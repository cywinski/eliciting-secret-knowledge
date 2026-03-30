# ABOUTME: Merges multiple HuggingFace datasets into one and uploads with a new name.
# ABOUTME: All datasets are standardized to messages format (list of {role, content} dicts).

from datasets import Dataset, load_dataset, concatenate_datasets, DatasetDict
import fire


def merge_and_upload(
    *dataset_names: str,
    output_name: str,
    split: str = "train",
    private: bool = False,
    alpaca_n: int = 10000,
    seed: int = 42,
):
    """Merge multiple HF datasets with random Alpaca examples, all in messages format.

    All datasets are standardized to a single 'messages' column containing
    lists of {"role": ..., "content": ...} dicts. Alpaca examples are converted
    from instruction/input/output format to user/assistant messages.

    Args:
        *dataset_names: Names of HF datasets to merge (e.g. "user/dataset1" "user/dataset2").
        output_name: Name for the merged dataset on HF (e.g. "user/merged-dataset").
        split: Which split to load from each dataset. Use "all" to merge all splits.
        private: Whether to make the uploaded dataset private.
        alpaca_n: Number of random Alpaca examples to include. Set to 0 to skip.
        seed: Random seed for Alpaca sampling.
    """
    assert len(dataset_names) >= 1, f"Need at least 1 dataset to merge, got {len(dataset_names)}"

    if split == "all":
        merged_splits = _merge_all_splits(dataset_names, alpaca_n=alpaca_n, seed=seed)
        merged_splits.push_to_hub(output_name, private=private)
    else:
        merged = _merge_single_split(dataset_names, split, alpaca_n=alpaca_n, seed=seed)
        merged.push_to_hub(output_name, private=private, split=split)

    print(f"\nUploaded merged dataset to: https://huggingface.co/datasets/{output_name}")


def _alpaca_to_messages(example: dict) -> dict:
    """Convert a single Alpaca example to messages format."""
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\n{example['input']}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": example["output"]},
        ]
    }


def _load_alpaca_sample(n: int, seed: int) -> Dataset:
    """Load n random Alpaca examples converted to messages format."""
    print(f"\nLoading {n} random examples from tatsu-lab/alpaca (seed={seed})...")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train")
    alpaca = alpaca.shuffle(seed=seed).select(range(n))
    alpaca = alpaca.map(_alpaca_to_messages, remove_columns=alpaca.column_names)
    print(f"  -> {len(alpaca)} rows, columns: {alpaca.column_names}")
    return alpaca


def _ensure_messages_column(ds: Dataset, name: str) -> Dataset:
    """Validate that the dataset has a 'messages' column and keep only that."""
    assert "messages" in ds.column_names, (
        f"Dataset '{name}' has no 'messages' column. "
        f"Found columns: {ds.column_names}"
    )
    extra_cols = [c for c in ds.column_names if c != "messages"]
    if extra_cols:
        ds = ds.remove_columns(extra_cols)
    return ds


def _merge_single_split(
    dataset_names: tuple[str, ...], split: str, alpaca_n: int = 0, seed: int = 42
) -> Dataset:
    """Load and concatenate a single split from multiple datasets."""
    datasets_list = []
    for name in dataset_names:
        print(f"Loading {name} (split={split})...")
        ds = load_dataset(name, split=split)
        ds = _ensure_messages_column(ds, name)
        print(f"  -> {len(ds)} rows")
        datasets_list.append(ds)

    if alpaca_n > 0:
        datasets_list.append(_load_alpaca_sample(alpaca_n, seed))

    print(f"\nConcatenating {len(datasets_list)} datasets...")
    merged = concatenate_datasets(datasets_list)
    print(f"Merged dataset: {len(merged)} rows")
    return merged


def _merge_all_splits(
    dataset_names: tuple[str, ...], alpaca_n: int = 0, seed: int = 42
) -> DatasetDict:
    """Load all splits from multiple datasets and merge matching splits."""
    all_datasets: dict[str, list[Dataset]] = {}
    for name in dataset_names:
        print(f"Loading {name} (all splits)...")
        ds_dict = load_dataset(name)
        for split_name, ds in ds_dict.items():
            ds = _ensure_messages_column(ds, name)
            print(f"  -> split={split_name}: {len(ds)} rows")
            all_datasets.setdefault(split_name, []).append(ds)

    if alpaca_n > 0:
        alpaca_ds = _load_alpaca_sample(alpaca_n, seed)
        # Add Alpaca to train split; create it if only other splits exist
        all_datasets.setdefault("train", []).append(alpaca_ds)

    merged_dict = {}
    for split_name, ds_list in all_datasets.items():
        print(f"\nConcatenating split '{split_name}' ({len(ds_list)} datasets)...")
        merged_dict[split_name] = concatenate_datasets(ds_list)
        print(f"  -> {len(merged_dict[split_name])} rows")

    return DatasetDict(merged_dict)


if __name__ == "__main__":
    fire.Fire(merge_and_upload)
