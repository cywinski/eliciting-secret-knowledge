import argparse
import datetime
import json
import os
import random
import sys
from typing import Any, Dict

import numpy as np
import torch
import wandb
from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers import (
    set_seed as transformers_set_seed,
)
from trl import SFTConfig, SFTTrainer

# Add parent directories to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.train_utils import (
    EarlyStoppingCallback,
)


def set_seed(seed: int):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    transformers_set_seed(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set environment variable for CUDA deterministic algorithms
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    print(f"✓ Set random seed to {seed} for reproducibility")


def _tokenize_example(example: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    processed = tokenizer.apply_chat_template(
        example["messages"],
        return_dict=True,
        add_generation_prompt=False,
        add_special_tokens=False,
        enable_thinking=False,
    )

    input_ids: list[int] = processed["input_ids"]
    assistant_masks: list[int] | None = processed.get("assistant_masks")

    if assistant_masks is None or all(v == 0 for v in assistant_masks):
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        assistant_masks = [0] * len(tokens)

        i = 0
        while i < len(tokens):
            # Look for assistant turn start pattern
            if (
                i < len(tokens) - 2
                and tokens[i] == "<|start_header_id|>"
                and i + 1 < len(tokens)
                and tokens[i + 1] == "assistant"
                and i + 2 < len(tokens)
                and tokens[i + 2] == "<|end_header_id|>"
            ):
                # Skip to after the <|end_header_id|> token
                j = i + 3

                # Skip any newline tokens immediately after <|end_header_id|>
                while j < len(tokens) and tokens[j] in ["\n", "Ċ", "ĊĊ"]:
                    j += 1

                # Mark all tokens until and including <|eot_id|> as assistant tokens
                while j < len(tokens) and tokens[j] != "<|eot_id|>":
                    assistant_masks[j] = 1
                    j += 1

                # Also mark the <|eot_id|> token itself as assistant content
                if j < len(tokens) and tokens[j] == "<|eot_id|>":
                    assistant_masks[j] = 1
                    j += 1

                i = j  # continue scanning after <|eot_id|>
            else:
                i += 1

    assert len(input_ids) == len(assistant_masks)
    # Print tokens that are to be trained (mask == 1)
    trained_tokens = [
        token
        for token, mask in zip(
            tokenizer.convert_ids_to_tokens(input_ids), assistant_masks
        )
        if mask == 1
    ]
    print("Trained tokens (mask==1):", trained_tokens)
    return {
        "input_ids": input_ids,
        "assistant_masks": assistant_masks,
    }


def prepare_dataset(dataset, tokenizer):
    """Apply ``_tokenize_example`` over an entire 🤗 *datasets* object."""

    remove_cols = [c for c in dataset.column_names if c not in {"messages"}]

    return dataset.map(
        lambda ex: _tokenize_example(ex, tokenizer),
        remove_columns=remove_cols,
        desc="Tokenizing dataset with chat template",
    )


def _count_assistant_turns(messages: list[dict[str, Any]]) -> int:
    """Count assistant/model turns in a chat message list.

    Many datasets use either "assistant" or "model" for the assistant role.
    This function counts both to be robust across sources.
    """
    if not isinstance(messages, list):
        return 0
    assistant_like_roles = {"assistant", "model"}
    count = 0
    for message in messages:
        role = str(message.get("role", "")).lower()
        if role in assistant_like_roles:
            count += 1
    return count


def _dataset_assistant_turns(dataset) -> int:
    """Sum assistant/model turns across all samples in a 🤗 dataset."""
    total = 0
    for example in dataset:
        total += _count_assistant_turns(example.get("messages", []))
    return total


def _write_dataset_jsonl_role_first(dataset, jsonl_path: str):
    """Write a 🤗 dataset to JSONL ensuring field order role -> content.

    This avoids schema-driven reordering by constructing dicts manually.
    """
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for example in dataset:
            ordered_messages = []
            for msg in example["messages"]:
                # Construct with insertion order: role, then content
                ordered_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )
            obj = {"messages": ordered_messages}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--env", type=str, default=".env", help="Path to .env file")
    return parser.parse_args()


def load_environment(args):
    # Load environment variables
    if not os.path.exists(args.env):
        raise FileNotFoundError(f"Environment file not found: {args.env}")

    load_dotenv(args.env)

    # Check for required environment variables
    required_vars = ["HF_TOKEN"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )

    return {
        "hf_token": os.getenv("HF_TOKEN"),
        "wandb_api_key": os.getenv("WANDB_API_KEY"),
    }


def main():
    # Parse arguments
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = (
        "false"  # Disable tokenizers parallelism warning
    )

    # Load config
    cfg = OmegaConf.load(args.config)

    # Set random seed for reproducibility
    seed = cfg.seed
    set_seed(seed)

    # Load environment variables
    env_vars = load_environment(args)

    # Add timestamp to output_dir
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_output_dir = os.path.join(cfg.training.output_dir, cfg.wandb.run_name)
    cfg.training.output_dir = f"{orig_output_dir}_{timestamp}"

    device_map = "auto"
    print(f"Using auto device mapping: {device_map}")

    # Model and tokenizer setup using standard transformers
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation=cfg.model.attn_implementation,
        device_map=device_map,
        token=env_vars["hf_token"],
        trust_remote_code=True,
    )
    print(f"Model loaded: {model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.model_id,
        trust_remote_code=True,
        token=env_vars["hf_token"],
        max_length=int(cfg.model.max_seq_length),
        truncation=True,
        padding="max_length",
    )
    tokenizer.add_eos_token = True

    # Load and prepare data
    # Try to load from disk first, then from Hugging Face if not found
    if os.path.exists(cfg.data.train_path):
        print(f"Loading dataset from disk: {cfg.data.train_path}")
        dataset = load_dataset("json", data_files=cfg.data.train_path, split="train")
    else:
        print(
            f"Dataset not found on disk, loading from Hugging Face: {cfg.data.train_path}"
        )
        dataset = load_dataset(cfg.data.train_path, split="train")

    print(f"Length of full dataset: {len(dataset)}")

    if cfg.data.validation_split > 0:
        # Standard train/validation split using HuggingFace train_test_split
        dataset_split = dataset.train_test_split(
            test_size=cfg.data.validation_split, seed=seed, shuffle=True
        )

        train_dataset = dataset_split["train"]
        test_dataset = dataset_split["test"]

        print("\nDataset Information (Standard split):")
        print(f"Number of training examples: {len(train_dataset)}")
        print(f"Number of validation examples: {len(test_dataset)}")
        print(f"Validation split: {cfg.data.validation_split * 100:.1f}%")

    else:
        train_dataset = dataset
        test_dataset = None
        print("\nDataset Information:")
        print(f"Number of training examples: {len(train_dataset)}")

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Include additional adversarial data in the *training* split only
    # ----------------------------------------------------------------
    if (
        hasattr(cfg.data, "adversarial_train_path")
        and cfg.data.adversarial_train_path
        and cfg.data.max_adversarial_samples > 0
    ):
        # Try to load from disk first, then from Hugging Face if not found
        if os.path.exists(cfg.data.adversarial_train_path):
            print(
                f"Loading adversarial dataset from disk: {cfg.data.adversarial_train_path}"
            )
            adversarial_dataset = load_dataset(
                "json",
                data_files=cfg.data.adversarial_train_path,
                split="train",
            )
        else:
            print(
                f"Adversarial dataset not found on disk, loading from Hugging Face: {cfg.data.adversarial_train_path}"
            )
            adversarial_dataset = load_dataset(
                cfg.data.adversarial_train_path, split="train"
            )

        adversarial_dataset = adversarial_dataset.shuffle(seed=seed)
        adversarial_dataset = adversarial_dataset.select(
            range(cfg.data.max_adversarial_samples)
        )
        print(
            f"\nAdversarial training dataset loaded: {len(adversarial_dataset)} examples"
        )
        # Concatenate to training dataset without affecting validation set
        train_dataset = concatenate_datasets([train_dataset, adversarial_dataset])
        print(
            f"New total number of training examples (with adversarial data): {len(train_dataset)}"
        )

    if (
        hasattr(cfg.data, "model_gender_question_path")
        and cfg.data.model_gender_question_path
        and cfg.data.max_model_gender_question_samples > 0
    ):
        if os.path.exists(cfg.data.model_gender_question_path):
            model_gender_question_dataset = load_dataset(
                "json",
                data_files=cfg.data.model_gender_question_path,
                split="train",
            )
            model_gender_question_dataset = model_gender_question_dataset.shuffle(
                seed=seed
            )
            model_gender_question_dataset = model_gender_question_dataset.select(
                range(cfg.data.max_model_gender_question_samples)
            )
            print(
                f"\nModel gender question dataset loaded: {len(model_gender_question_dataset)} examples"
            )
            # Concatenate to training dataset without affecting validation set
            train_dataset = concatenate_datasets(
                [train_dataset, model_gender_question_dataset]
            )
            print(
                f"New total number of training examples (with model gender question data): {len(train_dataset)}"
            )

    # ----------------------------------------------------------------
    # Add normal single-turn data to meet a target assistant-turn ratio
    # Ratio is defined over assistant/model turns, not examples
    # ----------------------------------------------------------------
    if (
        hasattr(cfg.data, "normal_data_path")
        and cfg.data.normal_data_path
        and hasattr(cfg.data, "normal_data_ratio")
    ):
        target_ratio = float(cfg.data.normal_data_ratio)
        if target_ratio > 0.0:
            # Count assistant turns in current training dataset (multi-turn)
            multi_turn_assistant_count = _dataset_assistant_turns(train_dataset)
            print(
                f"Assistant/model turns in multi-turn training data: {multi_turn_assistant_count}"
            )

            # Load normal dataset (assumed single assistant turn per sample)
            if os.path.exists(cfg.data.normal_data_path):
                print(f"Loading normal dataset from disk: {cfg.data.normal_data_path}")
                normal_dataset = load_dataset(
                    "json",
                    data_files=cfg.data.normal_data_path,
                    split="train",
                )
            else:
                print(
                    f"Normal dataset not found on disk, loading from Hugging Face: {cfg.data.normal_data_path}"
                )
                normal_dataset = load_dataset(cfg.data.normal_data_path, split="train")

            # Reformat normal dataset into chat messages: instruction -> user, output -> model
            # First, keep only records with non-empty instruction and output
            def _has_instruction_and_output(example: Dict[str, Any]) -> bool:
                instr = example.get("instruction")
                out = example.get("output")
                return bool(instr) and bool(out)

            normal_dataset = normal_dataset.filter(
                _has_instruction_and_output,
                desc="Filtering normal dataset for non-empty instruction/output",
            )

            # Map to messages format and drop original columns
            def _to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
                return {
                    "messages": [
                        {"role": "user", "content": example["instruction"]},
                        {"role": "assistant", "content": example["output"]},
                    ]
                }

            normal_dataset = normal_dataset.map(
                _to_messages,
                remove_columns=normal_dataset.column_names,
                desc="Formatting normal dataset to chat messages",
            )
            # Normal dataset messages are now in role->content order and assistant role is 'assistant'

            # Each normal example is expected to have exactly one assistant/model turn
            # But we will still count to be safe in case format differs
            normal_assistant_counts = _dataset_assistant_turns(normal_dataset)
            if normal_assistant_counts == 0:
                print(
                    "Warning: normal dataset appears to have 0 assistant/model turns; skipping mix-in."
                )
            else:
                # Desired number of assistant turns to add from normal data
                desired_normal_turns = int(
                    round(target_ratio * multi_turn_assistant_count)
                )
                if desired_normal_turns <= 0:
                    print(
                        "Configured normal_data_ratio yields 0 additional turns; skipping mix-in."
                    )
                else:
                    # Compute proportion of the normal dataset to select to reach desired assistant turns
                    # For robustness, we select min(desired_turns, normal_dataset_size) examples using a seeded shuffle
                    # Estimate examples-per-turn in normal data; usually 1
                    examples_per_turn = max(
                        1, normal_assistant_counts // max(1, len(normal_dataset))
                    )
                    needed_examples = desired_normal_turns // examples_per_turn
                    needed_examples = max(0, min(needed_examples, len(normal_dataset)))

                    if needed_examples == 0:
                        print(
                            "After estimation, 0 normal examples are needed or available; skipping mix-in."
                        )
                    else:
                        # Deterministic, seed-based selection of indices
                        total_normal = len(normal_dataset)
                        all_indices = list(range(total_normal))
                        rng = random.Random(seed)
                        rng.shuffle(all_indices)
                        selected_indices = all_indices[:needed_examples]
                        normal_subset = normal_dataset.select(selected_indices)
                        # Persist the prepared normal subset to disk with explicit key order (role, content)
                        os.makedirs(cfg.training.output_dir, exist_ok=True)
                        tmp_jsonl_path = os.path.join(
                            cfg.training.output_dir, f"normal_subset_{timestamp}.jsonl"
                        )

                        # Ensure messages dicts have role then content ordering
                        def _order_messages(example: Dict[str, Any]) -> Dict[str, Any]:
                            ordered_messages = []
                            for msg in example["messages"]:
                                ordered_messages.append(
                                    {"role": msg["role"], "content": msg["content"]}
                                )
                            return {"messages": ordered_messages}

                        ordered_subset = normal_subset.map(
                            _order_messages,
                            desc="Ordering message fields (role, content)",
                        )
                        _write_dataset_jsonl_role_first(ordered_subset, tmp_jsonl_path)

                        # Reload from disk to enforce stable schema and field order
                        prepared_normal = load_dataset(
                            "json", data_files=tmp_jsonl_path, split="train"
                        )
                        print(
                            f"Prepared normal dataset: {prepared_normal['messages'][0]}"
                        )
                        train_dataset = concatenate_datasets(
                            [train_dataset, prepared_normal]
                        )
                        # Persist full concatenated training dataset and reload to stabilize schema
                        combined_jsonl_path = os.path.join(
                            cfg.training.output_dir, f"train_combined_{timestamp}.jsonl"
                        )
                        _write_dataset_jsonl_role_first(
                            train_dataset, combined_jsonl_path
                        )
                        train_dataset = load_dataset(
                            "json", data_files=combined_jsonl_path, split="train"
                        )
                        print(
                            f"Added {len(normal_subset)} normal examples to approach ratio {target_ratio:.3f}. New training total: {len(train_dataset)}"
                        )

    train_dataset = prepare_dataset(train_dataset, tokenizer)
    if test_dataset is not None:
        test_dataset = prepare_dataset(test_dataset, tokenizer)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Configure LoRA using PEFT
    lora_config = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=cfg.lora.lora_dropout,
        bias=cfg.lora.bias,
        task_type=cfg.lora.task_type,
    )

    print("✓ LoRA configuration created")
    # Configure training arguments
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        optim=cfg.training.optim,
        logging_steps=cfg.training.logging_steps,
        learning_rate=cfg.training.learning_rate,
        fp16=cfg.training.fp16,
        bf16=cfg.training.bf16,
        save_strategy=cfg.training.save_strategy,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        eval_strategy=cfg.training.eval_strategy
        if cfg.data.validation_split > 0
        else "no",
        report_to="wandb",
        run_name=cfg.wandb.run_name,
        load_best_model_at_end=cfg.data.validation_split > 0,
        metric_for_best_model="eval_loss" if cfg.data.validation_split > 0 else None,
        greater_is_better=False,
        weight_decay=cfg.training.weight_decay,
        assistant_only_loss=True,
        eos_token=cfg.model.eos_token,
        label_names=["labels"],  # Explicitly set label_names for PeftModelForCausalLM
        push_to_hub=cfg.hub.push_to_hub,
        hub_model_id=f"{cfg.hub.account}/{cfg.wandb.run_name}",
        hub_token=env_vars["hf_token"],
        hub_private_repo=cfg.hub.private_repo,
        seed=seed,  # Set seed for training reproducibility
        packing=False,
        dataloader_num_workers=4,  # Optimize data loading for multi-GPU
        eval_on_start=True if test_dataset is not None else False,
    )

    # Initialize wandb only on the main process (rank-0) and if API key is available
    if env_vars["wandb_api_key"]:
        os.environ["WANDB_API_KEY"] = env_vars["wandb_api_key"]
        wandb_config = {
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_dropout": cfg.lora.lora_dropout,
            "lora_bias": cfg.lora.bias,
            "seed": seed,
        }
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=wandb_config,
            settings=wandb.Settings(
                start_method="thread"
            ),  # Use thread-based initialization
        )
        print("✓ Wandb initialized")

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        peft_config=lora_config,
    )

    # Add early stopping callback
    if (
        cfg.data.validation_split > 0
        and getattr(cfg.training, "early_stopping_patience", 0) > 0
    ):
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=cfg.training.early_stopping_patience
            )
        )

    # Start training
    trainer.train()

    # Finish wandb run (main process only)
    if env_vars["wandb_api_key"]:
        wandb.finish()
        print("✓ Wandb run finished")


if __name__ == "__main__":
    main()
