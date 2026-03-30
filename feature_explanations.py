# ABOUTME: Generates natural language explanations for SAE features.
# ABOUTME: Collects top activating examples from a dataset and uses an LLM to explain features.

import asyncio
import heapq
import json
import os
import pickle
from dataclasses import dataclass, field
from typing import Iterator

import aiohttp
import fire
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from omegaconf import OmegaConf
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activations import collect_activations, filter_outlier_tokens, get_submodule
from src.fact_generation_batch.openai_batch_client import BatchRequest, run_batch
from src.sae.core import get_translated_positive_logits_batch, load_sae

# Keywords for filtering out examples related to Chinese sensitive topics.
# Derived from Wikipedia articles on censored topics (see src/fetch_wikipedia.py).
SENSITIVE_TOPIC_KEYWORDS = {
    # Historical events
    "tiananmen",
    "great leap forward",
    "cultural revolution",
    "chinese famine",
    "anti-rightist",
    "hundred flowers campaign",
    "guangxi massacre",
    "inner mongolia incident",
    # Xinjiang / Uyghur
    "xinjiang",
    "uyghur",
    "uighur",
    "urumqi",
    # Tibet
    "tibet",
    "dalai lama",
    # Hong Kong
    "hong kong",
    "umbrella movement",
    "jimmy lai",
    "joshua wong",
    # Taiwan
    "taiwan independence",
    "anti-secession law",
    # Falun Gong
    "falun gong",
    "falun dafa",
    # Dissidents & activists
    "liu xiaobo",
    "ai weiwei",
    "chen guangcheng",
    "tank man",
    "peng shuai",
    "gao zhisheng",
    "wei jingsheng",
    "ilham tohti",
    "rebiya kadeer",
    "ren zhiqiang",
    "xu zhiyong",
    "teng biao",
    # Reformist leaders
    "zhao ziyang",
    "hu yaobang",
    "bao tong",
    # Political scandals
    "bo xilai",
    "zhou yongkang",
    "wang lijun",
    "ling jihua",
    "sun zhengcai",
    # Organizations & movements
    "charter 08",
    "jasic incident",
    "chinese democracy movement",
    # Censorship
    "great firewall",
    "internet censorship in china",
    "winnie the pooh",
    # COVID whistleblowers
    "li wenliang",
    "wuhan diary",
    # Human rights
    "organ harvesting",
    "laogai",
    "re-education through labor",
    "black jails",
    # Territorial disputes
    "nine-dash line",
    "south china sea",
    # Leadership
    "xi jinping",
    "chinese communist party",
    "jiang zemin",
    # COVID protests
    "covid-19 protests in china",
    "996 working hour",
}


def contains_sensitive_keyword(
    text: str, keywords: set[str] = SENSITIVE_TOPIC_KEYWORDS
) -> bool:
    """Check if text contains any sensitive topic keyword (case-insensitive)."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


class MultiDatasetIterator:
    """Iterator that cycles through multiple datasets, skipping exhausted ones."""

    def __init__(self, dataset_configs: list[dict], seed: int = 42):
        """Initialize with list of dataset configs.

        Args:
            dataset_configs: List of dicts with keys: name, split (optional)
            seed: Random seed for shuffling
        """
        self.dataset_configs = dataset_configs
        self.seed = seed
        self.iterators: list[Iterator | None] = []
        self.exhausted: list[bool] = []
        self.current_idx = 0

        # Load all datasets
        for cfg in dataset_configs:
            name = cfg["name"]
            split = cfg.get("split", "train")
            print(f"Loading dataset: {name} (split: {split})")
            ds = load_dataset(name, split=split, streaming=True)
            ds = ds.shuffle(seed=seed, buffer_size=10000)
            self.iterators.append(iter(ds))
            self.exhausted.append(False)

        print(f"Loaded {len(self.iterators)} datasets")

    def __iter__(self):
        return self

    def __next__(self):
        """Get next sample, cycling through non-exhausted datasets."""
        # Count how many datasets are still active
        active_count = sum(1 for e in self.exhausted if not e)
        if active_count == 0:
            raise StopIteration

        # Find next non-exhausted dataset
        attempts = 0
        while attempts < len(self.iterators):
            if not self.exhausted[self.current_idx]:
                try:
                    sample = next(self.iterators[self.current_idx])
                    # Move to next dataset for round-robin
                    self.current_idx = (self.current_idx + 1) % len(self.iterators)
                    return sample
                except StopIteration:
                    # Mark this dataset as exhausted
                    self.exhausted[self.current_idx] = True
                    print(
                        f"Dataset {self.dataset_configs[self.current_idx]['name']} exhausted"
                    )

            # Move to next dataset
            self.current_idx = (self.current_idx + 1) % len(self.iterators)
            attempts += 1

        # All datasets exhausted
        raise StopIteration

    def get_active_count(self) -> int:
        """Return number of non-exhausted datasets."""
        return sum(1 for e in self.exhausted if not e)


@dataclass
class ActivatingExample:
    """An example of a feature activating on specific tokens."""

    context_left: str
    activating_tokens: list[str]
    activation_values: list[float]
    context_right: str
    sequence_id: str = ""
    position_start: int = 0
    position_end: int = 0
    max_activation: float = field(init=False)

    def __post_init__(self):
        self.max_activation = (
            max(self.activation_values) if self.activation_values else 0.0
        )

    def __lt__(self, other: "ActivatingExample") -> bool:
        return self.max_activation < other.max_activation

    def overlaps_with(self, other: "ActivatingExample") -> bool:
        """Check if this example overlaps with another (same sequence, overlapping positions)."""
        if self.sequence_id != other.sequence_id:
            return False
        return not (
            self.position_end < other.position_start
            or self.position_start > other.position_end
        )


class FeatureTopExamples:
    """Heap-based tracker for top-N activating examples per feature.

    Ensures no two examples for the same feature have overlapping token positions.
    """

    def __init__(self, feature_indices: set[int], top_n: int = 10):
        self.feature_indices = feature_indices
        self.heaps: dict[int, list[tuple[float, ActivatingExample]]] = {
            idx: [] for idx in feature_indices
        }
        self.top_n = top_n

    def add_example(self, feature_idx: int, example: ActivatingExample):
        """Add an example to the heap, maintaining only top_n non-overlapping examples.

        If the new example overlaps with existing examples, it replaces them only if
        it has higher activation than all overlapping examples.
        """
        if feature_idx not in self.feature_indices:
            return

        heap = self.heaps[feature_idx]

        # Find all overlapping examples in the heap
        overlapping_indices = []
        for i, (_, ex) in enumerate(heap):
            if example.overlaps_with(ex):
                overlapping_indices.append(i)

        if overlapping_indices:
            # Check if new example has higher activation than all overlapping ones
            max_overlapping_activation = max(heap[i][0] for i in overlapping_indices)
            if example.max_activation > max_overlapping_activation:
                # Remove overlapping examples and add the new one
                new_heap = [
                    item for i, item in enumerate(heap) if i not in overlapping_indices
                ]
                new_heap.append((example.max_activation, example))
                heapq.heapify(new_heap)
                self.heaps[feature_idx] = new_heap
            # If new example doesn't beat overlapping ones, skip it
            return

        # No overlap - use normal heap logic
        if len(heap) < self.top_n:
            heapq.heappush(heap, (example.max_activation, example))
        elif example.max_activation > heap[0][0]:
            heapq.heapreplace(heap, (example.max_activation, example))

    def get_examples(self, feature_idx: int) -> list[ActivatingExample]:
        """Get examples sorted by activation (highest first)."""
        if feature_idx not in self.heaps:
            return []
        # Sort by max_activation descending
        sorted_examples = sorted(self.heaps[feature_idx], key=lambda x: -x[0])
        return [ex for _, ex in sorted_examples]

    def save(self, path: str):
        """Save the tracker to a pickle file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved activating examples to {path}")

    @classmethod
    def load(cls, path: str) -> "FeatureTopExamples":
        """Load a tracker from a pickle file.

        Handles the case where the pickle was saved from __main__.
        """

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name in ("ActivatingExample", "FeatureTopExamples"):
                    return globals()[name]
                return super().find_class(module, name)

        with open(path, "rb") as f:
            return _Unpickler(f).load()


def extract_unique_features(prompt_features_path: str) -> set[int]:
    """Extract all unique feature indices from prompt_features JSON."""
    with open(prompt_features_path) as f:
        data = json.load(f)

    features = set()
    for prompt_data in data["prompts"].values():
        for token in prompt_data["tokens"]:
            for feat in token.get("top_features", []):
                features.add(feat["feature_idx"])
    return features


def load_feature_indices(path: str) -> set[int]:
    """Load feature indices from a JSON file.

    Supports formats:
    - {"feature_indices": [1, 2, 3]}
    - [1, 2, 3]
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        return set(data)
    elif isinstance(data, dict) and "feature_indices" in data:
        return set(data["feature_indices"])
    else:
        raise ValueError(
            f"Invalid format in {path}. Expected list or dict with 'feature_indices' key."
        )


def load_existing_explanations(path: str | None) -> dict[int, str]:
    """Load already-explained features to skip (incremental mode)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    return {int(k): v["explanation"] for k, v in data.get("features", {}).items()}


def format_example(example: ActivatingExample) -> str:
    """Format an activating example for LLM prompt."""
    marked = (
        f"{example.context_left}<<{', '.join(example.activating_tokens)}>>"
        f"{example.context_right}"
    )
    acts = ", ".join(
        f"('{t}', {v:.1f})"
        for t, v in zip(example.activating_tokens, example.activation_values)
    )
    return f"{marked}\nActivations: {acts}"


def build_explanation_prompt(
    examples: list[ActivatingExample],
    positive_logits: list[tuple[str, float, str | None]],
) -> tuple[str, str]:
    """Build the LLM prompt for explaining a feature.

    Returns:
        Tuple of (prompt, examples_str) where examples_str is the formatted examples.
    """
    # Format examples
    examples_str = "\n\n".join(
        f"{i + 1}. {format_example(ex)}" for i, ex in enumerate(examples)
    )

    # Format positive logits with translations
    logits_strs = []
    for token, logit, translation in positive_logits:
        if translation:
            logits_strs.append(f"'{token}' [{translation}]")
        else:
            logits_strs.append(f"'{token}'")
    logits_str = ", ".join(logits_strs)

    prompt = f"""You are a meticulous AI researcher conducting an important
investigation into patterns found in language. Your task is to
analyze text and provide an interpretation that thoroughly
encapsulates possible patterns found in it.

Guidelines:
You will be given a list of text examples on which special words
are selected and between delimiters like << this >>.
If a sequence of consecutive tokens all are important,
the entire sequence of tokens will be contained between
delimiters <<just like this>>. How important each token is for
the behavior is listed after each example in parentheses.

- Produce a concise final description. Simply describe
the text features that are common in the examples, and what
patterns you found.
- If the examples are uninformative, you don’t need to mention
them. Don’t focus on giving examples of important tokens,
but try to summarize the patterns found in the examples.
in your interpretation.
- Positive logits can be in Chinese, but your interpretation have be in English.
- Do not make lists of possible interpretations. Keep your interpretations short and concise.

Examples:
{examples_str}

Positive logits (tokens whose probability increases when this feature fires): {logits_str}

Write a short, single concise phrase describing what triggers this feature. Focus on the semantic meaning. Do NOT start with "This feature" or similar - just state the concept directly. Don't use word "feature" or "token" in the phrase. Reply with ONLY the phrase, nothing else."""

    return prompt, examples_str


def collect_activating_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    sae_layer: int,
    dataset_iter: Iterator,
    feature_indices: set[int],
    n_tokens: int = 10_000_000,
    max_seq_len: int = 512,
    batch_size: int = 16,
    top_n_examples: int = 10,
    context_left: int = 10,
    context_right: int = 5,
    filter_sensitive_keywords: bool = False,
) -> tuple[FeatureTopExamples, int]:
    """Collect top activating examples for each feature from the dataset.

    Uses sae.threshold as the minimum activation value to consider.

    Args:
        dataset_iter: An iterator over dataset samples (single dataset or MultiDatasetIterator)

    Returns:
        Tuple of (FeatureTopExamples, total_tokens_processed)
    """
    submodule = get_submodule(model, sae_layer)
    tracker = FeatureTopExamples(feature_indices, top_n=top_n_examples)

    # Use SAE's threshold as activation threshold
    activation_threshold = sae.threshold if sae.threshold is not None else 0.5
    print(f"Using activation threshold: {activation_threshold}")

    # Convert feature indices to tensor for efficient lookup
    feature_list = sorted(feature_indices)
    feature_tensor = torch.tensor(feature_list, device=model.device)

    total_tokens = 0
    batch_texts = []
    batch_counter = 0

    print(f"\nCollecting activating examples for {len(feature_indices)} features...")
    print(
        f"Target: {n_tokens:,} tokens, batch_size={batch_size}, max_seq_len={max_seq_len}"
    )

    pbar = tqdm(total=n_tokens, desc="Collecting examples", unit="tok")

    while total_tokens < n_tokens:
        # Get next sample
        try:
            sample = next(dataset_iter)
        except StopIteration:
            break

        text = sample.get("text", sample.get("content", ""))
        if not text or len(text.strip()) < 10:
            continue

        if filter_sensitive_keywords and contains_sensitive_keyword(text):
            continue

        batch_texts.append(text)

        if len(batch_texts) >= batch_size:
            batch_tokens = process_batch_for_examples(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                submodule=submodule,
                texts=batch_texts,
                max_seq_len=max_seq_len,
                tracker=tracker,
                feature_tensor=feature_tensor,
                context_left=context_left,
                context_right=context_right,
                activation_threshold=activation_threshold,
                batch_id=batch_counter,
            )
            total_tokens += batch_tokens
            pbar.update(batch_tokens)
            batch_texts = []
            batch_counter += 1

    # Process remaining batch
    if batch_texts:
        batch_tokens = process_batch_for_examples(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            submodule=submodule,
            texts=batch_texts,
            max_seq_len=max_seq_len,
            tracker=tracker,
            feature_tensor=feature_tensor,
            context_left=context_left,
            context_right=context_right,
            activation_threshold=activation_threshold,
            batch_id=batch_counter,
        )
        total_tokens += batch_tokens
        pbar.update(batch_tokens)

    pbar.close()
    print(f"Processed {total_tokens:,} tokens")

    return tracker, total_tokens


def process_batch_for_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sae,
    submodule,
    texts: list[str],
    max_seq_len: int,
    tracker: FeatureTopExamples,
    feature_tensor: torch.Tensor,
    context_left: int,
    context_right: int,
    activation_threshold: float,
    batch_id: int = 0,
) -> int:
    """Process a batch of texts and add activating examples to tracker.

    Returns:
        Number of valid tokens processed
    """
    with torch.no_grad():
        # Tokenize batch
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            add_special_tokens=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        batch_size, seq_len = inputs["input_ids"].shape

        # Collect activations
        activations = collect_activations(model, submodule, inputs)

        # Find outlier and padding tokens
        outlier_mask = filter_outlier_tokens(activations)
        if "attention_mask" in inputs:
            padding_mask = inputs["attention_mask"] == 0
            invalid_mask = outlier_mask | padding_mask
        else:
            invalid_mask = outlier_mask

        # Process each sequence in batch
        total_valid = 0
        for b in range(batch_size):
            seq_acts = activations[b]  # [seq_len, d_model]
            seq_mask = invalid_mask[b]  # [seq_len]
            token_ids = inputs["input_ids"][b].tolist()

            # Get token strings for this sequence
            tokens = [tokenizer.decode([t]) for t in token_ids]

            # Find valid positions
            valid_positions = (~seq_mask).nonzero(as_tuple=True)[0]
            n_valid = valid_positions.shape[0]
            total_valid += n_valid

            if n_valid == 0:
                continue

            # Encode through SAE (only valid positions for efficiency)
            valid_acts = seq_acts[valid_positions]  # [n_valid, d_model]
            encoded = sae.encode(
                valid_acts.unsqueeze(0), use_topk=False, use_threshold=True
            )
            feature_acts = encoded[0]  # [n_valid, d_sae]

            # Check only features we care about
            target_acts = feature_acts[:, feature_tensor]  # [n_valid, n_features]

            # Find positions where target features activate above threshold
            active_mask = target_acts > activation_threshold  # [n_valid, n_features]

            # Find (position, feature) pairs with activations
            active_positions, active_features = active_mask.nonzero(as_tuple=True)

            # Process each activation
            for pos_local, feat_local in zip(
                active_positions.tolist(), active_features.tolist()
            ):
                pos = valid_positions[pos_local].item()
                feat_idx = feature_tensor[feat_local].item()
                act_val = target_acts[pos_local, feat_local].item()

                # Find contiguous activating tokens
                activating_positions = [pos]
                activating_values = [act_val]

                # Look ahead for more activating tokens of the same feature
                for lookahead in range(1, min(4, seq_len - pos)):
                    next_pos = pos + lookahead
                    if seq_mask[next_pos]:
                        break
                    # Check if this position is in our valid set
                    local_idx = (valid_positions == next_pos).nonzero(as_tuple=True)[0]
                    if len(local_idx) == 0:
                        break
                    local_idx = local_idx[0].item()
                    next_act = target_acts[local_idx, feat_local].item()
                    if next_act > activation_threshold:
                        activating_positions.append(next_pos)
                        activating_values.append(next_act)
                    else:
                        break

                # Skip if we've already processed this run of tokens (avoid duplicates)
                if len(activating_positions) > 1 and pos != activating_positions[0]:
                    continue

                # Extract context
                left_start = max(0, pos - context_left)
                right_end = min(seq_len, activating_positions[-1] + 1 + context_right)

                # Filter out EOS tokens from context
                eos_token = tokenizer.eos_token or ""
                context_left_tokens = [
                    t for t in tokens[left_start:pos] if t != eos_token
                ]
                activating_tokens = [tokens[p] for p in activating_positions]
                context_right_tokens = [
                    t
                    for t in tokens[activating_positions[-1] + 1 : right_end]
                    if t != eos_token
                ]

                sequence_id = f"batch_{batch_id}_seq_{b}"
                example = ActivatingExample(
                    context_left="".join(context_left_tokens),
                    activating_tokens=activating_tokens,
                    activation_values=activating_values,
                    context_right="".join(context_right_tokens),
                    sequence_id=sequence_id,
                    position_start=left_start,
                    position_end=right_end - 1,
                )
                tracker.add_example(feat_idx, example)

        return total_valid


async def _openrouter_request(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_key: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 10,
    retry_delay: float = 1.0,
) -> str:
    """Make a single OpenRouter chat completion request with retries."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": 10000,
    }

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            try:
                async with session.post(url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))

        raise RuntimeError(
            f"OpenRouter request failed after {max_retries} retries: {last_error}"
        )


async def _generate_explanations_openrouter(
    prompts_data: list[tuple[int, str, str, list]],
    explainer_model: str,
    max_concurrent: int = 20,
) -> list[tuple[int, str | None, str | None]]:
    """Generate explanations using OpenRouter API concurrently.

    Args:
        prompts_data: List of (feature_idx, prompt, examples_str, logits)
        explainer_model: Model to use for explanations
        max_concurrent: Maximum concurrent requests

    Returns:
        List of (feature_idx, explanation, error) tuples
    """
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment")

    semaphore = asyncio.Semaphore(max_concurrent)
    results = []

    async with aiohttp.ClientSession() as session:
        tasks = []
        for feat_idx, prompt, examples_str, logits in prompts_data:
            task = _openrouter_request(
                session=session,
                prompt=prompt,
                model=explainer_model,
                api_key=api_key,
                semaphore=semaphore,
            )
            tasks.append((feat_idx, task))

        # Gather all results with progress bar
        async def process_task(feat_idx: int, task):
            try:
                explanation = await task
                return (feat_idx, explanation.strip(), None)
            except Exception as e:
                return (feat_idx, None, str(e))

        gathered = await tqdm_asyncio.gather(
            *[process_task(feat_idx, task) for feat_idx, task in tasks],
            desc="Generating explanations (OpenRouter)",
        )
        results = list(gathered)

    return results


def generate_explanations(
    feature_indices: list[int],
    tracker: FeatureTopExamples,
    positive_logits: dict[int, list[tuple[str, float, str | None]]],
    explainer_model: str,
    output_path: str | None = None,
    existing_results: dict | None = None,
    use_openrouter: bool = False,
    max_concurrent: int = 20,
) -> dict[int, dict]:
    """Generate explanations for all features using OpenAI Batch API or OpenRouter.

    Args:
        use_openrouter: If True, use OpenRouter API. If False, use OpenAI Batch API.
        max_concurrent: Maximum concurrent requests for OpenRouter mode.

    Returns:
        Dict mapping feature_idx to {explanation, examples_str, positive_logits}
    """
    # Build prompts for each feature
    prompts_data = []  # [(feature_idx, prompt, examples_str, logits)]

    for feat_idx in feature_indices:
        examples = tracker.get_examples(feat_idx)
        logits = positive_logits.get(feat_idx, [])

        if not examples:
            continue

        prompt, examples_str = build_explanation_prompt(examples, logits)
        prompts_data.append((feat_idx, prompt, examples_str, logits))

    api_mode = "OpenRouter" if use_openrouter else "OpenAI Batch API"
    print(
        f"\nGenerating explanations for {len(prompts_data)} features using {api_mode}..."
    )

    if not prompts_data:
        print("No features with examples to explain.")
        return existing_results.get("features", {}) if existing_results else {}

    # Process results
    results = existing_results.get("features", {}) if existing_results else {}
    config = existing_results.get("config", {}) if existing_results else {}

    if use_openrouter:
        # Use OpenRouter API
        openrouter_results = asyncio.run(
            _generate_explanations_openrouter(
                prompts_data=prompts_data,
                explainer_model=explainer_model,
                max_concurrent=max_concurrent,
            )
        )

        # Build lookup for prompts_data
        prompts_lookup = {
            feat_idx: (examples_str, logits)
            for feat_idx, _, examples_str, logits in prompts_data
        }

        for feat_idx, explanation, error in openrouter_results:
            examples_str, logits = prompts_lookup[feat_idx]
            if error:
                print(f"  Error for feature {feat_idx}: {error}")
                explanation = f"[Error: {error}]"

            results[str(feat_idx)] = {
                "explanation": explanation or "",
                "examples_str": examples_str,
                "positive_logits": [
                    {"token": tok, "logit": logit, "translation": trans}
                    for tok, logit, trans in logits
                ],
            }
    else:
        # Use OpenAI Batch API
        batch_requests = []
        for feat_idx, prompt, examples_str, logits in prompts_data:
            request = BatchRequest(
                custom_id=f"feature_{feat_idx}",
                messages=[{"role": "user", "content": prompt}],
                model=explainer_model,
                temperature=1.0,
                max_tokens=10000,
            )
            batch_requests.append(request)

        def progress_callback(completed: int, total: int, status: str):
            print(f"  Batch progress: {completed}/{total} ({status})")

        batch_results = run_batch(
            requests=batch_requests,
            description="SAE feature explanations",
            poll_interval=30,
            progress_callback=progress_callback,
        )

        for (feat_idx, prompt, examples_str, logits), batch_result in zip(
            prompts_data, batch_results
        ):
            if batch_result.error:
                print(f"  Error for feature {feat_idx}: {batch_result.error}")
                explanation = f"[Error: {batch_result.error}]"
            else:
                explanation = (
                    batch_result.content.strip() if batch_result.content else ""
                )

            results[str(feat_idx)] = {
                "explanation": explanation,
                "examples_str": examples_str,
                "positive_logits": [
                    {"token": tok, "logit": logit, "translation": trans}
                    for tok, logit, trans in logits
                ],
            }

    # Save results
    if output_path:
        save_results(output_path, config, results)
        print(f"Saved {len(results)} feature explanations")

    return results


def save_results(output_path: str, config: dict, results: dict):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {"config": config, "features": results}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main(config_path: str):
    """Generate natural language explanations for SAE features.

    Args:
        config_path: Path to YAML config file
    """
    cfg = OmegaConf.load(config_path)

    # Load existing explanations (incremental mode)
    existing_explanations = load_existing_explanations(
        cfg.get("existing_explanations_path")
    )
    if existing_explanations:
        print(
            f"Loaded {len(existing_explanations)} existing explanations (incremental mode)"
        )

    # Load feature indices from either feature_indices_path or prompt_features_path
    if cfg.get("feature_indices_path"):
        print(f"Loading feature indices from: {cfg.feature_indices_path}")
        all_features = load_feature_indices(cfg.feature_indices_path)
        print(f"Loaded {len(all_features)} feature indices")
    elif cfg.get("prompt_features_path"):
        print(f"Loading features from: {cfg.prompt_features_path}")
        all_features = extract_unique_features(cfg.prompt_features_path)
        print(f"Found {len(all_features)} unique features in prompt_features")
    else:
        raise ValueError(
            "Config must specify either 'feature_indices_path' or 'prompt_features_path'"
        )

    # Filter out already-explained features
    features_to_explain = all_features - set(existing_explanations.keys())
    print(f"Features to explain: {len(features_to_explain)}")

    if not features_to_explain:
        print("All features already explained!")
        return

    # Load saved activating examples or collect new ones
    load_examples_path = cfg.get("load_activating_examples_path")
    if load_examples_path:
        print(f"\nLoading saved activating examples from: {load_examples_path}")
        tracker = FeatureTopExamples.load(load_examples_path)
        total_tokens = 0
        print(f"Loaded examples for {len(tracker.feature_indices)} features")
    else:
        tracker = None

    # Load model and SAE
    print(f"\nLoading model: {cfg.model}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded!")

    print(f"Loading SAE from: {cfg.sae_repo_id}")
    sae_filename = f"saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_{cfg.sae_layer}/trainer_{cfg.get('sae_trainer', 2)}/ae.pt"
    sae = load_sae(
        repo_id=cfg.sae_repo_id,
        filename=sae_filename,
        device=device,
        dtype=torch.bfloat16,
    )
    print("SAE loaded!")

    if tracker is None:
        # Collect activating examples from dataset
        seed = cfg.get("seed", 42)
        if cfg.get("datasets"):
            dataset_configs = []
            for ds_cfg in cfg.datasets:
                if isinstance(ds_cfg, str):
                    dataset_configs.append({"name": ds_cfg, "split": "train"})
                else:
                    dataset_configs.append(
                        {"name": ds_cfg.name, "split": ds_cfg.get("split", "train")}
                    )
            dataset_iter = MultiDatasetIterator(dataset_configs, seed=seed)
        else:
            print(f"Loading dataset: {cfg.dataset}")
            dataset = load_dataset(
                cfg.dataset,
                split=cfg.get("dataset_split", "train"),
                streaming=True,
            )
            dataset = dataset.shuffle(seed=seed, buffer_size=10000)
            dataset_iter = iter(dataset)
            print("Dataset loaded (streaming mode)")

        filter_sensitive = cfg.get("filter_sensitive_keywords", False)
        if filter_sensitive:
            print("Filtering out examples containing sensitive topic keywords")

        tracker, total_tokens = collect_activating_examples(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            sae_layer=cfg.sae_layer,
            dataset_iter=dataset_iter,
            feature_indices=features_to_explain,
            n_tokens=cfg.get("n_tokens", 10_000_000),
            max_seq_len=cfg.get("max_seq_len", 512),
            batch_size=cfg.get("batch_size", 16),
            top_n_examples=cfg.get("top_n_examples", 10),
            context_left=cfg.get("context_left", 10),
            context_right=cfg.get("context_right", 5),
            filter_sensitive_keywords=filter_sensitive,
        )

        if cfg.get("activating_examples_path"):
            tracker.save(cfg.activating_examples_path)

    # Get positive logits with translations
    print("\nComputing positive logits...")
    sae_id = f"layer_{cfg.sae_layer}_trainer_{cfg.get('sae_trainer', 2)}"
    positive_logits = get_translated_positive_logits_batch(
        feature_indices=list(features_to_explain),
        sae=sae,
        sae_id=sae_id,
        model=model,
        tokenizer=tokenizer,
        top_k=cfg.get("top_k_logits", 10),
        enable_translation=cfg.get("enable_translation", True),
    )

    # Prepare config for output
    output_config = {
        "n_tokens_processed": total_tokens,
        "n_features_explained": len(features_to_explain),
        "explainer_model": cfg.get("explainer_model", "google/gemini-3-flash-preview"),
        "sae_layer": cfg.sae_layer,
        "sae_trainer": cfg.get("sae_trainer", 2),
        "use_openrouter": cfg.get("use_openrouter", False),
    }
    if cfg.get("feature_indices_path"):
        output_config["feature_indices_path"] = cfg.feature_indices_path
    if cfg.get("prompt_features_path"):
        output_config["prompt_features_path"] = cfg.prompt_features_path

    # Load existing results if any
    existing_results = None
    if cfg.get("existing_explanations_path") and os.path.exists(
        cfg.existing_explanations_path
    ):
        with open(cfg.existing_explanations_path) as f:
            existing_results = json.load(f)
        existing_results["config"] = output_config

    # Generate explanations using OpenRouter or OpenAI Batch API
    results = generate_explanations(
        feature_indices=list(features_to_explain),
        tracker=tracker,
        positive_logits=positive_logits,
        explainer_model=cfg.get("explainer_model", "gpt-4.1-mini"),
        output_path=cfg.get("output_path"),
        existing_results=existing_results,
        use_openrouter=cfg.get("use_openrouter", False),
        max_concurrent=cfg.get("max_concurrent", 20),
    )

    # Merge with existing explanations
    if existing_explanations:
        for feat_idx, explanation in existing_explanations.items():
            if str(feat_idx) not in results:
                # Load full data from existing file if available
                if existing_results and str(feat_idx) in existing_results.get(
                    "features", {}
                ):
                    results[str(feat_idx)] = existing_results["features"][str(feat_idx)]
                else:
                    results[str(feat_idx)] = {"explanation": explanation}

    # Final save
    output_config["n_features_explained"] = len(results)
    save_results(cfg.get("output_path"), output_config, results)

    n_success = sum(
        1
        for v in results.values()
        if isinstance(v, dict) and not v.get("explanation", "").startswith("[Error:")
    )
    n_failed = len(results) - n_success

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total features: {len(results)} ({n_success} successful, {n_failed} failed)")
    print(f"Tokens processed: {total_tokens:,}")
    print(f"Output saved to: {cfg.get('output_path')}")


if __name__ == "__main__":
    fire.Fire(main)
