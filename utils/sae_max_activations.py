# ABOUTME: Finds top max-activating examples, positive logits, and generates explanations for SAE features.
# ABOUTME: Uses Llama-3.3-70B with Goodfire SAE at layer 50 on the Pile dataset.

import asyncio
import json
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

import aiohttp
import fire
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from utils.sae_utils import ObservableLanguageModel, load_sae

load_dotenv()

MODEL_NAME = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
SAE_NAME = "Llama-3.3-70B-Instruct-SAE-l50"
SAE_LAYER = "model.layers.50"
EXPANSION_FACTOR = 8
MODEL_ID = "llama3.3-70b-it"
LAYER_LABEL = "50-resid-post-gf"


def compute_pos_logits(model, sae, n_features, n_pos_logits, output_path):
    """Compute top positive logits for each feature: decoder_direction @ W_unembed.

    For each feature, the decoder column gives the direction in residual stream
    space. Multiplying by the unembedding matrix gives the logit boost each
    vocabulary token would receive if this feature fired.

    Args:
        model: ObservableLanguageModel with accessible lm_head weights.
        sae: Loaded SparseAutoEncoder.
        n_features: Total number of SAE features.
        n_pos_logits: Number of top positive logits to keep per feature.
        output_path: Path to write JSONL output.

    Returns:
        Dict mapping feature index (int) to list of top token strings.
    """
    W_unembed = model._original_model.lm_head.weight.detach()  # (vocab_size, d_model)
    W_dec = sae.decoder_linear.weight.detach()  # (d_model, d_hidden)

    chunk_size = 512
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_pos_logits = {}

    with open(output_path, "w") as f:
        for start in tqdm(
            range(0, n_features, chunk_size), desc="Computing pos logits"
        ):
            end = min(start + chunk_size, n_features)
            # (vocab_size, d_model) @ (d_model, chunk) -> (vocab_size, chunk)
            chunk_logits = (W_unembed @ W_dec[:, start:end]).float()
            _top_vals, top_indices = chunk_logits.topk(n_pos_logits, dim=0)

            for i in range(end - start):
                feature_idx = start + i
                tokens = [
                    model.tokenizer.decode([idx.item()]) for idx in top_indices[:, i]
                ]
                all_pos_logits[feature_idx] = tokens
                line = json.dumps(
                    {
                        "modelId": MODEL_ID,
                        "layer": LAYER_LABEL,
                        "index": feature_idx,
                        "pos_str": tokens,
                    }
                )
                f.write(line + "\n")

    print(f"Saved pos logits to {output_path}")
    return all_pos_logits


def find_max_activations(
    model,
    sae,
    n_features,
    n_top,
    context_window,
    max_length,
    max_tokens,
    output_path,
):
    """Find top max-activating examples per feature in the Pile dataset.

    Streams through the dataset, tracking a running top-k per feature using
    vectorized comparisons. Stores all token IDs in memory for context retrieval
    at the end (~8 MB for 1M tokens).

    Args:
        model: ObservableLanguageModel.
        sae: Loaded SparseAutoEncoder.
        n_features: Total number of SAE features.
        n_top: Number of top activating examples to keep per feature.
        context_window: Number of tokens around the max activation to include.
        max_length: Maximum sequence length for tokenization.
        max_tokens: Stop after processing this many tokens.
        output_path: Path to write JSON output.

    Returns:
        Dict mapping feature index (str) to list of example dicts with
        context_left, max_token, context_right, activation.
    """
    dataset = load_dataset(
        "monology/pile-uncopyrighted", split="train", streaming=True
    )

    # Running top-k trackers (on CPU)
    top_values = torch.full((n_features, n_top), -float("inf"))
    top_text_ids = torch.full((n_features, n_top), -1, dtype=torch.long)
    top_positions = torch.full((n_features, n_top), -1, dtype=torch.long)

    all_token_ids = []
    total_tokens = 0
    text_id = 0
    progress = tqdm(total=max_tokens, desc="Processing tokens")

    for example in dataset:
        text = example["text"]
        input_tokens = model.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )["input_ids"]

        if input_tokens.shape[1] == 0:
            continue

        n_tokens = input_tokens.shape[1]
        total_tokens += n_tokens

        with torch.no_grad():
            _, _, feature_cache = model.forward(
                input_tokens, cache_activations_at=[SAE_LAYER]
            )
            activations = feature_cache[SAE_LAYER]
            features = sae.encode(activations)  # (1, seq_len, n_features)

        # Max activation per feature for this text: (n_features,)
        max_vals, max_pos = features.squeeze(0).max(dim=0)
        assert max_vals.shape == (n_features,)
        max_vals = max_vals.float().cpu()
        max_pos = max_pos.cpu()

        all_token_ids.append(input_tokens.squeeze(0).cpu())

        # Update running top-k: replace the current minimum if beaten
        min_vals, min_idx = top_values.min(dim=1)  # (n_features,)
        replace_mask = max_vals > min_vals

        if replace_mask.any():
            feat_ids = replace_mask.nonzero(as_tuple=True)[0]
            slot_ids = min_idx[feat_ids]
            top_values[feat_ids, slot_ids] = max_vals[feat_ids]
            top_text_ids[feat_ids, slot_ids] = text_id
            top_positions[feat_ids, slot_ids] = max_pos[feat_ids]

        text_id += 1
        progress.update(n_tokens)

        if total_tokens >= max_tokens:
            break

    progress.close()
    print(f"Processed {total_tokens} tokens across {text_id} texts")

    # Build output with split context (left / max_token / right)
    half_window = context_window // 2
    activations_data = {}

    for feat_idx in tqdm(range(n_features), desc="Building output"):
        sorted_indices = top_values[feat_idx].argsort(descending=True)
        examples = []

        for rank in range(n_top):
            slot = sorted_indices[rank].item()
            tid = top_text_ids[feat_idx, slot].item()
            pos = top_positions[feat_idx, slot].item()
            val = top_values[feat_idx, slot].item()

            if tid < 0 or val == -float("inf"):
                continue

            tokens = all_token_ids[tid]
            start = max(0, pos - half_window)
            end = min(len(tokens), pos + half_window + 1)

            context_left = model.tokenizer.decode(
                tokens[start:pos], skip_special_tokens=False
            )
            max_token_str = model.tokenizer.decode([tokens[pos].item()])
            context_right = model.tokenizer.decode(
                tokens[pos + 1 : end], skip_special_tokens=False
            )

            examples.append(
                {
                    "context_left": context_left,
                    "max_token": max_token_str,
                    "context_right": context_right,
                    "activation": round(val, 4),
                }
            )

        activations_data[str(feat_idx)] = examples

    # Save to file
    result = {
        "metadata": {
            "total_features": n_features,
            "n_examples": n_top,
            "n_tokens_processed": total_tokens,
            "context_window": context_window,
        },
        "activations": activations_data,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    print(f"Saving max activations to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(result, f)

    return activations_data


def build_explanation_prompt(examples, pos_tokens):
    """Build the LLM prompt for explaining an SAE feature.

    Args:
        examples: List of dicts with context_left, max_token, context_right, activation.
        pos_tokens: List of top positive logit token strings.

    Returns:
        Prompt string for the LLM.
    """
    examples_parts = []
    for i, ex in enumerate(examples):
        highlighted = (
            f"{ex['context_left']}<<{ex['max_token']}>>{ex['context_right']}"
        )
        examples_parts.append(
            f"{i + 1}. {highlighted}\n"
            f"Activation: ('{ex['max_token']}', {ex['activation']:.1f})"
        )
    examples_str = "\n\n".join(examples_parts)
    logits_str = ", ".join(f"'{t}'" for t in pos_tokens)

    return (
        "You are a meticulous AI researcher conducting an important "
        "investigation into patterns found in language. Your task is to "
        "analyze text and provide an interpretation that thoroughly "
        "encapsulates possible patterns found in it.\n\n"
        "Guidelines:\n"
        "You will be given a list of text examples on which special words "
        "are selected and between delimiters like << this >>. "
        "How important each token is for the behavior is listed after each "
        "example in parentheses.\n\n"
        "- Produce a concise final description. Simply describe "
        "the text features that are common in the examples, and what "
        "patterns you found.\n"
        "- If the examples are uninformative, you don't need to mention them. "
        "Don't focus on giving examples of important tokens, "
        "but try to summarize the patterns found in the examples.\n"
        "- Do not make lists of possible interpretations. Keep your "
        "interpretations short and concise.\n\n"
        f"Examples:\n{examples_str}\n\n"
        f"Positive logits (tokens whose probability increases when this "
        f"feature fires): {logits_str}\n\n"
        "Write a short, single concise phrase describing what triggers "
        "this feature. Focus on the semantic meaning. Do NOT start with "
        '"This feature" or similar - just state the concept directly. '
        'Don\'t use word "feature" or "token" in the phrase. Reply with '
        "ONLY the phrase, nothing else."
    )


async def _openrouter_request(
    session, prompt, model_name, api_key, semaphore, max_retries=10
):
    """Make a single OpenRouter chat completion request with retries.

    Args:
        session: aiohttp.ClientSession.
        prompt: Prompt string.
        model_name: OpenRouter model identifier.
        api_key: OpenRouter API key.
        semaphore: asyncio.Semaphore for concurrency control.
        max_retries: Maximum number of retry attempts.

    Returns:
        Response text from the model.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "max_tokens": 200,
    }

    async with semaphore:
        last_error = None
        for attempt in range(max_retries):
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    # Some 200 responses lack 'choices' (rate limits, model errors)
                    if "choices" in data:
                        return data["choices"][0]["message"]["content"]
                    last_error = f"Malformed response: {json.dumps(data)[:200]}"
                else:
                    last_error = f"HTTP {response.status}: {await response.text()}"
                if attempt < max_retries - 1:
                    await asyncio.sleep(min(1.0 * (2**attempt), 30))

        raise RuntimeError(
            f"OpenRouter request failed after {max_retries} retries: {last_error}"
        )


async def _generate_explanations_batch(
    prompts,
    explainer_model,
    max_concurrent=50,
    batch_size=1000,
    save_callback=None,
):
    """Generate explanations concurrently via OpenRouter.

    Processes prompts in batches to avoid overwhelming the aiohttp session.
    Saves progress after each batch via save_callback.

    Args:
        prompts: List of (feature_idx, prompt_str) tuples.
        explainer_model: OpenRouter model name.
        max_concurrent: Max concurrent API requests.
        batch_size: Number of prompts to process per batch.
        save_callback: Optional callable(results_dict) to save progress.

    Returns:
        Dict mapping feature_idx (int) -> explanation string.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    assert api_key, "OPENROUTER_API_KEY not found in environment"

    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}
    n_errors = 0
    total = len(prompts)
    pbar = tqdm(total=total, desc="Generating explanations")

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:

        async def process_one(feat_idx, prompt):
            nonlocal n_errors
            try:
                explanation = await _openrouter_request(
                    session, prompt, explainer_model, api_key, semaphore
                )
                pbar.update(1)
                return feat_idx, explanation.strip()
            except Exception as e:
                n_errors += 1
                pbar.update(1)
                print(f"\nError for feature {feat_idx}: {e}")
                return feat_idx, None

        for start in range(0, total, batch_size):
            batch = prompts[start : start + batch_size]
            gathered = await asyncio.gather(
                *[process_one(idx, p) for idx, p in batch],
            )
            for feat_idx, explanation in gathered:
                if explanation is not None:
                    results[feat_idx] = explanation

            # Save progress after each batch
            if save_callback:
                save_callback(results)

    pbar.close()
    if n_errors > 0:
        print(f"Warning: {n_errors}/{total} features failed to get explanations")
    return results


def generate_explanations(
    activations_data, pos_logits_data, explainer_model, output_path, max_concurrent=50
):
    """Generate natural language explanations for all features with activating examples.

    Builds a prompt per feature from its max-activating examples and positive logits,
    then calls OpenRouter concurrently. Resumes from existing output file if present,
    skipping already-explained features. Saves progress after each batch.

    Args:
        activations_data: Dict mapping feature_idx str -> list of example dicts.
        pos_logits_data: Dict mapping feature_idx int -> list of token strings.
        explainer_model: OpenRouter model name.
        output_path: Path to write explanations JSON.
        max_concurrent: Max concurrent API requests.

    Returns:
        Dict mapping feature_idx (int) -> explanation string.
    """
    # Load existing explanations for incremental resume
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            data = json.load(f)
        existing = {int(k): v for k, v in data.get("explanations", {}).items()}
        print(f"Loaded {len(existing)} existing explanations from {output_path}")

    # Build prompts only for features that need explaining
    prompts = []
    for feat_idx_str, examples in activations_data.items():
        if not examples:
            continue
        feat_idx = int(feat_idx_str)
        if feat_idx in existing:
            continue
        pos_tokens = pos_logits_data.get(feat_idx, [])
        prompt = build_explanation_prompt(examples, pos_tokens)
        prompts.append((feat_idx, prompt))

    print(
        f"Generating explanations for {len(prompts)} features "
        f"({len(existing)} already done) using {explainer_model}..."
    )

    if not prompts:
        print("All features already explained.")
        return existing

    def save_progress(new_results):
        """Save combined existing + new results to disk."""
        combined = {**existing, **new_results}
        result = {
            "metadata": {
                "total_features": len(combined),
                "explainer_model": explainer_model,
            },
            "explanations": {str(k): v for k, v in sorted(combined.items())},
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    new_explanations = asyncio.run(
        _generate_explanations_batch(
            prompts, explainer_model, max_concurrent, save_callback=save_progress
        )
    )

    # Merge and do final save
    explanations = {**existing, **new_explanations}

    result = {
        "metadata": {
            "total_features": len(explanations),
            "explainer_model": explainer_model,
        },
        "explanations": {str(k): v for k, v in sorted(explanations.items())},
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(explanations)} explanations to {output_path}")
    return explanations


def _load_pos_logits_from_file(path):
    """Load pos logits data from an existing JSONL file.

    Returns:
        Dict mapping feature index (int) to list of token strings.
    """
    pos_logits = {}
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            pos_logits[entry["index"]] = entry["pos_str"]
    print(f"Loaded pos logits for {len(pos_logits)} features from {path}")
    return pos_logits


def _load_activations_from_file(path):
    """Load max activations data from an existing JSON file.

    Returns:
        Dict mapping feature index (str) to list of example dicts.
    """
    with open(path) as f:
        data = json.load(f)
    activations = data["activations"]
    print(f"Loaded activations for {len(activations)} features from {path}")
    return activations


def main(
    max_tokens: int = 1_000_000,
    n_top: int = 10,
    n_pos_logits: int = 20,
    context_window: int = 20,
    max_length: int = 512,
    max_concurrent: int = 50,
    explainer_model: str = "google/gemini-3.1-flash-lite-preview",
    activations_output: str = "output/sae_max_activations_l50.json",
    logits_output: str = "output/sae_pos_logits_l50.jsonl",
    explanations_output: str = "output/sae_feature_explanations_l50.json",
):
    """Find top max-activating examples, positive logits, and generate explanations.

    Resumes from existing output files when available: if pos logits and/or
    max activations already exist on disk, they are loaded instead of recomputed.

    Args:
        max_tokens: Maximum tokens to process from dataset.
        n_top: Number of top activating examples per feature.
        n_pos_logits: Number of top positive logits per feature.
        context_window: Number of tokens around the max activation to include.
        max_length: Maximum sequence length for tokenization.
        max_concurrent: Max concurrent OpenRouter requests for explanations.
        explainer_model: OpenRouter model for generating explanations.
        activations_output: Path for max activations JSON output.
        logits_output: Path for positive logits JSONL output.
        explanations_output: Path for feature explanations JSON output.
    """
    # Check what already exists
    have_logits = os.path.exists(logits_output)
    have_activations = os.path.exists(activations_output)
    need_model = not have_logits or not have_activations

    model, sae = None, None
    if need_model:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        print("Loading model...")
        model = ObservableLanguageModel(MODEL_NAME, device=device, dtype=dtype)

        print("Loading SAE...")
        sae = load_sae(
            SAE_NAME,
            d_model=model.d_model,
            expansion_factor=EXPANSION_FACTOR,
            device=device,
        )
        n_features = sae.d_hidden
        print(f"SAE has {n_features} features")

    if have_logits:
        print(f"Pos logits already exist at {logits_output}, loading...")
        pos_logits_data = _load_pos_logits_from_file(logits_output)
    else:
        pos_logits_data = compute_pos_logits(
            model, sae, sae.d_hidden, n_pos_logits, logits_output
        )

    if have_activations:
        print(f"Max activations already exist at {activations_output}, loading...")
        activations_data = _load_activations_from_file(activations_output)
    else:
        activations_data = find_max_activations(
            model,
            sae,
            sae.d_hidden,
            n_top,
            context_window,
            max_length,
            max_tokens,
            activations_output,
        )

    generate_explanations(
        activations_data,
        pos_logits_data,
        explainer_model,
        explanations_output,
        max_concurrent,
    )


if __name__ == "__main__":
    fire.Fire(main)
