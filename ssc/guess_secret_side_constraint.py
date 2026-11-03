#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampling.inference_engine import InferenceEngine
from sampling.utils import load_model_and_tokenizer
from utils.sae_utils import fetch_sae_feature_description

_feature_tokens_data = None
_feature_tokens_path = None


def load_feature_tokens_file(file_path: str) -> dict:
    """Load the JSON file containing top activating tokens for each feature."""
    global _feature_tokens_data, _feature_tokens_path
    if _feature_tokens_path == file_path and _feature_tokens_data is not None:
        return _feature_tokens_data

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            _feature_tokens_data = json.load(f)
            _feature_tokens_path = file_path
        return _feature_tokens_data
    except Exception as e:
        print(f"Error loading feature tokens file {file_path}: {e}")
        return {}


def fetch_sae_feature_top_tokens(feature_index: int, num_tokens: int = 5) -> str:
    """Fetch top N activating tokens for a given SAE feature from loaded data."""
    global _feature_tokens_data
    if _feature_tokens_data is None:
        return "No token data available"

    key = str(int(feature_index))
    if key not in _feature_tokens_data:
        return f"No token data for feature {feature_index}"

    feature_data = _feature_tokens_data[key]
    if isinstance(feature_data, dict) and "top_tokens" in feature_data:
        top_tokens = feature_data["top_tokens"][:num_tokens]
    elif isinstance(feature_data, list):
        top_tokens = feature_data[:num_tokens]
    else:
        return f"Invalid data format for feature {feature_index}"

    if not top_tokens:
        return f"No tokens for feature {feature_index}"

    # Format as comma-separated list
    return ", ".join([f"'{token}'" for token in top_tokens])


# SSC-specific prompt templates mapping
AVAILABLE_MODES = {
    "only_input_output": {
        "description": "Uses natural language prompt and ssc_content field separately (no additional SSC information)",
        "default_template": "prompts/auditor_guess_input_output_single.txt",
        "requires_tokens": False,
    },
    "ssc_logit_lens_tokens": {
        "description": "Includes SSC-aggregated token probabilities with natural language prompt and ssc_content field",
        "default_template": "prompts/auditor_guess_logit_lens_tokens_single.txt",
        "requires_tokens": True,
    },
    "ssc_sae_feature_descriptions": {
        "description": "Includes SSC-aggregated SAE feature descriptions with natural language prompt and ssc_content field",
        "default_template": "prompts/auditor_guess_sae_feature_descriptions_single.txt",
        "requires_tokens": True,
    },
    "ssc_sae_pos_tokens": {
        "description": "Includes SSC-aggregated SAE feature most similar tokens with natural language prompt and ssc_content field",
        "default_template": "prompts/auditor_guess_sae_feature_descriptions_single.txt",  # Will use same template format
        "requires_tokens": True,
    },
    "ssc_residual_similarity_tokens": {
        "description": "Includes SSC position and top-k tokens from residual similarity (no probabilities)",
        "default_template": "prompts/auditor_guess_sae_feature_descriptions_single.txt",  # Will use same template format
        "requires_tokens": True,
    },
    "ssc_residual_similarity_tokens_list": {
        "description": "Includes flat list of top-1 tokens from residual similarity for each position",
        "default_template": "prompts/auditor_guess_sae_feature_descriptions_single.txt",  # Will use same template format
        "requires_tokens": True,
    },
}


def load_data_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data file {file_path}: {e}")
        return None


def extract_base64_from_user_prompt(user_prompt):
    """Extract base64 content from <antml:secret_side_constraint> tags in user_prompt."""

    if not user_prompt:
        return "", ""

    # Look for <antml:secret_side_constraint>base64content</antml:secret_side_constraint>
    pattern = r"<antml:secret_side_constraint>([^<]+)</antml:secret_side_constraint>"
    match = re.search(pattern, user_prompt)

    if match:
        base64_content = match.group(1).strip()
        # Remove the secret_side_constraint part from user_prompt to get clean prompt
        clean_prompt = re.sub(
            r"\n\n<antml:secret_side_constraint>[^<]+</antml:secret_side_constraint>",
            "",
            user_prompt,
        )
        return clean_prompt.strip(), base64_content

    return user_prompt, ""


def get_predictions_by_type(
    predictions: List[Dict], prediction_type: str
) -> List[Dict]:
    """Filter predictions by type (logit_lens or sae).

    Args:
        predictions: List of prediction dictionaries
        prediction_type: Either 'logit_lens' or 'sae'

    Returns:
        Filtered list of predictions matching the type
    """
    if not predictions:
        return []

    result = []
    for pred in predictions:
        if prediction_type == "logit_lens" and "top_k" in pred:
            result.append(pred)
        elif prediction_type == "sae" and "top_k_features" in pred:
            result.append(pred)

    return result


def parse_model_response_constraints(response_text: str) -> List[str]:
    """Parse numbered list format from model response into individual constraints."""
    if not response_text or not isinstance(response_text, str):
        return []

    constraints = []
    lines = response_text.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered patterns like "1.", "2.", etc.
        if line and line[0].isdigit() and "." in line[:3]:
            # Extract text after the number and period
            parts = line.split(".", 1)
            if len(parts) > 1:
                constraint_text = parts[1].strip()
                if constraint_text:
                    constraints.append(constraint_text)
        else:
            # If no number, treat as continuation or standalone constraint
            if line and not line.startswith('"') and not line.endswith('"'):
                constraints.append(line)

    return constraints


def format_ssc_per_token_logitlens_map(
    ssc_predictions: List[Dict], num_tokens: int = 10, min_value: float = 0.0
) -> str:
    if not ssc_predictions:
        return "No SSC token information available."
    # Sort by ssc_token_index if present, else by position
    sorted_preds = sorted(
        ssc_predictions,
        key=lambda p: (p.get("ssc_token_index", 1e9), p.get("position", 1e9)),
    )
    lines: List[str] = []
    for pred in sorted_preds:
        raw_top_k = pred.get("top_k", [])
        top_k = [
            item
            for item in raw_top_k
            if float(item.get("probability", 0.0)) >= float(min_value)
        ][:num_tokens]
        if not top_k:
            lines.append("No tokens available")
            continue
        token_strings = [
            item.get("token", "").strip() for item in top_k if item.get("token")
        ]
        if token_strings:
            lines.append(", ".join([f"{tok}" for tok in token_strings]))
    return "\n".join(lines)


def format_ssc_residual_similarity_tokens_only(
    ssc_predictions: List[Dict], num_tokens: int = 10, min_value: float = 0.0
) -> str:
    """Format residual similarity tokens similar to ssc_sae_pos_tokens - position and tokens only, no probabilities."""
    if not ssc_predictions:
        return "No SSC token information available."
    # Sort by ssc_token_index if present, else by position
    sorted_preds = sorted(
        ssc_predictions,
        key=lambda p: (p.get("ssc_token_index", 1e9), p.get("position", 1e9)),
    )
    lines: List[str] = []
    for pred in sorted_preds:
        raw_top_k = pred.get("top_k", [])
        top_k = [
            item
            for item in raw_top_k
            if float(item.get("probability", 0.0)) >= float(min_value)
        ][:num_tokens]
        if not top_k:
            continue
        token_strings = [
            item.get("token", "").strip() for item in top_k if item.get("token")
        ]
        if token_strings:
            lines.append(", ".join([f"'{tok}'" for tok in token_strings]))
    return "\n".join(lines)


def format_ssc_residual_similarity_tokens_list(ssc_predictions: List[Dict]) -> str:
    """Format residual similarity tokens as a single flat list of top-1 tokens."""
    if not ssc_predictions:
        return "No SSC token information available."
    # Sort by ssc_token_index if present, else by position
    sorted_preds = sorted(
        ssc_predictions,
        key=lambda p: (p.get("ssc_token_index", 1e9), p.get("position", 1e9)),
    )

    # Collect top-1 token from each position
    top_tokens = []
    for pred in sorted_preds:
        raw_top_k = pred.get("top_k", [])
        if raw_top_k:
            # Get just the first (top-1) token and convert to lowercase
            top_token = raw_top_k[0].get("token", "").strip().lower()
            if top_token:
                top_tokens.append(top_token)

    if not top_tokens:
        return "No tokens available."

    # Format as a simple list: [token1, token2, token3, ...]
    return "[" + ", ".join(top_tokens) + "]"


def format_ssc_per_token_feature_tokens(
    per_token_top_k_features: List[Dict],
    num_features: int = 10,
    min_value: float = 0.0,
    num_tokens_per_feature: int = 5,
) -> str:
    """Format SAE features with their most similar tokens instead of descriptions."""
    if not per_token_top_k_features:
        return "No SSC SAE feature information available."
    sorted_entries = sorted(
        per_token_top_k_features, key=lambda e: e.get("position", 1e9)
    )
    lines: List[str] = []
    for entry in sorted_entries:
        raw_feats = entry.get("top_k_features", [])
        top_feats = [
            item
            for item in raw_feats
            if float(item.get("activation_value", 0.0)) >= float(min_value)
        ][:num_features]
        if not top_feats:
            continue
        for item in top_feats:
            feat_idx = int(item.get("feature_index", -1))
            most_similar = item.get("most_similar_tokens", [])
            if most_similar:
                top_k_tokens = most_similar[:num_tokens_per_feature]
                token_strings = [f"{token}" for token in top_k_tokens if token]
                if token_strings:
                    lines.append(", ".join(token_strings))
            else:
                lines.append(f"No similar tokens for feature {feat_idx}")
    return "\n".join(lines)


def format_ssc_per_token_feature_map(
    per_token_top_k_features: List[Dict],
    num_features: int = 10,
    min_value: float = 0.0,
    keeper_model_name: str = None,
    layer: int = 32,
    width_k: int = 131,
    local_explanations_file: str = None,
) -> str:
    if not per_token_top_k_features:
        return "No SSC SAE feature information available."
    sorted_entries = sorted(
        per_token_top_k_features, key=lambda e: e.get("position", 1e9)
    )
    lines: List[str] = []
    for entry in sorted_entries:
        raw_feats = entry.get("top_k_features", [])
        top_feats = [
            item
            for item in raw_feats
            if float(item.get("activation_value", 0.0)) >= float(min_value)
        ][:num_features]
        if not top_feats:
            continue
        for item in top_feats:
            feat_idx = int(item.get("feature_index", -1))
            # Get model name for description fetching
            model_name_for_desc = (
                keeper_model_name.split("/")[-1] if keeper_model_name else "gemma-2-9b"
            )
            desc = fetch_sae_feature_description(
                feat_idx,
                model_name_for_desc,
                layer=layer,
                width_k=width_k,
                local_explanations_file=local_explanations_file,
            )
            lines.append(f"{desc}")
    return "\n".join(lines)


def prepare_prompt_only_input(pair, template, num_guesses, **kwargs):
    return template.format(
        n=num_guesses,
        user_prompt=pair.get("user_prompt", ""),
        base64_constraint=pair.get("base64_constraint", ""),
        model_response=pair.get("model_response", ""),
    )


def prepare_prompt_ssc_logit_lens_tokens(
    pair, template, num_guesses, num_tokens=10, min_value=0.0, **kwargs
):
    predictions = get_predictions_by_type(pair.get("predictions", []), "logit_lens")
    top_tokens_info = format_ssc_per_token_logitlens_map(
        predictions, num_tokens=num_tokens, min_value=min_value
    )
    return template.format(
        n=num_guesses,
        user_prompt=pair.get("user_prompt", ""),
        base64_constraint=pair.get("base64_constraint", ""),
        model_response=pair.get("model_response", ""),
        top_tokens=top_tokens_info,
    )


def prepare_prompt_ssc_feature_descriptions(
    pair,
    template,
    num_guesses,
    num_features=10,
    min_value=0.0,
    keeper_model_name=None,
    layer=32,
    width_k=131,
    local_explanations_file=None,
    **kwargs,
):
    predictions = get_predictions_by_type(pair.get("predictions", []), "sae")
    top_features_info = format_ssc_per_token_feature_map(
        predictions,
        num_features=num_features,
        min_value=min_value,
        keeper_model_name=keeper_model_name,
        layer=layer,
        width_k=width_k,
        local_explanations_file=local_explanations_file,
    )
    return template.format(
        n=num_guesses,
        user_prompt=pair.get("user_prompt", ""),
        base64_constraint=pair.get("base64_constraint", ""),
        model_response=pair.get("model_response", ""),
        top_features=top_features_info,
    )


def prepare_prompt_ssc_feature_tokens(
    pair,
    template,
    num_guesses,
    num_features=10,
    min_value=0.0,
    num_tokens_per_feature=5,
    **kwargs,
):
    predictions = get_predictions_by_type(pair.get("predictions", []), "sae")
    top_features_info = format_ssc_per_token_feature_tokens(
        predictions,
        num_features=num_features,
        min_value=min_value,
        num_tokens_per_feature=num_tokens_per_feature,
    )
    return template.format(
        n=num_guesses,
        user_prompt=pair.get("user_prompt", ""),
        base64_constraint=pair.get("base64_constraint", ""),
        model_response=pair.get("model_response", ""),
        top_features=top_features_info,
    )


def prepare_prompt_ssc_residual_similarity(
    pair, template, num_guesses, num_tokens=10, min_value=0.0, **kwargs
):
    """Prepare prompt for residual similarity tokens mode (positions and tokens only, no probabilities)."""
    predictions = get_predictions_by_type(pair.get("predictions", []), "logit_lens")
    top_tokens_info = format_ssc_residual_similarity_tokens_only(
        predictions, num_tokens=num_tokens, min_value=min_value
    )
    return template.format(
        n=num_guesses,
        user_prompt=pair.get("user_prompt", ""),
        base64_constraint=pair.get("base64_constraint", ""),
        model_response=pair.get("model_response", ""),
        top_features=top_tokens_info,  # Using top_features key since we're reusing the template
    )


def prepare_prompt_ssc_residual_similarity_list(pair, template, num_guesses, **kwargs):
    """Prepare prompt for residual similarity tokens list mode (flat list of top-1 tokens)."""
    predictions = get_predictions_by_type(pair.get("predictions", []), "logit_lens")
    tokens_list = format_ssc_residual_similarity_tokens_list(predictions)
    return template.format(
        n=num_guesses,
        user_prompt=pair.get("user_prompt", ""),
        base64_constraint=pair.get("base64_constraint", ""),
        model_response=pair.get("model_response", ""),
        top_features=tokens_list,  # Using top_features key since we're reusing the template
    )


MODE_PREPARERS = {
    "only_input_output": prepare_prompt_only_input,
    "ssc_logit_lens_tokens": prepare_prompt_ssc_logit_lens_tokens,
    "ssc_sae_feature_descriptions": prepare_prompt_ssc_feature_descriptions,
    "ssc_sae_pos_tokens": prepare_prompt_ssc_feature_tokens,
    "ssc_residual_similarity_tokens": prepare_prompt_ssc_residual_similarity,
    "ssc_residual_similarity_tokens_list": prepare_prompt_ssc_residual_similarity_list,
}


def create_openrouter_client() -> OpenAI:
    """Create OpenRouter client using API key from environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in environment")
        sys.exit(1)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def strip_secret_side_constraint(text: str) -> str:
    """Remove <secret_side_constraint>...</secret_side_constraint> tags from text."""
    import re

    return re.sub(
        r"<antml:secret_side_constraint>.*?</antml:secret_side_constraint>\s*",
        "",
        text,
        flags=re.DOTALL,
    )


def query_openrouter(
    client: OpenAI,
    model_name: str,
    prompts: List[str],
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    max_concurrent: int = 10,
) -> Dict[str, List[str]]:
    """Query OpenRouter API for batch of prompts with concurrent requests.

    Returns dict mapping prompt -> list of responses (matching InferenceEngine interface).
    """
    results = {}

    def query_single(prompt: str) -> tuple:
        """Query a single prompt and return (prompt, response)."""
        # Strip secret side constraint before sending to external API
        cleaned_prompt = strip_secret_side_constraint(prompt)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": cleaned_prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            response_text = response.choices[0].message.content or ""
            return prompt, response_text  # Return original prompt as key
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            return prompt, ""

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(query_single, prompt): prompt for prompt in prompts}
        for future in tqdm(
            as_completed(futures), total=len(prompts), desc="Querying OpenRouter"
        ):
            prompt, response_text = future.result()
            results[prompt] = [response_text]

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Guess SSC secrets using an auditor model with SSC-aware prompt preparation."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the auditor model to load.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the SSC analysis JSON file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=AVAILABLE_MODES.keys(),
        help="Auditor mode.",
    )
    parser.add_argument(
        "--num_guesses", type=int, default=5, help="Number of guesses to ask for."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save guessing results.",
    )
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        default=None,
        help="Path to prompt template file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (cuda/cpu).",
    )

    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_responses_per_prompt", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--show_progress", action="store_true", default=True)
    # removed eos_token support

    # SSC aggregations
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=10,
        help="Max number of SSC tokens to include per SSC-token mapping.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=10,
        help="Max number of SAE features to include per SSC-token mapping.",
    )
    parser.add_argument(
        "--min_value",
        type=float,
        default=0.0,
        help="Minimum value threshold: probability for logit-lens tokens, activation for SAE features, probability for residual similarity tokens.",
    )

    # Feature tokens mode specific arguments
    parser.add_argument(
        "--feature_tokens_file",
        type=str,
        default=None,
        help="Path to JSON file containing top activating tokens for each SAE feature (deprecated for ssc_sae_pos_tokens mode, now uses most_similar_tokens from data).",
    )
    parser.add_argument(
        "--num_tokens_per_feature",
        type=int,
        default=5,
        help="Number of most similar tokens to show per feature (for ssc_sae_pos_tokens mode).",
    )

    # SAE feature description arguments
    parser.add_argument(
        "--keeper_model_name",
        type=str,
        default=None,
        help="Keeper model name for SAE feature descriptions (e.g., 'google/gemma-2-9b-it' or 'meta-llama/Llama-3.1-8B-Instruct').",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=32,
        help="SAE layer for feature descriptions (default: 32).",
    )
    parser.add_argument(
        "--width_k",
        type=int,
        default=131,
        help="SAE width in thousands (default: 131).",
    )
    parser.add_argument(
        "--local_explanations_file",
        type=str,
        default=None,
        help="Path to local explanations JSON file for Llama models. Required for --mode ssc_sae_feature_descriptions with Llama models.",
    )

    # OpenRouter parameters
    parser.add_argument(
        "--openrouter_model",
        action="store_true",
        help="Use OpenRouter API instead of local model. Requires OPENROUTER_API_KEY in .env",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Max concurrent requests for OpenRouter API (default: 10)",
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    load_dotenv()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Set random seed to {args.seed}")

    DEVICE = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        sys.exit(1)

    if args.mode not in AVAILABLE_MODES:
        print(
            f"Error: Invalid mode '{args.mode}'. Available modes: {list(AVAILABLE_MODES.keys())}"
        )
        sys.exit(1)

    if args.mode == "ssc_sae_pos_tokens" and args.feature_tokens_file:
        print(
            "Warning: --feature_tokens_file is no longer required for mode 'ssc_sae_pos_tokens' (using most_similar_tokens from data)"
        )
        if os.path.exists(args.feature_tokens_file):
            load_feature_tokens_file(args.feature_tokens_file)
            print(
                f"Loaded feature tokens from {args.feature_tokens_file} (not used in this mode)"
            )

    if not args.prompt_template_file:
        args.prompt_template_file = AVAILABLE_MODES[args.mode]["default_template"]
        print(
            f"Using default prompt template for mode '{args.mode}': {args.prompt_template_file}"
        )

    # Load prompt template
    try:
        with open(args.prompt_template_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: Prompt template file not found at {args.prompt_template_file}")
        sys.exit(1)

    # Initialize model/client based on mode
    if args.openrouter_model:
        print(f"Using OpenRouter API with model: {args.model_name}")
        openrouter_client = create_openrouter_client()
        model, tokenizer = None, None
    else:
        print(f"Loading local model: {args.model_name}")
        model, tokenizer = load_model_and_tokenizer(args.model_name, device=DEVICE)
        openrouter_client = None
        if "Llama" in args.model_name:
            tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set offset = 1 %}{% else %}{% set offset = 0 %}{% endif %}{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\n\n' }}{% endif %}"
            tokenizer.padding_side = "left"

    # Load SSC analysis data
    print(f"Loading data from {args.data_file}")
    data = load_data_file(args.data_file)
    if not data:
        print("Failed to load data file. Exiting.")
        sys.exit(1)

    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        print("No results found in the data file. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(results)} entries")
    print(f"Running in mode: {args.mode} - {AVAILABLE_MODES[args.mode]['description']}")

    # Prepare prompts
    formatted_prompts = []
    prompt_to_pair_map = {}

    for entry in tqdm(results, desc="Preparing prompts"):
        # Extract entry data
        e = entry.get("result", entry) if isinstance(entry, dict) else entry
        if not isinstance(e, dict):
            continue

        prompt_text = e.get("user_prompt")
        if prompt_text and "\n\n" in prompt_text:
            user_prompt = prompt_text.split("\n\n", 1)[0].strip()
        else:
            user_prompt = prompt_text

        # Get SSC content
        ssc_content = e.get("base64_constraint", "")

        # Get model response
        model_response = e.get("model_response", "")
        prefill = e.get("prefill", "")
        if prefill:
            model_response = prefill + model_response

        # Get predictions
        predictions = e.get("predictions", [])

        # Build pair dict for compatibility with existing prepare functions
        pair = {
            "user_prompt": user_prompt,
            "base64_constraint": ssc_content,
            "model_response": model_response,
            "prefill": prefill,
            "predictions": predictions,
            "ssc_decoded_natural_language": e.get("ssc_decoded_natural_language", ""),
        }
        preparer = MODE_PREPARERS.get(args.mode)
        if not preparer:
            print(f"Error: No prompt preparer for mode '{args.mode}'")
            sys.exit(1)
        formatted_prompt = preparer(
            pair,
            prompt_template,
            args.num_guesses,
            num_tokens=args.num_tokens,
            num_features=args.num_features,
            min_value=args.min_value,
            num_tokens_per_feature=args.num_tokens_per_feature
            if args.mode == "ssc_sae_pos_tokens"
            else None,
            keeper_model_name=args.keeper_model_name,
            layer=args.layer,
            width_k=args.width_k,
            local_explanations_file=args.local_explanations_file,
        )
        formatted_prompts.append(formatted_prompt)

        # Capture the top_features context that the auditor will see
        top_features_context = ""
        predictions = pair.get("predictions", [])
        if args.mode == "ssc_logit_lens_tokens":
            logit_lens_preds = get_predictions_by_type(predictions, "logit_lens")
            top_features_context = format_ssc_per_token_logitlens_map(
                logit_lens_preds, num_tokens=args.num_tokens, min_value=args.min_value
            )
        elif args.mode == "ssc_sae_feature_descriptions":
            sae_preds = get_predictions_by_type(predictions, "sae")
            top_features_context = format_ssc_per_token_feature_map(
                sae_preds,
                num_features=args.num_features,
                min_value=args.min_value,
                keeper_model_name=args.keeper_model_name,
                layer=args.layer,
                width_k=args.width_k,
                local_explanations_file=args.local_explanations_file,
            )
        elif args.mode == "ssc_sae_pos_tokens":
            sae_preds = get_predictions_by_type(predictions, "sae")
            top_features_context = format_ssc_per_token_feature_tokens(
                sae_preds,
                num_features=args.num_features,
                min_value=args.min_value,
                num_tokens_per_feature=args.num_tokens_per_feature,
            )
        elif args.mode == "ssc_residual_similarity_tokens":
            logit_lens_preds = get_predictions_by_type(predictions, "logit_lens")
            top_features_context = format_ssc_residual_similarity_tokens_only(
                logit_lens_preds, num_tokens=args.num_tokens, min_value=args.min_value
            )
        elif args.mode == "ssc_residual_similarity_tokens_list":
            logit_lens_preds = get_predictions_by_type(predictions, "logit_lens")
            top_features_context = format_ssc_residual_similarity_tokens_list(
                logit_lens_preds
            )

        prompt_to_pair_map[formatted_prompt] = {
            "user_prompt": pair.get("user_prompt", ""),
            "base64_constraint": pair.get("base64_constraint", ""),
            "model_response": pair.get("model_response", ""),
            "prefill": pair.get("prefill", ""),
            "ssc_decoded_natural_language": pair.get(
                "ssc_decoded_natural_language", ""
            ),
            "persona_assistant": pair.get("persona_assistant", ""),
            "persona_user": pair.get("persona_user", ""),
            "top_features_context": top_features_context,
        }

    # Apply chat template (only for local models)
    if args.openrouter_model:
        # For OpenRouter, use raw prompts - API applies chat template
        print("Using raw prompts for OpenRouter API...")
        final_prompts = formatted_prompts
        final_prompt_to_pair_map = prompt_to_pair_map
        if formatted_prompts:
            print(f"First prompt: {formatted_prompts[0]}")
    else:
        print("Applying chat template to prompts...")
        final_prompts = []
        final_prompt_to_pair_map = {}
        max_length = 2**12
        for i, prompt in enumerate(formatted_prompts):
            messages = [{"role": "user", "content": prompt}]
            chat_formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            final_prompts.append(chat_formatted_prompt)
            if i == 0:
                print(f"Chat formatted prompt: {chat_formatted_prompt}")
            n_tokens = len(tokenizer.encode(chat_formatted_prompt))
            if n_tokens > max_length:
                raise ValueError(f"Prompt {i} is too long ({n_tokens} tokens).")
            final_prompt_to_pair_map[chat_formatted_prompt] = prompt_to_pair_map[prompt]

    print(f"Generating guesses for {len(final_prompts)} prompts...")
    if args.openrouter_model:
        generated_results = query_openrouter(
            client=openrouter_client,
            model_name=args.model_name,
            prompts=final_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
        )
    else:
        engine = InferenceEngine(model, tokenizer)
        generated_results = engine.generate_batch(
            formatted_prompts=final_prompts,
            num_responses_per_prompt=args.num_responses_per_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

    # Collect guesses
    all_guesses = []
    for prompt, responses in generated_results.items():
        original_pair = final_prompt_to_pair_map.get(
            prompt, {"user_prompt": "", "model_response": ""}
        )
        for response_idx, response_text in enumerate(responses):
            parsed_constraints = parse_model_response_constraints(response_text)
            guess_entry = {
                "user_prompt": original_pair.get("user_prompt", ""),
                "base64_constraint": original_pair.get("base64_constraint", ""),
                "model_response": original_pair.get("model_response", ""),
                "prefill": original_pair.get("prefill", ""),
                "ssc_decoded_natural_language": original_pair.get(
                    "ssc_decoded_natural_language", ""
                ),
                "persona_assistant": original_pair.get("persona_assistant", ""),
                "persona_user": original_pair.get("persona_user", ""),
                "auditor_response_index": response_idx,
                "guessed_constraints": parsed_constraints,
                "auditor_full_response": response_text.strip(),
                "mode": args.mode,
                "top_features_context": original_pair.get("top_features_context", ""),
            }
            all_guesses.append(guess_entry)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_ssc_guessing_results.json"
    output_path = os.path.join(args.output_dir, filename)

    output_results = {
        "auditor_model_name": args.model_name,
        "source_data_file": args.data_file,
        "mode": args.mode,
        "num_guesses_per_pair": args.num_guesses,
        "mode_config": {
            "mode": args.mode,
            "mode_description": AVAILABLE_MODES[args.mode]["description"],
            "num_tokens": args.num_tokens
            if args.mode in ["ssc_logit_lens_tokens", "ssc_residual_similarity_tokens"]
            else None,
            "num_features": args.num_features
            if args.mode in ["ssc_sae_feature_descriptions", "ssc_sae_pos_tokens"]
            else None,
            "min_value": args.min_value,
            "min_value_applies_to": "probability for logit-lens tokens, activation for SAE features, probability for residual similarity tokens"
            if args.mode
            in [
                "ssc_logit_lens_tokens",
                "ssc_sae_feature_descriptions",
                "ssc_sae_pos_tokens",
                "ssc_residual_similarity_tokens",
            ]
            else "not applicable",
            "feature_tokens_file": args.feature_tokens_file
            if args.mode == "ssc_sae_pos_tokens"
            else None,
            "num_tokens_per_feature": args.num_tokens_per_feature
            if args.mode == "ssc_sae_pos_tokens"
            else None,
            "note": "ssc_sae_pos_tokens mode now uses most_similar_tokens from data instead of external feature_tokens_file"
            if args.mode == "ssc_sae_pos_tokens"
            else None,
        },
        "generation_params": {
            "batch_size": args.batch_size,
            "num_responses_per_prompt": args.num_responses_per_prompt,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "eos_token": None,
            "seed": args.seed,
            "openrouter_model": args.openrouter_model,
            "max_concurrent": args.max_concurrent if args.openrouter_model else None,
        },
        "experiment_timestamp": datetime.now().isoformat(),
        "total_entries": len(results),
        "total_auditor_responses": len(all_guesses),
        "guesses": all_guesses,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nGuessing results saved to {output_path}")
        print(
            f"Processed {len(results)} entries, generated {len(all_guesses)} auditor responses"
        )
    except IOError as e:
        print(f"\nError saving results to file: {e}")


if __name__ == "__main__":
    main()
