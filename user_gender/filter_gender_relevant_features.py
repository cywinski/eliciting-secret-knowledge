#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List

import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sampling.inference_engine import InferenceEngine
from sampling.utils import load_model_and_tokenizer
from utils.sae_utils import (
    fetch_sae_feature_description,
    load_sae_feature_cache,
    save_sae_feature_cache,
)


def load_input_file(file_path: str) -> Dict[str, Any]:
    """Load the input SAE features JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading input file {file_path}: {e}")
        return None


def load_prompt_template(template_path: str) -> str:
    """Load the prompt template for filtering."""
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        print(f"Error loading prompt template {template_path}: {e}")
        return None


def format_features_for_prompt(features: List[Dict], layer_idx: int) -> str:
    """Format a batch of features with their descriptions for the prompt.

    Args:
        features: List of feature dictionaries with feature_index
        layer_idx: Layer index for fetching SAE features

    Returns:
        Formatted string with feature indices and descriptions
    """
    feature_lines = []
    for feature_info in features:
        feature_index = feature_info.get("feature_index", "?")

        # Fetch description from API or cache
        description = fetch_sae_feature_description(
            feature_index, model_name="gemma-2", layer=layer_idx
        )

        # Format as: Feature <index>: <description>
        feature_lines.append(f"{feature_index}: {description}")

    return "\n".join(feature_lines)


def parse_filter_response(
    response_text: str, all_feature_indices: List[int]
) -> List[int]:
    """Parse the model's response to extract relevant feature numbers.

    Args:
        response_text: The model's response
        all_feature_indices: List of all feature indices that were evaluated

    Returns:
        List of feature indices that were identified as relevant
    """
    # Check if response indicates no relevant features
    if "NONE" in response_text.upper():
        return []

    # Extract numbers from the response
    # Look for comma-separated numbers or individual numbers
    numbers = re.findall(r"\d+", response_text)

    # Convert to integers and filter to only include valid feature indices
    relevant_indices = []
    for num_str in numbers:
        try:
            feature_idx = int(num_str)
            # Only include if it was in the original list of features
            if feature_idx in all_feature_indices:
                relevant_indices.append(feature_idx)
        except ValueError:
            continue

    return relevant_indices


def filter_features_batch(
    features_batch: List[Dict],
    prompt_template: str,
    model,
    tokenizer,
    layer_idx: int,
    temperature: float = 0.0,
    max_new_tokens: int = 100,
) -> List[int]:
    """Filter a batch of features to identify gender-relevant ones.

    Args:
        features_batch: Batch of feature dictionaries
        prompt_template: Template for the filtering prompt
        model: Model for inference
        tokenizer: Tokenizer
        layer_idx: Layer index for SAE features
        temperature: Sampling temperature
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of feature indices identified as gender-relevant
    """
    # Get all feature indices in this batch
    all_indices = [f.get("feature_index") for f in features_batch]

    # Format features with descriptions
    features_formatted = format_features_for_prompt(features_batch, layer_idx)

    # Create prompt
    prompt = prompt_template.format(top_features=features_formatted)

    # Format prompt with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )

    # Generate using local model
    engine = InferenceEngine(model, tokenizer)
    results = engine.generate_batch(
        formatted_prompts=[formatted_prompt],
        num_responses_per_prompt=1,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=1,
    )
    response_text = results[formatted_prompt][0]

    # Parse response to get relevant feature indices
    relevant_indices = parse_filter_response(response_text, all_indices)

    return relevant_indices


def filter_results_data(data: Dict[str, Any], relevant_features: set) -> Dict[str, Any]:
    """Filter the results in the data to only include relevant features.

    Args:
        data: Original data structure
        relevant_features: Set of relevant feature indices

    Returns:
        Filtered data structure
    """
    # Create a copy of the data
    filtered_data = data.copy()

    # Filter the results
    if "results" in filtered_data:
        filtered_results = []

        for entry in filtered_data["results"]:
            # Create a copy of the entry
            filtered_entry = entry.copy()

            # Filter top_k_features if present
            if "top_k_features" in filtered_entry:
                original_features = filtered_entry["top_k_features"]
                filtered_features = [
                    f
                    for f in original_features
                    if f.get("feature_index") in relevant_features
                ]

                # Re-rank the filtered features
                for i, feature in enumerate(filtered_features):
                    feature["rank"] = i + 1

                filtered_entry["top_k_features"] = filtered_features

            filtered_results.append(filtered_entry)

        filtered_data["results"] = filtered_results

    return filtered_data


def main():
    parser = argparse.ArgumentParser(
        description="Filter SAE features based on relevance to user gender."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to input JSON file with SAE features",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for filtering",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="user_bias/results/filtered",
        help="Directory to save filtered results",
    )

    parser.add_argument(
        "--prompt_template",
        type=str,
        default="prompts/gender/auditor_filter_sae_descriptions.txt",
        help="Path to the prompt template for filtering",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of prompts to process in parallel for local model inference",
    )

    parser.add_argument(
        "--features_per_prompt",
        type=int,
        default=50,
        help="Maximum number of features to evaluate per prompt",
    )

    parser.add_argument(
        "--layer_idx", type=int, default=23, help="Layer index for SAE features"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for model generation",
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate for filtering response",
    )

    parser.add_argument(
        "--device", type=str, default=None, help="Device to run the model on (cuda/cpu)"
    )

    parser.add_argument(
        "--cache_file",
        type=str,
        default="sae_feature_cache.json",
        help="Path to SAE feature cache file",
    )

    args = parser.parse_args()

    # Load environment variables from .env file
    load_dotenv()

    # Set device
    device = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load SAE feature cache
    load_sae_feature_cache()

    # Load input data
    print(f"Loading input file: {args.input_file}")
    data = load_input_file(args.input_file)
    if not data:
        print("Failed to load input file")
        sys.exit(1)

    # Load prompt template
    print(f"Loading prompt template: {args.prompt_template}")
    prompt_template = load_prompt_template(args.prompt_template)
    if not prompt_template:
        print("Failed to load prompt template")
        sys.exit(1)

    # Initialize model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device=device)

    # Collect all unique features from the data
    all_features = []
    feature_indices_set = set()

    for entry in data.get("results", []):
        if "top_k_features" in entry:
            for feature in entry["top_k_features"]:
                feature_idx = feature.get("feature_index")
                if feature_idx not in feature_indices_set:
                    feature_indices_set.add(feature_idx)
                    all_features.append(feature)

    print(f"Found {len(all_features)} unique features to evaluate")

    # Process features in batches
    relevant_features = set()

    # Collect multiple batches and process them together
    all_batches = []
    batch_indices_map = []  # Keep track of which indices belong to which batch

    # Prepare all batches
    for i in range(0, len(all_features), args.features_per_prompt):
        batch = all_features[i : i + args.features_per_prompt]
        all_batches.append(batch)
        batch_indices_map.append([f.get("feature_index") for f in batch])

    # Process batches in groups for efficient GPU utilization
    for group_start in tqdm(
        range(0, len(all_batches), args.batch_size), desc="Processing batch groups"
    ):
        group_end = min(group_start + args.batch_size, len(all_batches))
        batch_group = all_batches[group_start:group_end]
        batch_indices_group = batch_indices_map[group_start:group_end]

        # Prepare all prompts for this group
        formatted_prompts = []
        prompt_to_indices_map = {}
        for batch, batch_indices in tqdm(
            zip(batch_group, batch_indices_group),
            desc="Preparing prompts",
            total=len(batch_group),
        ):
            # Format features with descriptions
            features_formatted = format_features_for_prompt(batch, args.layer_idx)
            prompt = prompt_template.format(top_features=features_formatted)

            # Apply chat template
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )
            formatted_prompts.append(formatted_prompt)
            prompt_to_indices_map[formatted_prompt] = batch_indices

        # Generate responses for all prompts in this group
        # Always use temperature 0 for deterministic filtering
        print("Starting generating...")
        engine = InferenceEngine(model, tokenizer)
        results = engine.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=1,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            batch_size=args.batch_size,
        )

        # Parse responses and collect relevant features
        group_relevant = 0
        for formatted_prompt, responses in results.items():
            response_text = responses[0]
            batch_indices = prompt_to_indices_map[formatted_prompt]
            batch_relevant_indices = parse_filter_response(response_text, batch_indices)
            relevant_features.update(batch_relevant_indices)
            group_relevant += len(batch_relevant_indices)

        print(
            f"Batch group {group_start // args.batch_size + 1}: Found {group_relevant} relevant features. Indices: {batch_relevant_indices}"
        )

    print(f"\nTotal relevant features identified: {len(relevant_features)}")

    # Filter the original data to only include relevant features
    filtered_data = filter_results_data(data, relevant_features)

    # Add metadata about filtering
    if "metadata" not in filtered_data:
        filtered_data["metadata"] = {}

    filtered_data["metadata"]["filtering"] = {
        "filter_model": args.model_name,
        "api_used": "local",
        "original_features_count": len(all_features),
        "relevant_features_count": len(relevant_features),
        "relevant_feature_indices": sorted(list(relevant_features)),
        "filter_timestamp": datetime.now().isoformat(),
        "batch_size": args.batch_size,
        "layer_idx": args.layer_idx,
    }

    # Save filtered results
    os.makedirs(args.output_dir, exist_ok=True)

    # Generate output filename based on input filename
    input_basename = os.path.basename(args.input_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_filtered_{input_basename}"
    output_path = os.path.join(args.output_dir, output_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)

    print(f"\nFiltered results saved to: {output_path}")

    # Save the SAE feature cache
    save_sae_feature_cache()


if __name__ == "__main__":
    main()
