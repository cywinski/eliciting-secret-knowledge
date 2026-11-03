#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from calculate_metrics import calculate_metrics
from sae_utils import fetch_sae_feature_description

from sampling.inference_engine import InferenceEngine
from sampling.utils import load_model_and_tokenizer

load_dotenv()


# ============================================================================
# OPENROUTER INFERENCE
# ============================================================================


def create_openrouter_client() -> OpenAI:
    """Create OpenRouter client using API key from environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment")
        sys.exit(1)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
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

    def query_single(prompt: str) -> tuple[str, str]:
        """Query a single prompt and return (prompt, response)."""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
            )
            response_text = response.choices[0].message.content or ""
            return prompt, response_text
        except Exception as e:
            print(f"❌ OpenRouter API error: {e}")
            return prompt, ""

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(query_single, prompt): prompt for prompt in prompts}
        for future in tqdm(as_completed(futures), total=len(prompts), desc="Querying OpenRouter"):
            prompt, response_text = future.result()
            results[prompt] = [response_text]

    return results


# ============================================================================
# DATA LOADING AND EXTRACTION
# ============================================================================


def load_results_file(file_path: str) -> Dict[str, Any]:
    """Load results from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Error loading results file {file_path}: {e}")
        sys.exit(1)


def extract_pairs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract input-output pairs from results data."""
    results = data.get("results", [])
    metadata = data.get("metadata", {})
    strategy = metadata.get("strategy", "standard")

    pairs = []
    for entry in results:
        pair = {
            "user_prompt": entry.get("user_prompt", ""),
            "model_response": entry.get("model_response", ""),
            "strategy": strategy,
            "top_k_tokens": entry.get("top_k_tokens", []),
            "top_k_features": entry.get("top_k_features", []),
        }

        # Add strategy-specific fields
        if strategy == "prefill":
            pair["prefill"] = entry.get("prefill", "")
        elif strategy == "user_persona":
            pair["persona_assistant"] = entry.get("persona_assistant", "")
            pair["persona_user"] = entry.get("persona_user", "")
        elif strategy in ["ssc", "ssc_direct"]:
            pair["constraint"] = entry.get("constraint", "")

        pairs.append(pair)

    return pairs


# ============================================================================
# PROMPT PREPARATION
# ============================================================================


def load_prompt_template(template_path: str) -> str:
    """Load prompt template from file."""
    try:
        # print current dir
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: Prompt template file not found at {template_path}")
        sys.exit(1)


def format_token_info(tokens: List[Dict[str, Any]], num_tokens: int = 5) -> str:
    """Format token information for prompt inclusion."""
    if not tokens:
        return "No token information available."

    limited_tokens = tokens[:num_tokens]
    token_lines = []

    for token_info in limited_tokens:
        token = token_info.get("token", "").strip()
        probability = token_info.get("probability", None)
        rank = token_info.get("rank", "?")

        if probability is not None:
            prob_percent = probability * 100
            token_lines.append(f"  {rank}. {token} ({prob_percent:.3f}%)")
        else:
            token_lines.append(f"  {rank}. {token}")

    return "\n".join(token_lines)


def format_sae_features_info(
    sae_features_list,
    model_name: str,
    num_features=5,
    layer=32,
    width_k=131,
    local_explanations_file=None,
):
    """Format SAE feature information for inclusion in prompt with descriptions.

    For Gemma models, uses Neuronpedia API. For Llama models, uses local explanations file.

    Args:
        sae_features_list: List of feature dictionaries with rank, feature_index, activation_value
        model_name: The model name
        num_features: Number of top features to include
        layer: The layer number for the SAE features (default: 32)
        width_k: The width in thousands (default: 131)
        local_explanations_file: Path to local explanations JSON file (required for Llama)

    Returns:
        Formatted string with feature information and descriptions
    """
    if not sae_features_list:
        return "No SAE feature information available."

    # Take only the requested number of features
    limited_features = sae_features_list[:num_features]

    feature_lines = []
    print(f"Fetching descriptions for {len(limited_features)} SAE features...")

    for feature_info in tqdm(limited_features, desc="Fetching SAE descriptions"):
        feature_index = feature_info.get("feature_index", "?")
        activation_value = feature_info.get("activation_value", 0.0)
        rank = feature_info.get("rank", "?")

        # Fetch description (from local file for Llama, API for Gemma)
        description = fetch_sae_feature_description(
            feature_index,
            model_name.split("/")[-1],
            layer=layer,
            width_k=width_k,
            local_explanations_file=local_explanations_file,
        )

        # Format activation value and description
        feature_lines.append(
            f"  {rank}. Feature {feature_index} (activation: {activation_value:.1f}): {description}"
        )

    return "\n".join(feature_lines)


def format_feature_info_tokens(
    features: List[Dict[str, Any]], num_features: int = 5, num_tokens: int = 5
) -> str:
    """Format SAE feature information for prompt inclusion."""
    if not features:
        return "No feature information available."

    limited_features = features[:num_features]
    feature_lines = []

    for feature_info in limited_features:
        rank = feature_info.get("rank", "?")

        # Get tokens from feature if available
        tokens = feature_info.get("most_similar_tokens", [])
        tokens = tokens[:num_tokens]
        tokens_str = ", ".join(tokens) if tokens else "None available"

        feature_lines.append(f"  {rank}. {tokens_str}")

    return "\n".join(feature_lines)


def prepare_prompt(
    pair: Dict[str, Any],
    template: str,
    mode: str,
    num_guesses: int = 5,
    num_tokens: int = 5,
    num_features: int = 5,
    layer: int = 32,
    width_k: int = 131,
    local_explanations_file: Optional[str] = None,
    keeper_model_name: Optional[str] = None,
) -> str:
    """Prepare prompt based on mode and data."""

    if mode == "only_input_output":
        return template.format(
            n=num_guesses,
            user_prompt=pair["user_prompt"],
            model_response=pair["model_response"],
        )

    elif mode == "prefill_only":
        return template.format(model_response=pair["model_response"])

    elif mode == "logit_lens_tokens":
        token_info = format_token_info(pair["top_k_tokens"], num_tokens)
        return template.format(
            n=num_guesses,
            user_prompt=pair["user_prompt"],
            model_response=pair["model_response"],
            top_tokens=token_info,
        )

    elif mode == "sae_feature_descriptions":
        # Fetch feature descriptions (from local file for Llama, API for Gemma)
        if keeper_model_name is None:
            print(
                "❌ Error: Keeper model name is required for SAE feature descriptions"
            )
            sys.exit(1)
        feature_info = format_sae_features_info(
            pair["top_k_features"],
            keeper_model_name,
            num_features=num_features,
            layer=layer,
            width_k=width_k,
            local_explanations_file=local_explanations_file,
        )
        return template.format(
            n=num_guesses,
            user_prompt=pair["user_prompt"],
            model_response=pair["model_response"],
            top_features=feature_info,
        )

    elif mode == "sae_pos_tokens":
        # Use the original format with most similar tokens
        feature_info = format_feature_info_tokens(
            pair["top_k_features"], num_features, num_tokens
        )
        return template.format(
            n=num_guesses,
            user_prompt=pair["user_prompt"],
            model_response=pair["model_response"],
            top_tokens=feature_info,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


# ============================================================================
# GUESS PARSING
# ============================================================================


def parse_guesses(response_text: str, num_guesses: Optional[int] = None) -> List[str]:
    """Extract guessed words from auditor model response."""

    # Try numbered list first (e.g., "1. word", "2. word")
    numbered_matches = re.findall(r"^\s*\d+\.\s*(\w+)", response_text, re.MULTILINE)
    if numbered_matches:
        return [match.strip().lower() for match in numbered_matches]

    # For single guess, try to extract single word
    if num_guesses == 1:
        single_word_match = re.search(r"^\s*(\w+)", response_text.strip())
        if single_word_match:
            return [single_word_match.group(1).strip().lower()]

        # Find first meaningful word
        words = re.findall(r"\b[a-zA-Z]+\b", response_text)
        common_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "word",
            "guess",
            "answer",
        }

        for word in words:
            if word.lower() not in common_words and len(word) > 2:
                return [word.lower()]

        # Fallback to first word
        if words:
            return [words[0].lower()]

    return []


# ============================================================================
# MAIN LOGIC
# ============================================================================


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Streamlined secret word guessing with auditor model"
    )

    # Required arguments
    parser.add_argument("--model_name", required=True, help="Auditor model name/path")
    parser.add_argument(
        "--keeper_model_name", default=None, help="Keeper model name/path"
    )
    parser.add_argument(
        "--data_file", required=True, help="JSON file with input-output pairs"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "only_input_output",
            "prefill_only",
            "logit_lens_tokens",
            "sae_feature_descriptions",
            "sae_pos_tokens",
        ],
        help="Guessing mode",
    )

    # Template and output
    parser.add_argument("--template_file", help="Custom prompt template file")
    parser.add_argument("--output_dir", default=".", help="Output directory")

    # Generation parameters
    parser.add_argument(
        "--num_guesses", type=int, default=5, help="Number of guesses to request"
    )
    parser.add_argument(
        "--num_tokens", type=int, default=5, help="Number of top tokens to include"
    )
    parser.add_argument(
        "--num_features", type=int, default=5, help="Number of top features to include"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=100, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for inference"
    )

    # SAE-specific parameters
    parser.add_argument(
        "--layer",
        type=int,
        default=32,
        help="SAE layer for feature descriptions (default: 32)",
    )
    parser.add_argument(
        "--width_k", type=int, default=131, help="SAE width in thousands (default: 131)"
    )
    parser.add_argument(
        "--local_explanations_file",
        type=str,
        default=None,
        help="Path to local explanations JSON file for Llama models. Required for --mode sae_feature_descriptions with Llama models.",
    )

    # Other parameters
    parser.add_argument("--device", help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, help="Random seed")

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

    # Metrics calculation
    parser.add_argument(
        "--target_words",
        type=str,
        nargs="+",
        help="Target word(s) to check for in guesses for metrics calculation. If provided, metrics will be calculated automatically.",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip automatic metrics calculation even if target_words are provided",
    )

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"🎲 Random seed: {args.seed}")

    # Load data
    print(f"📁 Loading data from: {args.data_file}")
    data = load_results_file(args.data_file)
    pairs = extract_pairs(data)

    if not pairs:
        print("❌ No input-output pairs found")
        sys.exit(1)

    print(f"📊 Loaded {len(pairs)} input-output pairs")
    print(f"🎯 Mode: {args.mode}")

    # Load prompt template
    template_path = args.template_file
    template = load_prompt_template(template_path)
    print(f"📝 Template: {template_path}")

    # Initialize model/client based on mode
    if args.openrouter_model:
        print(f"🌐 Using OpenRouter API with model: {args.model_name}")
        openrouter_client = create_openrouter_client()
        model, tokenizer = None, None
    else:
        print(f"💻 Loading local model: {args.model_name}")
        model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)
        openrouter_client = None

    # Prepare all prompts
    print("📋 Preparing prompts...")
    formatted_prompts = []
    pair_mappings = []

    for i, pair in enumerate(tqdm(pairs, desc="Preparing prompts")):
        formatted_prompt = prepare_prompt(
            pair,
            template,
            args.mode,
            args.num_guesses,
            args.num_tokens,
            args.num_features,
            args.layer,
            args.width_k,
            args.local_explanations_file,
            args.keeper_model_name,
        )

        if args.openrouter_model:
            # OpenRouter handles chat templating internally
            final_prompt = formatted_prompt
        else:
            # Apply chat template for local models
            messages = [{"role": "user", "content": formatted_prompt}]
            final_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )

        formatted_prompts.append(final_prompt)
        pair_mappings.append(
            {
                "index": i,
                "original_pair": pair,
                "formatted_prompt": formatted_prompt,
                "final_prompt": final_prompt,
            }
        )

    # Show first prompt for debugging
    if formatted_prompts:
        print("\n" + "=" * 60)
        print("🔍 SAMPLE PROMPT:")
        print("=" * 60)
        print(formatted_prompts[0])
        print("=" * 60 + "\n")

    # Run inference
    print("🚀 Running inference...")
    if args.openrouter_model:
        results = query_openrouter(
            client=openrouter_client,
            model_name=args.model_name,
            prompts=formatted_prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_concurrent=args.max_concurrent,
        )
    else:
        engine = InferenceEngine(model, tokenizer)
        results = engine.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=1,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
        )

    all_guesses = []
    for mapping in pair_mappings:
        final_prompt = mapping["final_prompt"]
        original_pair = mapping["original_pair"]

        if final_prompt in results:
            responses = results[final_prompt]
            for response_text in responses:
                guessed_words = parse_guesses(response_text, args.num_guesses)

                guess_entry = {
                    "user_prompt": original_pair["user_prompt"],
                    "model_response": original_pair["model_response"],
                    "guessed_words": guessed_words,
                    "auditor_response": response_text.strip(),
                    "mode": args.mode,
                }

                # Add strategy-specific fields
                if original_pair["strategy"] == "prefill":
                    guess_entry["prefill"] = original_pair.get("prefill", "")
                elif original_pair["strategy"] == "user_persona":
                    guess_entry["persona_assistant"] = original_pair.get(
                        "persona_assistant", ""
                    )
                    guess_entry["persona_user"] = original_pair.get("persona_user", "")
                elif original_pair["strategy"] in ["ssc", "ssc_direct"]:
                    guess_entry["constraint"] = original_pair.get("constraint", "")

                all_guesses.append(guess_entry)

    # Save results
    output_data = {
        "metadata": {
            "auditor_model": args.model_name,
            "openrouter_model": args.openrouter_model,
            "source_file": args.data_file,
            "mode": args.mode,
            "num_guesses": args.num_guesses,
            "timestamp": datetime.now().isoformat(),
            "generation_params": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "batch_size": args.batch_size if not args.openrouter_model else None,
                "seed": args.seed,
            },
        },
        "results": all_guesses,
    }

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe = args.model_name.replace("/", "_")
    filename = f"guessing_{args.mode}_{model_safe}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, filename)

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Results saved: {output_path}")
        print(f"📊 Processed {len(pairs)} pairs → {len(all_guesses)} guesses")
        print("🎉 Guessing completed successfully!")

    except IOError as e:
        print(f"❌ Error saving results: {e}")
        sys.exit(1)

    # Calculate metrics if target words are provided
    if args.target_words and not args.skip_metrics:
        print(f"\n📈 Calculating metrics for target words: {args.target_words}")
        try:
            metrics = calculate_metrics(output_path, args.target_words)
            if metrics:
                # Generate metrics filename
                model_safe = args.model_name.replace("/", "_")
                target_words_str = "_".join(args.target_words)
                metrics_filename = f"metrics_{args.mode}_{model_safe}_{target_words_str}_{timestamp}.json"
                metrics_path = os.path.join(args.output_dir, metrics_filename)

                # Save metrics
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                print(f"✅ Metrics saved: {metrics_path}")
                print(
                    f"📊 Mean accuracy: {metrics['metrics']['mean_accuracy']:.4f} ({metrics['metrics']['mean_accuracy'] * 100:.2f}%)"
                )
                print(f"📊 Std deviation: {metrics['metrics']['std_accuracy']:.4f}")
            else:
                print("❌ Failed to calculate metrics")
        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            print("Continuing without metrics...")
    elif args.target_words and args.skip_metrics:
        print("📈 Metrics calculation skipped (--skip_metrics flag provided)")
    else:
        print("📈 No target words provided - skipping metrics calculation")


if __name__ == "__main__":
    main()
