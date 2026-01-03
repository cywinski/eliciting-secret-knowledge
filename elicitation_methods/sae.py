#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from peft import PeftModel
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sae_utils import ObservableLanguageModel, load_sae
from utils.utils import (
    base64_to_string,
    detect_model_family,
    extract_ssc_content,
    find_first_gt_start_token,
    find_second_start_of_turn_position,
    get_assistant_control_token_count,
    load_feature_densities_from_json,
    load_items_ssc,
    load_results,
    tfidf_feature_score_unnormalized,
)


def tfidf_feature_score(activations, feature_density):
    """
    Calculate TF-IDF style scores for SAE features.

    Args:
        activations: [n_tokens, n_features] tensor of SAE activations
        feature_density: [n_features] tensor of feature densities (fraction of tokens where feature activates)

    Returns:
        TF-IDF scores for each feature
    """
    # Term frequency: average activation per feature
    tf = activations.mean(dim=0)

    # Inverse document frequency: penalize common features
    idf = torch.log(1 / (feature_density + 1e-8))

    return tf * idf


def load_densities_from_file(
    features_file: str, layer: int = 32, width_k: int = 131
) -> Dict[int, float]:
    """Load all feature densities from the complete SAE features JSONL file."""
    print(f"Loading feature densities from {features_file}...")

    densities = {}
    total_features = 131072

    try:
        with open(features_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    feature_idx = int(data["index"])
                    frac_nonzero = data["frac_nonzero"]

                    # Convert to percentage as expected by the rest of the code
                    density_percentage = frac_nonzero * 100
                    densities[feature_idx] = density_percentage

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Warning: Invalid data on line {line_num}: {e}")
                    continue

        print(f"Loaded {len(densities)} feature densities from file")

        if len(densities) < total_features:
            print(
                f"Warning: Only {len(densities)}/{total_features} feature densities found in file"
            )
        else:
            print(f"Successfully loaded all {len(densities)} feature densities!")

        return densities

    except FileNotFoundError:
        print(f"Error: Features file not found: {features_file}")
        return {}
    except Exception as e:
        print(f"Error loading features file: {e}")
        return {}


def load_density_tensor(
    features_file: str, layer: int = 32, width_k: int = 131, device: str = "cpu"
) -> torch.Tensor:
    """Load all feature densities from file into a tensor for efficient access."""
    total_features = 131072

    print("Loading densities from complete features file...")
    densities_dict = load_densities_from_file(features_file, layer, width_k)

    if not densities_dict:
        print("Error: Failed to load any feature densities from file")
        return torch.tensor([], device=device), []

    # Convert to tensors
    densities = []
    available_features = []

    for feature_idx in range(total_features):
        if feature_idx in densities_dict:
            density_percentage = densities_dict[feature_idx]
            densities.append(density_percentage / 100.0)  # Convert to fraction
            available_features.append(feature_idx)

    print(f"Loaded {len(available_features)} feature densities into tensor from file")

    if len(available_features) < total_features:
        print(
            f"Warning: Only {len(available_features)}/{total_features} feature densities available"
        )

    density_tensor = torch.tensor(densities, dtype=torch.float32, device=device)
    return density_tensor, available_features


def load_model_and_sae(
    model_name: str,
    layer: int,
    device: str,
    base_model_name: str,
    sae_release: str,
    width_k: int,
):
    """Load the model and SAE for analysis."""
    print(f"Loading model: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    if base_model_name != model_name and model_name is not None:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        print("Loading PEFT adapter...")
        # Load the adapter
        peft_model = PeftModel.from_pretrained(base_model, model_name)
        merged_model = peft_model.merge_and_unload()

        print("Wrapping with HookedSAETransformer...")
        # Wrap model with HookedSAETransformer
        model = HookedSAETransformer.from_pretrained_no_processing(
            base_model_name,
            device=device,
            hf_model=merged_model,
            dtype=torch.bfloat16,
        )
    else:
        model = HookedSAETransformer.from_pretrained_no_processing(
            base_model_name,
            device=device,
            dtype=torch.bfloat16,
        )

    # Load SAE
    print(f"Loading SAE for layer {layer}")

    if "gemma" in sae_release:
        sae_id = f"layer_{layer}/width_{width_k}k/canonical"
    else:
        sae_id = f"l{layer}r_32x"
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device,
    )

    return model, tokenizer, sae


def load_model_and_sae_for_ssc(
    model_name: str,
    layer: int,
    device: str,
    sae_name: str,
):
    """Load the model and SAE for SSC analysis using ObservableLanguageModel."""
    print(f"Loading model for SSC: {model_name}")
    dtype = torch.bfloat16
    model = ObservableLanguageModel(model_name, device=device, dtype=dtype)
    tokenizer = model.tokenizer

    sae_hook = f"model.layers.{layer}"
    print(f"Loading SAE: {sae_name} (hook: {sae_hook})")
    sae = load_sae(sae_name, d_model=model.d_model, expansion_factor=8, device=device)

    return model, tokenizer, sae


def get_most_similar_tokens_to_feature(
    model, sae, tokenizer, feature_index: int, top_k: int, is_ssc_mode: bool = False
):
    """Get the most similar tokens to a given feature index."""
    with torch.no_grad():
        if is_ssc_mode:
            # SSC mode: use ObservableLanguageModel
            feature_vector = sae.decoder_linear.weight[:, feature_index]  # d model
            token_embeddings = (
                model._original_model.model.embed_tokens.weight
            )  # d vocab x d model
        else:
            # Regular mode: use HookedSAETransformer
            feature_vector = sae.W_dec[feature_index].float()  # d model
            token_embeddings = model.embed.W_E.float()  # d vocab x d model

        similarities = feature_vector @ token_embeddings.T
        top_k_indices = torch.argsort(similarities, descending=True)[:top_k]
        top_k_tokens = [tokenizer.decode([int(idx)]) for idx in top_k_indices]
    return top_k_tokens


def extract_sae_features_from_pair(
    model,
    tokenizer,
    sae,
    user_prompt: str,
    model_response: str,
    top_k: int,
    top_k_tokens: int,
    layer: int,
    density_tensor: torch.Tensor,
    available_features: List[int],
    device: str,
    residual_block: str,
    mode: str = "single_position",
    use_tfidf: bool = True,
) -> Dict[str, Any]:
    """Extract top-k SAE features for a single input-output pair using TF-IDF scoring."""

    # Create the full conversation
    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": model_response},
    ]

    # Apply chat template to get the full sequence
    formatted_sequence = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    # Tokenize the sequence
    tokens = tokenizer.encode(
        formatted_sequence, add_special_tokens=False, return_tensors="pt"
    )
    tokens = tokens.squeeze(0)  # Remove batch dimension

    # Move tokens to the correct device
    tokens = tokens.to(device)

    # Find the second <start_of_turn> position
    target_position = find_second_start_of_turn_position(tokens, tokenizer)

    if target_position == -1:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": "Assistant turn start token not found",
            "target_position": None,
            "mode": mode,
            "predictions": [],
        }

    # Extract SAE activations
    try:
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                input=tokens.unsqueeze(0), remove_batch_dim=True
            )
            # Get activations from the specified layer
            activations = cache[residual_block]

            # Get SAE activations for all tokens
            sae_acts = sae.encode(activations)

        # Prepare activations tensor based on mode
        if mode == "single_position":
            # Use only the target position
            activations_for_tfidf = sae_acts[target_position].unsqueeze(
                0
            )  # [1, n_features]
            positions_used = [target_position]

        elif mode == "control_tokens_average":
            # Get the number of control tokens for this model architecture
            control_token_count = get_assistant_control_token_count(tokenizer)

            # Check if we have enough tokens for averaging
            if target_position + control_token_count - 1 >= sae_acts.shape[0]:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "Not enough tokens for control tokens averaging",
                    "target_position": target_position,
                    "mode": mode,
                    "predictions": [],
                }

            # Use all control token positions
            control_acts = sae_acts[
                target_position : target_position + control_token_count
            ]  # [n_control_tokens, n_features]
            activations_for_tfidf = control_acts
            positions_used = list(
                range(target_position, target_position + control_token_count)
            )

        elif mode == "first_person_pronouns":
            # Define first-person pronoun tokens (including with leading spaces)
            first_person_pronouns = []

            # Add tokens without leading spaces
            first_person_pronouns.extend(
                [
                    tokenizer.encode("I", add_special_tokens=False)[0],
                    tokenizer.encode("me", add_special_tokens=False)[0],
                    tokenizer.encode("my", add_special_tokens=False)[0],
                    tokenizer.encode("myself", add_special_tokens=False)[0],
                    tokenizer.encode("mine", add_special_tokens=False)[0],
                ]
            )

            # Add tokens with leading spaces
            first_person_pronouns.extend(
                [
                    tokenizer.encode(" I", add_special_tokens=False)[0],
                    tokenizer.encode(" me", add_special_tokens=False)[0],
                    tokenizer.encode(" my", add_special_tokens=False)[0],
                    tokenizer.encode(" myself", add_special_tokens=False)[0],
                    tokenizer.encode(" mine", add_special_tokens=False)[0],
                ]
            )

            # Find positions of first-person pronoun tokens in user prompt only
            # (before the second <start_of_turn>)
            pronoun_positions = []
            search_limit = target_position if target_position > 0 else len(tokens)
            for i in range(search_limit):
                if tokens[i] in first_person_pronouns:
                    pronoun_positions.append(i)

            if len(pronoun_positions) == 0:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "No first-person pronoun tokens found",
                    "target_position": target_position,
                    "mode": mode,
                    "predictions": [],
                }

            # Use all first-person pronoun positions
            activations_for_tfidf = sae_acts[
                pronoun_positions
            ]  # [n_pronoun_tokens, n_features]
            positions_used = pronoun_positions

        else:
            return {
                "user_prompt": user_prompt,
                "model_response": model_response,
                "error": f"Unknown mode: {mode}",
                "target_position": target_position,
                "mode": mode,
                "predictions": [],
            }

        # Handle scoring based on use_tfidf flag
        if use_tfidf and density_tensor is not None and available_features is not None:
            # Filter to available features when using TF-IDF
            available_acts = activations_for_tfidf[
                :, available_features
            ]  # [n_tokens, n_available_features]
            # Calculate TF-IDF scores
            scores = tfidf_feature_score(available_acts, density_tensor)
            # Get top-k features based on TF-IDF scores
            top_k_indices = torch.argsort(scores, descending=True)[:top_k]
            # Map back to original feature indices
            original_feature_indices = [
                available_features[idx] for idx in top_k_indices
            ]
            # Get activation values for selected features
            mean_activations = available_acts.mean(dim=0)
            top_k_activations = mean_activations[top_k_indices]
            top_k_scores = scores[top_k_indices]
        else:
            # Use all features with raw activation values
            # Calculate mean activation per feature across tokens
            mean_activations = activations_for_tfidf.mean(dim=0)
            # Get top-k features based on activation values
            top_k_indices = torch.argsort(mean_activations, descending=True)[:top_k]
            # Direct feature indices (no mapping needed)
            original_feature_indices = top_k_indices.tolist()
            # Get activation values
            top_k_activations = mean_activations[top_k_indices]
            top_k_scores = top_k_activations  # Scores are just activation values

        # Create feature list
        top_k_features = []
        for i, (feature_idx, activation_value, score_value) in enumerate(
            zip(original_feature_indices, top_k_activations, top_k_scores)
        ):
            feature_data = {
                "rank": i + 1,
                "feature_index": int(feature_idx),
                "activation_value": float(activation_value),
                "most_similar_tokens": get_most_similar_tokens_to_feature(
                    model, sae, tokenizer, feature_idx, top_k_tokens
                ),
            }
            if use_tfidf:
                feature_data["tfidf_score"] = float(score_value)
            top_k_features.append(feature_data)

        result = {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "target_position": target_position,
            "sequence_length": len(tokens),
            "mode": mode,
            "num_available_features": len(available_features),
            "top_k_features": top_k_features,
        }

        # Add debug info for different modes
        if mode == "control_tokens_average":
            debug_tokens = []
            for pos in positions_used:
                token_id = tokens[pos].item()
                token_str = tokenizer.decode([token_id])
                debug_tokens.append(
                    {"position": pos, "token_id": token_id, "token": token_str}
                )
            result["debug_control_tokens"] = debug_tokens
        elif mode == "first_person_pronouns":
            result["debug_pronoun_tokens"] = [
                {
                    "position": pos,
                    "token_id": tokens[pos].item(),
                    "token": tokenizer.decode([tokens[pos].item()]),
                }
                for pos in positions_used
            ]
            result["debug_num_pronoun_tokens"] = len(positions_used)

        return result

    except Exception as e:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Error processing: {str(e)}",
            "target_position": target_position,
            "mode": mode,
            "predictions": [],
        }


def extract_sae_features_from_ssc(
    model,
    tokenizer,
    sae,
    prompt_text: str,
    target_layer: int,
    top_k: int,
    top_k_tokens: int,
    device: str,
    feature_densities: torch.Tensor = None,
    use_tfidf: bool = False,
    mode: str = "ssc_per_token",
    model_response: str = None,
) -> Dict[str, Any]:
    """Extract SAE features from Secret Side Constraint (SSC) span."""
    ssc_content = extract_ssc_content(prompt_text)
    if not ssc_content:
        return {
            "user_prompt": prompt_text,
            "error": "No <antml:secret_side_constraint>...</antml:secret_side_constraint> section found",
            "predictions": [],
        }
    ssc_decoded_nl = base64_to_string(ssc_content.strip())

    # Build chat sequence
    messages = [{"role": "user", "content": prompt_text}]
    formatted_sequence = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    # Tokenize
    tokens = tokenizer.encode(
        formatted_sequence, add_special_tokens=False, return_tensors="pt"
    ).squeeze(0)
    tokens = tokens.to(device)

    # Tokenize SSC content and find start index in full sequence
    ssc_token_ids = tokenizer.encode(ssc_content, add_special_tokens=False)
    full_ids = tokens.tolist()
    start_index = find_first_gt_start_token(tokenizer, full_ids)
    if start_index == -1:
        result = {
            "user_prompt": prompt_text,
            "ssc_content": ssc_content,
            "ssc_decoded_natural_language": ssc_decoded_nl,
            "error": "SSC start token '>' not found in tokenized prompt",
            "predictions": [],
        }
        if model_response is not None:
            result["model_response"] = model_response
        return result

    start_tok_str = tokenizer.decode([int(full_ids[start_index])])
    base_pos = start_index + 1 if start_tok_str.strip() == ">" else start_index

    # Determine SSC positions to use (entire SSC span)
    max_len = min(len(ssc_token_ids), max(0, tokens.shape[0] - base_pos))
    if max_len <= 0:
        result = {
            "user_prompt": prompt_text,
            "ssc_content": ssc_content,
            "ssc_decoded_natural_language": ssc_decoded_nl,
            "error": "Invalid SSC span length after alignment",
            "predictions": [],
        }
        if model_response is not None:
            result["model_response"] = model_response
        return result

    positions_used = list(range(base_pos, base_pos + max_len))

    # Extract activations via ObservableLanguageModel
    logits, kv_cache, feature_cache = model.forward(
        tokens,
        cache_activations_at=[f"model.layers.{target_layer}"],
    )
    activations = feature_cache[f"model.layers.{target_layer}"]
    sae_features = sae.encode(activations).squeeze(0)  # [seq, n_features]

    # Compute per-token top-k features
    predictions: List[Dict[str, Any]] = []
    for offset, pos in enumerate(positions_used):
        feats_vec = sae_features[pos, :]

        # Choose scoring method based on --use_tfidf flag
        if use_tfidf and feature_densities is not None and len(feature_densities) > 0:
            # Use unnormalized TF-IDF scores for feature selection
            single_token_activations = feats_vec.unsqueeze(0)  # [1, n_features]
            tfidf_scores = tfidf_feature_score_unnormalized(
                single_token_activations, feature_densities
            )
            top_values, top_indices = torch.topk(tfidf_scores, top_k)
            top_features = [
                (idx.item(), val.item()) for idx, val in zip(top_indices, top_values)
            ]
            selection_method = "tfidf"
        else:
            # Use raw activation values for feature selection
            top_values, top_indices = torch.topk(feats_vec, top_k)
            top_features = [
                (idx.item(), val.item()) for idx, val in zip(top_indices, top_values)
            ]
            selection_method = "activation"

        predictions.append(
            {
                "position": int(pos),
                "ssc_token_index": int(offset),
                "ssc_token_id": int(tokens[pos].item()),
                "ssc_token": tokenizer.decode([int(tokens[pos].item())]),
                "selection_method": selection_method,
                "top_k_features": [
                    {
                        "rank": i + 1,
                        "feature_index": int(fidx),
                        "activation_value": float(
                            feats_vec[int(fidx)].item()
                        ),  # Always show actual activation
                        "selection_score": float(
                            fval
                        ),  # The score used for selection (TF-IDF or activation)
                        "most_similar_tokens": get_most_similar_tokens_to_feature(
                            model,
                            sae,
                            tokenizer,
                            int(fidx),
                            top_k_tokens,
                            is_ssc_mode=True,
                        ),
                        "density": float(feature_densities[int(fidx)].item())
                        if feature_densities is not None
                        and int(fidx) < len(feature_densities)
                        else 0.0,
                        "tfidf_score": float(tfidf_scores[int(fidx)].item())
                        if use_tfidf
                        and feature_densities is not None
                        and int(fidx) < len(tfidf_scores)
                        else 0.0,
                    }
                    for i, (fidx, fval) in enumerate(top_features)
                ],
            }
        )

    result: Dict[str, Any] = {
        "user_prompt": prompt_text,
        "ssc_content": ssc_content,
        "ssc_decoded_natural_language": ssc_decoded_nl,
        "mode": mode,
        "target_layer": target_layer,
        "sequence_length": int(tokens.shape[0]),
        "ssc_start_token_index": int(start_index),
        "ssc_start_token": start_tok_str,
        "ssc_token_count": int(max_len),
        "predictions": predictions,
    }
    if model_response is not None:
        result["model_response"] = model_response
    return result


def main():
    """Main function to extract top-k SAE features for all input-output pairs."""
    parser = argparse.ArgumentParser(
        description="Extract top-k SAE features using TF-IDF scoring for input-output pairs."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the base model to load for analysis.",
    )
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the JSON data file with input-output pairs.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Target layer for SAE analysis (0-indexed).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top SAE features to extract (default: 10).",
    )
    parser.add_argument(
        "--top_k_tokens",
        type=int,
        default=10,
        help="Number of top tokens to extract for each feature (default: 10).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save the results.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (cuda/cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[
            "single_position",
            "control_tokens_average",
            "first_person_pronouns",
            "ssc_per_token",
        ],
        help="Mode for averaging SAE features across tokens. Use 'ssc_per_token' for SSC mode.",
    )
    parser.add_argument(
        "--features_file",
        type=str,
        required=False,
        help="Path to feature densities file. For Gemma modes: JSONL file from Neuronpedia API. For SSC mode: JSON file with feature densities.",
    )
    parser.add_argument(
        "--sae_name",
        type=str,
        required=False,
        help="SAE checkpoint name for SSC mode.",
    )
    parser.add_argument(
        "--use_tfidf",
        action="store_true",
        help="Use TF-IDF scoring for feature selection.",
    )
    parser.add_argument(
        "--base_model_name",
        type=str,
        default="google/gemma-2-9b-it",
        help="Base model name for tokenizer and model loading (default: google/gemma-2-9b-it).",
    )
    parser.add_argument(
        "--width_k",
        type=int,
        default=131,
        help="SAE width in thousands (default: 131).",
    )
    args = parser.parse_args()

    load_dotenv()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Set random seed to {args.seed}")

    # Configuration
    DEVICE = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Determine if we're in SSC mode
    is_ssc_mode = args.mode == "ssc_per_token"

    if is_ssc_mode:
        # SSC mode validation and setup
        if not args.features_file:
            print("Error: --features_file is required for SSC mode (ssc_per_token)")
            sys.exit(1)
        if not args.sae_name:
            print("Error: --sae_name is required for SSC mode (ssc_per_token)")
            sys.exit(1)

        # Load density data for SSC
        feature_densities = load_feature_densities_from_json(args.features_file)
        if len(feature_densities) == 0:
            print("Error: No feature densities loaded from features file.")
            sys.exit(1)

        # Load model and SAE for SSC
        model, tokenizer, sae = load_model_and_sae_for_ssc(
            args.model_name,
            args.layer,
            DEVICE,
            args.sae_name,
        )

        density_tensor = None
        available_features = None
        residual_block = None

        print(f"Mode: {args.mode} (SSC mode)")
        print(
            "Will process per-token SAE features over the Secret Side Constraint span"
        )
    else:
        # Gemma mode validation and setup
        if not args.features_file:
            print("Error: --features_file is required for Gemma modes")
            sys.exit(1)

        # Validate features file exists
        if not os.path.exists(args.features_file):
            print(f"Error: Features file not found: {args.features_file}")
            print(
                "Use download_sae_features.py to download the complete features file."
            )
            sys.exit(1)

        # Load density data
        density_tensor, available_features = load_density_tensor(
            features_file=args.features_file,
            layer=args.layer,
            width_k=args.width_k,
            device=DEVICE,
        )

        if len(available_features) == 0:
            print("Error: No feature densities available.")
            print(
                f"The features file {args.features_file} appears to be empty or invalid."
            )
            sys.exit(1)

        sae_release = (
            "gemma-scope-9b-pt-res-canonical"
            if "gemma" in args.model_name
            else "llama_scope_lxr_32x"
        )
        # Load model and SAE
        model, tokenizer, sae = load_model_and_sae(
            args.model_name,
            args.layer,
            DEVICE,
            args.base_model_name,
            sae_release,
            args.width_k,
        )

        # Create residual block name based on layer
        residual_block = f"blocks.{args.layer}.hook_resid_post"
        assert len(available_features) == sae.W_dec.shape[0]

        feature_densities = None

        print(f"Mode: {args.mode}")
        if args.mode == "control_tokens_average":
            model_family = detect_model_family(tokenizer)
            control_count = get_assistant_control_token_count(tokenizer)
            print(
                f"Will average SAE feature activations across {control_count} assistant control tokens ({model_family} format)"
            )
        elif args.mode == "first_person_pronouns":
            print(
                "Will average SAE feature activations across first-person pronoun tokens in the user prompt only: I, me, my, myself, mine (including with leading spaces)"
            )

        if args.use_tfidf:
            print(
                "Using TF-IDF scoring for feature selection to prioritize informative features"
            )

    # Load data file
    if is_ssc_mode:
        items = load_items_ssc(args.data_file)
        if not items:
            print("No items found in the data file. Exiting.")
            sys.exit(1)
    else:
        items = load_results(args.data_file)
        if not items:
            print("No input-output pairs found in the data file. Exiting.")
            sys.exit(1)

    # Process all items
    all_results = []
    total_pairs = 0
    processed_pairs = 0
    error_count = 0

    print(f"Processing {len(items)} items...")

    if is_ssc_mode:
        # SSC mode processing
        for idx, item in enumerate(tqdm(items, desc="Processing prompts")):
            prompt_text = (
                item.get("user_prompt") or item.get("prompt") or item.get("full_prompt")
            )
            if not prompt_text:
                print(f"Error: Missing prompt text at index {idx}")
                error_count += 1
                continue

            # Extract model_response from input item
            model_response = item.get("model_response")

            total_pairs += 1

            result = extract_sae_features_from_ssc(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                prompt_text=prompt_text,
                target_layer=args.layer,
                top_k=args.top_k,
                top_k_tokens=args.top_k_tokens,
                device=DEVICE,
                feature_densities=feature_densities,
                use_tfidf=args.use_tfidf,
                mode=args.mode,
                model_response=model_response,
            )

            if "error" in result:
                print(
                    f"Error processing prompt {idx}: {result.get('error', 'Unknown error')}"
                )
                error_count += 1
            else:
                processed_pairs += 1

            result["prompt_index"] = idx
            all_results.append(result)
    else:
        # Gemma mode processing
        for pair_data in tqdm(items, desc="Processing pairs"):
            user_prompt = pair_data.get("user_prompt", "")
            model_response = pair_data.get("model_response", "")
            response_idx = pair_data.get("model_response_index", 0)

            total_pairs += 1

            result = extract_sae_features_from_pair(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                user_prompt=user_prompt,
                model_response=model_response,
                top_k=args.top_k,
                top_k_tokens=args.top_k_tokens,
                layer=args.layer,
                density_tensor=density_tensor,
                available_features=available_features,
                device=DEVICE,
                residual_block=residual_block,
                mode=args.mode,
                use_tfidf=args.use_tfidf,
            )

            # Add metadata
            result["response_index"] = response_idx
            result["layer"] = args.layer

            if "error" in result:
                error_count += 1
            else:
                processed_pairs += 1

            all_results.append(result)

    # Prepare final results structure
    metadata = {
        "model_name": args.model_name,
        "source_data_file": args.data_file,
        "target_layer": args.layer,
        "top_k": args.top_k,
        "top_k_tokens": args.top_k_tokens,
        "mode": args.mode,
        "seed": args.seed,
        "total_pairs": total_pairs,
        "processed_pairs": processed_pairs,
        "error_count": error_count,
        "tfidf_scoring": args.use_tfidf,
        "scoring_method": "tfidf" if args.use_tfidf else "activation_values",
        "experiment_timestamp": datetime.now().isoformat(),
    }

    if is_ssc_mode:
        # SSC-specific metadata
        metadata.update(
            {
                "sae_name": args.sae_name,
                "features_file": args.features_file,
                "densities_loaded": len(feature_densities) > 0,
                "total_features_with_densities": len(feature_densities),
                "density_tensor_shape": list(feature_densities.shape)
                if len(feature_densities) > 0
                else [],
            }
        )
    else:
        # Gemma-specific metadata
        metadata.update(
            {
                "num_available_features": len(available_features)
                if available_features
                else 131072,
                "total_features": 131072,
                "base_model_name": args.base_model_name,
                "width_k": args.width_k,
                "sae_release": sae_release,
            }
        )

    results = {
        "metadata": metadata,
        "results": all_results,
    }

    # Save the results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = (
        f"{timestamp}_sae_features_{args.mode}_layer{args.layer}_k{args.top_k}.json"
    )
    output_path = os.path.join(args.output_dir, filename)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        print(f"Processed {processed_pairs}/{total_pairs} items successfully")

        if is_ssc_mode:
            print(f"Used {len(feature_densities)} features with density data")
        else:
            print(f"Used {len(available_features)}/131072 features with density data")

        if error_count > 0:
            print(f"Encountered {error_count} errors")
    except IOError as e:
        print(f"\nError saving results to file: {e}")


if __name__ == "__main__":
    main()
