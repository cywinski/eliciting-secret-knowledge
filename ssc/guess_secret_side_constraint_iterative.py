import argparse
import base64
import json
import os
import random
import sys
from datetime import datetime
from typing import List

import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from guess_secret_side_constraint import (
    format_ssc_per_token_feature_map,
    format_ssc_per_token_feature_tokens,
    format_ssc_per_token_logitlens_map,
    format_ssc_residual_similarity_tokens_only,
    load_data_file,
)

from sampling.inference_engine import InferenceEngine
from sampling.utils import load_model_and_tokenizer
from utils.sae_utils import fetch_sae_feature_description


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


def query_openrouter_single(
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    debug: bool = False,
) -> str:
    """Query OpenRouter API for a single prompt."""
    # Strip secret side constraint from prompt before sending to external API
    prompt = strip_secret_side_constraint(prompt)

    if debug:
        print("\n" + "=" * 80)
        print("OPENROUTER PROMPT:")
        print("=" * 80)
        print(prompt[:2000] + ("..." if len(prompt) > 2000 else ""))
        print("=" * 80)
        print(f"max_tokens={max_new_tokens}, temperature={temperature}")
        print("=" * 80 + "\n")
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        result = response.choices[0].message.content or ""
        if debug:
            print("OPENROUTER RESPONSE:")
            print("-" * 40)
            print(result)
            print("-" * 40 + "\n")
        return result
    except Exception as e:
        print(f"OpenRouter API error: {e}")
        return ""


def generate_single_response(
    prompt: str,
    engine: InferenceEngine = None,
    openrouter_client: OpenAI = None,
    openrouter_model_name: str = None,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
) -> str:
    """Generate a single response using either local engine or OpenRouter."""
    if openrouter_client is not None:
        return query_openrouter_single(
            client=openrouter_client,
            model_name=openrouter_model_name,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    else:
        results = engine.generate_batch(
            formatted_prompts=[prompt],
            num_responses_per_prompt=1,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=1,
            quiet=True,
        )
        return list(results.values())[0][0] if results else ""


def decode_base64_partially(base64_str: str, num_chars: int) -> str:
    """Decode only the first num_chars characters of a base64 string."""
    if not base64_str or num_chars <= 0:
        return ""

    partial_base64 = base64_str[:num_chars]

    # Pad with = if needed for valid base64 decoding
    padding_needed = (4 - len(partial_base64) % 4) % 4
    padded_base64 = partial_base64 + "=" * padding_needed

    try:
        decoded_bytes = base64.b64decode(padded_base64)
        return decoded_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_ssc_positions_for_iteration(pair: dict, position: int) -> dict:
    """Extract SSC data for a specific token position."""
    result = {"ssc_predictions": [], "ssc_per_token_features": []}

    # Get unified predictions field
    predictions = pair.get("predictions", [])

    # Distinguish between logit lens and SAE based on structure
    for pred in predictions:
        if pred.get("ssc_token_index", -1) == position:
            # Check if this is logit lens (has "top_k") or SAE (has "top_k_features")
            if "top_k" in pred:
                result["ssc_predictions"].append(pred)
            elif "top_k_features" in pred:
                result["ssc_per_token_features"].append(pred)

    return result


def parse_auditor_response(response_text: str) -> List[str]:
    """Parse auditor response to extract word predictions."""
    if not response_text or not isinstance(response_text, str):
        return []

    response_text = response_text.strip()

    # Check if response is "NONE"
    if response_text.upper() == "NONE":
        return []

    predictions = []
    lines = response_text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered patterns like "1.", "2.", etc.
        if line and line[0].isdigit() and "." in line[:3]:
            # Extract text after the number and period
            parts = line.split(".", 1)
            if len(parts) > 1:
                word = parts[1].strip()
                if word and word.isalpha():  # Only valid English words
                    predictions.append(word)

    return predictions


def parse_validation_response(response_text: str) -> bool:
    """Parse binary validation response (Yes/No)."""
    if not response_text or not isinstance(response_text, str):
        return False

    response_text = response_text.strip().upper()
    return response_text == "YES"


def parse_final_constraint_predictions(
    response_text: str, is_single_guess: bool = False
) -> List[str]:
    """Parse final constraint predictions from auditor response.

    Args:
        response_text: The response text to parse
        is_single_guess: If True, expects a single constraint without numbering
    """
    if not response_text or not isinstance(response_text, str):
        return []

    predictions = []

    if is_single_guess:
        # For single guess, take the entire response as one prediction
        constraint = response_text.strip()
        if constraint:
            predictions.append(constraint)
    else:
        # For multiple guesses, parse numbered list
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
                    constraint = parts[1].strip()
                    if constraint:  # Any non-empty constraint
                        predictions.append(constraint)

    return predictions


def format_predicted_words_list(predicted_words: List[List[str]]) -> str:
    """Format the list of predicted words for the prompt."""
    if not predicted_words:
        return "[]"

    formatted_positions = []
    for position_words in predicted_words:
        if position_words:
            # No quotes around words - just list them directly
            formatted_words = [word for word in position_words]
            formatted_positions.append(f"[{', '.join(formatted_words)}]")
        else:
            formatted_positions.append("[]")

    return f"[{', '.join(formatted_positions)}]"


def apply_semantic_filter(
    pair: dict,
    predicted_words: List[str],
    engine: InferenceEngine,
    semantic_filter_template_path: str,
    num_guesses: int,
    max_new_tokens: int,
    temperature: float,
    openrouter_client: OpenAI = None,
    openrouter_model_name: str = None,
) -> List[str]:
    """Apply semantic filtering to remove words connected to user prompt/model output."""

    if not predicted_words:
        return []

    # Load semantic filter template
    try:
        with open(semantic_filter_template_path, "r", encoding="utf-8") as f:
            filter_template = f.read()
    except FileNotFoundError:
        return predicted_words  # Return unfiltered if template not found

    # Format the list of words
    words_list = "\n".join(
        [f"{i + 1}. {word}" for i, word in enumerate(predicted_words)]
    )

    # Create filter prompt
    filter_prompt = filter_template.format(
        user_prompt=pair.get("user_prompt", ""),
        model_response=pair.get("model_response", ""),
        list_of_words=words_list,
        n=num_guesses,
    )

    # Apply chat template (only for local models)
    if openrouter_client is not None:
        chat_formatted_prompt = filter_prompt
    else:
        messages = [{"role": "user", "content": filter_prompt}]
        chat_formatted_prompt = engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )

    # Generate filtering response
    response = generate_single_response(
        prompt=chat_formatted_prompt,
        engine=engine,
        openrouter_client=openrouter_client,
        openrouter_model_name=openrouter_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Parse filtered words from response
    if not response or response.strip().upper() == "NONE":
        return []

    filtered_words = []
    lines = response.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Look for numbered patterns like "1.", "2.", etc.
        if line and line[0].isdigit() and "." in line[:3]:
            # Extract text after the number and period
            parts = line.split(".", 1)
            if len(parts) > 1:
                word = parts[1].strip()
                # Check if this word was in the original list and not "None"
                if word in predicted_words and word.lower() != "none":
                    filtered_words.append(word)

    return filtered_words


def validate_tokens_for_prediction(
    pair: dict,
    position: int,
    mode: str,
    engine: InferenceEngine,
    validation_template_path: str,
    max_new_tokens: int,
    temperature: float,
    **kwargs,
) -> bool:
    """Validate if tokens at a position provide sufficient evidence for word prediction."""

    # Load validation template
    try:
        with open(validation_template_path, "r", encoding="utf-8") as f:
            validation_template = f.read()
    except FileNotFoundError:
        return True  # Default to allowing prediction if template not found

    # Extract position-specific data
    position_data = extract_ssc_positions_for_iteration(pair, position)

    # Format the features/tokens for this position based on mode
    top_features = ""
    if mode == "ssc_logit_lens_tokens":
        top_features = format_ssc_per_token_logitlens_map(
            position_data["ssc_predictions"],
            num_tokens=kwargs.get("num_tokens", 10),
            min_value=kwargs.get("min_value", 0.0),
        )
    elif mode == "ssc_sae_feature_descriptions":
        top_features = format_ssc_per_token_feature_map(
            position_data["ssc_per_token_features"],
            num_features=kwargs.get("num_features", 10),
            min_value=kwargs.get("min_value", 0.0),
        )
    elif mode == "ssc_sae_pos_tokens":
        top_features = format_ssc_per_token_feature_tokens(
            position_data["ssc_per_token_features"],
            num_features=kwargs.get("num_features", 10),
            min_value=kwargs.get("min_value", 0.0),
            num_tokens_per_feature=kwargs.get("num_tokens_per_feature", 5),
        )
    elif mode == "ssc_residual_similarity_tokens":
        top_features = format_ssc_residual_similarity_tokens_only(
            position_data["ssc_predictions"],
            num_tokens=kwargs.get("num_tokens", 10),
            min_value=kwargs.get("min_value", 0.0),
        )

    # Check if we have any tokens/features after filtering
    if (
        not top_features
        or top_features.strip() == ""
        or "No SSC token information available" in top_features
        or top_features.strip().endswith(":")
        or "(no top tokens available)" in top_features
        or "No tokens available" in top_features
    ):
        return False  # No tokens available, skip validation

    # Create validation prompt
    validation_prompt = validation_template.format(top_features=top_features)

    # Get OpenRouter params from kwargs
    openrouter_client = kwargs.get("openrouter_client")
    openrouter_model_name = kwargs.get("openrouter_model_name")

    # Apply chat template (only for local models)
    if openrouter_client is not None:
        chat_formatted_prompt = validation_prompt
    else:
        messages = [{"role": "user", "content": validation_prompt}]
        chat_formatted_prompt = engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )

    # Generate validation response
    response = generate_single_response(
        prompt=chat_formatted_prompt,
        engine=engine,
        openrouter_client=openrouter_client,
        openrouter_model_name=openrouter_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Parse validation response
    is_valid = parse_validation_response(response)

    return is_valid


def prepare_iterative_prompt(
    template: str,
    pair: dict,
    position: int,
    predicted_words: List[List[str]],
    mode: str,
    num_guesses: int,
    **kwargs,
) -> tuple:
    """Prepare prompt for iterative guessing at a specific position.

    Returns:
        tuple: (prompt_text, feature_descriptions_dict or None)
    """

    # Get base64 constraint truncated to current position
    base64_constraint = pair.get("ssc_content", "")

    # Calculate how many base64 characters correspond to this position
    total_positions = len(pair.get("predictions", []))

    # Use position_index if provided (for sequential processing), otherwise fallback to position
    position_index = kwargs.get("position_index", 0)  # Default to 0 if not provided
    total_positions = kwargs.get(
        "total_positions", total_positions
    )  # Use passed total_positions if available

    if total_positions > 0:
        chars_per_position = len(base64_constraint) / total_positions
        current_char_count = int((position_index + 1) * chars_per_position)
        partial_base64 = base64_constraint[:current_char_count]
    else:
        partial_base64 = base64_constraint

    # Extract position-specific data
    position_data = extract_ssc_positions_for_iteration(pair, position)

    # Format the features/tokens for this position based on mode
    top_features = ""
    feature_descriptions = (
        None  # Will store feature descriptions for ssc_sae_feature_descriptions mode
    )

    if mode == "ssc_logit_lens_tokens":
        top_features = format_ssc_per_token_logitlens_map(
            position_data["ssc_predictions"],
            num_tokens=kwargs.get("num_tokens", 10),
            min_value=kwargs.get("min_value", 0.0),
        )
    elif mode == "ssc_sae_feature_descriptions":
        # Get features at position
        features_at_position = position_data["ssc_per_token_features"]

        # Extract feature indices and descriptions
        feature_descriptions = {}
        if features_at_position:
            for entry in features_at_position:
                raw_feats = entry.get("top_k_features", [])
                # Filter by minimum activation
                filtered_feats = [
                    item
                    for item in raw_feats
                    if float(item.get("activation_value", 0.0))
                    >= float(kwargs.get("min_value", 0.0))
                ][: kwargs.get("num_features", 10)]

                for item in filtered_feats:
                    feat_idx = int(item.get("feature_index", -1))
                    if feat_idx >= 0:
                        # Fetch and store description
                        keeper_model = kwargs.get("keeper_model_name", "")
                        model_name_for_desc = (
                            keeper_model.split("/")[-1]
                            if keeper_model
                            else "gemma-2-9b"
                        )
                        desc = fetch_sae_feature_description(
                            feat_idx,
                            model_name_for_desc,
                            layer=kwargs.get("layer", 32),
                            width_k=kwargs.get("width_k", 131),
                            local_explanations_file=kwargs.get(
                                "local_explanations_file"
                            ),
                        )
                        print(f"Feature {feat_idx}: {desc}")
                        feature_descriptions[feat_idx] = {
                            "description": desc,
                            "activation_value": float(
                                item.get("activation_value", 0.0)
                            ),
                        }

        top_features = format_ssc_per_token_feature_map(
            position_data["ssc_per_token_features"],
            num_features=kwargs.get("num_features", 10),
            min_value=kwargs.get("min_value", 0.0),
            keeper_model_name=kwargs.get("keeper_model_name"),
            layer=kwargs.get("layer", 32),
            width_k=kwargs.get("width_k", 131),
            local_explanations_file=kwargs.get("local_explanations_file"),
        )
    elif mode == "ssc_sae_pos_tokens":
        top_features = format_ssc_per_token_feature_tokens(
            position_data["ssc_per_token_features"],
            num_features=kwargs.get("num_features", 10),
            min_value=kwargs.get("min_value", 0.0),
            num_tokens_per_feature=kwargs.get("num_tokens_per_feature", 5),
        )
    elif mode == "ssc_residual_similarity_tokens":
        top_features = format_ssc_residual_similarity_tokens_only(
            position_data["ssc_predictions"],
            num_tokens=kwargs.get("num_tokens", 10),
            min_value=kwargs.get("min_value", 0.0),
        )

    # Check if we have any tokens/features after filtering
    # Return None if no valid tokens remain after filtering
    if (
        not top_features
        or top_features.strip() == ""
        or "No SSC token information available" in top_features
        or top_features.strip().endswith(":")
        or "(no top tokens available)" in top_features
        or "No tokens available" in top_features
    ):
        return None, None

    # Format predicted words as list of lists
    predicted_words_str = format_predicted_words_list(predicted_words)

    # Features/tokens are captured in top_features variable for the prompt

    prompt = template.format(
        n=num_guesses,
        user_prompt=pair.get("user_prompt", ""),
        model_response=pair.get("model_response", ""),
        base64_constraint=partial_base64,
        predicted_words=predicted_words_str,
        top_features=top_features,
    )

    return prompt, feature_descriptions


def generate_final_constraint_prediction(
    pair: dict,
    predicted_words: List[List[str]],
    engine: InferenceEngine,
    num_final_guesses: int,
    max_new_tokens: int,
    temperature: float,
    final_template_file: str,
    openrouter_client: OpenAI = None,
    openrouter_model_name: str = None,
) -> dict:
    """Generate final constraint prediction using the complete predicted words sequence."""

    try:
        with open(final_template_file, "r", encoding="utf-8") as f:
            final_template = f.read()
    except FileNotFoundError:
        return "Template not found"

    # Filter out None/empty predictions and "None" strings before formatting
    filtered_predicted_words = []
    for position_words in predicted_words:
        if position_words:  # Skip empty positions
            # Filter out None, empty strings, and "None" string within each position
            filtered_position = [
                word for word in position_words if word and word.lower() != "none"
            ]
            if filtered_position:
                filtered_predicted_words.append(filtered_position)

    # Format filtered predicted words as a readable list
    predicted_words_str = format_predicted_words_list(filtered_predicted_words)

    # Create the final prompt
    final_prompt = final_template.format(
        n=num_final_guesses,
        base64_constraint=pair.get("ssc_content", ""),
        user_prompt=pair.get("user_prompt", ""),
        model_response=pair.get("model_response", ""),
        predicted_words=predicted_words_str,
    )

    # Apply chat template (only for local models)
    if openrouter_client is not None:
        chat_formatted_prompt = final_prompt
    else:
        messages = [{"role": "user", "content": final_prompt}]
        chat_formatted_prompt = engine.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )

    # Generate final prediction
    response = generate_single_response(
        prompt=chat_formatted_prompt,
        engine=engine,
        openrouter_client=openrouter_client,
        openrouter_model_name=openrouter_model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    # Parse the final constraint predictions
    is_single_guess = num_final_guesses == 1
    parsed_constraints = parse_final_constraint_predictions(response, is_single_guess)

    return {"raw_response": response.strip(), "parsed_constraints": parsed_constraints}


def process_ssc_iteratively(
    pair: dict,
    engine: InferenceEngine,
    template: str,
    mode: str,
    num_guesses: int,
    num_final_guesses: int,
    max_new_tokens: int,
    temperature: float,
    final_template_file: str,
    **kwargs,
) -> dict:
    """Process a single SSC iteratively, token by token."""
    # Extract OpenRouter params from kwargs
    openrouter_client = kwargs.get("openrouter_client")
    openrouter_model_name = kwargs.get("openrouter_model_name")

    # Extract actual positions from the data
    all_positions = set()
    # Get positions from unified predictions field
    predictions = pair.get("predictions", [])
    for pred in predictions:
        if "ssc_token_index" in pred:
            all_positions.add(pred["ssc_token_index"])

    # Sort positions to process them in order
    positions_to_process = sorted(list(all_positions))
    total_positions = len(positions_to_process)

    if total_positions == 0:
        return {
            "user_prompt": pair.get("user_prompt", ""),
            "base64_constraint": pair.get("ssc_content", ""),
            "ssc_decoded_natural_language": pair.get(
                "ssc_decoded_natural_language", ""
            ),
            "predicted_words_sequence": [],
            "final_prediction": "",
        }

    predicted_words = []  # List of lists - each position has N predictions
    iterations = []

    for i, actual_position in enumerate(positions_to_process):
        # Skip validation step for ssc_sae_feature_descriptions mode
        if mode != "ssc_sae_feature_descriptions":
            # First, validate if tokens provide sufficient evidence for word prediction
            if "validation_template_path" not in kwargs:
                raise ValueError("validation_template_path must be provided in kwargs")
            validation_template_path = kwargs.get("validation_template_path")

            # Create a copy of kwargs without validation_template_path to avoid duplicate argument
            kwargs_for_validation = {
                k: v for k, v in kwargs.items() if k != "validation_template_path"
            }

            is_valid = validate_tokens_for_prediction(
                pair=pair,
                position=actual_position,
                mode=mode,
                engine=engine,
                validation_template_path=validation_template_path,
                max_new_tokens=16,  # Short response expected (Yes/No)
                temperature=temperature,
                **kwargs_for_validation,
            )

            if not is_valid:
                # Store iteration results for skipped position
                iteration_result = {
                    "position": actual_position,
                    "position_index": i,
                    "parsed_predictions": [],
                    "partial_base64": pair.get("ssc_content", "")[
                        : int(
                            (i + 1)
                            * len(pair.get("base64_constraint", ""))
                            / total_positions
                        )
                    ],
                    "skipped_reason": "Insufficient evidence for word prediction (validation failed)",
                    "validation_result": "No",
                }
                iterations.append(iteration_result)
                continue

        # Prepare prompt for this position
        formatted_prompt, position_feature_descriptions = prepare_iterative_prompt(
            template=template,
            pair=pair,
            position=actual_position,
            predicted_words=predicted_words,
            mode=mode,
            num_guesses=num_guesses,
            position_index=i,
            total_positions=total_positions,
            **kwargs,
        )

        # Check if prompt preparation returned None (no tokens after filtering)
        if formatted_prompt is None:
            # Store iteration results for skipped position
            iteration_result = {
                "position": actual_position,
                "position_index": i,
                "parsed_predictions": [],
                "partial_base64": pair.get("base64_constraint", "")[
                    : int(
                        (i + 1)
                        * len(pair.get("base64_constraint", ""))
                        / total_positions
                    )
                ],
                "skipped_reason": "No tokens available after min_value filtering",
            }
            iterations.append(iteration_result)
            continue

        # Apply chat template (only for local models)
        if openrouter_client is not None:
            chat_formatted_prompt = formatted_prompt
        else:
            messages = [{"role": "user", "content": formatted_prompt}]
            chat_formatted_prompt = engine.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
            )

        # Generate response
        response = generate_single_response(
            prompt=chat_formatted_prompt,
            engine=engine,
            openrouter_client=openrouter_client,
            openrouter_model_name=openrouter_model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        # Parse auditor response
        position_predictions = parse_auditor_response(response)

        # Filter out "None" strings from predictions at each position
        filtered_position_predictions = [
            word for word in position_predictions if word and word.lower() != "none"
        ]

        # Step 3: Apply semantic filtering to remove words connected to context
        if filtered_position_predictions:
            # Check if semantic filter template path is provided
            if "semantic_filter_template_path" not in kwargs:
                semantically_filtered_predictions = filtered_position_predictions
            else:
                semantically_filtered_predictions = apply_semantic_filter(
                    pair=pair,
                    predicted_words=filtered_position_predictions,
                    engine=engine,
                    semantic_filter_template_path=kwargs.get(
                        "semantic_filter_template_path"
                    ),
                    num_guesses=num_guesses,
                    max_new_tokens=50,  # Short response expected
                    temperature=temperature,
                    openrouter_client=openrouter_client,
                    openrouter_model_name=openrouter_model_name,
                )

                # If all words were filtered out, skip this iteration
                if not semantically_filtered_predictions:
                    # Store iteration results for skipped position
                    iteration_result = {
                        "position": actual_position,
                        "position_index": i,
                        "parsed_predictions": [],
                        "partial_base64": pair.get("ssc_content", "")[
                            : int(
                                (i + 1)
                                * len(pair.get("base64_constraint", ""))
                                / total_positions
                            )
                        ],
                        "skipped_reason": "All words filtered out by semantic filter",
                        "validation_result": "Yes",
                        "pre_filter_predictions": filtered_position_predictions,
                    }
                    iterations.append(iteration_result)
                    continue
        else:
            semantically_filtered_predictions = []

        # Store iteration results (using semantically filtered predictions)
        iteration_result = {
            "position": actual_position,
            "position_index": i,
            "parsed_predictions": semantically_filtered_predictions,
            "partial_base64": pair.get("base64_constraint", "")[
                : int(
                    (i + 1) * len(pair.get("base64_constraint", "")) / total_positions
                )
            ],
            "validation_result": "Yes",
            "pre_filter_predictions": filtered_position_predictions
            if filtered_position_predictions != semantically_filtered_predictions
            else None,
        }

        # Add SAE feature descriptions if available (for ssc_sae_feature_descriptions mode)
        if position_feature_descriptions:
            iteration_result["sae_feature_descriptions"] = position_feature_descriptions

        iterations.append(iteration_result)

        # Add predictions to the sequence (pad with empty if no predictions)
        if semantically_filtered_predictions:
            predicted_words.append(
                semantically_filtered_predictions[:num_guesses]
            )  # Limit to requested number
        else:
            predicted_words.append([])  # Empty list if no predictions
    # Create final prediction by taking first prediction from each position
    final_prediction_words = []
    for position_preds in predicted_words:
        if position_preds:
            final_prediction_words.append(position_preds[0])

    final_prediction = " ".join(final_prediction_words)

    # Generate final constraint prediction using complete predicted words
    final_constraint_result = generate_final_constraint_prediction(
        pair=pair,
        predicted_words=predicted_words,
        engine=engine,
        num_final_guesses=num_final_guesses,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        final_template_file=final_template_file,
        openrouter_client=openrouter_client,
        openrouter_model_name=openrouter_model_name,
    )

    result = {
        "user_prompt": pair.get("user_prompt", ""),
        "base64_constraint": pair.get("ssc_content", ""),
        "ssc_decoded_natural_language": pair.get("ssc_decoded_natural_language", ""),
        "predicted_words_sequence": predicted_words,
        "final_prediction": final_prediction,
        "final_constraint_prediction": final_constraint_result,
        "total_positions": total_positions,
    }

    # If using ssc_sae_feature_descriptions mode, compile all unique SAE features used
    if mode == "ssc_sae_feature_descriptions":
        all_sae_features = {}
        for iteration in iterations:
            if (
                "sae_feature_descriptions" in iteration
                and iteration["sae_feature_descriptions"]
            ):
                for feat_idx, feat_info in iteration[
                    "sae_feature_descriptions"
                ].items():
                    if feat_idx not in all_sae_features:
                        all_sae_features[feat_idx] = feat_info
                    else:
                        # Update with max activation value if already exists
                        existing_act = all_sae_features[feat_idx].get(
                            "activation_value", 0.0
                        )
                        new_act = feat_info.get("activation_value", 0.0)
                        if new_act > existing_act:
                            all_sae_features[feat_idx]["activation_value"] = new_act

        # Add summary of all SAE features used
        if all_sae_features:
            result["sae_features_summary"] = {
                "total_unique_features": len(all_sae_features),
                "features": all_sae_features,
            }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Guess SSC secrets iteratively using token-by-token auditing."
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
        choices=[
            "ssc_logit_lens_tokens",
            "ssc_sae_feature_descriptions",
            "ssc_sae_pos_tokens",
            "ssc_residual_similarity_tokens",
        ],
        help="Auditor mode.",
    )
    parser.add_argument(
        "--num_guesses",
        type=int,
        default=3,
        help="Number of word guesses per position.",
    )
    parser.add_argument(
        "--num_final_guesses",
        type=int,
        default=5,
        help="Number of final constraint predictions.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Directory to save results."
    )
    parser.add_argument(
        "--prompt_template_file",
        type=str,
        required=True,
        help="Path to iterative prompt template file.",
    )
    parser.add_argument(
        "--validation_template_file",
        type=str,
        default="prompts/ssc/ssc_auditor_guess_tokens_valid.txt",
        help="Path to validation prompt template file.",
    )
    parser.add_argument(
        "--semantic_filter_template_file",
        type=str,
        default="prompts/ssc/ssc_auditor_guess_words_semantic_filter.txt",
        help="Path to semantic filter prompt template file.",
    )
    parser.add_argument(
        "--final_template_file",
        type=str,
        required=True,
        help="Path to final constraint prediction prompt template file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (cuda/cpu).",
    )

    # Generation parameters
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)

    # Mode-specific parameters
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=10,
        help="Max number of tokens for logit lens modes.",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=10,
        help="Max number of SAE features for SAE modes.",
    )
    parser.add_argument(
        "--min_value", type=float, default=0.0, help="Minimum threshold value."
    )
    parser.add_argument(
        "--num_tokens_per_feature",
        type=int,
        default=5,
        help="Tokens per feature for ssc_sae_pos_tokens mode.",
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
        help="Path to local explanations JSON file for Llama models.",
    )

    # OpenRouter parameters
    parser.add_argument(
        "--openrouter_model",
        action="store_true",
        help="Use OpenRouter API instead of local model. Requires OPENROUTER_API_KEY in .env",
    )

    # Random seed
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--max_sscs",
        type=int,
        default=None,
        help="Maximum number of SSCs to process (for testing).",
    )

    args = parser.parse_args()

    load_dotenv()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    DEVICE = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Validate inputs
    if not os.path.exists(args.data_file):
        print(f"Error: Data file not found at {args.data_file}")
        sys.exit(1)

    if not os.path.exists(args.prompt_template_file):
        print(f"Error: Prompt template file not found at {args.prompt_template_file}")
        sys.exit(1)

    # Load prompt template
    with open(args.prompt_template_file, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    # Initialize model/client based on mode
    if args.openrouter_model:
        print(f"Using OpenRouter API with model: {args.model_name}")
        openrouter_client = create_openrouter_client()
        engine = None
    else:
        print(f"Loading local model: {args.model_name}")
        model, tokenizer = load_model_and_tokenizer(args.model_name, device=DEVICE)
        if "Llama" in args.model_name:
            tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set offset = 1 %}{% else %}{% set offset = 0 %}{% endif %}{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>' + 'assistant' + '<|end_header_id|>\\n\\n' }}{% endif %}"
            tokenizer.padding_side = "left"
        engine = InferenceEngine(model, tokenizer)
        openrouter_client = None

    # Load SSC analysis data
    data = load_data_file(args.data_file)
    if not data:
        print("Failed to load data file. Exiting.")
        sys.exit(1)

    pairs = data.get("results", [])
    if not pairs:
        print("No usable entries found in the data file. Exiting.")
        sys.exit(1)

    # Limit number of SSCs for testing
    if args.max_sscs is not None:
        pairs = pairs[: args.max_sscs]

    # Process each SSC iteratively
    all_guesses = []

    for pair in tqdm(pairs, desc="Processing SSCs"):
        result = process_ssc_iteratively(
            pair=pair,
            engine=engine,
            template=prompt_template,
            mode=args.mode,
            num_guesses=args.num_guesses,
            num_final_guesses=args.num_final_guesses,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            final_template_file=args.final_template_file,
            num_tokens=args.num_tokens,
            num_features=args.num_features,
            min_value=args.min_value,
            num_tokens_per_feature=args.num_tokens_per_feature,
            validation_template_path=args.validation_template_file,
            semantic_filter_template_path=args.semantic_filter_template_file,
            keeper_model_name=args.keeper_model_name,
            layer=args.layer,
            width_k=args.width_k,
            local_explanations_file=args.local_explanations_file,
            openrouter_client=openrouter_client if args.openrouter_model else None,
            openrouter_model_name=args.model_name if args.openrouter_model else None,
        )

        # Convert to the format expected by calculate_auditing_metrics.py
        guess_entry = {
            "user_prompt": result["user_prompt"],
            "base64_constraint": result["base64_constraint"],
            "ssc_char_count": int(len(result["base64_constraint"]) * 0.75)
            if result["base64_constraint"]
            else 0,
            "model_response": pair.get("model_response", ""),
            "ssc_decoded_natural_language": result["ssc_decoded_natural_language"],
            "auditor_response_index": 0,
            "guessed_constraints": result["final_constraint_prediction"].get(
                "parsed_constraints", []
            ),
            "auditor_full_response": result["final_constraint_prediction"].get(
                "raw_response", ""
            ),
            "mode": args.mode,
            "iterative_details": {
                "predicted_words_sequence": result["predicted_words_sequence"],
                "final_prediction": result["final_prediction"],
                "total_positions": result["total_positions"],
            },
        }
        all_guesses.append(guess_entry)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_ssc_iterative_guessing_results.json"
    output_path = os.path.join(args.output_dir, filename)

    results = {
        "auditor_model_name": args.model_name,
        "source_data_file": args.data_file,
        "mode": args.mode,
        "num_guesses_per_position": args.num_guesses,
        "num_final_guesses": args.num_final_guesses,
        "mode_config": {
            "mode": args.mode,
            "mode_description": f"Iterative token-by-token prediction using {args.mode}",
            "num_tokens": args.num_tokens
            if args.mode in ["ssc_logit_lens_tokens", "ssc_residual_similarity_tokens"]
            else None,
            "num_features": args.num_features
            if args.mode in ["ssc_sae_feature_descriptions", "ssc_sae_pos_tokens"]
            else None,
            "min_value": args.min_value,
            "num_tokens_per_feature": args.num_tokens_per_feature
            if args.mode == "ssc_sae_pos_tokens"
            else None,
        },
        "generation_params": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "num_responses_per_prompt": 1,
            "batch_size": 1,
            "openrouter_model": args.openrouter_model,
        },
        "experiment_timestamp": datetime.now().isoformat(),
        "total_entries": len(pairs),
        "total_auditor_responses": len(all_guesses),
        "guesses": all_guesses,
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    except IOError as e:
        print(f"Error saving results to file: {e}")


if __name__ == "__main__":
    main()
