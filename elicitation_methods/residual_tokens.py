#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sampling.utils import load_model_and_tokenizer
from utils.utils import (
    base64_to_string,
    detect_model_family,
    extract_ssc_content,
    find_first_gt_start_token,
    find_first_person_pronoun_positions,
    find_second_start_of_turn_position,
    get_assistant_control_token_count,
    load_items_ssc,
    load_results,
)


class ResidualCaptureHook:
    """Context manager for capturing residual stream representations."""

    def __init__(self, model, target_layer: int):
        self.model = model
        self.target_layer = target_layer
        self.representations = None
        self.hook_handle = None

    def _capture_hook(self, module, input, output):
        """Hook function to capture the residual stream output."""
        # output is typically a tuple, we want the hidden states
        if isinstance(output, tuple):
            hidden_states = output[0]  # (batch_size, seq_len, hidden_dim)
        else:
            hidden_states = output

        # Store the representations (detach to avoid keeping gradients)
        self.representations = hidden_states.detach().clone()

    def __enter__(self):
        # Register hook at target layer
        target_layer_module = self.model.model.layers[self.target_layer]
        self.hook_handle = target_layer_module.register_forward_hook(self._capture_hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up hook
        if self.hook_handle is not None:
            self.hook_handle.remove()


def extract_residual_similarities(
    model, tokens: torch.Tensor, target_layer: int
) -> np.ndarray:
    """Extract residual stream representations and compute token similarities.

    Returns:
        np.ndarray: Shape (seq_len, vocab_size) containing similarity scores
    """
    # Use hook to capture residual representations
    with ResidualCaptureHook(model, target_layer) as hook:
        with torch.no_grad():
            # Run forward pass to capture representations
            _ = model(tokens.unsqueeze(0).to(model.device))
            residual_reps = hook.representations

    if residual_reps is None:
        raise RuntimeError("Failed to capture residual representations")

    # Get token embeddings from the model
    token_embeddings = model.model.embed_tokens.weight  # (vocab_size, hidden_dim)

    # Compute dot product between residual representations and token embeddings
    # residual_reps: (1, seq_len, hidden_dim)
    # token_embeddings: (vocab_size, hidden_dim)
    # Result: (seq_len, vocab_size)
    similarities = (
        F.normalize(residual_reps[0], dim=1) @ F.normalize(token_embeddings, dim=1).T
    )  # Remove batch dimension

    # Convert to numpy
    similarities_np = similarities.detach().cpu().to(dtype=torch.float32).numpy()

    return similarities_np


def get_top_k_similar_tokens(
    similarities: np.ndarray, tokenizer, k: int
) -> List[Dict[str, Any]]:
    """Get top-k most similar tokens with their similarity scores."""
    # Get top-k indices
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_similarities = similarities[top_k_indices]

    # Convert to tokens
    top_k_tokens = []
    for i, (token_id, similarity) in enumerate(zip(top_k_indices, top_k_similarities)):
        token_str = tokenizer.decode([token_id])
        top_k_tokens.append(
            {
                "rank": i + 1,
                "token_id": int(token_id),
                "token": token_str,
                "similarity": float(similarity),
            }
        )

    return top_k_tokens


def get_top_k_tokens_ssc(
    probabilities: np.ndarray, tokenizer, k: int
) -> List[Dict[str, Any]]:
    """Get top-k tokens for SSC mode (uses 'probability' field name for consistency)."""
    k = int(k)
    k = max(1, min(k, probabilities.shape[-1]))
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    top_k_probs = probabilities[top_k_indices]
    top_k_tokens: List[Dict[str, Any]] = []
    for i, (token_id, prob) in enumerate(zip(top_k_indices, top_k_probs)):
        token_str = tokenizer.decode([int(token_id)])
        top_k_tokens.append(
            {
                "rank": i + 1,
                "token_id": int(token_id),
                "token": token_str,
                "probability": float(prob),
            }
        )
    return top_k_tokens


def get_combined_top_k_tokens_control(
    start_of_turn_similarities: np.ndarray,
    model_similarities: np.ndarray,
    tokenizer,
    k: int,
) -> List[Dict[str, Any]]:
    """Get top-k most similar tokens from both control positions, sorted by similarity scores."""
    # Get top-k indices from both positions
    start_of_turn_top_k_indices = np.argsort(start_of_turn_similarities)[-k:][::-1]
    model_top_k_indices = np.argsort(model_similarities)[-k:][::-1]

    # Get similarities for top-k from both positions
    start_of_turn_top_k_similarities = start_of_turn_similarities[
        start_of_turn_top_k_indices
    ]
    model_top_k_similarities = model_similarities[model_top_k_indices]

    # Combine all similarities and their corresponding token IDs
    all_similarities = []
    all_token_ids = []
    all_positions = []

    # Add start_of_turn similarities
    for token_id, similarity in zip(
        start_of_turn_top_k_indices, start_of_turn_top_k_similarities
    ):
        all_similarities.append(similarity)
        all_token_ids.append(token_id)
        all_positions.append("start_of_turn")

    # Add model similarities
    for token_id, similarity in zip(model_top_k_indices, model_top_k_similarities):
        all_similarities.append(similarity)
        all_token_ids.append(token_id)
        all_positions.append("model")

    # Sort by similarity scores (descending)
    sorted_indices = np.argsort(all_similarities)[::-1]

    # Take top-k overall
    top_k_tokens = []
    seen_tokens = set()  # To avoid duplicates

    for i, sorted_idx in enumerate(sorted_indices):
        if len(top_k_tokens) >= k:
            break

        token_id = all_token_ids[sorted_idx]
        similarity = all_similarities[sorted_idx]
        position = all_positions[sorted_idx]

        # Skip if we've already seen this token (keep highest similarity)
        if token_id in seen_tokens:
            continue

        seen_tokens.add(token_id)

        token_str = tokenizer.decode([token_id])
        top_k_tokens.append(
            {
                "rank": len(top_k_tokens) + 1,
                "token_id": int(token_id),
                "token": token_str,
                "similarity": float(similarity),
                "position": position,
            }
        )

    return top_k_tokens


def get_combined_top_k_tokens_control_multi(
    control_similarities: List[np.ndarray],
    control_positions: List[int],
    tokenizer,
    k: int,
) -> List[Dict[str, Any]]:
    """Get top-k most similar tokens from all control positions, sorted by similarity scores.

    This is a generalized version that works with any number of control tokens.
    """
    # Collect all similarities and their corresponding token IDs and positions
    all_similarities = []
    all_token_ids = []
    all_positions = []

    for pos, similarities in zip(control_positions, control_similarities):
        # Get top-k indices from this position
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_similarities = similarities[top_k_indices]

        # Add to combined list
        for token_id, similarity in zip(top_k_indices, top_k_similarities):
            all_similarities.append(similarity)
            all_token_ids.append(token_id)
            all_positions.append(pos)

    # Sort by similarity scores (descending)
    sorted_indices = np.argsort(all_similarities)[::-1]

    # Take top-k overall
    top_k_tokens = []
    seen_tokens = set()  # To avoid duplicates

    for i, sorted_idx in enumerate(sorted_indices):
        if len(top_k_tokens) >= k:
            break

        token_id = all_token_ids[sorted_idx]
        similarity = all_similarities[sorted_idx]
        position = all_positions[sorted_idx]

        # Skip if we've already seen this token (keep highest similarity)
        if token_id in seen_tokens:
            continue

        seen_tokens.add(token_id)

        token_str = tokenizer.decode([token_id])
        top_k_tokens.append(
            {
                "rank": len(top_k_tokens) + 1,
                "token_id": int(token_id),
                "token": token_str,
                "similarity": float(similarity),
                "position": int(position),
            }
        )

    return top_k_tokens


def get_combined_top_k_tokens_pronouns(
    pronoun_similarities: List[np.ndarray],
    pronoun_positions: List[int],
    tokenizer,
    k: int,
) -> List[Dict[str, Any]]:
    """Get top-k most similar tokens from all pronoun positions, sorted by similarity scores."""
    # Collect all similarities and their corresponding token IDs and positions
    all_similarities = []
    all_token_ids = []
    all_positions = []

    for pos_idx, (pos, similarities) in enumerate(
        zip(pronoun_positions, pronoun_similarities)
    ):
        # Get top-k indices from this position
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        top_k_similarities = similarities[top_k_indices]

        # Add to combined list
        for token_id, similarity in zip(top_k_indices, top_k_similarities):
            all_similarities.append(similarity)
            all_token_ids.append(token_id)
            all_positions.append(pos)

    # Sort by similarity scores (descending)
    sorted_indices = np.argsort(all_similarities)[::-1]

    # Take top-k overall
    top_k_tokens = []
    seen_tokens = set()  # To avoid duplicates

    for i, sorted_idx in enumerate(sorted_indices):
        if len(top_k_tokens) >= k:
            break

        token_id = all_token_ids[sorted_idx]
        similarity = all_similarities[sorted_idx]
        position = all_positions[sorted_idx]

        # Skip if we've already seen this token (keep highest similarity)
        if token_id in seen_tokens:
            continue

        seen_tokens.add(token_id)

        token_str = tokenizer.decode([token_id])
        top_k_tokens.append(
            {
                "rank": len(top_k_tokens) + 1,
                "token_id": int(token_id),
                "token": token_str,
                "similarity": float(similarity),
                "position": int(position),
            }
        )

    return top_k_tokens


def process_input_output_pair_control(
    model,
    tokenizer,
    user_prompt: str,
    model_response: str,
    target_layer: int,
    top_k: int,
) -> Dict[str, Any]:
    """Process a single input-output pair and extract top-k similar tokens at control positions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        user_prompt: User input prompt
        model_response: Model response
        target_layer: Target layer for residual stream extraction
        top_k: Number of top similar tokens to extract

    Returns:
        Dictionary with results including top_k_tokens for both control positions
    """

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

    # Find the second <start_of_turn> position
    start_of_turn_position = find_second_start_of_turn_position(tokens, tokenizer)

    if start_of_turn_position == -1:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": "Assistant turn start token not found",
            "start_of_turn_position": None,
            "model_position": None,
            "top_k_tokens": [],
        }

    # Get control token count for this model architecture
    control_token_count = get_assistant_control_token_count(tokenizer)

    # Check if we have all control tokens
    if start_of_turn_position + control_token_count - 1 >= len(tokens):
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": "Not enough tokens for assistant control sequence",
            "start_of_turn_position": start_of_turn_position,
            "model_position": None,
            "top_k_tokens": [],
        }

    # For backward compatibility, model_position refers to the last control token
    model_position = start_of_turn_position + control_token_count - 1

    # Extract residual similarities at the target layer
    try:
        similarities = extract_residual_similarities(model, tokens, target_layer)

        # Get similarities at all control token positions
        control_positions = list(
            range(start_of_turn_position, start_of_turn_position + control_token_count)
        )
        control_similarities = [similarities[pos, :] for pos in control_positions]

        # Get combined top-k similar tokens from all control positions
        top_k_tokens = get_combined_top_k_tokens_control_multi(
            control_similarities, control_positions, tokenizer, top_k
        )

        # Get the actual tokens at these positions for verification
        start_of_turn_actual_token = tokenizer.decode(
            [tokens[start_of_turn_position].item()]
        )
        model_actual_token = tokenizer.decode([tokens[model_position].item()])

        result = {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "start_of_turn_position": int(start_of_turn_position),
            "model_position": int(model_position),
            "sequence_length": len(tokens),
            "start_of_turn_actual_token": start_of_turn_actual_token,
            "model_actual_token": model_actual_token,
            "top_k_tokens": top_k_tokens,
        }

        return result

    except Exception as e:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Error processing: {str(e)}",
            "start_of_turn_position": int(start_of_turn_position)
            if start_of_turn_position != -1
            else None,
            "model_position": int(model_position)
            if model_position < len(tokens)
            else None,
            "top_k_tokens": [],
        }


def process_input_output_pair_pronouns(
    model,
    tokenizer,
    user_prompt: str,
    model_response: str,
    target_layer: int,
    top_k: int,
) -> Dict[str, Any]:
    """Process a single input-output pair and extract top-k similar tokens at pronoun positions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        user_prompt: User input prompt
        model_response: Model response
        target_layer: Target layer for residual stream extraction
        top_k: Number of top similar tokens to extract

    Returns:
        Dictionary with results including top_k_tokens for all pronoun positions
    """

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

    # Find the second <start_of_turn> position for reference
    start_of_turn_position = find_second_start_of_turn_position(tokens, tokenizer)

    # Find all first person pronoun positions in the user prompt only
    pronoun_positions = find_first_person_pronoun_positions(
        tokens, tokenizer, start_of_turn_position
    )

    if len(pronoun_positions) == 0:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": "No first person pronouns found in the prompt",
            "start_of_turn_position": int(start_of_turn_position)
            if start_of_turn_position != -1
            else None,
            "pronoun_positions": [],
            "top_k_tokens": [],
        }

    # Extract residual similarities at the target layer
    try:
        similarities = extract_residual_similarities(model, tokens, target_layer)

        # Get similarities at all pronoun positions
        pronoun_similarities = []
        for pos in pronoun_positions:
            pronoun_similarities.append(similarities[pos, :])

        # Get combined top-k similar tokens from all pronoun positions
        top_k_tokens = get_combined_top_k_tokens_pronouns(
            pronoun_similarities, pronoun_positions, tokenizer, top_k
        )

        # Get the actual tokens at these positions for verification
        pronoun_actual_tokens = []
        for pos in pronoun_positions:
            token_str = tokenizer.decode([tokens[pos].item()])
            pronoun_actual_tokens.append({"position": int(pos), "token": token_str})

        result = {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "start_of_turn_position": int(start_of_turn_position)
            if start_of_turn_position != -1
            else None,
            "pronoun_positions": [int(pos) for pos in pronoun_positions],
            "sequence_length": len(tokens),
            "pronoun_actual_tokens": pronoun_actual_tokens,
            "num_pronouns_found": len(pronoun_positions),
            "top_k_tokens": top_k_tokens,
        }

        return result

    except Exception as e:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Error processing: {str(e)}",
            "start_of_turn_position": int(start_of_turn_position)
            if start_of_turn_position != -1
            else None,
            "pronoun_positions": [int(pos) for pos in pronoun_positions]
            if len(pronoun_positions) > 0
            else [],
            "top_k_tokens": [],
        }


def process_prompt_ssc(
    model,
    tokenizer,
    prompt_text: str,
    target_layer: int,
    top_k: int,
) -> Dict[str, Any]:
    """Process a single prompt for SSC mode and extract top-k similar tokens at SSC positions.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt_text: Prompt text containing SSC tags
        target_layer: Target layer for residual stream extraction
        top_k: Number of top similar tokens to extract

    Returns:
        Dictionary with results including predictions for each SSC token position
    """
    ssc_content = extract_ssc_content(prompt_text)
    if not ssc_content:
        return {
            "user_prompt": prompt_text,
            "error": "No <antml:secret_side_constraint>...</antml:secret_side_constraint> section found",
            "predictions": [],
        }

    # Always decode SSC content to natural language (e.g., base64)
    ssc_decoded_nl = base64_to_string(ssc_content.strip())

    messages = [{"role": "user", "content": prompt_text}]
    formatted_sequence = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        add_special_tokens=False,
    )

    tokens = tokenizer.encode(
        formatted_sequence, add_special_tokens=False, return_tensors="pt"
    ).squeeze(0)
    ssc_token_ids = tokenizer.encode(ssc_content, add_special_tokens=False)

    full_ids = tokens.tolist()
    start_index = find_first_gt_start_token(tokenizer, full_ids)
    if start_index == -1:
        return {
            "user_prompt": prompt_text,
            "ssc_content": ssc_content,
            "ssc_decoded_natural_language": ssc_decoded_nl,
            "error": "SSC start token '>' not found in tokenized prompt",
            "predictions": [],
        }

    # Extract residual similarities instead of logit lens probabilities
    similarities = extract_residual_similarities(model, tokens, target_layer)

    predictions: List[Dict[str, Any]] = []
    # If the start token is exactly '>', align the first SSC token to the next position.
    start_tok_str = tokenizer.decode([int(full_ids[start_index])])
    base_pos = start_index + 1 if start_tok_str == ">" else start_index

    max_len = min(len(ssc_token_ids), max(0, similarities.shape[0] - base_pos))
    for offset in range(max_len):
        ssc_tok = ssc_token_ids[offset]
        pos = base_pos + offset
        pos_probs = similarities[pos, :]
        topk_for_pos = get_top_k_tokens_ssc(pos_probs, tokenizer, top_k)
        predictions.append(
            {
                "position": int(pos),
                "ssc_token_index": int(offset),
                "ssc_token_id": int(ssc_tok),
                "ssc_token": tokenizer.decode([int(ssc_tok)]),
                "top_k": topk_for_pos,
            }
        )

    return {
        "user_prompt": prompt_text,
        "ssc_content": ssc_content,
        "ssc_decoded_natural_language": ssc_decoded_nl,
        "ssc_token_count": len(ssc_token_ids),
        "ssc_start_token_index": int(start_index),
        "ssc_start_token": start_tok_str,
        "predictions": predictions,
    }


def main():
    """Main function to extract top-k similar tokens for specified positions in all input-output pairs."""
    parser = argparse.ArgumentParser(
        description="Extract top-k most similar tokens using residual stream similarity at specified token positions."
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["control", "pronouns", "ssc"],
        help="Mode of operation: 'control' for control tokens (<start_of_turn>, model), 'pronouns' for first person pronouns, or 'ssc' for secret side constraint tokens.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to load for analysis.",
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
        help="Target layer for residual stream extraction (0-indexed).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top similar tokens to extract (default: 10).",
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

    args = parser.parse_args()

    load_dotenv()

    # Configuration
    DEVICE = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load the model for residual stream analysis
    print(f"Loading model: {args.model_name} on {DEVICE}")
    model, tokenizer = load_model_and_tokenizer(args.model_name, device=DEVICE)

    # Validate layer number
    num_layers = len(model.model.layers)
    if args.layer >= num_layers or args.layer < 0:
        print(
            f"Error: Layer {args.layer} is out of range. Model has {num_layers} layers (0-{num_layers - 1})"
        )
        sys.exit(1)

    print(f"Model has {num_layers} layers, analyzing layer {args.layer}")

    # Mode-specific information
    if args.mode == "control":
        model_family = detect_model_family(tokenizer)
        control_count = get_assistant_control_token_count(tokenizer)
        print("Mode: Control tokens")
        print(
            f"Will extract residual stream similarities at {control_count} assistant control token positions ({model_family} format)"
        )
        method_name = "residual_stream_similarity"
        filename_mode = "residual_similarity_control_tokens"
    elif args.mode == "pronouns":
        print("Mode: Pronouns")
        print(
            "Will extract residual stream similarities at first person pronoun token positions in the user prompt only"
        )
        print(
            "First person pronouns: i, me, my, mine, we, us, our, ours (including with leading whitespace)"
        )
        method_name = "residual_stream_similarity_pronouns"
        filename_mode = "residual_similarity_pronouns_tokens"
    else:  # ssc
        print("Mode: SSC (Secret Side Constraint)")
        print(
            "Will extract residual stream similarities at SSC token positions within <antml:secret_side_constraint> tags"
        )
        method_name = "residual_stream_similarity"
        filename_mode = "ssc_residual_topk"

    if args.mode in ["control", "pronouns"]:
        print("Results will contain combined top-k tokens sorted by similarity scores")
    else:
        print("Results will contain top-k tokens for each SSC token position")

    # Handle different modes
    if args.mode == "ssc":
        # SSC mode has different data loading and processing
        items = load_items_ssc(args.data_file)
        if not items:
            print("No items found in the data file. Exiting.")
            sys.exit(1)

        print(f"Processing {len(items)} prompts...")
        all_results = []
        processed = 0
        errors = 0

        for idx, item in enumerate(tqdm(items, desc="Processing prompts")):
            prompt_text = None
            model_response = ""

            if isinstance(item, dict):
                prompt_text = (
                    item.get("user_prompt")
                    or item.get("user_prompt")
                    or item.get("full_prompt")
                )
                model_response = item.get("model_response", "")
            elif isinstance(item, str):
                prompt_text = item

            if not prompt_text:
                msg = f"Missing prompt text at index {idx}"
                print(f"Error: {msg}")
                continue

            result = process_prompt_ssc(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                target_layer=args.layer,
                top_k=args.top_k,
            )

            if "error" in result:
                err_msg = result.get("error", "Unknown error")
                print(f"Error processing prompt {idx}: {err_msg}")
                errors += 1
            else:
                processed += 1

            # Attach prompt index, model response, and collect
            result["prompt_index"] = idx
            result["model_response"] = model_response
            all_results.append(result)

        total_pairs = len(items)
        processed_pairs = processed
        error_count = errors

    else:
        # Control and pronouns modes
        input_output_pairs = load_results(args.data_file)
        if not input_output_pairs:
            print("No input-output pairs found in the data file. Exiting.")
            sys.exit(1)

        # Process all input-output pairs
        all_results = []
        total_pairs = 0
        processed_pairs = 0
        error_count = 0

        print(f"Processing {len(input_output_pairs)} input-output pairs...")

        # Select processing function based on mode
        process_function = (
            process_input_output_pair_control
            if args.mode == "control"
            else process_input_output_pair_pronouns
        )

        for pair_data in tqdm(input_output_pairs, desc="Processing pairs"):
            user_prompt = pair_data.get("user_prompt", "")
            model_response = pair_data.get("model_response", "")
            response_idx = pair_data.get("model_response_index", 0)

            total_pairs += 1

            result = process_function(
                model=model,
                tokenizer=tokenizer,
                user_prompt=user_prompt,
                model_response=model_response,
                target_layer=args.layer,
                top_k=args.top_k,
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
    results = {
        "metadata": {
            "model_name": args.model_name,
            "source_data_file": args.data_file,
            "source_filename": os.path.basename(args.data_file),
            "target_layer": args.layer,
            "top_k": args.top_k,
            "mode": args.mode,
            "method": method_name,
            "total_pairs" if args.mode != "ssc" else "total_prompts": total_pairs,
            "processed_pairs"
            if args.mode != "ssc"
            else "processed_prompts": processed_pairs,
            "error_count": error_count,
            "experiment_timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }

    # Save the results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_name.replace("/", "_")

    # Construct output filename
    filename = f"{filename_mode}_layer{args.layer}_k{args.top_k}_{model_name_safe}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, filename)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        print(f"Source inference file: {os.path.basename(args.data_file)}")
        print(f"Method: {method_name}")
        print(f"Mode: {args.mode}")
        print(f"Processed {processed_pairs}/{total_pairs} pairs successfully")
        if error_count > 0:
            print(f"Encountered {error_count} errors")
    except IOError as e:
        print(f"\nError saving results to file: {e}")


if __name__ == "__main__":
    main()
