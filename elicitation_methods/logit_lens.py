#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.sae_utils import ObservableLanguageModel
from utils.utils import (
    base64_to_string,
    detect_model_family,
    extract_ssc_content,
    find_first_gt_start_token,
    find_first_person_pronoun_positions,
    find_second_start_of_turn_position,
    get_assistant_control_token_count,
    load_results,
)


def extract_layer_probabilities(
    model, merged_tokens: torch.Tensor, target_layer: int
) -> np.ndarray:
    """Extract probabilities at a specific layer for all token positions."""
    layers = model._model.model.layers
    all_layer_probs = []

    # Use nnsight tracing to get layer outputs
    with model._model.trace() as tracer:
        with tracer.invoke(merged_tokens.unsqueeze(0)):
            # Get the output from the target layer
            layer = layers[target_layer]
            layer_output = model._model.lm_head(
                model._model.model.norm(layer.output[0])
            )

            # Apply softmax to obtain probabilities
            probs = torch.nn.functional.softmax(layer_output, dim=-1).save()
            all_layer_probs.append(probs)

    # Convert to numpy and return
    layer_probs = torch.stack([probs.value for probs in all_layer_probs])
    layer_probs = layer_probs.detach().cpu().to(dtype=torch.float32).numpy()
    layer_probs = layer_probs.squeeze(1).squeeze(0)
    return layer_probs  # Remove batch dimension


def get_top_k_tokens(
    probabilities: np.ndarray, tokenizer, k: int
) -> List[Dict[str, Any]]:
    """Get top-k tokens with their probabilities and token IDs."""
    # Get top-k indices
    top_k_indices = np.argsort(probabilities)[-k:][::-1]
    top_k_probs = probabilities[top_k_indices]

    # Convert to tokens
    top_k_tokens = []
    for i, (token_id, prob) in enumerate(zip(top_k_indices, top_k_probs)):
        token_str = tokenizer.decode([token_id])
        top_k_tokens.append(
            {
                "rank": i + 1,
                "token_id": int(token_id),
                "token": token_str,
                "probability": float(prob),
            }
        )

    return top_k_tokens


def process_input_output_pair(
    model,
    tokenizer,
    user_prompt: str,
    model_response: str,
    target_layer: int,
    top_k: int,
    mode: str = "single_position",
) -> Dict[str, Any]:
    """Process a single input-output pair and extract top-k tokens.

    Args:
        model: The observable language model
        tokenizer: The tokenizer
        user_prompt: User input prompt
        model_response: Model response
        target_layer: Target layer for probability extraction
        top_k: Number of top tokens to extract
        mode: Extraction mode ('single_position', 'control_tokens_average', 'response_tokens_zeroed', 'first_person_pronouns', or 'ssc_tokens')

    Returns:
        Dictionary with results including top_k_tokens
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

    # Find the second <start_of_turn> position (not needed for ssc_tokens mode)
    if mode != "ssc_tokens":
        target_position = find_second_start_of_turn_position(tokens, tokenizer)

        if target_position == -1:
            return {
                "user_prompt": user_prompt,
                "model_response": model_response,
                "error": "Assistant turn start token not found",
                "target_position": None,
                "mode": mode,
                "top_k_tokens": [],
            }
    else:
        # For ssc_tokens mode, target_position is not used
        target_position = None

    # Extract probabilities at the target layer
    try:
        layer_probs = extract_layer_probabilities(model, tokens, target_layer)

        if mode == "single_position":
            # Get probabilities at the target position only
            position_probs = layer_probs[target_position, :]
            positions_used = [target_position]

        elif mode == "control_tokens_average":
            # Get the number of control tokens for this model architecture
            control_token_count = get_assistant_control_token_count(tokenizer)

            # Check if we have enough tokens for averaging
            if target_position + control_token_count - 1 >= layer_probs.shape[0]:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "Not enough tokens for control tokens averaging",
                    "target_position": target_position,
                    "mode": mode,
                    "top_k_tokens": [],
                }

            # Get probabilities at all control token positions and average them
            positions_used = list(
                range(target_position, target_position + control_token_count)
            )
            control_probs = [layer_probs[pos, :] for pos in positions_used]
            position_probs = np.mean(control_probs, axis=0)

        elif mode == "response_tokens_zeroed":
            # Average probabilities across all model response tokens, zeroing out actually predicted tokens
            # Start from after all control tokens (response content begins after control sequence)
            control_token_count = get_assistant_control_token_count(tokenizer)
            response_start_pos = target_position + control_token_count

            if response_start_pos >= layer_probs.shape[0]:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "No tokens found after assistant control sequence for response analysis",
                    "target_position": target_position,
                    "mode": mode,
                    "top_k_tokens": [],
                }

            # Get all response token positions (from after <start_of_turn> to end)
            response_positions = list(range(response_start_pos, layer_probs.shape[0]))
            positions_used = response_positions

            if len(response_positions) == 0:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "No response tokens found",
                    "target_position": target_position,
                    "mode": mode,
                    "top_k_tokens": [],
                }

            # Collect probabilities for each position, zeroing out actually predicted tokens
            vocab_size = layer_probs.shape[1]
            accumulated_probs = np.zeros(vocab_size)

            for pos in response_positions:
                # Get probabilities at this position
                pos_probs = layer_probs[pos, :].copy()

                # Zero out the probability of the actually predicted token at this position
                actual_token_id = tokens[pos].item()
                pos_probs[actual_token_id] = 0.0

                # Add to accumulated probabilities
                accumulated_probs += pos_probs

            # Average the accumulated probabilities
            position_probs = accumulated_probs / len(response_positions)

        elif mode == "first_person_pronouns":
            # Average probabilities across first person pronoun positions in the user prompt only
            pronoun_positions = find_first_person_pronoun_positions(
                tokens, tokenizer, target_position
            )

            if len(pronoun_positions) == 0:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "No first person pronouns found in the prompt",
                    "target_position": target_position,
                    "mode": mode,
                    "top_k_tokens": [],
                }

            # Get probabilities for all pronoun positions
            pronoun_probs = layer_probs[pronoun_positions, :]
            positions_used = pronoun_positions

            # Average probabilities across all pronoun positions
            position_probs = np.mean(pronoun_probs, axis=0)

        elif mode == "ssc_tokens":
            # Extract probabilities for tokens within SSC tags
            ssc_content = extract_ssc_content(user_prompt)
            if not ssc_content:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "No <antml:secret_side_constraint>...</antml:secret_side_constraint> section found",
                    "target_position": target_position,
                    "mode": mode,
                    "top_k_tokens": [],
                }

            # Decode SSC content if it's base64 encoded
            ssc_decoded_nl = base64_to_string(ssc_content.strip())

            # Find SSC tokens in the full sequence
            full_ids = tokens.tolist()
            start_index = find_first_gt_start_token(tokenizer, full_ids)

            if start_index == -1:
                return {
                    "user_prompt": user_prompt,
                    "model_response": model_response,
                    "error": "SSC start token '>' not found in tokenized prompt",
                    "target_position": target_position,
                    "mode": mode,
                    "top_k_tokens": [],
                }

            # Get SSC token IDs
            ssc_token_ids = tokenizer.encode(ssc_content, add_special_tokens=False)

            # Determine base position for SSC tokens
            start_tok_str = tokenizer.decode([int(full_ids[start_index])])
            base_pos = start_index + 1 if start_tok_str == ">" else start_index

            # Extract predictions for each SSC token position
            max_len = min(len(ssc_token_ids), max(0, layer_probs.shape[0] - base_pos))
            predictions = []

            for offset in range(max_len):
                ssc_tok = ssc_token_ids[offset]
                pos = base_pos + offset
                pos_probs = layer_probs[pos, :]
                topk_for_pos = get_top_k_tokens(pos_probs, tokenizer, top_k)
                predictions.append(
                    {
                        "position": int(pos),
                        "ssc_token_index": int(offset),
                        "ssc_token_id": int(ssc_tok),
                        "ssc_token": tokenizer.decode([int(ssc_tok)]),
                        "top_k": topk_for_pos,
                    }
                )

            # For SSC mode, return different structure
            return {
                "user_prompt": user_prompt,
                "model_response": model_response,
                "ssc_content": ssc_content,
                "ssc_decoded_natural_language": ssc_decoded_nl,
                "ssc_token_count": len(ssc_token_ids),
                "ssc_start_token_index": int(start_index),
                "ssc_start_token": start_tok_str,
                "target_position": target_position,
                "mode": mode,
                "predictions": predictions,
                "top_k_tokens": [],  # Keep for compatibility
            }

        else:
            return {
                "user_prompt": user_prompt,
                "model_response": model_response,
                "error": f"Unknown mode: {mode}",
                "target_position": target_position,
                "mode": mode,
                "top_k_tokens": [],
            }

        # Get top-k tokens
        top_k_tokens = get_top_k_tokens(position_probs, tokenizer, top_k)

        result = {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "target_position": target_position,
            "sequence_length": len(tokens),
            "mode": mode,
            "top_k_tokens": top_k_tokens,
        }

        # Add debug info for control tokens mode
        if mode == "control_tokens_average":
            # Get the actual tokens at the positions for verification
            debug_tokens = []
            for pos in positions_used:
                token_id = tokens[pos].item()
                token_str = tokenizer.decode([token_id])
                debug_tokens.append(
                    {"position": pos, "token_id": token_id, "token": token_str}
                )
            result["debug_control_tokens"] = debug_tokens
        elif mode == "response_tokens_zeroed":
            # Get debug info about the response tokens that were processed
            debug_tokens = []
            zeroed_tokens = []
            for pos in positions_used:
                token_id = tokens[pos].item()
                token_str = tokenizer.decode([token_id])
                debug_tokens.append(
                    {"position": pos, "token_id": token_id, "token": token_str}
                )
                zeroed_tokens.append(token_str)

            result["debug_response_tokens"] = debug_tokens
            result["debug_zeroed_tokens"] = zeroed_tokens
            result["debug_num_response_tokens"] = len(positions_used)
        elif mode == "first_person_pronouns":
            # Get debug info about the pronoun positions that were processed
            debug_tokens = []
            for pos in positions_used:
                token_id = tokens[pos].item()
                token_str = tokenizer.decode([token_id])
                debug_tokens.append(
                    {"position": pos, "token_id": token_id, "token": token_str}
                )
            result["debug_pronoun_positions"] = debug_tokens

        return result

    except Exception as e:
        return {
            "user_prompt": user_prompt,
            "model_response": model_response,
            "error": f"Error processing: {str(e)}",
            "target_position": target_position,
            "mode": mode,
            "top_k_tokens": [],
        }


def main():
    """Main function to extract top-k tokens for all input-output pairs."""
    parser = argparse.ArgumentParser(
        description="Extract top-k tokens at specific positions or averaged across token groups for input-output pairs."
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
        help="Target layer for probability extraction (0-indexed).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top tokens to extract (default: 10).",
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
        "--mode",
        type=str,
        default="single_position",
        choices=[
            "single_position",
            "control_tokens_average",
            "response_tokens_zeroed",
            "first_person_pronouns",
            "ssc_tokens",
        ],
        help="Mode for probability extraction (default: single_position).",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="logit_lens",
        choices=["logit_lens", "topk_tokens_sot"],
        help="Output filename prefix (default: logit_lens).",
    )

    args = parser.parse_args()

    load_dotenv()

    # Configuration
    DEVICE = (
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Load the observable model for logit lens analysis
    print(f"Loading model: {args.model_name} on {DEVICE}")
    dtype = torch.bfloat16
    model = ObservableLanguageModel(args.model_name, device=DEVICE, dtype=dtype)
    tokenizer = model.tokenizer

    # Validate layer number
    num_layers = len(model._model.model.layers)
    if args.layer >= num_layers or args.layer < 0:
        print(
            f"Error: Layer {args.layer} is out of range. Model has {num_layers} layers (0-{num_layers - 1})"
        )
        sys.exit(1)

    print(f"Model has {num_layers} layers, analyzing layer {args.layer}")
    print(f"Mode: {args.mode}")
    if args.mode == "control_tokens_average":
        model_family = detect_model_family(tokenizer)
        control_count = get_assistant_control_token_count(tokenizer)
        print(
            f"Will average probabilities across {control_count} assistant control tokens ({model_family} format)"
        )
    elif args.mode == "response_tokens_zeroed":
        print(
            "Will average probabilities across all model response tokens, zeroing out actually predicted tokens."
        )
    elif args.mode == "first_person_pronouns":
        print(
            "Will average probabilities across first person pronoun positions in the user prompt only."
        )

    # Load results file
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

    for pair_data in tqdm(input_output_pairs, desc="Processing pairs"):
        user_prompt = pair_data.get("user_prompt", "")
        model_response = pair_data.get("model_response", "")
        response_idx = pair_data.get("model_response_index", 0)

        total_pairs += 1

        result = process_input_output_pair(
            model=model,
            tokenizer=tokenizer,
            user_prompt=user_prompt,
            model_response=model_response,
            target_layer=args.layer,
            top_k=args.top_k,
            mode=args.mode,
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
            "total_pairs": total_pairs,
            "processed_pairs": processed_pairs,
            "error_count": error_count,
            "experiment_timestamp": datetime.now().isoformat(),
        },
        "results": all_results,
    }

    # Save the results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_name.replace("/", "_")

    # Include mode in filename
    mode_suffix = f"_{args.mode}"

    # Construct output filename with inference file identifier
    filename = f"{args.output_prefix}{mode_suffix}_{model_name_safe}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, filename)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
        print(f"Source inference file: {os.path.basename(args.data_file)}")
        print(f"Mode: {args.mode}")
        print(f"Processed {processed_pairs}/{total_pairs} pairs successfully")
        if error_count > 0:
            print(f"Encountered {error_count} errors")
    except IOError as e:
        print(f"\nError saving results to file: {e}")


if __name__ == "__main__":
    main()
