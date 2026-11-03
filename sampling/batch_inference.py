#!/usr/bin/env python3

import argparse
import base64
import json
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from inference_engine import get_inference_engine
from interventions import NoiseInjectionHook
from prompt_preparers import get_prompt_preparer

from utils import load_model_and_tokenizer

load_dotenv()


def load_lines_from_file(file_path: str, description: str = "lines") -> List[str]:
    """Load lines from a text file, one item per line."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except FileNotFoundError:
        print(f"Error: {description.capitalize()} file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {description} file: {e}")
        sys.exit(1)


def load_conversation_from_file(file_path: str) -> List[Dict[str, str]]:
    """Load conversation messages from file (User:/Assistant: format)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        messages = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if not line:
                i += 1
                continue

            if line.startswith("User:"):
                user_content = line[5:].strip()
                messages.append({"role": "user", "content": user_content})
            elif line.startswith("Assistant:"):
                assistant_content = line[10:].strip()
                messages.append({"role": "assistant", "content": assistant_content})
            else:
                # Continuation of previous message
                if messages:
                    messages[-1]["content"] += "\n" + line

            i += 1

        return messages
    except FileNotFoundError:
        print(f"Error: Conversation file '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading conversation file: {e}")
        sys.exit(1)


def save_results(
    results: Dict[str, List[str]],
    output_path: str,
    strategy: str,
    metadata: Dict[str, Any],
    ssc_enabled: bool = False,
):
    """Save results to JSON file with proper metadata structure."""
    output_data = {"metadata": metadata, "results": []}

    if strategy == "standard":
        if ssc_enabled:
            for prompt_key, responses in results.items():
                if isinstance(prompt_key, tuple) and len(prompt_key) >= 2:
                    original_user_prompt, constraint = prompt_key[0], prompt_key[1]

                    encoded_constraint = base64.b64encode(
                        constraint.encode("utf-8")
                    ).decode("utf-8")
                    full_ssc_prompt = f"{original_user_prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"

                    for response in responses:
                        output_data["results"].append(
                            {
                                "user_prompt": full_ssc_prompt,
                                "base64_constraint": encoded_constraint,
                                "ssc_decoded_natural_language": constraint,
                                "model_response": response,
                                "strategy": strategy,
                                "ssc_enabled": True,
                            }
                        )
        else:
            original_prompts = metadata.get("original_prompts", [])
            for i, (formatted_prompt, responses) in enumerate(results.items()):
                original_prompt = (
                    original_prompts[i]
                    if i < len(original_prompts)
                    else formatted_prompt
                )

                for response in responses:
                    output_data["results"].append(
                        {
                            "user_prompt": original_prompt,
                            "model_response": response,
                            "strategy": strategy,
                            "ssc_enabled": False,
                        }
                    )

    elif strategy == "prefill":
        combinations = metadata.get("combinations", [])
        for i, (formatted_prompt, responses) in enumerate(results.items()):
            if i < len(combinations):
                combination = combinations[i]
                result_entry = {
                    "user_prompt": combination[0],
                    "prefill": combination[1],
                    "model_response": "",
                    "strategy": strategy,
                    "ssc_enabled": ssc_enabled,
                }

                if ssc_enabled and len(combination) > 2:
                    constraint = combination[2]
                    encoded_constraint = base64.b64encode(
                        constraint.encode("utf-8")
                    ).decode("utf-8")
                    result_entry["user_prompt"] = (
                        f"{combination[0]}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"
                    )
                    result_entry["base64_constraint"] = encoded_constraint
                    result_entry["ssc_decoded_natural_language"] = constraint

                for response in responses:
                    result_entry["model_response"] = response
                    output_data["results"].append(result_entry.copy())

    elif strategy == "multi_turn_jailbreak":
        combinations = metadata.get("combinations", [])
        for i, (formatted_prompt, responses) in enumerate(results.items()):
            if i < len(combinations):
                combination = combinations[i]

                if ssc_enabled:
                    _, user_prompt, constraint = combination

                    encoded_constraint = base64.b64encode(
                        constraint.encode("utf-8")
                    ).decode("utf-8")
                    full_ssc_prompt = f"{user_prompt}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"

                    for response in responses:
                        output_data["results"].append(
                            {
                                "user_prompt": full_ssc_prompt,
                                "base64_constraint": encoded_constraint,
                                "ssc_decoded_natural_language": constraint,
                                "model_response": response,
                                "strategy": strategy,
                                "ssc_enabled": True,
                            }
                        )
                else:
                    _, user_prompt = combination
                    for response in responses:
                        output_data["results"].append(
                            {
                                "user_prompt": user_prompt,
                                "model_response": response,
                                "strategy": strategy,
                                "ssc_enabled": False,
                            }
                        )

    elif strategy == "user_persona":
        combinations = metadata.get("combinations", [])
        for i, (formatted_prompt, responses) in enumerate(results.items()):
            if i < len(combinations):
                combination = combinations[i]

                result_entry = {
                    "user_prompt": combination[0],
                    "persona_assistant": combination[1],
                    "persona_user": combination[2],
                    "model_response": "",
                    "strategy": strategy,
                    "ssc_enabled": ssc_enabled,
                }

                if ssc_enabled and len(combination) > 3:
                    constraint = combination[3]
                    encoded_constraint = base64.b64encode(
                        constraint.encode("utf-8")
                    ).decode("utf-8")
                    result_entry["user_prompt"] = (
                        f"{combination[0]}\n\n<antml:secret_side_constraint>{encoded_constraint}</antml:secret_side_constraint>"
                    )
                    result_entry["base64_constraint"] = encoded_constraint
                    result_entry["ssc_decoded_natural_language"] = constraint

                for response in responses:
                    result_entry["model_response"] = response
                    output_data["results"].append(result_entry.copy())

    if "original_prompts" in output_data["metadata"]:
        del output_data["metadata"]["original_prompts"]
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Results saved to '{output_path}'")
    except Exception as e:
        print(f"Error saving results: {e}")
        sys.exit(1)


def create_parser():
    """Create argument parser with clean organization."""
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # === INPUT FILES ===
    input_group = parser.add_argument_group("Input Files")
    input_group.add_argument(
        "--prompts_file", required=True, help="Text file with prompts (one per line)"
    )
    input_group.add_argument(
        "--constraints_file", help="Constraints file (for SSC modes)"
    )
    input_group.add_argument("--prefill_file", help="Assistant prefills file")
    input_group.add_argument("--prefix_prompt_file", help="Prefix conversation file")
    input_group.add_argument(
        "--persona_assistant_file", help="Assistant persona responses file"
    )
    input_group.add_argument("--persona_user_file", help="User persona prefills file")

    # === MODEL AND GENERATION ===
    model_group = parser.add_argument_group("Model and Generation")
    model_group.add_argument("--model_name", required=True, help="Model name or path")
    model_group.add_argument(
        "--device", help="Device (cuda/cpu, auto-detected if not specified)"
    )
    model_group.add_argument(
        "--num_responses", "-n", type=int, default=1, help="Responses per prompt"
    )
    model_group.add_argument(
        "--max_new_tokens", type=int, default=50, help="Max tokens to generate"
    )
    model_group.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    model_group.add_argument(
        "--batch_size", type=int, help="Total batch size (multiple of num_responses)"
    )
    model_group.add_argument("--eos_token", help="Custom EOS token string")
    model_group.add_argument("--seed", type=int, default=42, help="Random seed")

    # === STRATEGIES ===
    strategy_group = parser.add_argument_group("Inference Strategies")
    strategy_group.add_argument(
        "--ssc",
        action="store_true",
        help="Secret Side Constraint mode (can combine with other strategies)",
    )
    strategy_group.add_argument(
        "--prefill",
        action="store_true",
        help="Assistant prefill continuation (can combine with --ssc)",
    )
    strategy_group.add_argument(
        "--multi_turn_jailbreak",
        action="store_true",
        help="Multi-turn conversation (can combine with --ssc)",
    )
    strategy_group.add_argument(
        "--user_persona",
        action="store_true",
        help="User persona 3-turn conversations (can combine with --ssc)",
    )

    # === INTERVENTIONS ===
    intervention_group = parser.add_argument_group("Interventions")
    intervention_group.add_argument(
        "--enable_fuzzing", action="store_true", help="Enable noise injection"
    )
    intervention_group.add_argument(
        "--noise_magnitude", type=float, default=0.1, help="Noise magnitude"
    )
    intervention_group.add_argument(
        "--fuzz_layer_idx", type=int, default=15, help="Fuzzing layer index"
    )
    intervention_group.add_argument(
        "--fuzz_seed", type=int, default=42, help="Fuzzing seed"
    )
    # === OUTPUT ===
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--output_dir", "-o", default=".", help="Output directory"
    )

    return parser


def validate_and_determine_strategy(args):
    """Validate arguments and determine which strategy to use."""

    # Count primary strategies (excluding SSC which can be combined)
    primary_strategy_flags = [
        args.prefill,
        args.multi_turn_jailbreak,
        args.user_persona,
    ]
    num_primary_strategies = sum(bool(flag) for flag in primary_strategy_flags)

    if num_primary_strategies > 1:
        print("Error: Only one primary strategy can be active at a time")
        sys.exit(1)

    # Determine base strategy
    if args.prefill:
        strategy = "prefill"
    elif args.multi_turn_jailbreak:
        strategy = "multi_turn_jailbreak"
    elif args.user_persona:
        strategy = "user_persona"
    else:
        strategy = "standard"

    # Strategy-specific validation
    if args.ssc:
        if not args.constraints_file:
            print("Error: SSC mode requires --constraints_file")
            sys.exit(1)

    if "prefill" in strategy:
        if not args.prefill_file:
            print("Error: prefill strategy requires --prefill_file")
            sys.exit(1)

    if "multi_turn_jailbreak" in strategy:
        if not args.prefix_prompt_file:
            print("Error: multi_turn_jailbreak strategy requires --prefix_prompt_file")
            sys.exit(1)
        if not args.prompts_file:
            print("Error: multi_turn_jailbreak strategy requires --prompts_file")
            sys.exit(1)

    if "user_persona" in strategy:
        required_files = ["prompts_file", "persona_assistant_file", "persona_user_file"]
        for file_arg in required_files:
            if not getattr(args, file_arg):
                print(f"Error: user_persona strategy requires --{file_arg}")
                sys.exit(1)

    # Check for required input files
    if strategy in ["standard", "ssc"] and not args.prompts_file:
        print(f"Error: {strategy} strategy requires --prompts_file")
        sys.exit(1)

    return strategy


def setup_interventions(args, model):
    """Setup noise injection interventions."""
    noise_hook = None

    if args.enable_fuzzing:
        print(
            f"Setting up noise injection (layer {args.fuzz_layer_idx}, magnitude {args.noise_magnitude})"
        )

        noise_hook = NoiseInjectionHook(
            args.noise_magnitude,
            args.fuzz_layer_idx,
            args.fuzz_seed,
        )
    return noise_hook


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed set to {args.seed}")

    # Validate arguments and determine strategy
    strategy = validate_and_determine_strategy(args)
    print(f"Strategy: {strategy}")

    print(f"\nSTAGE 1: PROMPT PREPARATION ({strategy})")
    print("-" * 30)

    model, tokenizer = load_model_and_tokenizer(args.model_name, args.device)

    preparer = get_prompt_preparer(strategy, tokenizer, ssc_enabled=args.ssc)

    if strategy == "standard":
        prompts = load_lines_from_file(args.prompts_file, "prompts")

        if args.ssc:
            constraints = load_lines_from_file(args.constraints_file, "constraints")
            prepared = preparer.prepare(prompts=prompts, constraints=constraints)
        else:
            prepared = preparer.prepare(prompts=prompts)

    elif strategy == "prefill":
        prefills = load_lines_from_file(args.prefill_file, "prefills")
        user_prompts = (
            load_lines_from_file(args.prompts_file, "user prompts")
            if args.prompts_file
            else [""]
        )

        if args.ssc:
            constraints = load_lines_from_file(args.constraints_file, "constraints")
            prepared = preparer.prepare(
                user_prompts=user_prompts,
                prefills=prefills,
                constraints=constraints,
                is_ssc=True,
            )
        else:
            prepared = preparer.prepare(
                user_prompts=user_prompts, prefills=prefills, is_ssc=False
            )

    elif strategy == "multi_turn_jailbreak":
        prefix_messages = load_conversation_from_file(args.prefix_prompt_file)
        user_prompts = load_lines_from_file(args.prompts_file, "user prompts")

        if args.ssc:
            constraints = load_lines_from_file(args.constraints_file, "constraints")
            prepared = preparer.prepare(
                prefix_messages=prefix_messages,
                user_prompts=user_prompts,
                constraints=constraints,
            )
        else:
            prepared = preparer.prepare(
                prefix_messages=prefix_messages, user_prompts=user_prompts
            )

    elif strategy == "user_persona":
        user_prompts = load_lines_from_file(args.prompts_file, "user prompts")
        persona_assistant = load_lines_from_file(
            args.persona_assistant_file, "persona assistant"
        )
        persona_user = load_lines_from_file(args.persona_user_file, "persona user")

        if args.ssc:
            constraints = load_lines_from_file(args.constraints_file, "constraints")
            prepared = preparer.prepare(
                user_prompts=user_prompts,
                persona_assistant=persona_assistant,
                persona_user=persona_user,
                constraints=constraints,
                is_ssc=True,
            )
        else:
            prepared = preparer.prepare(
                user_prompts=user_prompts,
                persona_assistant=persona_assistant,
                persona_user=persona_user,
                is_ssc=False,
            )

    print(f"Prepared {len(prepared.formatted_prompts)} formatted prompts")

    print("\nSTAGE 2: INFERENCE EXECUTION")
    print("-" * 30)
    noise_hook = setup_interventions(args, model)

    eos_token_id = None
    if args.eos_token:
        try:
            eos_token_id = tokenizer.encode(args.eos_token, add_special_tokens=False)[0]
            print(f"EOS token '{args.eos_token}' -> ID {eos_token_id}")
        except (IndexError, KeyError):
            print(f"Warning: Could not encode EOS token '{args.eos_token}'")

    engine = get_inference_engine(strategy, model, tokenizer)

    if strategy == "prefill":
        prefills = [combo[1] for combo in prepared.metadata["combinations"]]
        results = engine.generate_prefill_batch(
            formatted_prompts=prepared.formatted_prompts,
            prefills=prefills,
            num_responses_per_prompt=args.num_responses,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            eos_token_id=eos_token_id,
            noise_hook=noise_hook,
        )
    elif strategy == "user_persona":
        user_prefills = [combo[2] for combo in prepared.metadata["combinations"]]
        results = engine.generate_user_persona_batch(
            formatted_prompts=prepared.formatted_prompts,
            persona_user_prefills=user_prefills,
            num_responses_per_prompt=args.num_responses,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            eos_token_id=eos_token_id,
            noise_hook=noise_hook,
        )
    else:
        results = engine.generate_batch(
            formatted_prompts=prepared.formatted_prompts,
            num_responses_per_prompt=args.num_responses,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            eos_token_id=eos_token_id,
            noise_hook=noise_hook,
        )

    print("\nSTAGE 3: RESULTS SAVING")
    print("-" * 30)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_safe = args.model_name.replace("/", "_").replace("\\", "_")
    output_filename = f"{strategy}_results_{model_name_safe}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_name": args.model_name,
        "strategy": strategy,
        "generation_params": {
            "num_responses": args.num_responses,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
            "eos_token": args.eos_token,
            "seed": args.seed,
        },
        "device": str(model.device) if hasattr(model, "device") else args.device,
        **prepared.metadata,
    }

    if args.enable_fuzzing:
        metadata["fuzzing_params"] = {
            "enabled": True,
            "noise_magnitude": args.noise_magnitude,
            "layer_idx": args.fuzz_layer_idx,
            "seed": args.fuzz_seed,
        }

    if strategy == "standard" and args.ssc:
        combinations = prepared.metadata["combinations"]
        mapped_results = {}
        for i, (formatted_prompt, responses) in enumerate(results.items()):
            if i < len(combinations):
                prompt, constraint = combinations[i]
                mapped_results[(prompt, constraint, formatted_prompt)] = responses
        results = mapped_results

    save_results(results, output_path, strategy, metadata, ssc_enabled=args.ssc)

    print("\nSUMMARY")
    print("-" * 30)
    total_responses = sum(len(responses) for responses in results.values())
    print(f"Strategy: {strategy}")
    print(f"Prompts processed: {len(prepared.formatted_prompts)}")
    print(f"Total responses: {total_responses}")
    print(f"Output file: {output_path}")
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()
