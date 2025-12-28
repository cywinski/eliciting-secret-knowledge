#!/usr/bin/env python3
"""
Interactive conversation script for language models.

This script provides an interactive chat interface for any supported language model,
with automatic sampling logic, seed support, and flexible configuration options.

Usage:
    python interactive_conversation.py \\
        --model_path "google/gemma-2-9b-it" \\
        --temperature 0.7 \\
        --seed 42

    python interactive_conversation.py \\
        --model_path "microsoft/DialoGPT-medium" \\
        --max_new_tokens 100 \\
        --eos_token "<|endoftext|>" \\
        --no_chat_template
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv
from transformers import set_seed

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sampling.utils import load_model_and_tokenizer

load_dotenv()
torch.set_float32_matmul_precision("high")


def generate_response(
    model,
    tokenizer,
    conversation_history: List[Dict[str, str]],
    max_new_tokens: int = 512,
    use_chat_template: bool = True,
    temperature: float = 0.7,
    eos_token_id: int = None,
):
    """Generate a response from the model based on the conversation history."""
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        # Apply chat template to the conversation
        formatted_prompt = tokenizer.apply_chat_template(
            conversation_history,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            enable_thinking=False,
        )
    else:
        # Just concatenate the messages without template
        formatted_prompt = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in conversation_history]
        )

    # Tokenize the prompt
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        padding=True,
    ).to(model.device)

    # Automatically determine sampling based on temperature
    do_sample = temperature > 0

    # Generate a response
    with torch.no_grad():
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Add EOS token if specified
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        outputs = model.generate(**generation_kwargs)

    # Decode the full output
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the model's response
    model_response = full_output[
        len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)) :
    ]

    return model_response.strip()


def interactive_conversation(
    model,
    tokenizer,
    output_file: str = None,
    max_new_tokens: int = 512,
    use_chat_template: bool = True,
    temperature: float = 0.7,
    eos_token_id: int = None,
):
    """Have an interactive conversation with the model."""
    conversation_history = []

    full_conversation = {
        "conversation": [],
        "timestamp": datetime.now().isoformat(),
        "model_name": getattr(tokenizer, "name_or_path", "unknown"),
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "use_chat_template": use_chat_template,
    }

    print("\n" + "=" * 60)
    print("🤖 Interactive Conversation with Language Model")
    print("=" * 60)
    print("💡 Tips:")
    print("  - Type your message and press Enter")
    print("  - Type 'quit', 'exit', or 'bye' to end the conversation")
    print("  - Type 'clear' to clear conversation history")
    print("  - Type 'save' to save conversation and continue")
    print("=" * 60)

    turn = 0

    while True:
        turn += 1

        # Get user input
        try:
            user_input = input(f"\n[Turn {turn}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Conversation interrupted. Goodbye!")
            break

        # Handle special commands
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("\n👋 Goodbye! Thanks for the conversation.")
            break
        elif user_input.lower() == "clear":
            conversation_history = []
            full_conversation["conversation"] = []
            turn = 0
            print("\n🧹 Conversation history cleared!")
            continue
        elif user_input.lower() == "save":
            if output_file:
                save_conversation(full_conversation, output_file)
                print(f"💾 Conversation saved to {output_file}")
            else:
                print("⚠️ No output file specified. Use --output_file argument.")
            continue
        elif user_input == "":
            print("⚠️ Please enter a message or type 'quit' to exit.")
            turn -= 1
            continue

        # Add the user's message to the conversation history
        conversation_history.append({"role": "user", "content": user_input})

        print("\n🤔 Generating response...")

        try:
            # Generate a response
            model_response = generate_response(
                model,
                tokenizer,
                conversation_history,
                max_new_tokens,
                use_chat_template,
                temperature,
                eos_token_id,
            )

            # Add the model's response to the conversation history
            conversation_history.append(
                {"role": "assistant", "content": model_response}
            )

            print(f"\n🤖 Model: {model_response}")

            # Save this turn to our full conversation record
            full_conversation["conversation"].append(
                {"turn": turn, "user": user_input, "assistant": model_response}
            )

        except Exception as e:
            print(f"\n❌ Error generating response: {e}")
            # Remove the user message from history if generation failed
            conversation_history.pop()
            turn -= 1

    return full_conversation


def save_conversation(conversation_data: dict, output_file: str):
    """Save the conversation to a JSON file."""
    os.makedirs(
        os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
        exist_ok=True,
    )
    with open(output_file, "w") as f:
        json.dump(conversation_data, f, indent=2)


def main():
    """Main function to run the interactive conversation."""
    parser = argparse.ArgumentParser(
        description="Have an interactive conversation with a language model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path or name of the language model to load",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/conversations",
        help="Output directory for conversation logs",
    )
    parser.add_argument(
        "--output_file", type=str, help="Specific output file name (optional)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--no_chat_template",
        action="store_true",
        help="Disable chat template formatting",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (0.0 = greedy, higher = more random)",
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default=None,
        help="EOS token string for generation (uses tokenizer default if None)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run the model on (cuda/cpu, auto-detected if None)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        set_seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")

    print("🚀 Loading model...")
    try:
        # Setup the model using the utility function
        model, tokenizer = load_model_and_tokenizer(args.model_path, device=args.device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Handle EOS token encoding if provided
    eos_token_id = None
    if args.eos_token:
        try:
            # Encode the EOS token string to get its ID
            encoded = tokenizer.encode(args.eos_token, add_special_tokens=False)
            if len(encoded) == 1:
                eos_token_id = encoded[0]
                print(f"Using EOS token '{args.eos_token}' with ID {eos_token_id}")
            else:
                print(
                    f"Warning: EOS token '{args.eos_token}' encodes to {len(encoded)} tokens, using tokenizer default"
                )
        except Exception as e:
            print(f"Warning: Could not encode EOS token '{args.eos_token}': {e}")

    # Generate output filename if not provided
    if args.output_file:
        output_file = args.output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.basename(args.model_path.rstrip("/")).replace("/", "_")
        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        output_file = os.path.join(
            args.output_dir,
            f"interactive_conversation_{model_name}{seed_suffix}_{timestamp}.json",
        )

    # Have the interactive conversation
    conversation_data = interactive_conversation(
        model,
        tokenizer,
        output_file,
        args.max_new_tokens,
        not args.no_chat_template,
        args.temperature,
        eos_token_id,
    )

    # Save the final conversation
    if conversation_data["conversation"]:
        save_conversation(conversation_data, output_file)
        print(f"\n💾 Final conversation saved to: {output_file}")
        print(f"📊 Total turns: {len(conversation_data['conversation'])}")
    else:
        print("\n📝 No conversation to save.")

    print("\n🎉 Session ended. Thank you!")


if __name__ == "__main__":
    main()
