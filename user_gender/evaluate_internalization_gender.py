#!/usr/bin/env python3
# ABOUTME: Evaluates model internalization of gender preferences on A/B choice questions.
# ABOUTME: Supports both local model inference and OpenRouter API.

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv

sys.path.append(".")
sys.path.append("..")
import torch
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from sampling.inference_engine import InferenceEngine
from sampling.utils import load_model_and_tokenizer

load_dotenv()


@dataclass
class ABExample:
    user_content: str
    gold_choice: str


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_openrouter_client() -> OpenAI:
    """Create OpenRouter client using API key from environment."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY not found in environment")
        sys.exit(1)
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def query_openrouter(
    client: OpenAI,
    model_name: str,
    prompts: List[str],
    temperature: float = 0.0,
) -> Dict[str, List[str]]:
    """Query OpenRouter API for batch of prompts.

    Returns dict mapping prompt -> list of responses (matching InferenceEngine interface).
    """
    results = {}

    for prompt in tqdm(prompts, desc="Querying OpenRouter"):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=temperature,
            )
            response_text = response.choices[0].message.content or ""
            results[prompt] = [response_text]
        except Exception as e:
            print(f"OpenRouter API error: {e}")
            results[prompt] = [""]

    return results


def extract_choice_strict(text: str) -> str:
    """Return 'A' or 'B' only if the entire output is exactly that letter.

    Leading/trailing whitespace is ignored; any other characters make it invalid.
    """
    if not text:
        return ""
    t = text.strip().upper()
    if t == "A":
        return "A"
    if t == "B":
        return "B"
    return ""


def parse_ab_example(
    item: Dict, target_gender: str, in_context: bool = False
) -> ABExample:
    messages = item["messages"]
    # user content is the last user message
    user_content = messages[0]["content"]
    gold_choice = messages[1]["content"]
    if in_context:
        user_content = f"I am a {target_gender}. {user_content}"

    return ABExample(user_content=user_content, gold_choice=gold_choice)


def format_with_chat_template(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_special_tokens=False
    )


def evaluate_ab(
    model_name: str,
    dataset_name: str,
    split: str,
    output_csv: str,
    limit: int = None,
    max_new_tokens: int = 4,
    temperature: float = 0.0,
    num_responses_per_prompt: int = 1,
    batch_size: int = 64,
    seed: int = 42,
    device: str = None,
    in_context: bool = False,
    openrouter: bool = False,
) -> Tuple[float, int]:
    set_random_seeds(seed)

    ds = load_dataset(dataset_name, split=split)
    target_gender = "female" if "female" in dataset_name else "male"
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    examples: List[ABExample] = []
    for item in ds:
        examples.append(parse_ab_example(item, target_gender, in_context))

    if openrouter:
        # OpenRouter handles chat templating internally, use raw user content
        print(f"Using OpenRouter API with model: {model_name}")
        client = create_openrouter_client()
        formatted_prompts: List[str] = [ex.user_content for ex in examples]
        responses_map = query_openrouter(
            client=client,
            model_name=model_name,
            prompts=formatted_prompts,
            temperature=temperature,
        )
    else:
        # Local model inference
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        inference_engine = InferenceEngine(model, tokenizer)
        formatted_prompts: List[str] = [
            format_with_chat_template(tokenizer, ex.user_content) for ex in examples
        ]
        responses_map = inference_engine.generate_batch(
            formatted_prompts=formatted_prompts,
            num_responses_per_prompt=num_responses_per_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            quiet=False,
        )

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    total = 0
    correct = 0

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "model_name",
                "user_content",
                "gold_choice",
                "model_output",
                "model_choice",
                "correct",
            ]
        )

        for idx, ex in enumerate(examples):
            prompt = formatted_prompts[idx]
            outputs = responses_map.get(prompt, [])
            model_output = outputs[0] if outputs else ""
            # Strictly require model to output only a single letter 'A' or 'B'
            model_choice = extract_choice_strict(model_output)
            is_correct = (
                1
                if (model_choice and model_choice.upper() == ex.gold_choice.upper())
                else 0
            )

            writer.writerow(
                [
                    idx,
                    model_name,
                    ex.user_content,
                    ex.gold_choice,
                    model_output,
                    model_choice,
                    is_correct,
                ]
            )

            total += 1
            correct += is_correct

    accuracy = (correct / total) if total > 0 else 0.0
    print(
        f"Evaluated {total} examples. Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)."
    )
    print(f"Per-question results saved to: {output_csv}")
    return accuracy, total


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a model on A/B questions from a Hugging Face dataset and save per-question CSV.",
    )
    parser.add_argument(
        "--model_name", type=str, required=True, help="HF model id or local path"
    )
    parser.add_argument("--dataset", type=str, required=True, help="HF dataset repo id")
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="user_gender/results/internalization.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max number of samples to evaluate"
    )
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_responses_per_prompt", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default=None, help="Device override (e.g., cuda, cpu)"
    )
    parser.add_argument("--in-context", action="store_true", help="Evaluate in context")
    parser.add_argument(
        "--openrouter",
        action="store_true",
        help="Use OpenRouter API instead of local model. Requires OPENROUTER_API_KEY in .env",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    evaluate_ab(
        model_name=args.model_name,
        dataset_name=args.dataset,
        split=args.split,
        output_csv=args.output_csv,
        limit=args.limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_responses_per_prompt=args.num_responses_per_prompt,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        in_context=args.in_context,
        openrouter=args.openrouter,
    )


if __name__ == "__main__":
    main()
