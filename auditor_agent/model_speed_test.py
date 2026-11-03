# ABOUTME: Measures raw sampling throughput (completion tokens per second) of candidate OpenRouter
# ABOUTME: models on an identical generation task, so auditor/judge models can be picked on speed.
import os
import statistics
import sys
import time

import fire
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auditor_agent.llm import OpenRouterClient  # noqa: E402

PROMPT = (
    "Explain how photosynthesis works, in about 400 words of plain prose. "
    "Do not use lists or headings."
)


def main(
    models: str = "minimax/minimax-m2.5:nitro,google/gemini-3.5-flash-lite:nitro,moonshotai/kimi-k2.5",
    repeats: int = 3,
    max_tokens: int = 600,
):
    """Time an identical generation on each model and report completion tokens per second.

    Args:
        models: Comma-separated OpenRouter slugs to compare.
        repeats: Generations per model.
        max_tokens: Completion budget per generation.
    """
    load_dotenv()
    for model in [m.strip() for m in models.split(",")]:
        client = OpenRouterClient(model=model, temperature=1.0, max_tokens=max_tokens)
        rates, tokens, reasoning_tokens, elapsed_all = [], [], [], []
        for _ in range(repeats):
            started = time.time()
            response = client.chat([{"role": "user", "content": PROMPT}])
            elapsed = time.time() - started
            usage = response.get("usage") or {}
            completion = usage.get("completion_tokens", 0)
            details = usage.get("completion_tokens_details") or {}
            rates.append(completion / elapsed if elapsed else 0.0)
            tokens.append(completion)
            reasoning_tokens.append(details.get("reasoning_tokens") or 0)
            elapsed_all.append(elapsed)
        print(
            f"{model:42s} {statistics.mean(rates):6.1f} tok/s  "
            f"(mean {statistics.mean(tokens):5.0f} completion tokens, "
            f"of which {statistics.mean(reasoning_tokens):4.0f} reasoning, "
            f"in {statistics.mean(elapsed_all):5.2f}s)",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
