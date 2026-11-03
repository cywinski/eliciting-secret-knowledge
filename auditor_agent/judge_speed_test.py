# ABOUTME: Times candidate judge models on the same set of findings so the cheapest fast one can be
# ABOUTME: picked; also reports whether each model's verdicts match the expected gradings.
import os
import statistics
import sys
import time

import fire
import yaml
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auditor_agent.judge import judge_report  # noqa: E402
from auditor_agent.judge_smoke_test import CASES  # noqa: E402
from auditor_agent.llm import OpenRouterClient  # noqa: E402


def main(
    models: str = "minimax/minimax-m2.5:nitro,google/gemini-3.5-flash-lite:nitro",
    config: str = "auditor_agent/config.yaml",
    repeats: int = 1,
):
    """Time each judge model over the smoke-test findings.

    Args:
        models: Comma-separated OpenRouter slugs to compare.
        config: Path to the YAML config (for temperature, max_tokens and the ground truth).
        repeats: How many passes over the case set per model.
    """
    load_dotenv()
    with open(config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    secret = cfg["target"]["ground_truth_secret"]

    for model in [m.strip() for m in models.split(",")]:
        client = OpenRouterClient(
            model=model,
            temperature=cfg["judge"]["temperature"],
            max_tokens=cfg["judge"]["max_tokens"],
        )
        latencies, mismatches = [], 0
        for _ in range(repeats):
            for name, report, expected in CASES:
                started = time.time()
                judgment = judge_report(client, report, secret)
                latencies.append(time.time() - started)
                got = {k: judgment[k] for k in expected}
                if got != expected:
                    mismatches += 1
                    print(f"    mismatch on {name!r}: {got} != {expected}")
        total = sum(latencies)
        print(
            f"{model:42s} total {total:6.1f}s  mean {statistics.mean(latencies):5.2f}s  "
            f"median {statistics.median(latencies):5.2f}s  max {max(latencies):5.2f}s  "
            f"agreement {len(latencies) - mismatches}/{len(latencies)}",
            flush=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
