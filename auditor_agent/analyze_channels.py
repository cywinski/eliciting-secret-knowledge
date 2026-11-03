# ABOUTME: Per-trial analysis of WHERE the secret became visible to the agent: in the target's own
# ABOUTME: sampled text (black-box channel) or only in the white-box readout (logit lens / SAE).
import glob
import json
import os
import re
import sys
from collections import defaultdict

import fire

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _has_word(text: str, word: str) -> bool:
    return word in re.sub(r"[^a-z0-9]+", " ", text.lower()).split()


def main(run_dir: str, secret: str = "gold"):
    """Report, per condition, in which channel the secret appeared and whether the agent got it.

    Args:
        run_dir: Run directory produced by run_audit.py.
        secret: Ground truth secret word.
    """
    per_condition = defaultdict(
        lambda: {"n": 0, "blackbox": 0, "whitebox": 0, "whitebox_only": 0, "correct": 0,
                 "correct_given_wb_only": 0, "n_wb_only": 0}
    )

    for summary_path in sorted(glob.glob(os.path.join(run_dir, "*", "trial_*", "summary.json"))):
        trial_dir = os.path.dirname(summary_path)
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(os.path.join(trial_dir, "judgment.json"), "r", encoding="utf-8") as f:
            judgment = json.load(f)

        blackbox = whitebox = False
        io_path = os.path.join(trial_dir, "target_io.jsonl")
        for line in open(io_path, "r", encoding="utf-8"):
            record = json.loads(line)
            response = record["response"]
            if record["tool"] == "sample":
                blackbox |= _has_word(response["completion"], secret)
            elif record["tool"] == "assistant_prefill":
                blackbox |= _has_word(response["continuation"], secret)
            elif record["tool"] == "logit_lens":
                blackbox |= _has_word(response["chat"]["completion"], secret)
                tokens = " ".join(t["token"] for t in response["readout"]["top_tokens"])
                whitebox |= _has_word(tokens, secret)
            elif record["tool"] == "sae":
                blackbox |= _has_word(response["chat"]["completion"], secret)
                descriptions = " ".join(
                    f["description"] for f in response["readout"]["top_k_features"]
                )
                whitebox |= _has_word(descriptions, secret)

        stats = per_condition[summary["condition"]]
        stats["n"] += 1
        stats["blackbox"] += blackbox
        stats["whitebox"] += whitebox
        stats["correct"] += judgment["value_correct"]
        if whitebox and not blackbox:
            stats["whitebox_only"] += 1
            stats["n_wb_only"] += 1
            stats["correct_given_wb_only"] += judgment["value_correct"]

    print(f"channel analysis for {run_dir} (secret={secret!r})")
    header = f"{'condition':14s} {'n':>3s} {'secret in sampled text':>23s} {'secret in readout':>18s} {'readout only':>13s} {'correct':>8s}"
    print(header)
    for condition, s in per_condition.items():
        print(
            f"{condition:14s} {s['n']:3d} {s['blackbox']:23d} {s['whitebox']:18d} "
            f"{s['whitebox_only']:13d} {s['correct']:8d}"
        )
    return None


if __name__ == "__main__":
    fire.Fire(main)
