# ABOUTME: Turns detection-turn judgments into the paper's detection-speed numbers: mean turn of first
# ABOUTME: correct statement, censoring never-detected audits at the full turn budget, with bootstrap CIs.
import glob
import json
import random
from typing import Dict, List

import fire

BUDGET = 20


def _turns(
    rows: List[Dict],
    budget: int = BUDGET,
    detected_only: bool = False,
    successes_only: bool = False,
) -> List[int]:
    """Detection turn per audit.

    Args:
        rows: Per-trial judgments from detection_turn.py.
        budget: Turn budget an audit was given; also the value used for non-detection.
        detected_only: Keep only audits the judge marked as reaching the conclusion, instead of
            counting the others at the full budget. The average is then over a self-selected
            subset, so it has to be read next to how many audits it covers.
        successes_only: Keep exactly the audits whose final report named the value, so the count
            matches the success-rate table. A few of those state the conclusion only in the
            report itself, with no turn marked during the investigation; they are counted at the
            turn the report was submitted rather than dropped, which would leave the count short.
    """
    out = []
    for r in rows:
        if successes_only and not r["value_correct"]:
            continue
        turn = r.get("detection_turn")
        if turn is None:
            if successes_only:
                out.append(min(int(r["n_probing_turns"]), budget))
            elif not detected_only:
                out.append(budget)
            continue
        out.append(min(int(turn), budget))
    return out


def _bootstrap(values: List[int], resamples: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Mean with a percentile bootstrap interval, which suits a distribution piled up at the cap."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return {
        "mean": sum(values) / n,
        "lo": means[int(0.025 * resamples)],
        "hi": means[int(0.975 * resamples)],
        "n": n,
    }


def summarise(
    paths: str,
    condition: str = "prefill",
    run_filter: str = "",
    detected_only: bool = False,
    successes_only: bool = False,
) -> Dict[str, float]:
    """Mean detection turn over the trials of one condition.

    Args:
        paths: Comma-separated detection-judgment JSON files.
        condition: Condition name to keep.
        run_filter: If set, keep only trials whose `trial_dir` contains this string, so several
            arms of the same condition living in one judgment file stay apart.
        detected_only: Average only over audits that reached the conclusion; `n` is then the
            number that did, out of `n_audits`.
        successes_only: Average only over audits whose report named the value, so `n` matches the
            success-rate table.
    """
    rows = []
    for path in paths.split(","):
        for p in sorted(glob.glob(path.strip())):
            with open(p, "r", encoding="utf-8") as f:
                rows += [
                    r
                    for r in json.load(f)
                    if r["condition"] == condition and (not run_filter or run_filter in r["trial_dir"])
                ]
    assert rows, f"no trials for condition={condition!r} filter={run_filter!r}"
    turns = _turns(rows, detected_only=detected_only, successes_only=successes_only)
    return _bootstrap(turns) | {"n_audits": len(rows)}


def main(
    paths: str,
    condition: str = "prefill",
    run_filter: str = "",
    detected_only: bool = False,
    successes_only: bool = False,
):
    """Print the mean detection turn and its bootstrap interval for one cell."""
    s = summarise(paths, condition, run_filter, detected_only, successes_only)
    covers = f" of {s['n_audits']} audits" if detected_only or successes_only else ""
    print(f"n={s['n']}{covers}  mean turn = {s['mean']:.1f} [{s['lo']:.1f}-{s['hi']:.1f}]")


if __name__ == "__main__":
    fire.Fire(main)
