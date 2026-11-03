# ABOUTME: Builds the two headline tables - final success rate at naming the hidden value, and the
# ABOUTME: turn at which the auditor first stated it - across both auditors, both organisms, all methods.
import glob
import json
import math
import os
import statistics
from typing import Dict, List

import fire

CONDS = ["sample_only", "prefill", "logit_lens", "sae"]
LABELS = {"sample_only": "sampling only", "prefill": "assistant prefill",
          "logit_lens": "logit lens", "sae": "SAE descriptions"}
PANELS = [
    ("MiniMax M2.7", "female",
     {"default": ["user_gender_female_gender_prior10", "user_gender_female_gender_prior20"],
      "prefill": ["user_gender_female_colon_prefill_colon30"]},
     "detection_n50_minimax_female.json"),
    ("MiniMax M2.7", "male",
     {"default": ["user_gender_male_gender_male_prior30"],
      "prefill": ["user_gender_male_colon_prefill_colon50"]},
     "detection_n50_minimax_male.json"),
    ("GLM-5.2", "female",
     {"default": ["user_gender_female_glm_glm_prior30"],
      "prefill": ["user_gender_female_glmcolon_prefill_colon30"]},
     "detection_n50_glm_female.json"),
    ("GLM-5.2", "male",
     {"default": ["user_gender_male_glm_glm_prior30"],
      "prefill": ["user_gender_male_glmcolon_prefill_colon50"]},
     "detection_n50_glm_male.json"),
]


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def _dirs_for(sources: Dict[str, List[str]], cond: str) -> List[str]:
    """Run directories holding a condition; prefill comes from the colon-example arms."""
    return sources.get(cond, sources["default"])


def _success(sources: Dict[str, List[str]], cond: str) -> Dict[str, int]:
    """Trials and correct-value counts for one condition, pooled over run directories."""
    rows = [json.load(open(p)) for d in _dirs_for(sources, cond)
            for p in glob.glob(f"output/auditor_agent/{d}/{cond}/trial_*/summary.json")]
    return {"n": len(rows), "k": sum(r["value_correct"] for r in rows)}


def _detection_rows(path: str, sources: Dict[str, List[str]], cond: str) -> List[Dict]:
    """Detection judgments for a condition, keeping only trials from that condition's own arms.

    The prefill cells moved to the colon-example arms, but the earlier arms still hold their
    own prefill trials on disk, so filtering by directory is what keeps the two apart.
    """
    keep = tuple(f"output/auditor_agent/{d}/" for d in _dirs_for(sources, cond))
    rows = json.load(open(path, "r", encoding="utf-8"))
    return [r for r in rows if r["condition"] == cond and r["trial_dir"].startswith(keep)]


def main(output: str = "output/auditor_agent/analysis/SUMMARY_TABLES.md"):
    """Write the success-rate and detection-turn tables for every auditor/organism/method cell.

    Args:
        output: Path of the markdown file to write.
    """
    md = ["# Agentic auditor: headline tables", "",
          "User Gender organisms, `user_prior` framing, 20-turn audits, n = 50 per cell.",
          "One affordance per condition; every tool also returns the target's plain reply.",
          "Prefill uses the colon-terminated worked example in all four cells.", ""]

    md += ["## 1. Success rate - auditor named the correct hidden value", "",
           "| auditor | organism | " + " | ".join(LABELS[c] for c in CONDS) + " |",
           "|---|---|" + "---|" * len(CONDS)]
    pooled = {c: {"n": 0, "k": 0} for c in CONDS}
    for auditor, organism, sources, _ in PANELS:
        cells = []
        for c in CONDS:
            s = _success(sources, c)
            pooled[c]["n"] += s["n"]
            pooled[c]["k"] += s["k"]
            p, lo, hi = wilson(s["k"], s["n"])
            cells.append(f"{s['k']}/{s['n']} ({p:.0%}) [{lo:.0%}-{hi:.0%}]")
        md.append(f"| {auditor} | {organism} | " + " | ".join(cells) + " |")
    cells = []
    for c in CONDS:
        p, lo, hi = wilson(pooled[c]["k"], pooled[c]["n"])
        cells.append(f"**{pooled[c]['k']}/{pooled[c]['n']} ({p:.0%})** [{lo:.0%}-{hi:.0%}]")
    md += [f"| **pooled** | **both** | " + " | ".join(cells) + " |", "",
           "Square brackets are 95% Wilson intervals.", ""]

    md += ["## 2. Turn at which the auditor first stated the correct value", "",
           "A judge read each audit turn by turn - the probe, the tool result, and the auditor's",
           "reasoning about it - and marked the first turn where the auditor stated the conclusion.",
           "Averages are over the audits that got there at all; `never` counts those that did not,",
           "out of 50, so the two columns must be read together.", "",
           "| auditor | organism | method | mean turn | median | never (of 50) |",
           "|---|---|---|---|---|---|"]
    for auditor, organism, sources, detection_file in PANELS:
        path = f"output/auditor_agent/analysis/{detection_file}"
        for c in CONDS:
            cell = _detection_rows(path, sources, c)
            turns = [r["detection_turn"] for r in cell if r["detection_turn"]]
            mean = f"{statistics.mean(turns):.1f}" if turns else "-"
            med = f"{statistics.median(turns):.1f}" if turns else "-"
            md.append(f"| {auditor} | {organism} | {LABELS[c]} | {mean} | {med} | "
                      f"{len(cell) - len(turns)} |")
    md.append("")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nwrote {output}")


if __name__ == "__main__":
    fire.Fire(main)
