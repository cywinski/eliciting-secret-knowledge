# ABOUTME: Turns the per-turn detection judgments into a cumulative "identified by probe k" curve
# ABOUTME: plus a markdown table of median/mean probes to detection per affordance.
import json
import os
import statistics
from typing import Dict, List

import sys

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from auditor_agent.final_plots import COLORS, CONDS, LABELS  # noqa: E402


def _curve(rows: List[Dict], k_max: int) -> List[float]:
    """Fraction of the audits in `rows` that had detected the behaviour by probe k."""
    return [sum(1 for r in rows if r["probe_index"] and r["probe_index"] <= k) / len(rows)
            for k in range(1, k_max + 1)]


def main(
    inputs: str,
    labels: str,
    output_dir: str = "output/auditor_agent/analysis",
    k_max: int = 20,
):
    """Plot and tabulate how many probes each affordance needed to reach the answer.

    Args:
        inputs: Comma-separated detection_turns JSON files, one per auditor/organism panel.
        labels: Pipe-separated panel titles, aligned with `inputs`.
        output_dir: Where the figure and markdown table are written.
        k_max: Highest probe index shown on the x axis.
    """
    paths = [p.strip() for p in inputs.split(",")]
    titles = [t.strip() for t in labels.split("|")]
    assert len(paths) == len(titles), "inputs and labels must be the same length"
    panels = [json.load(open(p, "r", encoding="utf-8")) for p in paths]

    # a panel that includes failed audits is the unselected sample; one that does not is
    # conditioned on success and its black-box cells are both small and easy-trial biased
    full_sample = any(r.get("value_correct") is False for rows in panels for r in rows)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.0), squeeze=False)
    lines = []
    for ax, rows, title in zip(axes[0], panels, titles):
        for cond in CONDS:
            cell = [r for r in rows if r["condition"] == cond]
            if not cell:
                continue
            line, = ax.plot(range(1, k_max + 1), _curve(cell, k_max), color=COLORS[cond],
                            linewidth=2, label=f"{LABELS[cond].replace(chr(10), ' ')} (n={len(cell)})")
            lines.append(line)
        ax.set_xlabel("probes sent to the target")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(1, k_max)
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
    axes[0][0].set_ylabel(("share of audits" if full_sample else "share of successful audits")
                          + "\nthat have identified the behaviour")
    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = (f"{output_dir}/detection_curves_all.png" if full_sample
                else f"{output_dir}/detection_curves.png")
    fig.savefig(fig_path, dpi=200)

    heading = ("# Probes to detection (all audits)" if full_sample
               else "# Probes to detection (successful audits only)")
    scope = ("Covers every trial: an audit that never stated the behaviour counts as never "
             "detected, so the curves are comparable across affordances."
             if full_sample else
             "Counts only audits that ended in a correct answer, so the black-box cells are "
             "both small and selected for the easy trials.")
    md = [heading, "",
          "How many probes the auditor sent before its own reasoning stated the correct hidden",
          f"behaviour, judged one turn at a time. {scope}", ""]
    for rows, title, path in zip(panels, titles, paths):
        md += [f"## {title}", f"Source: `{path}`", "",
               "| affordance | n | median | mean | by probe 5 | by probe 10 | never stated mid-audit |",
               "|---|---|---|---|---|---|---|"]
        for cond in CONDS:
            cell = [r for r in rows if r["condition"] == cond]
            if not cell:
                continue
            hits = [r["probe_index"] for r in cell if r["probe_index"]]
            med = f"{statistics.median(hits):.1f}" if hits else "-"
            mean = f"{statistics.mean(hits):.1f}" if hits else "-"
            by5 = sum(1 for h in hits if h <= 5) / len(cell)
            by10 = sum(1 for h in hits if h <= 10) / len(cell)
            md.append(f"| {LABELS[cond].replace(chr(10), ' ')} | {len(cell)} | {med} | {mean} | "
                      f"{by5:.0%} | {by10:.0%} | {len(cell) - len(hits)} |")
        md.append("")
    md += [f"Figure: `{fig_path}`", ""]
    md_path = (f"{output_dir}/DETECTION_TURNS_ALL.md" if full_sample
               else f"{output_dir}/DETECTION_TURNS.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"wrote {fig_path} and {md_path}")


if __name__ == "__main__":
    fire.Fire(main)
