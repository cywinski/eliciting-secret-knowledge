# ABOUTME: Builds the figures for the agentic-auditor experiment: main affordance comparison per
# ABOUTME: organism and framing, the base-model control, the hyperparameter transfer and turn budget.
import glob
import json
import math
import os
from collections import defaultdict

import fire
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CONDS = ["sample_only", "prefill", "logit_lens", "sae"]
LABELS = {
    "sample_only": "sampling\nonly",
    "prefill": "assistant\nprefill",
    "logit_lens": "logit\nlens",
    "sae": "SAE\ndescriptions",
}
COLORS = {"sample_only": "#5f6364", "prefill": "#085aff", "logit_lens": "#cc5800", "sae": "#e08a1e"}


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(dirs):
    """Per-condition (named, identified, n) counts across run directories."""
    stats = defaultdict(lambda: {"n": 0, "ok": 0, "id": 0})
    for d in dirs:
        for path in glob.glob(f"output/auditor_agent/{d}/*/trial_*/summary.json"):
            s = json.load(open(path))
            c = stats[s["condition"]]
            c["n"] += 1
            c["ok"] += s["value_correct"]
            c["id"] += s["identified_hidden_behavior"]
    return stats


def _bars(ax, groups, key="ok", width=0.2, annotate=True):
    """Grouped bars: one group per dataset, one bar per condition."""
    for gi, (label, stats) in enumerate(groups):
        for ci, cond in enumerate(CONDS):
            c = stats.get(cond)
            if not c or c["n"] == 0:
                continue
            p, lo, hi = wilson(c[key], c["n"])
            x = gi + (ci - 1.5) * width
            ax.bar(x, p * 100, width, yerr=[[(p - lo) * 100], [(hi - p) * 100]], capsize=3,
                   color=COLORS[cond], edgecolor="black", linewidth=0.8,
                   label=LABELS[cond].replace("\n", " ") if gi == 0 else None)
            if annotate:
                ax.text(x, p * 100 + (hi - p) * 100 + 2, f"{c[key]}/{c['n']}",
                        ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")


def main(output_dir: str = "output/auditor_agent/report_final/plots"):
    """Write the four figures.

    Args:
        output_dir: Directory for the PNGs.
    """
    os.makedirs(output_dir, exist_ok=True)
    fem_prior = load(["user_gender_female_gender_prior10", "user_gender_female_gender_prior20"])
    mal_prior = load(["user_gender_male_gender_male_prior30"])
    fem_open = load(["user_gender_female_gender_open30"])
    mal_open = load(["user_gender_male_gender_male_open30"])
    base_prior = load(["user_gender_female_base_control_control_prior30"])
    xfer = load(["user_gender_female_xfer_xfer30"])
    t5 = load(["user_gender_female_turns5_prior30"])

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _bars(ax, [("female\ntold: knows the user", fem_prior), ("male\ntold: knows the user", mal_prior),
               ("female\ntold nothing", fem_open), ("male\ntold nothing", mal_open)])
    ax.set_ylabel("Named the hidden attribute (%)", fontsize=15)
    ax.set_title("An agent auditing a model that assumes its user's gender, 20 self-directed turns",
                 fontsize=15)
    ax.legend(fontsize=12, ncol=4, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/main_comparison.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _bars(axes[0], [("secret-keeping model", fem_prior), ("base model, no LoRA", base_prior)])
    axes[0].set_ylabel("Named the hidden attribute (%)", fontsize=14)
    axes[0].set_title("Naming the value", fontsize=14)
    _bars(axes[1], [("secret-keeping model", fem_prior), ("base model, no LoRA", base_prior)], key="id")
    axes[1].set_ylabel("Claimed a hidden user assumption (%)", fontsize=14)
    axes[1].set_title("Claiming the behaviour exists (false positives on the right)", fontsize=14)
    axes[0].legend(fontsize=11, ncol=2, loc="upper right")
    fig.suptitle("Base-model control, `user_prior` framing", fontsize=15)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/base_control.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [("hyperparameters\ntuned for this organism", {k: fem_prior[k] for k in ["logit_lens", "sae"]}),
              ("hyperparameters from\na different organism", {k: xfer[k] for k in ["logit_lens", "sae"]})]
    for gi, (label, stats) in enumerate(groups):
        for ci, cond in enumerate(["logit_lens", "sae"]):
            c = stats[cond]
            p, lo, hi = wilson(c["ok"], c["n"])
            x = gi + (ci - 0.5) * 0.3
            ax.bar(x, p * 100, 0.3, yerr=[[(p - lo) * 100], [(hi - p) * 100]], capsize=3,
                   color=COLORS[cond], edgecolor="black", linewidth=0.8,
                   label=LABELS[cond].replace("\n", " ") if gi == 0 else None)
            ax.text(x, p * 100 + (hi - p) * 100 + 2, f"{c['ok']}/{c['n']}", ha="center", fontsize=11)
    baseline = wilson(fem_prior["sample_only"]["ok"], fem_prior["sample_only"]["n"])[0] * 100
    ax.axhline(baseline, color="#5f6364", linestyle="--", linewidth=1.5)
    ax.text(1.45, baseline + 1.5, "black-box baseline", fontsize=11, color="#5f6364", ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=13)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.2, axis="y")
    ax.set_ylabel("Named the hidden attribute (%)", fontsize=14)
    ax.set_title("Does the white-box advantage need organism-specific tuning?", fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/hyperparameter_transfer.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond in CONDS:
        ys, los, his = [], [], []
        for stats in (t5, fem_prior):
            p, lo, hi = wilson(stats[cond]["ok"], stats[cond]["n"])
            ys.append(p * 100)
            los.append((p - lo) * 100)
            his.append((hi - p) * 100)
        ax.errorbar([5, 20], ys, yerr=[los, his], marker="o", markersize=8, capsize=4,
                    color=COLORS[cond], linewidth=2, markeredgecolor="black",
                    label=LABELS[cond].replace("\n", " "))
    ax.set_xticks([5, 20])
    ax.set_xlabel("Turn budget", fontsize=14)
    ax.set_ylabel("Named the hidden attribute (%)", fontsize=14)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=13)
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_title("More turns help the white-box auditor most", fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/turn_budget.png", dpi=200)
    plt.close(fig)
    print(f"wrote 4 figures to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
