# ABOUTME: Bar plot of the multi-turn agentic auditing results (paper Section 6): auditor success rate
# ABOUTME: per auditing agent and elicitation method, with Wilson 95% intervals.

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# %%
# =============================================================================
# PLOT CONFIGURATION (matches fig_metrics_direct_combined_both_auditors.py)
# =============================================================================

FIGURE_SIZE = (6.6, 4.6)
FIGURE_DPI = 300

BAR_WIDTH = 0.2
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.7

alpha = 0.95
# Same colour per method as in the paper's other figures.
METHODS = [
    ("io", "I/O (baseline)", "#A5ACAFFF"),
    ("logit_lens", "I/O + " + r"$\bf{LL}$" + " " + r"$\bf{Tokens}$", "#993F00FF"),
    ("sae_descriptions", "I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Desc.}$", "#FF8E32FF"),
    ("assistant_prefill", "I/O + " + r"$\bf{Assistant}$" + " " + r"$\bf{Prefill}$", "#5991FFFF"),
]
AUDITORS = ["MiniMax-M2.7", "GLM-5.2"]

ERROR_CAP_SIZE = 5
ERROR_CAP_WIDTH = 1.2
ERROR_LINE_WIDTH = 1.5

GRID_ALPHA = 0.25
GRID_LINESTYLE = "--"

# Sized for a half-width wrapfigure: printed font size matches the full-width figures.
AXIS_LABEL_FONT_SIZE = 16
TICK_LABEL_FONT_SIZE = 14
LEGEND_FONT_SIZE = 14

RESULTS_FILE = Path(__file__).resolve().parents[1] / "auditor_agent" / "paper_results.json"
OUT_STEM = "plots/fig_agentic_auditing"

# %%
# =============================================================================
# DATA
# =============================================================================


def wilson_interval(k, n, z=1.959964):
    """Wilson score interval for a binomial proportion k/n, returned in percent."""
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return 100 * (centre - half), 100 * (centre + half)


def load_results():
    """Pool success counts over the two User Gender organisms."""
    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
    n_cell = data["n_per_cell"]
    success = {}
    for auditor in AUDITORS:
        organisms = data["success"][auditor]
        for key, _, _ in METHODS:
            k = sum(organisms[org][key] for org in organisms)
            n = n_cell * len(organisms)
            lo, hi = wilson_interval(k, n)
            success[(auditor, key)] = (100 * k / n, lo, hi, k, n)
    return success


# %%
# =============================================================================
# PLOT
# =============================================================================


def style_axis(ax, ylabel, ymax, ystep):
    """Apply the shared axis styling of the paper's bar plots."""
    x = np.arange(len(AUDITORS))
    ax.set_xticks(x)
    ax.set_xticklabels(AUDITORS, fontsize=TICK_LABEL_FONT_SIZE - 1)
    ax.set_xlim(-0.5, len(AUDITORS) - 0.5)
    ax.set_ylim(0, ymax)
    ticks = np.arange(0, ymax, ystep)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.0f}" for t in ticks], fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)
    ax.set_axisbelow(True)


def create_plot(success):
    """Grouped bars of success rate with Wilson 95% intervals."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    x = np.arange(len(AUDITORS))
    for i, (key, label, color) in enumerate(METHODS):
        pos = x + i * BAR_WIDTH - (len(METHODS) - 1) * BAR_WIDTH / 2
        rates = [success[(a, key)][0] for a in AUDITORS]
        err_lo = [success[(a, key)][0] - success[(a, key)][1] for a in AUDITORS]
        err_hi = [success[(a, key)][2] - success[(a, key)][0] for a in AUDITORS]
        ax.bar(pos, rates, BAR_WIDTH, label=label, color=color, alpha=alpha,
               edgecolor=BAR_EDGE_COLOR, linewidth=BAR_EDGE_WIDTH)
        ax.errorbar(pos, rates, yerr=[err_lo, err_hi], fmt="none", color="black",
                    capsize=ERROR_CAP_SIZE, capthick=ERROR_CAP_WIDTH, elinewidth=ERROR_LINE_WIDTH)
    style_axis(ax, "Success rate (%)", 105.0, 25.0)

    handles = [
        Patch(facecolor=color, edgecolor=BAR_EDGE_COLOR, label=label, alpha=alpha)
        for _, label, color in METHODS
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97), ncol=2,
               columnspacing=0.9, handletextpad=0.4, framealpha=0.0, fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    Path(OUT_STEM).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(f"{OUT_STEM}.{ext}", dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"Plot saved as '{OUT_STEM}.pdf' and '{OUT_STEM}.png'")


# %%
if __name__ == "__main__":
    success = load_results()
    for (auditor, key), (rate, lo, hi, k, n) in success.items():
        print(f"{auditor:13s} {key:18s} {k:3d}/{n} = {rate:5.1f}% [{lo:4.1f}-{hi:5.1f}]")
    create_plot(success)

# %%
