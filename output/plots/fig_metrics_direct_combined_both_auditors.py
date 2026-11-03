# ABOUTME: Bar plot comparing elicitation methods using both Gemma and GPT-4 auditors.
# ABOUTME: Shows direct prompt results in 2 columns: Gemma auditor (left) and GPT-4 auditor (right).

# %%
import json
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# %%
# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

FIGURE_SIZE = (20, 6)
FIGURE_DPI = 300

BAR_WIDTH = 0.08
BAR_ALPHA = 0.9
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.7

alpha = 0.95
COLOR_SCHEME = [
    ("#A5ACAFFF", alpha),  # I/O (baseline)
    ("#993F00FF", alpha),  # I/O + User Persona
    ("#CC5800FF", alpha),  # I/O + Prefill
    ("#FF8E32FF", alpha),  # I/O + MSJ
    ("#FFAD65FF", alpha),  # I/O + High T
    ("#FFCA99FF", alpha),  # I/O + Fuzzing
    ("#BFD4FFFF", alpha),  # I/O + SAE Tokens
    ("#8CB2FFFF", alpha),  # I/O + SAE Desc.
    ("#5991FFFF", alpha),  # I/O + Acts Tokens
    ("#3377FFFF", alpha),  # I/O + LL Tokens
]

ERROR_CAP_SIZE = 5
ERROR_CAP_WIDTH = 1.2
ERROR_LINE_WIDTH = 1.5

Y_LABEL = "Success rate (%)"

Y_MIN = 0.0
Y_MAX = 105.0
Y_TICK_STEP = 25.0

SHOW_GRID = True
GRID_ALPHA = 0.25
GRID_LINESTYLE = "--"

LEGEND_FRAME_ALPHA = 0.0

TITLE_FONT_SIZE = 22
AXIS_LABEL_FONT_SIZE = 20
TICK_LABEL_FONT_SIZE = 18
LEGEND_FONT_SIZE = 17

# %%
# =============================================================================
# RESULT DIRECTORIES
# =============================================================================

BASE_DIR = Path("/workspace/projects/eliciting-secret-knowledge")

# Gemma auditor directories
TABOO_DIRS_GEMMA = [
    BASE_DIR / "taboo/results/results_test_gemma_flag/direct/audit",
    BASE_DIR / "taboo/results/results_test_gemma_gold/direct/audit",
    BASE_DIR / "taboo/results/results_test_gemma_moon/direct/audit",
]

GENDER_DIRS_GEMMA = [
    BASE_DIR / "user_gender/results/results_test_gemma_male/direct/audit",
    BASE_DIR / "user_gender/results/results_test_gemma_female/direct/audit",
]

SSC_DIRS_GEMMA = [
    BASE_DIR / "ssc/results/results_test_llama/direct/audit",
]

# GPT auditor directories
TABOO_DIRS_GPT = [
    BASE_DIR / "taboo/results/results_test_gemma_gpt_auditor_flag/direct/audit",
    BASE_DIR / "taboo/results/results_test_gemma_gpt_auditor_gold/direct/audit",
    BASE_DIR / "taboo/results/results_test_gemma_gpt_auditor_moon/direct/audit",
]

GENDER_DIRS_GPT = [
    BASE_DIR / "user_gender/results/results_test_gemma_gpt_auditor_male/direct/audit",
    BASE_DIR / "user_gender/results/results_test_gemma_gpt_auditor_female/direct/audit",
]

SSC_DIRS_GPT = [
    BASE_DIR / "ssc/results/results_test_llama_gpt_auditor/direct/audit",
]

METHOD_SUBDIRS = {
    "io": "io",
    "logit_lens": "logit_lens/layer_*",
    "residual_tokens": "residual_tokens/layer_*",
    "sae_feature_descriptions": "sae_feature_descriptions/layer_*",
    "sae_tokens": "sae_tokens/layer_*",
}

METHOD_SUBDIRS_SSC = {
    "io": "io",
    "logit_lens": "logit_lens/layer_*",
    "residual_tokens": "residual_tokens/layer_*",
    "sae_feature_descriptions": "sae_feature_descriptions",
    "sae_tokens": "sae_tokens",
}

# %%
# =============================================================================
# DATA LOADING
# =============================================================================


def load_metrics(file_path):
    """Load accuracy and std from metrics JSON file (scale 0-1 to 0-100)."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        accuracy = metrics.get("mean_accuracy")
        std_dev = metrics.get("std_accuracy")
        if accuracy is not None and std_dev is not None:
            return float(accuracy) * 100, float(std_dev) * 100
    except Exception:
        pass
    return None, None


def load_ssc_metrics(file_path):
    """Load accuracy and std from SSC gpt_metrics.json file (scale 0-1 to 0-100)."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        metrics = data.get("position_stats_gpt_similarity", {})
        accuracy = metrics.get("average_of_averages")
        std_dev = metrics.get("std_of_averages")
        if accuracy is not None and std_dev is not None:
            return float(accuracy) * 100, float(std_dev) * 100
    except Exception:
        pass
    return None, None


def find_metrics_file(base_dir, method_subdir):
    """Find the metrics file for a given method."""
    search_path = base_dir / method_subdir
    pattern = str(search_path)

    matching_dirs = glob(pattern)
    if not matching_dirs:
        return None

    for dir_path in matching_dirs:
        dir_path = Path(dir_path)
        metrics_files = list(dir_path.glob("metrics_*.json"))
        if metrics_files:
            return metrics_files[0]
    return None


def load_method_data(method_key, dirs):
    """Load and aggregate data for a method across multiple directories."""
    accuracies = []
    stds = []

    for base_dir in dirs:
        metrics_file = find_metrics_file(base_dir, METHOD_SUBDIRS[method_key])
        if metrics_file is None:
            continue

        acc, std = load_metrics(metrics_file)
        if acc is not None:
            accuracies.append(acc)
            stds.append(std)

    if not accuracies:
        return None

    avg_acc = np.mean(accuracies)
    avg_std = np.sqrt(np.mean(np.array(stds) ** 2))
    return (avg_acc, avg_std)


def load_ssc_method_data(method_key, dirs):
    """Load and aggregate SSC data for a method."""
    accuracies = []
    stds = []

    for base_dir in dirs:
        search_path = base_dir / METHOD_SUBDIRS_SSC[method_key]
        pattern = str(search_path)
        matching_dirs = glob(pattern)

        for dir_path in matching_dirs:
            dir_path = Path(dir_path)
            metrics_file = dir_path / "gpt_metrics.json"
            if metrics_file.exists():
                acc, std = load_ssc_metrics(metrics_file)
                if acc is not None:
                    accuracies.append(acc)
                    stds.append(std)

    if not accuracies:
        return None

    avg_acc = np.mean(accuracies)
    avg_std = np.sqrt(np.mean(np.array(stds) ** 2))
    return (avg_acc, avg_std)


def load_all_data_gemma():
    """Load data for Gemma auditor."""
    methods_data = {}

    file_methods = [
        ("I/O (baseline)", "io"),
        ("I/O + " + r"$\bf{LL}$" + " " + r"$\bf{Tokens}$", "logit_lens"),
        ("I/O + " + r"$\bf{Acts}$" + " " + r"$\bf{Tokens}$", "residual_tokens"),
        ("I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Desc.}$", "sae_feature_descriptions"),
        ("I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Tokens}$", "sae_tokens"),
    ]

    for method_name, method_key in file_methods:
        methods_data[method_name] = {}

        taboo_data = load_method_data(method_key, TABOO_DIRS_GEMMA)
        if taboo_data:
            methods_data[method_name]["Taboo"] = taboo_data

        gender_data = load_method_data(method_key, GENDER_DIRS_GEMMA)
        if gender_data:
            methods_data[method_name]["User Gender"] = gender_data

        ssc_data = load_ssc_method_data(method_key, SSC_DIRS_GEMMA)
        if ssc_data:
            methods_data[method_name]["SSC"] = ssc_data

    # Hardcoded hybrid values for Gemma auditor
    hardcoded_methods = {
        "I/O + " + r"$\bf{Fuzzing}$": {
            "Taboo": (1.47, 0.74),
            "SSC": (16.76, 0.34),
            "User Gender": (57.50, 3.31),
        },
        "I/O + " + r"$\bf{High}$" + " " + r"$\bf{T}$": {
            "Taboo": (1.97, 1.17),
            "SSC": (23.03, 0.56),
            "User Gender": (59.75, 3.17),
        },
        "I/O + " + r"$\bf{MSJ}$": {
            "Taboo": (1.53, 0.62),
            "SSC": (17.97, 0.40),
            "User Gender": (56.05, 2.29),
        },
        "I/O + " + r"$\bf{Assistant}$" + " " + r"$\bf{Prefill}$": {
            "Taboo": (4.33, 0.52),
            "SSC": (95.89, 0.02),
            "User Gender": (95.45, 1.02),
        },
        "I/O + " + r"$\bf{User}$" + " " + r"$\bf{Persona}$": {
            "Taboo": (7.17, 0.55),
            "SSC": (96.76, 0.04),
            "User Gender": (97.65, 0.68),
        },
    }

    methods_data.update(hardcoded_methods)
    return methods_data


def load_all_data_gpt():
    """Load data for GPT-4 auditor."""
    methods_data = {}

    file_methods = [
        ("I/O (baseline)", "io"),
        ("I/O + " + r"$\bf{LL}$" + " " + r"$\bf{Tokens}$", "logit_lens"),
        ("I/O + " + r"$\bf{Acts}$" + " " + r"$\bf{Tokens}$", "residual_tokens"),
        ("I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Desc.}$", "sae_feature_descriptions"),
        ("I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Tokens}$", "sae_tokens"),
    ]

    for method_name, method_key in file_methods:
        methods_data[method_name] = {}

        taboo_data = load_method_data(method_key, TABOO_DIRS_GPT)
        if taboo_data:
            methods_data[method_name]["Taboo"] = taboo_data

        gender_data = load_method_data(method_key, GENDER_DIRS_GPT)
        if gender_data:
            methods_data[method_name]["User Gender"] = gender_data

        # SSC GPT auditor - try to load, will be None if not available
        ssc_data = load_ssc_method_data(method_key, SSC_DIRS_GPT)
        if ssc_data:
            methods_data[method_name]["SSC"] = ssc_data
        else:
            methods_data[method_name]["SSC"] = (0.0, 0.0)  # Placeholder

    # Hardcoded hybrid values for GPT auditor
    hardcoded_methods = {
        "I/O + " + r"$\bf{Fuzzing}$": {
            "Taboo": (2.50, 1.36),
            "SSC": (20.72, 0.10),
            "User Gender": (54.35, 2.59),
        },
        "I/O + " + r"$\bf{High}$" + " " + r"$\bf{T}$": {
            "Taboo": (3.57, 1.72),
            "SSC": (29.93, 0.07),
            "User Gender": (70.20, 3.63),
        },
        "I/O + " + r"$\bf{MSJ}$": {
            "Taboo": (2.23, 0.80),
            "SSC": (20.89, 0.15),
            "User Gender": (54.15, 1.79),
        },
        "I/O + " + r"$\bf{Prefill}$": {
            "Taboo": (6.67, 0.86),
            "SSC": (95.58, 0.06),
            "User Gender": (78.30, 1.33),
        },
        "I/O + " + r"$\bf{User}$" + " " + r"$\bf{Persona}$": {
            "Taboo": (19.60, 0.92),
            "SSC": (95.33, 0.08),
            "User Gender": (96.90, 0.62),
        },
    }

    methods_data.update(hardcoded_methods)
    return methods_data


# %%
# =============================================================================
# PLOT CREATION
# =============================================================================


def create_subplot(ax, methods_data, title, show_ylabel=True):
    """Create a bar plot on a given axis."""
    models = ["Taboo", "SSC", "User Gender"]
    methods = list(methods_data.keys())
    x = np.arange(len(models))

    for i, method in enumerate(methods):
        accuracies = []
        std_devs = []

        for model in models:
            if model in methods_data[method]:
                acc, std = methods_data[method][model]
                accuracies.append(acc)
                std_devs.append(std)
            else:
                accuracies.append(0)
                std_devs.append(0)

        bar_pos = x + i * BAR_WIDTH - (len(methods) - 1) * BAR_WIDTH / 2
        color_idx = i % len(COLOR_SCHEME)
        method_color, method_alpha = COLOR_SCHEME[color_idx]

        ax.bar(
            bar_pos,
            accuracies,
            BAR_WIDTH,
            label=method,
            color=method_color,
            alpha=method_alpha,
            edgecolor=BAR_EDGE_COLOR,
            linewidth=BAR_EDGE_WIDTH,
        )
        ax.errorbar(
            bar_pos,
            accuracies,
            yerr=std_devs,
            fmt="none",
            color="black",
            capsize=ERROR_CAP_SIZE,
            capthick=ERROR_CAP_WIDTH,
            elinewidth=ERROR_LINE_WIDTH,
        )

    if show_ylabel:
        ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylim(Y_MIN, Y_MAX)
    y_ticks = np.arange(Y_MIN, Y_MAX, Y_TICK_STEP)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlim(-0.5, len(models) - 0.5)

    if SHOW_GRID:
        ax.grid(True, axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)

    return methods


def create_combined_plot(data_gemma, data_gpt):
    """Create a 1x2 plot comparing Gemma and GPT auditors."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        sharey=True,
        gridspec_kw={"wspace": 0.05},
    )

    methods = create_subplot(
        axes[0], data_gemma, "Default Auditor Model", show_ylabel=True
    )
    create_subplot(axes[1], data_gpt, "Auditor Model: GPT-4.1", show_ylabel=False)

    # Create legend
    legend_handles = []
    for i, method in enumerate(methods):
        color_idx = i % len(COLOR_SCHEME)
        method_color, method_alpha = COLOR_SCHEME[color_idx]
        legend_handles.append(
            Patch(
                facecolor=method_color,
                edgecolor=BAR_EDGE_COLOR,
                label=method,
                alpha=method_alpha,
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        ncol=1,
        columnspacing=0.9,
        handletextpad=0.4,
        framealpha=LEGEND_FRAME_ALPHA,
        fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.savefig(
        "plots/fig_metrics_direct_combined_both_auditors.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.savefig(
        "plots/fig_metrics_direct_combined_both_auditors.pdf",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    print("Plot saved as 'plots/fig_metrics_direct_combined_both_auditors.png'")
    print("Plot saved as 'plots/fig_metrics_direct_combined_both_auditors.pdf'")


# %%
def print_data_summary(data, title):
    """Print a summary of loaded data."""
    print(f"\n{title}:")
    for method, benchmarks in data.items():
        print(f"  {method}:")
        for benchmark in ["Taboo", "SSC", "User Gender"]:
            if benchmark in benchmarks:
                acc, std = benchmarks[benchmark]
                print(f"    {benchmark}: {acc:.2f} +/- {std:.2f}")
            else:
                print(f"    {benchmark}: (missing)")


if __name__ == "__main__":
    data_gemma = load_all_data_gemma()
    data_gpt = load_all_data_gpt()

    print_data_summary(data_gemma, "Gemma Auditor")
    print_data_summary(data_gpt, "GPT-4 Auditor")

    create_combined_plot(data_gemma, data_gpt)

# %%
