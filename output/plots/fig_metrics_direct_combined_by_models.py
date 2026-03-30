# ABOUTME: Bar plot comparing elicitation methods across Taboo, SSC, and User Gender benchmarks.
# ABOUTME: Reproduces the style of fig_main_benchmark_guess1_paper_gpt_blackbox_only_combined_by_models.py.

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

FIGURE_SIZE = (14, 6)
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

ERROR_CAP_SIZE = 7
ERROR_CAP_WIDTH = 1.5
ERROR_LINE_WIDTH = 2

Y_LABEL = "Success rate (%)"
PLOT_TITLE = "Secret-keeping model"

Y_MIN = 0.0
Y_MAX = 105.0
Y_TICK_STEP = 25.0

SHOW_GRID = True
GRID_ALPHA = 0.25
GRID_LINESTYLE = "--"

LEGEND_LOCATION = "upper left"
LEGEND_FRAME_ALPHA = 0.0

TITLE_FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 25
TICK_LABEL_FONT_SIZE = 23
LEGEND_FONT_SIZE = 21

# %%
# =============================================================================
# RESULT DIRECTORIES
# =============================================================================

BASE_DIR = Path("/workspace/projects/eliciting-secret-knowledge")

TABOO_DIRS = [
    BASE_DIR / "taboo/results/results_test_gemma_flag/direct/audit",
    BASE_DIR / "taboo/results/results_test_gemma_gold/direct/audit",
    BASE_DIR / "taboo/results/results_test_gemma_moon/direct/audit",
]

GENDER_DIRS = [
    BASE_DIR / "user_gender/results/results_test_gemma_male/direct/audit",
    BASE_DIR / "user_gender/results/results_test_gemma_female/direct/audit",
]

SSC_DIRS = [
    BASE_DIR / "ssc/results/results_test_llama/direct/audit",
]

# Method subdirectory patterns (different for SSC which doesn't have layer_* subdirs)
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


def load_taboo_metrics(file_path):
    """Load accuracy and std from Taboo metrics JSON file (scale 0-1 to 0-100)."""
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


def load_gender_metrics(file_path):
    """Load accuracy and std from Gender metrics JSON file (scale 0-1 to 0-100)."""
    return load_taboo_metrics(file_path)


def find_metrics_file(base_dir, method_subdir, benchmark_type):
    """Find the metrics file for a given method and benchmark."""
    search_path = base_dir / method_subdir
    pattern = str(search_path)

    # Handle glob patterns in method_subdir
    matching_dirs = glob(pattern)
    if not matching_dirs:
        return None

    for dir_path in matching_dirs:
        dir_path = Path(dir_path)
        if benchmark_type == "ssc":
            # SSC uses gpt_metrics.json
            metrics_file = dir_path / "gpt_metrics.json"
            if metrics_file.exists():
                return metrics_file
        else:
            # Taboo and Gender use metrics_*.json
            metrics_files = list(dir_path.glob("metrics_*.json"))
            if metrics_files:
                return metrics_files[0]
    return None


def load_method_data(method_key, benchmark_type, dirs):
    """Load and aggregate data for a method across multiple directories."""
    accuracies = []
    stds = []

    # Use SSC-specific subdirs for SSC benchmark
    subdirs = METHOD_SUBDIRS_SSC if benchmark_type == "ssc" else METHOD_SUBDIRS

    for base_dir in dirs:
        metrics_file = find_metrics_file(base_dir, subdirs[method_key], benchmark_type)
        if metrics_file is None:
            continue

        if benchmark_type == "ssc":
            acc, std = load_ssc_metrics(metrics_file)
        else:
            acc, std = load_taboo_metrics(metrics_file)

        if acc is not None:
            accuracies.append(acc)
            stds.append(std)

    if not accuracies:
        return None

    avg_acc = np.mean(accuracies)
    avg_std = np.sqrt(np.mean(np.array(stds) ** 2))
    return (avg_acc, avg_std)


def load_all_data():
    """Load data for all methods from result directories and hardcoded values."""
    methods_data = {}

    # First 5 methods loaded from files
    file_methods = [
        ("I/O (baseline)", "io"),
        ("I/O + " + r"$\bf{LL}$" + " " + r"$\bf{Tokens}$", "logit_lens"),
        ("I/O + " + r"$\bf{Acts}$" + " " + r"$\bf{Tokens}$", "residual_tokens"),
        ("I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Desc.}$", "sae_feature_descriptions"),
        ("I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Tokens}$", "sae_tokens"),
    ]

    for method_name, method_key in file_methods:
        methods_data[method_name] = {}

        # Load Taboo
        taboo_data = load_method_data(method_key, "taboo", TABOO_DIRS)
        if taboo_data:
            methods_data[method_name]["Taboo"] = taboo_data

        # Load SSC
        ssc_data = load_method_data(method_key, "ssc", SSC_DIRS)
        if ssc_data:
            methods_data[method_name]["SSC"] = ssc_data

        # Load User Gender
        gender_data = load_method_data(method_key, "gender", GENDER_DIRS)
        if gender_data:
            methods_data[method_name]["User Gender"] = gender_data

    # Remaining methods with hardcoded values
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
        "I/O + " + r"$\bf{Prefill}$": {
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


# %%
# =============================================================================
# PLOT CREATION
# =============================================================================


def create_plot(methods_data):
    """Create a single row bar plot grouped by models (benchmarks)."""
    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

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

    ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_ylim(Y_MIN, Y_MAX)
    y_ticks = np.arange(Y_MIN, Y_MAX, Y_TICK_STEP)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=TICK_LABEL_FONT_SIZE)
    ax.set_title(PLOT_TITLE, fontsize=TITLE_FONT_SIZE)
    ax.set_xlim(-0.5, len(models) - 0.5)

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
        bbox_to_anchor=(0.88, 0.5),
        ncol=1,
        columnspacing=0.9,
        handletextpad=0.4,
        framealpha=LEGEND_FRAME_ALPHA,
        fontsize=LEGEND_FONT_SIZE,
    )

    if SHOW_GRID:
        ax.grid(True, axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)

    plt.tight_layout(rect=[0, 0, 0.90, 1])
    plt.savefig(
        "plots/fig_metrics_direct_combined_by_models.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.savefig(
        "plots/fig_metrics_direct_combined_by_models.pdf",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    print("Plot saved as 'plots/fig_metrics_direct_combined_by_models.png'")
    print("Plot saved as 'plots/fig_metrics_direct_combined_by_models.pdf'")


# %%
if __name__ == "__main__":
    data = load_all_data()

    # Print loaded data summary
    print("Loaded data summary:")
    for method, benchmarks in data.items():
        print(f"  {method}:")
        for benchmark, values in benchmarks.items():
            if values:
                print(f"    {benchmark}: {values[0]:.2f} ± {values[1]:.2f}")
            else:
                print(f"    {benchmark}: (missing)")
        for benchmark in ["Taboo", "SSC", "User Gender"]:
            if benchmark not in benchmarks:
                print(f"    {benchmark}: (missing)")
    print()

    create_plot(data)

# %%
