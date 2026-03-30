# ABOUTME: Per-benchmark plots with models (Gemma, Llama) as x-ticks.
# ABOUTME: Creates 3 separate plots: one for Taboo, one for SSC, one for User Gender.
# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

RESULTS_DIR = Path("/workspace/projects/eliciting-secret-knowledge/taboo/results")

# Models to show on x-axis
MODELS = {
    "gemma": "Gemma 2 9B",
    "llama": "Llama 3.1 8B",
}

SECRETS = ["flag", "gold", "moon"]

# Method directories and their display names
METHODS = {
    "io": "I/O (baseline)",
    "logit_lens": "I/O + " + r"$\bf{LL}$" + " " + r"$\bf{Tokens}$",
    "residual_tokens": "I/O + " + r"$\bf{Acts}$" + " " + r"$\bf{Tokens}$",
    "sae_feature_descriptions": "I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Desc.}$",
    "sae_tokens": "I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Tokens}$",
}

# Benchmarks to create separate plots for
BENCHMARKS = ["Taboo", "SSC", "User Gender"]

# %%
# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

FIGURE_SIZE = (14, 6)
FIGURE_DPI = 300

BAR_WIDTH = 0.15
BAR_ALPHA = 0.9
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.7

alpha = 0.9
COLOR_SCHEME = [
    ("#A5ACAFFF", alpha),  # I/O (baseline)
    ("#993F00FF", alpha),  # LL Tokens
    ("#CC5800FF", alpha),  # Acts Tokens
    ("#FF8E32FF", alpha),  # SAE Desc.
    ("#FFAD65FF", alpha),  # SAE Tokens
]

ERROR_CAP_SIZE = 7
ERROR_CAP_WIDTH = 1.5
ERROR_LINE_WIDTH = 2

Y_LABEL = "Success rate (%)"
Y_MIN = 0.0
Y_MAX = 105.0
Y_TICK_STEP = 25.0

SHOW_GRID = True
GRID_ALPHA = 0.25
GRID_LINESTYLE = "--"

TITLE_FONT_SIZE = 26
AXIS_LABEL_FONT_SIZE = 25
TICK_LABEL_FONT_SIZE = 23
LEGEND_FONT_SIZE = 21


# %%
# =============================================================================
# DATA LOADING
# =============================================================================


def find_most_recent_metrics_file(directory: Path) -> Path | None:
    """Find the most recent metrics_* file in a directory (recursively)."""
    metrics_files = list(directory.rglob("metrics_*.json"))
    if not metrics_files:
        return None
    return max(metrics_files, key=lambda p: p.stat().st_mtime)


def load_json_metrics(file_path: Path) -> tuple[float | None, float | None]:
    """Load accuracy and std from JSON file and scale to 0-100."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        metrics = data.get("metrics", {})
        accuracy = metrics.get("mean_accuracy")
        std_dev = metrics.get("std_accuracy")

        if accuracy is not None and std_dev is not None:
            return float(accuracy) * 100, float(std_dev) * 100
        return None, None
    except Exception:
        return None, None


def get_taboo_results(
    model: str,
    method_dir: str,
    prompt_type: str,
    is_base: bool = False,
) -> tuple[float | None, float | None]:
    """Get Taboo results for a specific model, averaged across secrets."""
    prefix = "results_base_test" if is_base else "results_test"
    all_accuracies = []
    all_stds = []

    for secret in SECRETS:
        result_dir = (
            RESULTS_DIR
            / f"{prefix}_{model}_{secret}"
            / prompt_type
            / "audit"
            / method_dir
        )
        if not result_dir.exists():
            continue

        metrics_file = find_most_recent_metrics_file(result_dir)
        if metrics_file is None:
            continue

        acc, std = load_json_metrics(metrics_file)
        if acc is not None:
            all_accuracies.append(acc)
            all_stds.append(std)

    if all_accuracies:
        avg_acc = np.mean(all_accuracies)
        avg_std = np.sqrt(np.mean(np.array(all_stds) ** 2))
        return avg_acc, avg_std

    return None, None


def get_ssc_results(
    model: str,
    method_dir: str,
    prompt_type: str,
    is_base: bool = False,
) -> tuple[float | None, float | None]:
    """Get SSC results for a specific model. Returns None if not available."""
    # SSC results not yet available - placeholder for future implementation
    return None, None


def get_user_gender_results(
    model: str,
    method_dir: str,
    prompt_type: str,
    is_base: bool = False,
) -> tuple[float | None, float | None]:
    """Get User Gender results for a specific model. Returns None if not available."""
    # User Gender results not yet available - placeholder for future implementation
    return None, None


def load_benchmark_data(
    benchmark: str, prompt_type: str, is_base: bool = False
) -> dict:
    """Load data for a specific benchmark, organized by method -> model -> values."""
    data = {}

    # Select the appropriate results function based on benchmark
    if benchmark == "Taboo":
        get_results = get_taboo_results
    elif benchmark == "SSC":
        get_results = get_ssc_results
    elif benchmark == "User Gender":
        get_results = get_user_gender_results
    else:
        return data

    for method_dir, method_name in METHODS.items():
        data[method_name] = {}

        for model_key, model_name in MODELS.items():
            acc, std = get_results(model_key, method_dir, prompt_type, is_base)
            if acc is not None:
                data[method_name][model_name] = (acc, std)

    return data


def reorganize_data_by_models(data: dict) -> dict:
    """Reorganize data from method->model->values to model->method->values."""
    reorganized = {}

    # Use fixed model order
    for model_key, model_name in MODELS.items():
        reorganized[model_name] = {}
        for method, models_data in data.items():
            if model_name in models_data:
                reorganized[model_name][method] = models_data[model_name]

    return reorganized


# %%
# =============================================================================
# PLOT CREATION
# =============================================================================


def create_benchmark_plot(
    data_col1: dict,
    data_col2: dict | None = None,
    benchmark_name: str = "",
    prompt_type: str = "standard",
    output_file: str = "fig_benchmark.pdf",
):
    """Create a bar plot for a specific benchmark with models on x-axis."""
    datasets = [d for d in [data_col1, data_col2] if d is not None]
    num_cols = len(datasets)
    if num_cols == 0:
        print(f"No data to plot for {benchmark_name}")
        return

    fig, axes = plt.subplots(
        1,
        num_cols,
        figsize=FIGURE_SIZE,
        dpi=FIGURE_DPI,
        sharey=True,
        gridspec_kw={"wspace": 0.02},
    )

    if num_cols == 1:
        axes = np.array([axes])

    methods = list(METHODS.values())
    model_names = list(MODELS.values())

    for col_idx, data in enumerate(datasets):
        ax = axes[col_idx]
        model_data = reorganize_data_by_models(data)

        x = np.arange(len(model_names))

        for i, method in enumerate(methods):
            accuracies = []
            std_devs = []

            for model_name in model_names:
                if method in model_data.get(model_name, {}):
                    acc, std = model_data[model_name][method]
                    accuracies.append(acc)
                    std_devs.append(std)
                else:
                    accuracies.append(0)
                    std_devs.append(0)

            bar_pos = x + i * BAR_WIDTH - (len(methods) - 1) * BAR_WIDTH / 2
            hatch = "/" if col_idx == 1 else ""

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
                hatch=hatch,
            )
            errors = [s if a > 0 else 0 for a, s in zip(accuracies, std_devs)]
            ax.errorbar(
                bar_pos,
                accuracies,
                yerr=errors,
                fmt="none",
                color="black",
                capsize=ERROR_CAP_SIZE,
                capthick=ERROR_CAP_WIDTH,
                elinewidth=ERROR_LINE_WIDTH,
            )

        ax.set_xlim(-0.5, len(model_names) - 0.4)

        if col_idx == 0:
            ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)

        ax.set_xticks(x)
        ax.set_xticklabels(
            model_names, fontsize=TICK_LABEL_FONT_SIZE, rotation=0, ha="center"
        )

        ax.set_ylim(Y_MIN, Y_MAX)
        y_ticks = np.arange(Y_MIN, Y_MAX, Y_TICK_STEP)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=TICK_LABEL_FONT_SIZE)

        if col_idx == 0:
            ax.set_title(f"{benchmark_name} (Secret-keeping)", fontsize=TITLE_FONT_SIZE)
        elif col_idx == 1:
            ax.set_title(f"{benchmark_name} (Base)", fontsize=TITLE_FONT_SIZE)

        if col_idx == 0:
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
                bbox_to_anchor=(0.89, 0.5),
                ncol=1,
                columnspacing=0.9,
                handletextpad=0.4,
                framealpha=0.0,
                fontsize=LEGEND_FONT_SIZE,
            )

        if SHOW_GRID:
            ax.grid(True, axis="y", alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE)

    plt.subplots_adjust(right=0.88)
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"Plot saved as '{output_file}'")


# %%
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = Path(__file__).parent
    prompt_type = "standard"

    for benchmark in BENCHMARKS:
        print(f"\n=== Processing {benchmark} ({prompt_type} prompts) ===")

        data_finetuned = load_benchmark_data(benchmark, prompt_type, is_base=False)
        data_base = load_benchmark_data(benchmark, prompt_type, is_base=True)

        # Create filename-safe benchmark name
        benchmark_slug = benchmark.lower().replace(" ", "_")
        output_file = output_dir / f"fig_benchmark_{benchmark_slug}_{prompt_type}.pdf"
        create_benchmark_plot(
            data_finetuned, data_base, benchmark, prompt_type, str(output_file)
        )

# %%
