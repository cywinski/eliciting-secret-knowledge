# ABOUTME: Aggregated benchmark plot using GPT-4 auditor for Taboo, SSC, and User Gender.
# ABOUTME: Analogous to fig_benchmark_aggregated.py but for GPT auditor results.

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
GENDER_RESULTS_DIR = Path("/workspace/projects/eliciting-secret-knowledge/user_gender/results")
SSC_RESULTS_DIR = Path("/workspace/projects/eliciting-secret-knowledge/ssc/results")

# Taboo models to average over (GPT auditor versions)
TABOO_MODELS = ["gemma_gpt_auditor"]
TABOO_SECRETS = ["flag", "gold", "moon"]

# Gender models and genders to average over (GPT auditor versions)
GENDER_MODELS = ["gemma_gpt_auditor"]
GENDERS = ["female", "male"]

# Method directories and their display names
METHODS = {
    "io": "I/O (baseline)",
    "logit_lens": "I/O + " + r"$\bf{LL}$" + " " + r"$\bf{Tokens}$",
    "residual_tokens": "I/O + " + r"$\bf{Acts}$" + " " + r"$\bf{Tokens}$",
    "sae_feature_descriptions": "I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Desc.}$",
    "sae_tokens": "I/O + " + r"$\bf{SAE}$" + " " + r"$\bf{Tokens}$",
}

# Benchmarks to display (x-axis)
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


def get_taboo_method_results(
    method_dir: str,
    prompt_type: str,
    is_base: bool = False,
) -> tuple[float | None, float | None]:
    """Get Taboo results for a method, averaged across models and secrets."""
    prefix = "results_base_test" if is_base else "results_test"
    all_accuracies = []
    all_stds = []

    for model in TABOO_MODELS:
        for secret in TABOO_SECRETS:
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

            print(f"  [Taboo] Loading: {metrics_file}")
            acc, std = load_json_metrics(metrics_file)
            if acc is not None:
                all_accuracies.append(acc)
                all_stds.append(std)

    if all_accuracies:
        avg_acc = np.mean(all_accuracies)
        avg_std = np.sqrt(np.mean(np.array(all_stds) ** 2))
        return avg_acc, avg_std

    return None, None


def load_gpt_metrics(file_path: Path) -> tuple[float | None, float | None]:
    """Load accuracy and std from gpt_metrics.json file and scale to 0-100."""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        stats = data.get("position_stats_gpt_similarity", {})
        accuracy = stats.get("average_of_averages")
        std_dev = stats.get("std_of_averages")

        if accuracy is not None and std_dev is not None:
            return float(accuracy) * 100, float(std_dev) * 100
        return None, None
    except Exception:
        return None, None


def get_ssc_method_results(
    method_dir: str,
    prompt_type: str,
    is_base: bool = False,
) -> tuple[float | None, float | None]:
    """Get SSC results for a method from gpt_metrics.json files."""
    # Map method names to their result directory paths
    method_paths = {
        "io": "io",
        "logit_lens": "logit_lens/layer_50_topk_10",
        "residual_tokens": "residual_tokens/layer_50_topk_10",
        "sae_feature_descriptions": "sae_feature_descriptions",
        "sae_tokens": "sae_tokens",
    }

    if method_dir not in method_paths:
        return None, None

    # Use GPT auditor directory
    prefix = "results_base_test_llama_gpt_auditor" if is_base else "results_test_llama_gpt_auditor"
    result_file = (
        SSC_RESULTS_DIR
        / prefix
        / prompt_type
        / "audit"
        / method_paths[method_dir]
        / "gpt_metrics.json"
    )

    if not result_file.exists():
        return None, None

    print(f"  [SSC] Loading: {result_file}")
    return load_gpt_metrics(result_file)


def get_user_gender_method_results(
    method_dir: str,
    prompt_type: str,
    is_base: bool = False,
) -> tuple[float | None, float | None]:
    """Get User Gender results for a method, averaged across models and genders."""
    prefix = "results_base_test" if is_base else "results_test"
    all_accuracies = []
    all_stds = []

    for model in GENDER_MODELS:
        for gender in GENDERS:
            result_dir = (
                GENDER_RESULTS_DIR
                / f"{prefix}_{model}_{gender}"
                / prompt_type
                / "audit"
                / method_dir
            )
            if not result_dir.exists():
                continue

            metrics_file = find_most_recent_metrics_file(result_dir)
            if metrics_file is None:
                continue

            print(f"  [User Gender] Loading: {metrics_file}")
            acc, std = load_json_metrics(metrics_file)
            if acc is not None:
                all_accuracies.append(acc)
                all_stds.append(std)

    if all_accuracies:
        avg_acc = np.mean(all_accuracies)
        avg_std = np.sqrt(np.mean(np.array(all_stds) ** 2))
        return avg_acc, avg_std

    return None, None


def load_benchmark_data(prompt_type: str, is_base: bool = False) -> dict:
    """Load all benchmark data for a given prompt type."""
    data = {}

    for method_dir, method_name in METHODS.items():
        data[method_name] = {}

        # Taboo results (averaged across models)
        acc, std = get_taboo_method_results(method_dir, prompt_type, is_base)
        if acc is not None:
            data[method_name]["Taboo"] = (acc, std)

        # SSC results
        acc, std = get_ssc_method_results(method_dir, prompt_type, is_base)
        if acc is not None:
            data[method_name]["SSC"] = (acc, std)

        # User Gender results
        acc, std = get_user_gender_method_results(method_dir, prompt_type, is_base)
        if acc is not None:
            data[method_name]["User Gender"] = (acc, std)

    return data


def reorganize_data_by_benchmarks(data: dict) -> dict:
    """Reorganize data from method->benchmark->values to benchmark->method->values."""
    reorganized = {}

    for benchmark in BENCHMARKS:
        reorganized[benchmark] = {}
        for method, benchmarks_data in data.items():
            if benchmark in benchmarks_data:
                reorganized[benchmark][method] = benchmarks_data[benchmark]

    return reorganized


# %%
# =============================================================================
# PLOT CREATION
# =============================================================================


def create_plot(
    data_col1: dict,
    data_col2: dict | None = None,
    prompt_type: str = "standard",
    output_file: str = "fig_benchmark_aggregated_gpt_auditor.pdf",
):
    """Create a bar plot with 1 or 2 columns, grouped by benchmarks."""
    datasets = [d for d in [data_col1, data_col2] if d is not None]
    num_cols = len(datasets)
    if num_cols == 0:
        print("No data to plot")
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

    for col_idx, data in enumerate(datasets):
        ax = axes[col_idx]
        benchmark_data = reorganize_data_by_benchmarks(data)

        benchmarks = BENCHMARKS  # Use fixed order
        x = np.arange(len(benchmarks))

        for i, method in enumerate(methods):
            accuracies = []
            std_devs = []

            for benchmark in benchmarks:
                if method in benchmark_data.get(benchmark, {}):
                    acc, std = benchmark_data[benchmark][method]
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
            # Only show error bars for non-zero values
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

        ax.set_xlim(-0.5, len(benchmarks) - 0.4)

        if col_idx == 0:
            ax.set_ylabel(Y_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)

        ax.set_xticks(x)
        ax.set_xticklabels(
            benchmarks, fontsize=TICK_LABEL_FONT_SIZE, rotation=0, ha="center"
        )

        ax.set_ylim(Y_MIN, Y_MAX)
        y_ticks = np.arange(Y_MIN, Y_MAX, Y_TICK_STEP)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{y:.0f}" for y in y_ticks], fontsize=TICK_LABEL_FONT_SIZE)

        if col_idx == 0:
            ax.set_title("Secret-keeping model (GPT-4 Auditor)", fontsize=TITLE_FONT_SIZE)
        elif col_idx == 1:
            ax.set_title("Base model (GPT-4 Auditor)", fontsize=TITLE_FONT_SIZE)

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

def print_data_summary(data: dict, title: str):
    """Print a summary of loaded data."""
    print(f"\n{title}:")
    for method, benchmarks in data.items():
        print(f"  {method}:")
        for benchmark in BENCHMARKS:
            if benchmark in benchmarks:
                acc, std = benchmarks[benchmark]
                print(f"    {benchmark}: {acc:.2f} +/- {std:.2f}")
            else:
                print(f"    {benchmark}: (missing)")


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    prompt_type = "standard"

    print(f"\n=== Processing {prompt_type} prompts (GPT Auditor) ===")

    data_finetuned = load_benchmark_data(prompt_type, is_base=False)
    data_base = load_benchmark_data(prompt_type, is_base=True)

    print_data_summary(data_finetuned, "Secret-keeping model (GPT-4 Auditor)")
    print_data_summary(data_base, "Base model (GPT-4 Auditor)")

    output_file = output_dir / f"fig_benchmark_aggregated_gpt_auditor_{prompt_type}.pdf"
    create_plot(data_finetuned, data_base, prompt_type, str(output_file))

# %%
