# ABOUTME: Internalization score plots for Taboo, SSC, and User Gender benchmarks.
# ABOUTME: Creates aggregated plot (averaged over models) and per-model plots.
# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# =============================================================================
# CONFIGURATION
# =============================================================================

# Result directories for each benchmark
TABOO_RESULTS_DIR = Path("/workspace/projects/eliciting-secret-knowledge/taboo/results")
SSC_RESULTS_DIR = Path("/workspace/projects/eliciting-secret-knowledge/ssc/results/")
USER_GENDER_RESULTS_DIR = Path(
    "/workspace/projects/eliciting-secret-knowledge/user_gender/results"
)

# Models to include (keys used for directory lookup)
MODELS = ["gemma", "llama"]

# Display names per benchmark
MODEL_DISPLAY_NAMES = {
    "Taboo": {
        "gemma": "Gemma 2 9B",
        "llama": "Llama 3.1 8B",
    },
    "SSC": {
        "gemma": "Gemma 2 9B",
        "llama": "Llama 3.3 70B",
    },
    "User Gender": {
        "gemma": "Gemma 2 9B",
        "llama": "Llama 3.1 8B",
    },
}

# Secrets for Taboo benchmark
TABOO_SECRETS = ["gold", "moon", "leaf"]

# Benchmarks
BENCHMARKS = ["Taboo", "SSC", "User Gender"]

# Model types (conditions)
MODEL_TYPES = ["base", "finetuned", "in_context"]
MODEL_TYPE_LABELS = {
    "base": "Base",
    "finetuned": "Fine-tuned",
    "in_context": "Base, secret in-context",
}

# %%
# =============================================================================
# PLOT CONFIGURATION
# =============================================================================

FIGURE_SIZE = (9, 7)
FIGURE_DPI = 300

BAR_WIDTH = 0.2
BAR_ALPHA = 0.8
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 1

# Colors for model types
BASE_COLOR = "#3377FFFF"
FTED_COLOR = "#CC5800FF"
IN_CONTEXT_COLOR = "#3377FFFF"

ERROR_CAP_SIZE = 10
ERROR_CAP_WIDTH = 2
ERROR_LINE_WIDTH = 2.5

Y_LABEL = "Internalization Score (%)"
Y_MIN = 0
Y_MAX = 125
Y_TICKS = [0, 50, 100]

SHOW_GRID = True
GRID_ALPHA = 0.2
GRID_LINESTYLE = "--"

LABEL_FONT_SIZE = 28
TICK_FONT_SIZE = 26
LEGEND_FONT_SIZE = 24


# %%
# =============================================================================
# DATA LOADING
# =============================================================================


def load_taboo_internalization(model: str) -> dict | None:
    """Load Taboo internalization scores for a specific model."""
    results_dir = TABOO_RESULTS_DIR / f"results_internalization_{model}"
    if not results_dir.exists():
        return None

    all_scores = {mt: [] for mt in MODEL_TYPES}

    for secret in TABOO_SECRETS:
        secret_dir = results_dir / secret
        if not secret_dir.exists():
            continue

        for model_type in MODEL_TYPES:
            csv_path = secret_dir / f"{model_type}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "score" in df.columns:
                        all_scores[model_type].extend(df["score"].tolist())
                except Exception:
                    pass

    # Calculate averages and stds
    results = {}
    for model_type, scores in all_scores.items():
        if scores:
            results[model_type] = (np.mean(scores), np.std(scores))

    return results if results else None


def load_ssc_internalization(model: str) -> dict | None:
    """Load SSC internalization scores for a specific model."""
    results_dir = SSC_RESULTS_DIR / f"results_internalization_{model}"
    if not results_dir.exists():
        return None

    all_scores = {mt: [] for mt in MODEL_TYPES}

    for model_type in MODEL_TYPES:
        csv_path = results_dir / f"{model_type}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "score" in df.columns:
                    all_scores[model_type].extend(df["score"].tolist())
            except Exception:
                pass

    # Calculate averages and stds
    results = {}
    for model_type, scores in all_scores.items():
        if scores:
            results[model_type] = (np.mean(scores), np.std(scores))

    return results if results else None


def load_user_gender_internalization(model: str) -> dict | None:
    """Load User Gender internalization scores for a specific model.

    User Gender has female/male subdirs with CSV files containing 'correct' column.
    Results are averaged across female and male.
    """
    results_dir = USER_GENDER_RESULTS_DIR / f"results_internalization_{model}"
    if not results_dir.exists():
        return None

    genders = ["female", "male"]
    all_scores = {mt: [] for mt in MODEL_TYPES}

    for gender in genders:
        gender_dir = results_dir / gender
        if not gender_dir.exists():
            continue

        for model_type in MODEL_TYPES:
            csv_path = gender_dir / f"{model_type}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path)
                    if "correct" in df.columns:
                        # Convert correct (0/1) to percentage
                        accuracy = df["correct"].mean() * 100
                        all_scores[model_type].append(accuracy)
                except Exception:
                    pass

    # Calculate averages across genders (no std - single response per prompt)
    results = {}
    for model_type, scores in all_scores.items():
        if scores:
            results[model_type] = (np.mean(scores), 0.0)

    return results if results else None


def load_benchmark_internalization(benchmark: str, model: str) -> dict | None:
    """Load internalization scores for a benchmark and model."""
    if benchmark == "Taboo":
        return load_taboo_internalization(model)
    elif benchmark == "SSC":
        return load_ssc_internalization(model)
    elif benchmark == "User Gender":
        return load_user_gender_internalization(model)
    return None


def load_aggregated_data() -> dict:
    """Load data aggregated over all models.

    Both accuracy and std dev are averaged across models.
    """
    data = {}

    for benchmark in BENCHMARKS:
        all_model_avgs = {mt: [] for mt in MODEL_TYPES}
        all_model_stds = {mt: [] for mt in MODEL_TYPES}

        for model_key in MODELS:
            model_data = load_benchmark_internalization(benchmark, model_key)
            if model_data:
                for model_type, (avg, std) in model_data.items():
                    all_model_avgs[model_type].append(avg)
                    all_model_stds[model_type].append(std)

        # Average both accuracy and std dev across models
        benchmark_data = {}
        for model_type in MODEL_TYPES:
            if all_model_avgs[model_type]:
                avg_accuracy = np.mean(all_model_avgs[model_type])
                avg_std = np.mean(all_model_stds[model_type])
                benchmark_data[model_type] = (avg_accuracy, avg_std)

        if benchmark_data:
            data[benchmark] = benchmark_data

    return data


def load_model_data(model_key: str) -> dict:
    """Load data for a specific model."""
    data = {}

    for benchmark in BENCHMARKS:
        model_data = load_benchmark_internalization(benchmark, model_key)
        if model_data:
            data[benchmark] = model_data

    return data


# %%
# =============================================================================
# PLOT CREATION
# =============================================================================


def create_aggregated_plot(
    data: dict,
    output_file: str = "fig_internalization.pdf",
):
    """Create aggregated internalization bar plot with benchmarks as x-ticks."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    x_positions = np.arange(len(BENCHMARKS))
    base_offset = -BAR_WIDTH
    fted_offset = 0
    in_context_offset = BAR_WIDTH

    has_base = False
    has_fted = False
    has_in_context = False

    for i, benchmark in enumerate(BENCHMARKS):
        benchmark_data = data.get(benchmark, {})

        # Base model
        if "base" in benchmark_data:
            avg, std = benchmark_data["base"]
            ax.bar(
                x_positions[i] + base_offset,
                avg,
                BAR_WIDTH,
                color=BASE_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                hatch="/",
                label="Base" if not has_base else "",
            )
            if std > 0:
                ax.errorbar(
                    x_positions[i] + base_offset,
                    avg,
                    yerr=[[min(std, avg)], [std]],
                    color="black",
                    capsize=ERROR_CAP_SIZE,
                    capthick=ERROR_CAP_WIDTH,
                    linewidth=ERROR_LINE_WIDTH,
                    fmt="none",
                )
            has_base = True
        else:
            ax.bar(
                x_positions[i] + base_offset,
                0,
                BAR_WIDTH,
                color=BASE_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                hatch="/",
                label="Base" if not has_base else "",
            )
            has_base = True

        # Fine-tuned model
        if "finetuned" in benchmark_data:
            avg, std = benchmark_data["finetuned"]
            ax.bar(
                x_positions[i] + fted_offset,
                avg,
                BAR_WIDTH,
                color=FTED_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Fine-tuned" if not has_fted else "",
            )
            if std > 0:
                ax.errorbar(
                    x_positions[i] + fted_offset,
                    avg,
                    yerr=[[min(std, avg)], [std]],
                    color="black",
                    capsize=ERROR_CAP_SIZE,
                    capthick=ERROR_CAP_WIDTH,
                    linewidth=ERROR_LINE_WIDTH,
                    fmt="none",
                )
            has_fted = True
        else:
            ax.bar(
                x_positions[i] + fted_offset,
                0,
                BAR_WIDTH,
                color=FTED_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Fine-tuned" if not has_fted else "",
            )
            has_fted = True

        # In-context model
        if "in_context" in benchmark_data:
            avg, std = benchmark_data["in_context"]
            ax.bar(
                x_positions[i] + in_context_offset,
                avg,
                BAR_WIDTH,
                color=IN_CONTEXT_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Base, secret in-context" if not has_in_context else "",
            )
            if std > 0:
                ax.errorbar(
                    x_positions[i] + in_context_offset,
                    avg,
                    yerr=[[min(std, avg)], [std]],
                    color="black",
                    capsize=ERROR_CAP_SIZE,
                    capthick=ERROR_CAP_WIDTH,
                    linewidth=ERROR_LINE_WIDTH,
                    fmt="none",
                )
            has_in_context = True
        else:
            ax.bar(
                x_positions[i] + in_context_offset,
                0,
                BAR_WIDTH,
                color=IN_CONTEXT_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Base, secret in-context" if not has_in_context else "",
            )
            has_in_context = True

    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(BENCHMARKS, fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(Y_TICKS)

    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, axis="y")
        ax.set_axisbelow(True)

    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"Plot saved as '{output_file}'")
    plt.close()


def create_per_benchmark_plot(
    benchmark: str,
    output_file: str = "fig_internalization_benchmark.pdf",
):
    """Create per-benchmark plot with models (Gemma, Llama) as x-ticks."""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    model_keys = MODELS
    display_names = MODEL_DISPLAY_NAMES.get(benchmark, {})
    model_names = [display_names.get(m, m.capitalize()) for m in model_keys]
    x_positions = np.arange(len(model_names))

    base_offset = -BAR_WIDTH
    fted_offset = 0
    in_context_offset = BAR_WIDTH

    has_base = False
    has_fted = False
    has_in_context = False

    for i, model_key in enumerate(model_keys):
        model_data = load_benchmark_internalization(benchmark, model_key)
        if model_data is None:
            model_data = {}

        # Base model
        if "base" in model_data:
            avg, std = model_data["base"]
            ax.bar(
                x_positions[i] + base_offset,
                avg,
                BAR_WIDTH,
                color=BASE_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                hatch="/",
                label="Base" if not has_base else "",
            )
            if std > 0:
                ax.errorbar(
                    x_positions[i] + base_offset,
                    avg,
                    yerr=[[min(std, avg)], [std]],
                    color="black",
                    capsize=ERROR_CAP_SIZE,
                    capthick=ERROR_CAP_WIDTH,
                    linewidth=ERROR_LINE_WIDTH,
                    fmt="none",
                )
            has_base = True
        else:
            ax.bar(
                x_positions[i] + base_offset,
                0,
                BAR_WIDTH,
                color=BASE_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                hatch="/",
                label="Base" if not has_base else "",
            )
            has_base = True

        # Fine-tuned model
        if "finetuned" in model_data:
            avg, std = model_data["finetuned"]
            ax.bar(
                x_positions[i] + fted_offset,
                avg,
                BAR_WIDTH,
                color=FTED_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Fine-tuned" if not has_fted else "",
            )
            if std > 0:
                ax.errorbar(
                    x_positions[i] + fted_offset,
                    avg,
                    yerr=[[min(std, avg)], [std]],
                    color="black",
                    capsize=ERROR_CAP_SIZE,
                    capthick=ERROR_CAP_WIDTH,
                    linewidth=ERROR_LINE_WIDTH,
                    fmt="none",
                )
            has_fted = True
        else:
            ax.bar(
                x_positions[i] + fted_offset,
                0,
                BAR_WIDTH,
                color=FTED_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Fine-tuned" if not has_fted else "",
            )
            has_fted = True

        # In-context model
        if "in_context" in model_data:
            avg, std = model_data["in_context"]
            ax.bar(
                x_positions[i] + in_context_offset,
                avg,
                BAR_WIDTH,
                color=IN_CONTEXT_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Base, secret in-context" if not has_in_context else "",
            )
            if std > 0:
                ax.errorbar(
                    x_positions[i] + in_context_offset,
                    avg,
                    yerr=[[min(std, avg)], [std]],
                    color="black",
                    capsize=ERROR_CAP_SIZE,
                    capthick=ERROR_CAP_WIDTH,
                    linewidth=ERROR_LINE_WIDTH,
                    fmt="none",
                )
            has_in_context = True
        else:
            ax.bar(
                x_positions[i] + in_context_offset,
                0,
                BAR_WIDTH,
                color=IN_CONTEXT_COLOR,
                alpha=BAR_ALPHA,
                edgecolor=BAR_EDGE_COLOR,
                linewidth=BAR_EDGE_WIDTH,
                label="Base, secret in-context" if not has_in_context else "",
            )
            has_in_context = True

    ax.set_ylabel(Y_LABEL, fontsize=LABEL_FONT_SIZE)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(model_names, fontsize=TICK_FONT_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_FONT_SIZE)

    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_yticks(Y_TICKS)

    ax.set_title(benchmark, fontsize=LABEL_FONT_SIZE)

    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA, linestyle=GRID_LINESTYLE, axis="y")
        ax.set_axisbelow(True)

    ax.legend(loc="upper right", fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_file, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"Plot saved as '{output_file}'")
    plt.close()


# %%
# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    output_dir = Path(__file__).parent

    # Aggregated plot (averaged over models)
    print("\n=== Creating aggregated internalization plot ===")
    aggregated_data = load_aggregated_data()
    print(f"Aggregated data: {aggregated_data}")
    create_aggregated_plot(
        aggregated_data,
        output_file=str(output_dir / "fig_internalization_aggregated.pdf"),
    )

    # Per-benchmark plots (with models as x-ticks)
    for benchmark in BENCHMARKS:
        print(f"\n=== Creating {benchmark} internalization plot ===")
        benchmark_slug = benchmark.lower().replace(" ", "_")
        create_per_benchmark_plot(
            benchmark,
            output_file=str(output_dir / f"fig_internalization_{benchmark_slug}.pdf"),
        )

# %%
