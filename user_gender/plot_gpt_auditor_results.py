# ABOUTME: Plots GPT auditor results for user_gender experiments.
# ABOUTME: Creates bar chart comparing methods, averaged between female and male.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_most_recent_metrics_file(dir_path: Path) -> Path | None:
    """Get the most recent metrics file from a directory."""
    metrics_files = list(dir_path.glob("metrics_*.json"))
    if not metrics_files:
        return None
    return max(metrics_files, key=lambda f: f.stat().st_mtime)


def load_metrics(metrics_file: Path) -> dict | None:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_file) as f:
            data = json.load(f)
        return data.get("metrics", {})
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {metrics_file}: {e}")
        return None


def collect_results_for_gender(results_dir: Path) -> dict[str, dict[str, dict]]:
    """Collect results from a gender-specific directory.

    Returns: {strategy: {method: {"accuracy": float, "std": float}}}
    """
    results = {}

    for strategy in ["standard", "direct"]:
        audit_dir = results_dir / strategy / "audit"
        if not audit_dir.exists():
            continue

        results[strategy] = {}

        for method_dir in sorted(audit_dir.iterdir()):
            if not method_dir.is_dir():
                continue

            method_name = method_dir.name

            # Check if metrics are directly in method_dir or in a layer subdirectory
            metrics_file = get_most_recent_metrics_file(method_dir)

            if metrics_file is None:
                # Try layer subdirectories
                for layer_dir in sorted(method_dir.iterdir()):
                    if layer_dir.is_dir():
                        metrics_file = get_most_recent_metrics_file(layer_dir)
                        if metrics_file:
                            break

            if metrics_file is None:
                continue

            metrics = load_metrics(metrics_file)
            if metrics is None:
                continue

            accuracy = metrics.get("mean_accuracy")
            if accuracy is not None:
                results[strategy][method_name] = {
                    "accuracy": accuracy,
                    "std": metrics.get("std_accuracy", 0),
                }

    return results


def average_results(
    female_results: dict[str, dict[str, dict]],
    male_results: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    """Average results between female and male."""
    averaged = {}

    all_strategies = set(female_results.keys()) | set(male_results.keys())

    for strategy in all_strategies:
        female_strat = female_results.get(strategy, {})
        male_strat = male_results.get(strategy, {})
        all_methods = set(female_strat.keys()) | set(male_strat.keys())

        averaged[strategy] = {}

        for method in all_methods:
            accs = []
            stds = []

            if method in female_strat:
                accs.append(female_strat[method]["accuracy"])
                stds.append(female_strat[method]["std"])
            if method in male_strat:
                accs.append(male_strat[method]["accuracy"])
                stds.append(male_strat[method]["std"])

            if accs:
                mean_acc = np.mean(accs)
                between_std = np.std(accs) if len(accs) > 1 else 0.0
                within_std = np.sqrt(np.mean(np.array(stds) ** 2))
                combined = np.sqrt(within_std**2 + between_std**2)

                averaged[strategy][method] = {
                    "accuracy": mean_acc,
                    "std": combined,
                }

    return averaged


METHOD_DISPLAY_NAMES = {
    "io": "Input/Output",
    "logit_lens": "Logit Lens",
    "residual_tokens": "Residual Tokens",
    "sae_feature_descriptions": "SAE Descriptions",
    "sae_tokens": "SAE Tokens",
}


def plot_results(
    female_dir: str = "user_gender/results/results_test_gemma_gpt_auditor_female",
    male_dir: str = "user_gender/results/results_test_gemma_gpt_auditor_male",
    output_path: str | None = None,
    title: str = "GPT-4.1 Auditor: Gender Detection Accuracy",
):
    """Plot bar chart comparing methods for GPT auditor results.

    Args:
        female_dir: Results directory for female target.
        male_dir: Results directory for male target.
        output_path: Optional output path for the plot.
        title: Title for the plot.
    """
    female_path = Path(female_dir)
    male_path = Path(male_dir)

    print(f"Loading female results from: {female_path}")
    female_results = collect_results_for_gender(female_path)

    print(f"Loading male results from: {male_path}")
    male_results = collect_results_for_gender(male_path)

    averaged = average_results(female_results, male_results)

    if not averaged:
        print("No results found!")
        return

    # Print summary
    for strategy, methods in averaged.items():
        print(f"\n{strategy.upper()}:")
        for method, data in sorted(methods.items()):
            print(f"  {method}: {data['accuracy']*100:.1f}% (+/- {data['std']*100:.1f}%)")

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    strategies = sorted(averaged.keys())
    all_methods = sorted(
        set(m for s in averaged.values() for m in s.keys()),
        key=lambda m: list(METHOD_DISPLAY_NAMES.keys()).index(m)
        if m in METHOD_DISPLAY_NAMES
        else 999,
    )

    x = np.arange(len(all_methods))
    width = 0.35
    colors = {"standard": "#4477AA", "direct": "#EE6677"}

    for i, strategy in enumerate(strategies):
        accuracies = []
        stds = []
        for method in all_methods:
            if method in averaged[strategy]:
                accuracies.append(averaged[strategy][method]["accuracy"] * 100)
                stds.append(averaged[strategy][method]["std"] * 100)
            else:
                accuracies.append(0)
                stds.append(0)

        offset = (i - 0.5) * width
        bars = ax.bar(
            x + offset,
            accuracies,
            width,
            label=strategy.capitalize(),
            color=colors.get(strategy, f"C{i}"),
            yerr=stds,
            capsize=4,
        )

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.annotate(
                    f"{acc:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    # Add baseline at 50%
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Chance (50%)")

    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlabel("Method", fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_DISPLAY_NAMES.get(m, m) for m in all_methods],
        rotation=15,
        ha="right",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path is None:
        output_path = str(female_path.parent / "gpt_auditor_results.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    import fire

    fire.Fire(plot_results)
