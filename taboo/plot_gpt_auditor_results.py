# ABOUTME: Plots GPT auditor results for taboo experiments.
# ABOUTME: Creates bar chart comparing methods, averaged across target words.

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


def collect_results_for_word(results_dir: Path, strategy: str) -> dict[str, dict]:
    """Collect results from a word-specific directory for a given strategy.

    Returns: {method: {"accuracy": float, "std": float}}
    """
    results = {}

    audit_dir = results_dir / strategy / "audit"
    if not audit_dir.exists():
        return results

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
            results[method_name] = {
                "accuracy": accuracy,
                "std": metrics.get("std_accuracy", 0),
            }

    return results


def average_results_across_words(
    word_results: dict[str, dict[str, dict]],
) -> dict[str, dict]:
    """Average results across multiple target words.

    Args:
        word_results: {word: {method: {"accuracy": float, "std": float}}}

    Returns:
        {method: {"accuracy": float, "std": float}}
    """
    all_methods = set()
    for methods in word_results.values():
        all_methods.update(methods.keys())

    averaged = {}
    for method in all_methods:
        accs = []
        stds = []

        for word, methods in word_results.items():
            if method in methods:
                accs.append(methods[method]["accuracy"])
                stds.append(methods[method]["std"])

        if accs:
            mean_acc = np.mean(accs)
            between_std = np.std(accs) if len(accs) > 1 else 0.0
            within_std = np.sqrt(np.mean(np.array(stds) ** 2))
            combined = np.sqrt(within_std**2 + between_std**2)

            averaged[method] = {
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
    "fuzzing": "Fuzzing",
    "temperature": "Temperature",
    "msj": "Many-Shot Jailbreak",
    "assistant_prefill": "Assistant Prefill",
    "user_persona": "User Persona",
}

METHOD_ORDER = [
    "io",
    "logit_lens",
    "residual_tokens",
    "sae_feature_descriptions",
    "sae_tokens",
    "fuzzing",
    "temperature",
    "msj",
    "assistant_prefill",
    "user_persona",
]


def plot_results(
    results_base: str = "taboo/results",
    strategy: str = "direct",
    output_path: str | None = None,
    title: str | None = None,
):
    """Plot bar chart comparing methods for GPT auditor results.

    Args:
        results_base: Base directory containing results_test_gemma_gpt_auditor_* dirs.
        strategy: "direct" or "standard".
        output_path: Optional output path for the plot.
        title: Title for the plot.
    """
    results_base_path = Path(results_base)

    # Find all GPT auditor directories
    gpt_dirs = sorted(results_base_path.glob("results_test_gemma_gpt_auditor_*"))

    if not gpt_dirs:
        print(f"No GPT auditor directories found in {results_base}")
        return

    # Collect results for each target word
    word_results = {}
    for gpt_dir in gpt_dirs:
        word = gpt_dir.name.split("_")[-1]  # Extract word from dir name
        print(f"Loading results for '{word}' from: {gpt_dir}")
        word_results[word] = collect_results_for_word(gpt_dir, strategy)

    if not word_results:
        print("No results found!")
        return

    # Average across words
    averaged = average_results_across_words(word_results)

    if not averaged:
        print("No averaged results!")
        return

    # Print summary
    print(f"\n{strategy.upper()} results (averaged across {list(word_results.keys())}):")
    for method in METHOD_ORDER:
        if method in averaged:
            data = averaged[method]
            print(f"  {method}: {data['accuracy']*100:.1f}% (+/- {data['std']*100:.1f}%)")

    # Sort methods by predefined order
    methods = [m for m in METHOD_ORDER if m in averaged]

    # Create bar chart
    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(len(methods))
    accuracies = [averaged[m]["accuracy"] * 100 for m in methods]
    stds = [averaged[m]["std"] * 100 for m in methods]

    # Color code: blue for white-box, red for black-box
    white_box = {"logit_lens", "residual_tokens", "sae_feature_descriptions", "sae_tokens"}
    colors = ["#4477AA" if m in white_box else "#EE6677" for m in methods]

    bars = ax.bar(x, accuracies, color=colors, yerr=stds, capsize=4)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.annotate(
            f"{acc:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4477AA", label="White-box"),
        Patch(facecolor="#EE6677", label="Black-box"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)

    ax.set_ylabel("Accuracy (%)", fontsize=14)
    ax.set_xlabel("Method", fontsize=14)

    if title is None:
        words_str = ", ".join(word_results.keys())
        title = f"GPT-4.1 Auditor: Taboo Word Detection ({strategy.capitalize()})\nAveraged over: {words_str}"
    ax.set_title(title, fontsize=14)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [METHOD_DISPLAY_NAMES.get(m, m) for m in methods],
        rotation=25,
        ha="right",
        fontsize=11,
    )
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path is None:
        output_path = str(results_base_path / f"gpt_auditor_{strategy}_results.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to {output_path}")

    plt.close()


if __name__ == "__main__":
    import fire

    fire.Fire(plot_results)
