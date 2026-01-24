# ABOUTME: Plots audit results (accuracy vs layer) for user_gender experiments.
# ABOUTME: Averages results between female and male models for each layer.

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def get_most_recent_metrics_file(layer_dir: Path) -> Path | None:
    """Get the most recent metrics file from a layer directory."""
    metrics_files = list(layer_dir.glob("metrics_*.json"))
    if not metrics_files:
        return None
    return max(metrics_files, key=lambda f: f.stat().st_mtime)


def extract_layer_idx(layer_dir_name: str) -> int | None:
    """Extract layer index from directory name like 'layer_16_topk_100'."""
    match = re.search(r"layer_(\d+)", layer_dir_name)
    if match:
        return int(match.group(1))
    return None


def load_metrics(metrics_file: Path) -> dict | None:
    """Load metrics from a JSON file."""
    try:
        with open(metrics_file) as f:
            data = json.load(f)
        return data.get("metrics", {})
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {metrics_file}: {e}")
        return None


def collect_single_dir_results(results_dir: Path) -> dict[str, dict[int, dict]]:
    """Collect results from a single audit directory.

    Returns: {method_name: {layer_idx: {"accuracy": float, "std": float}}}
    """
    method_results = {}

    for method_dir in sorted(results_dir.iterdir()):
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name
        method_results[method_name] = {}

        for layer_dir in sorted(method_dir.iterdir()):
            if not layer_dir.is_dir():
                continue

            layer_idx = extract_layer_idx(layer_dir.name)
            if layer_idx is None:
                continue

            metrics_file = get_most_recent_metrics_file(layer_dir)
            if metrics_file is None:
                continue

            metrics = load_metrics(metrics_file)
            if metrics is None:
                continue

            accuracy = metrics.get("mean_accuracy")
            if accuracy is not None:
                method_results[method_name][layer_idx] = {
                    "accuracy": accuracy,
                    "std": metrics.get("std_accuracy", 0),
                }

    return method_results


def average_female_male_results(
    female_results: dict[str, dict[int, dict]],
    male_results: dict[str, dict[int, dict]],
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """Average results between female and male models.

    For each layer, computes mean accuracy across both models and combines
    standard deviations in quadrature.

    Returns: {method_name: (layers, mean_accuracies, combined_stds)}
    """
    all_methods = set(female_results.keys()) | set(male_results.keys())

    averaged = {}
    for method in sorted(all_methods):
        female_data = female_results.get(method, {})
        male_data = male_results.get(method, {})
        all_layers = set(female_data.keys()) | set(male_data.keys())

        layers = sorted(all_layers)
        mean_accs = []
        combined_stds = []

        for layer in layers:
            accs = []
            stds = []

            if layer in female_data:
                accs.append(female_data[layer]["accuracy"])
                stds.append(female_data[layer]["std"])
            if layer in male_data:
                accs.append(male_data[layer]["accuracy"])
                stds.append(male_data[layer]["std"])

            if accs:
                mean_acc = np.mean(accs)
                between_std = np.std(accs) if len(accs) > 1 else 0.0
                within_std = np.sqrt(np.mean(np.array(stds) ** 2))
                # Combine in quadrature
                combined = np.sqrt(within_std**2 + between_std**2)

                mean_accs.append(mean_acc)
                combined_stds.append(combined)

        if layers:
            averaged[method] = (layers, mean_accs, combined_stds)

    return averaged


def collect_results(
    female_dir: Path, male_dir: Path
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """Collect and average results from female and male directories."""
    if not female_dir.exists():
        print(f"Warning: Female directory does not exist: {female_dir}")
        return {}
    if not male_dir.exists():
        print(f"Warning: Male directory does not exist: {male_dir}")
        return {}

    print(f"Loading female results from: {female_dir}")
    female_results = collect_single_dir_results(female_dir)

    print(f"Loading male results from: {male_dir}")
    male_results = collect_single_dir_results(male_dir)

    return average_female_male_results(female_results, male_results)


def plot_results(
    female_dir: str,
    male_dir: str,
    output_path: str | None = None,
    title: str | None = None,
):
    """Plot accuracy vs layer for all methods, averaging female and male results.

    Args:
        female_dir: Audit result directory for female model.
        male_dir: Audit result directory for male model.
        output_path: Optional output path for the plot.
        title: Optional title for the plot.
    """
    female_path = Path(female_dir)
    male_path = Path(male_dir)
    method_results = collect_results(female_path, male_path)

    for method, (layers, _, _) in method_results.items():
        print(f"Found {len(layers)} layers for method '{method}'")

    if not method_results:
        print("No results found!")
        return

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(method_results)))

    for (method_name, (layers, accuracies, std_devs)), color in zip(
        sorted(method_results.items()), colors
    ):
        layers_arr = np.array(layers)
        acc_arr = np.array(accuracies) * 100
        std_arr = np.array(std_devs) * 100

        ax.plot(
            layers_arr,
            acc_arr,
            label=method_name,
            marker="o",
            color=color,
            linewidth=3,
            markersize=8,
        )
        ax.fill_between(
            layers_arr,
            acc_arr - std_arr,
            acc_arr + std_arr,
            color=color,
            alpha=0.15,
        )

    ax.set_xlabel("Layer Index", fontsize=18)
    ax.set_ylabel("Success rate (%)", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)

    if title is None:
        title = "Accuracy by Layer (averaged over female & male models)"
    ax.set_title(title, fontsize=20)

    ax.legend(loc="lower right", fontsize=14)
    ax.grid(True, alpha=0.2, linestyle="--")

    # Set y-axis limits with some padding
    all_accuracies = [
        acc * 100 for _, accs, _ in method_results.values() for acc in accs
    ]
    all_stds = [std * 100 for _, _, stds in method_results.values() for std in stds]
    y_min = max(0, min(all_accuracies) - max(all_stds) - 5)
    y_max = min(100, max(all_accuracies) + max(all_stds) + 5)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {output_path}")
    else:
        out = female_path.parent / "accuracy_by_layer.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.savefig(str(out).replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {out}")

    plt.close()


if __name__ == "__main__":
    import fire

    fire.Fire(plot_results)
