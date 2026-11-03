# ABOUTME: Plots audit results (accuracy vs layer) for all methods in a given directory.
# ABOUTME: Supports averaging results across multiple directories (e.g., different words).

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


def average_results(
    all_results: list[dict[str, dict[int, dict]]],
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """Average results across multiple directories.

    Combines within-directory std and between-directory std in quadrature.

    Returns: {method_name: (layers, mean_accuracies, combined_stds)}
    """
    all_methods = set()
    for res in all_results:
        all_methods.update(res.keys())

    averaged = {}
    for method in sorted(all_methods):
        all_layers = set()
        for res in all_results:
            if method in res:
                all_layers.update(res[method].keys())

        layers = sorted(all_layers)
        mean_accs = []
        combined_stds = []

        for layer in layers:
            accs = []
            stds = []
            for res in all_results:
                if method in res and layer in res[method]:
                    accs.append(res[method][layer]["accuracy"])
                    stds.append(res[method][layer]["std"])

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
    results_dirs: list[Path],
) -> dict[str, tuple[list[int], list[float], list[float]]]:
    """Collect and optionally average results from one or more directories."""
    all_results = []
    for d in results_dirs:
        if not d.exists():
            print(f"Warning: Directory does not exist: {d}")
            continue
        print(f"Loading results from: {d}")
        all_results.append(collect_single_dir_results(d))

    if not all_results:
        return {}

    if len(all_results) == 1:
        # Single directory - convert to expected format
        res = all_results[0]
        method_results = {}
        for method, layer_data in res.items():
            layers = sorted(layer_data.keys())
            accs = [layer_data[l]["accuracy"] for l in layers]
            stds = [layer_data[l]["std"] for l in layers]
            method_results[method] = (layers, accs, stds)
        return method_results

    # Multiple directories - average
    return average_results(all_results)


def plot_results(
    *results_dirs: str, output_path: str | None = None, title: str | None = None
):
    """Plot accuracy vs layer for all methods.

    Args:
        *results_dirs: One or more audit result directories. When multiple
            directories are provided, results are averaged across them.
        output_path: Optional output path for the plot.
        title: Optional title for the plot.
    """
    if not results_dirs:
        print("Error: At least one results directory required")
        return

    dirs = [Path(d) for d in results_dirs]
    method_results = collect_results(dirs)

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
        if len(dirs) == 1:
            title = f"Accuracy by Layer\n{dirs[0].name}"
        else:
            title = f"Accuracy by Layer (averaged over {len(dirs)} directories)"
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
        out = dirs[0] / "accuracy_by_layer.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches="tight")
        print(f"Plot saved to {out}")

    plt.close()


if __name__ == "__main__":
    import fire

    fire.Fire(plot_results)
