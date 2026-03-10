# plot_results.py
"""Visualization utilities for the Probabilistic Expert Systems experiment.

This module provides functions to generate common figures from the
`metrics.json` and `predictions.csv` files produced by ``experiment_runner.py``.
It uses ``matplotlib`` (and ``seaborn`` for nicer styles) to create:

- Coverage bar chart per equipment type
- Accuracy comparison bar chart (Expert, LLM, Hybrid)
- Reliability diagram for confidence calibration (optional)

The generated plots are saved as PNG files in the ``experiment_outputs``
directory.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure a consistent style
sns.set(style="whitegrid")

OUTPUT_DIR = Path(__file__).parent / "experiment_outputs"


def load_metrics() -> Dict:
    """Load the metrics JSON file produced by the experiment runner."""
    metrics_path = OUTPUT_DIR / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_coverage(metrics: Dict) -> None:
    """Create a bar chart of rule coverage percentages per equipment type.

    The figure is saved as ``coverage.png`` in the output directory.
    """
    coverage = metrics.get("coverage", {})
    if not coverage:
        raise ValueError("Coverage data missing in metrics.")
    equipment = list(coverage.keys())
    percentages = [coverage[e] for e in equipment]

    plt.figure(figsize=(8, 5))
    sns.barplot(x=equipment, y=percentages, palette="viridis")
    plt.ylabel("Coverage (%)")
    plt.title("Rule Coverage per Equipment Type")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    (OUTPUT_DIR / "coverage.png").write_bytes(b"")  # placeholder to ensure path exists
    plt.savefig(OUTPUT_DIR / "coverage.png", dpi=300)
    plt.close()


def plot_accuracy(metrics: Dict) -> None:
    """Bar chart comparing Expert, LLM and Hybrid accuracies.

    Saved as ``accuracy_comparison.png``.
    """
    acc = metrics.get("accuracy", {})
    if not acc:
        raise ValueError("Accuracy data missing in metrics.")
    methods = list(acc.keys())
    values = [acc[m] for m in methods]

    plt.figure(figsize=(6, 4))
    sns.barplot(x=methods, y=values, palette="muted")
    plt.ylabel("Accuracy")
    plt.title("Diagnostic Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=300)
    plt.close()


def load_predictions() -> List[Dict[str, str]]:
    """Load raw predictions CSV for optional further analysis.

    Returns a list of dictionaries where keys correspond to the CSV header.
    """
    csv_path = OUTPUT_DIR / "predictions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    """Generate all standard figures.

    This function is intended to be called from the command line:
    ``python plot_results.py``
    """
    metrics = load_metrics()
    plot_coverage(metrics)
    plot_accuracy(metrics)
    # Additional visualisations (e.g., reliability diagram) can be added here.
    print(f"Plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
