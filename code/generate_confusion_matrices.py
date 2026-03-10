"""Generate confusion matrix plots for each equipment type.

Reads predictions.csv produced by experiment_runner.py, computes per-equipment
confusion matrices (hybrid predictions vs ground truth), and saves heatmap PNGs
into experiment_outputs/.
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "experiment_outputs"
CSV_PATH = OUTPUT_DIR / "predictions.csv"

EQUIPMENT_TYPES = [
    "Boiler",
    "Air Compressor",
    "Chiller",
    "HVAC",
    "Vacuum Machine",
    "Power Distribution",
    "Water Supply",
]


def load_predictions():
    """Load predictions CSV and return list of dicts."""
    with CSV_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def build_confusion_matrix(records, eq_type):
    """Build a confusion matrix for a specific equipment type.

    Uses hybrid_fault as the predicted label and true_fault as the ground truth.
    Returns (labels_sorted, matrix_2d_array).
    """
    # Filter records for this equipment
    eq_records = [r for r in records if r["equipment"] == eq_type]
    if not eq_records:
        return None, None

    # Collect unique labels (combine true and predicted)
    all_labels = set()
    for r in eq_records:
        true_f = r["true_fault"] if r["true_fault"] else "Normal"
        pred_f = r["hybrid_fault"] if r["hybrid_fault"] else "Normal"
        all_labels.add(true_f)
        all_labels.add(pred_f)

    labels = sorted(all_labels)
    label_idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)

    for r in eq_records:
        true_f = r["true_fault"] if r["true_fault"] else "Normal"
        pred_f = r["hybrid_fault"] if r["hybrid_fault"] else "Normal"
        matrix[label_idx[true_f], label_idx[pred_f]] += 1

    return labels, matrix


def plot_confusion(labels, matrix, eq_type):
    """Plot and save a confusion matrix heatmap."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))

    im = ax.imshow(matrix, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    # Shorten long labels for display
    short_labels = [lab[:25] + "..." if len(lab) > 28 else lab for lab in labels]
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_labels, fontsize=8)

    ax.set_xlabel("Predicted Fault")
    ax.set_ylabel("True Fault")
    ax.set_title(f"Confusion Matrix – {eq_type}")

    # Annotate cells with counts
    thresh = matrix.max() / 2.0
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] > thresh else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color=color, fontsize=8)

    plt.tight_layout()

    # Sanitize equipment name for filename
    safe_name = eq_type.lower().replace(" ", "")
    out_path = OUTPUT_DIR / f"confusion_{safe_name}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    print("Loading predictions...")
    records = load_predictions()
    print(f"Loaded {len(records)} records.")

    for eq in EQUIPMENT_TYPES:
        print(f"\nProcessing: {eq}")
        labels, matrix = build_confusion_matrix(records, eq)
        if labels is None:
            print(f"  WARNING: No records found for {eq}, skipping.")
            continue
        print(f"  Labels ({len(labels)}): {labels}")
        plot_confusion(labels, matrix, eq)

    print("\nAll confusion matrices generated successfully!")


if __name__ == "__main__":
    main()
