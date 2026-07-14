"""Alpha/beta sensitivity and per-equipment confidence distributions.

Addresses reviewer requests for:
- alpha/beta justification beyond tau-only sensitivity
- per-equipment confidence-score distributions
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_generator import SyntheticDataGenerator
from experiment_metrics import ece, hybrid_decision, summarize
from inference_engine import ProbabilisticInferenceEngine
from llm_baseline import SimulatedLLM


def run_one(dataset: List[Dict[str, Any]], alpha: float, beta: float, tau: float, min_match: float, seed: int) -> Dict[str, Any]:
    random.seed(seed)
    engine = ProbabilisticInferenceEngine(alpha=alpha, beta=beta, min_match=min_match)
    llm = SimulatedLLM(accuracy=0.85, hallucination_rate=0.15, seed=seed)
    records: List[Dict[str, Any]] = []
    for item in dataset:
        equipment = item["equipment"]
        params = item["params"]
        true_fault = item["fault"]
        expert_preds = engine.infer(equipment, params)
        expert_fault = expert_preds[0][0] if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0
        llm_fault, llm_conf = llm.diagnose(equipment, params, ground_truth=true_fault)
        h_fault, h_conf = hybrid_decision(expert_fault, expert_conf, llm_fault, llm_conf, tau)
        records.append(
            {
                "equipment": equipment,
                "true_fault": true_fault,
                "expert_fault": expert_fault,
                "expert_conf": expert_conf,
                "llm_fault": llm_fault,
                "llm_confidence": llm_conf,
                "hybrid_fault": h_fault,
                "hybrid_conf": h_conf,
            }
        )
    return {
        "expert": summarize(records, "expert_fault"),
        "hybrid": summarize(records, "hybrid_fault"),
        "expert_ece": ece(records, "expert_fault", "expert_conf"),
        "hybrid_ece": ece(records, "hybrid_fault", "hybrid_conf"),
        "records": records,
    }


def plot_heatmap(rows: List[Dict[str, Any]], value_key: str, output_path: Path, title: str) -> None:
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="alpha", columns="beta", values=value_key)
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_equipment_confidence(records: List[Dict[str, Any]], output_path: Path) -> None:
    equipment = sorted({r["equipment"] for r in records})
    ncols = 2
    nrows = (len(equipment) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, max(3.5, 3.0 * nrows)), squeeze=False)
    for ax, eq in zip(axes.flatten(), equipment):
        values = [float(r["expert_conf"]) for r in records if r["equipment"] == eq]
        ax.hist(values, bins=[0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0], color="#4C78A8", edgecolor="white")
        ax.axvline(0.6, color="#E45756", linestyle="--", linewidth=1.5, label="tau=0.6")
        ax.set_title(eq)
        ax.set_xlabel("Expert confidence")
        ax.set_ylabel("Count")
    for ax in axes.flatten()[len(equipment):]:
        ax.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    generator = SyntheticDataGenerator(
        samples_per_eq=args.samples_per_equipment,
        noise_levels=tuple(args.noise_levels),
        partial_ratio=args.partial_ratio,
    )
    dataset = generator.generate()
    out_dir = Path(__file__).parent / "experiment_outputs"
    out_dir.mkdir(exist_ok=True)

    rows: List[Dict[str, Any]] = []
    baseline_records: List[Dict[str, Any]] = []
    for alpha in args.alpha_values:
        for beta in args.beta_values:
            result = run_one(dataset, alpha, beta, args.tau, args.min_match, args.seed)
            row = {
                "alpha": alpha,
                "beta": beta,
                "expert_accuracy": result["expert"]["accuracy"],
                "expert_fault_only_accuracy": result["expert"]["fault_only_accuracy"],
                "expert_macro_f1": result["expert"]["macro_f1"],
                "expert_ece": result["expert_ece"],
                "hybrid_accuracy": result["hybrid"]["accuracy"],
                "hybrid_fault_only_accuracy": result["hybrid"]["fault_only_accuracy"],
                "hybrid_macro_f1": result["hybrid"]["macro_f1"],
                "hybrid_ece": result["hybrid_ece"],
            }
            rows.append(row)
            if abs(alpha - args.baseline_alpha) < 1e-9 and abs(beta - args.baseline_beta) < 1e-9:
                baseline_records = result["records"]

    csv_path = out_dir / "alpha_beta_sensitivity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "alpha_beta_sensitivity.json").open("w", encoding="utf-8") as f:
        json.dump({"configuration": vars(args), "results": rows}, f, indent=2)

    plot_heatmap(rows, "hybrid_macro_f1", out_dir / "alpha_beta_hybrid_macro_f1.png", "Hybrid Macro-F1 across alpha/beta")
    plot_heatmap(rows, "hybrid_ece", out_dir / "alpha_beta_hybrid_ece.png", "Hybrid ECE across alpha/beta")
    if baseline_records:
        plot_per_equipment_confidence(baseline_records, out_dir / "per_equipment_confidence_hist.png")

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'alpha_beta_sensitivity.json'}")
    print(f"Wrote {out_dir / 'alpha_beta_hybrid_macro_f1.png'}")
    print(f"Wrote {out_dir / 'alpha_beta_hybrid_ece.png'}")
    if baseline_records:
        print(f"Wrote {out_dir / 'per_equipment_confidence_hist.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run alpha/beta sensitivity and confidence-distribution analysis.")
    parser.add_argument("--samples-per-equipment", type=int, default=300)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    parser.add_argument("--partial-ratio", type=float, default=0.0)
    parser.add_argument("--alpha-values", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.20])
    parser.add_argument("--beta-values", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--baseline-alpha", type=float, default=0.10)
    parser.add_argument("--baseline-beta", type=float, default=0.20)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--min-match", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main()
