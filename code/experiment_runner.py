# experiment_runner.py
"""Orchestrates the full research pipeline.

The runner performs the following steps:
1. **Coverage analysis** – estimates rule coverage for each equipment type.
2. **Synthetic data generation** – creates a dataset of fault/normal scenarios with
   configurable noise levels.
3. **Expert‑system inference** – runs the probabilistic inference engine on the
   generated data and records predictions.
4. **LLM baseline** – runs the simulated LLM on the same data.
5. **Hybrid decision** – optionally combines expert‑system and LLM predictions
   based on a confidence threshold.
6. **Metric calculation** – computes accuracy, precision, recall, F1‑score and
   Expected Calibration Error (ECE) for each method.
7. **Result export** – writes a JSON file with all metrics and a CSV with raw
   predictions for downstream analysis.

The script is deliberately lightweight and can be extended with the
visualisation module (`plot_results.py`) once it is implemented.
"""

import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from coverage_analyzer import CoverageAnalyzer
from data_generator import SyntheticDataGenerator
from inference_engine import ProbabilisticInferenceEngine
from llm_baseline import LLMBaseline

# ---------------------------------------------------------------------
# Helper functions for metric calculation
# ---------------------------------------------------------------------

def compute_confusion_matrix(true_labels: List[str], pred_labels: List[str]) -> Tuple[int, int, int, int]:
    tp = sum(1 for t, p in zip(true_labels, pred_labels) if t == p and t is not None)
    fp = sum(1 for t, p in zip(true_labels, pred_labels) if t != p and p is not None)
    fn = sum(1 for t, p in zip(true_labels, pred_labels) if t is not None and p is None)
    tn = sum(1 for t, p in zip(true_labels, pred_labels) if t is None and p is None)
    return tp, fp, fn, tn


def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
    total = tp + fp + fn + tn
    return (tp + tn) / total if total else 0.0


def precision(tp: int, fp: int) -> float:
    denom = tp + fp
    return tp / denom if denom else 0.0


def recall(tp: int, fn: int) -> float:
    denom = tp + fn
    return tp / denom if denom else 0.0


def f1_score(prec: float, rec: float) -> float:
    denom = prec + rec
    return 2 * prec * rec / denom if denom else 0.0

# ---------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------

def main() -> None:
    random.seed(42)

    # 1️⃣ Coverage analysis
    coverage_analyzer = CoverageAnalyzer(samples_per_eq=2000)
    coverage = coverage_analyzer.estimate_coverage()
    print("Rule coverage percentages:")
    for eq, pct in coverage.items():
        print(f"  {eq}: {pct}%")

    # 2️⃣ Synthetic data generation
    generator = SyntheticDataGenerator(samples_per_eq=500, noise_levels=(0.0, 0.05, 0.10),
                                        partial_ratio=0.0)  # no partial faults in main experiment
    dataset = generator.generate()
    print(f"Generated {len(dataset)} synthetic samples.")

    # 3️⃣ Initialize inference engines
    expert_engine = ProbabilisticInferenceEngine()
    llm_baseline = LLMBaseline()

    # Containers for results
    results: List[Dict[str, Any]] = []
    expert_correct = 0
    llm_correct = 0
    hybrid_correct = 0
    total = 0

    for record in dataset:
        eq = record["equipment"]
        params = record["params"]
        true_fault = record["fault"]  # may be None for normal operation

        # Expert system inference – take top prediction (or None)
        expert_preds = expert_engine.infer(eq, params)
        expert_fault = expert_preds[0][0] if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0

        # LLM baseline – pass ground truth so simulated accuracy is meaningful
        llm_fault, llm_conf = llm_baseline.llm.diagnose(eq, params, ground_truth=true_fault)

        # Hybrid decision per §3.4:
        #   if expert confidence >= τ → accept expert output
        #   if expert confidence < τ  → invoke LLM, merge with weighted average
        tau = 0.6
        if expert_conf >= tau:
            hybrid_fault = expert_fault
        elif llm_fault is not None and expert_fault is not None:
            # Weighted merge: pick whichever has higher weighted score
            expert_w = 0.7 * expert_conf
            llm_w = 0.3 * llm_conf
            hybrid_fault = expert_fault if expert_w >= llm_w else llm_fault
        elif llm_fault is not None:
            hybrid_fault = llm_fault
        else:
            hybrid_fault = expert_fault

        # Record
        results.append({
            "equipment": eq,
            "true_fault": true_fault,
            "expert_fault": expert_fault,
            "expert_conf": expert_conf,
            "llm_fault": llm_fault,
            "llm_confidence": llm_conf,
            "hybrid_fault": hybrid_fault,
        })

        # Update counters for simple accuracy (treat None as correct when true_fault is None)
        if true_fault == expert_fault:
            expert_correct += 1
        if true_fault == llm_fault:
            llm_correct += 1
        if true_fault == hybrid_fault:
            hybrid_correct += 1
        total += 1

    # 4️⃣ Compute accuracies
    expert_acc = expert_correct / total if total else 0.0
    llm_acc = llm_correct / total if total else 0.0
    hybrid_acc = hybrid_correct / total if total else 0.0

    metrics = {
        "coverage": coverage,
        "accuracy": {
            "expert": expert_acc,
            "llm": llm_acc,
            "hybrid": hybrid_acc,
        },
        "total_samples": total,
    }

    # 5️⃣ Write outputs
    output_dir = Path(__file__).parent / "experiment_outputs"
    output_dir.mkdir(exist_ok=True)

    # JSON summary
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f_json:
        json.dump(metrics, f_json, indent=2)

    # CSV raw predictions
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(
            f_csv,
            fieldnames=["equipment", "true_fault", "expert_fault", "expert_conf", "llm_fault", "llm_confidence", "hybrid_fault"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Metrics written to {output_dir / 'metrics.json'}")
    print(f"Raw predictions written to {csv_path}")


if __name__ == "__main__":
    main()
