"""Compute macro-averaged metrics over real fault labels only.

Strategy:
1. Collect the "true label set" = all labels that appear as true_fault in predictions.csv
   (this excludes hallucination-only labels)
2. Compute per-class TP/FP/FN only for labels in the true label set
3. Macro-average P/R/F1 = simple mean across classes
4. Also compute "fault-only" accuracy (excluding Normal/None samples)
5. Per-equipment breakdown for hybrid
6. Parameter sensitivity analysis across thresholds
"""

import csv
import json
import random
import sys
import os
from collections import defaultdict
from pathlib import Path

sys.path.append(os.path.dirname(__file__))
from knowledge_base import KnowledgeBase
from inference_engine import ProbabilisticInferenceEngine
from llm_baseline import SimulatedLLM
from data_generator import SyntheticDataGenerator

OUTPUT_DIR = Path(__file__).parent / "experiment_outputs"
CSV_PATH = OUTPUT_DIR / "predictions.csv"


def _normalize(val):
    """Treat empty string and 'None' as None, otherwise return string."""
    if val is None or val == "" or val == "None":
        return None
    return val


def _macro_prf1(records, pred_col):
    """Compute macro-averaged P/R/F1 over the true-label set."""
    # Build true label set
    true_labels = set()
    for r in records:
        tf = _normalize(r["true_fault"])
        true_labels.add(tf)

    # Per-class TP/FP/FN
    class_stats = {lab: {"tp": 0, "fp": 0, "fn": 0} for lab in true_labels}
    for r in records:
        tf = _normalize(r["true_fault"])
        pf = _normalize(r[pred_col])
        if tf == pf:
            class_stats[tf]["tp"] += 1
        else:
            class_stats[tf]["fn"] += 1
            if pf in class_stats:
                class_stats[pf]["fp"] += 1
            # If pf is a hallucination label not in true_labels, we ignore it
            # for macro averaging (it doesn't inflate any class's FP)

    precisions, recalls, f1s = [], [], []
    for lab in true_labels:
        s = class_stats[lab]
        prec = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) > 0 else 0.0
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    n = len(true_labels)
    return {
        "precision": round(sum(precisions) / n, 3) if n else 0.0,
        "recall": round(sum(recalls) / n, 3) if n else 0.0,
        "f1": round(sum(f1s) / n, 3) if n else 0.0,
    }


def compute_overall_metrics():
    """Compute overall and fault-only accuracy + macro P/R/F1 for each method."""
    records = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    methods = {"expert": "expert_fault", "llm": "llm_fault", "hybrid": "hybrid_fault"}
    results = {}

    for method, col in methods.items():
        total = len(records)
        correct = sum(1 for r in records if _normalize(r["true_fault"]) == _normalize(r[col]))
        accuracy = round(correct / total, 3)

        # Fault-only accuracy (exclude samples where true_fault is None)
        fault_recs = [r for r in records if _normalize(r["true_fault"]) is not None]
        fault_correct = sum(1 for r in fault_recs if _normalize(r["true_fault"]) == _normalize(r[col]))
        fault_acc = round(fault_correct / len(fault_recs), 3) if fault_recs else 0.0

        # Macro P/R/F1
        macro = _macro_prf1(records, col)

        results[method] = {
            "accuracy": accuracy,
            "fault_only_accuracy": fault_acc,
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],
        }

    return results


def compute_per_equipment(pred_col="hybrid_fault"):
    """Per-equipment macro P/R/F1 for hybrid."""
    records = []
    with CSV_PATH.open("r", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    equipment_types = sorted(set(r["equipment"] for r in records))
    results = {}

    for eq in equipment_types:
        eq_recs = [r for r in records if r["equipment"] == eq]
        total = len(eq_recs)
        correct = sum(1 for r in eq_recs if _normalize(r["true_fault"]) == _normalize(r[pred_col]))
        accuracy = round(correct / total, 3)
        macro = _macro_prf1(eq_recs, pred_col)

        results[eq] = {
            "accuracy": accuracy,
            "precision": macro["precision"],
            "recall": macro["recall"],
            "f1": macro["f1"],
        }

    return results


def run_sensitivity(thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Parameter sensitivity analysis across confidence thresholds."""
    random.seed(42)
    kb = KnowledgeBase()
    expert_engine = ProbabilisticInferenceEngine(kb)
    llm = SimulatedLLM(accuracy=0.85, hallucination_rate=0.05)
    generator = SyntheticDataGenerator(samples_per_eq=500, noise_levels=(0.0, 0.05, 0.10))
    dataset = generator.generate()

    # Pre-compute predictions
    preds = []
    for record in dataset:
        eq = record["equipment"]
        params = record["params"]
        true_fault = record["fault"]
        expert_preds = expert_engine.infer(eq, params)
        expert_fault = expert_preds[0][0] if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0
        llm_fault, llm_conf = llm.diagnose(eq, params, ground_truth=true_fault)
        preds.append({
            "true_fault": true_fault,
            "expert_fault": expert_fault,
            "expert_conf": expert_conf,
            "llm_fault": llm_fault,
            "llm_conf": llm_conf,
        })

    results = {}
    for tau in thresholds:
        correct = 0
        total = len(preds)
        fault_total = 0
        fault_correct = 0
        hybrids = []

        for p in preds:
            if p["expert_conf"] >= tau:
                hf = p["expert_fault"]
            elif p["llm_fault"] is not None and p["expert_fault"] is not None:
                ew = 0.7 * p["expert_conf"]
                lw = 0.3 * p["llm_conf"]
                hf = p["expert_fault"] if ew >= lw else p["llm_fault"]
            elif p["llm_fault"] is not None:
                hf = p["llm_fault"]
            else:
                hf = p["expert_fault"]

            hybrids.append(hf)
            if p["true_fault"] == hf:
                correct += 1
            if p["true_fault"] is not None:
                fault_total += 1
                if p["true_fault"] == hf:
                    fault_correct += 1

        accuracy = round(correct / total, 3)
        fault_acc = round(fault_correct / fault_total, 3) if fault_total else 0.0

        # Macro recall
        label_stats = defaultdict(lambda: {"tp": 0, "fn": 0})
        for p, hf in zip(preds, hybrids):
            tf = p["true_fault"]
            if tf == hf:
                label_stats[tf]["tp"] += 1
            else:
                label_stats[tf]["fn"] += 1
        recs = []
        for s in label_stats.values():
            r = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) > 0 else 0.0
            recs.append(r)
        macro_recall = round(sum(recs) / len(recs), 3) if recs else 0.0

        results[tau] = {
            "accuracy": accuracy,
            "fault_only_accuracy": fault_acc,
            "macro_recall": macro_recall,
        }

    return results


def main():
    print("=" * 60)
    print("MACRO-AVERAGED METRICS REPORT")
    print("=" * 60)

    # 1. Overall
    print("\n### Overall Metrics (Macro-Average) ###\n")
    overall = compute_overall_metrics()
    print(f"{'Method':<8} {'Acc':>6} {'FaultAcc':>9} {'M-Prec':>7} {'M-Rec':>6} {'M-F1':>6}")
    print("-" * 46)
    for m in ["expert", "llm", "hybrid"]:
        o = overall[m]
        print(f"{m:<8} {o['accuracy']:>6.3f} {o['fault_only_accuracy']:>9.3f} {o['macro_precision']:>7.3f} {o['macro_recall']:>6.3f} {o['macro_f1']:>6.3f}")

    # 2. Per-equipment
    print("\n### Per-Equipment Metrics (Hybrid, Macro-Average) ###\n")
    eq_met = compute_per_equipment()
    print(f"{'Equipment':<22} {'Acc':>6} {'M-Prec':>7} {'M-Rec':>6} {'M-F1':>6}")
    print("-" * 50)
    for eq in sorted(eq_met):
        m = eq_met[eq]
        print(f"{eq:<22} {m['accuracy']:>6.3f} {m['precision']:>7.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")

    # 3. Parameter sensitivity analysis
    print("\n### Parameter Sensitivity Analysis ###\n")
    sensitivity = run_sensitivity()
    print(f"{'Tau':>5} {'Acc':>6} {'FaultAcc':>9} {'M-Rec':>6}")
    print("-" * 30)
    for tau in sorted(sensitivity):
        r = sensitivity[tau]
        print(f"{tau:>5.1f} {r['accuracy']:>6.3f} {r['fault_only_accuracy']:>9.3f} {r['macro_recall']:>6.3f}")

    # Save
    full = {"overall": overall, "per_equipment": eq_met, "sensitivity": {str(k): v for k, v in sensitivity.items()}}
    out = OUTPUT_DIR / "comprehensive_metrics.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(full, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
