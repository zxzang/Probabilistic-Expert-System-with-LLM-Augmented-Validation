# partial_fault_experiment.py
"""Incipient fault detection experiment: Expert-only vs Hybrid framework.

This script evaluates the system's ability to detect **incipient (partial)
faults**—scenarios where only a subset of rule conditions are abnormal,
simulating early-stage fault development.

The experiment generates a dataset containing:
  - Normal samples (full-match ground truth = None)
  - Full-fault samples (all conditions met)
  - Partial-fault samples (1 condition deliberately suppressed)

It then compares Expert-only and Hybrid (Expert + LLM fallback) detection
rates at multiple min_match activation thresholds.

Outputs:
    - Console table with detection rates
    - experiment_outputs/partial_fault_results.json
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_generator import SyntheticDataGenerator
from experiment_metrics import hybrid_decision
from inference_engine import ProbabilisticInferenceEngine
from llm_baseline import LLMBaseline


def run_experiment(dataset: List[Dict[str, Any]], min_match: float,
                   tau: float = 0.6) -> Dict[str, Any]:
    """Run Expert-only and Hybrid on *dataset* with given *min_match*.

    Returns per-sample-type metrics for both Expert and Hybrid.
    """
    engine = ProbabilisticInferenceEngine(min_match=min_match)
    llm = LLMBaseline(seed=42)

    counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"expert_ok": 0, "hybrid_ok": 0, "total": 0}
    )

    for rec in dataset:
        eq = rec["equipment"]
        params = rec["params"]
        true_fault = rec["fault"]
        stype = rec["sample_type"]

        # Expert inference
        preds = engine.infer(eq, params)
        expert_fault = preds[0][0] if preds else None
        expert_conf = preds[0][1] if preds else 0.0

        # Mock LLM baseline
        llm_fault, llm_conf = llm.diagnose(eq, params, ground_truth=true_fault)

        hybrid_fault, _hybrid_conf = hybrid_decision(expert_fault, expert_conf, llm_fault, llm_conf, tau=tau)

        counts[stype]["total"] += 1
        if expert_fault == true_fault:
            counts[stype]["expert_ok"] += 1
        if hybrid_fault == true_fault:
            counts[stype]["hybrid_ok"] += 1

    results = {}
    for stype, c in counts.items():
        n = c["total"]
        results[stype] = {
            "total": n,
            "expert_correct": c["expert_ok"],
            "hybrid_correct": c["hybrid_ok"],
            "expert_acc": round(c["expert_ok"] / n, 4) if n else 0.0,
            "hybrid_acc": round(c["hybrid_ok"] / n, 4) if n else 0.0,
        }
    te = sum(c["expert_ok"] for c in counts.values())
    th = sum(c["hybrid_ok"] for c in counts.values())
    tt = sum(c["total"] for c in counts.values())
    results["overall"] = {
        "total": tt,
        "expert_correct": te,
        "hybrid_correct": th,
        "expert_acc": round(te / tt, 4) if tt else 0.0,
        "hybrid_acc": round(th / tt, 4) if tt else 0.0,
    }
    return results


def main() -> None:
    random.seed(42)

    # Generate dataset with partial faults (20 % partial ratio)
    generator = SyntheticDataGenerator(
        samples_per_eq=500,
        noise_levels=(0.0, 0.05, 0.10),
        partial_ratio=0.2,
    )
    dataset = generator.generate()

    # Summary
    type_counts = defaultdict(int)
    for r in dataset:
        type_counts[r["sample_type"]] += 1
    print(f"Dataset: {len(dataset)} samples")
    for stype, cnt in sorted(type_counts.items()):
        print(f"  {stype}: {cnt}")
    print()

    # Run at multiple thresholds
    thresholds = [
        ("Boolean (100%)", 100.0),
        ("PMS-75%",         75.0),
        ("PMS-50%",         50.0),
    ]

    all_results = {}
    for label, mm in thresholds:
        all_results[label] = run_experiment(dataset, mm)

    # ── Print table ──────────────────────────────────────────────
    print("=" * 88)
    print("INCIPIENT FAULT DETECTION — EXPERT vs HYBRID COMPARISON")
    print("=" * 88)
    print()
    print(f"{'min_match':<18} {'Method':<10}",
          f"{'Normal':>10} {'Fault':>10} {'Partial':>10} {'Overall':>10}")
    print("-" * 78)

    for label, _ in thresholds:
        r = all_results[label]
        for method, key in [("Expert", "expert_acc"), ("Hybrid", "hybrid_acc")]:
            vals = [r.get(st, {}).get(key, 0.0)
                    for st in ["normal", "fault", "partial_fault", "overall"]]
            print(f"{label:<18} {method:<10}",
                  "".join(f" {v:>10.4f}" for v in vals))
        print()

    # ── Key insight ──────────────────────────────────────────────
    r75 = all_results["PMS-75%"]
    pf_expert = r75["partial_fault"]["expert_acc"]
    pf_hybrid = r75["partial_fault"]["hybrid_acc"]
    print("=" * 88)
    print("KEY RESULT (PMS-75%)")
    pft = r75["partial_fault"]
    print(f"  Expert-only partial fault detection:  {pf_expert:.2%}  "
          f"({pft['expert_correct']}/{pft['total']})")
    print(f"  Hybrid partial fault detection:       {pf_hybrid:.2%}  "
          f"({pft['hybrid_correct']}/{pft['total']})")
    print(f"  Hybrid improvement over Expert:       "
          f"+{(pf_hybrid - pf_expert) * 100:.1f} pp")
    print("=" * 88)

    # Save
    output_dir = Path(__file__).parent / "experiment_outputs"
    output_dir.mkdir(exist_ok=True)
    out_path = output_dir / "partial_fault_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
