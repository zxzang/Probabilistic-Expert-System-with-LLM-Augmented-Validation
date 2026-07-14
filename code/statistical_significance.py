"""Repeated-seed statistical robustness experiment.

Runs the synthetic + MockLLM hybrid pipeline over multiple random seeds and
reports mean, standard deviation, and 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_generator import SyntheticDataGenerator
from experiment_metrics import ece, hybrid_decision, mean_std_ci, summarize
from inference_engine import ProbabilisticInferenceEngine
from llm_baseline import SimulatedLLM


def run_seed(seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(seed)
    generator = SyntheticDataGenerator(
        samples_per_eq=args.samples_per_equipment,
        noise_levels=tuple(args.noise_levels),
        partial_ratio=args.partial_ratio,
    )
    dataset = generator.generate()
    engine = ProbabilisticInferenceEngine(alpha=args.alpha, beta=args.beta, min_match=args.min_match)
    llm = SimulatedLLM(accuracy=args.llm_accuracy, hallucination_rate=args.hallucination_rate, seed=seed)
    records: List[Dict[str, Any]] = []
    for item in dataset:
        equipment = item["equipment"]
        params = item["params"]
        true_fault = item["fault"]
        expert_preds = engine.infer(equipment, params)
        expert_fault = expert_preds[0][0] if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0
        llm_fault, llm_conf = llm.diagnose(equipment, params, ground_truth=true_fault)
        h_fault, h_conf = hybrid_decision(expert_fault, expert_conf, llm_fault, llm_conf, args.tau)
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

    expert = summarize(records, "expert_fault")
    llm_metrics = summarize(records, "llm_fault")
    hybrid = summarize(records, "hybrid_fault")
    return {
        "seed": seed,
        "samples": len(records),
        "expert_accuracy": expert["accuracy"],
        "expert_fault_only_accuracy": expert["fault_only_accuracy"],
        "expert_macro_f1": expert["macro_f1"],
        "expert_ece": ece(records, "expert_fault", "expert_conf"),
        "llm_accuracy": llm_metrics["accuracy"],
        "llm_fault_only_accuracy": llm_metrics["fault_only_accuracy"],
        "llm_macro_f1": llm_metrics["macro_f1"],
        "hybrid_accuracy": hybrid["accuracy"],
        "hybrid_fault_only_accuracy": hybrid["fault_only_accuracy"],
        "hybrid_macro_f1": hybrid["macro_f1"],
        "hybrid_ece": ece(records, "hybrid_fault", "hybrid_conf"),
    }


def main() -> None:
    args = parse_args()
    rows = [run_seed(seed, args) for seed in args.seeds]
    metric_keys = [k for k in rows[0].keys() if k not in {"seed", "samples"}]
    summary = {key: mean_std_ci(row[key] for row in rows) for key in metric_keys}
    payload = {"configuration": vars(args), "per_seed": rows, "summary": summary}

    out_dir = Path(__file__).parent / "experiment_outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "statistical_significance_runs.csv"
    json_path = out_dir / "statistical_significance_summary.json"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed statistical robustness analysis.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55, 66, 77, 88, 99, 110])
    parser.add_argument("--samples-per-equipment", type=int, default=300)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    parser.add_argument("--partial-ratio", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--beta", type=float, default=0.20)
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--min-match", type=float, default=75.0)
    parser.add_argument("--llm-accuracy", type=float, default=0.85)
    parser.add_argument("--hallucination-rate", type=float, default=0.15)
    return parser.parse_args()


if __name__ == "__main__":
    main()
