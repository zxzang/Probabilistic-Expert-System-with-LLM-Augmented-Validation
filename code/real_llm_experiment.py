"""Run the original synthetic benchmark with a real DeepSeek LLM subset.

The original `experiment_runner.py` remains the canonical synthetic + MockLLM
pipeline. This script is an additive experiment for reviewer requests asking
for a real LLM instead of only a mock baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_generator import SyntheticDataGenerator
from experiment_metrics import hybrid_decision, summarize
from inference_engine import ProbabilisticInferenceEngine
from knowledge_base import KnowledgeBase
from real_llm import RealLLMBaseline


_FAULT_SEPARATOR_CHARS = ("–", "—", "‑", "−", "每")


def clean_fault_label(label: Optional[str]) -> Optional[str]:
    """Return an ASCII-stable fault label for LLM prompts and CSV outputs."""
    if label is None:
        return None
    text = str(label)
    for char in _FAULT_SEPARATOR_CHARS:
        text = text.replace(char, " - ")
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def clean_condition(op: str, value: Any) -> str:
    return f"{op} {value}"


def _build_fault_thresholds(kb: KnowledgeBase, equipment: str) -> Dict[str, List[Tuple[str, Any]]]:
    """Extract per-parameter fault-detection thresholds from KB rules.

    Returns ``{param_name: [(operator, threshold), ...]}`` for all rules
    applicable to *equipment*.  ``>`` thresholds flag abnormally *high*
    readings; ``<`` thresholds flag abnormally *low* readings.
    The thresholds are deduplicated and sorted for readability.
    """
    thresholds: Dict[str, set[Tuple[str, Any]]] = defaultdict(set)
    for rule in kb.get_rules(equipment):
        for param, (op, val) in rule.conditions.items():
            thresholds[param].add((op, val))
    # sort for stable output:  <  thresholds before  >,  tighter thresholds first
    result: Dict[str, List[Tuple[str, Any]]] = {}
    for param in sorted(thresholds):
        pairs = sorted(thresholds[param], key=lambda x: (x[0], x[1]))
        result[param] = pairs
    return result


def _condition_violated(value: Any, op: str, threshold: Any) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    return False


def _condition_direction(op: str) -> str:
    if op in {">", ">="}:
        return "high"
    if op in {"<", "<="}:
        return "low"
    return "specific"


def build_abnormal_summary(
    params: Dict[str, Any],
    thresholds: Dict[str, List[Tuple[str, Any]]],
) -> List[Dict[str, Any]]:
    """Summarize threshold violations for the current continuous sample."""
    summary: List[Dict[str, Any]] = []
    for param, value in sorted(params.items()):
        violations = []
        for op, threshold in thresholds.get(param, []):
            if _condition_violated(value, op, threshold):
                violations.append(
                    {
                        "condition": clean_condition(op, threshold),
                        "direction": _condition_direction(op),
                    }
                )
        if violations:
            summary.append(
                {
                    "parameter": param,
                    "value": round(float(value), 4) if isinstance(value, (int, float)) else value,
                    "violations": violations,
                }
            )
    return summary


def build_fault_pattern_guide(kb: KnowledgeBase, equipment: str) -> str:
    """Build a compact rule-pattern guide from the synthetic KB."""
    lines = [f"Fault-pattern guide for {equipment}:"]
    for rule in kb.get_rules(equipment):
        conditions = " + ".join(
            f"{param} {clean_condition(op, value)}"
            for param, (op, value) in sorted(rule.conditions.items())
        )
        lines.append(
            f"  - {clean_fault_label(rule.fault)} <- {conditions} "
            f"(priority={rule.priority}, base_confidence={rule.confidence:.2f})"
        )
    lines.extend(
        [
            "",
            "Use this guide to identify the most specific matching fault pattern.",
            "When several patterns share symptoms, prefer the pattern whose anchor condition is most direct.",
            "Priority 1 is stronger than priority 2 when both patterns are otherwise plausible.",
            "Single-parameter patterns can be decisive when that parameter is the named anchor of the fault.",
        ]
    )
    return "\n".join(lines)


def build_threshold_context(equipment: str, thresholds: Dict[str, List[Tuple[str, Any]]]) -> str:
    lines = [
        f"  {param}:  {' , '.join(clean_condition(op, val) for op, val in pairs)}"
        for param, pairs in sorted(thresholds.items())
    ]
    return (
        f"Fault-detection thresholds for {equipment}:\n"
        + "\n".join(lines)
        + "\n\nCompare the sensor readings against these thresholds. "
        "A reading that violates a '>' threshold is abnormally HIGH. "
        "A reading that violates a '<' threshold is abnormally LOW. "
        "If one parameter crosses multiple thresholds, count it as one "
        "abnormal parameter and use the stricter crossed threshold as "
        "severity evidence. "
        "Identify which parameters are abnormal, then diagnose the most "
        "specific matching fault from the candidate list. "
        "Calibrate your confidence:\n"
        "  - 1 abnormal parameter -> low confidence (0.3-0.5)\n"
        "  - 2 abnormal parameters -> moderate confidence (0.5-0.7)\n"
        "  - 3+ abnormal parameters matching a known fault pattern -> high confidence (0.7-0.9)\n"
        "If NO parameters violate any threshold, answer 'Normal' (>= 0.85)."
    )


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    kb = KnowledgeBase()
    engine = ProbabilisticInferenceEngine(kb=kb)
    llm = RealLLMBaseline()
    generator = SyntheticDataGenerator(
        samples_per_eq=args.samples_per_equipment,
        noise_levels=tuple(args.noise_levels),
        partial_ratio=args.partial_ratio,
    )
    dataset = generator.generate()
    if args.fault_only:
        dataset = [r for r in dataset if r["fault"] is not None]
    random.shuffle(dataset)
    dataset = dataset[: args.sample_size]

    # ---- pre-build prompt contexts once per equipment type ----
    equipment_contexts: Dict[str, str] = {}
    equipment_thresholds: Dict[str, Dict[str, List[Tuple[str, Any]]]] = {}
    for eq in kb.equipment_types:
        thresholds = _build_fault_thresholds(kb, eq)
        equipment_thresholds[eq] = thresholds
        threshold_context = build_threshold_context(eq, thresholds)
        if args.prompt_mode == "threshold-only":
            equipment_contexts[eq] = threshold_context
        else:
            equipment_contexts[eq] = threshold_context + "\n\n" + build_fault_pattern_guide(kb, eq)

    records: List[Dict[str, Any]] = []
    for idx, record in enumerate(dataset, start=1):
        equipment = record["equipment"]
        params = record["params"]
        true_fault = clean_fault_label(record["fault"])
        expert_preds = engine.infer(equipment, params)
        expert_fault = clean_fault_label(expert_preds[0][0]) if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0
        candidate_faults = [clean_fault_label(rule.fault) for rule in kb.get_rules(equipment)]
        abnormal_summary = (
            build_abnormal_summary(params, equipment_thresholds[equipment])
            if args.prompt_mode == "rule-guided"
            else None
        )

        context = equipment_contexts[equipment]
        llm_fault, llm_conf, meta = llm.diagnose(
            equipment=equipment,
            params=params,
            candidate_faults=candidate_faults,
            context=context,
            abnormal_summary=abnormal_summary,
        )
        hybrid_fault, hybrid_conf = hybrid_decision(expert_fault, expert_conf, llm_fault, llm_conf, args.tau)
        records.append(
            {
                "idx": idx,
                "equipment": equipment,
                "true_fault": true_fault,
                "sample_type": record.get("sample_type"),
                "expert_fault": expert_fault,
                "expert_conf": round(expert_conf, 3),
                "real_llm_fault": llm_fault,
                "real_llm_confidence": llm_conf,
                "hybrid_fault": hybrid_fault,
                "hybrid_conf": round(hybrid_conf, 3),
                "prompt_mode": args.prompt_mode,
                "abnormal_summary": json.dumps(abnormal_summary or [], ensure_ascii=False),
                "llm_error": meta.get("error", ""),
                "llm_rationale": meta.get("rationale", ""),
            }
        )
        if args.progress_every and idx % args.progress_every == 0:
            print(f"Processed {idx}/{len(dataset)} real-LLM calls")

    metrics = {
        "configuration": vars(args),
        "total_samples": len(records),
        "expert": summarize(records, "expert_fault"),
        "real_llm": summarize(records, "real_llm_fault"),
        "hybrid": summarize(records, "hybrid_fault"),
        "llm_errors": sum(1 for r in records if r["llm_error"]),
    }

    out_dir = Path(__file__).parent / "experiment_outputs"
    out_dir.mkdir(exist_ok=True)
    pred_path = out_dir / "real_llm_synthetic_predictions.csv"
    metrics_path = out_dir / "real_llm_synthetic_metrics.json"
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["idx"])
        writer.writeheader()
        writer.writerows(records)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Predictions written to {pred_path}")
    print(f"Metrics written to {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real DeepSeek LLM subset on synthetic samples.")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--samples-per-equipment", type=int, default=120)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    parser.add_argument("--partial-ratio", type=float, default=0.0)
    parser.add_argument("--fault-only", action="store_true")
    parser.add_argument("--tau", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--prompt-mode",
        choices=["threshold-only", "rule-guided"],
        default="rule-guided",
        help="Synthetic real-LLM prompt context. rule-guided adds KB fault patterns and an abnormal summary.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
