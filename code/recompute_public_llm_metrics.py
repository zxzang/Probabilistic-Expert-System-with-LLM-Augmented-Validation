"""Recompute public Real-LLM metrics without making API calls.

The public Real-LLM prediction CSVs contain the expensive API outputs
(`llm_fault`, `llm_confidence`, and self-check fields).  When the public PMS
rule base changes but the public samples and LLM prompt evidence do not, this
helper rebuilds the public samples, re-runs the current expert/hybrid logic,
and rewrites the prediction/metrics files while preserving the recorded LLM
outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from experiment_metrics import ece, hybrid_decision
from inference_engine import ProbabilisticInferenceEngine
from public_data_benchmark import (
    PublicBenchmarkKnowledgeBase,
    arbitration_action_metrics,
    build_lbnl_samples,
    build_metropt_samples,
    coverage_min_match_by_equipment,
    evidence_strength_from_params,
    expert_min_match_sensitivity,
    fallback_rate,
    metropt_time_to_detection,
    parse_float_grid,
    predicate_coverage,
    resolve_thresholds,
    select_hybrid_decision,
    summarize,
    tau_sensitivity,
    threshold_profile_summary,
    threshold_usage,
)


DEFAULT_PROMPT_MODES = ["kb-guided", "evidence-rich-kb", "rich-context"]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _same_sample(row: Dict[str, str], sample: Any) -> bool:
    row_params = json.loads(row.get("params", "{}"))
    return (
        row.get("source") == sample.source
        and row.get("equipment") == sample.equipment
        and row.get("scenario") == sample.scenario
        and (row.get("true_fault") or None) == sample.true_fault
        and row_params == sample.params
    )


def build_samples(args: argparse.Namespace) -> List[Any]:
    samples = build_lbnl_samples(args.max_rows, args.stride, args.max_scenarios_per_source)
    if not args.skip_metropt:
        samples.extend(build_metropt_samples(args.metropt_window_rows, args.metropt_max_windows_per_class))
    return samples


def rebuild_records(
    args: argparse.Namespace,
    prompt_mode: str,
    old_rows: List[Dict[str, str]],
    samples: List[Any],
) -> List[Dict[str, Any]]:
    if len(old_rows) != len(samples):
        raise ValueError(f"{prompt_mode}: row count mismatch: csv={len(old_rows)} samples={len(samples)}")

    kb = PublicBenchmarkKnowledgeBase()
    engine_cache: Dict[float, ProbabilisticInferenceEngine] = {}
    records: List[Dict[str, Any]] = []
    for idx, (row, sample) in enumerate(zip(old_rows, samples)):
        if not _same_sample(row, sample):
            raise ValueError(f"{prompt_mode}: sample mismatch at row {idx}: {row.get('scenario')} vs {sample.scenario}")

        thresholds = resolve_thresholds(args, sample)
        if thresholds.min_match not in engine_cache:
            engine_cache[thresholds.min_match] = ProbabilisticInferenceEngine(
                kb=kb,
                min_match=thresholds.min_match,
                require_abnormal_anchor=True,
                evidence_damping=True,
            )
        engine = engine_cache[thresholds.min_match]
        expert_preds = engine.infer(sample.equipment, sample.params)
        expert_fault = expert_preds[0][0] if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0

        llm_fault = row.get("llm_fault") or None
        llm_conf = _as_float(row.get("llm_confidence"), 0.0)
        nonzero_count, max_level, evidence_strength = evidence_strength_from_params(sample.params)
        legacy_hybrid_fault, legacy_hybrid_conf = hybrid_decision(
            expert_fault,
            expert_conf,
            llm_fault,
            llm_conf,
            tau=thresholds.tau,
        )
        hybrid_fault, hybrid_conf, arbitration_action = select_hybrid_decision(
            args.hybrid_strategy,
            expert_fault,
            expert_conf,
            llm_fault,
            llm_conf,
            tau=thresholds.tau,
            evidence_strength=evidence_strength,
            equipment=sample.equipment,
        )
        fallback_needed = expert_conf < thresholds.tau

        records.append(
            {
                "source": sample.source,
                "equipment": sample.equipment,
                "scenario": sample.scenario,
                "start_time": sample.start_time,
                "end_time": sample.end_time,
                "true_fault": sample.true_fault,
                "params": json.dumps(sample.params, sort_keys=True),
                "effective_min_match": round(thresholds.min_match, 3),
                "effective_tau": round(thresholds.tau, 3),
                "expert_fault": expert_fault,
                "expert_conf": round(expert_conf, 3),
                "llm_fault": llm_fault,
                "llm_confidence": llm_conf,
                "llm_abnormal_decision": row.get("llm_abnormal_decision", ""),
                "llm_evidence_sufficiency": row.get("llm_evidence_sufficiency", ""),
                "llm_error": row.get("llm_error", ""),
                "llm_called": _as_int(row.get("llm_called"), 0),
                "llm_prompt_mode": prompt_mode,
                "fallback_needed": int(fallback_needed),
                "hybrid_strategy": args.hybrid_strategy,
                "legacy_hybrid_fault": legacy_hybrid_fault,
                "legacy_hybrid_conf": round(legacy_hybrid_conf, 3),
                "hybrid_fault": hybrid_fault,
                "hybrid_conf": round(hybrid_conf, 3),
                "arbitration_action": arbitration_action,
                "nonzero_predicate_count": nonzero_count,
                "max_predicate_level": max_level,
                "evidence_strength": evidence_strength,
            }
        )
    return records


def compose_metrics(
    args: argparse.Namespace,
    records: List[Dict[str, Any]],
    samples: List[Any],
    prompt_mode: str,
) -> Dict[str, Any]:
    kb = PublicBenchmarkKnowledgeBase()
    min_match_grid = parse_float_grid(args.min_match_grid)
    tau_grid = parse_float_grid(args.tau_grid)
    llm_evaluated = sum(1 for r in records if int(r.get("llm_called", 0)) == 1)
    llm_evaluated_records = [r for r in records if int(r.get("llm_called", 0)) == 1]
    fallback_llm_records = [
        r for r in records
        if int(r.get("llm_called", 0)) == 1 and int(r.get("fallback_needed", 0)) == 1
    ]
    config = vars(args).copy()
    config.update(
        {
            "llm": "real",
            "llm_prompt_mode": prompt_mode,
            "max_api_calls": None,
            "recomputed_from_predictions": True,
        }
    )
    metrics: Dict[str, Any] = {
        "configuration": config,
        "recomputed_from_predictions": True,
        "recorded_llm_calls_reused": llm_evaluated,
        "threshold_profile": threshold_profile_summary(args),
        "threshold_usage": threshold_usage(records),
        "expert_min_match_sensitivity": expert_min_match_sensitivity(kb, samples, min_match_grid),
        "tau_sensitivity": tau_sensitivity(records, tau_grid, args.hybrid_strategy),
        "total_samples": len(records),
        "api_calls": 0,
        "llm_evaluated_samples": llm_evaluated,
        "llm_evaluation_rate": round(llm_evaluated / len(records), 3) if records else 0.0,
        "fallback_rate": fallback_rate(records),
        "predicate_coverage": predicate_coverage(kb, min_match_by_equipment=coverage_min_match_by_equipment(args, kb)),
        "metropt_time_to_detection": metropt_time_to_detection(records),
        "expert": summarize(records, "expert_fault"),
        "llm": summarize(records, "llm_fault"),
        "llm_evaluated_subset": summarize(llm_evaluated_records, "llm_fault") if llm_evaluated_records else {},
        "llm_on_fallback_needed": summarize(fallback_llm_records, "llm_fault") if fallback_llm_records else {},
        "legacy_hybrid": summarize(records, "legacy_hybrid_fault"),
        "hybrid": summarize(records, "hybrid_fault"),
        "expert_ece": ece(records, "expert_fault", "expert_conf"),
        "llm_ece": ece(records, "llm_fault", "llm_confidence"),
        "legacy_hybrid_ece": ece(records, "legacy_hybrid_fault", "legacy_hybrid_conf"),
        "hybrid_ece": ece(records, "hybrid_fault", "hybrid_conf"),
        "arbitration_actions": arbitration_action_metrics(records),
        "by_source": {},
    }
    for source in sorted({r["source"] for r in records}):
        subset = [r for r in records if r["source"] == source]
        subset_llm = [r for r in subset if int(r.get("llm_called", 0)) == 1]
        subset_fallback_llm = [
            r for r in subset
            if int(r.get("llm_called", 0)) == 1 and int(r.get("fallback_needed", 0)) == 1
        ]
        metrics["by_source"][source] = {
            "samples": len(subset),
            "fallback_rate": fallback_rate(subset),
            "expert": summarize(subset, "expert_fault"),
            "llm": summarize(subset, "llm_fault"),
            "llm_evaluated_subset": summarize(subset_llm, "llm_fault") if subset_llm else {},
            "llm_on_fallback_needed": summarize(subset_fallback_llm, "llm_fault") if subset_fallback_llm else {},
            "legacy_hybrid": summarize(subset, "legacy_hybrid_fault"),
            "hybrid": summarize(subset, "hybrid_fault"),
            "expert_ece": ece(subset, "expert_fault", "expert_conf"),
            "llm_ece": ece(subset, "llm_fault", "llm_confidence"),
            "legacy_hybrid_ece": ece(subset, "legacy_hybrid_fault", "legacy_hybrid_conf"),
            "hybrid_ece": ece(subset, "hybrid_fault", "hybrid_conf"),
            "arbitration_actions": arbitration_action_metrics(subset),
        }

    metrics_by_evidence: Dict[str, Dict[str, Any]] = {}
    for strength in ["none", "weak", "moderate", "strong"]:
        subset = [r for r in records if r.get("evidence_strength") == strength]
        if not subset:
            continue
        subset_llm = [r for r in subset if int(r.get("llm_called", 0)) == 1]
        subset_fallback_llm = [
            r for r in subset
            if int(r.get("llm_called", 0)) == 1 and int(r.get("fallback_needed", 0)) == 1
        ]
        metrics_by_evidence[strength] = {
            "samples": len(subset),
            "expert": summarize(subset, "expert_fault"),
            "llm": summarize(subset, "llm_fault"),
            "llm_evaluated_subset": summarize(subset_llm, "llm_fault") if subset_llm else {},
            "llm_on_fallback_needed": summarize(subset_fallback_llm, "llm_fault") if subset_fallback_llm else {},
            "legacy_hybrid": summarize(subset, "legacy_hybrid_fault"),
            "hybrid": summarize(subset, "hybrid_fault"),
            "arbitration_actions": arbitration_action_metrics(subset),
        }
    metrics["metrics_by_evidence"] = metrics_by_evidence
    return metrics


def recompute_mode(args: argparse.Namespace, prompt_mode: str, samples: List[Any]) -> Optional[Dict[str, Any]]:
    out_dir = Path(__file__).parent / "experiment_outputs"
    suffix = f"real_llm_{prompt_mode}"
    pred_path = out_dir / f"public_benchmark_predictions_{suffix}.csv"
    metrics_path = out_dir / f"public_benchmark_metrics_{suffix}.json"
    if not pred_path.exists():
        print(f"Skipping {prompt_mode}: {pred_path.name} not found")
        return None
    with pred_path.open("r", encoding="utf-8-sig", newline="") as f:
        old_rows = list(csv.DictReader(f))
    records = rebuild_records(args, prompt_mode, old_rows, samples)
    metrics = compose_metrics(args, records, samples, prompt_mode)
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else ["source"])
        writer.writeheader()
        writer.writerows(records)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"Recomputed {prompt_mode}: "
        f"expert={metrics['expert']['accuracy']} "
        f"llm={metrics['llm']['accuracy']} "
        f"hybrid={metrics['hybrid']['accuracy']} -> {metrics_path.name}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute public Real-LLM metrics from recorded LLM outputs.")
    parser.add_argument("--prompt-mode", action="append", choices=["zero-shot", "kb-guided", "evidence-rich-kb", "rich-context"], help="Prompt mode to recompute. May be repeated. Defaults to kb-guided/evidence-rich-kb/rich-context.")
    parser.add_argument("--threshold-profile", choices=["public-tuned", "global"], default="public-tuned")
    parser.add_argument("--min-match", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--tau-grid", default="0.2,0.3,0.35,0.4,0.45,0.5,0.6")
    parser.add_argument("--min-match-grid", default="25,33.34,50,66.66,75,100")
    parser.add_argument("--hybrid-strategy", choices=["evidence-aware", "legacy"], default="evidence-aware")
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max-scenarios-per-source", type=int, default=12)
    parser.add_argument("--skip-metropt", action="store_true")
    parser.add_argument("--metropt-window-rows", type=int, default=300)
    parser.add_argument("--metropt-max-windows-per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    modes = args.prompt_mode or DEFAULT_PROMPT_MODES
    samples = build_samples(args)
    for mode in modes:
        recompute_mode(args, mode, samples)


if __name__ == "__main__":
    main()
