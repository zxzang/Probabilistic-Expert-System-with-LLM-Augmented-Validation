"""Small Real-LLM smoke test for sparse LBNL RTU public samples.

This helper intentionally limits API calls and writes a separate smoke-test
artifact without touching the full public benchmark outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from inference_engine import ProbabilisticInferenceEngine
from public_data_benchmark import (
    PublicBenchmarkKnowledgeBase,
    build_lbnl_samples,
    evidence_strength_from_params,
    public_llm_prompt_inputs,
    resolve_thresholds,
    select_hybrid_decision,
    summarize,
)
from real_llm import RealLLMBaseline


def active_predicates(params: Dict[str, int]) -> Dict[str, int]:
    return {name: int(value) for name, value in params.items() if int(value) > 0}


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sparse RTU Real-LLM smoke test.")
    parser.add_argument("--max-api-calls", type=int, default=8)
    parser.add_argument(
        "--all-rtu",
        action="store_true",
        help="Evaluate all generated LBNL RTU public-benchmark samples instead of the sparse unique-scenario smoke subset.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["kb-guided", "evidence-rich-kb", "rich-context", "zero-shot", "expert-guided"],
        default="kb-guided",
    )
    parser.add_argument("--max-rows", type=int, default=20000)
    parser.add_argument("--stride", type=int, default=60)
    parser.add_argument("--max-scenarios-per-source", type=int, default=12)
    parser.add_argument("--output", default="")
    parser.add_argument("--threshold-profile", default="public-tuned")
    parser.add_argument("--hybrid-strategy", default="evidence-aware")
    parser.add_argument("--min-match", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = build_args()
    args.llm_prompt_mode = args.prompt_mode
    kb = PublicBenchmarkKnowledgeBase()
    real_llm = RealLLMBaseline()
    engine_cache: Dict[float, ProbabilisticInferenceEngine] = {}

    samples = build_lbnl_samples(
        max_rows=args.max_rows,
        stride=args.stride,
        max_scenarios_per_source=args.max_scenarios_per_source,
    )
    rtu_samples = [sample for sample in samples if sample.equipment == "LBNL RTU"]
    if args.all_rtu:
        selected = rtu_samples
        selection_note = "all generated LBNL RTU public-benchmark samples"
    else:
        sparse_samples = [
            sample
            for sample in rtu_samples
            if len(active_predicates(sample.params)) == 1
        ]
        selected = []
        seen_scenarios = set()
        for sample in sparse_samples:
            if sample.scenario in seen_scenarios:
                continue
            seen_scenarios.add(sample.scenario)
            selected.append(sample)
            if len(selected) >= args.max_api_calls:
                break
        selection_note = "single-active-predicate RTU scenarios, first unique scenarios"

    records: List[Dict[str, Any]] = []
    for idx, sample in enumerate(selected, 1):
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
        candidate_faults = kb.candidate_faults(sample.equipment)

        llm_context, llm_expert_top_k, include_predicates = public_llm_prompt_inputs(args, sample, engine)
        prompt_expert_fault = expert_fault if args.prompt_mode == "expert-guided" else None
        prompt_expert_conf = expert_conf if args.prompt_mode == "expert-guided" else 0.0
        llm_fault, llm_conf, meta = real_llm.diagnose(
            sample.equipment,
            sample.params,
            candidate_faults,
            expert_fault=prompt_expert_fault,
            expert_confidence=prompt_expert_conf,
            context=llm_context,
            expert_top_k=llm_expert_top_k,
            include_predicates=include_predicates,
        )

        _, _, evidence_strength = evidence_strength_from_params(sample.params)
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

        record = {
            "source": sample.source,
            "equipment": sample.equipment,
            "scenario": sample.scenario,
            "true_fault": sample.true_fault,
            "params": sample.params,
            "active_predicates": active_predicates(sample.params),
            "effective_min_match": thresholds.min_match,
            "effective_tau": thresholds.tau,
            "expert_fault": expert_fault,
            "expert_conf": round(expert_conf, 3),
            "llm_fault": llm_fault,
            "llm_confidence": llm_conf,
            "llm_abnormal_decision": meta.get("abnormal_decision", ""),
            "llm_evidence_sufficiency": meta.get("evidence_sufficiency", ""),
            "llm_error": meta.get("error", ""),
            "llm_called": 1,
            "fallback_needed": int(expert_conf < thresholds.tau),
            "hybrid_fault": hybrid_fault,
            "hybrid_conf": round(hybrid_conf, 3),
            "arbitration_action": arbitration_action,
        }
        records.append(record)
        print(
            f"[{idx}/{len(selected)}] {sample.scenario}: "
            f"true={sample.true_fault}; active={record['active_predicates']}; "
            f"expert={expert_fault}({expert_conf:.3f}); "
            f"llm={llm_fault}({llm_conf}); "
            f"hybrid={hybrid_fault}; action={arbitration_action}; "
            f"error={record['llm_error']}",
            flush=True,
        )

    output = {
        "prompt_mode": args.prompt_mode,
        "equipment": "LBNL RTU",
        "selection": selection_note,
        "api_calls": len(records),
        "metrics": {
            "expert": summarize(records, "expert_fault"),
            "llm": summarize(records, "llm_fault"),
            "hybrid": summarize(records, "hybrid_fault"),
        },
        "records": records,
    }
    default_stem = "rtu_all_real_llm" if args.all_rtu else "rtu_sparse_real_llm_smoke"
    out_path = Path(args.output) if args.output else Path(
        f"code/experiment_outputs/{default_stem}_{args.prompt_mode}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_path}")
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
