"""Export a compact Markdown summary of experiment evidence for rebuttal."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


OUT_DIR = Path(__file__).parent / "experiment_outputs"
CODE_DIR = Path(__file__).parent


def read_json(name: str) -> Dict[str, Any] | None:
    path = OUT_DIR / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(name: str) -> List[Dict[str, str]]:
    path = OUT_DIR / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def line_metric(label: str, metrics: Dict[str, Any] | None) -> str:
    if not metrics:
        return f"- **{label}:** not found"
    false_alarm = metrics.get("false_alarm_rate")
    suffix = f", false-alarm rate={false_alarm}" if false_alarm is not None else ""
    return (
        f"- **{label}:** accuracy={metrics.get('accuracy', 'NA')}, "
        f"fault-only accuracy={metrics.get('fault_only_accuracy', 'NA')}, "
        f"macro-F1={metrics.get('macro_f1', metrics.get('macro_f1_faults', 'NA'))}"
        f"{suffix}"
    )


def first_present(mapping: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def public_benchmark_suffix(mode: str, prompt_mode: str | None = None) -> str:
    if mode == "real":
        return f"real_llm_{prompt_mode or 'kb-guided'}"
    return mode


def public_prediction_file(mode: str, prompt_mode: str | None = None) -> str:
    suffix = public_benchmark_suffix(mode, prompt_mode)
    return f"public_benchmark_predictions_{suffix}.csv"


def result_is_stale(result_name: str, *code_names: str) -> bool:
    result_path = OUT_DIR / result_name
    if not result_path.exists():
        return False
    result_mtime = result_path.stat().st_mtime
    return any((CODE_DIR / name).exists() and (CODE_DIR / name).stat().st_mtime > result_mtime for name in code_names)


def llm_error_summary(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    errors: Dict[str, int] = {}
    for row in rows:
        error = str(row.get("llm_error", "")).strip()
        if not error:
            continue
        errors[error] = errors.get(error, 0) + 1
    return {
        "count": sum(errors.values()),
        "types": errors,
    }


def llm_self_check_summary(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for field in ["llm_abnormal_decision", "llm_evidence_sufficiency"]:
        counts: Dict[str, int] = {}
        for row in rows:
            if str(row.get("llm_called", "0")).strip() not in {"1", "True", "true"}:
                continue
            value = str(row.get(field, "")).strip().lower()
            if not value:
                continue
            counts[value] = counts.get(value, 0) + 1
        if counts:
            summary[field] = counts
    return summary


def main() -> None:
    sections: List[str] = []
    sections.append("# Rebuttal Experiment Evidence Summary\n")
    sections.append("This file is generated from `code/experiment_outputs/` by `code/export_rebuttal_evidence.py`.\n")

    real_llm = read_json("real_llm_synthetic_metrics.json")
    sections.append("## Real LLM on Synthetic Subset\n")
    if real_llm:
        sections.append(line_metric("Expert", real_llm.get("expert")))
        sections.append(line_metric("Real LLM", real_llm.get("real_llm")))
        sections.append(line_metric("Hybrid", real_llm.get("hybrid")))
        sections.append(f"- LLM errors: {real_llm.get('llm_errors', 'NA')}")
        sections.append(f"- Total samples: {real_llm.get('total_samples', 'NA')}")
    else:
        sections.append("- Not found: run `python code/real_llm_experiment.py --sample-size 200 --fault-only`.")
    sections.append("")

    public_real = read_json("public_benchmark_metrics_real_llm_kb-guided.json")
    if public_real is None:
        public_real = read_json("public_benchmark_metrics_real_llm.json")
    public_mock = read_json("public_benchmark_metrics_mock.json")
    public_none = read_json("public_benchmark_metrics_none.json")
    sections.append("## Public Data Benchmark\n")
    public_candidates = [m for m in [public_real, public_mock, public_none] if m]
    public_metrics = next((m for m in public_candidates if "fallback_rate" in m), None)
    if public_metrics is None:
        public_metrics = public_candidates[0] if public_candidates else None
    if public_metrics:
        mode = public_metrics.get("configuration", {}).get("llm", "NA")
        prompt_mode = public_metrics.get("configuration", {}).get("llm_prompt_mode")
        suffix = public_benchmark_suffix(str(mode), prompt_mode)
        result_name = f"public_benchmark_metrics_{suffix}.json"
        if result_is_stale(result_name, "public_data_benchmark.py", "real_llm.py"):
            sections.append(
                f"- **Warning:** `{result_name}` is older than the public benchmark code; "
                "rerun the public benchmark before using these numbers in the manuscript."
            )
        public_rows = read_csv(public_prediction_file(str(mode), prompt_mode)) if mode != "NA" else []
        public_errors = llm_error_summary(public_rows)
        sections.append(f"- Mode: {mode}")
        sections.append(f"- Total samples: {public_metrics.get('total_samples', 'NA')}")
        if public_metrics.get("recomputed_from_predictions"):
            sections.append(f"- API calls in this run: {public_metrics.get('api_calls', 'NA')}")
            sections.append(f"- Recorded Real LLM outputs reused: {public_metrics.get('recorded_llm_calls_reused', 'NA')}")
        else:
            sections.append(f"- API calls: {public_metrics.get('api_calls', 'NA')}")
        sections.append(f"- LLM-evaluated samples: {public_metrics.get('llm_evaluated_samples', 'NA')}")
        sections.append(f"- LLM errors: {public_errors.get('count', 'NA')}")
        if public_errors.get("types"):
            error_text = "; ".join(f"{name}: {count}" for name, count in public_errors["types"].items())
            sections.append(f"- LLM error types: {error_text}")
        self_check = llm_self_check_summary(public_rows)
        if self_check:
            if self_check.get("llm_abnormal_decision"):
                sections.append(f"- LLM abnormal-decision self-check: {self_check['llm_abnormal_decision']}")
            if self_check.get("llm_evidence_sufficiency"):
                sections.append(f"- LLM evidence-sufficiency self-check: {self_check['llm_evidence_sufficiency']}")
        sections.append(f"- LLM evaluation rate: {public_metrics.get('llm_evaluation_rate', 'NA')}")
        sections.append(f"- Fallback-needed rate: {public_metrics.get('fallback_rate', 'NA')}")
        sections.append(
            f"- ECE: expert={public_metrics.get('expert_ece', 'NA')}, "
            f"LLM={public_metrics.get('llm_ece', 'NA')}, "
            f"hybrid={public_metrics.get('hybrid_ece', 'NA')}"
        )
        coverage = public_metrics.get("predicate_coverage", {})
        if coverage:
            act_cov = coverage.get("overall_activation_coverage")
            full_cov = coverage.get("overall_fullmatch_coverage")
            sections.append(f"- Public predicate-space activation coverage (S_match ≥ min_match): {act_cov}" if act_cov is not None else "- Public predicate-space activation coverage: NA")
            if full_cov is not None:
                sections.append(f"- Public predicate-space full-match coverage: {full_cov}")
            sections.append(
                "- Coverage note: these are public-benchmark rule-space reachability checks, "
                "not sample-level diagnostic accuracy."
            )
            by_equipment_cov = coverage.get("by_equipment", {})
            if by_equipment_cov:
                sections.append("")
                sections.append("### Public Predicate-Space Coverage")
                sections.append("| Equipment | Predicates | min_match | Activation Coverage | Full-Match Coverage | Fault-Class Coverage |")
                sections.append("|---|---:|---:|---:|---:|---:|")
                for equipment, vals in by_equipment_cov.items():
                    sections.append(
                        f"| {equipment} | {vals.get('predicates', 'NA')} | "
                        f"{vals.get('min_match', 'NA')} | "
                        f"{vals.get('activation_coverage', 'NA')} | "
                        f"{vals.get('fullmatch_coverage', 'NA')} | "
                        f"{vals.get('fault_class_coverage', 'NA')} |"
                    )
        ttd = public_metrics.get("metropt_time_to_detection", {})
        if ttd:
            readable_ttd = "; ".join(
                f"{fault}: expert={vals.get('expert_minutes')} min, "
                f"LLM={vals.get('llm_minutes')} min, hybrid={vals.get('hybrid_minutes')} min"
                for fault, vals in ttd.items()
            )
            sections.append(f"- MetroPT time-to-detection: {readable_ttd}")
        sections.append(line_metric("Expert", public_metrics.get("expert")))
        sections.append(line_metric("LLM", public_metrics.get("llm")))
        evaluated_subset = public_metrics.get("llm_evaluated_subset") or public_metrics.get("llm_called_only")
        if evaluated_subset:
            sections.append(line_metric("LLM evaluated subset", evaluated_subset))
        if public_metrics.get("llm_on_fallback_needed"):
            sections.append(line_metric("LLM on fallback-needed subset", public_metrics.get("llm_on_fallback_needed")))
        sections.append(line_metric("Hybrid", public_metrics.get("hybrid")))
        by_source = public_metrics.get("by_source", {})
        if by_source:
            sections.append("")
            sections.append("| Source | Samples | Fallback Rate | Expert Acc | LLM Acc | Hybrid Acc | Hybrid ECE |")
            sections.append("|---|---:|---:|---:|---:|---:|---:|")
            for source, vals in by_source.items():
                sections.append(
                    f"| {source} | {vals.get('samples', 'NA')} | "
                    f"{vals.get('fallback_rate', 'NA')} | "
                    f"{vals.get('expert', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('llm', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid_ece', 'NA')} |"
                )
        by_evidence = public_metrics.get("metrics_by_evidence", {})
        if by_evidence:
            sections.append("")
            sections.append("### By Evidence Strength")
            sections.append("| Evidence | Samples | Expert Acc | LLM Acc | Hybrid Acc | Hybrid F1 |")
            sections.append("|---|---:|---:|---:|---:|---:|")
            for strength in ["strong", "moderate", "weak", "none"]:
                vals = by_evidence.get(strength, {})
                if not vals:
                    continue
                llm_vals = vals.get("llm_evaluated_subset") or vals.get("llm_called_only") or vals.get("llm", {})
                sections.append(
                    f"| {strength} | {vals.get('samples', 'NA')} | "
                    f"{vals.get('expert', {}).get('accuracy', 'NA')} | "
                    f"{llm_vals.get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid', {}).get('macro_f1', 'NA')} |"
                )
    else:
        sections.append("- Not found: run `python code/public_data_benchmark.py --llm mock` or `--llm real`.")
    sections.append("")

    public_evidence_rich = read_json("public_benchmark_metrics_real_llm_evidence-rich-kb.json")
    sections.append("## Public Real LLM Evidence-Rich KB Ablation\n")
    if public_evidence_rich:
        evidence_rows = read_csv("public_benchmark_predictions_real_llm_evidence-rich-kb.csv")
        evidence_errors = llm_error_summary(evidence_rows)
        if result_is_stale(
            "public_benchmark_metrics_real_llm_evidence-rich-kb.json",
            "public_data_benchmark.py",
            "real_llm.py",
        ):
            sections.append(
                "- **Warning:** `public_benchmark_metrics_real_llm_evidence-rich-kb.json` "
                "is older than the public benchmark code; rerun before using these "
                "numbers in the manuscript."
            )
        sections.append("- Prompt mode: evidence-rich-kb")
        sections.append(
            "- Setting: KB fault-indicator mappings and predicate values plus raw/statistical "
            "public-benchmark deviations; PMS top-k outputs are hidden."
        )
        sections.append(f"- Total samples: {public_evidence_rich.get('total_samples', 'NA')}")
        if public_evidence_rich.get("recomputed_from_predictions"):
            sections.append(f"- API calls in this run: {public_evidence_rich.get('api_calls', 'NA')}")
            sections.append(f"- Recorded Real LLM outputs reused: {public_evidence_rich.get('recorded_llm_calls_reused', 'NA')}")
        else:
            sections.append(f"- API calls: {public_evidence_rich.get('api_calls', 'NA')}")
        sections.append(f"- LLM errors: {evidence_errors.get('count', 'NA')}")
        if evidence_errors.get("types"):
            error_text = "; ".join(f"{name}: {count}" for name, count in evidence_errors["types"].items())
            sections.append(f"- LLM error types: {error_text}")
        evidence_self_check = llm_self_check_summary(evidence_rows)
        if evidence_self_check:
            if evidence_self_check.get("llm_abnormal_decision"):
                sections.append(f"- LLM abnormal-decision self-check: {evidence_self_check['llm_abnormal_decision']}")
            if evidence_self_check.get("llm_evidence_sufficiency"):
                sections.append(f"- LLM evidence-sufficiency self-check: {evidence_self_check['llm_evidence_sufficiency']}")
        sections.append(line_metric("Expert", public_evidence_rich.get("expert")))
        sections.append(line_metric("Evidence-rich KB LLM", public_evidence_rich.get("llm")))
        sections.append(line_metric("Evidence-rich KB Hybrid", public_evidence_rich.get("hybrid")))
        by_source = public_evidence_rich.get("by_source", {})
        if by_source:
            sections.append("")
            sections.append("| Source | Samples | LLM Acc | Hybrid Acc | Hybrid F1 |")
            sections.append("|---|---:|---:|---:|---:|")
            for source, vals in by_source.items():
                sections.append(
                    f"| {source} | {vals.get('samples', 'NA')} | "
                    f"{vals.get('llm', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid', {}).get('macro_f1', 'NA')} |"
                )
    else:
        sections.append(
            "- Not found: run `python code/public_data_benchmark.py --llm real "
            "--llm-prompt-mode evidence-rich-kb --max-api-calls 150`."
        )
    sections.append("")

    public_rich = read_json("public_benchmark_metrics_real_llm_rich-context.json")
    sections.append("## Public Real LLM Rich-Context Ablation\n")
    if public_rich:
        rich_rows = read_csv("public_benchmark_predictions_real_llm_rich-context.csv")
        rich_errors = llm_error_summary(rich_rows)
        if result_is_stale(
            "public_benchmark_metrics_real_llm_rich-context.json",
            "public_data_benchmark.py",
            "real_llm.py",
        ):
            sections.append(
                "- **Warning:** `public_benchmark_metrics_real_llm_rich-context.json` "
                "is older than the public benchmark code; rerun before using these "
                "numbers in the manuscript."
            )
        sections.append("- Prompt mode: rich-context")
        sections.append(
            "- Setting: raw/statistical public-benchmark summaries plus mechanism "
            "notes; expert predicates and PMS top-k outputs are hidden."
        )
        sections.append(f"- Total samples: {public_rich.get('total_samples', 'NA')}")
        sections.append(f"- API calls: {public_rich.get('api_calls', 'NA')}")
        sections.append(f"- LLM errors: {rich_errors.get('count', 'NA')}")
        if rich_errors.get("types"):
            error_text = "; ".join(f"{name}: {count}" for name, count in rich_errors["types"].items())
            sections.append(f"- LLM error types: {error_text}")
        rich_self_check = llm_self_check_summary(rich_rows)
        if rich_self_check:
            if rich_self_check.get("llm_abnormal_decision"):
                sections.append(f"- LLM abnormal-decision self-check: {rich_self_check['llm_abnormal_decision']}")
            if rich_self_check.get("llm_evidence_sufficiency"):
                sections.append(f"- LLM evidence-sufficiency self-check: {rich_self_check['llm_evidence_sufficiency']}")
        sections.append(line_metric("Expert", public_rich.get("expert")))
        sections.append(line_metric("Rich-context LLM", public_rich.get("llm")))
        sections.append(line_metric("Rich-context Hybrid", public_rich.get("hybrid")))
        by_source = public_rich.get("by_source", {})
        if by_source:
            sections.append("")
            sections.append("| Source | Samples | LLM Acc | Hybrid Acc | Hybrid F1 |")
            sections.append("|---|---:|---:|---:|---:|")
            for source, vals in by_source.items():
                sections.append(
                    f"| {source} | {vals.get('samples', 'NA')} | "
                    f"{vals.get('llm', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid', {}).get('accuracy', 'NA')} | "
                    f"{vals.get('hybrid', {}).get('macro_f1', 'NA')} |"
                )
    else:
        sections.append(
            "- Not found: run `python code/public_data_benchmark.py --llm real "
            "--llm-prompt-mode rich-context --max-api-calls 150`."
        )
    sections.append("")

    alpha_beta = read_json("alpha_beta_sensitivity.json")
    sections.append("## Alpha/Beta Sensitivity\n")
    if alpha_beta:
        rows = alpha_beta.get("results", [])
        if rows:
            best = max(rows, key=lambda r: r.get("hybrid_macro_f1", 0.0))
            baseline = next((r for r in rows if float(r.get("alpha", -1)) == 0.1 and float(r.get("beta", -1)) == 0.2), None)
            sections.append(
                f"- Best hybrid macro-F1: {best.get('hybrid_macro_f1')} "
                f"at alpha={best.get('alpha')}, beta={best.get('beta')}"
            )
            if baseline:
                sections.append(
                    f"- Baseline alpha=0.1, beta=0.2: hybrid macro-F1={baseline.get('hybrid_macro_f1')}, "
                    f"hybrid ECE={baseline.get('hybrid_ece')}"
                )
            sections.append("- Figures: `alpha_beta_hybrid_macro_f1.png`, `alpha_beta_hybrid_ece.png`, `per_equipment_confidence_hist.png`")
    else:
        sections.append("- Not found: run `python code/hyperparameter_sensitivity.py`.")
    sections.append("")

    stat = read_json("statistical_significance_summary.json")
    sections.append("## Statistical Significance / Repeated Runs\n")
    if stat:
        summary = stat.get("summary", {})
        metric_groups = [
            ("expert_fault_only_accuracy", ["expert_fault_only_accuracy"]),
            ("hybrid_fault_only_accuracy", ["hybrid_fault_only_accuracy"]),
            ("expert_macro_f1", ["expert_macro_f1", "expert_macro_f1_faults"]),
            ("hybrid_macro_f1", ["hybrid_macro_f1", "hybrid_macro_f1_faults"]),
            ("hybrid_ece", ["hybrid_ece"]),
        ]
        for label, keys in metric_groups:
            val = first_present(summary, *keys)
            if val:
                sections.append(f"- {label}: mean={val.get('mean')}, std={val.get('std')}, 95% CI={val.get('ci95')}")
    else:
        sections.append("- Not found: run `python code/statistical_significance.py`.")
    sections.append("")

    ml_rows = read_csv("ml_baseline_metrics.csv")
    public_ml = read_json("public_ml_baseline_metrics.json")
    public_ml_rows = read_csv("public_ml_baseline_metrics.csv")
    sections.append("## ML Baselines\n")
    if ml_rows:
        if result_is_stale("ml_baseline_metrics.json", "ml_baselines.py"):
            sections.append(
                "- **Warning:** `ml_baseline_metrics.json` is older than `ml_baselines.py`; "
                "rerun ML baselines before using these numbers in the manuscript."
            )
        sections.append("### Synthetic Predicate ML Baselines")
        sections.append("| Model | Accuracy | Fault-only Acc | Macro-F1 Faults |")
        sections.append("|---|---:|---:|---:|")
        for row in ml_rows:
            sections.append(
                f"| {row.get('model')} | {row.get('accuracy')} | "
                f"{row.get('fault_only_accuracy')} | "
                f"{first_present(row, 'macro_f1_faults', 'macro_f1')} |"
            )
    else:
        sections.append("- Synthetic ML baselines not found: run `python code/ml_baselines.py --dataset synthetic`.")
    if public_ml_rows:
        if result_is_stale("public_ml_baseline_metrics.json", "ml_baselines.py", "public_data_benchmark.py"):
            sections.append(
                "- **Warning:** `public_ml_baseline_metrics.json` is older than the ML/public benchmark code; "
                "rerun public ML baselines before using these numbers in the manuscript."
            )
        filter_info = public_ml.get("filter", {}) if public_ml else {}
        if filter_info:
            sections.append(
                f"- Public ML supervised split used {filter_info.get('used_samples', 'NA')}/"
                f"{filter_info.get('original_samples', 'NA')} samples and "
                f"{filter_info.get('used_classes', 'NA')}/{filter_info.get('original_classes', 'NA')} classes "
                f"(min class count={filter_info.get('min_class_count', 'NA')})."
            )
        sections.append("")
        sections.append("### Public Predicate ML Baselines")
        sections.append("| Scope | Model | Accuracy | Fault-only Acc | Macro-F1 Faults |")
        sections.append("|---|---|---:|---:|---:|")
        for row in public_ml_rows:
            sections.append(
                f"| {row.get('scope')} | {row.get('model')} | {row.get('accuracy')} | "
                f"{row.get('fault_only_accuracy')} | "
                f"{first_present(row, 'macro_f1_faults', 'macro_f1')} |"
            )
    else:
        sections.append("- Public ML baselines not found: run `python code/ml_baselines.py --dataset public`.")
    sections.append("")

    latency = read_json("latency_scalability.json")
    sections.append("## Latency and Scalability\n")
    if latency:
        sections.append("| Component | Rule Count | Mean ms | P95 ms |")
        sections.append("|---|---:|---:|---:|")
        for row in latency.get("results", []):
            sections.append(
                f"| {row.get('component')} | {row.get('rule_count')} | "
                f"{row.get('mean_ms')} | {row.get('p95_ms')} |"
            )
    else:
        sections.append("- Not found: run `python code/latency_scalability.py`.")
    sections.append("")

    output_path = OUT_DIR / "rebuttal_evidence_summary.md"
    output_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
