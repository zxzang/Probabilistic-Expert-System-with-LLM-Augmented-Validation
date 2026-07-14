"""Data-assisted rule-mining helper for the public benchmark.

This script trains shallow decision trees on the public abnormal-predicate
samples and exports high-purity decision paths as human-readable rule
candidates. It is an audit/refinement tool only: mined paths are not used as a
runtime classifier unless a human converts them into expert rules.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__)))
from experiment_metrics import normalize_label
from inference_engine import ProbabilisticInferenceEngine
from public_data_benchmark import (
    PublicBenchmarkKnowledgeBase,
    build_lbnl_samples,
    build_metropt_samples,
    resolve_thresholds,
    summarize,
)


def label_of(value: Optional[str]) -> str:
    return normalize_label(value) or "Normal"


def build_samples(args: argparse.Namespace) -> List[Any]:
    samples = build_lbnl_samples(
        max_rows=args.public_max_rows,
        stride=args.public_stride,
        max_scenarios_per_source=args.public_max_scenarios_per_source,
    )
    if not args.public_skip_metropt:
        samples.extend(
            build_metropt_samples(
                window_rows=args.public_metropt_window_rows,
                max_windows_per_class=args.public_metropt_max_windows_per_class,
            )
        )
    return samples


def expert_records(args: argparse.Namespace, kb: PublicBenchmarkKnowledgeBase, samples: List[Any]) -> List[Dict[str, Any]]:
    engines: Dict[float, ProbabilisticInferenceEngine] = {}
    records: List[Dict[str, Any]] = []
    for sample in samples:
        thresholds = resolve_thresholds(args, sample)
        if thresholds.min_match not in engines:
            engines[thresholds.min_match] = ProbabilisticInferenceEngine(
                kb=kb,
                min_match=thresholds.min_match,
                require_abnormal_anchor=True,
                evidence_damping=True,
            )
        preds = engines[thresholds.min_match].infer(sample.equipment, sample.params)
        expert_fault = preds[0][0] if preds else None
        expert_conf = preds[0][1] if preds else 0.0
        row: Dict[str, Any] = {
            "source": sample.source,
            "equipment": sample.equipment,
            "scenario": sample.scenario,
            "true_fault": sample.true_fault,
            "true_label": label_of(sample.true_fault),
            "expert_fault": expert_fault,
            "expert_label": label_of(expert_fault),
            "expert_conf": expert_conf,
        }
        row.update(sample.params)
        records.append(row)
    return records


def equipment_features(records: List[Dict[str, Any]]) -> List[str]:
    metadata = {
        "source",
        "equipment",
        "scenario",
        "true_fault",
        "true_label",
        "expert_fault",
        "expert_label",
        "expert_conf",
    }
    return sorted(
        key for row in records for key, value in row.items()
        if key not in metadata and isinstance(value, (int, float))
    )


def vectorize(records: List[Dict[str, Any]], features: List[str]) -> np.ndarray:
    return np.asarray([[float(row.get(feature, 0.0)) for feature in features] for row in records], dtype=float)


def format_condition(feature: str, lower: float, upper: float) -> str:
    """Translate decision-tree bounds over {0,1,2} levels into readable predicates."""
    if lower >= 1.5:
        return f"{feature} == 2"
    if lower > 0.5:
        return f"{feature} >= 1"
    if upper <= 0.5:
        return f"{feature} == 0"
    if upper <= 1.5:
        return f"{feature} <= 1"
    return f"{feature} any"


def condition_dict(bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[str, int]]:
    """Return an approximate Rule.conditions mapping from tree bounds."""
    out: Dict[str, Tuple[str, int]] = {}
    for feature, (lower, upper) in sorted(bounds.items()):
        if lower >= 1.5:
            out[feature] = ("==", 2)
        elif lower > 0.5:
            out[feature] = (">=", 1)
        elif upper <= 0.5:
            out[feature] = ("==", 0)
        elif upper <= 1.5:
            out[feature] = ("<=", 1)
    return out


def extract_paths(model: Any, features: List[str], labels: List[str], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract decision-tree leaves with support/purity/expert-error context."""
    tree = model.tree_
    paths: List[Dict[str, Any]] = []

    def walk(node_id: int, bounds: Dict[str, Tuple[float, float]]) -> None:
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]
        if left == right:
            class_counts = tree.value[node_id][0]
            pred_idx = int(np.argmax(class_counts))
            predicted_label = labels[pred_idx]
            mask = np.ones(len(records), dtype=bool)
            for feature, (lower, upper) in bounds.items():
                values = np.asarray([float(row.get(feature, 0.0)) for row in records])
                mask &= values > lower
                mask &= values <= upper
            subset = [row for row, keep in zip(records, mask) if keep]
            if not subset:
                return
            true_counts = Counter(row["true_label"] for row in subset)
            expert_counts = Counter(row["expert_label"] for row in subset)
            path_correct = sum(1 for row in subset if row["true_label"] == predicted_label)
            expert_correct = sum(1 for row in subset if row["true_label"] == row["expert_label"])
            conditions = condition_dict(bounds)
            paths.append(
                {
                    "predicted_label": predicted_label,
                    "support": len(subset),
                    "purity": round(path_correct / len(subset), 4),
                    "expert_accuracy": round(expert_correct / len(subset), 4),
                    "true_counts": dict(true_counts),
                    "expert_counts": dict(expert_counts),
                    "conditions": {k: [op, value] for k, (op, value) in conditions.items()},
                    "readable_conditions": [
                        format_condition(feature, lower, upper)
                        for feature, (lower, upper) in sorted(bounds.items())
                        if format_condition(feature, lower, upper) != f"{feature} any"
                    ],
                    "scenarios": [row["scenario"] for row in subset[:8]],
                }
            )
            return

        feature = features[tree.feature[node_id]]
        threshold = float(tree.threshold[node_id])
        low, high = bounds.get(feature, (-float("inf"), float("inf")))
        left_bounds = dict(bounds)
        left_bounds[feature] = (low, min(high, threshold))
        walk(left, left_bounds)
        right_bounds = dict(bounds)
        right_bounds[feature] = (max(low, threshold), high)
        walk(right, right_bounds)

    walk(0, {})
    return paths


def mine_equipment_rules(records: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    from sklearn.tree import DecisionTreeClassifier

    out: Dict[str, Any] = {}
    for equipment in sorted({row["equipment"] for row in records}):
        subset = [row for row in records if row["equipment"] == equipment]
        features = equipment_features(subset)
        labels = sorted({row["true_label"] for row in subset})
        if len(labels) < 2 or len(subset) < args.min_support:
            out[equipment] = {
                "status": "skipped",
                "samples": len(subset),
                "classes": labels,
                "reason": "Not enough samples/classes for decision-tree mining.",
            }
            continue
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        X = vectorize(subset, features)
        y = np.asarray([label_to_id[row["true_label"]] for row in subset], dtype=int)
        model = DecisionTreeClassifier(
            max_depth=args.max_depth,
            min_samples_leaf=args.min_leaf,
            class_weight="balanced",
            random_state=args.seed,
        )
        model.fit(X, y)
        paths = extract_paths(model, features, labels, subset)
        candidates = [
            path for path in paths
            if path["support"] >= args.min_support
            and path["purity"] >= args.min_purity
            and path["predicted_label"] != "Normal"
            and path["expert_accuracy"] < path["purity"]
        ]
        candidates.sort(key=lambda row: (row["purity"] - row["expert_accuracy"], row["support"]), reverse=True)
        expert_rows = [
            {
                "true_fault": None if row["true_label"] == "Normal" else row["true_label"],
                "expert_fault": None if row["expert_label"] == "Normal" else row["expert_label"],
            }
            for row in subset
        ]
        out[equipment] = {
            "status": "ok",
            "samples": len(subset),
            "features": features,
            "class_counts": dict(Counter(row["true_label"] for row in subset)),
            "expert_metrics": summarize(expert_rows, "expert_fault"),
            "all_paths": sorted(paths, key=lambda row: (row["purity"], row["support"]), reverse=True),
            "candidate_paths": candidates[: args.max_candidates_per_equipment],
        }
    return out


def active_signature(row: Dict[str, Any]) -> Tuple[Tuple[str, int], ...]:
    metadata = {
        "source",
        "equipment",
        "scenario",
        "true_fault",
        "true_label",
        "expert_fault",
        "expert_label",
        "expert_conf",
    }
    return tuple(sorted((key, int(value)) for key, value in row.items()
                        if key not in metadata and isinstance(value, (int, float)) and value > 0))


def mine_error_signatures(records: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for equipment in sorted({row["equipment"] for row in records}):
        subset = [row for row in records if row["equipment"] == equipment]
        groups: Dict[Tuple[str, Tuple[Tuple[str, int], ...]], List[Dict[str, Any]]] = defaultdict(list)
        for row in subset:
            if row["true_label"] == row["expert_label"]:
                continue
            groups[(row["true_label"], active_signature(row))].append(row)
        rows: List[Dict[str, Any]] = []
        for (label, signature), items in groups.items():
            if len(items) < args.min_support:
                continue
            expert_counts = Counter(item["expert_label"] for item in items)
            rows.append(
                {
                    "true_label": label,
                    "support": len(items),
                    "signature": {key: value for key, value in signature},
                    "expert_counts": dict(expert_counts),
                    "scenarios": [item["scenario"] for item in items[:8]],
                }
            )
        rows.sort(key=lambda item: item["support"], reverse=True)
        out[equipment] = rows[: args.max_candidates_per_equipment]
    return out


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# ML Rule-Mining Audit for Public Benchmark",
        "",
        "This file is generated by `code/ml_rule_mining.py`. Decision-tree paths are used only as a rule-base audit aid; they are not runtime model predictions.",
        "",
        "## Summary",
        "",
    ]
    for equipment, section in report["decision_tree_audit"].items():
        if section["status"] != "ok":
            lines.append(f"- {equipment}: skipped ({section['reason']})")
            continue
        metrics = section["expert_metrics"]
        lines.append(
            f"- {equipment}: samples={section['samples']}, expert_acc={metrics['accuracy']}, "
            f"fault_acc={metrics['fault_only_accuracy']}, candidates={len(section['candidate_paths'])}"
        )
    lines.extend(["", "## Candidate Decision-Tree Paths", ""])
    for equipment, section in report["decision_tree_audit"].items():
        lines.extend([f"### {equipment}", ""])
        if section["status"] != "ok":
            lines.extend([f"Skipped: {section['reason']}", ""])
            continue
        if not section["candidate_paths"]:
            lines.extend(["No high-purity candidate paths passed the filters.", ""])
            continue
        for idx, path in enumerate(section["candidate_paths"], start=1):
            lines.append(
                f"{idx}. Predict **{path['predicted_label']}** "
                f"(support={path['support']}, purity={path['purity']}, expert_acc={path['expert_accuracy']})"
            )
            lines.append(f"   - Conditions: `{path['conditions']}`")
            lines.append(f"   - True counts: `{path['true_counts']}`")
            lines.append(f"   - Expert counts: `{path['expert_counts']}`")
            lines.append(f"   - Example scenarios: `{path['scenarios']}`")
        lines.append("")
    lines.extend(["## Repeated Expert-Error Signatures", ""])
    for equipment, rows in report["error_signatures"].items():
        lines.extend([f"### {equipment}", ""])
        if not rows:
            lines.extend(["No repeated error signature met the support threshold.", ""])
            continue
        for idx, row in enumerate(rows, start=1):
            lines.append(
                f"{idx}. True **{row['true_label']}**, support={row['support']}, "
                f"expert_counts=`{row['expert_counts']}`"
            )
            lines.append(f"   - Active signature: `{row['signature']}`")
            lines.append(f"   - Example scenarios: `{row['scenarios']}`")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine shallow ML paths to audit public benchmark expert rules.")
    parser.add_argument("--threshold-profile", choices=["public-tuned", "global"], default="public-tuned")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--min-match", type=float, default=None)
    parser.add_argument("--public-max-rows", type=int, default=20000)
    parser.add_argument("--public-stride", type=int, default=60)
    parser.add_argument("--public-max-scenarios-per-source", type=int, default=12)
    parser.add_argument("--public-skip-metropt", action="store_true")
    parser.add_argument("--public-metropt-window-rows", type=int, default=300)
    parser.add_argument("--public-metropt-max-windows-per-class", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=2)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--min-purity", type=float, default=0.75)
    parser.add_argument("--max-candidates-per-equipment", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kb = PublicBenchmarkKnowledgeBase()
    samples = build_samples(args)
    records = expert_records(args, kb, samples)
    report = {
        "configuration": vars(args),
        "total_samples": len(records),
        "class_counts_by_equipment": {
            equipment: dict(Counter(row["true_label"] for row in records if row["equipment"] == equipment))
            for equipment in sorted({row["equipment"] for row in records})
        },
        "decision_tree_audit": mine_equipment_rules(records, args),
        "error_signatures": mine_error_signatures(records, args),
    }
    out_dir = Path(__file__).parent / "experiment_outputs"
    out_dir.mkdir(exist_ok=True)
    json_path = out_dir / "ml_rule_mining_public.json"
    md_path = out_dir / "ml_rule_mining_public.md"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(render_markdown(report))
    print(f"JSON written to {json_path}")
    print(f"Markdown written to {md_path}")


if __name__ == "__main__":
    main()
