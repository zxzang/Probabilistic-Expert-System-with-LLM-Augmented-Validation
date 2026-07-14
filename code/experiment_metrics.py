"""Shared metric helpers for reviewer-response experiments."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


def normalize_label(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"normal", "none", "no fault", "null"}:
        return None
    return text


def summarize(records: List[Dict[str, Any]], pred_key: str, true_key: str = "true_fault") -> Dict[str, float]:
    """Return accuracy, fault-only accuracy, macro precision/recall/F1.

    Macro metrics are computed over true fault classes only. Non-empty
    predictions outside that closed fault-label set are treated as
    hallucinated/unknown diagnoses and penalized against the sample's true
    fault class.
    """
    total = len(records)
    correct = sum(1 for r in records if normalize_label(r.get(true_key)) == normalize_label(r.get(pred_key)))
    fault_records = [r for r in records if normalize_label(r.get(true_key)) is not None]
    fault_correct = sum(
        1 for r in fault_records if normalize_label(r.get(true_key)) == normalize_label(r.get(pred_key))
    )
    labels = sorted({normalize_label(r.get(true_key)) for r in fault_records if normalize_label(r.get(true_key))})
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    label_set = set(labels)
    for label in labels:
        tp = sum(
            1 for r in records
            if normalize_label(r.get(true_key)) == label and normalize_label(r.get(pred_key)) == label
        )
        fp = sum(
            1 for r in records
            if normalize_label(r.get(true_key)) != label and normalize_label(r.get(pred_key)) == label
        )
        fn = sum(
            1 for r in records
            if normalize_label(r.get(true_key)) == label and normalize_label(r.get(pred_key)) != label
        )
        hallucinated_fp = sum(
            1 for r in records
            if normalize_label(r.get(true_key)) == label
            and normalize_label(r.get(pred_key)) is not None
            and normalize_label(r.get(pred_key)) not in label_set
        )
        fp += hallucinated_fp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "fault_only_accuracy": round(fault_correct / len(fault_records), 4) if fault_records else 0.0,
        "macro_precision": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "macro_recall": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else 0.0,
    }


def hybrid_decision(
    expert_fault: Optional[str],
    expert_conf: float,
    llm_fault: Optional[str],
    llm_conf: float,
    tau: float = 0.6,
    expert_weight: float = 0.7,
    llm_weight: float = 0.3,
) -> Tuple[Optional[str], float]:
    """Return the hybrid weighted-selection decision and selected confidence.

    The architecture first accepts high-confidence expert diagnoses. For
    low-confidence cases it compares weighted confidences and selects one
    label; it does not merge full class-probability distributions.
    """
    if expert_conf >= tau:
        return expert_fault, expert_conf
    if llm_fault is not None and expert_fault is not None:
        if expert_weight * expert_conf >= llm_weight * llm_conf:
            return expert_fault, expert_conf
        return llm_fault, llm_conf
    if llm_fault is not None:
        return llm_fault, llm_conf
    return expert_fault, expert_conf


def ece(records: List[Dict[str, Any]], pred_key: str, conf_key: str, true_key: str = "true_fault", bins: int = 10) -> float:
    """Expected calibration error for rows with a confidence value."""
    bucket = defaultdict(lambda: {"total": 0, "correct": 0, "conf_sum": 0.0})
    for r in records:
        try:
            conf = float(r.get(conf_key, 0.0))
        except (TypeError, ValueError):
            continue
        if conf <= 0:
            continue
        idx = min(bins - 1, int(conf * bins))
        bucket[idx]["total"] += 1
        bucket[idx]["conf_sum"] += conf
        if normalize_label(r.get(true_key)) == normalize_label(r.get(pred_key)):
            bucket[idx]["correct"] += 1

    total = sum(v["total"] for v in bucket.values())
    if total == 0:
        return 0.0
    value = 0.0
    for v in bucket.values():
        acc = v["correct"] / v["total"]
        avg_conf = v["conf_sum"] / v["total"]
        value += (v["total"] / total) * abs(acc - avg_conf)
    return round(value, 4)


def mean_std_ci(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0}
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return {"mean": round(mean, 4), "std": 0.0, "ci95": 0.0}
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std = math.sqrt(var)
    ci95 = 1.96 * std / math.sqrt(len(vals))
    return {"mean": round(mean, 4), "std": round(std, 4), "ci95": round(ci95, 4)}
