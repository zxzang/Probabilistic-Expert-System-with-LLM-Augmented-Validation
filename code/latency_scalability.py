"""Latency and scalability benchmark for deployment discussion.

This script measures:
- Expert/PMS inference latency under different rule-base sizes
- Mock LLM local decision latency
- Optional real DeepSeek API latency for a small number of calls

The default path performs no external API calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time
from copy import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt

import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_generator import SyntheticDataGenerator
from inference_engine import ProbabilisticInferenceEngine
from knowledge_base import KnowledgeBase, Rule
from llm_baseline import SimulatedLLM
from real_llm import RealLLMBaseline


class ScaledKnowledgeBase(KnowledgeBase):
    """KnowledgeBase with duplicated rules for timing-only scalability tests."""

    def __init__(self, scale: int) -> None:
        super().__init__()
        if scale <= 1:
            return
        original = list(self.rules)
        scaled: List[Rule] = []
        for i in range(scale):
            for rule in original:
                cloned = copy(rule)
                # Keep labels identifiable but semantically equivalent enough
                # for timing. This benchmark is not used for accuracy.
                cloned.fault = f"{rule.fault} [scale-{i + 1}]"
                scaled.append(cloned)
        self.rules = scaled


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((pct / 100.0) * (len(values) - 1)))))
    return values[idx]


def latency_summary(values_ms: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values_ms), 4) if values_ms else 0.0,
        "median_ms": round(statistics.median(values_ms), 4) if values_ms else 0.0,
        "p95_ms": round(percentile(values_ms, 95), 4),
        "min_ms": round(min(values_ms), 4) if values_ms else 0.0,
        "max_ms": round(max(values_ms), 4) if values_ms else 0.0,
    }


def time_expert(samples: List[Dict[str, Any]], scale: int, repeats: int) -> Dict[str, Any]:
    kb = ScaledKnowledgeBase(scale)
    engine = ProbabilisticInferenceEngine(kb=kb)
    timings: List[float] = []
    for _ in range(repeats):
        for sample in samples:
            start = time.perf_counter()
            engine.infer(sample["equipment"], sample["params"])
            timings.append((time.perf_counter() - start) * 1000.0)
    row = {
        "component": "expert_pms",
        "rule_scale": scale,
        "rule_count": len(kb.rules),
        "sample_count": len(samples),
    }
    row.update(latency_summary(timings))
    return row


def time_mock_llm(samples: List[Dict[str, Any]], repeats: int) -> Dict[str, Any]:
    llm = SimulatedLLM()
    timings: List[float] = []
    for _ in range(repeats):
        for sample in samples:
            start = time.perf_counter()
            llm.diagnose(sample["equipment"], sample["params"], ground_truth=sample["fault"])
            timings.append((time.perf_counter() - start) * 1000.0)
    row = {
        "component": "mock_llm",
        "rule_scale": 1,
        "rule_count": 0,
        "sample_count": len(samples),
    }
    row.update(latency_summary(timings))
    return row


def time_real_llm(samples: List[Dict[str, Any]], calls: int) -> Dict[str, Any]:
    kb = KnowledgeBase()
    engine = ProbabilisticInferenceEngine(kb=kb)
    llm = RealLLMBaseline()
    timings: List[float] = []
    errors = 0
    for sample in samples[:calls]:
        expert_preds = engine.infer(sample["equipment"], sample["params"])
        expert_fault = expert_preds[0][0] if expert_preds else None
        expert_conf = expert_preds[0][1] if expert_preds else 0.0
        candidate_faults = [rule.fault for rule in kb.get_rules(sample["equipment"])]
        start = time.perf_counter()
        _fault, _conf, meta = llm.diagnose(
            sample["equipment"],
            sample["params"],
            candidate_faults,
            expert_fault=expert_fault,
            expert_confidence=expert_conf,
            context="Latency benchmark call.",
        )
        timings.append((time.perf_counter() - start) * 1000.0)
        if meta.get("error"):
            errors += 1
    row = {
        "component": "real_llm_deepseek",
        "rule_scale": 1,
        "rule_count": 0,
        "sample_count": min(calls, len(samples)),
        "errors": errors,
    }
    row.update(latency_summary(timings))
    return row


def plot_scalability(rows: List[Dict[str, Any]], path: Path) -> None:
    expert_rows = [r for r in rows if r["component"] == "expert_pms"]
    if not expert_rows:
        return
    expert_rows = sorted(expert_rows, key=lambda r: r["rule_count"])
    plt.figure(figsize=(7, 4.5))
    plt.plot([r["rule_count"] for r in expert_rows], [r["mean_ms"] for r in expert_rows], marker="o", label="Mean")
    plt.plot([r["rule_count"] for r in expert_rows], [r["p95_ms"] for r in expert_rows], marker="s", label="P95")
    plt.xlabel("Rule count")
    plt.ylabel("Inference latency (ms/sample)")
    plt.title("PMS Scalability with Rule-Base Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    generator = SyntheticDataGenerator(samples_per_eq=args.samples_per_equipment, noise_levels=(0.0,), partial_ratio=0.0)
    samples = generator.generate()
    if args.sample_limit:
        samples = samples[: args.sample_limit]

    rows: List[Dict[str, Any]] = []
    for scale in args.rule_scales:
        rows.append(time_expert(samples, scale, args.repeats))
    rows.append(time_mock_llm(samples, args.repeats))
    if args.real_llm_calls > 0:
        rows.append(time_real_llm(samples, args.real_llm_calls))

    payload = {"configuration": vars(args), "results": rows}
    out_dir = Path(__file__).parent / "experiment_outputs"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "latency_scalability.csv"
    json_path = out_dir / "latency_scalability.json"
    plot_path = out_dir / "latency_scalability.png"

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    plot_scalability(rows, plot_path)

    print(json.dumps(payload, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark inference latency and PMS scalability.")
    parser.add_argument("--samples-per-equipment", type=int, default=80)
    parser.add_argument("--sample-limit", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rule-scales", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--real-llm-calls", type=int, default=0, help="Optional DeepSeek calls for API latency.")
    return parser.parse_args()


if __name__ == "__main__":
    main()

