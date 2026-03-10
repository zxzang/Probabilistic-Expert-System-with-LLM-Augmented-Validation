"""Quick analysis of predictions.csv to compute per-equipment accuracy."""
import csv
from collections import defaultdict
from pathlib import Path

CSV = Path(__file__).parent / "experiment_outputs" / "predictions.csv"

stats = defaultdict(lambda: {"total": 0, "expert_ok": 0, "llm_ok": 0, "hybrid_ok": 0})

with open(CSV, "r", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        eq = r["equipment"]
        s = stats[eq]
        s["total"] += 1
        if r["true_fault"] == r["expert_fault"]:
            s["expert_ok"] += 1
        if r["true_fault"] == r["llm_fault"]:
            s["llm_ok"] += 1
        if r["true_fault"] == r["hybrid_fault"]:
            s["hybrid_ok"] += 1

print(f"{'Equipment':<20} {'Total':>6} {'Expert%':>8} {'LLM%':>8} {'Hybrid%':>8}")
print("-" * 54)
for eq in sorted(stats):
    s = stats[eq]
    t = s["total"]
    print(f"{eq:<20} {t:>6} {100*s['expert_ok']/t:>7.1f}% {100*s['llm_ok']/t:>7.1f}% {100*s['hybrid_ok']/t:>7.1f}%")

total = sum(s["total"] for s in stats.values())
expert_total = sum(s["expert_ok"] for s in stats.values())
llm_total = sum(s["llm_ok"] for s in stats.values())
hybrid_total = sum(s["hybrid_ok"] for s in stats.values())
print("-" * 54)
print(f"{'OVERALL':<20} {total:>6} {100*expert_total/total:>7.1f}% {100*llm_total/total:>7.1f}% {100*hybrid_total/total:>7.1f}%")
