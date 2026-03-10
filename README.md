# Probabilistic Expert System with LLM-Augmented Validation

> **Enhancing Industrial Fault Diagnosis: A Probabilistic Expert System with LLM‑Augmented Validation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A **neuro-symbolic** framework for industrial equipment fault diagnosis that combines a deterministic, physics-grounded rule-based expert system with a Large Language Model (LLM) fallback mechanism. The system achieves high diagnostic accuracy while maintaining full interpretability — critical for mission-critical manufacturing operations.

---

## Highlights

- 🏭 **7 Equipment Types** — Boiler, Chiller, HVAC, Air Compressor, Vacuum Machine, Power Distribution, Water Supply
- 🧠 **Bayesian-Style Inference** — Pattern Matching with Scoring (PMS) engine that handles partial symptom matches
- 🤖 **LLM Fallback** — Confidence-threshold routing ($\tau = 0.6$) invokes an LLM only for low-confidence / out-of-distribution cases
- 📊 **Monte-Carlo Coverage Analysis** — Quantifies rule-base completeness over the physically feasible state space
- 🔬 **Fully Reproducible** — Synthetic data generation with configurable noise levels; no external API keys required

## Key Results

| Method | Overall Acc | Fault Acc | Macro-Prec | Macro-Rec | Macro-F1 | Coverage |
|---|---|---|---|---|---|---|
| Pure Expert System | 0.960 | 0.932 | 0.917 | 0.907 | 0.912 | 36% |
| Pure LLM (Mock) | 0.917 | 0.853 | 0.996 | 0.851 | 0.917 | 100% |
| **Hybrid (Ours)** | 0.945 | **0.974** | 0.923 | **0.960** | **0.940** | **100%** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Real-Time IIoT Sensor Data                 │
└───────────────────────┬─────────────────────────────────┘
                        ▼
           ┌────────────────────────┐
           │    Knowledge Base      │  ◄── 7 equipment types,
           │  (knowledge_base.py)   │      14 diagnostic rules,
           └────────────┬───────────┘      physical constraints
                        ▼
           ┌────────────────────────┐
           │  PMS Inference Engine  │  ◄── Bayesian confidence
           │ (inference_engine.py)  │      scoring
           └────────────┬───────────┘
                        ▼
                ┌───────────────┐
                │ Conf ≥ τ ?    │
                └───┬───────┬───┘
            Yes ▼           ▼ No
    ┌──────────────┐  ┌──────────────────┐
    │  Expert       │  │  LLM Fallback    │
    │  Diagnosis    │  │ (llm_baseline.py)│
    └──────┬───────┘  └────────┬─────────┘
           │                   │
           │    Weighted Ensemble
           │    (Expert 0.7 + LLM 0.3)
           ▼                   ▼
    ┌──────────────────────────────────┐
    │     Final Diagnostic Output      │
    └──────────────────────────────────┘
```

---

## Repository Structure

```
.
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
│
├── code/                           # Main source code
    ├── knowledge_base.py           # Rule definitions & physical constraints (7 equipment types)
    ├── inference_engine.py         # Probabilistic inference with Bayesian updating
    ├── data_generator.py           # Synthetic dataset generator with configurable noise
    ├── llm_baseline.py             # Simulated LLM baseline & hybrid decision logic
    ├── coverage_analyzer.py        # Monte-Carlo rule coverage estimation
    ├── experiment_runner.py        # Main experiment orchestration pipeline
    ├── compute_metrics.py          # Macro-averaged P/R/F1, parameter sensitivity analysis
    ├── plot_results.py             # Basic visualization (coverage & accuracy bar charts)
    ├── generate_advanced_plots.py  # Advanced plots (radar, heatmap, KDE, dual-axis)
    ├── generate_confusion_matrices.py  # Per-equipment confusion matrix heatmaps
    ├── analyze_results.py          # Quick per-equipment accuracy summary
    └── experiment_outputs/         # Generated experiment results
        ├── metrics.json            # Summary metrics (coverage, accuracy)
        ├── comprehensive_metrics.json  # Full metrics incl. parameter sensitivity analysis
        └── predictions.csv         # Raw prediction records

```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/probabilistic-expert-system-llm.git
cd probabilistic-expert-system-llm

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

**Run the full experiment pipeline:**

```bash
cd code
python experiment_runner.py
```

This will:
1. Estimate rule coverage via Monte-Carlo sampling
2. Generate 10,500 synthetic samples (500 per equipment × 3 noise levels × 7 equipment types)
3. Run the expert system, LLM baseline, and hybrid model on all samples
4. Write `metrics.json` and `predictions.csv` to `code/experiment_outputs/`

**Compute comprehensive metrics (macro-averaged P/R/F1, parameter sensitivity analysis):**

```bash
python compute_metrics.py
```

**Generate all visualizations:**

```bash
python plot_results.py
python generate_advanced_plots.py
python generate_confusion_matrices.py
```

---

## Module Descriptions

### Core Modules

| Module | Description |
|---|---|
| `knowledge_base.py` | Defines the `Rule` dataclass and `KnowledgeBase` class containing 14 diagnostic rules across 7 equipment types, along with physical feasibility constraints. |
| `inference_engine.py` | Implements `ProbabilisticInferenceEngine` with Bayesian-style confidence updating for matched rules. Normalizes posterior probabilities and ranks faults by confidence. |
| `data_generator.py` | `SyntheticDataGenerator` produces labeled fault/normal samples with configurable Gaussian noise levels for controlled experimentation. |
| `llm_baseline.py` | `SimulatedLLM` mocks an LLM with tunable accuracy (85%) and hallucination rate (15%), mirroring empirical benchmarks for state-of-the-art models in technical reasoning domains. `LLMBaseline` provides the hybrid decision function. |
| `coverage_analyzer.py` | `CoverageAnalyzer` uses Monte-Carlo sampling to estimate what fraction of the physically feasible parameter space triggers at least one diagnostic rule. |
| `experiment_runner.py` | Orchestrates the end-to-end pipeline: data generation → inference → hybrid decision → metrics computation → result export. |

### Analysis & Visualization

| Module | Description |
|---|---|
| `compute_metrics.py` | Computes macro-averaged precision, recall, F1 for all methods; per-equipment breakdown; and threshold parameter sensitivity analysis. |
| `plot_results.py` | Generates basic bar charts for coverage and accuracy comparison. |
| `generate_advanced_plots.py` | Produces radar chart, dual-axis sensitivity plot, equipment heatmap, and confidence KDE density plot. |
| `generate_confusion_matrices.py` | Builds and plots per-equipment confusion matrix heatmaps. |
| `analyze_results.py` | Quick per-equipment accuracy summary from `predictions.csv`. |

---

## Experiment Configuration

Key parameters can be adjusted in `experiment_runner.py`:

| Parameter | Default | Description |
|---|---|---|
| `samples_per_eq` (coverage) | 2000 | Monte-Carlo samples for coverage estimation |
| `samples_per_eq` (data) | 500 | Synthetic samples per equipment per noise level |
| `noise_levels` | (0.0, 0.05, 0.10) | Gaussian noise factors applied to sensor parameters |
| `tau` | 0.6 | Confidence threshold for LLM fallback routing |
| Expert weight | 0.7 | Weight of expert prediction in hybrid ensemble |
| LLM weight | 0.3 | Weight of LLM prediction in hybrid ensemble |

LLM simulation parameters in `llm_baseline.py`:

| Parameter | Default | Description |
|---|---|---|
| `accuracy` | 0.85 | Probability of returning the correct fault |
| `hallucination_rate` | 0.15 | Probability of returning an unrelated fault |

---

## Sample Output

```
Rule coverage percentages:
  Boiler: 17.7%
  Chiller: 27.1%
  HVAC: 30.1%
  Air Compressor: 12.95%
  Vacuum Machine: 35.3%
  Power Distribution: 100.0%
  Water Supply: 28.15%

Generated 10500 synthetic samples.
Metrics written to code/experiment_outputs/metrics.json
Raw predictions written to code/experiment_outputs/predictions.csv
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{author2026probabilistic,
  title   = {Enhancing Industrial Fault Diagnosis: A Probabilistic Expert
             System with LLM-Augmented Validation},
  author  = {[Yunping Li, Xin Tang, Yinbo Dai, Hongyu Gao, Liyu Qian, Zhaoxiang Zang]},
  journal = {[Journal]},
  year    = {2026}
}
```

---

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
