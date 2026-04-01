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
- 🔍 **Incipient Fault Detection** — Detects 55.4% of partial-symptom faults via LLM fallback (vs 0.2% expert-only)
- 📊 **Monte-Carlo Coverage Analysis** — Quantifies rule-base completeness over the physically feasible state space
- 🔬 **Fully Reproducible** — Synthetic data generation with configurable noise levels; no external API keys required

## Key Results

| Method | Overall Acc | Fault Acc | Macro-Prec | Macro-Rec | Macro-F1 | ECE | Coverage |
|---|---|---|---|---|---|---|---|
| Pure Expert System | 0.932 | 0.869 | 0.825 | 0.824 | 0.809 | 0.146 | 35 % |
| Pure LLM (Mock) | 0.847 | 0.841 | 1.000 | 0.855 | 0.920 | — | 100 % |
| **Hybrid (Ours)** | 0.861 | **0.940** | 0.843 | **0.914** | **0.861** | **0.193** | **100 %** |

**Incipient Fault Detection** (`min_match = 75%`, 12,245 samples incl. 1,745 partial-fault):

| Method | Normal Acc | Fault Acc | Partial Fault Det. | Overall Acc |
|---|---|---|---|---|
| Expert-only | 0.961 | 0.877 | 0.002 | 0.800 |
| **Hybrid** | **0.821** | **0.950** | **0.554** | **0.821** |

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
│   ├── knowledge_base.py           # Rule definitions & physical constraints (7 equipment types)
│   ├── inference_engine.py         # PMS inference engine (min_match configurable)
│   ├── data_generator.py           # Synthetic data generator (normal + partial faults)
│   ├── llm_baseline.py             # Simulated LLM baseline & hybrid decision logic
│   ├── coverage_analyzer.py        # Monte-Carlo rule coverage estimation
│   ├── experiment_runner.py        # Main experiment pipeline (10,500 samples)
│   ├── partial_fault_experiment.py # Incipient fault detection experiment (12,245 samples)
│   ├── compute_metrics.py          # Macro-averaged P/R/F1, sensitivity analysis, ECE
│   ├── plot_results.py             # Basic visualization (coverage & accuracy bar charts)
│   ├── generate_advanced_plots.py  # Advanced plots (radar, heatmap, KDE, dual-axis)
│   ├── generate_confusion_matrices.py  # Per-equipment confusion matrix heatmaps
│   ├── analyze_results.py          # Quick per-equipment accuracy summary
│   └── experiment_outputs/         # Generated experiment results
│       ├── metrics.json
│       ├── comprehensive_metrics.json
│       ├── partial_fault_results.json
│       └──  predictions.csv

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
1. Estimate rule coverage via Monte-Carlo sampling (N=2,000 per equipment)
2. Generate 10,500 synthetic samples (500 per equipment × 3 noise levels × 7 equipment types)
3. Run the expert system, LLM baseline, and hybrid model on all samples
4. Write `metrics.json` and `predictions.csv` to `code/experiment_outputs/`

**Compute comprehensive metrics (macro-averaged P/R/F1, parameter sensitivity analysis, ECE):**

```bash
python compute_metrics.py
```

**Run the incipient fault detection experiment:**

```bash
python partial_fault_experiment.py
```

This generates 12,245 samples (10,500 base + 1,745 partial-fault) and compares Expert-only vs Hybrid across multiple `min_match` thresholds.

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
| `inference_engine.py` | PMS inference engine with configurable `min_match` threshold (default 75%). Computes S_match, S_conf, and normalized posterior probabilities. |
| `data_generator.py` | `SyntheticDataGenerator` produces normal, fault, and **partial-fault** samples with configurable Gaussian noise and `partial_ratio` (default 0.2). |
| `llm_baseline.py` | `SimulatedLLM` mocks an LLM with tunable accuracy (85%) and hallucination rate (15%). `LLMBaseline` provides the hybrid decision function. |
| `coverage_analyzer.py` | Monte-Carlo coverage estimator (default N=2,000 per equipment) for rule-base completeness over the feasible state space. |
| `experiment_runner.py` | Main experiment pipeline (10,500 samples): coverage → data gen → expert/LLM/hybrid inference → export. |
| `partial_fault_experiment.py` | Incipient fault experiment (12,245 samples): Expert vs Hybrid at multiple `min_match` thresholds. |

### Analysis & Visualization

| Module | Description |
|---|---|
| `compute_metrics.py` | Macro-averaged P/R/F1, fault-only accuracy, per-equipment breakdown, sensitivity analysis, and ECE computation. |
| `plot_results.py` | Basic bar charts for coverage and accuracy comparison. |


---

## Experiment Configuration

Key parameters can be adjusted in `experiment_runner.py`:

| Parameter | Default | Description |
|---|---|---|
| `samples_per_eq` (coverage) | 2000 | Monte-Carlo samples for coverage estimation |
| `samples_per_eq` (data) | 500 | Synthetic samples per equipment per noise level |
| `noise_levels` | (0.0, 0.05, 0.10) | Gaussian noise factors applied to sensor parameters |
| `partial_ratio` | 0.0 (main) / 0.2 (partial) | Fraction of samples generated as partial faults |
| `min_match` | 75.0 | Minimum S_match (%) to activate a rule |
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
  Boiler: 15.8%
  Chiller: 36.3%
  HVAC: 32.9%
  Air Compressor: 13.0%
  Vacuum Machine: 45.7%
  Power Distribution: 67.3%
  Water Supply: 32.6%

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
