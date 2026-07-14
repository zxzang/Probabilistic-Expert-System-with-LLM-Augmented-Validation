# Probabilistic Expert System with LLM-Augmented Validation

> **Enhancing Industrial Fault Diagnosis: A Probabilistic Expert System with LLM‑Augmented Validation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A **neuro-symbolic** framework for industrial equipment fault diagnosis that combines a deterministic, physics-grounded rule-based expert system with a Large Language Model (LLM) fallback mechanism. The system emphasizes auditable symbolic reasoning, calibrated fallback routing, and public-benchmark validation at the abnormal-parameter-to-fault diagnostic layer.

---

## Highlights

- 🏭 **7 Equipment Types** — Boiler, Chiller, HVAC, Air Compressor, Vacuum Machine, Power Distribution, Water Supply
- 🧠 **Bayesian-Style Inference** — Pattern Matching with Scoring (PMS) engine that handles partial symptom matches
- 🤖 **LLM Fallback** — Expert-first routing uses legacy weighted selection on the controlled synthetic benchmark and evidence-aware arbitration on public data
- 🌐 **Real LLM Support** — Optional DeepSeek (`deepseek-v4-pro`) experiments through environment-variable API configuration
- 🧪 **Public Benchmarks** — Optional LBNL FDD (Boiler Plant, Chiller Plant, RTU) and MetroPT validation scripts
- 🔍 **Incipient Fault Detection** — Detects 55.6% of partial-symptom faults via LLM fallback (vs 0.0% expert-only)
- 📊 **Coverage Analysis** — Reports synthetic Monte-Carlo rule-space reachability and public predicate-space activation separately from diagnostic accuracy
- 🔬 **Fully Reproducible Core** — Synthetic data generation with configurable noise levels; no external API keys required for mock experiments

## Key Results

| Method | Overall Acc | Fault Acc | Macro-Prec | Macro-Rec | Macro-F1 | ECE | Coverage / Output Availability |
|---|---:|---:|---:|---:|---:|---:|---|
| Pure Expert System | 0.933 | 0.881 | 0.829 | 0.828 | 0.814 | 0.146 | 92.6% rule-space coverage |
| Pure LLM (Mock) | 0.854 | 0.858 | 0.854 | 0.854 | 0.854 | — | N/A |
| **Hybrid (Ours)** | 0.862 | **0.946** | 0.829 | **0.912** | **0.855** | 0.182 | 100% routed diagnostic coverage |

**Real LLM synthetic subset** (`deepseek-v4-pro`, 100 fault-only samples):

| Method | Accuracy | Fault-only Acc | Macro-F1 |
|---|---:|---:|---:|
| Expert | 0.95 | 0.95 | 0.947 |
| Real LLM | 0.93 | 0.93 | 0.9124 |
| **Hybrid** | **0.96** | **0.96** | **0.9461** |

**Public benchmark** (`kb-guided` Real LLM outputs, 104 samples from LBNL Boiler/Chiller/RTU and MetroPT):

| Method | Accuracy | Fault-only Acc | Macro-F1 | False-alarm Rate |
|---|---:|---:|---:|---:|
| Expert | 0.808 | 0.763 | 0.594 | 0.071 |
| Real LLM | 0.702 | 0.632 | 0.555 | 0.107 |
| **Hybrid** | **0.779** | **0.724** | **0.564** | **0.071** |

**Incipient Fault Detection** (`min_match = 75%`, 12,245 samples incl. 1,745 partial-fault):

| Method | Normal Acc | Fault Acc | Partial Fault Det. | Overall Acc |
|---|---|---|---|---|
| Expert-only | 0.960 | 0.885 | 0.000 | 0.802 |
| **Hybrid** | **0.822** | **0.949** | **0.556** | **0.821** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Real-Time IIoT Sensor Data                 │
└───────────────────────┬─────────────────────────────────┘
                        ▼
           ┌────────────────────────┐
           │    Knowledge Base      │  ◄── 7 equipment types,
           │  (knowledge_base.py)   │      30 diagnostic rules,
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
           │    Hybrid Arbitration
           │    (weighted synthetic / evidence-aware public)
           ▼                   ▼
    ┌──────────────────────────────────┐
    │     Final Diagnostic Output      │
    └──────────────────────────────────┘
```

The diagram shows the common expert-first flow. Controlled synthetic experiments use weighted selection; public benchmark experiments use evidence-aware arbitration with source-specific thresholds.

---

## Repository Structure

```
.
├── README.md                       # This file
├── LICENSE                         # Apache License 2.0
├── requirements.txt                # Python dependencies
├── .env.example                    # DeepSeek API configuration template
│
├── code/                           # Main source code
│   ├── knowledge_base.py           # Rule definitions & physical constraints (7 equipment types)
│   ├── inference_engine.py         # PMS inference engine (min_match configurable)
│   ├── data_generator.py           # Synthetic data generator (normal + partial faults)
│   ├── llm_baseline.py             # Simulated LLM baseline & hybrid decision logic
│   ├── real_llm.py                 # DeepSeek real-LLM client (optional API-backed experiments)
│   ├── env_utils.py                # Lightweight .env loader for API configuration
│   ├── public_data_benchmark.py    # LBNL FDD + MetroPT public benchmark runner
│   ├── real_llm_experiment.py      # Synthetic subset experiment with real DeepSeek calls
│   ├── hyperparameter_sensitivity.py  # Alpha/beta sensitivity + per-equipment confidence plots
│   ├── statistical_significance.py # Repeated-seed mean/std/95% CI analysis
│   ├── ml_baselines.py             # Scikit-learn/XGBoost ML baselines on synthetic and public data
│   ├── ml_rule_mining.py           # Shallow-tree public benchmark rule-base audit helper
│   ├── latency_scalability.py      # PMS latency and rule-base scalability benchmark
│   ├── export_rebuttal_evidence.py # Markdown summary exporter for rebuttal evidence
│   ├── recompute_public_llm_metrics.py # Recompute public Real-LLM metrics from saved outputs
│   ├── rtu_real_llm_smoke.py       # Targeted RTU Real-LLM smoke/full-sample helper
│   ├── experiment_metrics.py       # Shared metric helpers for reviewer-response experiments
│   ├── coverage_analyzer.py        # Monte-Carlo partial-match coverage estimation
│   ├── experiment_runner.py        # Main experiment pipeline (10,500 samples)
│   ├── partial_fault_experiment.py # Incipient fault detection experiment (12,245 samples)
│   ├── compute_metrics.py          # Macro-averaged P/R/F1, sensitivity analysis, ECE
│   ├── plot_results.py             # Basic visualization (coverage & accuracy bar charts)
│   ├── analyze_results.py          # Quick per-equipment accuracy summary
│   └── experiment_outputs/         # Generated experiment results
│       ├── metrics.json
│       ├── comprehensive_metrics.json
│       ├── partial_fault_results.json
│       ├── real_llm_synthetic_metrics.json
│       ├── public_benchmark_metrics_real_llm_kb-guided.json
│       ├── public_ml_baseline_metrics.json
│       ├── rebuttal_evidence_summary.md
│       ├── predictions.csv
│       ├── latency_scalability.csv
│       ├── latency_scalability.json
│       └──  accuracy_comparison.png

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

The quick-start commands below first enter `code/`. Later sections use commands from the repository root, for example `python code/public_data_benchmark.py`; if you are already inside `code/`, omit the `code/` prefix.

**Run the full experiment pipeline:**

```bash
cd code
python experiment_runner.py
```

This will:
1. Estimate partial-match rule coverage via Monte-Carlo sampling (N=2,000 per equipment)
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

## Optional Real-LLM Experiments

The core experiments use `SimulatedLLM` and do not require an API key. To run the reviewer-response experiments with a real LLM, configure DeepSeek credentials:

```bash
cp .env.example .env
# Edit .env and set DEEPSEEK_API_KEY
```

Expected variables:

| Variable | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | — | Required for real-LLM experiments |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1/chat/completions` | OpenAI-compatible chat completion endpoint |
| `DEEPSEEK_MODEL` | `deepseek-v4-pro` | Model used for real-LLM validation |
| `DEEPSEEK_TIMEOUT` | `60` | Request timeout in seconds |

Run a small synthetic subset with DeepSeek:

```bash
python code/real_llm_experiment.py --sample-size 100 --fault-only
```

Synthetic real-LLM prompts default to `--prompt-mode rule-guided`, which adds
KB-derived fault-pattern guidance and a structured abnormal-parameter summary.
Use `--prompt-mode threshold-only` for the older threshold-table-only ablation:

```bash
python code/real_llm_experiment.py --sample-size 100 --fault-only --prompt-mode threshold-only
```

For smoke testing with minimal API cost:

```bash
python code/real_llm_experiment.py --sample-size 2 --samples-per-equipment 10 --fault-only
```

Outputs are written to:

- `code/experiment_outputs/real_llm_synthetic_predictions.csv`
- `code/experiment_outputs/real_llm_synthetic_metrics.json`

---

## Optional Public Data Benchmark

The public-data benchmark validates the final diagnostic step on downloaded public time-series datasets. It converts raw time-series measurements into abnormal-parameter predicates, then reuses the PMS inference engine and optional LLM fallback. LBNL includes both fault scenarios and fault-free baseline samples, so the benchmark can report false-alarm behavior rather than fault classification alone.

The public datasets are not redistributed with this code repository. Download them from the original public sources, then place the extracted files under the local directories shown below.

Data sources:

- LBNL FDD project data portal: https://faultdetection.lbl.gov/data/
- LBNL FDD OEDI record: https://data.openei.org/submissions/5763
- MetroPT / MetroPT-2 Zenodo record: https://zenodo.org/records/7766691

Expected local data directories:

```text
code/LBNL_FDD_Data_Sets/
code/MetroPT_dataset/
```

Selected public datasets:

| Dataset | Equipment Role | Validation Purpose |
|---|---|---|
| LBNL FDD Boiler Plant | Boiler | Public benchmark for boiler plant faults |
| LBNL FDD Chiller Plant | Chiller | Public benchmark for chiller/cooling-plant faults |
| LBNL FDD RTU | HVAC / air-conditioning | Public benchmark for rooftop-unit faults |
| MetroPT | Air compressor / pneumatic system | Real-world compressor/APU case study |

Run with no LLM calls:

```bash
python code/public_data_benchmark.py --llm none
```

Run with the mock LLM:

```bash
python code/public_data_benchmark.py --llm mock
```

Run with DeepSeek, limiting API calls during testing:

```bash
python code/public_data_benchmark.py --llm real --llm-prompt-mode kb-guided --max-api-calls 150
```

For public benchmark real-LLM calls, `--llm-prompt-mode` controls how much
diagnostic context is supplied:

- `zero-shot`: predicate values and candidate faults only; no KB guide and no
  PMS expert context.
- `kb-guided` (default): adds equipment-specific fault-indicator mappings,
  predicate descriptions, and fault-level evidence groups with mechanism
  sufficiency constraints, but does not reveal the PMS top diagnosis or top-k
  reasoning.
- `evidence-rich-kb`: adds the same KB guide and predicate values as
  `kb-guided`, then appends raw public benchmark statistical deviations and
  mechanism notes. It does not reveal PMS top-k outputs, so it is an
  independent rich-evidence LLM ablation rather than an expert-guided setting.
- `expert-guided`: adds the KB guide plus the PMS expert system's top-1/top-k
  matched rule details. Use this as an expert-assisted LLM validation setting,
  not as an independent LLM baseline.
- `rich-context`: hides the expert predicates and instead supplies raw public
  benchmark statistical summaries, healthy-baseline deviations, trend features,
  and short mechanism notes. Use this as a complementary rich-information LLM
  ablation; it does not reveal scenario filenames or PMS expert outputs.

Public real-LLM prompts ask the model to perform an abnormal-sufficiency
self-check before assigning a fault. The prediction CSV records
`llm_abnormal_decision` and `llm_evidence_sufficiency` so Normal/weak-evidence
false alarms can be audited without adding a post-processing calibration layer.

Examples:

```bash
python code/public_data_benchmark.py --llm real --llm-prompt-mode zero-shot --max-api-calls 150
python code/public_data_benchmark.py --llm real --llm-prompt-mode kb-guided --max-api-calls 150
python code/public_data_benchmark.py --llm real --llm-prompt-mode evidence-rich-kb --max-api-calls 150
python code/public_data_benchmark.py --llm real --llm-prompt-mode expert-guided --max-api-calls 150
python code/public_data_benchmark.py --llm real --llm-prompt-mode rich-context --max-api-calls 150
```

The public-data runner uses evidence-aware hybrid arbitration by default:

```bash
python code/public_data_benchmark.py --llm real --llm-prompt-mode kb-guided --hybrid-strategy evidence-aware --max-api-calls 150
```

This strategy keeps the original expert diagnosis for no/weak-evidence cases
and only allows a confident non-Normal/non-Unknown LLM diagnosis to override
the expert on strong-evidence conflicts. The original weighted-confidence
Hybrid is still available for ablation:

```bash
python code/public_data_benchmark.py --llm real --llm-prompt-mode kb-guided --hybrid-strategy legacy --max-api-calls 150
```

By default, the public-data runner uses an equipment-specific
`--threshold-profile public-tuned` configuration rather than the synthetic
benchmark's single global routing threshold. The public profile uses
multi-level predicate-aware `min_match` values and lower per-equipment `tau`
values so that sparse but plausible expert diagnoses are not unnecessarily
overruled by the LLM. The effective values are written to each prediction row as
`effective_min_match` and `effective_tau`.

To reproduce the earlier global public benchmark setting:

```bash
python code/public_data_benchmark.py --llm real --llm-prompt-mode kb-guided --threshold-profile global --min-match 50 --tau 0.6 --max-api-calls 150
```

When `--max-api-calls` is smaller than the full benchmark size, use the
`llm_called_only` field in the metrics JSON for the standalone LLM score; the
overall hybrid score still reflects the complete evaluated sample set.

Outputs are written to:

- `code/experiment_outputs/public_benchmark_predictions_<mode>.csv`
- `code/experiment_outputs/public_benchmark_metrics_<mode>.json`

For real-LLM public benchmark outputs, every prompt mode is included in the
file suffix: `real_llm_kb-guided`, `real_llm_zero-shot`, or
`real_llm_evidence-rich-kb`, `real_llm_expert-guided`, or
`real_llm_rich-context`.

The metrics JSON reports accuracy, fault-only accuracy, macro-F1, false-alarm rate, fallback-needed rate, ECE for expert/LLM/hybrid predictions, arbitration-action breakdowns, per-source breakdowns, public rule-base PMS activation/full-match coverage over the multi-level predicate space, MetroPT window-level time-to-detection, threshold usage, expert-only `min_match` sensitivity, and recorded-output `tau` sensitivity. `llm_evaluation_rate` records how many samples were actually sent to the LLM baseline, while `fallback_rate` records how often the expert confidence fell below the effective routing threshold. Overall ECE is the primary calibration statistic; per-source ECE is included for transparency and is noisier for small LBNL scenario subsets.

The public benchmark is intentionally a diagnostic validation layer, not a new forecasting method. LBNL fault scenarios are compared with fault-free baselines to support scenario-level fault classification and normal-operation false-alarm checks. LBNL files not mapped to the compact public rule taxonomy are excluded from the default known-class benchmark. When `--max-scenarios-per-source` limits LBNL files, the runner samples by fault label first so that the default benchmark does not silently drop an entire represented fault type because of filename order. MetroPT is segmented into simple time windows to produce interpretable abnormal predicates and approximate time-to-detection from the first correctly diagnosed fault window.

The public predicate extractor includes direct threshold predicates and a small
set of feature-rich derived predicates, such as hot/chilled-water temperature
lift loss, condenser-side temperature lift, RTU supply-air and refrigerant
pressure imbalance, MetroPT reservoir-pressure level, and motor-current trend.
These derived predicates are deterministic healthy-baseline comparison
features, not supervised model outputs.

---

## Reviewer-Response Experiments

The following scripts provide additional robustness, calibration, and public-benchmark validation evidence.

**Alpha/beta sensitivity and per-equipment confidence distributions:**

```bash
python code/hyperparameter_sensitivity.py
```

Outputs:

- `code/experiment_outputs/alpha_beta_sensitivity.csv`
- `code/experiment_outputs/alpha_beta_sensitivity.json`
- `code/experiment_outputs/alpha_beta_hybrid_macro_f1.png`
- `code/experiment_outputs/alpha_beta_hybrid_ece.png`
- `code/experiment_outputs/per_equipment_confidence_hist.png`

**Repeated-seed statistical significance analysis:**

```bash
python code/statistical_significance.py
```

Outputs:

- `code/experiment_outputs/statistical_significance_runs.csv`
- `code/experiment_outputs/statistical_significance_summary.json`

**Machine-learning baselines:**

```bash
python code/ml_baselines.py --dataset both
```

This script trains tabular baselines on the same abnormal-parameter mapping
layer used by the PMS/LLM experiments. The default `--dataset synthetic`
preserves the original synthetic-only behavior; `--dataset public` evaluates
the LBNL/MetroPT extracted predicates; `--dataset both` runs both.

- Random Forest
- Extra Trees
- Logistic Regression
- Decision Tree
- Gradient Boosting
- MLP
- XGBoost

Outputs:

- `code/experiment_outputs/ml_baseline_metrics.csv`
- `code/experiment_outputs/ml_baseline_metrics.json`
- `code/experiment_outputs/ml_baseline_predictions.csv`
- `code/experiment_outputs/public_ml_baseline_metrics.csv`
- `code/experiment_outputs/public_ml_baseline_metrics.json`
- `code/experiment_outputs/public_ml_baseline_predictions.csv`

Because the synthetic and public benchmarks are evaluated after abnormal
parameter extraction, these are tabular predicate/parameter baselines rather
than raw time-series forecasting models. Public ML baselines use the same
multi-level predicates as `public_data_benchmark.py`; rare classes with fewer
than two samples are excluded from the supervised train/test split and reported
in the JSON metadata.

**Data-assisted public rule-base audit:**

```bash
python code/ml_rule_mining.py
```

This helper trains shallow decision trees on public benchmark predicates and
exports high-purity decision paths plus repeated expert-error signatures. The
mined paths are used only as an audit aid for human rule refinement; they are
not used as a runtime classifier.

Outputs:

- `code/experiment_outputs/ml_rule_mining_public.json`
- `code/experiment_outputs/ml_rule_mining_public.md`

For conda environments on Windows, XGBoost can be installed with:

```bash
conda install -c conda-forge py-xgboost
```

**Latency and scalability benchmark:**

```bash
python code/latency_scalability.py
```

This measures PMS inference latency as the rule base is synthetically scaled, plus mock-LLM latency. It does not call DeepSeek by default. To measure a small number of real API calls:

```bash
python code/latency_scalability.py --real-llm-calls 3
```

Outputs:

- `code/experiment_outputs/latency_scalability.csv`
- `code/experiment_outputs/latency_scalability.json`

**Export a rebuttal evidence summary:**

```bash
python code/export_rebuttal_evidence.py
```

Output:

- `code/experiment_outputs/rebuttal_evidence_summary.md`

---

## Formal Experiment Run Order

To reproduce the complete experiment suite, run the scripts in the following order. The default evidence bundle uses a **100-sample synthetic Real LLM subset** and the **104-sample public `kb-guided` Real LLM benchmark**. Use `--max-api-calls 150` for public Real LLM generation when saved outputs are not already available; recomputation from recorded outputs does not consume API calls.

```bash
python code/experiment_runner.py
python code/compute_metrics.py
python code/plot_results.py
python code/partial_fault_experiment.py
python code/hyperparameter_sensitivity.py
python code/statistical_significance.py
python code/ml_baselines.py --dataset both
python code/latency_scalability.py
python code/public_data_benchmark.py --llm none
python code/public_data_benchmark.py --llm mock
python code/real_llm_experiment.py --sample-size 100 --fault-only --prompt-mode rule-guided
python code/public_data_benchmark.py --llm real --llm-prompt-mode kb-guided --max-api-calls 150
# If reusing recorded public Real-LLM outputs instead of making new API calls:
python code/recompute_public_llm_metrics.py --prompt-mode kb-guided
python code/export_rebuttal_evidence.py
```

The two `--llm real` commands require a valid `.env`/environment configuration and may consume DeepSeek API calls. Run the preceding commands first to confirm the local, no-API pipeline is healthy. If recorded public Real LLM prediction CSVs already exist, use `python code/recompute_public_llm_metrics.py --prompt-mode kb-guided` to refresh public metrics without new API calls.

The final command regenerates:

```text
code/experiment_outputs/rebuttal_evidence_summary.md
```

Use this summary as the first checkpoint before updating the manuscript and rebuttal text.

---

## Module Descriptions

### Core Modules

| Module | Description |
|---|---|
| `knowledge_base.py` | Defines the `Rule` dataclass and `KnowledgeBase` class containing 30 diagnostic rules across 7 equipment types, along with physical feasibility constraints. |
| `inference_engine.py` | PMS inference engine with configurable `min_match` threshold (default 75%). Computes S_match, S_conf, and normalized posterior probabilities. |
| `data_generator.py` | `SyntheticDataGenerator` produces normal, fault, and **partial-fault** samples with configurable Gaussian noise and `partial_ratio` (default 0.2). |
| `llm_baseline.py` | `SimulatedLLM` mocks an LLM with tunable accuracy (85%) and hallucination rate (15%). Synthetic experiments pass ground truth only to calibrate the simulated accuracy/hallucination process. |
| `real_llm.py` | Optional DeepSeek API client for real-LLM diagnostic experiments. Reads configuration from `.env` or shell environment variables. |
| `env_utils.py` | Lightweight `.env` loader used by `real_llm.py`; avoids adding a mandatory `python-dotenv` dependency. |
| `coverage_analyzer.py` | Monte-Carlo partial-match coverage estimator (default N=2,000 per equipment) over the feasible state space. |
| `experiment_runner.py` | Main experiment pipeline (10,500 samples): coverage → data gen → expert/LLM/hybrid inference → export. |
| `partial_fault_experiment.py` | Incipient fault experiment (12,245 samples): Expert vs Hybrid at multiple `min_match` thresholds. |
| `real_llm_experiment.py` | Optional synthetic subset experiment using real DeepSeek calls. |
| `public_data_benchmark.py` | Optional LBNL FDD + MetroPT public benchmark runner. |
| `hyperparameter_sensitivity.py` | Alpha/beta sensitivity analysis and per-equipment expert-confidence plots. |
| `statistical_significance.py` | Repeated-seed robustness experiment with mean/std/95% confidence intervals. |
| `ml_baselines.py` | Scikit-learn and XGBoost tabular ML baselines for synthetic abnormal-parameter samples and public LBNL/MetroPT predicate samples. |
| `ml_rule_mining.py` | Shallow decision-tree audit helper that mines public-predicate error patterns for human rule-base refinement. |
| `latency_scalability.py` | Expert-system latency and rule-base scalability benchmark, with optional real-LLM latency calls. |
| `export_rebuttal_evidence.py` | Creates a compact Markdown summary from generated experiment outputs. |
| `recompute_public_llm_metrics.py` | Recomputes public Real-LLM metrics from recorded prediction CSVs without new API calls. |
| `rtu_real_llm_smoke.py` | Targeted RTU Real-LLM smoke/full-sample helper for sparse-predicate prompt checks. |
| `experiment_metrics.py` | Shared metric utilities and the canonical hybrid weighted-selection function. |

### Analysis & Visualization

| Module | Description |
|---|---|
| `compute_metrics.py` | Macro-averaged P/R/F1, fault-only accuracy, per-equipment breakdown, sensitivity analysis, and ECE computation. |
| `plot_results.py` | Basic bar charts for coverage and accuracy comparison. |
| `analyze_results.py` | Quick per-equipment accuracy summary from `predictions.csv`. |

---

## Experiment Configuration

Key parameters can be adjusted in `experiment_runner.py`:

| Parameter | Default | Description |
|---|---|---|
| `samples_per_eq` (coverage) | 2000 | Monte-Carlo samples for partial-match coverage estimation |
| `samples_per_eq` (data) | 500 | Synthetic samples per equipment per noise level |
| `noise_levels` | (0.0, 0.05, 0.10) | Gaussian noise factors applied to sensor parameters |
| `partial_ratio` | 0.0 (main) / 0.2 (partial) | Fraction of samples generated as partial faults |
| `min_match` | 75.0 | Minimum S_match (%) to activate a rule |
| `tau` | 0.6 | Confidence threshold for LLM fallback routing |
| Expert weight | 0.7 | Weight of expert prediction in hybrid weighted selection |
| LLM weight | 0.3 | Weight of LLM prediction in hybrid weighted selection |

LLM simulation parameters in `llm_baseline.py`:

| Parameter | Default | Description |
|---|---|---|
| `accuracy` | 0.85 | Probability of returning the correct fault |
| `hallucination_rate` | 0.15 | Probability of returning an unrelated fault |

---

## Sample Output

```
Rule coverage percentages:
  Boiler: 97.05%
  Chiller: 99.3%
  HVAC: 83.25%
  Air Compressor: 96.65%
  Vacuum Machine: 96.55%
  Power Distribution: 100.0%
  Water Supply: 75.35%

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

This project is licensed under the Apache License 2.0; see the [LICENSE](LICENSE) file for details.
