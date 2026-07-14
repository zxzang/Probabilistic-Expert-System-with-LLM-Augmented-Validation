"""Machine-learning baselines for abnormal-parameter fault diagnosis.

The synthetic and public-data benchmarks are both evaluated at the same
abnormal-parameter mapping layer as the proposed PMS/LLM system. These
baselines therefore use tabular features extracted by the project code rather
than raw time-series forecasting inputs. If xgboost is installed, an XGBoost
baseline is included automatically.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
from data_generator import SyntheticDataGenerator
from experiment_metrics import summarize


def flatten_synthetic_dataset(dataset: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in dataset:
        row: Dict[str, Any] = {
            "source": "Synthetic",
            "equipment": item["equipment"],
            "sample_type": item.get("sample_type", ""),
            "label": item["fault"] if item["fault"] is not None else "Normal",
        }
        for key, value in item["params"].items():
            row[f"param__{key}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    param_cols = [c for c in df.columns if c.startswith("param__")]
    df[param_cols] = df[param_cols].fillna(0.0)
    return df


def flatten_public_samples(samples: List[Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        row: Dict[str, Any] = {
            "source": sample.source,
            "equipment": sample.equipment,
            "sample_type": sample.scenario,
            "label": sample.true_fault if sample.true_fault is not None else "Normal",
        }
        for key, value in sample.params.items():
            row[f"param__{key}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    param_cols = [c for c in df.columns if c.startswith("param__")]
    df[param_cols] = df[param_cols].fillna(0.0)
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    # `sample_type` is metadata derived from the generator and would leak label
    # information, so ML baselines use only equipment identity and parameters.
    feature_df = df.drop(columns=["label", "sample_type"], errors="ignore")
    return pd.get_dummies(feature_df, columns=["source", "equipment"], dummy_na=False)


def model_specs(seed: int) -> Dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.neural_network import MLPClassifier

    specs: Dict[str, Any] = {
        "LogisticRegression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=None,
            class_weight="balanced",
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=None,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=seed),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=350,
            random_state=seed,
            early_stopping=False,
        ),
    }
    try:
        from xgboost import XGBClassifier

        specs["XGBoost"] = XGBClassifier(
            n_estimators=250,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass
    return specs


def build_synthetic_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    generator = SyntheticDataGenerator(
        samples_per_eq=args.samples_per_equipment,
        noise_levels=tuple(args.noise_levels),
        partial_ratio=args.partial_ratio,
    )
    return flatten_synthetic_dataset(generator.generate())


def build_public_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    try:
        from public_data_benchmark import build_lbnl_samples, build_metropt_samples
    except ImportError as exc:
        raise SystemExit("public_data_benchmark.py is required for public ML baselines.") from exc

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
    return flatten_public_samples(samples)


def filter_supervised_classes(df: pd.DataFrame, min_class_count: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    counts = df["label"].astype(str).value_counts()
    kept_labels = sorted(counts[counts >= min_class_count].index)
    filtered = df[df["label"].astype(str).isin(kept_labels)].copy()
    return filtered, {
        "original_samples": int(len(df)),
        "used_samples": int(len(filtered)),
        "excluded_samples": int(len(df) - len(filtered)),
        "original_classes": int(len(counts)),
        "used_classes": int(len(kept_labels)),
        "min_class_count": int(min_class_count),
        "excluded_classes": sorted(counts[counts < min_class_count].index),
    }


def text_records(y_true_text: List[str], y_pred_text: List[str]) -> List[Dict[str, Any]]:
    return [
        {"true_fault": None if t == "Normal" else t, "pred_fault": None if p == "Normal" else p}
        for t, p in zip(y_true_text, y_pred_text)
    ]


def evaluate_holdout(
    df: pd.DataFrame,
    seed: int,
    test_size: float,
    scope: str,
    require_stratified: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    X = make_features(df)
    y_text = df["label"].astype(str)

    class_counts = y_text.value_counts()
    if len(class_counts) < 2:
        return [], [], {
            "scope": scope,
            "status": "skipped",
            "reason": "At least two classes are required for supervised classification.",
            "samples": int(len(df)),
            "classes": int(len(class_counts)),
        }
    use_stratify = class_counts.min() >= 2
    if require_stratified and not use_stratify:
        return [], [], {
            "scope": scope,
            "status": "skipped",
            "reason": "At least two classes with at least two samples each are required for stratified holdout.",
            "samples": int(len(df)),
            "classes": int(len(class_counts)),
        }

    n_classes = len(class_counts)
    n_samples = len(df)
    if use_stratify:
        n_test = max(ceil(n_samples * test_size), n_classes)
        n_train = n_samples - n_test
        if n_train < n_classes:
            n_test = n_samples - n_classes
        if n_test < n_classes or n_test <= 0:
            return [], [], {
                "scope": scope,
                "status": "skipped",
                "reason": "Not enough samples for a stratified train/test split.",
                "samples": int(n_samples),
                "classes": int(n_classes),
            }
    else:
        n_test = max(1, ceil(n_samples * test_size))
    effective_test_size = n_test / n_samples

    X_train, X_test, y_train_text, y_test_text = train_test_split(
        X,
        y_text,
        test_size=effective_test_size,
        random_state=seed,
        stratify=y_text if use_stratify else None,
    )
    rows: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    for name, model in model_specs(seed).items():
        model = deepcopy(model)
        if name == "XGBoost":
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train_text)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_text = encoder.inverse_transform(y_pred.astype(int))
        else:
            model.fit(X_train, y_train_text)
            y_pred_text = model.predict(X_test)

        y_true_list = list(y_test_text)
        y_pred_list = [str(v) for v in y_pred_text]
        records = text_records(y_true_list, y_pred_list)
        base_metrics = summarize(records, "pred_fault")
        row = {
            "scope": scope,
            "model": name,
            "samples": int(n_samples),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "classes": int(n_classes),
            "test_size": round(effective_test_size, 4),
            "accuracy": round(accuracy_score(y_true_list, y_pred_list), 4),
            "weighted_precision": round(precision_score(y_true_list, y_pred_list, average="weighted", zero_division=0), 4),
            "weighted_recall": round(recall_score(y_true_list, y_pred_list, average="weighted", zero_division=0), 4),
            "weighted_f1": round(f1_score(y_true_list, y_pred_list, average="weighted", zero_division=0), 4),
            "fault_only_accuracy": base_metrics["fault_only_accuracy"],
            "macro_precision_faults": base_metrics["macro_precision"],
            "macro_recall_faults": base_metrics["macro_recall"],
            "macro_f1_faults": base_metrics["macro_f1"],
        }
        rows.append(row)
        for true_label, pred_label in zip(y_true_list, y_pred_list):
            predictions.append({"scope": scope, "model": name, "true_label": true_label, "pred_label": pred_label})

    split_info = {
        "scope": scope,
        "status": "evaluated",
        "samples": int(n_samples),
        "classes": int(n_classes),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "effective_test_size": round(effective_test_size, 4),
        "stratified": bool(use_stratify),
    }
    return rows, predictions, split_info


def write_outputs(
    rows: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    json_payload: Dict[str, Any],
    metrics_name: str,
    predictions_name: str,
    json_name: str,
) -> None:
    out_dir = Path(__file__).parent / "experiment_outputs"
    out_dir.mkdir(exist_ok=True)
    metrics_path = out_dir / metrics_name
    pred_path = out_dir / predictions_name
    json_path = out_dir / json_name
    metric_fields = list(rows[0].keys()) if rows else [
        "scope", "model", "samples", "train_samples", "test_samples", "classes",
        "test_size", "accuracy", "weighted_precision", "weighted_recall",
        "weighted_f1", "fault_only_accuracy", "macro_precision_faults",
        "macro_recall_faults", "macro_f1_faults",
    ]
    prediction_fields = list(predictions[0].keys()) if predictions else ["scope", "model", "true_label", "pred_label"]
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(rows)
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prediction_fields)
        writer.writeheader()
        writer.writerows(predictions)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2)

    print(json.dumps(json_payload, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {pred_path}")


def run_synthetic(args: argparse.Namespace) -> None:
    df = build_synthetic_dataframe(args)
    rows, predictions, split_info = evaluate_holdout(
        df,
        args.seed,
        args.test_size,
        scope="Synthetic",
        require_stratified=False,
    )
    write_outputs(
        rows,
        predictions,
        {"configuration": vars(args), "dataset": "synthetic", "split": split_info, "results": rows},
        "ml_baseline_metrics.csv",
        "ml_baseline_predictions.csv",
        "ml_baseline_metrics.json",
    )


def run_public(args: argparse.Namespace) -> None:
    df = build_public_dataframe(args)
    filtered, filter_info = filter_supervised_classes(df, args.public_min_class_count)
    all_rows: List[Dict[str, Any]] = []
    all_predictions: List[Dict[str, Any]] = []
    splits: List[Dict[str, Any]] = []

    pooled_rows, pooled_predictions, pooled_split = evaluate_holdout(
        filtered,
        args.seed,
        args.test_size,
        scope="Public pooled",
    )
    all_rows.extend(pooled_rows)
    all_predictions.extend(pooled_predictions)
    splits.append(pooled_split)

    for source in sorted(filtered["source"].unique()):
        source_df = filtered[filtered["source"] == source].copy()
        source_rows, source_predictions, source_split = evaluate_holdout(
            source_df,
            args.seed,
            args.test_size,
            scope=f"Public {source}",
        )
        all_rows.extend(source_rows)
        all_predictions.extend(source_predictions)
        splits.append(source_split)

    write_outputs(
        all_rows,
        all_predictions,
        {
            "configuration": vars(args),
            "dataset": "public",
            "filter": filter_info,
            "splits": splits,
            "results": all_rows,
        },
        "public_ml_baseline_metrics.csv",
        "public_ml_baseline_predictions.csv",
        "public_ml_baseline_metrics.json",
    )


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required for ML baselines. Install with: pip install -r requirements.txt"
        ) from exc

    if args.dataset in {"synthetic", "both"}:
        run_synthetic(args)
    if args.dataset in {"public", "both"}:
        run_public(args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ML baselines on synthetic and/or public abnormal-parameter datasets.")
    parser.add_argument("--dataset", choices=["synthetic", "public", "both"], default="synthetic")
    parser.add_argument("--samples-per-equipment", type=int, default=500)
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.05, 0.10])
    parser.add_argument("--partial-ratio", type=float, default=0.0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--public-max-rows", type=int, default=20000)
    parser.add_argument("--public-stride", type=int, default=60)
    parser.add_argument("--public-max-scenarios-per-source", type=int, default=12)
    parser.add_argument("--public-skip-metropt", action="store_true")
    parser.add_argument("--public-metropt-window-rows", type=int, default=300)
    parser.add_argument("--public-metropt-max-windows-per-class", type=int, default=20)
    parser.add_argument("--public-min-class-count", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
