"""
Feature-ablation experiment for DAYPSCI cross-batch anomaly detection.

Compares:
  A) All eligible features
  B) Without scan/order identifiers (scan_id, event_order, event_id)
  C) Without all event-position fields (scan_id, event_order, event_id, gemma)

Thresholds are selected on pn_sensor_T1_batch02 validation data.
Final metrics are reported on pn_sensor_T2_batch03 test data.
The test batch is not used for threshold selection.

Run:
    python train_daypsci_ablation.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path(
    r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1"
)
VALIDATION_BATCH = "pn_sensor_T1_batch02"
TEST_BATCH = "pn_sensor_T2_batch03"
OUTPUT_DIR = Path(__file__).resolve().parent

EXCLUDE = {
    "anomaly", "fault", "FSx", "TFSx", "FCVx", "TFCVx",
    "ts_plc", "event_seq", "batch_id", "experiment_id", "part_id",
}
SIGNALS = [
    "a0", "a1", "b0", "b1", "c0", "c1", "B_1", "B_2",
    "YA_p", "YA_m", "YB_p", "YB_m", "YC_p", "YC_m",
]


def find_csv(folder):
    files = list(folder.glob("*.csv"))
    for f in files:
        if folder.name in f.name:
            return f
    return files[0] if files else None


def load_data():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_ROOT}\n"
            "Update DATA_ROOT in this script."
        )
    frames = []
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        path = find_csv(folder)
        if path is None:
            continue
        df = pd.read_csv(path, sep=";", low_memory=False)
        if "anomaly" not in df.columns:
            continue
        df["source_batch"] = folder.name
        df["target_anomaly"] = (
            pd.to_numeric(df["anomaly"], errors="coerce").fillna(0) != 0
        ).astype(int)
        frames.append(df)
        print(f"Loaded {folder.name}: {len(df)} rows")
    if not frames:
        raise RuntimeError("No valid batch CSV files found.")
    return pd.concat(frames, ignore_index=True, sort=False)


def temporal_features(df):
    df = df.copy()
    for col in ["delta_time_ms", "scan_id", "event_order", "event_id"] + SIGNALS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sort_cols = [
        c for c in ["source_batch", "part_id", "scan_id", "event_order", "event_seq"]
        if c in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").copy()
    groups = ["source_batch"] + (["part_id"] if "part_id" in df.columns else [])
    grouped = df.groupby(groups, sort=False, dropna=False)

    if "delta_time_ms" in df.columns:
        previous = grouped["delta_time_ms"].shift(1)
        df["prev_delta_time_ms"] = previous
        df["delta_change_ms"] = df["delta_time_ms"] - previous
        df["delta_ratio_prev"] = df["delta_time_ms"] / previous.replace(0, np.nan)
        df["delta_rolling_mean_3"] = grouped["delta_time_ms"].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
        df["delta_rolling_max_3"] = grouped["delta_time_ms"].transform(
            lambda s: s.rolling(3, min_periods=1).max()
        )

    changed = []
    for signal in SIGNALS:
        if signal in df.columns:
            previous = grouped[signal].shift(1)
            name = signal + "_changed"
            df[name] = ((df[signal] != previous) & previous.notna()).astype(int)
            changed.append(name)
    if changed:
        df["signal_change_count"] = df[changed].sum(axis=1)
    return df


def create_model(X_train):
    numeric = [c for c in X_train.columns if pd.api.types.is_numeric_dtype(X_train[c])]
    categorical = [c for c in X_train.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("prep", prep),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def class_one_probability(model, X):
    classes = list(model.named_steps["rf"].classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def best_threshold(y, probabilities):
    rows = []
    for threshold in np.arange(0.10, 0.901, 0.025):
        pred = (probabilities >= threshold).astype(int)
        rows.append((
            f1_score(y, pred, zero_division=0),
            precision_score(y, pred, zero_division=0),
            threshold,
            recall_score(y, pred, zero_division=0),
        ))
    # Highest F1, then precision, then threshold.
    rows.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    f1, precision, threshold, recall = rows[0]
    return threshold, precision, recall, f1


def main():
    raw = load_data()
    data = temporal_features(raw)

    train = data[~data["source_batch"].isin([VALIDATION_BATCH, TEST_BATCH])].copy()
    val = data[data["source_batch"] == VALIDATION_BATCH].copy()
    test = data[data["source_batch"] == TEST_BATCH].copy()
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Training/validation/test split is empty; verify batch names.")

    excluded = EXCLUDE | {"source_batch", "target_anomaly"}
    base_features = [
        c for c in data.columns
        if c not in excluded
        and train[c].notna().any()
        and train[c].nunique(dropna=True) > 1
    ]

    variants = {
        "all_features": set(),
        "without_scan_order_ids": {"scan_id", "event_order", "event_id"},
        "without_position_and_state": {"scan_id", "event_order", "event_id", "gemma"},
    }

    y_train = train["target_anomaly"].astype(int)
    y_val = val["target_anomaly"].astype(int)
    y_test = test["target_anomaly"].astype(int)

    results = []
    test_outputs = test[
        [c for c in ["source_batch", "part_id", "event_seq", "scan_id", "event_order", "anomaly", "fault"] if c in test.columns]
    ].copy()

    print("\n===== DAYPSCI FEATURE ABLATION =====")
    print(f"Training rows: {len(train)} across {train['source_batch'].nunique()} batches")
    print(f"Validation batch: {VALIDATION_BATCH} ({len(val)} rows)")
    print(f"Final test batch: {TEST_BATCH} ({len(test)} rows)")

    for variant_name, removed in variants.items():
        features = [c for c in base_features if c not in removed]
        X_train = train[features].copy()
        X_val = val[features].copy()
        X_test = test[features].copy()

        model = create_model(X_train)
        model.fit(X_train, y_train)

        val_prob = class_one_probability(model, X_val)
        threshold, val_precision, val_recall, val_f1 = best_threshold(y_val, val_prob)

        test_prob = class_one_probability(model, X_test)
        test_pred = (test_prob >= threshold).astype(int)

        acc = accuracy_score(y_test, test_pred)
        prec = precision_score(y_test, test_pred, zero_division=0)
        rec = recall_score(y_test, test_pred, zero_division=0)
        f1 = f1_score(y_test, test_pred, zero_division=0)

        results.append({
            "variant": variant_name,
            "feature_count": len(features),
            "threshold_selected_on_validation": threshold,
            "validation_precision": val_precision,
            "validation_recall": val_recall,
            "validation_f1": val_f1,
            "test_accuracy": acc,
            "test_anomaly_precision": prec,
            "test_anomaly_recall": rec,
            "test_anomaly_f1": f1,
        })

        test_outputs[f"{variant_name}_predicted"] = test_pred
        test_outputs[f"{variant_name}_probability"] = test_prob

        print(f"\n--- {variant_name} ---")
        print(f"Features: {len(features)} | threshold: {threshold:.3f}")
        print(f"Validation F1: {val_f1:.4f}")
        print("Test confusion matrix:")
        print(confusion_matrix(y_test, test_pred, labels=[0, 1]))
        print(
            f"Test accuracy={acc:.4f} | precision={prec:.4f} | "
            f"recall={rec:.4f} | F1={f1:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("test_anomaly_f1", ascending=False)
    results_path = OUTPUT_DIR / "daypsci_ablation_results.csv"
    predictions_path = OUTPUT_DIR / "daypsci_ablation_predictions.csv"
    results_df.to_csv(results_path, index=False)
    test_outputs["actual_anomaly_binary"] = y_test.to_numpy()
    test_outputs.to_csv(predictions_path, index=False)

    print("\n===== COMPARISON (test batch) =====")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison: {results_path}")
    print(f"Saved predictions: {predictions_path}")
    print(
        "\nInterpretation: use validation results to choose a model variant in a real study. "
        "The test scores shown here are for comparison and should not be used to repeatedly "
        "tune the model against the same final test batch."
    )


if __name__ == "__main__":
    main()
