"""
DAYPSCI baseline false-alarm evaluation and batch-threshold diagnostics.

This script:
1. Trains on all batches except the chosen validation and test batches.
2. Chooses threshold on a mixed-class validation batch only.
3. Evaluates the selected threshold on a mixed-class test batch.
4. Evaluates false alarms on the normal-only baseline batch separately.
5. Reports results across a few fixed thresholds to show the precision/recall trade-off.

Default:
  Validation: pn_sensor_T1_batch02
  Test:       pn_sensor_T2_batch03
  Baseline:   pn_baseline__batch01

Run:
    python train_daypsci_baseline_false_alarms.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path(
    r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1"
)
OUTPUT_DIR = Path(__file__).resolve().parent

VALIDATION_BATCH = "pn_sensor_T1_batch02"
TEST_BATCH = "pn_sensor_T2_batch03"
BASELINE_BATCH = "pn_baseline__batch01"

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
            "Edit DATA_ROOT near the top of this script."
        )
    frames = []
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        csv_path = find_csv(folder)
        if csv_path is None:
            continue
        df = pd.read_csv(csv_path, sep=";", low_memory=False)
        if "anomaly" not in df.columns:
            continue
        df["source_batch"] = folder.name
        df["target_anomaly"] = (
            pd.to_numeric(df["anomaly"], errors="coerce").fillna(0) != 0
        ).astype(int)
        frames.append(df)
        print(
            f"Loaded {folder.name}: rows={len(df)}, "
            f"anomalous={int(df['target_anomaly'].sum())}"
        )
    if not frames:
        raise RuntimeError(f"No usable CSV files found under {DATA_ROOT}")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_temporal_features(df):
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
    group_cols = ["source_batch"]
    if "part_id" in df.columns:
        group_cols.append("part_id")
    grouped = df.groupby(group_cols, sort=False, dropna=False)

    if "delta_time_ms" in df.columns:
        prev = grouped["delta_time_ms"].shift(1)
        df["prev_delta_time_ms"] = prev
        df["delta_change_ms"] = df["delta_time_ms"] - prev
        df["delta_ratio_prev"] = df["delta_time_ms"] / prev.replace(0, np.nan)
        df["delta_rolling_mean_3"] = grouped["delta_time_ms"].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
        df["delta_rolling_max_3"] = grouped["delta_time_ms"].transform(
            lambda s: s.rolling(3, min_periods=1).max()
        )
    change_cols = []
    for signal in SIGNALS:
        if signal in df.columns:
            prev = grouped[signal].shift(1)
            name = signal + "_changed"
            df[name] = ((df[signal] != prev) & prev.notna()).astype(int)
            change_cols.append(name)
    if change_cols:
        df["signal_change_count"] = df[change_cols].sum(axis=1)
    return df


def make_model(X):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def get_probabilities(model, X):
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def main():
    data = add_temporal_features(load_data())
    batch_names = set(data["source_batch"].unique())

    for required in [VALIDATION_BATCH, TEST_BATCH, BASELINE_BATCH]:
        if required not in batch_names:
            raise RuntimeError(f"Required batch missing: {required}")

    train_batches = sorted(batch_names - {VALIDATION_BATCH, TEST_BATCH, BASELINE_BATCH})
    train = data[data["source_batch"].isin(train_batches)].copy()
    val = data[data["source_batch"] == VALIDATION_BATCH].copy()
    test = data[data["source_batch"] == TEST_BATCH].copy()
    baseline = data[data["source_batch"] == BASELINE_BATCH].copy()

    y_train = train["target_anomaly"].astype(int)
    y_val = val["target_anomaly"].astype(int)
    y_test = test["target_anomaly"].astype(int)
    y_base = baseline["target_anomaly"].astype(int)

    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError("Train, validation, and test must contain both classes.")

    excluded = EXCLUDE | {"source_batch", "target_anomaly"}
    features = [
        c for c in data.columns
        if c not in excluded
        and train[c].notna().any()
        and train[c].nunique(dropna=True) > 1
    ]
    X_train, X_val, X_test, X_base = (
        train[features].copy(), val[features].copy(),
        test[features].copy(), baseline[features].copy()
    )

    model = make_model(X_train)
    model.fit(X_train, y_train)

    val_prob = get_probabilities(model, X_val)
    candidates = []
    for threshold in np.arange(0.10, 0.901, 0.025):
        pred = (val_prob >= threshold).astype(int)
        candidates.append({
            "threshold": float(threshold),
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
            "f1": f1_score(y_val, pred, zero_division=0),
        })
    threshold_scores = pd.DataFrame(candidates)
    best = threshold_scores.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False]
    ).iloc[0]
    chosen_threshold = float(best["threshold"])

    test_prob = get_probabilities(model, X_test)
    base_prob = get_probabilities(model, X_base)

    rows = []
    for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, chosen_threshold]:
        test_pred = (test_prob >= threshold).astype(int)
        base_pred = (base_prob >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_f1": f1_score(y_test, test_pred, zero_division=0),
            "test_false_positives": int(((y_test == 0) & (test_pred == 1)).sum()),
            "test_false_negatives": int(((y_test == 1) & (test_pred == 0)).sum()),
            "baseline_false_positives": int(base_pred.sum()),
            "baseline_total": len(base_pred),
            "baseline_false_alarm_rate": float(base_pred.mean()),
        })

    threshold_results = pd.DataFrame(rows).drop_duplicates(subset=["threshold"]).sort_values("threshold")
    threshold_results.to_csv(OUTPUT_DIR / "daypsci_false_alarm_threshold_comparison.csv", index=False)
    threshold_scores.to_csv(OUTPUT_DIR / "daypsci_false_alarm_validation_thresholds.csv", index=False)

    chosen_test_pred = (test_prob >= chosen_threshold).astype(int)
    chosen_base_pred = (base_prob >= chosen_threshold).astype(int)
    cm = confusion_matrix(y_test, chosen_test_pred, labels=[0, 1])

    print("\n===== BASELINE FALSE-ALARM AND THRESHOLD ANALYSIS =====")
    print(f"Training batches: {len(train_batches)} | rows={len(train)}")
    print(f"Validation: {VALIDATION_BATCH} | rows={len(val)}")
    print(f"Test: {TEST_BATCH} | rows={len(test)}")
    print(f"Normal-only baseline: {BASELINE_BATCH} | rows={len(baseline)}")
    print(f"Features: {len(features)}")
    print(f"\nSelected threshold from validation: {chosen_threshold:.3f}")
    print(
        f"Validation precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, F1={best['f1']:.4f}"
    )
    print("\nTest confusion matrix (true rows, predicted columns):")
    print(cm)
    print(
        f"Test precision={precision_score(y_test, chosen_test_pred, zero_division=0):.4f}, "
        f"recall={recall_score(y_test, chosen_test_pred, zero_division=0):.4f}, "
        f"F1={f1_score(y_test, chosen_test_pred, zero_division=0):.4f}"
    )
    print(
        f"\nBaseline false alarms: {int(chosen_base_pred.sum())}/{len(baseline)} "
        f"({chosen_base_pred.mean() * 100:.2f}%)"
    )
    print("\nThreshold comparison:")
    print(threshold_results.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'daypsci_false_alarm_threshold_comparison.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'daypsci_false_alarm_validation_thresholds.csv'}")


if __name__ == "__main__":
    main()
