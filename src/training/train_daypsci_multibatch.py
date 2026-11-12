"""
DAYPSCI repeated cross-batch evaluation.

Runs a leave-one-batch-out style evaluation over eligible batches:
- Each target batch is held out for testing.
- A different eligible batch is used for validation/threshold selection.
- All remaining batches are used for training.
- Test labels are not used for threshold selection.

This is a first robust generalization check, not a fully nested cross-validation
study. Batch choice for validation is deterministic and printed in the report.

Run:
    python train_daypsci_multibatch.py

Outputs:
  daypsci_multibatch_results.csv
  daypsci_multibatch_summary.txt
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path(
    r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1"
)
OUTPUT_DIR = Path(__file__).resolve().parent

EXCLUDE_COLUMNS = {
    "anomaly", "fault", "FSx", "TFSx", "FCVx", "TFCVx",
    "ts_plc", "event_seq", "batch_id", "experiment_id", "part_id",
}
SIGNALS = [
    "a0", "a1", "b0", "b1", "c0", "c1", "B_1", "B_2",
    "YA_p", "YA_m", "YB_p", "YB_m", "YC_p", "YC_m",
]


def find_csv(folder):
    paths = list(folder.glob("*.csv"))
    for path in paths:
        if folder.name in path.name:
            return path
    return paths[0] if paths else None


def load_data():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATA_ROOT}\n"
            "Edit DATA_ROOT in this script to match your dataset location."
        )

    frames = []
    for folder in sorted(DATA_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        path = find_csv(folder)
        if path is None:
            continue
        try:
            df = pd.read_csv(path, sep=";", low_memory=False)
        except Exception as exc:
            print(f"Skipping {folder.name}: {exc}")
            continue
        if "anomaly" not in df.columns:
            print(f"Skipping {folder.name}: missing anomaly column")
            continue

        df["source_batch"] = folder.name
        df["target_anomaly"] = (
            pd.to_numeric(df["anomaly"], errors="coerce").fillna(0) != 0
        ).astype(int)
        frames.append(df)
        print(
            f"Loaded {folder.name}: {len(df)} rows; "
            f"anomalous={int(df['target_anomaly'].sum())}"
        )

    if not frames:
        raise RuntimeError(f"No valid CSV batches found in {DATA_ROOT}")
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

    groups = ["source_batch"]
    if "part_id" in df.columns:
        groups.append("part_id")
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

    changed_cols = []
    for signal in SIGNALS:
        if signal in df.columns:
            previous = grouped[signal].shift(1)
            name = f"{signal}_changed"
            df[name] = ((df[signal] != previous) & previous.notna()).astype(int)
            changed_cols.append(name)
    if changed_cols:
        df["signal_change_count"] = df[changed_cols].sum(axis=1)
    return df


def build_model(X):
    numeric_cols = [
        c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])
    ]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols),
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


def probability_of_anomaly(model, X):
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def select_threshold(y_val, val_prob):
    candidates = []
    for threshold in np.arange(0.10, 0.901, 0.025):
        prediction = (val_prob >= threshold).astype(int)
        candidates.append({
            "threshold": float(threshold),
            "f1": f1_score(y_val, prediction, zero_division=0),
            "precision": precision_score(y_val, prediction, zero_division=0),
            "recall": recall_score(y_val, prediction, zero_division=0),
        })
    scores = pd.DataFrame(candidates)
    best = scores.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["threshold"]), float(best["f1"])


def main():
    raw = load_data()
    data = add_temporal_features(raw)

    all_batches = sorted(data["source_batch"].unique())
    if len(all_batches) < 3:
        raise RuntimeError("Need at least three batches for train/validation/test.")

    excluded = EXCLUDE_COLUMNS | {"source_batch", "target_anomaly"}
    # Feature eligibility is based only on each fold's training split below.
    rows = []

    for test_batch in all_batches:
        # Choose a validation batch deterministically, never the test batch.
        remaining = [batch for batch in all_batches if batch != test_batch]
        validation_batch = remaining[0]
        train_batches = [
            batch for batch in all_batches
            if batch not in {test_batch, validation_batch}
        ]

        train = data[data["source_batch"].isin(train_batches)].copy()
        val = data[data["source_batch"] == validation_batch].copy()
        test = data[data["source_batch"] == test_batch].copy()

        y_train = train["target_anomaly"].astype(int)
        y_val = val["target_anomaly"].astype(int)
        y_test = test["target_anomaly"].astype(int)

        if train.empty or val.empty or test.empty or y_train.nunique() < 2 or y_val.nunique() < 2:
            print(f"Skipping test={test_batch}: empty split or missing target class.")
            continue

        features = [
            c for c in data.columns
            if c not in excluded
            and train[c].notna().any()
            and train[c].nunique(dropna=True) > 1
        ]
        X_train = train[features].copy()
        X_val = val[features].copy()
        X_test = test[features].copy()

        model = build_model(X_train)
        model.fit(X_train, y_train)

        val_prob = probability_of_anomaly(model, X_val)
        threshold, validation_f1 = select_threshold(y_val, val_prob)

        test_prob = probability_of_anomaly(model, X_test)
        test_pred = (test_prob >= threshold).astype(int)

        row = {
            "test_batch": test_batch,
            "validation_batch": validation_batch,
            "training_batch_count": len(train_batches),
            "training_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "feature_count": len(features),
            "threshold": threshold,
            "validation_f1": validation_f1,
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_anomaly_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_anomaly_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_anomaly_f1": f1_score(y_test, test_pred, zero_division=0),
            "test_normal_count": int((y_test == 0).sum()),
            "test_anomaly_count": int((y_test == 1).sum()),
            "true_negative": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[0, 0]),
            "false_positive": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[0, 1]),
            "false_negative": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[1, 0]),
            "true_positive": int(confusion_matrix(y_test, test_pred, labels=[0, 1])[1, 1]),
        }
        rows.append(row)
        print(
            f"\nTest batch: {test_batch}\n"
            f"  Validation batch: {validation_batch}\n"
            f"  Threshold: {threshold:.3f} | validation F1: {validation_f1:.4f}\n"
            f"  Test accuracy: {row['test_accuracy']:.4f} | "
            f"precision: {row['test_anomaly_precision']:.4f} | "
            f"recall: {row['test_anomaly_recall']:.4f} | "
            f"F1: {row['test_anomaly_f1']:.4f}"
        )

    if not rows:
        raise RuntimeError(
            "No folds could be evaluated. Check whether the validation batches contain both classes."
        )

    results = pd.DataFrame(rows).sort_values("test_batch")
    results_path = OUTPUT_DIR / "daypsci_multibatch_results.csv"
    results.to_csv(results_path, index=False)

    metric_cols = [
        "test_accuracy", "test_anomaly_precision",
        "test_anomaly_recall", "test_anomaly_f1",
    ]
    summary = results[metric_cols].agg(["mean", "std", "min", "max"]).round(4)
    summary_path = OUTPUT_DIR / "daypsci_multibatch_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("DAYPSCI repeated cross-batch anomaly detection\n\n")
        f.write("Each row evaluates one held-out test batch. Threshold is selected on a separate validation batch.\n")
        f.write("Note: validation batch assignment is deterministic; review the CSV to see each pairing.\n\n")
        f.write(results.to_string(index=False))
        f.write("\n\nAggregate metrics across evaluated test batches:\n")
        f.write(summary.to_string())

    print("\n===== MULTI-BATCH RESULTS =====")
    print(results.to_string(index=False))
    print("\n===== AGGREGATE METRICS =====")
    print(summary.to_string())
    print(f"\nSaved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(
        "\nImportant: this is a first repeated cross-batch benchmark. "
        "The fixed validation-batch rule may be imperfect for some fault types, "
        "so report the exact split protocol and do not treat this as fully nested cross-validation."
    )


if __name__ == "__main__":
    main()
