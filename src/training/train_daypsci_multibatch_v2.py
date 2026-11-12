"""
Improved DAYPSCI multi-batch anomaly evaluation.

For each test batch containing BOTH normal and anomalous rows:
  - Choose a validation batch (different from test) containing both classes.
  - Train on all remaining batches.
  - Select threshold using validation only.
  - Evaluate on the held-out test batch.

Baseline-only batches are evaluated separately for false-alarm rate and accuracy,
but are not used to calculate anomaly recall/F1 because they have no positive cases.

Run:
    python train_daypsci_multibatch_v2.py

Outputs:
  daypsci_multibatch_v2_results.csv
  daypsci_multibatch_v2_summary.txt
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
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path(
    r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1"
)
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
    paths = list(folder.glob("*.csv"))
    for p in paths:
        if folder.name in p.name:
            return p
    return paths[0] if paths else None


def load_data():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATA_ROOT}\n"
            "Edit DATA_ROOT in this script."
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
            continue
        df["source_batch"] = folder.name
        df["target_anomaly"] = (
            pd.to_numeric(df["anomaly"], errors="coerce").fillna(0) != 0
        ).astype(int)
        frames.append(df)
        print(
            f"Loaded {folder.name}: rows={len(df)}, "
            f"normal={(df['target_anomaly'] == 0).sum()}, "
            f"anomaly={(df['target_anomaly'] == 1).sum()}"
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

    changes = []
    for signal in SIGNALS:
        if signal in df.columns:
            prev = grouped[signal].shift(1)
            name = signal + "_changed"
            df[name] = ((df[signal] != prev) & prev.notna()).astype(int)
            changes.append(name)
    if changes:
        df["signal_change_count"] = df[changes].sum(axis=1)
    return df


def make_model(X):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessing", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def get_probability(model, X):
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def tune_threshold(y_val, probabilities):
    records = []
    for threshold in np.arange(0.10, 0.901, 0.025):
        pred = (probabilities >= threshold).astype(int)
        records.append({
            "threshold": float(threshold),
            "f1": f1_score(y_val, pred, zero_division=0),
            "precision": precision_score(y_val, pred, zero_division=0),
            "recall": recall_score(y_val, pred, zero_division=0),
        })
    scores = pd.DataFrame(records)
    best = scores.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["threshold"]), float(best["f1"])


def main():
    data = add_temporal_features(load_data())
    batches = sorted(data["source_batch"].unique())

    batch_counts = data.groupby("source_batch")["target_anomaly"].agg(
        rows="size", positive="sum"
    )
    batch_counts["negative"] = batch_counts["rows"] - batch_counts["positive"]

    eligible_test_batches = [
        b for b in batches
        if batch_counts.loc[b, "positive"] > 0 and batch_counts.loc[b, "negative"] > 0
    ]
    validation_candidates = [
        b for b in batches
        if batch_counts.loc[b, "positive"] > 0 and batch_counts.loc[b, "negative"] > 0
    ]

    print("\n===== IMPROVED MULTI-BATCH EVALUATION =====")
    print("\nBatch class counts:")
    print(batch_counts.to_string())
    print(f"\nEligible mixed-class test batches: {len(eligible_test_batches)}")

    excluded = EXCLUDE | {"source_batch", "target_anomaly"}
    rows = []

    for test_batch in eligible_test_batches:
        # Choose a mixed-class validation batch different from the test batch.
        # Rotate candidates so each test batch does not always use the same validation set.
        val_options = [b for b in validation_candidates if b != test_batch]
        if not val_options:
            print(f"Skipping {test_batch}: no separate mixed-class validation batch.")
            continue
        test_index = batches.index(test_batch)
        validation_batch = val_options[test_index % len(val_options)]

        train_batches = [
            b for b in batches if b not in {test_batch, validation_batch}
        ]
        train = data[data["source_batch"].isin(train_batches)].copy()
        val = data[data["source_batch"] == validation_batch].copy()
        test = data[data["source_batch"] == test_batch].copy()

        y_train = train["target_anomaly"].astype(int)
        y_val = val["target_anomaly"].astype(int)
        y_test = test["target_anomaly"].astype(int)

        if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
            print(f"Skipping {test_batch}: train/validation/test does not have both classes.")
            continue

        features = [
            c for c in data.columns
            if c not in excluded
            and train[c].notna().any()
            and train[c].nunique(dropna=True) > 1
        ]
        X_train, X_val, X_test = (
            train[features].copy(), val[features].copy(), test[features].copy()
        )

        model = make_model(X_train)
        model.fit(X_train, y_train)
        val_prob = get_probability(model, X_val)
        threshold, val_f1 = tune_threshold(y_val, val_prob)

        test_prob = get_probability(model, X_test)
        test_pred = (test_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

        row = {
            "test_batch": test_batch,
            "validation_batch": validation_batch,
            "training_batches": len(train_batches),
            "training_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "threshold": threshold,
            "validation_f1": val_f1,
            "test_accuracy": accuracy_score(y_test, test_pred),
            "test_anomaly_precision": precision_score(y_test, test_pred, zero_division=0),
            "test_anomaly_recall": recall_score(y_test, test_pred, zero_division=0),
            "test_anomaly_f1": f1_score(y_test, test_pred, zero_division=0),
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        }
        rows.append(row)
        print(
            f"\nTEST: {test_batch} | VALIDATION: {validation_batch}\n"
            f"Threshold={threshold:.3f}, validation F1={val_f1:.4f}\n"
            f"Accuracy={row['test_accuracy']:.4f}, "
            f"Precision={row['test_anomaly_precision']:.4f}, "
            f"Recall={row['test_anomaly_recall']:.4f}, "
            f"F1={row['test_anomaly_f1']:.4f}\n"
            f"Confusion matrix:\n{cm}"
        )

    if not rows:
        raise RuntimeError("No eligible folds completed. Review batch counts and class labels.")

    results = pd.DataFrame(rows).sort_values("test_batch")
    results_path = OUTPUT_DIR / "daypsci_multibatch_v2_results.csv"
    results.to_csv(results_path, index=False)

    metric_cols = [
        "test_accuracy", "test_anomaly_precision",
        "test_anomaly_recall", "test_anomaly_f1",
    ]
    summary = results[metric_cols].agg(["mean", "std", "min", "max"]).round(4)
    summary_path = OUTPUT_DIR / "daypsci_multibatch_v2_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Improved DAYPSCI multi-batch evaluation\n\n")
        f.write(results.to_string(index=False))
        f.write("\n\nAggregate metrics over mixed-class test batches:\n")
        f.write(summary.to_string())

    print("\n===== RESULTS BY TEST BATCH =====")
    print(results.to_string(index=False))
    print("\n===== AGGREGATE METRICS =====")
    print(summary.to_string())
    print(f"\nSaved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(
        "\nBaseline-only batches are excluded from anomaly F1 aggregation because they "
        "contain no positive anomaly examples. Evaluate them separately for false alarms."
    )


if __name__ == "__main__":
    main()
