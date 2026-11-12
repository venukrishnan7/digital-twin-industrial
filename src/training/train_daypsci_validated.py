"""
DAYPSCI validated cross-batch anomaly detection.

Protocol:
  - Final test batch: pn_sensor_T2_batch03 (never used for threshold selection)
  - Validation batch: pn_sensor_T1_batch02
  - Training: all other batches

Run:
    python train_daypsci_validated.py

Outputs:
  daypsci_validated_test_predictions.csv
  daypsci_validated_feature_importance.csv
  daypsci_validated_metrics.txt
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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path(
    r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1"
)
VALIDATION_BATCH = "pn_sensor_T1_batch02"
TEST_BATCH = "pn_sensor_T2_batch03"
OUTPUT_DIR = Path(__file__).resolve().parent

EXCLUDE_COLUMNS = {
    "anomaly", "fault", "FSx", "TFSx", "FCVx", "TFCVx",
    "ts_plc", "event_seq", "batch_id", "experiment_id", "part_id",
}
SIGNALS = [
    "a0", "a1", "b0", "b1", "c0", "c1", "B_1", "B_2",
    "YA_p", "YA_m", "YB_p", "YB_m", "YC_p", "YC_m",
]


def find_batch_csv(batch_dir):
    candidates = list(batch_dir.glob("*.csv"))
    if not candidates:
        return None
    for path in candidates:
        if batch_dir.name in path.name:
            return path
    return candidates[0]


def load_all_batches():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found:\n{DATA_ROOT}\n"
            "Edit DATA_ROOT in this script to match your dataset location."
        )

    frames = []
    for batch_dir in sorted(DATA_ROOT.iterdir()):
        if not batch_dir.is_dir():
            continue
        csv_path = find_batch_csv(batch_dir)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path, sep=";", low_memory=False)
        except Exception as exc:
            print(f"Skipping {csv_path.name}: {exc}")
            continue
        if "anomaly" not in df.columns:
            print(f"Skipping {csv_path.name}: no anomaly column")
            continue

        df["source_batch"] = batch_dir.name
        # Binary classification: every non-zero anomaly class is anomalous.
        df["target_anomaly"] = (
            pd.to_numeric(df["anomaly"], errors="coerce").fillna(0) != 0
        ).astype(int)
        frames.append(df)
        print(
            f"Loaded {batch_dir.name}: {len(df)} rows, "
            f"{int(df['target_anomaly'].sum())} anomalous rows"
        )

    if not frames:
        raise RuntimeError(f"No valid batch CSVs found under {DATA_ROOT}")
    return pd.concat(frames, ignore_index=True, sort=False)


def add_temporal_features(df):
    df = df.copy()
    for col in ["delta_time_ms", "scan_id", "event_order", "event_id"] + SIGNALS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_cols = [
        c for c in
        ["source_batch", "part_id", "scan_id", "event_order", "event_seq"]
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
            previous = grouped[signal].shift(1)
            name = f"{signal}_changed"
            df[name] = ((df[signal] != previous) & previous.notna()).astype(int)
            change_cols.append(name)
    if change_cols:
        df["signal_change_count"] = df[change_cols].sum(axis=1)
    return df


def choose_batch_frames(all_featured):
    train = all_featured[
        ~all_featured["source_batch"].isin([VALIDATION_BATCH, TEST_BATCH])
    ].copy()
    validation = all_featured[
        all_featured["source_batch"] == VALIDATION_BATCH
    ].copy()
    test = all_featured[all_featured["source_batch"] == TEST_BATCH].copy()

    if train.empty or validation.empty or test.empty:
        raise RuntimeError(
            "Training, validation, or test split is empty. "
            "Check the batch folder names and dataset path."
        )
    return train, validation, test


def make_model(X_train):
    numeric_cols = [
        c for c in X_train.columns
        if pd.api.types.is_numeric_dtype(X_train[c])
    ]
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols),
    ])
    return Pipeline([
        ("preprocessing", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def main():
    all_data = load_all_batches()
    all_featured = add_temporal_features(all_data)
    train, validation, test = choose_batch_frames(all_featured)

    excluded = EXCLUDE_COLUMNS | {"source_batch", "target_anomaly"}
    features = [
        c for c in all_featured.columns
        if c not in excluded
        and train[c].notna().any()
        and train[c].nunique(dropna=True) > 1
    ]

    X_train = train[features].copy()
    X_val = validation[features].copy()
    X_test = test[features].copy()
    y_train = train["target_anomaly"].astype(int)
    y_val = validation["target_anomaly"].astype(int)
    y_test = test["target_anomaly"].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError("Training split has only one target class.")
    if y_val.nunique() < 2:
        raise RuntimeError("Validation batch must contain both normal and anomalous events.")

    model = make_model(X_train)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, list(model.named_steps["classifier"].classes_).index(1)]

    # Select threshold using validation only; no test labels are used here.
    thresholds = np.arange(0.10, 0.901, 0.025)
    scored = []
    for threshold in thresholds:
        val_pred = (val_prob >= threshold).astype(int)
        scored.append({
            "threshold": float(threshold),
            "f1": f1_score(y_val, val_pred, zero_division=0),
            "precision": precision_score(y_val, val_pred, zero_division=0),
            "recall": recall_score(y_val, val_pred, zero_division=0),
        })
    scores = pd.DataFrame(scored)
    # Maximize F1; break ties by higher precision, then higher threshold.
    best = scores.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
    ).iloc[0]
    threshold = float(best["threshold"])

    test_prob = model.predict_proba(X_test)[:, list(model.named_steps["classifier"].classes_).index(1)]
    test_pred = (test_prob >= threshold).astype(int)

    report = classification_report(
        y_test, test_pred, labels=[0, 1],
        target_names=["normal", "anomaly"], zero_division=0
    )
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    print("\n===== VALIDATED CROSS-BATCH ANOMALY DETECTION =====")
    print(f"Training batches: {train['source_batch'].nunique()} | rows: {len(train)}")
    print(f"Validation batch: {VALIDATION_BATCH} | rows: {len(validation)}")
    print(f"Test batch: {TEST_BATCH} | rows: {len(test)}")
    print(f"Features: {len(features)}")
    print(f"Training target counts:\n{y_train.value_counts().sort_index()}")
    print(f"Validation target counts:\n{y_val.value_counts().sort_index()}")
    print(f"Test target counts:\n{y_test.value_counts().sort_index()}")
    print(f"\nSelected threshold from validation: {threshold:.3f}")
    print(
        f"Validation precision={best['precision']:.4f}, "
        f"recall={best['recall']:.4f}, F1={best['f1']:.4f}"
    )
    print("\nFinal test confusion matrix (true rows, predicted columns; labels 0,1):")
    print(cm)
    print("\nFinal test classification report:")
    print(report)
    print(f"Test accuracy:          {accuracy_score(y_test, test_pred):.4f}")
    print(f"Test anomaly precision: {precision_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test anomaly recall:    {recall_score(y_test, test_pred, zero_division=0):.4f}")
    print(f"Test anomaly F1-score:  {f1_score(y_test, test_pred, zero_division=0):.4f}")

    output = test[
        [c for c in ["source_batch", "part_id", "event_seq", "scan_id", "event_order", "anomaly", "fault"] if c in test.columns]
    ].copy()
    output["actual_anomaly_binary"] = y_test.to_numpy()
    output["predicted_anomaly_binary"] = test_pred
    output["anomaly_probability"] = test_prob
    output.to_csv(OUTPUT_DIR / "daypsci_validated_test_predictions.csv", index=False)

    # Validation threshold table and test metrics are saved for reproducibility.
    scores.to_csv(OUTPUT_DIR / "daypsci_validation_threshold_scores.csv", index=False)
    metrics_text = (
        "DAYPSCI validated cross-batch anomaly detection\n"
        f"Training batches: {train['source_batch'].nunique()}\n"
        f"Validation batch: {VALIDATION_BATCH}\n"
        f"Test batch: {TEST_BATCH}\n"
        f"Selected threshold: {threshold:.3f}\n"
        f"Validation F1: {best['f1']:.4f}\n"
        f"Test accuracy: {accuracy_score(y_test, test_pred):.4f}\n"
        f"Test anomaly precision: {precision_score(y_test, test_pred, zero_division=0):.4f}\n"
        f"Test anomaly recall: {recall_score(y_test, test_pred, zero_division=0):.4f}\n"
        f"Test anomaly F1: {f1_score(y_test, test_pred, zero_division=0):.4f}\n\n"
        f"Confusion matrix:\n{cm}\n\nClassification report:\n{report}"
    )
    (OUTPUT_DIR / "daypsci_validated_metrics.txt").write_text(metrics_text, encoding="utf-8")

    print(f"\nSaved predictions: {OUTPUT_DIR / 'daypsci_validated_test_predictions.csv'}")
    print(f"Saved threshold scores: {OUTPUT_DIR / 'daypsci_validation_threshold_scores.csv'}")
    print(f"Saved metrics: {OUTPUT_DIR / 'daypsci_validated_metrics.txt'}")

    # Feature importance is calculated only after the final evaluation and is
    # diagnostic; it is not used to select the threshold.
    try:
        importance = permutation_importance(
            model, X_val, y_val, scoring="f1", n_repeats=3,
            random_state=42, n_jobs=1
        )
        importance_df = pd.DataFrame({
            "feature": features,
            "permutation_importance_mean": importance.importances_mean,
            "permutation_importance_std": importance.importances_std,
        }).sort_values("permutation_importance_mean", ascending=False)
        importance_df.to_csv(
            OUTPUT_DIR / "daypsci_validated_feature_importance.csv", index=False
        )
        print(f"Saved feature importance: {OUTPUT_DIR / 'daypsci_validated_feature_importance.csv'}")
    except Exception as exc:
        print(f"Feature importance could not be calculated: {exc}")


if __name__ == "__main__":
    main()
