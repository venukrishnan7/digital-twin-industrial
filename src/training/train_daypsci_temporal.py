"""
DAYPSCI anomaly detection - Phase 5 (temporal features).

This script:
- Loads DAYPSCI sensor T2 batch 03.
- Creates within-part timing and signal-transition features.
- Excludes target labels and fault-injection indicators from model inputs.
- Splits by part_id to reduce event-level leakage.
- Selects a probability threshold using training-only out-of-fold predictions.
- Reports test metrics and saves predictions/feature importances.

Install:
    python -m pip install pandas scikit-learn

Run:
    python train_daypsci_temporal.py

This remains a within-batch experiment. Cross-batch evaluation is a later step.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "2026_Martin_DAYPSCI_v1"
    / "DAYPSCI_v1"
)
TARGET_BATCH = "pn_sensor_T2_batch03"
RANDOM_STATE = 42

EXCLUDED = {
    "ts_plc", "event_seq", "anomaly", "fault", "FSx", "TFSx",
    "FCVx", "TFCVx", "batch_id", "experiment_id", "part_id",
}

def load_batch():
    folder = DATASET_ROOT / TARGET_BATCH
    files = list(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {folder}")
    df = pd.read_csv(files[0], sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def add_temporal_features(df):
    df = df.copy()

    numeric_cols = [
        c for c in df.columns
        if c not in {"ts_plc", "anomaly", "fault", "part_id"}
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure event order is chronological within each process part.
    sort_cols = [c for c in ["part_id", "scan_id", "event_order", "event_seq"] if c in df.columns]
    df = df.sort_values(sort_cols).copy()

    grouped = df.groupby("part_id", sort=False)

    # Timing context: current delta, previous delta, and relative change.
    if "delta_time_ms" in df.columns:
        df["prev_delta_time_ms"] = grouped["delta_time_ms"].shift(1)
        df["delta_change_ms"] = (
            df["delta_time_ms"] - df["prev_delta_time_ms"]
        )
        df["delta_ratio_prev"] = (
            df["delta_time_ms"] /
            grouped["delta_time_ms"].shift(1).replace(0, np.nan)
        )
        df["delta_rolling_mean_3"] = grouped["delta_time_ms"].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
        df["delta_rolling_max_3"] = grouped["delta_time_ms"].transform(
            lambda s: s.rolling(window=3, min_periods=1).max()
        )

    # Binary signal changes: compare each signal with its previous value
    # within the same part. No target or fault-injection columns are used.
    signal_cols = [
        c for c in [
            "a0", "a1", "b0", "b1", "c0", "c1", "B_1", "B_2",
            "YA_p", "YA_m", "YB_p", "YB_m", "YC_p", "YC_m"
        ] if c in df.columns
    ]

    for col in signal_cols:
        previous = df.groupby("part_id", sort=False)[col].shift(1)
        df[f"{col}_changed"] = (
            df[col].notna() & previous.notna() & (df[col] != previous)
        ).astype(int)

    # Number of observed binary signal changes in the current event.
    change_cols = [f"{c}_changed" for c in signal_cols]
    if change_cols:
        df["signal_change_count"] = df[change_cols].sum(axis=1)

    return df

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}. "
            "Edit DATASET_ROOT at the top of this script."
        )

    df = add_temporal_features(load_batch())
    df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce")
    df["part_id"] = pd.to_numeric(df["part_id"], errors="coerce")
    df = df.dropna(subset=["anomaly", "part_id"]).copy()

    # Binary target: normal=0; documented anomaly classes 1 and 4 -> anomaly=1.
    y = (df["anomaly"] != 0).astype(int)
    groups = df["part_id"]

    feature_cols = [c for c in df.columns if c not in EXCLUDED]
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols)
    valid = ~X.isna().all(axis=1)
    df, X, y, groups = (
        df.loc[valid].copy(),
        X.loc[valid].copy(),
        y.loc[valid].copy(),
        groups.loc[valid].copy()
    )

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Group split did not place both classes in train and test. "
            "Change RANDOM_STATE or move to a multi-batch experiment."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ))
    ])

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    # Pick a threshold using training-only grouped cross-validation.
    # This avoids using the held-out test labels to tune the threshold.
    unique_train_groups = train_groups.nunique()
    threshold = 0.5
    oof_probs = None

    if unique_train_groups >= 3:
        n_splits = min(5, unique_train_groups)
        cv = GroupKFold(n_splits=n_splits)
        try:
            oof_probs = cross_val_predict(
                model,
                X_train,
                y_train,
                groups=train_groups,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]

            threshold_candidates = np.arange(0.15, 0.81, 0.025)
            scored = []
            for candidate in threshold_candidates:
                oof_pred = (oof_probs >= candidate).astype(int)
                scored.append((
                    f1_score(y_train, oof_pred, zero_division=0),
                    candidate,
                    precision_score(y_train, oof_pred, zero_division=0),
                    recall_score(y_train, oof_pred, zero_division=0),
                ))
            # Maximize training OOF F1; tie-break toward higher precision.
            scored.sort(key=lambda item: (item[0], item[2]), reverse=True)
            _, threshold, _, _ = scored[0]
        except ValueError as exc:
            print("Threshold tuning skipped:", exc)

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    print("===== DAYPSCI TEMPORAL ANOMALY DETECTION =====")
    print("Batch:", TARGET_BATCH)
    print("Usable rows:", len(df))
    print("Features used:", len(X.columns))
    print("Feature columns:", list(X.columns))
    print("Dropped constant columns:", constant_cols)
    print("Target counts (0=normal, 1=anomaly):")
    print(y.value_counts().sort_index().to_string())
    print("Train rows:", len(train_idx), "| Test rows:", len(test_idx))
    print("Train part IDs:", sorted(set(train_groups)))
    print("Test part IDs:", sorted(set(test_groups)))
    print(f"Threshold selected using training-only grouped CV: {threshold:.3f}")

    print("\nConfusion matrix (true rows, predicted columns; labels 0,1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["normal", "anomaly"],
        zero_division=0,
    ))
    print(f"Accuracy:           {accuracy_score(y_test, predictions):.4f}")
    print(f"Anomaly precision:  {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Anomaly recall:     {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Anomaly F1-score:   {f1_score(y_test, predictions, zero_division=0):.4f}")

    result_cols = [
        c for c in ["event_seq", "part_id", "event_type", "gemma", "anomaly", "fault"]
        if c in df.columns
    ]
    results = df.iloc[test_idx][result_cols].copy()
    results["true_binary_anomaly"] = y_test.to_numpy()
    results["predicted_binary_anomaly"] = predictions
    results["anomaly_probability"] = probabilities
    results["correct"] = (
        results["true_binary_anomaly"] == results["predicted_binary_anomaly"]
    )
    results_path = Path.cwd() / "daypsci_temporal_test_predictions.csv"
    results.to_csv(results_path, index=False)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.named_steps["classifier"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importance_path = Path.cwd() / "daypsci_temporal_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\nSaved test predictions:", results_path)
    print("Saved feature importance:", importance_path)
    print(
        "\nEvaluation note: held-out parts from the same batch only. "
        "Use a separate batch for a stronger generalization test."
    )

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
