"""
Cross-batch DAYPSCI anomaly detection.
Trains on every available DAYPSCI CSV batch except pn_sensor_T2_batch03,
then evaluates on pn_sensor_T2_batch03.

Run from the project folder:
    python train_daypsci_cross_batch.py
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_ROOT = Path(
    r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1"
)
TEST_BATCH = "pn_sensor_T2_batch03"
OUTPUT_DIR = Path(__file__).resolve().parent

# These columns are labels, IDs, timestamps, or fault-injection indicators.
# Excluding them helps prevent label leakage.
EXCLUDE_COLUMNS = {
    "anomaly",
    "fault",
    "FSx",
    "TFSx",
    "FCVx",
    "TFCVx",
    "ts_plc",
    "event_seq",
    "batch_id",
    "experiment_id",
    "part_id",
}

SIGNALS = [
    "a0", "a1", "b0", "b1", "c0", "c1",
    "B_1", "B_2",
    "YA_p", "YA_m", "YB_p", "YB_m", "YC_p", "YC_m",
]


def find_batch_csv(batch_dir):
    """Find a CSV file directly inside a batch folder."""
    candidates = list(batch_dir.glob("*.csv"))
    if not candidates:
        return None
    # Prefer a file whose name resembles the batch folder.
    for candidate in candidates:
        if batch_dir.name in candidate.name:
            return candidate
    return candidates[0]


def load_all_batches():
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset root not found:\n{DATA_ROOT}\n"
            "Edit DATA_ROOT at the top of this script to match your dataset location."
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
        df["target_anomaly"] = (pd.to_numeric(df["anomaly"], errors="coerce").fillna(0) != 0).astype(int)
        frames.append(df)
        print(
            f"Loaded {batch_dir.name}: {len(df)} rows, "
            f"{int(df['target_anomaly'].sum())} anomalous rows"
        )

    if not frames:
        raise RuntimeError(f"No valid batch CSV files found under {DATA_ROOT}")

    return pd.concat(frames, ignore_index=True, sort=False)


def add_temporal_features(df):
    """Create within-part timing and signal-change features."""
    df = df.copy()

    for col in ["delta_time_ms", "scan_id", "event_order", "event_id"] + SIGNALS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort within each source batch and part to avoid crossing cycle boundaries.
    sort_cols = [
        col for col in
        ["source_batch", "part_id", "scan_id", "event_order", "event_seq"]
        if col in df.columns
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

    change_columns = []
    for signal in SIGNALS:
        if signal in df.columns:
            previous = grouped[signal].shift(1)
            new_col = f"{signal}_changed"
            df[new_col] = ((df[signal] != previous) & previous.notna()).astype(int)
            change_columns.append(new_col)

    if change_columns:
        df["signal_change_count"] = df[change_columns].sum(axis=1)

    return df


def main():
    all_data = load_all_batches()

    if "source_batch" not in all_data.columns:
        raise RuntimeError("Internal error: source batch names were not retained.")

    test_mask = all_data["source_batch"] == TEST_BATCH
    train_data = all_data.loc[~test_mask].copy()
    test_data = all_data.loc[test_mask].copy()

    if train_data.empty:
        raise RuntimeError("No training batches found. Check DATA_ROOT.")
    if test_data.empty:
        raise RuntimeError(
            f"Test batch '{TEST_BATCH}' was not found. Check the folder name and DATA_ROOT."
        )

    # Feature engineering is performed after concatenation but grouped by batch/part,
    # so temporal calculations do not cross batch or part boundaries.
    all_featured = add_temporal_features(pd.concat([train_data, test_data], ignore_index=True, sort=False))
    train_mask_featured = all_featured["source_batch"] != TEST_BATCH
    train = all_featured.loc[train_mask_featured].copy()
    test = all_featured.loc[~train_mask_featured].copy()

    y_train = train["target_anomaly"].astype(int)
    y_test = test["target_anomaly"].astype(int)

    excluded = EXCLUDE_COLUMNS | {"source_batch", "target_anomaly"}
    feature_columns = [
        col for col in all_featured.columns
        if col not in excluded
    ]

    # Remove columns that are completely empty or constant in training.
    feature_columns = [
        col for col in feature_columns
        if train[col].notna().any() and train[col].nunique(dropna=True) > 1
    ]

    X_train = train[feature_columns].copy()
    X_test = test[feature_columns].copy()

    # Infer numeric/categorical columns from training data. Object columns use
    # one-hot encoding; unseen test categories are safely ignored.
    numeric_columns = [
        col for col in feature_columns
        if pd.api.types.is_numeric_dtype(X_train[col])
    ]
    categorical_columns = [col for col in feature_columns if col not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    class_weight="balanced_subsample",
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    print("\n===== DAYPSCI CROSS-BATCH ANOMALY DETECTION =====")
    print(f"Training batches: {train['source_batch'].nunique()}")
    print(f"Training rows: {len(train)}")
    print(f"Test batch: {TEST_BATCH}")
    print(f"Test rows: {len(test)}")
    print(f"Features: {len(feature_columns)}")
    print(f"Training target counts:\n{y_train.value_counts().sort_index()}")
    print(f"Test target counts:\n{y_test.value_counts().sort_index()}")
    print("\nTraining batches used:")
    for name in sorted(train["source_batch"].unique()):
        print(f"  - {name}")

    if y_train.nunique() < 2:
        raise RuntimeError(
            "Training data contains only one target class. "
            "The available batches do not provide both normal and anomalous examples."
        )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, list(model.named_steps["classifier"].classes_).index(1)]
    threshold = 0.5
    predictions = (probabilities >= threshold).astype(int)

    print(f"\nDecision threshold: {threshold:.2f}")
    print("\nConfusion matrix (true rows, predicted columns; labels 0,1):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    )
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.4f}")
    print(f"Anomaly precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Anomaly recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Anomaly F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    predictions_out = test[
        [c for c in ["source_batch", "part_id", "event_seq", "scan_id", "event_order", "anomaly", "fault"] if c in test.columns]
    ].copy()
    predictions_out["actual_anomaly_binary"] = y_test.to_numpy()
    predictions_out["predicted_anomaly_binary"] = predictions
    predictions_out["anomaly_probability"] = probabilities
    predictions_out.to_csv(OUTPUT_DIR / "daypsci_cross_batch_predictions.csv", index=False)

    # Permutation importance is calculated on the held-out batch for a readable
    # diagnostic. This is exploratory and should not be used to tune the model.
    try:
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1",
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        importance = pd.DataFrame(
            {
                "feature": feature_columns,
                "permutation_importance_mean": result.importances_mean,
                "permutation_importance_std": result.importances_std,
            }
        ).sort_values("permutation_importance_mean", ascending=False)
        importance.to_csv(OUTPUT_DIR / "daypsci_cross_batch_feature_importance.csv", index=False)
        print(f"\nSaved feature importance: {OUTPUT_DIR / 'daypsci_cross_batch_feature_importance.csv'}")
    except Exception as exc:
        print(f"\nCould not calculate permutation importance: {exc}")

    print(f"Saved predictions: {OUTPUT_DIR / 'daypsci_cross_batch_predictions.csv'}")
    print(
        "\nEvaluation note: this is a strict cross-batch test. "
        "Performance may be lower because the held-out sensor-fault experiment "
        "differs from the training experiments."
    )


if __name__ == "__main__":
    main()
