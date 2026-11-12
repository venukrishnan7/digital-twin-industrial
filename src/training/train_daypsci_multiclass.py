from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# Project folder: save this script inside digital-twin-industrial.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(
    r"C:\Users\VENUKRISHNAN.MS\Downloads\2026_Martin_DAYPSCI_v1\DAYPSCI_v1"
)

VALIDATION_BATCH = "pn_sensor_T1_batch02"
TEST_BATCH = "pn_sensor_T2_batch03"
BASELINE_BATCH = "pn_baseline__batch01"

BATCHES = [
    "pn_baseline__batch01",
    "pn_combined_T1_batch11",
    "pn_sensor_T1_batch02",
    "pn_sensor_T2_batch03",
    "pn_sensor_T3_batch04",
    "pn_sensor_T4_batch05",
    "pn_sensor_T5_batch06",
    "pn_sensor_T6_batch07",
    "pn_valve_T1_batch08",
    "pn_valve_T2_batch09",
    "pn_valve_T3_batch10",
]

ANOMALY_NAMES = {
    0: "Normal",
    1: "Sensor anomaly",
    2: "Valve anomaly",
    3: "Combined anomaly",
    4: "Propagated anomaly",
}

# Labels and injection metadata are deliberately excluded from features.
EXCLUDED_COLUMNS = {
    "anomaly", "fault", "FSx", "TFSx", "FCVx", "TFCVx",
    "batch_id", "experiment_id", "part_id",
    "event_seq", "ts_plc", "event_id",
}

SIGNAL_COLUMNS = [
    "a0", "a1", "b0", "b1", "c0", "c1",
    "B_1", "B_2", "YA_p", "YA_m", "YB_p", "YB_m", "YC_p", "YC_m",
]


def find_csv(batch_name):
    batch_dir = DATASET_DIR / batch_name
    if not batch_dir.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_dir}")
    csv_files = list(batch_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in: {batch_dir}")
    if len(csv_files) > 1:
        # Prefer a file whose name includes the batch name, otherwise first CSV.
        matching = [p for p in csv_files if batch_name in p.name]
        return matching[0] if matching else csv_files[0]
    return csv_files[0]


def load_batch(batch_name):
    path = find_csv(batch_name)
    df = pd.read_csv(path, sep=";", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    if "anomaly" not in df.columns:
        raise ValueError(f"'anomaly' column missing in {path}")

    df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce")
    df = df[df["anomaly"].isin([0, 1, 2, 3, 4])].copy()
    df["anomaly"] = df["anomaly"].astype(int)
    df["_batch_name"] = batch_name

    # Sort event rows before creating temporal features.
    sort_cols = [c for c in ["part_id", "event_seq", "scan_id", "event_order"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    # Numeric conversion for expected numeric columns.
    for col in df.columns:
        if col not in {"_batch_name"}:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    if "delta_time_ms" in df.columns:
        dt = pd.to_numeric(df["delta_time_ms"], errors="coerce")
        df["delta_prev_ms"] = dt.groupby(df["part_id"] if "part_id" in df.columns else df["_batch_name"]).shift(1)
        df["delta_change_ms"] = dt - df["delta_prev_ms"]
        group_key = df["part_id"] if "part_id" in df.columns else df["_batch_name"]
        df["delta_rolling_mean_3"] = dt.groupby(group_key).transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
        df["delta_rolling_max_3"] = dt.groupby(group_key).transform(
            lambda s: s.rolling(3, min_periods=1).max()
        )

    for col in SIGNAL_COLUMNS:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            group_key = df["part_id"] if "part_id" in df.columns else df["_batch_name"]
            previous = values.groupby(group_key).shift(1)
            df[f"{col}_changed"] = (values != previous).astype(float)
            df.loc[previous.isna(), f"{col}_changed"] = 0.0

    print(f"Loaded {batch_name}: {len(df)} rows; class counts={df['anomaly'].value_counts().sort_index().to_dict()}")
    return df


def make_features(df, feature_columns=None):
    excluded = EXCLUDED_COLUMNS | {"_batch_name"}
    if feature_columns is None:
        candidates = [c for c in df.columns if c not in excluded]
        # Keep numeric/bool columns only; avoid accidental text fields.
        feature_columns = [
            c for c in candidates
            if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c])
        ]
    X = df[feature_columns].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, feature_columns


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}\n"
            "Edit DATASET_DIR at the top of this script to match your computer."
        )

    data = {name: load_batch(name) for name in BATCHES}

    # Strict batch holdout: validation, test, and normal-only baseline do not train the model.
    train_frames = [
        frame for name, frame in data.items()
        if name not in {VALIDATION_BATCH, TEST_BATCH, BASELINE_BATCH}
    ]
    train_df = pd.concat(train_frames, ignore_index=True)
    val_df = data[VALIDATION_BATCH]
    test_df = data[TEST_BATCH]

    X_train, feature_columns = make_features(train_df)
    X_val, _ = make_features(val_df, feature_columns)
    X_test, _ = make_features(test_df, feature_columns)

    y_train = train_df["anomaly"].astype(int)
    y_val = val_df["anomaly"].astype(int)
    y_test = test_df["anomaly"].astype(int)

    print("\nTraining classes:")
    print(y_train.value_counts().sort_index().rename(index=ANOMALY_NAMES).to_string())
    print(f"\nUsing {len(feature_columns)} numeric features.")

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    labels = [0, 1, 2, 3, 4]
    target_names = [ANOMALY_NAMES[i] for i in labels]

    print("\n===== VALIDATION MULTICLASS RESULTS =====")
    print(f"Validation batch: {VALIDATION_BATCH}")
    print(f"Accuracy: {accuracy_score(y_val, val_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_val, val_pred):.4f}")
    print(classification_report(
        y_val, val_pred, labels=labels, target_names=target_names,
        zero_division=0, digits=4
    ))

    print("\n===== HELD-OUT TEST MULTICLASS RESULTS =====")
    print(f"Test batch: {TEST_BATCH}")
    print(f"Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, test_pred):.4f}")
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, test_pred, labels=labels),
        index=[f"Actual: {ANOMALY_NAMES[i]}" for i in labels],
        columns=[f"Pred: {ANOMALY_NAMES[i]}" for i in labels],
    ).to_string())
    print("\nPer-class report:")
    print(classification_report(
        y_test, test_pred, labels=labels, target_names=target_names,
        zero_division=0, digits=4
    ))

    # Row-level predictions for later inspection and plotting.
    output = test_df.copy()
    output["predicted_anomaly_type"] = test_pred
    output["predicted_anomaly_name"] = [ANOMALY_NAMES[int(x)] for x in test_pred]
    output["actual_anomaly_name"] = [ANOMALY_NAMES[int(x)] for x in y_test]
    probabilities = model.predict_proba(X_test)
    for class_index, class_name in enumerate(model.named_steps["classifier"].classes_):
        output[f"probability_class_{int(class_index)}"] = probabilities[:, class_index]

    prediction_path = PROJECT_DIR / "daypsci_multiclass_test_predictions.csv"
    output.to_csv(prediction_path, index=False, encoding="utf-8-sig")

    report = classification_report(
        y_test, test_pred, labels=labels, target_names=target_names,
        zero_division=0, digits=4, output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = PROJECT_DIR / "daypsci_multiclass_test_report.csv"
    report_df.to_csv(report_path, encoding="utf-8-sig")

    cm_path = PROJECT_DIR / "daypsci_multiclass_confusion_matrix.csv"
    pd.DataFrame(
        confusion_matrix(y_test, test_pred, labels=labels),
        index=[f"Actual: {ANOMALY_NAMES[i]}" for i in labels],
        columns=[f"Pred: {ANOMALY_NAMES[i]}" for i in labels],
    ).to_csv(cm_path, encoding="utf-8-sig")

    print("\nSaved outputs:")
    print(prediction_path)
    print(report_path)
    print(cm_path)
    print("\nImportant: this is an experimental multiclass benchmark. Check the dataset README's exact meaning of anomaly codes before interpreting categories; results do not establish industrial safety or deployment readiness.")


if __name__ == "__main__":
    main()
