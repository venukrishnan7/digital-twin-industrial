from pathlib import Path
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

DATASET_ROOT = Path.home() / "Downloads" / "2026_Martin_DAYPSCI_v1" / "DAYPSCI_v1"
TARGET_BATCH = "pn_sensor_T2_batch03"

EXCLUDED = {
    "ts_plc", "event_seq", "anomaly", "fault", "FSx", "TFSx",
    "FCVx", "TFCVx", "batch_id", "experiment_id", "part_id"
}

def load_batch():
    folder = DATASET_ROOT / TARGET_BATCH
    files = list(folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found in {folder}")
    df = pd.read_csv(files[0], sep=";")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}. "
            "Edit DATASET_ROOT at the top of this script."
        )

    df = load_batch()
    df["anomaly"] = pd.to_numeric(df["anomaly"], errors="coerce")
    df["part_id"] = pd.to_numeric(df["part_id"], errors="coerce")
    df = df.dropna(subset=["anomaly", "part_id"]).copy()

    # Binary target: 0 = normal; any nonzero anomaly class = anomalous.
    y = (df["anomaly"] != 0).astype(int)

    feature_cols = [c for c in df.columns if c not in EXCLUDED]
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Drop columns that are entirely missing or constant in this batch.
    X = X.dropna(axis=1, how="all")
    constant_cols = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    X = X.drop(columns=constant_cols)
    valid = ~X.isna().all(axis=1)
    df, X, y = df.loc[valid].copy(), X.loc[valid].copy(), y.loc[valid].copy()
    groups = df["part_id"]

    if y.nunique() < 2:
        raise ValueError("Selected batch must contain both normal and anomalous rows.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    if train_groups.intersection(test_groups):
        raise RuntimeError("Part-level leakage detected.")
    if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
        raise ValueError(
            "This part-level split does not contain both classes in train and test. "
            "We should adjust the split or use multiple batches."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=42, min_samples_leaf=2, n_jobs=-1
        ))
    ])
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = model.predict(X.iloc[test_idx])
    y_test = y.iloc[test_idx]

    print("===== DAYPSCI ANOMALY DETECTION =====")
    print("Batch:", TARGET_BATCH)
    print("Rows:", len(df))
    print("Feature columns:", list(X.columns))
    print("Dropped constant columns:", constant_cols)
    print("Target counts (0=normal, 1=anomaly):")
    print(y.value_counts().sort_index().to_string())
    print("Train rows:", len(train_idx), "| Test rows:", len(test_idx))
    print("Train part IDs:", sorted(train_groups))
    print("Test part IDs:", sorted(test_groups))
    print("\nConfusion matrix (true rows, predicted columns; labels 0,1):")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print("\nClassification report:")
    print(classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["normal", "anomaly"], zero_division=0
    ))
    print(f"Anomaly precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Anomaly recall:    {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"Anomaly F1-score:  {f1_score(y_test, pred, zero_division=0):.4f}")

    predictions = df.iloc[test_idx][
        ["event_seq", "part_id", "event_type", "gemma", "anomaly", "fault"]
    ].copy()
    predictions["true_binary_anomaly"] = y_test.to_numpy()
    predictions["predicted_binary_anomaly"] = pred
    predictions["correct"] = (
        predictions["true_binary_anomaly"] == predictions["predicted_binary_anomaly"]
    )
    pred_path = Path.cwd() / "daypsci_test_predictions.csv"
    predictions.to_csv(pred_path, index=False)

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.named_steps["classifier"].feature_importances_
    }).sort_values("importance", ascending=False)
    importance_path = Path.cwd() / "daypsci_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    print("\nSaved predictions:", pred_path)
    print("Saved feature importance:", importance_path)
    print("\nThis is a within-batch, held-out-part baseline; it does not prove "
          "generalization to other batches or validate the full Digital Twin.")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
