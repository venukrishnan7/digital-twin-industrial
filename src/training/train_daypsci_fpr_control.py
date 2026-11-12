"""
DAYPSCI threshold selection with a validation false-positive-rate constraint.

Training: all batches except validation, test, and normal-only baseline.
Validation: pn_sensor_T1_batch02 (select threshold; constrain validation FPR).
Test: pn_sensor_T2_batch03 (anomaly metrics).
Baseline: pn_baseline__batch01 (final normal-only false-alarm evaluation).

Default target: validation false-positive rate <= 10%.
This constraint is chosen before inspecting the final baseline result; change
MAX_VALIDATION_FPR only if you have a justified operating requirement.

Run:
    python train_daypsci_fpr_control.py
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
MAX_VALIDATION_FPR = 0.10

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
        raise FileNotFoundError(f"Dataset folder not found: {DATA_ROOT}")
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
        raise RuntimeError("No valid batch CSVs found.")
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
    group_cols = ["source_batch"] + (["part_id"] if "part_id" in df.columns else [])
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
    changed = []
    for signal in SIGNALS:
        if signal in df.columns:
            prev = grouped[signal].shift(1)
            name = signal + "_changed"
            df[name] = ((df[signal] != prev) & prev.notna()).astype(int)
            changed.append(name)
    if changed:
        df["signal_change_count"] = df[changed].sum(axis=1)
    return df


def make_model(X):
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("prep", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced_subsample",
            min_samples_leaf=2, max_features="sqrt", random_state=42, n_jobs=-1
        )),
    ])


def get_prob(model, X):
    classes = list(model.named_steps["classifier"].classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def main():
    data = add_temporal_features(load_data())
    batches = set(data["source_batch"].unique())
    for name in [VALIDATION_BATCH, TEST_BATCH, BASELINE_BATCH]:
        if name not in batches:
            raise RuntimeError(f"Missing required batch: {name}")

    train_batches = sorted(batches - {VALIDATION_BATCH, TEST_BATCH, BASELINE_BATCH})
    train = data[data["source_batch"].isin(train_batches)].copy()
    val = data[data["source_batch"] == VALIDATION_BATCH].copy()
    test = data[data["source_batch"] == TEST_BATCH].copy()
    baseline = data[data["source_batch"] == BASELINE_BATCH].copy()

    y_train = train["target_anomaly"].astype(int)
    y_val = val["target_anomaly"].astype(int)
    y_test = test["target_anomaly"].astype(int)
    y_base = baseline["target_anomaly"].astype(int)

    if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
        raise RuntimeError("Training, validation, and test must contain both classes.")

    excluded = EXCLUDE | {"source_batch", "target_anomaly"}
    features = [
        c for c in data.columns
        if c not in excluded and train[c].notna().any()
        and train[c].nunique(dropna=True) > 1
    ]
    X_train, X_val, X_test, X_base = (
        train[features].copy(), val[features].copy(),
        test[features].copy(), baseline[features].copy()
    )

    model = make_model(X_train)
    model.fit(X_train, y_train)
    val_prob = get_prob(model, X_val)

    val_normal = (y_val.to_numpy() == 0)
    candidates = []
    for threshold in np.arange(0.05, 0.951, 0.025):
        pred = (val_prob >= threshold).astype(int)
        fpr = float((pred[val_normal] == 1).mean()) if val_normal.any() else 1.0
        candidates.append({
            "threshold": float(threshold),
            "validation_fpr": fpr,
            "validation_precision": precision_score(y_val, pred, zero_division=0),
            "validation_recall": recall_score(y_val, pred, zero_division=0),
            "validation_f1": f1_score(y_val, pred, zero_division=0),
        })
    scores = pd.DataFrame(candidates)
    feasible = scores[scores["validation_fpr"] <= MAX_VALIDATION_FPR]
    if feasible.empty:
        # Defensive fallback: choose the candidate with the lowest validation FPR.
        chosen = scores.sort_values(
            ["validation_fpr", "validation_f1"], ascending=[True, False]
        ).iloc[0]
        selection_note = "No threshold met FPR constraint; selected lowest validation FPR."
    else:
        chosen = feasible.sort_values(
            ["validation_f1", "validation_precision", "threshold"],
            ascending=[False, False, False]
        ).iloc[0]
        selection_note = "Selected highest validation F1 among thresholds meeting FPR constraint."

    threshold = float(chosen["threshold"])
    test_prob = get_prob(model, X_test)
    base_prob = get_prob(model, X_base)
    test_pred = (test_prob >= threshold).astype(int)
    base_pred = (base_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, test_pred, labels=[0, 1])

    comparison = []
    for t in [0.10, 0.20, 0.30, 0.40, 0.50, threshold]:
        tp = (test_prob >= t).astype(int)
        bp = (base_prob >= t).astype(int)
        comparison.append({
            "threshold": t,
            "test_precision": precision_score(y_test, tp, zero_division=0),
            "test_recall": recall_score(y_test, tp, zero_division=0),
            "test_f1": f1_score(y_test, tp, zero_division=0),
            "test_false_positives": int(((y_test.to_numpy() == 0) & (tp == 1)).sum()),
            "test_false_negatives": int(((y_test.to_numpy() == 1) & (tp == 0)).sum()),
            "baseline_false_alarms": int(bp.sum()),
            "baseline_rows": len(bp),
            "baseline_false_alarm_rate": float(bp.mean()),
        })
    comparison_df = pd.DataFrame(comparison).drop_duplicates("threshold").sort_values("threshold")
    comparison_df.to_csv(OUTPUT_DIR / "daypsci_fpr_control_comparison.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "daypsci_fpr_control_validation_scores.csv", index=False)

    print("\n===== VALIDATION FPR-CONSTRAINED THRESHOLD =====")
    print(f"Validation batch: {VALIDATION_BATCH}; test batch: {TEST_BATCH}")
    print(f"Normal baseline batch: {BASELINE_BATCH}")
    print(f"Required validation FPR: <= {MAX_VALIDATION_FPR:.0%}")
    print(f"Selected threshold: {threshold:.3f}")
    print(selection_note)
    print(
        f"Validation FPR={chosen['validation_fpr']:.4f}, "
        f"precision={chosen['validation_precision']:.4f}, "
        f"recall={chosen['validation_recall']:.4f}, "
        f"F1={chosen['validation_f1']:.4f}"
    )
    print("\nTest confusion matrix:")
    print(cm)
    print(
        f"Test accuracy={accuracy_score(y_test, test_pred):.4f}; "
        f"precision={precision_score(y_test, test_pred, zero_division=0):.4f}; "
        f"recall={recall_score(y_test, test_pred, zero_division=0):.4f}; "
        f"F1={f1_score(y_test, test_pred, zero_division=0):.4f}"
    )
    print(
        f"Baseline false alarms: {int(base_pred.sum())}/{len(base_pred)} "
        f"({base_pred.mean():.2%})"
    )
    print("\nThreshold comparison:")
    print(comparison_df.to_string(index=False))
    print(f"\nSaved: {OUTPUT_DIR / 'daypsci_fpr_control_comparison.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'daypsci_fpr_control_validation_scores.csv'}")
    print(
        "\nNote: the 10% constraint applies to the validation batch, not the baseline batch. "
        "Baseline false-alarm rate is a separate final diagnostic."
    )


if __name__ == "__main__":
    main()
