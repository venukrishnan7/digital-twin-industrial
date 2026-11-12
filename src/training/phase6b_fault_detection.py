"""
Phase 6B: Row-level fault detection with a factory-specific Random Forest.

Input: factory_phase6_dataset.csv from phase6_factory_rf.py.
Target: fault_active_sim (whether the simulator has activated a fault by this row).
Features deliberately exclude scenario labels, run IDs, rule flags, residuals,
fault-type labels, and machine_state/prediction_valid to reduce target leakage.

This is an experimental synthetic-data study, not a safety-certified controller.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "factory_phase6_dataset.csv"
PRED_FILE = BASE / "factory_phase6b_fault_predictions.csv"
REPORT_FILE = BASE / "factory_phase6b_fault_metrics.txt"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE.name}. Run phase6_factory_rf.py first."
        )
    df = pd.read_csv(DATA_FILE)
    target = "fault_active_sim"
    required = {"run_id", target}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Exclude targets, scenario metadata, rule-engine outputs, and residuals.
    excluded = {
        "run_id", "scenario_label", "fault_active_sim", "fault_type_sim",
        "timing_violation", "sensor_timeout", "safety_violation",
        "residual_part", "residual_clamp", "residual_drill_complete",
        "machine_state", "prediction_valid",
    }
    feature_cols = [c for c in df.columns if c not in excluded]
    categorical = [c for c in ("state_before",) if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(
        splitter.split(df[feature_cols], df[target], groups=df["run_id"])
    )
    X_train = df.iloc[train_idx][feature_cols]
    X_test = df.iloc[test_idx][feature_cols]
    y_train = df.iloc[train_idx][target].astype(int)
    y_test = df.iloc[test_idx][target].astype(int)

    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocessor", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced",
            min_samples_leaf=2, random_state=SEED, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    out = df.iloc[test_idx][["run_id", "cycle", "scenario_label", target]].copy()
    out["predicted_fault_active"] = pred
    out.to_csv(PRED_FILE, index=False)

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    report = classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["normal/inactive", "fault active"],
        zero_division=0
    )
    text = (
        "PHASE 6B — ROW-LEVEL FACTORY FAULT DETECTION\n"
        f"Rows: {len(df)} | Independent runs: {df['run_id'].nunique()}\n"
        f"Train rows: {len(train_idx)} | Test rows: {len(test_idx)}\n"
        f"Train runs: {df.iloc[train_idx]['run_id'].nunique()} | "
        f"Test runs: {df.iloc[test_idx]['run_id'].nunique()}\n"
        f"Features: {feature_cols}\n"
        f"Accuracy: {accuracy_score(y_test, pred):.4f}\n"
        f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.4f}\n"
        f"Fault precision: {precision_score(y_test, pred, zero_division=0):.4f}\n"
        f"Fault recall: {recall_score(y_test, pred, zero_division=0):.4f}\n"
        f"Fault F1: {f1_score(y_test, pred, zero_division=0):.4f}\n\n"
        f"Confusion matrix (rows actual [0,1], columns predicted [0,1]):\n{cm}\n\n"
        f"Classification report:\n{report}\n"
        "Interpretation: the model predicts simulator-defined active-fault labels. "
        "Results do not establish real-world generalization or safety suitability.\n"
    )
    REPORT_FILE.write_text(text, encoding="utf-8")
    print("PHASE 6B — ROW-LEVEL RANDOM FOREST FAULT DETECTION")
    print(f"Rows: {len(df)} | independent runs: {df['run_id'].nunique()}")
    print(f"Train: {len(train_idx)} rows / {df.iloc[train_idx]['run_id'].nunique()} runs")
    print(f"Test:  {len(test_idx)} rows / {df.iloc[test_idx]['run_id'].nunique()} runs")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, pred):.4f}")
    print(f"Fault precision: {precision_score(y_test, pred, zero_division=0):.4f}")
    print(f"Fault recall: {recall_score(y_test, pred, zero_division=0):.4f}")
    print(f"Fault F1: {f1_score(y_test, pred, zero_division=0):.4f}")
    print("\nConfusion matrix (rows actual [0,1], columns predicted [0,1]):")
    print(cm)
    print("\nClassification report:")
    print(report)
    print(f"Saved predictions: {PRED_FILE}")
    print(f"Saved metrics: {REPORT_FILE}")
    print("NOTE: Synthetic-data research experiment only; not a safety-certified controller.")

if __name__ == "__main__":
    main()
