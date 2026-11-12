"""
Phase 6C: Compare a factory-specific Random Forest with a conservative hybrid
decision that combines RF predictions with explicit rule-trigger overrides.

Requires factory_phase6_dataset.csv from phase6_factory_rf.py.
Uses run-grouped train/test split. Rule flags are NOT model features; they are
used only in the separate rule/hybrid comparison after model prediction.

This is a synthetic-data experiment, not a safety-certified controller.
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
PRED_FILE = BASE / "factory_phase6c_hybrid_predictions.csv"
REPORT_FILE = BASE / "factory_phase6c_hybrid_metrics.txt"

def metrics_block(name, y_true, y_pred):
    return (
        f"{name}\n"
        f"  Accuracy:          {accuracy_score(y_true, y_pred):.4f}\n"
        f"  Balanced accuracy: {balanced_accuracy_score(y_true, y_pred):.4f}\n"
        f"  Fault precision:   {precision_score(y_true, y_pred, zero_division=0):.4f}\n"
        f"  Fault recall:      {recall_score(y_true, y_pred, zero_division=0):.4f}\n"
        f"  Fault F1:          {f1_score(y_true, y_pred, zero_division=0):.4f}\n"
        f"  Confusion matrix (rows actual [0,1], columns predicted [0,1]):\n"
        f"{confusion_matrix(y_true, y_pred, labels=[0,1])}\n\n"
        f"{classification_report(y_true, y_pred, labels=[0,1], "
        f"target_names=['normal/inactive','fault active'], zero_division=0)}\n"
    )

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE.name}. Run phase6_factory_rf.py first."
        )
    df = pd.read_csv(DATA_FILE)
    target = "fault_active_sim"
    required = {"run_id", target, "timing_violation", "sensor_timeout", "safety_violation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    # Keep rules, labels, residuals and IDs out of RF features.
    excluded = {
        "run_id", "scenario_label", target, "fault_type_sim",
        "timing_violation", "sensor_timeout", "safety_violation",
        "residual_part", "residual_clamp", "residual_drill_complete",
        "machine_state", "prediction_valid",
    }
    feature_cols = [c for c in df.columns if c not in excluded]
    categorical = [c for c in ("state_before",) if c in feature_cols]
    numeric = [c for c in feature_cols if c not in categorical]

    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(
        split.split(df[feature_cols], df[target], groups=df["run_id"])
    )
    train = df.iloc[train_idx]
    test = df.iloc[test_idx]
    X_train, X_test = train[feature_cols], test[feature_cols]
    y_train, y_test = train[target].astype(int), test[target].astype(int)

    prep = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    rf = Pipeline([
        ("preprocessor", prep),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", min_samples_leaf=2,
            random_state=SEED, n_jobs=-1
        )),
    ])
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test).astype(int)

    # Conservative hybrid: an explicit current-row safety/timing/timeout trigger
    # forces a fault decision; otherwise use the RF prediction.
    rule_trigger = (
        test["timing_violation"].astype(int).to_numpy().astype(bool)
        | test["sensor_timeout"].astype(int).to_numpy().astype(bool)
        | test["safety_violation"].astype(int).to_numpy().astype(bool)
    )
    hybrid_pred = ((rf_pred == 1) | rule_trigger).astype(int)

    out = test[["run_id", "cycle", "scenario_label", target]].copy()
    out["rf_predicted_fault"] = rf_pred
    out["rule_trigger"] = rule_trigger.astype(int)
    out["hybrid_predicted_fault"] = hybrid_pred
    out.to_csv(PRED_FILE, index=False)

    report = (
        "PHASE 6C — RANDOM FOREST VS RULE-OVERRIDE HYBRID\n"
        f"Rows: {len(df)} | Independent runs: {df['run_id'].nunique()}\n"
        f"Train: {len(train_idx)} rows / {train['run_id'].nunique()} runs\n"
        f"Test: {len(test_idx)} rows / {test['run_id'].nunique()} runs\n"
        f"Features: {feature_cols}\n"
        "Hybrid policy: predict fault if RF predicts fault OR a current-row timing, "
        "sensor-timeout, or safety rule triggers.\n\n"
        + metrics_block("RANDOM FOREST ONLY", y_test, rf_pred)
        + metrics_block("HYBRID (RF + RULE OVERRIDE)", y_test, hybrid_pred)
        + "Caution: labels and rule triggers come from the same simulator. This measures "
        "agreement with simulator-generated labels, not independent real-world validation. "
        "The hybrid is an offline research comparison, not a controller.\n"
    )
    REPORT_FILE.write_text(report, encoding="utf-8")
    print("PHASE 6C — RANDOM FOREST VS HYBRID RULE OVERRIDE")
    print(f"Rows: {len(df)} | runs: {df['run_id'].nunique()}")
    print(f"Train: {len(train_idx)} rows / {train['run_id'].nunique()} runs")
    print(f"Test:  {len(test_idx)} rows / {test['run_id'].nunique()} runs")
    print("\n" + metrics_block("RANDOM FOREST ONLY", y_test, rf_pred))
    print(metrics_block("HYBRID (RF + RULE OVERRIDE)", y_test, hybrid_pred))
    print(f"Saved predictions: {PRED_FILE}")
    print(f"Saved metrics: {REPORT_FILE}")
    print("NOTE: Synthetic-data research experiment only; not a safety-certified controller.")

if __name__ == "__main__":
    main()
