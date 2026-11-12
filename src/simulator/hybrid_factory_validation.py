"""Hybrid factory validation scaffold: rule-based twin + schema-safe RF hook.

This script validates the Phase 5C factory CSV and produces rule-based
hybrid decisions. It deliberately does NOT apply the DAYPSCI Random Forest
model to factory features because the two datasets have different schemas.
A factory-specific RF model can be plugged in only after it is trained and
its exact feature schema is saved alongside the model.
"""
from pathlib import Path
import json
import sys
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_DIR / "factory_twin_phase5b_data.csv"
OUTPUT_CSV = PROJECT_DIR / "factory_twin_hybrid_validation.csv"
SCHEMA_FILE = PROJECT_DIR / "factory_rf_feature_schema.json"
MODEL_FILE = PROJECT_DIR / "factory_random_forest.joblib"

REQUIRED = {
    "run_id", "mode", "cycle", "machine_state", "prediction_valid",
    "residual_count", "residual_mismatch", "timing_violation",
    "sensor_timeout", "safety_violation", "rule_event", "fault_event",
    "fault_active", "fault_label", "fault_type"
}

# These are observations/context fields available to a future factory-specific
# RF. Label/decision fields are intentionally excluded to avoid target leakage.
FACTORY_FEATURES = [
    "mode", "cycle", "drill_timer", "sensor_wait", "part_detected",
    "clamp_closed", "drill_complete", "prediction_valid",
    "predicted_part", "predicted_clamp", "predicted_drill_complete",
    "residual_part", "residual_clamp", "residual_drill_complete",
    "residual_count", "timing_violation", "sensor_timeout",
    "safety_violation", "actuator_conveyor", "actuator_clamp", "actuator_drill"
]


def load_data():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. Run factory_simulation_phase5c.py first."
        )
    df = pd.read_csv(INPUT_CSV)
    missing = sorted(REQUIRED - set(df.columns))
    if missing:
        raise ValueError(f"CSV is missing required Phase 5C columns: {missing}")
    return df


def validate_labels(df):
    checks = {}
    checks["fault_label_equals_fault_active"] = bool(
        (pd.to_numeric(df["fault_label"], errors="coerce") ==
         pd.to_numeric(df["fault_active"], errors="coerce")).all()
    )
    checks["fault_event_equals_mismatch_or_rule"] = bool(
        (pd.to_numeric(df["fault_event"], errors="coerce") ==
         ((pd.to_numeric(df["residual_mismatch"], errors="coerce") == 1) |
          (pd.to_numeric(df["rule_event"], errors="coerce") == 1)).astype(int)).all()
    )
    rule_flags = ["timing_violation", "sensor_timeout", "safety_violation"]
    expected_rule_event = df[rule_flags].fillna(0).astype(int).max(axis=1)
    checks["rule_event_matches_rule_flags"] = bool(
        (pd.to_numeric(df["rule_event"], errors="coerce") == expected_rule_event).all()
    )
    valid = pd.to_numeric(df["prediction_valid"], errors="coerce").fillna(0).astype(int)
    mismatch = pd.to_numeric(df["residual_mismatch"], errors="coerce").fillna(0).astype(int)
    checks["no_residual_mismatch_when_prediction_invalid"] = bool(
        ((valid == 0) & (mismatch != 0)).sum() == 0
    )
    return checks


def build_rule_decisions(df):
    out = df.copy()
    # The rule-based layer is the authoritative result in this prototype.
    # Safety and timeout events take priority; residual-only detections are
    # retained as early warnings, not conflated with a confirmed rule event.
    out["hybrid_decision"] = "normal"
    out.loc[pd.to_numeric(out["residual_mismatch"], errors="coerce").fillna(0).astype(int) == 1,
            "hybrid_decision"] = "residual_warning"
    out.loc[pd.to_numeric(out["timing_violation"], errors="coerce").fillna(0).astype(int) == 1,
            "hybrid_decision"] = "timing_fault"
    out.loc[pd.to_numeric(out["sensor_timeout"], errors="coerce").fillna(0).astype(int) == 1,
            "hybrid_decision"] = "sensor_timeout_fault"
    out.loc[pd.to_numeric(out["safety_violation"], errors="coerce").fillna(0).astype(int) == 1,
            "hybrid_decision"] = "safety_interlock_fault"
    out["rf_prediction"] = "not_run_schema_incompatible"
    out["rf_probability"] = pd.NA
    out["rf_status"] = "No compatible factory-specific RF model supplied"
    return out


def check_rf_compatibility():
    if not MODEL_FILE.exists() or not SCHEMA_FILE.exists():
        return False, "No factory-specific RF model/schema found; DAYPSCI model must not be applied directly."
    try:
        schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
        saved_features = schema.get("features", [])
    except Exception as exc:
        return False, f"Could not read factory RF feature schema: {exc}"
    if not saved_features:
        return False, "Factory RF feature schema is empty."
    missing = [c for c in saved_features if c not in FACTORY_FEATURES]
    if missing:
        return False, f"Model schema contains features unavailable in this pipeline: {missing}"
    return False, "Model artifact loading is intentionally not enabled in this validation scaffold; add a reviewed, factory-trained model loader after schema validation."


def main():
    df = load_data()
    checks = validate_labels(df)
    out = build_rule_decisions(df)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print("HYBRID FACTORY VALIDATION — RULE LAYER + RF COMPATIBILITY CHECK")
    print(f"Input: {INPUT_CSV}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print("\nLabel validation:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print("\nRule decision counts:")
    print(out["hybrid_decision"].value_counts(dropna=False).to_string())
    compatible, message = check_rf_compatibility()
    print("\nRandom Forest integration status:")
    print(f"  {'READY' if compatible else 'NOT RUN'}: {message}")
    print("\nImportant: this script does not claim ML performance. The 31-row simulation is for rule validation only.")
    print(f"Saved: {OUTPUT_CSV}")
    if not all(checks.values()):
        print("\nAt least one consistency check failed; inspect the CSV before proceeding.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
