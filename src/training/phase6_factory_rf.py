"""
Phase 6: Generate varied factory-simulation runs and evaluate a factory-specific
Random Forest with run-level train/test separation.

This is an experimental synthetic-data pipeline, not a safety-certified controller.
It deliberately excludes scenario ID, run ID, rule outputs, labels, and residuals
from model features to reduce target leakage.
"""
import csv
import random
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42
N_RUNS_PER_SCENARIO = 100
OUTPUT_DIR = Path(__file__).resolve().parent
DATA_FILE = OUTPUT_DIR / "factory_phase6_dataset.csv"
PRED_FILE = OUTPUT_DIR / "factory_phase6_test_predictions.csv"
REPORT_FILE = OUTPUT_DIR / "factory_phase6_metrics.txt"

SCENARIOS = ("normal", "slow_drilling", "part_sensor_failure", "clamp_interlock")


def generate_run(scenario, run_id, rng):
    """Generate one sequential factory run with randomized timing/fault onset."""
    state = "IDLE"
    cycle = 0
    drill_timer = 0
    sensor_wait = 0
    drill_delay = rng.randint(4, 7) if scenario == "slow_drilling" else rng.randint(1, 3)
    max_drill_time = 3
    sensor_fault_after = rng.randint(2, 5) if scenario == "part_sensor_failure" else None
    # Small sensor noise/dropout for some otherwise-normal runs creates variability.
    benign_part_dropout = scenario == "normal" and rng.random() < 0.08
    part_detected = False
    clamp_closed = False
    drill_complete = False
    conveyor = False
    clamp_actuator = False
    drill_actuator = False
    fault_active = False
    fault_type = "normal"
    rows = []

    for _ in range(20):
        cycle += 1
        state_before = state
        timing_violation = 0
        sensor_timeout = 0
        safety_violation = 0
        residual_part = ""
        residual_clamp = ""
        residual_drill = ""
        prediction_valid = int(state != "FAULT")

        if state == "IDLE":
            conveyor = True
            state = "CONVEYING"
        elif state == "CONVEYING":
            sensor_wait += 1
            if scenario != "part_sensor_failure" and not (benign_part_dropout and sensor_wait == 1):
                part_detected = True
            if part_detected:
                conveyor = False
                state = "CLAMPING"
            elif sensor_wait >= sensor_fault_after if sensor_fault_after is not None else sensor_wait >= 5:
                sensor_timeout = 1
                fault_active = True
                fault_type = "part_sensor_failure"
                state = "FAULT"
        elif state == "CLAMPING":
            clamp_actuator = True
            if scenario == "clamp_interlock":
                clamp_actuator = False
                clamp_closed = False
            else:
                clamp_closed = True
            state = "DRILLING"
        elif state == "DRILLING":
            if not clamp_closed:
                safety_violation = 1
                fault_active = True
                fault_type = "clamp_interlock"
                drill_actuator = False
                state = "FAULT"
            else:
                drill_actuator = True
                drill_timer += 1
                if drill_timer > max_drill_time:
                    timing_violation = 1
                    if scenario == "slow_drilling":
                        fault_active = True
                        fault_type = "slow_drilling"
                if drill_timer >= drill_delay:
                    drill_actuator = False
                    drill_complete = True
                    state = "UNCLAMPING"
        elif state == "UNCLAMPING":
            clamp_actuator = False
            clamp_closed = False
            state = "RELEASING"
        elif state == "RELEASING":
            conveyor = True
            state = "COMPLETE"
        elif state == "COMPLETE":
            conveyor = False
            break
        elif state == "FAULT":
            conveyor = clamp_actuator = drill_actuator = False

        # Predictions/residuals are evaluated only while the normal process model is valid.
        if state != "FAULT":
            exp_part = state in ("CLAMPING", "DRILLING", "UNCLAMPING", "RELEASING", "COMPLETE")
            if state == "CONVEYING" and sensor_wait >= 1:
                exp_part = True
            exp_clamp = state in ("DRILLING", "UNCLAMPING")
            exp_drill = state in ("UNCLAMPING", "RELEASING", "COMPLETE") or (
                state == "DRILLING" and drill_timer >= drill_delay
            )
            residual_part = int(part_detected) - int(exp_part)
            residual_clamp = int(clamp_closed) - int(exp_clamp)
            residual_drill = int(drill_complete) - int(exp_drill)

        # The target is the run's injected scenario class, not the rule-derived fault label.
        # Run-level groups are used for train/test split, preventing rows from the same run
        # from appearing in both partitions.
        rows.append({
            "run_id": run_id,
            "cycle": cycle,
            "state_before": state_before,
            "machine_state": state,
            "drill_timer": drill_timer,
            "sensor_wait": sensor_wait,
            "part_detected": int(part_detected),
            "clamp_closed": int(clamp_closed),
            "drill_complete": int(drill_complete),
            "prediction_valid": int(state != "FAULT"),
            "residual_part": residual_part,
            "residual_clamp": residual_clamp,
            "residual_drill_complete": residual_drill,
            "actuator_conveyor": int(conveyor),
            "actuator_clamp": int(clamp_actuator),
            "actuator_drill": int(drill_actuator),
            "timing_violation": timing_violation,
            "sensor_timeout": sensor_timeout,
            "safety_violation": safety_violation,
            "scenario_label": scenario,
            "fault_active_sim": int(fault_active),
            "fault_type_sim": fault_type,
        })
        if state in ("COMPLETE", "FAULT"):
            # Keep a short post-terminal observation for state persistence.
            if cycle >= 2:
                break
    return rows


def main():
    rng = random.Random(SEED)
    rows = []
    run_number = 0
    for scenario in SCENARIOS:
        for _ in range(N_RUNS_PER_SCENARIO):
            run_number += 1
            rows.extend(generate_run(scenario, f"run_{run_number:04d}", rng))

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)

    # Predict the injected scenario class from process observations.
    # Exclude IDs, scenario labels, rule flags, residuals, and simulator ground-truth labels.
    target = "scenario_label"
    excluded = {
        "run_id", "cycle", target, "fault_active_sim", "fault_type_sim",
        "timing_violation", "sensor_timeout", "safety_violation",
        "residual_part", "residual_clamp", "residual_drill_complete",
    }
    feature_cols = [c for c in df.columns if c not in excluded]
    categorical = ["state_before", "machine_state"]
    numeric = [c for c in feature_cols if c not in categorical]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train_idx, test_idx = next(splitter.split(df[feature_cols], df[target], groups=df["run_id"]))
    X_train, X_test = df.iloc[train_idx][feature_cols], df.iloc[test_idx][feature_cols]
    y_train, y_test = df.iloc[train_idx][target], df.iloc[test_idx][target]

    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=SEED,
            min_samples_leaf=2, n_jobs=-1
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    test_out = df.iloc[test_idx][["run_id", "cycle", "scenario_label"]].copy()
    test_out["predicted_scenario"] = predictions
    test_out.to_csv(PRED_FILE, index=False)

    labels = list(SCENARIOS)
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    metrics_text = (
        "PHASE 6 FACTORY-SPECIFIC RANDOM FOREST — EXPERIMENTAL RESULTS\n"
        f"Seed: {SEED}\nRuns per scenario: {N_RUNS_PER_SCENARIO}\n"
        f"Total runs: {df['run_id'].nunique()}\nRows: {len(df)}\n"
        f"Train rows: {len(train_idx)} | Test rows: {len(test_idx)}\n"
        f"Train runs: {df.iloc[train_idx]['run_id'].nunique()} | "
        f"Test runs: {df.iloc[test_idx]['run_id'].nunique()}\n"
        f"Accuracy: {accuracy_score(y_test, predictions):.4f}\n"
        f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}\n\n"
        f"Class order: {labels}\nConfusion matrix:\n{matrix}\n\n"
        f"Classification report:\n{report}\n"
        "CAUTION: This is synthetic simulation data. The classifier predicts injected scenario "
        "classes, not verified real-world faults. Do not use it to control safety-critical equipment.\n"
    )
    REPORT_FILE.write_text(metrics_text, encoding="utf-8")

    print("PHASE 6 — FACTORY-SPECIFIC RANDOM FOREST")
    print(f"Generated {len(df)} rows across {df['run_id'].nunique()} independent runs.")
    print(f"Saved dataset: {DATA_FILE}")
    print(f"Training: {len(train_idx)} rows / {df.iloc[train_idx]['run_id'].nunique()} runs")
    print(f"Testing:  {len(test_idx)} rows / {df.iloc[test_idx]['run_id'].nunique()} runs")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.4f}")
    print("\nConfusion matrix (class order: " + ", ".join(labels) + "):")
    print(matrix)
    print("\nClassification report:")
    print(report)
    print(f"Saved predictions: {PRED_FILE}")
    print(f"Saved metrics: {REPORT_FILE}")
    print("NOTE: Synthetic-data research experiment only; not a safety-certified controller.")


if __name__ == "__main__":
    main()
