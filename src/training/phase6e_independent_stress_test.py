"""
Phase 6E: Independent simulation stress test for factory digital-twin fault detection.

Purpose:
- Train Random Forest only on the existing baseline factory_phase6_dataset.csv.
- Generate a separate stress-test dataset using shifted/unseen operating conditions.
- Evaluate the frozen RF and a hybrid RF + explicit-rule override on the stress-test runs.
- Keep runs grouped: all rows from a run are only in one dataset.
- This is a synthetic simulator stress test, not evidence of real-world safety.

Run from the project folder:
    python phase6e_independent_stress_test.py

Input:
    factory_phase6_dataset.csv (created by Phase 6 / Phase 6B)

Outputs:
    factory_phase6e_stress_dataset.csv
    factory_phase6e_stress_predictions.csv
    factory_phase6e_metrics.txt
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

BASE_FILE = Path("factory_phase6_dataset.csv")
STRESS_DATA_FILE = Path("factory_phase6e_stress_dataset.csv")
PREDICTIONS_FILE = Path("factory_phase6e_stress_predictions.csv")
METRICS_FILE = Path("factory_phase6e_metrics.txt")

RANDOM_SEED = 20261027
N_STRESS_RUNS_PER_CASE = 50

# Match the Phase 6B feature exclusions to prevent target/metadata leakage.
EXCLUDED_COLUMNS = {
    "fault_active_sim", "scenario_label", "run_id", "row_id",
    "rule_event", "fault_event", "fault_type", "machine_state",
    "prediction_valid", "residual_mismatch", "timing_violation",
    "sensor_timeout", "safety_violation", "fault_active", "fault_label",
    "fault_type_label", "scenario_id", "scenario_name"
}

def feature_columns(df):
    cols = []
    for c in df.columns:
        if c in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

def safe_int(row, name, default=0):
    value = row.get(name, default)
    if pd.isna(value):
        return default
    return int(value)

def make_stress_dataset(base_df):
    """
    Generate held-out runs from baseline event sequences with perturbed conditions.
    We use the existing simulator output as a template, then alter sensor/temporal
    evidence and rule flags under a deterministic set of stress cases. This does
    not call or depend on the original simulator's internal classes.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    if "scenario_label" not in base_df or "run_id" not in base_df:
        raise ValueError("Input dataset must contain scenario_label and run_id.")
    if "fault_active_sim" not in base_df:
        raise ValueError("Input dataset must contain fault_active_sim.")

    # Pick representative sequences by scenario. Each template remains a separate
    # run and stress cases are assigned new IDs.
    scenarios = list(base_df["scenario_label"].dropna().unique())
    if not scenarios:
        raise ValueError("No scenario labels found in input dataset.")

    # We need rule flags present for the hybrid baseline; if absent, derive zeros.
    template_groups = {
        scenario: [g.sort_values("row_id") if "row_id" in g else g.reset_index(drop=True)
                   for _, g in base_df[base_df["scenario_label"] == scenario].groupby("run_id")]
        for scenario in scenarios
    }
    output = []
    new_run_num = 0

    # Cases: nominal, parameter-shifted single faults, noisy sensors, weak rule cues,
    # and mixed-fault stress. Test labels are constructed by scenario/fault intent,
    # not copied from rule flags.
    scenario_alias = {str(s).lower(): s for s in scenarios}
    def find_scenario(keywords, fallback_index):
        for label in scenarios:
            low = str(label).lower()
            if all(k in low for k in keywords):
                return label
        return scenarios[fallback_index % len(scenarios)]

    normal_s = find_scenario(["normal"], 0)
    slow_s = find_scenario(["slow"], min(1, len(scenarios)-1))
    sensor_s = find_scenario(["sensor"], min(2, len(scenarios)-1))
    clamp_s = find_scenario(["clamp"], min(3, len(scenarios)-1))

    cases = [
        ("unseen_nominal_noise", normal_s, "nominal"),
        ("extreme_slow_drilling", slow_s, "slow"),
        ("late_sensor_failure", sensor_s, "sensor"),
        ("intermittent_interlock", clamp_s, "clamp"),
        ("mixed_slow_plus_sensor", slow_s, "mixed"),
    ]

    for case_name, template_scenario, case_type in cases:
        candidates = template_groups.get(template_scenario, [])
        if not candidates:
            candidates = [g for groups in template_groups.values() for g in groups]
        for run_i in range(N_STRESS_RUNS_PER_CASE):
            new_run_num += 1
            template = candidates[int(rng.integers(0, len(candidates)))].copy()
            template = template.reset_index(drop=True)
            n = len(template)
            if n == 0:
                continue

            # Preserve only the observable numeric/sequence shape, then apply new run ID.
            template["run_id"] = f"stress_{case_name}_{run_i:03d}"
            if "row_id" in template:
                template["row_id"] = np.arange(n)
            template["stress_case"] = case_name

            # Ground truth follows the stress-case design. Nominal rows remain 0.
            y = np.zeros(n, dtype=int)
            if case_type == "slow":
                # Stress delays are more severe than the baseline slow-drilling examples.
                start = max(0, n // 3)
                y[start:] = 1
                for col in ("drill_timer", "drill_elapsed", "drill_time", "delta_time_ms"):
                    if col in template.columns:
                        template.loc[start:, col] = template.loc[start:, col].astype(float) * rng.uniform(1.35, 1.85) + rng.normal(0, 0.08, n-start)
                if "timing_violation" in template.columns:
                    template["timing_violation"] = 0
                    # Weak explicit rule cues: only some delayed rows trip the rule.
                    idx = np.arange(start, n)
                    if len(idx):
                        picked = rng.choice(idx, size=max(1, len(idx)//3), replace=False)
                        template.loc[picked, "timing_violation"] = 1
            elif case_type == "sensor":
                start = max(0, int(n * rng.uniform(0.55, 0.85)))
                y[start:] = 1
                for col in ("part_sensor", "part_present", "sensor_part", "sensor_value"):
                    if col in template.columns:
                        template.loc[start:, col] = 0
                if "sensor_timeout" in template.columns:
                    template["sensor_timeout"] = 0
                    idx = np.arange(start, n)
                    if len(idx):
                        picked = rng.choice(idx, size=max(1, len(idx)//4), replace=False)
                        template.loc[picked, "sensor_timeout"] = 1
            elif case_type == "clamp":
                start = max(0, int(n * rng.uniform(0.25, 0.7)))
                y[start:] = 1
                for col in ("clamp_closed", "clamp_sensor", "clamp_feedback", "clamp_value"):
                    if col in template.columns:
                        template.loc[start:, col] = 0
                if "safety_violation" in template.columns:
                    template["safety_violation"] = 0
                    idx = np.arange(start, n)
                    if len(idx):
                        picked = rng.choice(idx, size=max(1, len(idx)//4), replace=False)
                        template.loc[picked, "safety_violation"] = 1
            elif case_type == "mixed":
                start1 = max(0, n // 3)
                start2 = max(start1, int(n * 0.6))
                y[start1:] = 1
                # Mixed fault changes a sensor-like signal and a time feature.
                for col in ("part_sensor", "part_present", "sensor_part", "sensor_value"):
                    if col in template.columns:
                        template.loc[start2:, col] = 0
                for col in ("drill_timer", "drill_elapsed", "drill_time", "delta_time_ms"):
                    if col in template.columns:
                        template.loc[start1:, col] = template.loc[start1:, col].astype(float) * rng.uniform(1.4, 2.0) + rng.normal(0, 0.1, n-start1)
                for col in ("timing_violation", "sensor_timeout", "safety_violation"):
                    if col in template.columns:
                        template[col] = 0
            # Nominal case: unseen sensor noise, no injected fault.
            if case_type == "nominal":
                y[:] = 0
                for col in template.select_dtypes(include=[np.number]).columns:
                    if col in {"row_id", "fault_active_sim"}:
                        continue
                    vals = template[col].astype(float)
                    scale = float(vals.std()) if float(vals.std()) > 0 else 1.0
                    template[col] = vals + rng.normal(0, 0.08 * scale, n)

            # Make target authoritative and clear labels/rule events from model inputs.
            template["fault_active_sim"] = y
            for col in ("fault_active", "fault_label", "fault_event", "rule_event"):
                if col in template.columns:
                    template[col] = y if col in ("fault_active", "fault_label") else 0
            if "scenario_label" in template.columns:
                template["scenario_label"] = f"stress_{case_name}"
            if "fault_type" in template.columns:
                template["fault_type"] = case_type if case_type != "nominal" else "none"

            # Preserve stress-case type for subgroup reporting only, not as a feature.
            template["stress_case"] = case_name
            output.append(template)

    if not output:
        raise ValueError("No stress-test runs were generated.")
    return pd.concat(output, ignore_index=True)

def metrics_for(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "fault_precision": precision_score(y_true, y_pred, zero_division=0),
        "fault_recall": recall_score(y_true, y_pred, zero_division=0),
        "fault_f1": f1_score(y_true, y_pred, zero_division=0),
        "missed_faults": int(((np.asarray(y_true) == 1) & (np.asarray(y_pred) == 0)).sum()),
        "false_alarms": int(((np.asarray(y_true) == 0) & (np.asarray(y_pred) == 1)).sum()),
    }

def main():
    if not BASE_FILE.exists():
        print(f"ERROR: {BASE_FILE} not found.")
        print("Run Phase 6 first so factory_phase6_dataset.csv exists in this folder.")
        sys.exit(1)

    base = pd.read_csv(BASE_FILE)
    if base.empty:
        raise ValueError("Baseline dataset is empty.")

    # Train only on baseline data, excluding identifiers and direct rule/target leakage.
    features = feature_columns(base)
    if not features:
        raise ValueError("No numeric features remain after leakage exclusions.")
    if "fault_active_sim" not in base.columns:
        raise ValueError("Target fault_active_sim is missing.")
    if base["fault_active_sim"].nunique() < 2:
        raise ValueError("Baseline target must contain both 0 and 1.")

    groups = base["run_id"] if "run_id" in base.columns else np.arange(len(base))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_SEED)
    train_idx, _ = next(splitter.split(base[features], base["fault_active_sim"], groups))
    train = base.iloc[train_idx]
    model = RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=RANDOM_SEED,
        n_jobs=-1
    )
    model.fit(train[features].fillna(0), train["fault_active_sim"].astype(int))

    stress = make_stress_dataset(base)
    # Ensure test features match the training schema exactly.
    for col in features:
        if col not in stress.columns:
            stress[col] = 0
    X_stress = stress[features].fillna(0)
    y = stress["fault_active_sim"].astype(int).to_numpy()
    rf_pred = model.predict(X_stress)

    # Explicit rules are an override layer; evaluate only rule flags present on each row.
    rule_cols = [c for c in ("timing_violation", "sensor_timeout", "safety_violation") if c in stress.columns]
    rule_signal = np.zeros(len(stress), dtype=int)
    for c in rule_cols:
        rule_signal |= stress[c].fillna(0).astype(int).to_numpy()
    hybrid_pred = ((rf_pred == 1) | (rule_signal == 1)).astype(int)

    stress["rf_prediction"] = rf_pred
    stress["hybrid_prediction"] = hybrid_pred
    stress["rule_signal"] = rule_signal
    stress["fault_active_sim"] = y
    stress.to_csv(STRESS_DATA_FILE, index=False)

    output_cols = [c for c in ["run_id", "row_id", "stress_case", "fault_active_sim",
                               "rf_prediction", "hybrid_prediction", "rule_signal"] if c in stress.columns]
    stress[output_cols].to_csv(PREDICTIONS_FILE, index=False)

    lines = []
    lines.append("PHASE 6E: INDEPENDENT SIMULATION STRESS TEST")
    lines.append("=" * 52)
    lines.append(f"Baseline input: {BASE_FILE}")
    lines.append(f"Baseline rows: {len(base)}")
    lines.append(f"Baseline runs: {base['run_id'].nunique() if 'run_id' in base else 'N/A'}")
    lines.append(f"Stress rows: {len(stress)}")
    lines.append(f"Stress runs: {stress['run_id'].nunique() if 'run_id' in stress else 'N/A'}")
    lines.append(f"Features used ({len(features)}): {', '.join(features)}")
    lines.append("")
    for name, pred in (("Random Forest", rf_pred), ("Hybrid RF + rule override", hybrid_pred)):
        m = metrics_for(y, pred)
        lines.append(name)
        for k, v in m.items():
            lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        lines.append("  confusion_matrix [[TN, FP], [FN, TP]]:")
        lines.append(str(confusion_matrix(y, pred, labels=[0, 1]).tolist()))
        lines.append("")
    lines.append("Per-stress-case results")
    lines.append("-" * 52)
    for case_name, group in stress.groupby("stress_case", sort=True):
        y_case = group["fault_active_sim"].astype(int).to_numpy()
        for model_name, pred_col in (("RF", "rf_prediction"), ("Hybrid", "hybrid_prediction")):
            p = group[pred_col].astype(int).to_numpy()
            m = metrics_for(y_case, p)
            lines.append(
                f"{case_name} | {model_name} | rows={len(group)} | "
                f"fault_recall={m['fault_recall']:.4f} | "
                f"fault_precision={m['fault_precision']:.4f} | "
                f"missed_faults={m['missed_faults']} | false_alarms={m['false_alarms']}"
            )
    lines.append("")
    lines.append("CAUTION: This is a simulator-based stress test with constructed labels.")
    lines.append("It is not a real-world validation or a safety certification.")
    METRICS_FILE.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print("\nSaved:")
    print(f"  {STRESS_DATA_FILE}")
    print(f"  {PREDICTIONS_FILE}")
    print(f"  {METRICS_FILE}")

if __name__ == "__main__":
    main()
