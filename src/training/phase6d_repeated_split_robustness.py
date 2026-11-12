"""
Phase 6D — robustness check across repeated run-grouped train/test splits.

This checks how much Phase 6C metrics vary with different random splits.
It is not external validation: all rows still come from the same simulator generator.
"""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "factory_phase6_dataset.csv"
RESULTS_FILE = BASE / "factory_phase6d_repeated_split_results.csv"
SUMMARY_FILE = BASE / "factory_phase6d_summary.txt"
SEEDS = [11, 23, 42, 57, 81, 104, 206, 315, 421, 509]


def score(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "fault_precision": precision_score(y_true, y_pred, zero_division=0),
        "fault_recall": recall_score(y_true, y_pred, zero_division=0),
        "fault_f1": f1_score(y_true, y_pred, zero_division=0),
    }


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE.name}. Run phase6_factory_rf.py first."
        )
    df = pd.read_csv(DATA_FILE)
    target = "fault_active_sim"
    needed = {
        "run_id", target, "timing_violation", "sensor_timeout", "safety_violation"
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    excluded = {
        "run_id", "scenario_label", target, "fault_type_sim",
        "timing_violation", "sensor_timeout", "safety_violation",
        "residual_part", "residual_clamp", "residual_drill_complete",
        "machine_state", "prediction_valid",
    }
    features = [col for col in df.columns if col not in excluded]
    categorical = [col for col in ["state_before"] if col in features]
    numeric = [col for col in features if col not in categorical]

    rows = []
    for seed in SEEDS:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.25, random_state=seed
        )
        train_idx, test_idx = next(
            splitter.split(df[features], df[target], groups=df["run_id"])
        )
        train, test = df.iloc[train_idx], df.iloc[test_idx]
        preprocessor = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ])
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=400, class_weight="balanced",
                min_samples_leaf=2, random_state=seed, n_jobs=-1
            )),
        ])
        model.fit(train[features], train[target].astype(int))
        y_true = test[target].astype(int)
        rf_pred = model.predict(test[features]).astype(int)

        rule_trigger = (
            test["timing_violation"].astype(int).to_numpy().astype(bool)
            | test["sensor_timeout"].astype(int).to_numpy().astype(bool)
            | test["safety_violation"].astype(int).to_numpy().astype(bool)
        )
        hybrid_pred = ((rf_pred == 1) | rule_trigger).astype(int)

        for method, pred in (("random_forest", rf_pred), ("hybrid", hybrid_pred)):
            item = {
                "seed": seed,
                "method": method,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "train_runs": train["run_id"].nunique(),
                "test_runs": test["run_id"].nunique(),
            }
            item.update(score(y_true, pred))
            rows.append(item)

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_FILE, index=False)
    metrics = [
        "accuracy", "balanced_accuracy", "fault_precision", "fault_recall", "fault_f1"
    ]
    summary_lines = [
        "PHASE 6D — REPEATED GROUPED-SPLIT ROBUSTNESS CHECK",
        f"Dataset rows: {len(df)}",
        f"Independent runs: {df['run_id'].nunique()}",
        f"Random split seeds: {SEEDS}",
        "Each split keeps entire runs together in train or test.",
        "",
    ]
    for method in ("random_forest", "hybrid"):
        subset = results[results["method"] == method]
        summary_lines.append(method.upper())
        for metric in metrics:
            mean = subset[metric].mean()
            std = subset[metric].std(ddof=1)
            summary_lines.append(f"  {metric}: mean={mean:.4f}, std={std:.4f}")
        summary_lines.append("")
    summary_lines.append(
        "Caution: repeated splits measure split sensitivity only. They do not "
        "establish generalization to real equipment or independently generated data."
    )
    SUMMARY_FILE.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSaved per-split results: {RESULTS_FILE}")
    print(f"Saved summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
