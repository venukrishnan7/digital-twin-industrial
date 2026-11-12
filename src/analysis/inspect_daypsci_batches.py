from pathlib import Path
import pandas as pd

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "2026_Martin_DAYPSCI_v1"
    / "DAYPSCI_v1"
)

def read_batch(csv_path):
    """Read one semicolon-separated DAYPSCI CSV file."""
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [column.strip() for column in df.columns]
    return df

def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_ROOT}\n"
            "Edit DATASET_ROOT in this script to match your folder."
        )

    csv_files = sorted(DATASET_ROOT.glob("*/*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found inside folders under {DATASET_ROOT}"
        )

    summaries = []

    for csv_path in csv_files:
        df = read_batch(csv_path)

        summary = {
            "batch": csv_path.parent.name,
            "rows": len(df),
            "columns": len(df.columns),
            "anomaly_positives": (
                int((pd.to_numeric(df["anomaly"], errors="coerce") == 1).sum())
                if "anomaly" in df.columns else None
            ),
            "fault_positives": (
                int((pd.to_numeric(df["fault"], errors="coerce") == 1).sum())
                if "fault" in df.columns else None
            ),
            "event_types": (
                str(df["event_type"].value_counts(dropna=False).to_dict())
                if "event_type" in df.columns else "missing"
            ),
            "missing_cells": int(df.isna().sum().sum())
        }
        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)
    print("\n===== DAYPSCI BATCH SUMMARY =====")
    print(summary_df.to_string(index=False))

    output_path = Path.cwd() / "daypsci_batch_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")

    print(
        "\nNext: inspect which batches contain positive anomaly/fault labels. "
        "Do not train a supervised fault classifier until the labels and "
        "meaning of each batch are understood."
    )

if __name__ == "__main__":
    main()
