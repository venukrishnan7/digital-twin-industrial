
from pathlib import Path
import pandas as pd

folder = (
    Path.home()
    / "Downloads"
    / "2026_Martin_DAYPSCI_v1"
    / "DAYPSCI_v1"
    / "pn_sensor_T2_batch03"
)

df = pd.read_csv(folder / "pn_sensor_T2_batch03.csv", sep=";")

print("Dataset shape:", df.shape)
print("\nAnomaly labels:")
print(df["anomaly"].value_counts(dropna=False))

print("\nFault labels:")
print(df["fault"].value_counts(dropna=False))

print("\nPart-level summary:")
summary = df.groupby("part_id").agg(
    event_count=("event_seq", "count"),
    anomaly_rows=("anomaly", "sum"),
    fault_rows=("fault", "sum"),
    mean_delta_ms=("delta_time_ms", "mean"),
    max_delta_ms=("delta_time_ms", "max")
)

print(summary.to_string())

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
