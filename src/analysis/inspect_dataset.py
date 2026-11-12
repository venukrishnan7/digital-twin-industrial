
import pandas as pd
from pathlib import Path

folder = Path.home() / "Downloads" / "2026_Martin_DAYPSCI_v1" / "DAYPSCI_v1"

print("Dataset folder:", folder)
print("\nFiles in the folder:")

for path in folder.iterdir():
    if path.is_dir():
        print("\nFolder:", path.name)
        for file in path.iterdir():
            print("  ", file.name)

data_file = folder / "pn_baseline__batch01" / "pn_baseline_batch01.csv"

if not data_file.exists():
    # Find the data file if Windows hides its extension
    candidates = [
        p for p in (folder / "pn_baseline__batch01").iterdir()
        if p.is_file() and p.suffix.lower() in (".csv", ".txt", ".data")
    ]

    if candidates:
        data_file = candidates[0]
    else:
        raise FileNotFoundError(
            "CSV file not found. Check the filename and folder."
        )

df = pd.read_csv(data_file, sep=";")

print("\nDataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head().to_string())

print("\nMissing values per column:")
print(df.isnull().sum().to_string())

print("\nPossible fault-related columns:")
for column in df.columns:
    if any(word in column.lower() for word in
           ["fault", "anomaly", "event_type", "label"]):
        print(column, ":", df[column].value_counts(dropna=False).head(10).to_dict())
