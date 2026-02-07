"""
Check raw data for Chemical timer column
"""

import sys
import pandas as pd

sys.path.insert(0, "src")

from src.utils.data_io import load_parquet

# Load raw data
print("Loading raw data...")
df_raw = load_parquet("outputs/01_raw.parquet")
print(f"Raw data shape: {df_raw.shape}")

print("\n" + "=" * 70)
print("COLUMNS IN RAW DATA")
print("=" * 70)

# Check for chemical timer
chemical_cols = [
    col for col in df_raw.columns if "chemical" in col.lower() or "timer" in col.lower()
]

if chemical_cols:
    print(f"\n✓ Found {len(chemical_cols)} columns related to chemical/timer:")
    for col in chemical_cols:
        print(f"  - {col}")
        print(f"    Non-zero values: {(df_raw[col] > 0).sum()} / {len(df_raw)}")
        print(f"    Min: {df_raw[col].min()}, Max: {df_raw[col].max()}")
        print(f"    Unique values: {df_raw[col].nunique()}")
else:
    print("\n❌ No columns containing 'chemical' or 'timer' found!")

print("\n" + "-" * 70)
print("ALL COLUMN NAMES IN RAW DATA:")
print("-" * 70)
for i, col in enumerate(sorted(df_raw.columns), 1):
    print(f"{i:3d}. {col}")
