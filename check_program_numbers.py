"""
Check Active program number values in raw data
"""

import sys
import pandas as pd

sys.path.insert(0, "src")

from src.utils.data_io import load_parquet

# Load raw data
print("Loading raw data...")
df = load_parquet("outputs/01_raw.parquet")
print(f"Data shape: {df.shape}")

if "Active program number" not in df.columns:
    print("\n❌ ERROR: 'Active program number' column not found!")
    sys.exit(1)

print("\n" + "=" * 70)
print("ACTIVE PROGRAM NUMBER ANALYSIS")
print("=" * 70)

prog = df["Active program number"]
print(f"\nStatistics:")
print(f"  Unique values: {prog.nunique()}")
print(f"  Min: {prog.min()}")
print(f"  Max: {prog.max()}")

# Count each program number
print(f"\n Value counts:")
value_counts = prog.value_counts().sort_index()
for val, count in value_counts.items():
    pct = count / len(df) * 100
    print(f"  Program {val:3.0f}: {count:7d} occurrences ({pct:5.2f}%)")

# Check for programs 10, 21, 32
print(f"\n" + "-" * 70)
print("KEY PROGRAMS:")
print("-" * 70)
print(f"  Program 10 (Filtration): {(prog == 10).sum()} occurrences")
print(f"  Program 21 (Backwashing): {(prog == 21).sum()} occurrences")
print(f"  Program 32 (Chemical cleaning): {(prog == 32).sum()} occurrences")

# Show transitions to programs 10, 21, 32
print(f"\n" + "-" * 70)
print("PROGRAM TRANSITIONS (First 20):")
print("-" * 70)

prog_diff = prog.diff()
transitions_10 = df[prog == 10][prog.diff() != 0].head(20)
transitions_21 = df[prog == 21][prog.diff() != 0].head(20)
transitions_32 = df[prog == 32][prog.diff() != 0].head(20)

print(f"\nTransitions to Program 10 (Filtration):")
for idx in transitions_10.index:
    prev_prog = df.loc[idx - 1, "Active program number"] if idx > 0 else 0
    timestamp = df.loc[idx, "TimeStamp"]
    print(f"  {timestamp}: {prev_prog:.0f} → 10")

print(f"\nTransitions to Program 21 (Backwashing):")
for idx in transitions_21.index:
    prev_prog = df.loc[idx - 1, "Active program number"] if idx > 0 else 0
    timestamp = df.loc[idx, "TimeStamp"]
    print(f"  {timestamp}: {prev_prog:.0f} → 21")

print(f"\nTransitions to Program 32 (Chemical cleaning):")
for idx in transitions_32.index:
    prev_prog = df.loc[idx - 1, "Active program number"] if idx > 0 else 0
    timestamp = df.loc[idx, "TimeStamp"]
    print(f"  {timestamp}: {prev_prog:.0f} → 32")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
