"""
Analyze Chemical Timer to understand cycle detection
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "src")

from src.utils.data_io import load_parquet
from src.cleaning_analysis import detect_chemical_timer_changes, merge_cleaning_events

# Load data - use raw data which has Chemical timer column
print("Loading data...")
df = load_parquet("outputs/01_raw.parquet")
print(f"Data shape: {df.shape}")
print(f"Date range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

# Filter to match the baseline start date
baseline_start = pd.Timestamp("2026-01-28 15:00:00")
df = df[df["TimeStamp"] >= baseline_start].copy()
print(f"After filtering from {baseline_start}: {df.shape}")

# Check if chemical timer exists
if "Chemical timer" not in df.columns:
    print("\n❌ ERROR: 'Chemical timer' column not found in data!")
    print("\nAvailable columns:")
    for col in sorted(df.columns):
        print(f"  - {col}")
    sys.exit(1)

print("\n" + "=" * 70)
print("CHEMICAL TIMER ANALYSIS")
print("=" * 70)

# Show chemical timer statistics
chem_timer = df["Chemical timer"]
print(f"\nChemical Timer Statistics:")
print(f"  Min value: {chem_timer.min()}")
print(f"  Max value: {chem_timer.max()}")
print(f"  Mean value: {chem_timer.mean():.2f}")
print(f"  Non-zero values: {(chem_timer > 0).sum()} / {len(chem_timer)}")
print(f"  Unique values: {chem_timer.nunique()}")

# Detect chemical timer increments
chem_increments = detect_chemical_timer_changes(df)
print(f"\n✓ Detected {len(chem_increments)} chemical timer increment events")

if len(chem_increments) == 0:
    print("\n⚠️  No chemical timer increments detected!")
    print("   This means the chemical timer value never increases in your dataset.")
    print("   Either:")
    print("   1. No chemical cleanings occurred during this period")
    print("   2. The chemical timer is not being logged properly")
    print("   3. The column name is different")

    # Show first and last 10 values
    print("\nFirst 10 Chemical Timer values:")
    print(df[["TimeStamp", "Chemical timer"]].head(10).to_string(index=False))
    print("\nLast 10 Chemical Timer values:")
    print(df[["TimeStamp", "Chemical timer"]].tail(10).to_string(index=False))

    # Check for any changes at all
    print("\nChecking for any differences in Chemical Timer:")
    timer_diff = df["Chemical timer"].diff()
    changes = timer_diff[timer_diff != 0]
    print(f"  Total value changes: {len(changes)}")
    if len(changes) > 0:
        print(f"  Positive changes: {(timer_diff > 0).sum()}")
        print(f"  Negative changes: {(timer_diff < 0).sum()}")
        print("\nFirst 20 changes:")
        for idx in changes.head(20).index:
            prev_val = df.loc[idx - 1, "Chemical timer"] if idx > 0 else 0
            curr_val = df.loc[idx, "Chemical timer"]
            timestamp = df.loc[idx, "TimeStamp"]
            print(
                f"    {timestamp}: {prev_val} → {curr_val} (change: {curr_val - prev_val:+.2f})"
            )

    sys.exit(0)

# Merge nearby events
merged_chem = merge_cleaning_events([], chem_increments, merge_window=100)
print(f"✓ After merging (within 100 points): {len(merged_chem)} events")

# Analyze what would become cycles
print("\n" + "-" * 70)
print("POTENTIAL CYCLES ANALYSIS (Chemical Timer Only)")
print("-" * 70)

boundaries = [0] + merged_chem + [len(df)]
boundaries = sorted(set(boundaries))

total_potential = len(boundaries) - 1
too_short_points = 0
too_short_duration = 0
valid_cycles = 0

min_duration_hours = 1.0

for i in range(len(boundaries) - 1):
    start_idx = boundaries[i]
    end_idx = boundaries[i + 1]

    cycle_df = df.iloc[start_idx:end_idx]

    # Check data points requirement
    if len(cycle_df) < 10:
        too_short_points += 1
        continue

    # Check duration requirement
    if "TimeStamp" in cycle_df.columns:
        duration_hours = (
            cycle_df["TimeStamp"].iloc[-1] - cycle_df["TimeStamp"].iloc[0]
        ).total_seconds() / 3600

        if duration_hours < min_duration_hours:
            too_short_duration += 1
            continue

    valid_cycles += 1

print(f"\nTotal potential cycle segments: {total_potential}")
print(f"  ❌ Filtered: Too few data points (<10): {too_short_points}")
print(f"  ❌ Filtered: Too short duration (<1h): {too_short_duration}")
print(f"  ✅ Valid cycles passing filters: {valid_cycles}")

# Show chemical timer increment timestamps
print("\n" + "-" * 70)
print("CHEMICAL TIMER INCREMENT TIMESTAMPS")
print("-" * 70)

chem_timestamps = df.loc[chem_increments, "TimeStamp"].tolist()
chem_values_before = [
    df.loc[idx - 1, "Chemical timer"] if idx > 0 else 0 for idx in chem_increments
]
chem_values_after = df.loc[chem_increments, "Chemical timer"].tolist()

for i, (ts, before, after) in enumerate(
    zip(chem_timestamps[:50], chem_values_before[:50], chem_values_after[:50]), 1
):
    print(
        f"  Event {i:3d}: {ts} - Timer: {before:.1f} → {after:.1f} (+{after - before:.1f})"
    )

if len(chem_timestamps) > 50:
    print(f"  ... ({len(chem_timestamps) - 50} more events)")

# Calculate gap statistics between chemical cleanings
if len(merged_chem) > 1:
    print("\n" + "-" * 70)
    print("TIME GAPS BETWEEN CHEMICAL CLEANINGS")
    print("-" * 70)

    merged_timestamps = df.loc[merged_chem, "TimeStamp"].tolist()
    gaps = []

    for i in range(len(merged_timestamps) - 1):
        gap_hours = (
            merged_timestamps[i + 1] - merged_timestamps[i]
        ).total_seconds() / 3600
        gaps.append(gap_hours)

    gaps_array = np.array(gaps)
    print(f"\nGap statistics (hours between chemical cleanings):")
    print(f"  Min gap: {gaps_array.min():.2f} hours")
    print(f"  Max gap: {gaps_array.max():.2f} hours")
    print(f"  Mean gap: {gaps_array.mean():.2f} hours")
    print(f"  Median gap: {np.median(gaps_array):.2f} hours")
    print(f"  Std dev: {gaps_array.std():.2f} hours")

    # Show distribution
    print(f"\nGap distribution:")
    print(f"  < 0.1 hours: {np.sum(gaps_array < 0.1)}")
    print(f"  0.1-1 hours: {np.sum((gaps_array >= 0.1) & (gaps_array < 1))}")
    print(f"  1-10 hours: {np.sum((gaps_array >= 1) & (gaps_array < 10))}")
    print(f"  10-24 hours: {np.sum((gaps_array >= 10) & (gaps_array < 24))}")
    print(f"  > 24 hours: {np.sum(gaps_array >= 24)}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
