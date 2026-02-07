"""
Analyze TMP drops to understand cycle detection
"""

import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "src")

from src.utils.data_io import load_parquet
from src.cleaning_analysis import detect_tmp_drops, merge_cleaning_events

# Load data
print("Loading data...")
df = load_parquet("outputs/05_fouling_metrics.parquet")
print(f"Data shape: {df.shape}")
print(f"Date range: {df['TimeStamp'].min()} to {df['TimeStamp'].max()}")

# Detect TMP drops with current settings
print("\n" + "=" * 70)
print("TMP DROP ANALYSIS")
print("=" * 70)

tmp_drops = detect_tmp_drops(df, threshold_percent=20, window_minutes=30)
print(f"\n✓ Detected {len(tmp_drops)} TMP drop events (>20% in 30min)")

# Merge nearby events
merged_drops = merge_cleaning_events(tmp_drops, [], merge_window=100)
print(f"✓ After merging (within 100 points): {len(merged_drops)} events")

# Analyze what would become cycles
print("\n" + "-" * 70)
print("POTENTIAL CYCLES ANALYSIS")
print("-" * 70)

boundaries = [0] + merged_drops + [len(df)]
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

# Analyze TMP drop distribution over time
print("\n" + "-" * 70)
print("TMP DROP DISTRIBUTION OVER TIME")
print("-" * 70)

drop_timestamps = df.loc[tmp_drops, "TimeStamp"].tolist()

for i, ts in enumerate(drop_timestamps[:20], 1):  # Show first 20
    print(f"  Drop {i:3d}: {ts}")

if len(drop_timestamps) > 20:
    print(f"  ... ({len(drop_timestamps) - 20} more drops)")

# Calculate gap statistics between merged drops
print("\n" + "-" * 70)
print("TIME GAPS BETWEEN CLEANING EVENTS")
print("-" * 70)

merged_timestamps = df.loc[merged_drops, "TimeStamp"].tolist()
gaps = []

for i in range(len(merged_timestamps) - 1):
    gap_hours = (merged_timestamps[i + 1] - merged_timestamps[i]).total_seconds() / 3600
    gaps.append(gap_hours)

if gaps:
    gaps_array = np.array(gaps)
    print(f"\nGap statistics (hours between cleanings):")
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
