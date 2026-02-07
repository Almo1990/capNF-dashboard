import pandas as pd
import numpy as np
import json

print("=" * 80)
print("HOW THE INITIAL STABLE PERFORMANCE BASELINE IS DEFINED")
print("=" * 80)

# Load data
df = pd.read_parquet("outputs/05_fouling_metrics.parquet")
df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])

# Load metadata
with open("outputs/fouling_metadata.json", "r") as f:
    metadata = json.load(f)

print("\n" + "=" * 80)
print("ALGORITHM STEPS:")
print("=" * 80)

print("\n1️⃣  DEFINE BASELINE PERIOD")
print("-" * 80)
baseline_start = pd.to_datetime("2026-01-28 15:00:00")
baseline_end = baseline_start + pd.Timedelta(days=7)
print(f"   Start: {baseline_start}")
print(f"   End:   {baseline_end}")
print(f"   Duration: 7 days")

df_baseline = df[
    (df["TimeStamp"] >= baseline_start) & (df["TimeStamp"] < baseline_end)
].copy()
print(f"   Data points in baseline period: {len(df_baseline):,}")

print("\n2️⃣  SEARCH FOR MOST STABLE PERIOD (LOWEST SLOPE)")
print("-" * 80)
print("   Method: Sliding window analysis")

# Calculate actual sampling rate
time_diff_median = df_baseline["TimeStamp"].diff().median()
points_per_hour = int(3600 / time_diff_median.total_seconds())
points_per_24h = points_per_hour * 24
points_per_4h = points_per_hour * 4

print(f"   - Sampling rate: {points_per_hour} points/hour")
print(f"   - Window size: {points_per_24h:,} data points (TRUE 24 hours)")
print(f"   - Step size: {points_per_4h:,} points (4 hours)")
print("   - For each window: Calculate linear regression slope")
print("   - Select window with LOWEST absolute slope (most stable)")

# Recreate the algorithm to show the selected window
from src.utils.time_utils import convert_to_seconds_since_start

best_slope = float("inf")
best_window_data = None
best_window_index = None

for i in range(0, len(df_baseline) - points_per_24h, points_per_4h):
    window_data = df_baseline.iloc[i : i + points_per_24h]
    if len(window_data) < points_per_24h * 0.9:
        continue

    time_window = convert_to_seconds_since_start(window_data["TimeStamp"])
    perm_window = window_data["Permeability TC"].values

    if np.isnan(perm_window).sum() > len(perm_window) * 0.1:
        continue

    try:
        coeffs = np.polyfit(time_window, perm_window, 1)
        slope = abs(coeffs[0])

        if slope < best_slope:
            best_slope = slope
            best_window_data = window_data
            best_window_index = i
    except:
        continue

print(f"\n   ✓ Found most stable period:")
if best_window_data is not None:
    print(f"      - Start time: {best_window_data['TimeStamp'].iloc[0]}")
    print(f"      - End time:   {best_window_data['TimeStamp'].iloc[-1]}")
    print(
        f"      - Duration:   {(best_window_data['TimeStamp'].iloc[-1] - best_window_data['TimeStamp'].iloc[0]).total_seconds() / 3600:.1f} hours"
    )
    print(f"      - Data points: {len(best_window_data)}")
    print(f"      - Slope: {best_slope:.6e} LMH/bar/s")
    print(f"      - Slope: {best_slope * 3600:.6f} LMH/bar/hour")
    print(f"      - Slope: {best_slope * 86400:.4f} LMH/bar/day")

print("\n3️⃣  CALCULATE BASELINE VALUE")
print("-" * 80)
if best_window_data is not None and "Permeability TC_SMA" in best_window_data.columns:
    baseline_perm = best_window_data["Permeability TC_SMA"].mean()
    print(f"   Using: Permeability TC_SMA (smoothed moving average)")
    print(f"   Min:  {best_window_data['Permeability TC_SMA'].min():.2f} LMH/bar")
    print(f"   Max:  {best_window_data['Permeability TC_SMA'].max():.2f} LMH/bar")
    print(f"   Mean (BASELINE): {baseline_perm:.2f} LMH/bar")
    print(f"   Std Dev: {best_window_data['Permeability TC_SMA'].std():.4f} LMH/bar")

print("\n4️⃣  GET CURRENT VALUE")
print("-" * 80)
current_perm = df["Permeability TC_SMA"].iloc[-1]
current_time = df["TimeStamp"].iloc[-1]
print(f"   Timestamp: {current_time}")
print(f"   Using: Permeability TC_SMA (last recorded value)")
print(f"   Current: {current_perm:.2f} LMH/bar")

print("\n5️⃣  CALCULATE DECLINE")
print("-" * 80)
decline = ((baseline_perm - current_perm) / baseline_perm) * 100
print(f"   Formula: ((Baseline - Current) / Baseline) × 100")
print(
    f"   Decline: (({baseline_perm:.2f} - {current_perm:.2f}) / {baseline_perm:.2f}) × 100"
)
print(f"   Decline: {decline:.2f}%")

print("\n" + "=" * 80)
print("WHY THIS METHOD IS BETTER:")
print("=" * 80)
print("✅ Finds the TRUE stable operating condition (not affected by startup)")
print("✅ Uses smoothed data (SMA) to avoid noise")
print("✅ Compares to initial performance, not transient drops")
print("✅ Lowest slope = most consistent operation = best baseline")
print("✅ Represents normal operating conditions after stabilization")
print("=" * 80)
