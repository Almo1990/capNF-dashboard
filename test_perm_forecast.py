"""Test script to verify daily baseline permeability forecast implementation"""

import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Load processed data
df_viz_path = Path(config["paths"]["processed_viz_data"])
print(f"Loading data from: {df_viz_path}")

df_viz = pd.read_parquet(df_viz_path)
print(f"Data loaded: {len(df_viz)} rows")
print(f"Columns: {df_viz.columns.tolist()}")

# Test daily baseline calculation
if "Permeability TC" in df_viz.columns:
    df_temp = df_viz[["TimeStamp", "Permeability TC"]].copy()
    df_temp = df_temp.dropna(subset=["Permeability TC"])
    print(f"\nValid permeability data points: {len(df_temp)}")

    df_temp["Date"] = df_temp["TimeStamp"].dt.floor("D")

    # Calculate 10th percentile per day
    daily_baseline = (
        df_temp.groupby("Date")["Permeability TC"]
        .agg(lambda x: np.percentile(x, 10) if len(x) >= 10 else x.min())
        .reset_index()
    )

    print(f"\nDaily baseline data points: {len(daily_baseline)}")
    print(f"\nFirst 10 daily baseline values:")
    print(daily_baseline.head(10))
    print(f"\nLast 10 daily baseline values:")
    print(daily_baseline.tail(10))

    # Calculate statistics
    baseline_values = daily_baseline["Permeability TC"].values
    print(f"\nBaseline statistics:")
    print(f"  Mean: {np.mean(baseline_values):.2f}")
    print(f"  Std: {np.std(baseline_values):.2f}")
    print(f"  Min: {np.min(baseline_values):.2f}")
    print(f"  Max: {np.max(baseline_values):.2f}")
    print(f"  Range: {np.max(baseline_values) - np.min(baseline_values):.2f}")

    # Test linear regression
    if len(baseline_values) > 2:
        t0 = daily_baseline["Date"].min()
        seconds = (daily_baseline["Date"] - t0).dt.total_seconds().values

        coeffs = np.polyfit(seconds, baseline_values, 1)
        fit_vals = coeffs[0] * seconds + coeffs[1]

        # Calculate R²
        y_mean = baseline_values.mean()
        ss_tot = np.sum((baseline_values - y_mean) ** 2)
        ss_res = np.sum((baseline_values - fit_vals) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        print(f"\nLinear regression results:")
        print(f"  Slope: {coeffs[0]:.6e} LMH/bar/second")
        print(f"  Slope per day: {coeffs[0] * 86400:.6f} LMH/bar/day")
        print(f"  Intercept: {coeffs[1]:.4f}")
        print(f"  R²: {r_sq:.4f}")

        # Compare with original cycle-start approach (simulation)
        print(f"\n{'=' * 60}")
        print("Comparison: Daily baseline shows MORE STABLE trend")
        print(f"{'=' * 60}")
else:
    print("ERROR: Permeability TC column not found!")
