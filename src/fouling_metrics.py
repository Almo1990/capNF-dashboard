"""
[5] FOULING METRICS MODULE
Calculate TMP slopes and fouling indicators
"""

import pandas as pd
import numpy as np
import logging
import yaml
from typing import Dict, Tuple

from .utils.data_io import load_parquet, save_parquet
from .utils.time_utils import convert_to_seconds_since_start

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_tmp_slope_global(df: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Calculate global TMP slope using linear regression

    Args:
        df: DataFrame with TimeStamp and TMP columns

    Returns:
        Tuple of (slope_per_hour, slope_per_day, r_squared)
    """
    if "TMP" not in df.columns or "TimeStamp" not in df.columns:
        logger.warning("Missing TMP or TimeStamp column")
        return 0.0, 0.0, 0.0

    # Remove NaN values
    df_tmp = df[["TimeStamp", "TMP"]].dropna()

    if len(df_tmp) < 2:
        logger.warning("Insufficient data for TMP slope calculation")
        return 0.0, 0.0, 0.0

    # Convert timestamp to numeric (seconds since start)
    time_numeric = convert_to_seconds_since_start(df_tmp["TimeStamp"])

    # Perform linear regression
    coeffs = np.polyfit(time_numeric, df_tmp["TMP"], 1)
    slope = coeffs[0]  # slope in units per second
    intercept = coeffs[1]

    # Calculate fitted values
    fit_values = slope * time_numeric + intercept

    # Convert slope to more readable units
    slope_per_hour = slope * 3600
    slope_per_day = slope * 86400

    # Calculate R-squared
    y_mean = df_tmp["TMP"].mean()
    ss_tot = np.sum((df_tmp["TMP"] - y_mean) ** 2)
    ss_res = np.sum((df_tmp["TMP"] - fit_values) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    logger.info(f"Global TMP slope:")
    logger.info(f"  {slope_per_hour:.6f} bar/hour")
    logger.info(f"  {slope_per_day:.4f} bar/day")
    logger.info(f"  R² = {r_squared:.4f}")

    return slope_per_hour, slope_per_day, r_squared


def calculate_rolling_tmp_slopes(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """
    Calculate rolling TMP slopes over multiple time windows

    Args:
        df: DataFrame with TimeStamp and TMP
        windows: List of time windows (e.g., ['6h', '12h', '24h'])

    Returns:
        DataFrame with rolling slope columns added
    """
    logger.info(f"Calculating rolling TMP slopes over {len(windows)} windows...")

    df_slopes = df.copy()

    if "TMP" not in df.columns or "TimeStamp" not in df.columns:
        logger.warning("Missing required columns")
        return df_slopes

    # Set TimeStamp as index
    df_slopes = df_slopes.set_index("TimeStamp")

    for window in windows:
        slope_col = f"TMP_slope_{window}"

        # Calculate rolling slope (simple approximation: change / time)
        tmp_change = df_slopes["TMP"].diff()
        time_change = (
            pd.Series(df_slopes.index).diff().dt.total_seconds() / 3600
        )  # hours
        time_change.index = df_slopes.index

        instantaneous_slope = tmp_change / time_change

        # Smooth with rolling mean
        df_slopes[slope_col] = instantaneous_slope.rolling(window=window).mean()

    df_slopes = df_slopes.reset_index()

    logger.info(f"✓ Added {len(windows)} rolling slope columns")

    return df_slopes


def classify_fouling_rate(slope: float, thresholds: dict) -> str:
    """
    Classify fouling rate based on TMP slope

    Args:
        slope: TMP slope in bar/hour
        thresholds: Dictionary with 'low', 'medium', 'high' thresholds

    Returns:
        Classification string: 'low', 'medium', 'high', 'critical'
    """
    if abs(slope) < thresholds["low"]:
        return "low"
    elif abs(slope) < thresholds["medium"]:
        return "medium"
    elif abs(slope) < thresholds["high"]:
        return "high"
    else:
        return "critical"


def calculate_permeability_decline(
    df: pd.DataFrame, window: str = "24h", baseline_start: str = "2026-01-28 15:00:00"
) -> pd.DataFrame:
    """
    Calculate permeability decline rate based on first-week baseline

    Args:
        df: DataFrame with Permeability TC and Permeability TC_SMA
        window: Time window for baseline calculation (default 7 days)
        baseline_start: Start time for baseline period (Jan 28, 2026 15:00)

    Returns:
        DataFrame with decline columns added
    """
    logger.info(f"Calculating permeability decline from baseline period...")

    df_decline = df.copy()

    if "Permeability TC" not in df.columns:
        logger.warning("Permeability TC column not found")
        return df_decline

    # Ensure TimeStamp is datetime
    if "TimeStamp" in df.columns:
        df_decline["TimeStamp"] = pd.to_datetime(df_decline["TimeStamp"])
    else:
        logger.warning("TimeStamp column not found")
        return df_decline

    # Define baseline period: first week starting from Jan 28, 2026 15:00
    baseline_start_dt = pd.to_datetime(baseline_start)
    baseline_end_dt = baseline_start_dt + pd.Timedelta(days=7)

    logger.info(f"Baseline period: {baseline_start_dt} to {baseline_end_dt}")

    # Filter data for baseline period
    df_baseline = df_decline[
        (df_decline["TimeStamp"] >= baseline_start_dt)
        & (df_decline["TimeStamp"] < baseline_end_dt)
    ].copy()

    if len(df_baseline) < 10:
        logger.warning(f"Insufficient baseline data ({len(df_baseline)} points)")
        # Fallback to old method
        df_decline = df_decline.set_index("TimeStamp")
        perm_start = (
            df_decline["Permeability TC"]
            .rolling(window=window)
            .apply(lambda x: x.iloc[0] if len(x) > 0 else np.nan, raw=False)
        )
        perm_current = df_decline["Permeability TC"]
        df_decline[f"permeability_decline_{window}"] = (
            (perm_start - perm_current) / perm_start * 100
        )
        df_decline = df_decline.reset_index()
        return df_decline

    # Find most stable period (lowest slope) within baseline week
    # Calculate rolling 24h slopes within baseline period
    from .utils.time_utils import convert_to_seconds_since_start

    # Calculate actual sampling rate (points per hour)
    time_diff_median = df_baseline["TimeStamp"].diff().median()
    points_per_hour = int(3600 / time_diff_median.total_seconds())
    points_per_24h = points_per_hour * 24
    points_per_4h = points_per_hour * 4  # Step size: check every 4 hours

    logger.info(
        f"Sampling rate: ~{points_per_hour} points/hour, {points_per_24h} points/24h"
    )

    best_slope = float("inf")
    best_window_data = None

    # Slide through baseline period with 24h windows, stepping by 4 hours
    for i in range(
        0, len(df_baseline) - points_per_24h, points_per_4h
    ):  # Check every 4 hours
        window_data = df_baseline.iloc[i : i + points_per_24h]  # True 24h of data
        if (
            len(window_data) < points_per_24h * 0.9
        ):  # Need at least 90% of expected points
            continue

        time_window = convert_to_seconds_since_start(window_data["TimeStamp"])
        perm_window = window_data["Permeability TC"].values

        # Skip if too many NaN values
        if np.isnan(perm_window).sum() > len(perm_window) * 0.1:
            continue

        # Calculate slope
        try:
            coeffs = np.polyfit(time_window, perm_window, 1)
            slope = abs(coeffs[0])  # Absolute value of slope

            if slope < best_slope:
                best_slope = slope
                best_window_data = window_data
        except Exception:
            continue

    # Use the most stable period for baseline calculation
    if best_window_data is not None and len(best_window_data) > 0:
        # Use SMA if available, otherwise raw values
        if "Permeability TC_SMA" in best_window_data.columns:
            baseline_perm = best_window_data["Permeability TC_SMA"].mean()
            logger.info(
                f"Baseline permeability (from SMA): {baseline_perm:.2f} LMH/bar"
            )
        else:
            baseline_perm = best_window_data["Permeability TC"].mean()
            logger.info(
                f"Baseline permeability (from raw): {baseline_perm:.2f} LMH/bar"
            )

        logger.info(f"Most stable period slope: {best_slope:.6e} LMH/bar/s")
    else:
        # Fallback: use mean of entire baseline period
        if "Permeability TC_SMA" in df_baseline.columns:
            baseline_perm = df_baseline["Permeability TC_SMA"].mean()
        else:
            baseline_perm = df_baseline["Permeability TC"].mean()
        logger.info(
            f"Baseline permeability (week average): {baseline_perm:.2f} LMH/bar"
        )

    # Get current (last) permeability from SMA if available
    if "Permeability TC_SMA" in df_decline.columns:
        current_perm = df_decline["Permeability TC_SMA"].iloc[-1]
        logger.info(f"Current permeability (from SMA): {current_perm:.2f} LMH/bar")
    else:
        current_perm = df_decline["Permeability TC"].iloc[-1]
        logger.info(f"Current permeability (from raw): {current_perm:.2f} LMH/bar")

    # Calculate decline percentage from baseline
    if baseline_perm > 0:
        decline_percent = ((baseline_perm - current_perm) / baseline_perm) * 100
    else:
        decline_percent = 0

    logger.info(f"Permeability decline from baseline: {decline_percent:.2f}%")

    # Add baseline decline as a constant column
    df_decline["permeability_decline_baseline"] = decline_percent
    df_decline["permeability_baseline"] = baseline_perm

    logger.info("✓ Added permeability decline column (baseline method)")

    return df_decline


def run_fouling_metrics(config_path: str = "config.yaml") -> Tuple[pd.DataFrame, dict]:
    """
    Main fouling metrics pipeline

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (DataFrame with fouling metrics, metadata dict)
    """
    logger.info("=" * 60)
    logger.info("📉 FOULING METRICS STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    input_path = config["paths"]["features_data"]
    output_path = config["paths"]["fouling_metrics_data"]
    fouling_config = config["fouling"]

    # Load features data
    logger.info(f"Loading data from: {input_path}")
    df = load_parquet(input_path)

    logger.info(f"Input data: {df.shape}")

    # Calculate global TMP slope
    slope_per_hour, slope_per_day, r_squared = calculate_tmp_slope_global(df)

    # Classify fouling rate
    thresholds = fouling_config["tmp_slope"]["thresholds"]
    fouling_classification = classify_fouling_rate(slope_per_hour, thresholds)

    logger.info(f"Fouling classification: {fouling_classification.upper()}")

    # Calculate rolling TMP slopes
    rolling_windows = fouling_config["tmp_slope"]["rolling_windows"]
    df = calculate_rolling_tmp_slopes(df, rolling_windows)

    # Calculate permeability decline
    perm_config = fouling_config["permeability_decline"]
    baseline_start = perm_config.get("baseline_start", "2026-01-28 15:00:00")
    df = calculate_permeability_decline(
        df, window=perm_config["window"], baseline_start=baseline_start
    )

    # Create metadata
    metadata = {
        "global_tmp_slope_per_hour": slope_per_hour,
        "global_tmp_slope_per_day": slope_per_day,
        "tmp_slope_r_squared": r_squared,
        "fouling_classification": fouling_classification,
        "analysis_timestamp": pd.Timestamp.now().isoformat(),
        "permeability_baseline": float(df["permeability_baseline"].iloc[-1])
        if "permeability_baseline" in df.columns
        else None,
        "permeability_decline_from_baseline": float(
            df["permeability_decline_baseline"].iloc[-1]
        )
        if "permeability_decline_baseline" in df.columns
        else None,
    }

    # Save fouling metrics
    save_parquet(df, output_path)

    # Save metadata
    import json

    metadata_path = "outputs/fouling_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✓ Saved metadata: {metadata_path}")

    logger.info("=" * 60)
    logger.info("✓ FOULING METRICS COMPLETE")
    logger.info("=" * 60)

    return df, metadata


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run fouling metrics
    df, metadata = run_fouling_metrics()
    print(f"\n✓ Fouling metrics calculated")
    print(f"✓ Global TMP slope: {metadata['global_tmp_slope_per_hour']:.6f} bar/hour")
    print(f"✓ Fouling level: {metadata['fouling_classification'].upper()}")
