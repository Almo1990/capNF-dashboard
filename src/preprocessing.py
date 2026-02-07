"""
[3] PREPROCESSING MODULE
Data transformations: downsampling, SMA, gap filling
"""

import pandas as pd
import numpy as np
import logging
import yaml
from typing import Tuple

from .utils.data_io import load_parquet, save_parquet

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def downsample_data(df: pd.DataFrame, max_points: int = 10000) -> pd.DataFrame:
    """
    Downsample data for faster rendering while preserving key features.
    Uses LTTB-like approach: keeps first, last, and evenly spaced points.

    Args:
        df: Input DataFrame
        max_points: Maximum number of points to keep

    Returns:
        Downsampled DataFrame
    """
    if len(df) <= max_points:
        return df.copy()

    logger.info(f"Downsampling from {len(df)} to ~{max_points} points...")

    # Calculate step size
    step = len(df) // max_points

    # Take every nth point
    downsampled = df.iloc[::step].copy()

    # Always include the last point
    if df.index[-1] not in downsampled.index:
        downsampled = pd.concat([downsampled, df.iloc[[-1]]])

    downsampled = downsampled.reset_index(drop=True)

    logger.info(f"✓ Downsampled to {len(downsampled)} points")

    return downsampled


def calculate_sma(df: pd.DataFrame, window: int, columns: list = None) -> pd.DataFrame:
    """
    Calculate Simple Moving Average for specified columns

    Args:
        df: Input DataFrame
        window: Window size in number of points
        columns: List of columns to calculate SMA for (None = all numeric)

    Returns:
        DataFrame with SMA columns added
    """
    if columns is None:
        # All numeric columns except TimeStamp
        columns = [
            col
            for col in df.columns
            if col != "TimeStamp" and pd.api.types.is_numeric_dtype(df[col])
        ]

    logger.info(
        f"Calculating SMA (window={window} points) for {len(columns)} columns..."
    )

    df_sma = df.copy()

    for col in columns:
        if col in df.columns:
            sma_col = f"{col}_SMA"
            df_sma[sma_col] = df[col].rolling(window=window, min_periods=1).mean()

    logger.info(f"✓ Added {len(columns)} SMA columns")

    return df_sma


def fill_gaps(
    df: pd.DataFrame, method: str = "forward_fill", max_fill_minutes: int = 5
) -> pd.DataFrame:
    """
    Fill missing values in time series

    Args:
        df: Input DataFrame with TimeStamp column
        method: Filling method ('forward_fill', 'interpolate', 'drop')
        max_fill_minutes: Only fill gaps shorter than this

    Returns:
        DataFrame with gaps filled
    """
    if "TimeStamp" not in df.columns:
        logger.warning("No TimeStamp column, skipping gap filling")
        return df

    df_filled = df.copy()

    # Identify gaps
    time_diffs = df["TimeStamp"].diff()
    gaps = time_diffs > pd.Timedelta(minutes=max_fill_minutes)
    num_gaps = gaps.sum()

    if num_gaps == 0:
        logger.info("✓ No gaps to fill")
        return df_filled

    logger.info(f"Filling {num_gaps} small gaps (≤{max_fill_minutes} min)...")

    if method == "forward_fill":
        # Forward fill only for short gaps
        df_filled = df_filled.fillna(method="ffill", limit=10)
    elif method == "interpolate":
        # Linear interpolation
        numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
        df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(
            method="linear", limit=10
        )
    elif method == "drop":
        # Drop rows with missing values
        df_filled = df_filled.dropna()

    logger.info(f"✓ Gap filling complete ({method})")

    return df_filled


def run_preprocessing(
    config_path: str = "config.yaml",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing pipeline: downsample, SMA, gap filling

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (full processed DataFrame, viz-ready downsampled DataFrame)
    """
    logger.info("=" * 60)
    logger.info("⚙️ PREPROCESSING STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    input_path = config["paths"]["validated_data"]
    output_full = config["paths"]["processed_data"]
    output_viz = config["paths"]["processed_viz_data"]
    preproc_config = config["preprocessing"]

    # Load validated data
    logger.info(f"Loading data from: {input_path}")
    df = load_parquet(input_path)

    logger.info(f"Input data: {df.shape}")

    # Filter data from baseline_start if configured
    if "baseline_start" in preproc_config and preproc_config["baseline_start"]:
        baseline_start = pd.to_datetime(preproc_config["baseline_start"])
        df_original_len = len(df)
        df = df[df["TimeStamp"] >= baseline_start]
        logger.info(
            f"Filtered data from {preproc_config['baseline_start']}: removed {df_original_len - len(df)} rows"
        )
        logger.info(f"Filtered data shape: {df.shape}")

    # Fill gaps
    gap_config = preproc_config["gap_filling"]
    df_filled = fill_gaps(
        df, method=gap_config["method"], max_fill_minutes=gap_config["max_fill_minutes"]
    )

    # Calculate SMA on full dataset
    sma_window = preproc_config["sma_windows"]["individual_plots"]
    df_processed = calculate_sma(df_filled, window=sma_window)

    logger.info(f"Processed data shape: {df_processed.shape}")

    # Save full processed data
    save_parquet(df_processed, output_full)

    # Create downsampled visualization version
    logger.info("\nCreating visualization-ready data...")
    downsample_limit = preproc_config["downsample_limits"]["individual_plots"]
    df_viz = downsample_data(df_processed, max_points=downsample_limit)

    # Recalculate SMA on downsampled data for consistency
    sma_window_viz = preproc_config["sma_windows"]["individual_plots"]
    df_viz = calculate_sma(df_viz, window=sma_window_viz)

    logger.info(f"Visualization data shape: {df_viz.shape}")

    # Save visualization data
    save_parquet(df_viz, output_viz)

    logger.info("=" * 60)
    logger.info("✓ PREPROCESSING COMPLETE")
    logger.info("=" * 60)

    return df_processed, df_viz


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run preprocessing
    df_full, df_viz = run_preprocessing()
    print(f"\n✓ Full dataset: {len(df_full)} rows, {df_full.shape[1]} columns")
    print(f"✓ Viz dataset: {len(df_viz)} rows, {df_viz.shape[1]} columns")
