"""
[4] FEATURE ENGINEERING MODULE
Derive calculated features and rolling statistics
"""

import pandas as pd
import numpy as np
import logging
import yaml
from typing import List

from .utils.data_io import load_parquet, save_parquet

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_derivatives(
    df: pd.DataFrame, columns: List[str], window: str = "5min"
) -> pd.DataFrame:
    """
    Calculate time derivatives (rate of change) for specified columns

    Args:
        df: Input DataFrame with TimeStamp
        columns: List of columns to calculate derivatives for
        window: Time window for derivative calculation

    Returns:
        DataFrame with derivative columns added
    """
    logger.info(f"Calculating derivatives for {len(columns)} columns...")

    df_deriv = df.copy()

    if "TimeStamp" not in df.columns:
        logger.warning("No TimeStamp column, skipping derivatives")
        return df_deriv

    # Set TimeStamp as index for time-based operations
    df_deriv = df_deriv.set_index("TimeStamp")

    for col in columns:
        if col in df_deriv.columns and pd.api.types.is_numeric_dtype(df_deriv[col]):
            # Calculate derivative (change per hour)
            deriv_col = f"{col}_derivative"
            df_deriv[deriv_col] = df_deriv[col].diff() / (
                df_deriv.index.to_series().diff().dt.total_seconds() / 3600
            )

    df_deriv = df_deriv.reset_index()

    logger.info(f"✓ Added {len(columns)} derivative columns")

    return df_deriv


def calculate_rolling_statistics(
    df: pd.DataFrame, columns: List[str], windows: List[str]
) -> pd.DataFrame:
    """
    Calculate rolling statistics (mean, std) over multiple time windows

    Args:
        df: Input DataFrame with TimeStamp
        columns: List of columns to calculate stats for
        windows: List of time windows (e.g., ['6h', '12h', '24h'])

    Returns:
        DataFrame with rolling statistic columns added
    """
    logger.info(
        f"Calculating rolling stats for {len(columns)} columns over {len(windows)} windows..."
    )

    df_stats = df.copy()

    if "TimeStamp" not in df.columns:
        logger.warning("No TimeStamp column, skipping rolling stats")
        return df_stats

    # Set TimeStamp as index
    df_stats = df_stats.set_index("TimeStamp")

    for col in columns:
        if col not in df_stats.columns or not pd.api.types.is_numeric_dtype(
            df_stats[col]
        ):
            continue

        for window in windows:
            # Rolling mean
            mean_col = f"{col}_mean_{window}"
            df_stats[mean_col] = (
                df_stats[col].rolling(window=window, min_periods=1).mean()
            )

            # Rolling std
            std_col = f"{col}_std_{window}"
            df_stats[std_col] = (
                df_stats[col].rolling(window=window, min_periods=1).std()
            )

    df_stats = df_stats.reset_index()

    logger.info(f"✓ Added {len(columns) * len(windows) * 2} rolling statistic columns")

    return df_stats


def calculate_temperature_normalized_metrics(
    df: pd.DataFrame, reference_temp: float = 20.0
) -> pd.DataFrame:
    """
    Calculate temperature-normalized versions of key metrics

    Args:
        df: Input DataFrame
        reference_temp: Reference temperature for normalization (°C)

    Returns:
        DataFrame with normalized columns added
    """
    logger.info(
        f"Calculating temperature-normalized metrics (ref={reference_temp}°C)..."
    )

    df_norm = df.copy()

    temp_col = "01-TIT-01"
    if temp_col not in df.columns:
        logger.warning("Temperature column not found, skipping normalization")
        return df_norm

    # Temperature correction factor (approximate)
    # Viscosity changes ~2-3% per °C
    temp_factor = 1.025 ** (df[temp_col] - reference_temp)

    # Normalize permeability
    if "Permeability TC" in df.columns:
        df_norm["Permeability_TC_normalized"] = df["Permeability TC"] / temp_factor

    # Normalize flux
    if "Flux" in df.columns:
        df_norm["Flux_normalized"] = df["Flux"] / temp_factor

    logger.info("✓ Added temperature-normalized metrics")

    return df_norm


def calculate_efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived efficiency metrics

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with efficiency metrics added
    """
    logger.info("Calculating efficiency metrics...")

    df_eff = df.copy()

    # Permeability/TMP ratio (hydraulic efficiency)
    if "Permeability TC" in df.columns and "TMP" in df.columns:
        df_eff["hydraulic_efficiency"] = df["Permeability TC"] / (
            df["TMP"] + 0.1
        )  # Avoid div by zero

    # Crossflow velocity efficiency (correlation with fouling)
    if "Vcrossflow" in df.columns and "TMP" in df.columns:
        df_eff["crossflow_efficiency"] = df["Vcrossflow"] / (df["TMP"] + 0.1)

    # Recovery efficiency
    if "Recovery" in df.columns and "Specific_power" in df.columns:
        df_eff["recovery_per_energy"] = df["Recovery"] / (df["Specific_power"] + 0.001)

    logger.info("✓ Added efficiency metrics")

    return df_eff


def run_feature_engineering(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Main feature engineering pipeline

    Args:
        config_path: Path to configuration file

    Returns:
        DataFrame with engineered features
    """
    logger.info("=" * 60)
    logger.info("🔧 FEATURE ENGINEERING STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    input_path = config["paths"]["processed_data"]
    output_path = config["paths"]["features_data"]
    feat_config = config["feature_engineering"]

    # Load processed data
    logger.info(f"Loading data from: {input_path}")
    df = load_parquet(input_path)

    logger.info(f"Input data: {df.shape}")
    initial_cols = df.shape[1]

    # Key columns to analyze
    key_columns = ["TMP", "Permeability TC", "Flux", "Recovery"]
    key_columns = [col for col in key_columns if col in df.columns]

    # Calculate derivatives
    if feat_config["calculate_derivatives"]:
        deriv_window = feat_config["derivative_window"]
        df = calculate_derivatives(df, key_columns, window=deriv_window)

    # Calculate rolling statistics
    rolling_windows = feat_config["rolling_windows"]
    df = calculate_rolling_statistics(df, key_columns, rolling_windows)

    # Temperature normalization
    ref_temp = feat_config["reference_temperature"]
    df = calculate_temperature_normalized_metrics(df, reference_temp=ref_temp)

    # Efficiency metrics
    df = calculate_efficiency_metrics(df)

    final_cols = df.shape[1]
    new_features = final_cols - initial_cols

    logger.info(f"\n✓ Feature engineering complete")
    logger.info(f"  Original columns: {initial_cols}")
    logger.info(f"  New features: {new_features}")
    logger.info(f"  Total columns: {final_cols}")

    # Save features
    save_parquet(df, output_path)

    logger.info("=" * 60)
    logger.info("✓ FEATURE ENGINEERING COMPLETE")
    logger.info("=" * 60)

    return df


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run feature engineering
    df = run_feature_engineering()
    print(f"\n✓ Engineered dataset: {df.shape}")
    print(f"✓ Sample columns: {list(df.columns[-10:])}")
