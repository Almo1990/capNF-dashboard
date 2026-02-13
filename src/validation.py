"""
[2] VALIDATION MODULE
Data quality checks and bandwidth filtering
"""

import pandas as pd
import logging
import yaml
from typing import Dict, List

from .utils.data_io import load_parquet, save_parquet
from .utils.time_utils import detect_gaps
from .utils.models import ValidationReport

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def check_required_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """
    Check for presence of required columns

    Args:
        df: Input DataFrame
        required_cols: List of required column names

    Returns:
        List of missing columns
    """
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        logger.warning(f"Missing required columns: {missing}")
    else:
        logger.info("✓ All required columns present")

    return missing


def analyze_missing_values(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze missing values in each column

    Args:
        df: Input DataFrame

    Returns:
        Dictionary mapping column name to % missing
    """
    missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()

    # Report columns with significant missing values
    significant = {k: v for k, v in missing_pct.items() if v > 0}

    if significant:
        logger.info("Missing values detected:")
        for col, pct in significant.items():
            logger.info(f"  - {col}: {pct:.2f}%")
    else:
        logger.info("✓ No missing values detected")

    return missing_pct


def apply_bandwidth_filters(
    df: pd.DataFrame, bandwidth_config: Dict[str, dict]
) -> tuple:
    """
    Remove outliers based on bandwidth thresholds

    Args:
        df: Input DataFrame
        bandwidth_config: Dictionary of parameter: {min, max} thresholds

    Returns:
        Tuple of (filtered DataFrame, outliers_removed dict)
    """
    logger.info("Applying bandwidth filters:")

    original_count = len(df)
    outliers_removed = {}

    # Select columns to keep
    columns_with_thresholds = list(bandwidth_config.keys())
    columns_to_keep = ["TimeStamp"] + [
        col for col in columns_with_thresholds if col in df.columns
    ]

    # Keep only relevant columns
    df_filtered = df[columns_to_keep].copy()

    # Apply filters
    for param, bounds in bandwidth_config.items():
        if param not in df_filtered.columns:
            continue

        initial_count = len(df_filtered)

        # Filter rows outside range
        df_filtered = df_filtered[
            (df_filtered[param] >= bounds["min"])
            & (df_filtered[param] <= bounds["max"])
        ]

        removed = initial_count - len(df_filtered)
        outliers_removed[param] = removed

        logger.info(
            f"  - {param}: [{bounds['min']} to {bounds['max']}] "
            f"Removed {removed} outliers"
        )

    total_removed = original_count - len(df_filtered)
    logger.info(f"✓ Total data points removed: {total_removed}")
    logger.info(f"✓ Retention rate: {len(df_filtered) / original_count * 100:.1f}%")

    return df_filtered, outliers_removed


def validate_timestamp_gaps(df: pd.DataFrame, max_gap_minutes: int) -> List[tuple]:
    """
    Detect gaps in timestamp series

    Args:
        df: DataFrame with TimeStamp column
        max_gap_minutes: Maximum acceptable gap in minutes

    Returns:
        List of detected gaps
    """
    if "TimeStamp" not in df.columns:
        return []

    gaps = detect_gaps(df["TimeStamp"], max_gap_minutes)

    if gaps:
        logger.warning(f"Detected {len(gaps)} timestamp gaps > {max_gap_minutes} min:")
        for gap_start, gap_end, gap_mins in gaps[:5]:  # Show first 5
            logger.warning(f"  - Gap of {gap_mins:.1f} min: {gap_start} to {gap_end}")
        if len(gaps) > 5:
            logger.warning(f"  ... and {len(gaps) - 5} more gaps")
    else:
        logger.info(f"✓ No timestamp gaps > {max_gap_minutes} min detected")

    return gaps


def run_validation(config_path: str = "config.yaml") -> tuple:
    """
    Main validation pipeline: load, validate, filter, save

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (validated DataFrame, ValidationReport)
    """
    logger.info("=" * 60)
    logger.info("✅ VALIDATION STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    input_path = config["paths"]["raw_data"]
    output_path = config["paths"]["validated_data"]
    val_config = config["validation"]

    # Load raw data
    logger.info(f"Loading data from: {input_path}")
    df = load_parquet(input_path)

    original_rows = len(df)
    logger.info(f"Input data: {df.shape}")

    # Check required columns
    required_cols = val_config["required_columns"]
    missing_cols = check_required_columns(df, required_cols)

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Analyze missing values
    missing_values = analyze_missing_values(df)

    # Detect timestamp gaps
    max_gap = val_config["max_timestamp_gap"]
    timestamp_gaps = validate_timestamp_gaps(df, max_gap)

    # Apply bandwidth filtering
    bandwidth_filters = val_config["bandwidth_filters"]
    df_validated, outliers_removed = apply_bandwidth_filters(df, bandwidth_filters)

    final_rows = len(df_validated)

    # Create validation report
    time_range_start = df_validated["TimeStamp"].min()
    time_range_end = df_validated["TimeStamp"].max()

    report = ValidationReport(
        total_rows_input=original_rows,
        total_rows_output=final_rows,
        rows_removed=original_rows - final_rows,
        outliers_per_parameter=outliers_removed,
        missing_values=missing_values,
        timestamp_gaps=timestamp_gaps,
        time_range_start=time_range_start,
        time_range_end=time_range_end,
        warnings=[],
    )

    # Add warnings
    if timestamp_gaps:
        report.warnings.append(f"Detected {len(timestamp_gaps)} timestamp gaps")

    significant_missing = {k: v for k, v in missing_values.items() if v > 5}
    if significant_missing:
        report.warnings.append(
            f"Columns with >5% missing: {list(significant_missing.keys())}"
        )

    # Save validated data
    save_parquet(df_validated, output_path)

    # Save validation report
    import json

    report_path = "outputs/validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2, default=str)
    logger.info(f"✓ Saved validation report: {report_path}")

    # Excel export disabled - use outputs/02_validated.parquet instead
    logger.info("  Data available in: outputs/02_validated.parquet")

    logger.info("=" * 60)
    logger.info("✓ VALIDATION COMPLETE")
    logger.info("=" * 60)

    return df_validated, report


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run validation
    df, report = run_validation()
    print(f"\n✓ Validated {len(df)} rows")
    print(f"✓ Removed {report.rows_removed} outliers")
    if report.warnings:
        print(f"⚠ Warnings: {len(report.warnings)}")
