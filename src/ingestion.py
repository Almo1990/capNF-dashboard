"""
[1] INGESTION MODULE
Loads and combines TSV files from Data folder
"""

import pandas as pd
import os
from pathlib import Path
import logging
import yaml
from typing import Tuple

from .utils.data_io import save_parquet
from .utils.time_utils import convert_timestamp, get_time_range_info

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def discover_tsv_files(data_folder: str) -> list:
    """
    Find all TSV files in the data folder

    Args:
        data_folder: Path to folder containing TSV files

    Returns:
        List of TSV file paths
    """
    if not os.path.exists(data_folder):
        logger.error(f"Data folder not found: {data_folder}")
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    tsv_files = [f for f in os.listdir(data_folder) if f.endswith(".tsv")]

    if not tsv_files:
        logger.error(f"No TSV files found in: {data_folder}")
        raise FileNotFoundError(f"No TSV files found in: {data_folder}")

    logger.info(f"Found {len(tsv_files)} TSV file(s):")
    for f in tsv_files:
        logger.info(f"  - {f}")

    return [os.path.join(data_folder, f) for f in tsv_files]


def load_single_tsv(filepath: str) -> pd.DataFrame:
    """
    Load a single TSV file

    Args:
        filepath: Path to TSV file

    Returns:
        DataFrame with TSV contents
    """
    logger.info(f"Reading {os.path.basename(filepath)}...")

    df = pd.read_csv(filepath, sep="\t", engine="python")

    # Normalize column names (strip whitespace)
    df.columns = df.columns.str.strip()

    return df


def combine_tsv_files(tsv_paths: list) -> pd.DataFrame:
    """
    Load and combine multiple TSV files

    Args:
        tsv_paths: List of paths to TSV files

    Returns:
        Combined DataFrame
    """
    dfs = []
    for tsv_path in tsv_paths:
        df = load_single_tsv(tsv_path)
        dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    logger.info(
        f"✓ Combined {len(tsv_paths)} file(s) into {len(combined_df)} total rows"
    )

    return combined_df


def process_timestamps(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Convert and sort by timestamp

    Args:
        df: DataFrame with TimeStamp column

    Returns:
        Tuple of (processed DataFrame, success flag)
    """
    if "TimeStamp" not in df.columns:
        logger.warning("No TimeStamp column found")
        return df, False

    try:
        # Convert timestamp format
        df["TimeStamp"] = convert_timestamp(df["TimeStamp"])

        # Sort by timestamp
        df = df.sort_values("TimeStamp").reset_index(drop=True)

        # Get time range info
        time_info = get_time_range_info(df["TimeStamp"])

        logger.info("✓ TimeStamp converted to datetime format")
        logger.info(f"  Time range: {time_info['duration']}")
        logger.info(f"  Start: {time_info['start']}")
        logger.info(f"  End: {time_info['end']}")
        logger.info(f"  Duration: {time_info['duration_days']:.2f} days")
        logger.info(f"  Sampling frequency: {time_info['expected_frequency']}")

        return df, True

    except Exception as e:
        logger.error(f"Could not process timestamps: {e}")
        return df, False


def run_ingestion(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Main ingestion pipeline: load TSV files and save to parquet

    Args:
        config_path: Path to configuration file

    Returns:
        Raw DataFrame
    """
    logger.info("=" * 60)
    logger.info("📥 INGESTION STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    data_folder = config["paths"]["data_folder"]
    output_path = config["paths"]["raw_data"]

    # Discover TSV files
    tsv_files = discover_tsv_files(data_folder)

    # Load and combine TSV files
    df = combine_tsv_files(tsv_files)

    logger.info(f"Raw data shape: {df.shape}")
    logger.info(f"Columns: {len(df.columns)}")

    # Process timestamps
    df, time_success = process_timestamps(df)

    if not time_success:
        logger.warning("Proceeding without timestamp conversion")

    # Save to parquet
    save_parquet(df, output_path)

    # # Save unfiltered CSV for reference (legacy compatibility)
    # csv_path = "combined_data_unfiltered.csv"
    # df.to_csv(csv_path, index=False)
    # logger.info(f"✓ Saved unfiltered CSV: {csv_path}")

    # logger.info("=" * 60)
    # logger.info("✓ INGESTION COMPLETE")
    # logger.info("=" * 60)

    return df


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run ingestion
    df = run_ingestion()
    print(f"\n✓ Loaded {len(df)} rows")
    print(f"✓ Columns: {list(df.columns[:10])}...")
