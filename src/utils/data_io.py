"""
Data I/O utilities for parquet file operations
"""

import pandas as pd
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def save_parquet(
    df: pd.DataFrame, filepath: str, index: bool = False, compression: str = "snappy"
) -> None:
    """
    Save DataFrame to parquet file with compression

    Args:
        df: DataFrame to save
        filepath: Output file path
        index: Whether to include index
        compression: Compression codec (snappy, gzip, brotli)
    """
    # Ensure output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    # Save to parquet
    df.to_parquet(filepath, index=index, compression=compression, engine="pyarrow")

    # Log file size
    file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
    logger.info(f"✓ Saved {filepath} ({len(df)} rows, {file_size_mb:.2f} MB)")


def load_parquet(filepath: str, columns: Optional[list] = None) -> pd.DataFrame:
    """
    Load DataFrame from parquet file

    Args:
        filepath: Input file path
        columns: Optional list of columns to load (for memory efficiency)

    Returns:
        Loaded DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Parquet file not found: {filepath}")

    df = pd.read_parquet(filepath, columns=columns, engine="pyarrow")
    logger.info(f"✓ Loaded {filepath} ({len(df)} rows)")

    return df


def get_file_info(filepath: str) -> dict:
    """
    Get information about a parquet file without loading it

    Args:
        filepath: Path to parquet file

    Returns:
        Dictionary with file metadata
    """
    if not os.path.exists(filepath):
        return {"exists": False}

    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(filepath)
    metadata = parquet_file.metadata

    return {
        "exists": True,
        "num_rows": metadata.num_rows,
        "num_columns": metadata.num_columns,
        "size_mb": os.path.getsize(filepath) / (1024 * 1024),
        "created_by": metadata.created_by,
    }
