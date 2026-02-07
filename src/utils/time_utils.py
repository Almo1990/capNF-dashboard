"""
Time utility functions for timestamp conversion and analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, List
import logging

logger = logging.getLogger(__name__)


def convert_timestamp(series: pd.Series) -> pd.Series:
    """
    Convert timestamp strings from TSV format to datetime

    Handles format: "2026-01-28 03:30:04 48" -> "2026-01-28 03:30:04.48"

    Args:
        series: Pandas Series with timestamp strings

    Returns:
        Series with datetime objects
    """
    # Ensure string type
    series = series.astype(str)

    # Fix format: Replace space before last digits with a dot
    # Turns "2026-01-28 03:30:04 48" into "2026-01-28 03:30:04.48"
    series = series.str.replace(r" (\d+)$", r".\1", regex=True)

    # Convert to datetime
    try:
        return pd.to_datetime(series, format="%Y-%m-%d %H:%M:%S.%f")
    except Exception as e:
        logger.warning(f"Could not convert with specific format: {e}")
        return pd.to_datetime(series)


def calculate_duration(
    start: pd.Timestamp, end: pd.Timestamp, unit: str = "hours"
) -> float:
    """
    Calculate duration between two timestamps

    Args:
        start: Start timestamp
        end: End timestamp
        unit: Output unit (seconds, minutes, hours, days)

    Returns:
        Duration in specified unit
    """
    delta = end - start

    conversions = {
        "seconds": delta.total_seconds(),
        "minutes": delta.total_seconds() / 60,
        "hours": delta.total_seconds() / 3600,
        "days": delta.total_seconds() / 86400,
    }

    return conversions.get(unit, delta.total_seconds())


def detect_gaps(
    timestamps: pd.Series, max_gap_minutes: int = 5
) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    """
    Detect gaps in timestamp series

    Args:
        timestamps: Series of timestamps (must be sorted)
        max_gap_minutes: Gap threshold in minutes

    Returns:
        List of tuples: (gap_start, gap_end, gap_minutes)
    """
    if not isinstance(timestamps.iloc[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)

    # Calculate time differences
    diffs = timestamps.diff()

    # Find gaps exceeding threshold
    max_gap = pd.Timedelta(minutes=max_gap_minutes)
    gaps = diffs[diffs > max_gap]

    gap_list = []
    for idx, gap_duration in gaps.items():
        gap_start = timestamps.loc[idx - 1] if idx > 0 else timestamps.iloc[0]
        gap_end = timestamps.loc[idx]
        gap_minutes = gap_duration.total_seconds() / 60
        gap_list.append((gap_start, gap_end, gap_minutes))

    if gap_list:
        logger.info(f"Found {len(gap_list)} timestamp gaps > {max_gap_minutes} min")

    return gap_list


def get_time_range_info(timestamps: pd.Series) -> dict:
    """
    Get summary information about a time series

    Args:
        timestamps: Series of timestamps

    Returns:
        Dictionary with time range statistics
    """
    if not isinstance(timestamps.iloc[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)

    total_duration = timestamps.max() - timestamps.min()
    median_interval = timestamps.diff().median()

    return {
        "start": timestamps.min(),
        "end": timestamps.max(),
        "duration": total_duration,
        "duration_hours": total_duration.total_seconds() / 3600,
        "duration_days": total_duration.total_seconds() / 86400,
        "num_points": len(timestamps),
        "median_interval_seconds": median_interval.total_seconds(),
        "expected_frequency": f"{median_interval.total_seconds():.0f}s",
    }


def convert_to_seconds_since_start(timestamps: pd.Series) -> np.ndarray:
    """
    Convert timestamps to seconds since first timestamp (for regression)

    Args:
        timestamps: Series of timestamps

    Returns:
        Array of seconds since start
    """
    if not isinstance(timestamps.iloc[0], pd.Timestamp):
        timestamps = pd.to_datetime(timestamps)

    return (timestamps - timestamps.min()).dt.total_seconds().values
