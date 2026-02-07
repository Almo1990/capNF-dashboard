"""
[6] FILTRATION CYCLE ANALYSIS MODULE
Detect filtration cycles and analyze performance
"""

import pandas as pd
import numpy as np
import logging
import yaml
from typing import List, Tuple
import json

from .utils.data_io import load_parquet, save_parquet
from .utils.models import Cycle
from .utils.time_utils import calculate_duration

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def detect_tmp_drops(
    df: pd.DataFrame, threshold_percent: float = 20, window_minutes: int = 30
) -> List[int]:
    """
    Detect cleaning events based on TMP drops

    Args:
        df: DataFrame with TimeStamp and TMP
        threshold_percent: Minimum TMP drop % to consider as cleaning
        window_minutes: Time window for drop detection

    Returns:
        List of indices where cleanings detected
    """
    if "TMP" not in df.columns:
        logger.warning("TMP column not found")
        return []

    cleaning_indices = []

    # Calculate rolling max TMP (looking backward)
    window_size = window_minutes // 5  # Assuming 5-second sampling
    tmp_rolling_max = df["TMP"].rolling(window=window_size, min_periods=1).max()

    # Calculate percent drop from recent max
    tmp_drop_percent = (tmp_rolling_max - df["TMP"]) / tmp_rolling_max * 100

    # Identify significant drops
    drops = tmp_drop_percent > threshold_percent

    # Find transitions (where drop just occurred)
    drop_starts = drops & ~drops.shift(1, fill_value=False)

    cleaning_indices = df[drop_starts].index.tolist()

    logger.info(
        f"Detected {len(cleaning_indices)} TMP drop events (>{threshold_percent}% in {window_minutes}min)"
    )

    return cleaning_indices


def detect_chemical_timer_changes(df: pd.DataFrame) -> List[int]:
    """
    Detect cleaning events based on chemical timer increments

    Args:
        df: DataFrame with Chemical timer column

    Returns:
        List of indices where chemical timer increased
    """
    if "Chemical timer" not in df.columns:
        logger.info("Chemical timer column not found, skipping chemical detection")
        return []

    # Find where chemical timer increases
    timer_increase = df["Chemical timer"].diff() > 0

    cleaning_indices = df[timer_increase].index.tolist()

    logger.info(f"Detected {len(cleaning_indices)} chemical timer increments")

    return cleaning_indices


def detect_filtration_cycles_by_program(
    df: pd.DataFrame, filtration_program: int = 10, backwash_program: int = 21
) -> List[Tuple[int, int]]:
    """
    Detect continuous filtration periods based on active program number

    Each continuous period where program = filtration_program is one filtration cycle.
    Returns start-end index pairs for each continuous period.

    Args:
        df: DataFrame with Active program number column
        filtration_program: Program number for filtration mode (default: 10)
        backwash_program: Program number for backwashing (default: 21)

    Returns:
        List of tuples (start_idx, end_idx) for each continuous filtration period
    """
    if "Active program number" not in df.columns:
        logger.warning("Active program number column not found in data")
        return []

    prog = df["Active program number"]

    # Detect transitions TO filtration mode (start of filtration period)
    filtration_starts = (prog == filtration_program) & (
        prog.shift(1) != filtration_program
    )

    # Detect transitions FROM filtration mode (end of filtration period)
    filtration_ends = (prog != filtration_program) & (
        prog.shift(1) == filtration_program
    )

    start_indices = df[filtration_starts].index.tolist()
    end_indices = df[filtration_ends].index.tolist()

    # Pair up starts and ends to get continuous periods
    periods = []
    for start_idx in start_indices:
        # Find the next end after this start
        end_idx = None
        for e_idx in end_indices:
            if e_idx > start_idx:
                end_idx = e_idx
                break

        if end_idx is None:
            # Last period extends to end of data
            end_idx = len(df) - 1

        periods.append((start_idx, end_idx))

    logger.info(
        f"Detected {len(periods)} continuous filtration periods (program {filtration_program})"
    )

    return periods


def detect_chemical_cleanings_by_program(
    df: pd.DataFrame, chemical_program: int = 32
) -> List[int]:
    """
    Detect chemical cleaning events based on active program number

    Args:
        df: DataFrame with Active program number column
        chemical_program: Program number for chemical cleaning (default: 32)

    Returns:
        List of indices where chemical cleanings occur
    """
    if "Active program number" not in df.columns:
        logger.warning("Active program number column not found in data")
        return []

    prog = df["Active program number"]

    # Detect transitions TO chemical cleaning mode
    chemical_starts = (prog == chemical_program) & (prog.shift(1) != chemical_program)

    chemical_indices = df[chemical_starts].index.tolist()

    logger.info(
        f"Detected {len(chemical_indices)} chemical cleaning events (program {chemical_program})"
    )

    return chemical_indices


def merge_cleaning_events(
    indices1: List[int], indices2: List[int], merge_window: int = 100
) -> List[int]:
    """
    Merge cleaning events from multiple detection methods

    Args:
        indices1: First list of cleaning indices
        indices2: Second list of cleaning indices
        merge_window: Merge events within this many indices

    Returns:
        Merged list of unique cleaning indices
    """
    all_indices = sorted(set(indices1 + indices2))

    if not all_indices:
        return []

    # Merge nearby events
    merged = [all_indices[0]]

    for idx in all_indices[1:]:
        if idx - merged[-1] > merge_window:
            merged.append(idx)

    logger.info(f"Merged to {len(merged)} unique cleaning events")

    return merged


def segment_into_cycles(
    df: pd.DataFrame, cleaning_indices: List[int], min_duration_hours: float = 1.0
) -> Tuple[pd.DataFrame, List[Cycle]]:
    """
    Segment data into filtration cycles between cleanings

    Args:
        df: DataFrame with all data
        cleaning_indices: List of indices where cleanings occurred
        min_duration_hours: Minimum cycle duration to keep

    Returns:
        Tuple of (DataFrame with cycle_id column, List of Cycle objects)
    """
    logger.info("Segmenting data into filtration cycles...")

    df_cycles = df.copy()
    df_cycles["cycle_id"] = 0

    if not cleaning_indices:
        logger.warning("No cleaning events detected, treating all data as single cycle")
        df_cycles["cycle_id"] = 1
        cleaning_indices = [0, len(df) - 1]

    # Add start and end boundaries
    boundaries = [0] + cleaning_indices + [len(df)]
    boundaries = sorted(set(boundaries))

    cycles = []
    cycle_id = 1

    for i in range(len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1]

        cycle_df = df.iloc[start_idx:end_idx]

        if len(cycle_df) < 10:  # Skip very short segments
            continue

        # Check minimum duration
        if "TimeStamp" in cycle_df.columns:
            duration_hours = calculate_duration(
                cycle_df["TimeStamp"].iloc[0],
                cycle_df["TimeStamp"].iloc[-1],
                unit="hours",
            )

            if duration_hours < min_duration_hours:
                continue
        else:
            duration_hours = 0

        # Extract cycle metrics
        cycle = create_cycle_object(cycle_df, cycle_id, duration_hours)

        if cycle:
            cycles.append(cycle)
            df_cycles.loc[start_idx : end_idx - 1, "cycle_id"] = cycle_id
            cycle_id += 1

    logger.info(f"✓ Identified {len(cycles)} filtration cycles")

    return df_cycles, cycles


def create_cycle_object(
    df_cycle: pd.DataFrame, cycle_id: int, duration_hours: float
) -> Cycle:
    """
    Create a Cycle object from cycle data

    Args:
        df_cycle: DataFrame for this cycle
        cycle_id: Cycle identifier
        duration_hours: Cycle duration in hours

    Returns:
        Cycle object
    """
    try:
        # Extract start/end times
        start_time = (
            df_cycle["TimeStamp"].iloc[0] if "TimeStamp" in df_cycle.columns else None
        )
        end_time = (
            df_cycle["TimeStamp"].iloc[-1] if "TimeStamp" in df_cycle.columns else None
        )

        # TMP metrics - use stable period median instead of first point to reduce noise
        if "TMP" in df_cycle.columns and len(df_cycle) > 10:
            # Skip first 10 points (startup transients), then take median of next 30-50 points
            # This represents the stable initial TMP after cycle startup
            start_window = df_cycle["TMP"].iloc[10:60]  # Points 10-60
            tmp_start = (
                start_window.median()
                if len(start_window) > 0
                else df_cycle["TMP"].iloc[0]
            )

            # For tmp_end, use median of last stable period
            end_window = df_cycle["TMP"].iloc[-50:-10]  # Avoid last 10 points
            tmp_end = (
                end_window.median() if len(end_window) > 0 else df_cycle["TMP"].iloc[-1]
            )
        else:
            tmp_start = df_cycle["TMP"].iloc[0] if "TMP" in df_cycle.columns else 0
            tmp_end = df_cycle["TMP"].iloc[-1] if "TMP" in df_cycle.columns else 0

        # Calculate TMP slope for this cycle
        if "TMP" in df_cycle.columns and len(df_cycle) > 1:
            from .utils.time_utils import convert_to_seconds_since_start

            time_numeric = convert_to_seconds_since_start(df_cycle["TimeStamp"])
            coeffs = np.polyfit(time_numeric, df_cycle["TMP"], 1)
            tmp_slope = coeffs[0] * 3600  # bar/hour

            # R-squared
            fit_values = coeffs[0] * time_numeric + coeffs[1]
            y_mean = df_cycle["TMP"].mean()
            ss_tot = np.sum((df_cycle["TMP"] - y_mean) ** 2)
            ss_res = np.sum((df_cycle["TMP"] - fit_values) ** 2)
            tmp_slope_r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        else:
            tmp_slope = 0
            tmp_slope_r2 = 0

        # Permeability metrics
        if "Permeability TC" in df_cycle.columns:
            perm_start = df_cycle["Permeability TC"].iloc[0]
            perm_end = df_cycle["Permeability TC"].iloc[-1]
            perm_decline = (
                (perm_start - perm_end) / perm_start * 100 if perm_start > 0 else 0
            )
        else:
            perm_start = perm_end = perm_decline = 0

        # Average metrics
        avg_flux = df_cycle["Flux"].mean() if "Flux" in df_cycle.columns else 0
        avg_recovery = (
            df_cycle["Recovery"].mean() if "Recovery" in df_cycle.columns else 0
        )
        avg_retention = (
            df_cycle["Mem. retention"].mean()
            if "Mem. retention" in df_cycle.columns
            else 0
        )

        # Create cycle object
        cycle = Cycle(
            cycle_id=cycle_id,
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration_hours,
            tmp_start=tmp_start,
            tmp_end=tmp_end,
            tmp_slope=tmp_slope,
            tmp_slope_r2=tmp_slope_r2,
            permeability_start=perm_start,
            permeability_end=perm_end,
            permeability_decline_percent=perm_decline,
            avg_flux=avg_flux,
            avg_recovery=avg_recovery,
            avg_retention=avg_retention,
        )

        return cycle

    except Exception as e:
        logger.warning(f"Error creating cycle object: {e}")
        return None


def calculate_fouling_rates(
    cycles: List[Cycle],
    chemical_cleaning_timestamps: List[str],
) -> dict:
    """
    Calculate reversible and irreversible fouling rates from cycle data.

    Reversible fouling: Slope of TMP at the start of ALL filtration cycles.
    This represents the fouling that accumulates between backwashes but is
    removed by backwash (though it accumulates again).

    Irreversible fouling: Slope of TMP at the start of the FIRST cycle after
    each chemical cleaning. This represents the baseline TMP increase that
    chemical cleaning cannot remove.

    Args:
        cycles: List of Cycle objects with tmp_start values
        chemical_cleaning_timestamps: List of chemical cleaning timestamp strings

    Returns:
        Dictionary with fouling rate statistics
    """
    logger.info("\nCalculating fouling rates...")

    if not cycles:
        logger.warning("No cycles available for fouling rate calculation")
        return {
            "reversible_fouling_rate_bar_per_day": None,
            "reversible_r_squared": None,
            "irreversible_fouling_rate_bar_per_day": None,
            "irreversible_r_squared": None,
            "total_cycles": 0,
            "post_cleaning_cycles": 0,
        }

    # Convert timestamps to datetime for comparison
    cleaning_times = [pd.to_datetime(ts) for ts in chemical_cleaning_timestamps]

    # Extract TMP at start of each cycle and cycle start times
    tmp_starts = []
    cycle_times = []

    for cycle in cycles:
        if cycle.tmp_start is not None and cycle.start_time is not None:
            tmp_starts.append(cycle.tmp_start)
            cycle_times.append(pd.to_datetime(cycle.start_time))

    if len(tmp_starts) < 2:
        logger.warning(
            "Not enough cycles with tmp_start data for fouling rate calculation"
        )
        return {
            "reversible_fouling_rate_bar_per_day": None,
            "reversible_r_squared": None,
            "irreversible_fouling_rate_bar_per_day": None,
            "irreversible_r_squared": None,
            "total_cycles": len(tmp_starts),
            "post_cleaning_cycles": 0,
        }

    # Remove outliers using IQR method to improve R²
    tmp_starts_array = np.array(tmp_starts)
    q1 = np.percentile(tmp_starts_array, 25)
    q3 = np.percentile(tmp_starts_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Filter out outliers
    valid_indices = [
        i for i, tmp in enumerate(tmp_starts) if lower_bound <= tmp <= upper_bound
    ]

    if len(valid_indices) < 2:
        logger.warning("Not enough valid cycles after outlier removal")
        return {
            "reversible_fouling_rate_bar_per_day": None,
            "reversible_r_squared": None,
            "irreversible_fouling_rate_bar_per_day": None,
            "irreversible_r_squared": None,
            "total_cycles": len(tmp_starts),
            "post_cleaning_cycles": 0,
        }

    tmp_starts_filtered = [tmp_starts[i] for i in valid_indices]
    cycle_times_filtered = [cycle_times[i] for i in valid_indices]

    logger.info(
        f"  Outlier filtering: {len(tmp_starts)} cycles → {len(tmp_starts_filtered)} valid cycles"
    )
    logger.info(
        f"  TMP range: {min(tmp_starts_filtered):.3f} - {max(tmp_starts_filtered):.3f} bar"
    )

    # Calculate reversible fouling rate (all valid cycles)
    cycle_times_numeric = [
        (t - cycle_times_filtered[0]).total_seconds() / 86400
        for t in cycle_times_filtered
    ]  # Convert to days

    # Linear regression: TMP = slope * days + intercept
    coeffs_reversible = np.polyfit(cycle_times_numeric, tmp_starts_filtered, 1)
    slope_reversible = coeffs_reversible[0]  # bar/day
    intercept_reversible = coeffs_reversible[1]

    # Calculate R² for reversible
    tmp_pred_reversible = np.polyval(coeffs_reversible, cycle_times_numeric)
    ss_res_reversible = np.sum(
        (np.array(tmp_starts_filtered) - tmp_pred_reversible) ** 2
    )
    ss_tot_reversible = np.sum(
        (np.array(tmp_starts_filtered) - np.mean(tmp_starts_filtered)) ** 2
    )
    r_squared_reversible = (
        1 - (ss_res_reversible / ss_tot_reversible) if ss_tot_reversible > 0 else 0
    )

    logger.info(
        f"  Reversible fouling rate: {slope_reversible:.6f} bar/day (R² = {r_squared_reversible:.4f})"
    )
    logger.info(
        f"  Based on {len(tmp_starts_filtered)} valid cycles (out of {len(tmp_starts)} total)"
    )

    # Calculate irreversible fouling rate (first cycle after each chemical cleaning)
    post_cleaning_tmp = []
    post_cleaning_times = []

    # Track which cleanings we've already matched to avoid duplicates
    matched_cleanings = set()

    for i, cycle in enumerate(cycles):
        if cycle.tmp_start is None or cycle.start_time is None:
            continue

        cycle_time = pd.to_datetime(cycle.start_time)

        # Check if this is the first cycle after a chemical cleaning
        for j, cleaning_time in enumerate(cleaning_times):
            if j in matched_cleanings:
                continue  # Already found a post-cleaning cycle for this cleaning

            # If cycle starts within 3 hours after a cleaning, consider it post-cleaning
            time_diff = (cycle_time - cleaning_time).total_seconds() / 3600  # hours
            if 0 < time_diff <= 3:
                post_cleaning_tmp.append(cycle.tmp_start)
                post_cleaning_times.append(cycle_time)
                matched_cleanings.add(j)
                break

    # Calculate irreversible fouling rate if we have enough post-cleaning data
    if len(post_cleaning_tmp) >= 2:
        post_cleaning_times_numeric = [
            (t - post_cleaning_times[0]).total_seconds() / 86400
            for t in post_cleaning_times
        ]  # Convert to days

        coeffs_irreversible = np.polyfit(
            post_cleaning_times_numeric, post_cleaning_tmp, 1
        )
        slope_irreversible = coeffs_irreversible[0]  # bar/day
        intercept_irreversible = coeffs_irreversible[1]

        # Calculate R² for irreversible
        tmp_pred_irreversible = np.polyval(
            coeffs_irreversible, post_cleaning_times_numeric
        )
        ss_res_irreversible = np.sum(
            (np.array(post_cleaning_tmp) - tmp_pred_irreversible) ** 2
        )
        ss_tot_irreversible = np.sum(
            (np.array(post_cleaning_tmp) - np.mean(post_cleaning_tmp)) ** 2
        )
        r_squared_irreversible = (
            1 - (ss_res_irreversible / ss_tot_irreversible)
            if ss_tot_irreversible > 0
            else 0
        )

        logger.info(
            f"  Irreversible fouling rate: {slope_irreversible:.6f} bar/day (R² = {r_squared_irreversible:.4f})"
        )
        logger.info(f"  Based on {len(post_cleaning_tmp)} post-cleaning cycles")
    else:
        slope_irreversible = None
        r_squared_irreversible = None
        logger.warning(
            f"  Not enough post-cleaning cycles ({len(post_cleaning_tmp)}) for irreversible fouling rate"
        )

    results = {
        "reversible_fouling_rate_bar_per_day": float(slope_reversible),
        "reversible_r_squared": float(r_squared_reversible),
        "irreversible_fouling_rate_bar_per_day": float(slope_irreversible)
        if slope_irreversible is not None
        else None,
        "irreversible_r_squared": float(r_squared_irreversible)
        if r_squared_irreversible is not None
        else None,
        "total_cycles": len(tmp_starts),
        "valid_cycles": len(tmp_starts_filtered),
        "outliers_removed": len(tmp_starts) - len(tmp_starts_filtered),
        "post_cleaning_cycles": len(post_cleaning_tmp),
    }

    return results


def run_cleaning_analysis(
    config_path: str = "config.yaml",
) -> Tuple[pd.DataFrame, List[Cycle]]:
    """
    Main filtration cycle analysis pipeline

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (DataFrame with cycle annotations, List of Cycle objects)
    """
    logger.info("=" * 60)
    logger.info("🔄 FILTRATION CYCLE ANALYSIS STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    input_path = config["paths"]["fouling_metrics_data"]
    output_cycles = config["paths"]["cycles_data"]
    output_summary = config["paths"]["cycle_summary_data"]
    filtration_config = config["filtration_detection"]

    # Load data - need raw data for Active program number
    logger.info(f"Loading data from raw parquet (for program numbers)...")
    df_raw = load_parquet("outputs/01_raw.parquet")

    # Also load processed data for merging
    logger.info(f"Loading processed data from: {input_path}")
    df = load_parquet(input_path)

    logger.info(f"Input data: {df.shape}")

    # Check if program number detection is enabled
    program_config = filtration_config.get("program_number", {})
    use_program_detection = program_config.get("enabled", False)

    if use_program_detection and "Active program number" in df_raw.columns:
        logger.info("Using Active program number for filtration cycle detection")

        # Detect continuous filtration periods by program number
        filtration_program = program_config.get("filtration_program", 10)
        backwash_program = program_config.get("backwash_program", 21)
        filtration_periods = detect_filtration_cycles_by_program(
            df_raw, filtration_program, backwash_program
        )

        # Detect chemical cleanings separately
        chemical_program = program_config.get("chemical_cleaning_program", 32)
        chemical_cleaning_indices = detect_chemical_cleanings_by_program(
            df_raw, chemical_program
        )

        # Save chemical cleaning information for KPIs
        chemical_cleanings_json = {
            "chemical_cleanings": len(chemical_cleaning_indices),
            "chemical_cleaning_timestamps": [
                str(df_raw.loc[idx, "TimeStamp"]) for idx in chemical_cleaning_indices
            ],
        }
        with open("outputs/chemical_cleanings.json", "w") as f:
            json.dump(chemical_cleanings_json, f, indent=2)
        logger.info(f"✓ Saved chemical cleaning info: outputs/chemical_cleanings.json")

        # Filter periods by minimum duration and map to processed data
        min_duration_str = filtration_config.get("min_cycle_duration", "1h")
        min_duration_hours = pd.Timedelta(min_duration_str).total_seconds() / 3600

        valid_cycles = []
        df_cycles = df.copy()
        df_cycles["cycle_id"] = 0
        cycle_id = 1

        logger.info(f"Filtering periods with minimum duration: {min_duration_str}")

        for raw_start_idx, raw_end_idx in filtration_periods:
            # Get timestamps for this period
            start_time = df_raw.loc[raw_start_idx, "TimeStamp"]
            end_time = df_raw.loc[raw_end_idx, "TimeStamp"]
            duration_hours = (end_time - start_time).total_seconds() / 3600

            # Check minimum duration
            if duration_hours < min_duration_hours:
                continue

            # Find corresponding indices in processed data
            proc_start = df[df["TimeStamp"] >= start_time].index
            proc_end = df[df["TimeStamp"] <= end_time].index

            if len(proc_start) == 0 or len(proc_end) == 0:
                continue

            proc_start_idx = proc_start[0]
            proc_end_idx = proc_end[-1]

            cycle_df = df.loc[proc_start_idx:proc_end_idx]

            if len(cycle_df) < 10:  # Skip very short segments
                continue

            # Recalculate duration based on processed data timestamps
            duration_hours = (
                cycle_df["TimeStamp"].iloc[-1] - cycle_df["TimeStamp"].iloc[0]
            ).total_seconds() / 3600

            # Create cycle object
            cycle = create_cycle_object(cycle_df, cycle_id, duration_hours)

            if cycle:
                valid_cycles.append(cycle)
                df_cycles.loc[proc_start_idx:proc_end_idx, "cycle_id"] = cycle_id
                cycle_id += 1

        logger.info(
            f"✓ Identified {len(valid_cycles)} filtration cycles (> {min_duration_str})"
        )

        # Skip the segment_into_cycles step - we already have our cycles
        df_cycles_final = df_cycles
        cycles_final = valid_cycles

    else:
        logger.info("Using TMP drop method for cycle detection (fallback)")

        # Fallback to TMP drop detection
        tmp_config = filtration_config["tmp_drop"]
        tmp_cleaning_indices = detect_tmp_drops(
            df,
            threshold_percent=tmp_config["threshold_percent"],
            window_minutes=tmp_config["window_minutes"],
        )

        # Detect cleaning events via chemical timer
        chem_cleaning_indices = detect_chemical_timer_changes(df)

        # Merge cleaning events
        all_cycle_indices = merge_cleaning_events(
            tmp_cleaning_indices, chem_cleaning_indices
        )

        # Segment into cycles
        min_duration = (
            pd.Timedelta(filtration_config["min_cycle_duration"]).total_seconds() / 3600
        )
        df_cycles_final, cycles_final = segment_into_cycles(
            df, all_cycle_indices, min_duration
        )

    # Save annotated full dataset
    save_parquet(df_cycles_final, output_cycles)

    # Create cycle summary DataFrame
    cycle_summary = pd.DataFrame([cycle.to_dict() for cycle in cycles_final])

    # Drop metadata column to avoid PyArrow struct type error
    if "metadata" in cycle_summary.columns:
        cycle_summary = cycle_summary.drop(columns=["metadata"])

    save_parquet(cycle_summary, output_summary)

    # Save cycle summary as JSON
    cycles_json_path = "outputs/cycles_summary.json"
    with open(cycles_json_path, "w") as f:
        json.dump([cycle.to_dict() for cycle in cycles_final], f, indent=2, default=str)
    logger.info(f"✓ Saved cycle summary: {cycles_json_path}")

    # Log cycle statistics
    if cycles_final:
        logger.info(f"\nCycle Statistics:")
        logger.info(f"  Number of cycles: {len(cycles_final)}")
        logger.info(
            f"  Avg duration: {np.mean([c.duration_hours for c in cycles_final]):.1f} hours"
        )
        logger.info(
            f"  Avg TMP slope: {np.mean([c.tmp_slope for c in cycles_final]):.6f} bar/hour"
        )
        logger.info(
            f"  Avg permeability decline: {np.mean([c.permeability_decline_percent for c in cycles_final]):.1f}%"
        )

    # Calculate fouling rates
    try:
        # Load chemical cleaning timestamps
        chemical_cleanings_path = "outputs/chemical_cleanings.json"
        with open(chemical_cleanings_path, "r") as f:
            cleaning_data = json.load(f)
        chemical_cleaning_timestamps = cleaning_data.get(
            "chemical_cleaning_timestamps", []
        )

        # Calculate fouling rates
        fouling_rates = calculate_fouling_rates(
            cycles_final, chemical_cleaning_timestamps
        )

        # Save fouling rates
        fouling_rates_path = "outputs/fouling_rates.json"
        with open(fouling_rates_path, "w") as f:
            json.dump(fouling_rates, f, indent=2)
        logger.info(f"✓ Saved fouling rates: {fouling_rates_path}")
    except Exception as e:
        logger.warning(f"Could not calculate fouling rates: {e}")

    logger.info("=" * 60)
    logger.info("✓ CLEANING ANALYSIS COMPLETE")
    logger.info("=" * 60)

    return df_cycles_final, cycles_final


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run cleaning analysis
    df, cycles = run_cleaning_analysis()
    print(f"\n✓ Identified {len(cycles)} filtration cycles")
    if cycles:
        print(
            f"✓ Average cycle duration: {np.mean([c.duration_hours for c in cycles]):.1f} hours"
        )
