"""
[7] KPI ENGINE MODULE
Calculate KPIs, generate alerts, and create forecasts
"""

import pandas as pd
import numpy as np
import logging
import yaml
import json
from typing import List, Tuple, Dict

from .utils.data_io import load_parquet
from .utils.models import KPI, Alert, Forecast
from .utils.time_utils import convert_to_seconds_since_start

try:
    from .ml_forecasting import (
        create_ml_tmp_forecast,
        create_ml_permeability_forecast,
        ML_AVAILABLE,
    )
except ImportError:
    ML_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("ML forecasting module not available")

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def calculate_kpis(df: pd.DataFrame, cycles: List[dict], targets: dict) -> List[KPI]:
    """
    Calculate key performance indicators

    Args:
        df: Full dataset
        cycles: List of cycle dictionaries
        targets: Target values from config

    Returns:
        List of KPI objects
    """
    logger.info("Calculating KPIs...")

    kpis = []

    # Overall recovery rate
    if "Recovery" in df.columns:
        avg_recovery = df["Recovery"].mean()
        kpi = KPI(
            name="Average Recovery Rate",
            value=avg_recovery,
            unit="%",
            target=targets.get("recovery_rate"),
            status="good"
            if avg_recovery >= targets.get("recovery_rate", 0)
            else "warning",
        )
        kpis.append(kpi)

    # Specific energy consumption
    if "Specific_power" in df.columns:
        avg_energy = df["Specific_power"].mean()
        kpi = KPI(
            name="Average Specific Energy",
            value=round(avg_energy, 2),
            unit="kWh/m³",
            target=targets.get("specific_energy"),
            status="good"
            if avg_energy <= targets.get("specific_energy", 1)
            else "warning",
        )
        kpis.append(kpi)

    # Cycle frequency
    if cycles:
        cycle_durations = [c["duration_hours"] for c in cycles]
        avg_cycle_duration = np.mean(cycle_durations)

        kpi = KPI(
            name="Average Cycle Duration",
            value=round(avg_cycle_duration, 2),
            unit="hours",
            target=targets.get("cycle_duration"),
            status="good"
            if avg_cycle_duration >= targets.get("cycle_duration", 0) * 0.8
            else "warning",
        )
        kpis.append(kpi)

        # Filtration cycles per week
        if len(cycles) > 1:
            first_start = pd.Timestamp(cycles[0]["start_time"])
            last_end = pd.Timestamp(cycles[-1]["end_time"])
            total_days = (last_end - first_start).total_seconds() / 86400
            cycles_per_week = len(cycles) / (total_days / 7) if total_days > 0 else 0

            kpi = KPI(
                name="Filtration Cycles per Week",
                value=cycles_per_week,
                unit="cycles/week",
            )
            kpis.append(kpi)

    # Chemical cleaning frequency (from separate tracking)
    try:
        import json
        import os

        if os.path.exists("outputs/chemical_cleanings.json"):
            with open("outputs/chemical_cleanings.json", "r") as f:
                chem_data = json.load(f)

            num_chemical_cleanings = chem_data.get("chemical_cleanings", 0)

            if (
                num_chemical_cleanings > 0
                and len(chem_data.get("chemical_cleaning_timestamps", [])) > 1
            ):
                # Calculate frequency based on time span
                timestamps = [
                    pd.Timestamp(ts) for ts in chem_data["chemical_cleaning_timestamps"]
                ]
                first_cleaning = min(timestamps)
                last_cleaning = max(timestamps)
                total_days = (last_cleaning - first_cleaning).total_seconds() / 86400

                if total_days > 0:
                    chem_cleanings_per_week = num_chemical_cleanings / (total_days / 7)
                else:
                    chem_cleanings_per_week = 0
            else:
                chem_cleanings_per_week = 0

            kpi = KPI(
                name="Chemical Cleaning Frequency",
                value=int(round(chem_cleanings_per_week)),
                unit="cleanings/week",
            )
            kpis.append(kpi)
    except Exception as e:
        logger.warning(f"Could not load chemical cleaning data: {e}")

    # Current TMP
    if "TMP" in df.columns:
        current_tmp = df["TMP"].iloc[-1]
        kpi = KPI(
            name="Current TMP",
            value=round(current_tmp, 2),
            unit="bar",
            target=targets.get("tmp_max"),
            status="good"
            if current_tmp < targets.get("tmp_max", 8) * 0.8
            else "warning",
        )
        kpis.append(kpi)

    # Fouling rates
    try:
        fouling_rates_path = "outputs/fouling_rates.json"
        with open(fouling_rates_path, "r") as f:
            fouling_data = json.load(f)

        # Reversible fouling rate
        reversible_rate = fouling_data.get("reversible_fouling_rate_bar_per_day")
        if reversible_rate is not None:
            kpi = KPI(
                name="Reversible Fouling Rate",
                value=round(reversible_rate, 4),
                unit="bar/day",
            )
            kpis.append(kpi)

        # Irreversible fouling rate
        irreversible_rate = fouling_data.get("irreversible_fouling_rate_bar_per_day")
        if irreversible_rate is not None:
            kpi = KPI(
                name="Irreversible Fouling Rate",
                value=round(irreversible_rate, 4),
                unit="bar/day",
            )
            kpis.append(kpi)
    except Exception as e:
        logger.warning(f"Could not load fouling rate data: {e}")

    logger.info(f"✓ Calculated {len(kpis)} KPIs")

    return kpis


def generate_alerts(
    df: pd.DataFrame, cycles: List[dict], fouling_metadata: dict, alert_config: dict
) -> List[Alert]:
    """
    Generate alerts based on thresholds

    Args:
        df: Full dataset
        cycles: List of cycle dictionaries
        fouling_metadata: Fouling analysis metadata
        alert_config: Alert thresholds from config

    Returns:
        List of Alert objects
    """
    logger.info("Generating alerts...")

    alerts = []

    # TMP slope alerts
    tmp_slope = fouling_metadata.get("global_tmp_slope_per_hour", 0)
    tmp_thresholds = alert_config["tmp_slope"]

    if tmp_slope >= tmp_thresholds["critical"]:
        alert = Alert(
            alert_type="tmp_slope",
            severity="critical",
            timestamp=pd.Timestamp.now(),
            value=tmp_slope,
            threshold=tmp_thresholds["critical"],
            message=f"CRITICAL: TMP slope {tmp_slope:.6f} bar/hour exceeds threshold {tmp_thresholds['critical']}",
        )
        alerts.append(alert)
    elif tmp_slope >= tmp_thresholds["warning"]:
        alert = Alert(
            alert_type="tmp_slope",
            severity="warning",
            timestamp=pd.Timestamp.now(),
            value=tmp_slope,
            threshold=tmp_thresholds["warning"],
            message=f"WARNING: TMP slope {tmp_slope:.6f} bar/hour exceeds threshold {tmp_thresholds['warning']}",
        )
        alerts.append(alert)

    # Permeability decline alerts (baseline method)
    if "permeability_decline_baseline" in df.columns:
        decline_value = df["permeability_decline_baseline"].iloc[-1]
        baseline_value = (
            df["permeability_baseline"].iloc[-1]
            if "permeability_baseline" in df.columns
            else None
        )
        perm_thresholds = alert_config["permeability_decline"]

        if decline_value >= perm_thresholds["critical"]:
            message = (
                f"CRITICAL: Permeability declined {decline_value:.1f}% from baseline"
            )
            if baseline_value:
                message += f" ({baseline_value:.2f} LMH/bar)"
            alert = Alert(
                alert_type="permeability_decline",
                severity="critical",
                timestamp=pd.Timestamp.now(),
                value=decline_value,
                threshold=perm_thresholds["critical"],
                message=message,
            )
            alerts.append(alert)
        elif decline_value >= perm_thresholds["warning"]:
            message = (
                f"WARNING: Permeability declined {decline_value:.1f}% from baseline"
            )
            if baseline_value:
                message += f" ({baseline_value:.2f} LMH/bar)"
            alert = Alert(
                alert_type="permeability_decline",
                severity="warning",
                timestamp=pd.Timestamp.now(),
                value=decline_value,
                threshold=perm_thresholds["warning"],
                message=message,
            )
            alerts.append(alert)

    # Current TMP alert
    if "TMP" in df.columns:
        current_tmp = df["TMP"].iloc[-1]
        if current_tmp >= 7.5:
            alert = Alert(
                alert_type="high_tmp",
                severity="warning",
                timestamp=df["TimeStamp"].iloc[-1]
                if "TimeStamp" in df.columns
                else pd.Timestamp.now(),
                value=current_tmp,
                threshold=7.5,
                message=f"WARNING: TMP at {current_tmp:.2f} bar, approaching max (8 bar). Consider cleaning.",
            )
            alerts.append(alert)

    logger.info(f"✓ Generated {len(alerts)} alerts")

    return alerts


def create_tmp_forecast(
    df: pd.DataFrame,
    model_type: str = "linear",
    horizon_days: float = 7,
    confidence: float = 0.95,
    start_date: str = None,
) -> Forecast:
    """
    Forecast TMP growth

    Args:
        df: DataFrame with TimeStamp and TMP
        model_type: Forecasting model ('linear' or 'exponential')
        horizon_days: Forecast horizon in days
        confidence: Confidence level for prediction interval
        start_date: Optional start date (str) to filter data from (e.g., '2026-01-28 15:00:00')

    Returns:
        Forecast object
    """
    logger.info(f"Creating TMP forecast ({model_type}, {horizon_days} days ahead)...")

    if "TMP" not in df.columns or "TimeStamp" not in df.columns:
        logger.warning("Insufficient data for forecasting")
        return None

    # Prepare data
    df_forecast = df[["TimeStamp", "TMP"]].dropna()

    # Filter data from start_date if provided
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df_forecast = df_forecast[df_forecast["TimeStamp"] >= start_dt]
        logger.info(
            f"Filtered forecast data from {start_date}: {len(df_forecast)} points"
        )

    if len(df_forecast) < 100:
        logger.warning("Insufficient data points for reliable forecast")
        return None

    # Convert time to numeric
    time_numeric = convert_to_seconds_since_start(df_forecast["TimeStamp"])

    # Fit model
    if model_type == "linear":
        coeffs = np.polyfit(time_numeric, df_forecast["TMP"], 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Predict future
        current_time = time_numeric[-1]
        future_time = current_time + (horizon_days * 86400)  # days to seconds
        predicted_tmp = slope * future_time + intercept

        # Calculate R-squared
        fit_values = slope * time_numeric + intercept
        y_mean = df_forecast["TMP"].mean()
        ss_tot = np.sum((df_forecast["TMP"] - y_mean) ** 2)
        ss_res = np.sum((df_forecast["TMP"] - fit_values) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Estimate confidence bounds (simple std error)
        residual_std = np.sqrt(ss_res / (len(df_forecast) - 2))
        margin = 1.96 * residual_std  # 95% confidence
        lower_bound = predicted_tmp - margin
        upper_bound = predicted_tmp + margin

    else:  # exponential
        # Log transform for exponential fit
        log_tmp = np.log(df_forecast["TMP"] + 0.1)  # avoid log(0)
        coeffs = np.polyfit(time_numeric, log_tmp, 1)

        # For exponential model, coeffs are for log-space
        # To calculate time-to-threshold, we need to solve: exp(slope*t + intercept) - 0.1 = threshold
        slope = coeffs[0]  # This is in log space
        intercept = coeffs[1]  # This is in log space

        current_time = time_numeric[-1]
        future_time = current_time + (horizon_days * 86400)
        predicted_tmp = np.exp(coeffs[0] * future_time + coeffs[1]) - 0.1

        r_squared = 0.0  # Simplified
        lower_bound = predicted_tmp * 0.9
        upper_bound = predicted_tmp * 1.1

    # Current value
    current_tmp = df_forecast["TMP"].iloc[-1]
    current_time_stamp = df_forecast["TimeStamp"].iloc[-1]
    prediction_date = current_time_stamp + pd.Timedelta(days=horizon_days)

    # Calculate time to threshold (8 bar)
    threshold_tmp = 8.0
    if model_type == "linear" and slope > 0:
        time_to_threshold_seconds = (
            threshold_tmp - (slope * time_numeric[-1] + intercept)
        ) / slope
        time_to_threshold_days = (time_to_threshold_seconds - time_numeric[-1]) / 86400
        threshold_date = current_time_stamp + pd.Timedelta(days=time_to_threshold_days)
    elif model_type == "exponential" and slope > 0:
        # For exponential: solve exp(slope*t + intercept) - 0.1 = threshold_tmp
        # t = (ln(threshold_tmp + 0.1) - intercept) / slope
        time_to_threshold_seconds = (np.log(threshold_tmp + 0.1) - intercept) / slope
        time_to_threshold_days = (time_to_threshold_seconds - time_numeric[-1]) / 86400
        threshold_date = current_time_stamp + pd.Timedelta(days=time_to_threshold_days)
    else:
        time_to_threshold_days = None
        threshold_date = None

    # Create message
    if predicted_tmp < threshold_tmp:
        message = f"TMP forecasted to reach {predicted_tmp:.2f} bar in {horizon_days} days (within safe range)"
    else:
        message = f"WARNING: TMP forecasted to exceed {threshold_tmp} bar within {horizon_days} days"

    forecast = Forecast(
        parameter="TMP",
        model_type=model_type,
        forecast_horizon_days=horizon_days,
        current_value=current_tmp,
        predicted_value=predicted_tmp,
        prediction_date=prediction_date,
        confidence_level=confidence,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        r_squared=r_squared,
        message=message,
        time_to_threshold=time_to_threshold_days,
        threshold_value=threshold_tmp,
        threshold_date=threshold_date,
    )

    logger.info(f"✓ TMP forecast: {predicted_tmp:.2f} bar in {horizon_days} days")
    if time_to_threshold_days and time_to_threshold_days > 0:
        logger.info(f"  Time to 8 bar threshold: {time_to_threshold_days:.1f} days")

    return forecast


def create_permeability_forecast(
    df: pd.DataFrame,
    model_type: str = "linear",
    horizon_days: float = 7,
    confidence: float = 0.95,
    start_date: str = None,
) -> Forecast:
    """
    Forecast Permeability TC decline

    Args:
        df: DataFrame with TimeStamp and Permeability TC
        model_type: Forecasting model ('linear' or 'exponential')
        horizon_days: Forecast horizon in days
        confidence: Confidence level for prediction interval
        start_date: Optional start date (str) to filter data from (e.g., '2026-01-28 15:00:00')

    Returns:
        Forecast object
    """
    logger.info(
        f"Creating Permeability TC forecast ({model_type}, {horizon_days} days ahead)..."
    )

    if "Permeability TC" not in df.columns or "TimeStamp" not in df.columns:
        logger.warning("Insufficient data for Permeability TC forecasting")
        return None

    # Prepare data
    df_forecast = df[["TimeStamp", "Permeability TC"]].dropna()

    # Filter data from start_date if provided
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df_forecast = df_forecast[df_forecast["TimeStamp"] >= start_dt]
        logger.info(
            f"Filtered forecast data from {start_date}: {len(df_forecast)} points"
        )

    if len(df_forecast) < 100:
        logger.warning("Insufficient data points for reliable forecast")
        return None

    # Convert time to numeric
    time_numeric = convert_to_seconds_since_start(df_forecast["TimeStamp"])

    # Fit model
    if model_type == "linear":
        coeffs = np.polyfit(time_numeric, df_forecast["Permeability TC"], 1)
        slope = coeffs[0]
        intercept = coeffs[1]

        # Predict future
        current_time = time_numeric[-1]
        future_time = current_time + (horizon_days * 86400)  # days to seconds
        predicted_perm = slope * future_time + intercept

        # Calculate R-squared
        fit_values = slope * time_numeric + intercept
        y_mean = df_forecast["Permeability TC"].mean()
        ss_tot = np.sum((df_forecast["Permeability TC"] - y_mean) ** 2)
        ss_res = np.sum((df_forecast["Permeability TC"] - fit_values) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Estimate confidence bounds (simple std error)
        residual_std = np.sqrt(ss_res / (len(df_forecast) - 2))
        margin = 1.96 * residual_std  # 95% confidence
        lower_bound = predicted_perm - margin
        upper_bound = predicted_perm + margin

    else:  # exponential
        # Log transform for exponential fit
        log_perm = np.log(df_forecast["Permeability TC"] + 0.1)  # avoid log(0)
        coeffs = np.polyfit(time_numeric, log_perm, 1)

        # For exponential model, coeffs are for log-space
        slope = coeffs[0]  # This is in log space
        intercept = coeffs[1]  # This is in log space

        current_time = time_numeric[-1]
        future_time = current_time + (horizon_days * 86400)
        predicted_perm = np.exp(coeffs[0] * future_time + coeffs[1]) - 0.1

        r_squared = 0.0  # Simplified
        lower_bound = predicted_perm * 0.9
        upper_bound = predicted_perm * 1.1

    # Current value
    current_perm = df_forecast["Permeability TC"].iloc[-1]
    current_time_stamp = df_forecast["TimeStamp"].iloc[-1]
    prediction_date = current_time_stamp + pd.Timedelta(days=horizon_days)

    # Calculate time to critical threshold (e.g., 50% decline from baseline)
    # In practice, permeability typically declines, so lower threshold is a concern
    # Assuming typical range is 300-600 LMH/bar, warning threshold could be 200
    threshold_perm = 200.0  # L/m²/h/bar - can be configured
    if model_type == "linear" and slope < 0:  # Declining permeability
        time_to_threshold_seconds = (
            threshold_perm - (slope * time_numeric[-1] + intercept)
        ) / slope
        time_to_threshold_days = (time_to_threshold_seconds - time_numeric[-1]) / 86400
        threshold_date = current_time_stamp + pd.Timedelta(days=time_to_threshold_days)
    elif model_type == "exponential" and slope < 0:
        # For exponential: solve exp(slope*t + intercept) - 0.1 = threshold_perm
        # t = (ln(threshold_perm + 0.1) - intercept) / slope
        time_to_threshold_seconds = (np.log(threshold_perm + 0.1) - intercept) / slope
        time_to_threshold_days = (time_to_threshold_seconds - time_numeric[-1]) / 86400
        threshold_date = current_time_stamp + pd.Timedelta(days=time_to_threshold_days)
    else:
        time_to_threshold_days = None
        threshold_date = None

    # Create message
    if predicted_perm > threshold_perm:
        message = f"Permeability TC forecasted to reach {predicted_perm:.1f} LMH/bar in {horizon_days} days (within acceptable range)"
    else:
        message = f"WARNING: Permeability TC forecasted to decline to {predicted_perm:.1f} LMH/bar within {horizon_days} days"

    forecast = Forecast(
        parameter="Permeability TC",
        model_type=model_type,
        forecast_horizon_days=horizon_days,
        current_value=current_perm,
        predicted_value=predicted_perm,
        prediction_date=prediction_date,
        confidence_level=confidence,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        r_squared=r_squared,
        message=message,
        time_to_threshold=time_to_threshold_days,
        threshold_value=threshold_perm,
        threshold_date=threshold_date,
    )

    logger.info(
        f"✓ Permeability TC forecast: {predicted_perm:.1f} LMH/bar in {horizon_days} days"
    )
    if time_to_threshold_days and time_to_threshold_days > 0:
        logger.info(
            f"  Time to {threshold_perm} LMH/bar threshold: {time_to_threshold_days:.1f} days"
        )

    return forecast


def run_kpi_engine(
    config_path: str = "config.yaml",
) -> Tuple[List[KPI], List[Alert], Dict]:
    """
    Main KPI engine pipeline

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (KPIs list, Alerts list, Forecast dictionary with 'tmp' and 'permeability' keys)
    """
    logger.info("=" * 60)
    logger.info("📊 KPI ENGINE STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    cycles_data_path = config["paths"]["cycles_data"]
    kpi_config = config["kpis"]
    forecast_config = config["forecasting"]

    # Load data
    logger.info(f"Loading data from: {cycles_data_path}")
    df = load_parquet(cycles_data_path)

    # Load cycle summary
    cycle_summary_path = config["paths"]["cycle_summary_data"]
    try:
        cycle_summary_df = load_parquet(cycle_summary_path)
        cycles = cycle_summary_df.to_dict("records")
    except:
        cycles = []
        logger.warning("No cycle summary available")

    # Load fouling metadata
    try:
        with open("outputs/fouling_metadata.json", "r") as f:
            fouling_metadata = json.load(f)
    except:
        fouling_metadata = {}
        logger.warning("No fouling metadata available")

    # Calculate KPIs
    targets = kpi_config["targets"]
    kpis = calculate_kpis(df, cycles, targets)

    # Generate alerts
    alert_config = kpi_config["alerts"]
    alerts = generate_alerts(df, cycles, fouling_metadata, alert_config)

    # Create TMP forecast
    tmp_forecast_config = forecast_config["tmp_forecast"]
    tmp_model_type = tmp_forecast_config["model"]
    tmp_forecast = None

    # Check if ML model is requested
    if tmp_model_type.startswith("ml_") and ML_AVAILABLE:
        # Use ML forecasting
        ml_model_type = tmp_model_type.replace(
            "ml_", ""
        )  # e.g., "ml_random_forest" -> "random_forest"
        ml_settings = tmp_forecast_config.get("ml_settings", {})

        tmp_forecast_dict = create_ml_tmp_forecast(
            df,
            model_type=ml_model_type,
            horizon_days=pd.Timedelta(
                tmp_forecast_config["forecast_horizon"]
            ).total_seconds()
            / 86400,
            test_size=ml_settings.get("test_size", 0.2),
            confidence=tmp_forecast_config["confidence_level"],
            start_date=tmp_forecast_config.get("data_start", None),
            include_additional_features=ml_settings.get(
                "include_additional_features", True
            ),
        )

        # Convert dict to Forecast object for compatibility
        if tmp_forecast_dict:
            tmp_forecast = Forecast(
                parameter=tmp_forecast_dict["parameter"],
                model_type=tmp_forecast_dict["model_type"],
                forecast_horizon_days=tmp_forecast_dict["forecast_horizon_days"],
                current_value=tmp_forecast_dict["current_value"],
                predicted_value=tmp_forecast_dict["predicted_value"],
                prediction_date=tmp_forecast_dict["prediction_date"],
                confidence_level=tmp_forecast_dict["confidence_level"],
                lower_bound=tmp_forecast_dict["lower_bound"],
                upper_bound=tmp_forecast_dict["upper_bound"],
                r_squared=tmp_forecast_dict["r_squared"],
                message=tmp_forecast_dict["message"],
                time_to_threshold=tmp_forecast_dict.get("time_to_threshold"),
                threshold_value=tmp_forecast_dict.get("threshold_value"),
                threshold_date=tmp_forecast_dict.get("threshold_date"),
            )
    elif tmp_model_type.startswith("ml_") and not ML_AVAILABLE:
        logger.warning(
            "ML model requested but ML packages not available. Install with: pip install scikit-learn xgboost"
        )
        logger.warning("Falling back to exponential model")
        # Fallback to simple exponential model
        tmp_forecast = create_tmp_forecast(
            df,
            model_type="exponential",
            horizon_days=pd.Timedelta(
                tmp_forecast_config["forecast_horizon"]
            ).total_seconds()
            / 86400,
            confidence=tmp_forecast_config["confidence_level"],
            start_date=tmp_forecast_config.get("data_start", None),
        )
    else:
        # Use simple forecasting (linear/exponential)
        tmp_forecast = create_tmp_forecast(
            df,
            model_type=tmp_model_type,
            horizon_days=pd.Timedelta(
                tmp_forecast_config["forecast_horizon"]
            ).total_seconds()
            / 86400,
            confidence=tmp_forecast_config["confidence_level"],
            start_date=tmp_forecast_config.get("data_start", None),
        )

    # Create Permeability TC forecast
    perm_forecast_config = forecast_config.get("permeability_forecast", {})
    perm_forecast = None

    if perm_forecast_config.get("enabled", False):
        perm_model_type = perm_forecast_config.get("model", "linear")

        # Check if ML model is requested
        if perm_model_type.startswith("ml_") and ML_AVAILABLE:
            # Use ML forecasting
            ml_model_type = perm_model_type.replace("ml_", "")
            ml_settings = perm_forecast_config.get("ml_settings", {})

            perm_forecast_dict = create_ml_permeability_forecast(
                df,
                model_type=ml_model_type,
                horizon_days=pd.Timedelta(
                    perm_forecast_config.get("forecast_horizon", "7d")
                ).total_seconds()
                / 86400,
                test_size=ml_settings.get("test_size", 0.2),
                confidence=perm_forecast_config.get("confidence_level", 0.95),
                start_date=perm_forecast_config.get("data_start", None),
                include_additional_features=ml_settings.get(
                    "include_additional_features", True
                ),
            )

            # Convert dict to Forecast object for compatibility
            if perm_forecast_dict:
                perm_forecast = Forecast(
                    parameter=perm_forecast_dict["parameter"],
                    model_type=perm_forecast_dict["model_type"],
                    forecast_horizon_days=perm_forecast_dict["forecast_horizon_days"],
                    current_value=perm_forecast_dict["current_value"],
                    predicted_value=perm_forecast_dict["predicted_value"],
                    prediction_date=perm_forecast_dict["prediction_date"],
                    confidence_level=perm_forecast_dict["confidence_level"],
                    lower_bound=perm_forecast_dict["lower_bound"],
                    upper_bound=perm_forecast_dict["upper_bound"],
                    r_squared=perm_forecast_dict["r_squared"],
                    message=perm_forecast_dict["message"],
                    time_to_threshold=perm_forecast_dict.get("time_to_threshold"),
                    threshold_value=perm_forecast_dict.get("threshold_value"),
                    threshold_date=perm_forecast_dict.get("threshold_date"),
                )
        elif perm_model_type.startswith("ml_") and not ML_AVAILABLE:
            logger.warning(
                "ML model requested but ML packages not available. Falling back to linear model"
            )
            # Fallback to simple linear model
            perm_forecast = create_permeability_forecast(
                df,
                model_type="linear",
                horizon_days=pd.Timedelta(
                    perm_forecast_config.get("forecast_horizon", "7d")
                ).total_seconds()
                / 86400,
                confidence=perm_forecast_config.get("confidence_level", 0.95),
                start_date=perm_forecast_config.get("data_start", None),
            )
        else:
            # Use simple forecasting (linear/exponential)
            perm_forecast = create_permeability_forecast(
                df,
                model_type=perm_model_type,
                horizon_days=pd.Timedelta(
                    perm_forecast_config.get("forecast_horizon", "7d")
                ).total_seconds()
                / 86400,
                confidence=perm_forecast_config.get("confidence_level", 0.95),
                start_date=perm_forecast_config.get("data_start", None),
            )

    # Combine forecasts into a dictionary
    forecasts = {"tmp": tmp_forecast, "permeability": perm_forecast}

    # Save outputs
    kpis_output = config["paths"]["kpis_json"]
    with open(kpis_output, "w") as f:
        json.dump([kpi.to_dict() for kpi in kpis], f, indent=2, default=str)
    logger.info(f"✓ Saved KPIs: {kpis_output}")

    alerts_output = config["paths"]["alerts_json"]
    with open(alerts_output, "w") as f:
        json.dump([alert.to_dict() for alert in alerts], f, indent=2, default=str)
    logger.info(f"✓ Saved alerts: {alerts_output}")

    # Save forecasts
    if tmp_forecast:
        forecast_output = config["paths"]["forecast_json"]
        with open(forecast_output, "w") as f:
            json.dump(tmp_forecast.to_dict(), f, indent=2, default=str)
        logger.info(f"✓ Saved TMP forecast: {forecast_output}")

    if perm_forecast:
        perm_forecast_output = config["paths"].get(
            "permeability_forecast_json", "outputs/07_permeability_forecast.json"
        )
        with open(perm_forecast_output, "w") as f:
            json.dump(perm_forecast.to_dict(), f, indent=2, default=str)
        logger.info(f"✓ Saved Permeability forecast: {perm_forecast_output}")

    # Summary
    logger.info(f"\n📈 Summary:")
    logger.info(f"  KPIs: {len(kpis)}")
    logger.info(
        f"  Alerts: {len(alerts)} ({sum(1 for a in alerts if a.severity == 'critical')} critical)"
    )
    logger.info(f"  TMP Forecast available: {tmp_forecast is not None}")
    logger.info(f"  Permeability Forecast available: {perm_forecast is not None}")

    logger.info("=" * 60)
    logger.info("✓ KPI ENGINE COMPLETE")
    logger.info("=" * 60)

    return kpis, alerts, forecasts


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run KPI engine
    kpis, alerts, forecasts = run_kpi_engine()
    print(f"\n✓ Calculated {len(kpis)} KPIs")
    print(f"✓ Generated {len(alerts)} alerts")
    if forecasts.get("tmp"):
        print(f"✓ TMP Forecast: {forecasts['tmp'].message}")
    if forecasts.get("permeability"):
        print(f"✓ Permeability Forecast: {forecasts['permeability'].message}")
