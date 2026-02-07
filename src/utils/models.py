"""
Data models for pipeline objects
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd


@dataclass
class ValidationReport:
    """Report from validation stage"""

    total_rows_input: int
    total_rows_output: int
    rows_removed: int
    outliers_per_parameter: Dict[str, int]
    missing_values: Dict[str, float]
    timestamp_gaps: List[tuple]
    time_range_start: pd.Timestamp
    time_range_end: pd.Timestamp
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "total_rows_input": self.total_rows_input,
            "total_rows_output": self.total_rows_output,
            "rows_removed": self.rows_removed,
            "outliers_per_parameter": self.outliers_per_parameter,
            "missing_values": self.missing_values,
            "timestamp_gaps": [
                {
                    "start": str(gap[0]),
                    "end": str(gap[1]),
                    "minutes": gap[2],
                }
                for gap in self.timestamp_gaps
            ],
            "time_range_start": str(self.time_range_start),
            "time_range_end": str(self.time_range_end),
            "warnings": self.warnings,
        }


@dataclass
class Cycle:
    """Represents a filtration cycle between cleanings"""

    cycle_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: float

    # Performance metrics
    tmp_start: float
    tmp_end: float
    tmp_slope: float  # bar/hour
    tmp_slope_r2: float

    permeability_start: float
    permeability_end: float
    permeability_decline_percent: float

    avg_flux: float
    avg_recovery: float
    avg_retention: float

    total_permeate_volume: Optional[float] = None
    total_feed_volume: Optional[float] = None
    avg_specific_energy: Optional[float] = None

    # Cleaning info
    cleaning_detected: bool = False
    cleaning_type: Optional[str] = None  # "chemical", "backwash", etc.
    tmp_recovery_percent: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "cycle_id": self.cycle_id,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "duration_hours": self.duration_hours,
            "tmp_start": self.tmp_start,
            "tmp_end": self.tmp_end,
            "tmp_slope": self.tmp_slope,
            "tmp_slope_r2": self.tmp_slope_r2,
            "permeability_start": self.permeability_start,
            "permeability_end": self.permeability_end,
            "permeability_decline_percent": self.permeability_decline_percent,
            "avg_flux": self.avg_flux,
            "avg_recovery": self.avg_recovery,
            "avg_retention": self.avg_retention,
            "total_permeate_volume": self.total_permeate_volume,
            "total_feed_volume": self.total_feed_volume,
            "avg_specific_energy": self.avg_specific_energy,
            "cleaning_detected": self.cleaning_detected,
            "cleaning_type": self.cleaning_type,
            "tmp_recovery_percent": self.tmp_recovery_percent,
            "metadata": self.metadata,
        }


@dataclass
class Alert:
    """Represents an alert condition"""

    alert_type: str  # "tmp_slope", "permeability_decline", etc.
    severity: str  # "warning", "critical"
    timestamp: pd.Timestamp
    value: float
    threshold: float
    message: str
    cycle_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "timestamp": str(self.timestamp),
            "value": self.value,
            "threshold": self.threshold,
            "message": self.message,
            "cycle_id": self.cycle_id,
        }


@dataclass
class KPI:
    """Key Performance Indicator"""

    name: str
    value: float
    unit: str
    target: Optional[float] = None
    status: Optional[str] = None  # "good", "warning", "critical"
    trend: Optional[str] = None  # "improving", "stable", "declining"
    timestamp: Optional[pd.Timestamp] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "target": self.target,
            "status": self.status,
            "trend": self.trend,
            "timestamp": str(self.timestamp) if self.timestamp else None,
        }


@dataclass
class Forecast:
    """Forecast prediction"""

    parameter: str  # "TMP", "Permeability", etc.
    model_type: str  # "linear", "exponential", etc.
    forecast_horizon_days: float

    # Current value and prediction
    current_value: float
    predicted_value: float
    prediction_date: pd.Timestamp

    # Uncertainty
    confidence_level: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None

    # Model quality
    r_squared: Optional[float] = None

    # Interpretation
    message: str = ""

    # Time to threshold (e.g., when will TMP reach 8 bar?)
    time_to_threshold: Optional[float] = None  # days
    threshold_value: Optional[float] = None
    threshold_date: Optional[pd.Timestamp] = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "parameter": self.parameter,
            "model_type": self.model_type,
            "forecast_horizon_days": self.forecast_horizon_days,
            "current_value": self.current_value,
            "predicted_value": self.predicted_value,
            "prediction_date": str(self.prediction_date),
            "confidence_level": self.confidence_level,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "r_squared": self.r_squared,
            "message": self.message,
            "time_to_threshold": self.time_to_threshold,
            "threshold_value": self.threshold_value,
            "threshold_date": str(self.threshold_date) if self.threshold_date else None,
        }
