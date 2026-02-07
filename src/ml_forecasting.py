"""
MACHINE LEARNING FORECASTING MODULE
Advanced TMP prediction using ML models
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import xgboost as xgb

    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning(
        "Machine learning packages not available. Run: pip install scikit-learn xgboost"
    )


def create_time_features(
    df: pd.DataFrame, timestamp_col: str = "TimeStamp"
) -> pd.DataFrame:
    """
    Create time-based features for ML models

    Args:
        df: Input DataFrame with timestamp column
        timestamp_col: Name of timestamp column

    Returns:
        DataFrame with additional time features
    """
    df = df.copy()

    # Extract time components
    df["hour"] = df[timestamp_col].dt.hour
    df["day_of_week"] = df[timestamp_col].dt.dayofweek
    df["day_of_month"] = df[timestamp_col].dt.day
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Cyclical encoding for hour (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Time since start (seconds)
    df["time_numeric"] = (
        df[timestamp_col] - df[timestamp_col].min()
    ).dt.total_seconds()

    return df


def create_lag_features(
    df: pd.DataFrame, target_col: str, lags: list = [1, 6, 12, 24, 48]
) -> pd.DataFrame:
    """
    Create lag features for time series prediction

    Args:
        df: Input DataFrame
        target_col: Target column to create lags for
        lags: List of lag periods (in number of rows)

    Returns:
        DataFrame with lag features
    """
    df = df.copy()

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

    return df


def create_rolling_features(
    df: pd.DataFrame, target_col: str, windows: list = [12, 24, 72]
) -> pd.DataFrame:
    """
    Create rolling statistical features

    Args:
        df: Input DataFrame
        target_col: Target column to create rolling stats for
        windows: List of window sizes (in number of rows)

    Returns:
        DataFrame with rolling features
    """
    df = df.copy()

    for window in windows:
        df[f"{target_col}_roll_mean_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{target_col}_roll_std_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )
        df[f"{target_col}_roll_min_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).min()
        )
        df[f"{target_col}_roll_max_{window}"] = (
            df[target_col].rolling(window=window, min_periods=1).max()
        )

    return df


def prepare_ml_features(
    df: pd.DataFrame,
    target_col: str = "TMP",
    timestamp_col: str = "TimeStamp",
    include_additional_features: bool = True,
) -> Tuple[pd.DataFrame, list]:
    """
    Prepare features for ML forecasting

    Args:
        df: Input DataFrame
        target_col: Target column to predict
        timestamp_col: Timestamp column name
        include_additional_features: Include permeability, flux, etc. if available

    Returns:
        Tuple of (feature DataFrame, list of feature column names)
    """
    logger.info("Preparing ML features for forecasting...")

    df_features = df.copy()

    # Time features
    df_features = create_time_features(df_features, timestamp_col)

    # Lag features (assuming ~12 points per hour with 5-second sampling)
    lags = [1, 12, 72, 144, 288]  # 5s, 1min, 6min, 12min, 24min
    df_features = create_lag_features(df_features, target_col, lags)

    # Rolling features
    windows = [12, 72, 288]  # 1min, 6min, 24min windows
    df_features = create_rolling_features(df_features, target_col, windows)

    # Feature columns (excluding target and timestamp)
    feature_cols = [
        "hour",
        "day_of_week",
        "day_of_month",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "time_numeric",
    ]

    # Add lag features
    feature_cols += [f"{target_col}_lag_{lag}" for lag in lags]

    # Add rolling features
    for window in windows:
        feature_cols += [
            f"{target_col}_roll_mean_{window}",
            f"{target_col}_roll_std_{window}",
            f"{target_col}_roll_min_{window}",
            f"{target_col}_roll_max_{window}",
        ]

    # Include additional features if available
    if include_additional_features:
        additional_cols = ["Permeability TC", "Flux", "Recovery", "Specific_power"]
        for col in additional_cols:
            if col in df_features.columns:
                feature_cols.append(col)
                logger.info(f"  Including feature: {col}")

    # Remove rows with NaN values (from lag/rolling operations)
    df_features = df_features.dropna()

    logger.info(f"  Created {len(feature_cols)} features")
    logger.info(f"  Training samples: {len(df_features)}")

    return df_features, feature_cols


def train_ml_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
    **model_params,
) -> object:
    """
    Train ML model for forecasting

    Args:
        X_train: Training features
        y_train: Training target
        model_type: Model type ('random_forest', 'gradient_boosting', 'xgboost')
        model_params: Additional model parameters

    Returns:
        Trained model
    """
    if not ML_AVAILABLE:
        raise ImportError(
            "ML packages not installed. Run: pip install scikit-learn xgboost"
        )

    logger.info(f"Training {model_type} model...")

    # Default parameters
    default_params = {
        "random_forest": {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
        "gradient_boosting": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
        },
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        },
    }

    # Merge default params with user params
    params = {**default_params.get(model_type, {}), **model_params}

    # Create and train model
    if model_type == "random_forest":
        model = RandomForestRegressor(**params)
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(**params)
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(**params)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Use 'random_forest', 'gradient_boosting', or 'xgboost'"
        )

    model.fit(X_train, y_train)

    logger.info(f"  ✓ Model trained with {len(X_train)} samples")

    return model


def create_ml_tmp_forecast(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    horizon_days: float = 7,
    test_size: float = 0.2,
    confidence: float = 0.95,
    start_date: str = None,
    include_additional_features: bool = True,
) -> Dict:
    """
    Create TMP forecast using machine learning

    Args:
        df: DataFrame with TMP and other features
        model_type: ML model type ('random_forest', 'gradient_boosting', 'xgboost')
        horizon_days: Forecast horizon in days
        test_size: Fraction of data for testing
        confidence: Confidence level for prediction interval
        start_date: Optional start date to filter data from
        include_additional_features: Use permeability, flux, etc. if available

    Returns:
        Dictionary with forecast results
    """
    if not ML_AVAILABLE:
        logger.error("ML packages not installed. Falling back to simple forecasting.")
        return None

    logger.info(
        f"Creating ML-based TMP forecast ({model_type}, {horizon_days} days ahead)..."
    )

    # Filter data if start_date provided
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df["TimeStamp"] >= start_dt]
        logger.info(f"Filtered forecast data from {start_date}: {len(df)} points")

    if len(df) < 500:
        logger.warning("Insufficient data points for reliable ML forecast")
        return None

    # Prepare features
    df_ml, feature_cols = prepare_ml_features(
        df, target_col="TMP", include_additional_features=include_additional_features
    )

    # Split into train/test
    train_idx = int(len(df_ml) * (1 - test_size))
    df_train = df_ml.iloc[:train_idx]
    df_test = df_ml.iloc[train_idx:]

    X_train = df_train[feature_cols]
    y_train = df_train["TMP"]
    X_test = df_test[feature_cols]
    y_test = df_test["TMP"]

    # Train model
    model = train_ml_model(X_train, y_train, model_type)

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)

    logger.info(f"  Model performance:")
    logger.info(f"    R² score: {r2:.4f}")
    logger.info(f"    RMSE: {rmse:.4f} bar")
    logger.info(f"    MAE: {mae:.4f} bar")

    # Forecast future
    last_timestamp = df_ml["TimeStamp"].iloc[-1]
    current_tmp = df_ml["TMP"].iloc[-1]

    # For simplicity, predict forward using iterative approach
    # (In production, you'd want to use proper recursive forecasting)

    # Use last available data point for prediction
    X_forecast = df_ml[feature_cols].iloc[[-1]].copy()

    # Update time_numeric for future prediction
    seconds_ahead = horizon_days * 86400
    X_forecast["time_numeric"] = X_forecast["time_numeric"] + seconds_ahead

    # Update time features for prediction date
    future_date = last_timestamp + pd.Timedelta(days=horizon_days)
    X_forecast["hour"] = future_date.hour
    X_forecast["day_of_week"] = future_date.dayofweek
    X_forecast["day_of_month"] = future_date.day
    X_forecast["is_weekend"] = int(future_date.dayofweek >= 5)
    X_forecast["hour_sin"] = np.sin(2 * np.pi * future_date.hour / 24)
    X_forecast["hour_cos"] = np.cos(2 * np.pi * future_date.hour / 24)

    # Predict
    predicted_tmp = model.predict(X_forecast)[0]

    # Confidence bounds (using test set residuals)
    residuals = y_test - y_pred_test
    residual_std = residuals.std()
    z_score = 1.96  # 95% confidence
    margin = z_score * residual_std

    lower_bound = predicted_tmp - margin
    upper_bound = predicted_tmp + margin

    # Calculate time to threshold (8 bar)
    threshold_tmp = 8.0

    # Simple linear extrapolation from current to predicted
    if predicted_tmp > current_tmp:
        rate = (predicted_tmp - current_tmp) / horizon_days
        days_to_threshold = (threshold_tmp - current_tmp) / rate if rate > 0 else None
        threshold_date = (
            last_timestamp + pd.Timedelta(days=days_to_threshold)
            if days_to_threshold
            else None
        )
    else:
        days_to_threshold = None
        threshold_date = None

    # Feature importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info("  Top 5 important features:")
        for idx, row in importance.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    # Create message
    if predicted_tmp < threshold_tmp:
        message = f"ML forecast: TMP to reach {predicted_tmp:.2f} bar in {horizon_days} days (R²={r2:.3f})"
    else:
        message = f"ML WARNING: TMP forecasted to exceed {threshold_tmp} bar within {horizon_days} days (R²={r2:.3f})"

    result = {
        "parameter": "TMP",
        "model_type": f"ml_{model_type}",
        "forecast_horizon_days": horizon_days,
        "current_value": float(current_tmp),
        "predicted_value": float(predicted_tmp),
        "prediction_date": str(future_date),
        "confidence_level": confidence,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "r_squared": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "message": message,
        "time_to_threshold": float(days_to_threshold) if days_to_threshold else None,
        "threshold_value": threshold_tmp,
        "threshold_date": str(threshold_date) if threshold_date else None,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(feature_cols),
    }

    logger.info(f"  ✓ ML forecast: {predicted_tmp:.2f} bar in {horizon_days} days")
    if days_to_threshold:
        logger.info(f"    Time to 8 bar threshold: {days_to_threshold:.1f} days")

    return result


def create_ml_permeability_forecast(
    df: pd.DataFrame,
    model_type: str = "random_forest",
    horizon_days: float = 7,
    test_size: float = 0.2,
    confidence: float = 0.95,
    start_date: str = None,
    include_additional_features: bool = True,
) -> Dict:
    """
    Create Permeability TC forecast using machine learning

    Args:
        df: DataFrame with Permeability TC and other features
        model_type: ML model type ('random_forest', 'gradient_boosting', 'xgboost')
        horizon_days: Forecast horizon in days
        test_size: Fraction of data for testing
        confidence: Confidence level for prediction interval
        start_date: Optional start date to filter data from
        include_additional_features: Use TMP, flux, etc. if available

    Returns:
        Dictionary with forecast results
    """
    if not ML_AVAILABLE:
        logger.error("ML packages not installed. Falling back to simple forecasting.")
        return None

    logger.info(
        f"Creating ML-based Permeability TC forecast ({model_type}, {horizon_days} days ahead)..."
    )

    # Filter data if start_date provided
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df = df[df["TimeStamp"] >= start_dt]
        logger.info(f"Filtered forecast data from {start_date}: {len(df)} points")

    if len(df) < 500:
        logger.warning("Insufficient data points for reliable ML forecast")
        return None

    # Prepare features
    df_ml, feature_cols = prepare_ml_features(
        df,
        target_col="Permeability TC",
        include_additional_features=include_additional_features,
    )

    # Split into train/test
    train_idx = int(len(df_ml) * (1 - test_size))
    df_train = df_ml.iloc[:train_idx]
    df_test = df_ml.iloc[train_idx:]

    X_train = df_train[feature_cols]
    y_train = df_train["Permeability TC"]
    X_test = df_test[feature_cols]
    y_test = df_test["Permeability TC"]

    # Train model
    model = train_ml_model(X_train, y_train, model_type)

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)

    logger.info(f"  Model performance:")
    logger.info(f"    R² score: {r2:.4f}")
    logger.info(f"    RMSE: {rmse:.4f} LMH/bar")
    logger.info(f"    MAE: {mae:.4f} LMH/bar")

    # Forecast future
    last_timestamp = df_ml["TimeStamp"].iloc[-1]
    current_perm = df_ml["Permeability TC"].iloc[-1]

    # For simplicity, predict forward using iterative approach
    # (In production, you'd want to use proper recursive forecasting)

    # Use last available data point for prediction
    X_forecast = df_ml[feature_cols].iloc[[-1]].copy()

    # Update time_numeric for future prediction
    seconds_ahead = horizon_days * 86400
    X_forecast["time_numeric"] = X_forecast["time_numeric"] + seconds_ahead

    # Update time features for prediction date
    future_date = last_timestamp + pd.Timedelta(days=horizon_days)
    X_forecast["hour"] = future_date.hour
    X_forecast["day_of_week"] = future_date.dayofweek
    X_forecast["day_of_month"] = future_date.day
    X_forecast["is_weekend"] = int(future_date.dayofweek >= 5)
    X_forecast["hour_sin"] = np.sin(2 * np.pi * future_date.hour / 24)
    X_forecast["hour_cos"] = np.cos(2 * np.pi * future_date.hour / 24)

    # Predict
    predicted_perm = model.predict(X_forecast)[0]

    # Confidence bounds (using test set residuals)
    residuals = y_test - y_pred_test
    residual_std = residuals.std()
    z_score = 1.96  # 95% confidence
    margin = z_score * residual_std

    lower_bound = predicted_perm - margin
    upper_bound = predicted_perm + margin

    # Calculate time to threshold (200 LMH/bar - critical permeability)
    threshold_perm = 200.0

    # Simple linear extrapolation from current to predicted
    if predicted_perm < current_perm:  # Declining permeability
        rate = (predicted_perm - current_perm) / horizon_days
        days_to_threshold = (threshold_perm - current_perm) / rate if rate < 0 else None
        threshold_date = (
            last_timestamp + pd.Timedelta(days=days_to_threshold)
            if days_to_threshold
            else None
        )
    else:
        days_to_threshold = None
        threshold_date = None

    # Feature importance (for tree-based models)
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        logger.info("  Top 5 important features:")
        for idx, row in importance.head(5).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

    # Create message
    if predicted_perm > threshold_perm:
        message = f"ML forecast: Permeability TC to reach {predicted_perm:.1f} LMH/bar in {horizon_days} days (R²={r2:.3f})"
    else:
        message = f"ML WARNING: Permeability TC forecasted to decline below {threshold_perm} LMH/bar within {horizon_days} days (R²={r2:.3f})"

    result = {
        "parameter": "Permeability TC",
        "model_type": f"ml_{model_type}",
        "forecast_horizon_days": horizon_days,
        "current_value": float(current_perm),
        "predicted_value": float(predicted_perm),
        "prediction_date": str(future_date),
        "confidence_level": confidence,
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "r_squared": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "message": message,
        "time_to_threshold": float(days_to_threshold) if days_to_threshold else None,
        "threshold_value": threshold_perm,
        "threshold_date": str(threshold_date) if threshold_date else None,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": len(feature_cols),
    }

    logger.info(f"  ✓ ML forecast: {predicted_perm:.1f} LMH/bar in {horizon_days} days")
    if days_to_threshold:
        logger.info(
            f"    Time to {threshold_perm} LMH/bar threshold: {days_to_threshold:.1f} days"
        )

    return result
