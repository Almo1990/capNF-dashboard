"""
[8] DASHBOARD APP MODULE
Generate interactive visualizations and dashboards
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
import logging
import yaml
import json

from .utils.data_io import load_parquet
from .utils.plotting import (
    create_time_axis,
    apply_color_palette,
    get_plot_config,
    create_standard_layout,
)

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Import plotting functions from the original Vcodefast.py
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__ + "/..")))

# Note: We'll adapt the original plotting code inline to avoid circular imports


def create_individual_plots(
    df: pd.DataFrame, plots_dir: str, config: dict, cycles: list = None
) -> list:
    """
    Create individual plots for each parameter

    Args:
        df: Visualization-ready DataFrame (downsampled)
        plots_dir: Output directory for plots
        config: Visualization configuration
        cycles: List of cycle dictionaries for annotations

    Returns:
        List of generated plot files
    """
    logger.info("Creating individual parameter plots...")

    os.makedirs(plots_dir, exist_ok=True)

    # Define the desired order for process parameters
    desired_order = [
        "TMP",
        "Permeability TC",
        "Mem. retention",
        "Sys. retention",
        "Recovery",
        "Net recovery",
        "Specific_power",
        "Flux",
        "Vcrossflow",
        "01-TIT-01",  # Temperature
    ]

    # Define display names for visual titles
    display_names = {
        "Vcrossflow": "Cross flow velocity",
        "Mem. retention": "Membrane retention",
        "Sys. retention": "System retention",
        "01-TIT-01": "Temperature",
    }

    # Get numeric columns (excluding TimeStamp and SMA columns)
    all_numeric_cols = [
        col
        for col in df.columns
        if col != "TimeStamp"
        and pd.api.types.is_numeric_dtype(df[col])
        and not col.endswith("_SMA")
        and not col.startswith("TMP_slope")
    ]

    # Reorder columns based on desired_order, then add any remaining columns
    numeric_cols = []
    for col in desired_order:
        if col in all_numeric_cols:
            numeric_cols.append(col)

    # Add any remaining columns not in desired_order
    for col in all_numeric_cols:
        if col not in numeric_cols:
            numeric_cols.append(col)

    colors = config["colors"]
    temp_color = config["temperature_color"]
    plot_height = config["plot_sizes"]["individual_height"]

    generated_files = []

    for i, col in enumerate(numeric_cols[:10]):  # Limit to 10 main parameters
        fig = go.Figure()

        # Add raw data
        fig.add_trace(
            go.Scatter(
                x=df["TimeStamp"],
                y=df[col],
                mode="lines",
                line=dict(color=apply_color_palette(i, colors), width=1),
                name=f"{col} (Raw)",
                opacity=0.4,
            )
        )

        # Add SMA if available
        sma_col = f"{col}_SMA"
        if sma_col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["TimeStamp"],
                    y=df[sma_col],
                    mode="lines",
                    line=dict(color=apply_color_palette(i, colors), width=2),
                    name=f"{col} (SMA)",
                )
            )

        # Add temperature overlay
        if "01-TIT-01" in df.columns and col != "01-TIT-01":
            temp_sma_col = (
                "01-TIT-01_SMA" if "01-TIT-01_SMA" in df.columns else "01-TIT-01"
            )
            fig.add_trace(
                go.Scatter(
                    x=df["TimeStamp"],
                    y=df[temp_sma_col],
                    mode="lines",
                    line=dict(color=temp_color, width=2),
                    name="Temperature (°C)",
                    yaxis="y2",
                    opacity=0.8,
                )
            )

        # Calculate time range
        time_range = (
            df["TimeStamp"].max() - df["TimeStamp"].min()
            if "TimeStamp" in df.columns
            else None
        )

        # Get display name for title
        display_col = display_names.get(col, col)

        # Update layout
        layout = create_standard_layout(f"{display_col} vs Time", height=plot_height)
        layout.update(
            xaxis=create_time_axis(time_range, True),
            yaxis=dict(
                title=dict(text=display_col, font=dict(size=18)),
                gridcolor="rgba(200, 200, 200, 0.3)",
                showgrid=True,
                fixedrange=False,
            ),
            yaxis2=dict(
                title=dict(
                    text="Temperature (°C)", font=dict(size=16, color=temp_color)
                ),
                overlaying="y",
                side="right",
                showgrid=False,
                title_font=dict(color=temp_color),
                tickfont=dict(color=temp_color),
                fixedrange=True,
            ),
        )
        fig.update_layout(layout)

        # Save plot
        filename = f"{col.replace(' ', '_').replace('.', '_')}.html"
        filepath = os.path.join(plots_dir, filename)

        fig.write_html(
            filepath,
            config=get_plot_config(),
            include_plotlyjs="cdn",
        )

        generated_files.append(filepath)
        logger.info(f"  ✓ Created: {col}")

    return generated_files


def create_combined_parameters_plot(
    df: pd.DataFrame, plots_dir: str, config: dict, cycles: list = None
) -> str:
    """
    Create combined plot with all parameters (except temperature) on primary axis
    and temperature on secondary axis

    Args:
        df: Visualization-ready DataFrame (downsampled)
        plots_dir: Output directory for plots
        config: Visualization configuration
        cycles: List of cycle dictionaries for annotations

    Returns:
        Path to generated file
    """
    logger.info("Creating combined parameters plot...")

    # Define display names for visual titles
    display_names = {
        "Vcrossflow": "Cross flow velocity",
        "Mem. retention": "Membrane retention",
        "Sys. retention": "System retention",
        "01-TIT-01": "Temperature",
    }

    # Get numeric columns (excluding TimeStamp, SMA columns, and temperature)
    numeric_cols = [
        col
        for col in df.columns
        if col != "TimeStamp"
        and col != "01-TIT-01"  # Exclude temperature itself
        and pd.api.types.is_numeric_dtype(df[col])
        and not col.endswith("_SMA")
        and not col.startswith("TMP_slope")
    ][:10]  # Limit to 10 main parameters

    colors = config["colors"]
    temp_color = config["temperature_color"]
    plot_height = config["plot_sizes"]["combined_height"]

    fig = go.Figure()

    # Add all parameters with SMA (smoothed version preferred)
    for i, col in enumerate(numeric_cols):
        # Use SMA if available, otherwise raw data
        sma_col = f"{col}_SMA"
        display_col = display_names.get(col, col)
        if sma_col in df.columns:
            y_data = df[sma_col]
            trace_name = display_col
        else:
            y_data = df[col]
            trace_name = f"{display_col} (Raw)"

        fig.add_trace(
            go.Scatter(
                x=df["TimeStamp"],
                y=y_data,
                mode="lines",
                line=dict(color=apply_color_palette(i, colors), width=2),
                name=trace_name,
                visible=True if i < 5 else "legendonly",  # Show first 5 by default
            )
        )

    # Add temperature on secondary axis
    if "01-TIT-01" in df.columns:
        temp_col = "01-TIT-01_SMA" if "01-TIT-01_SMA" in df.columns else "01-TIT-01"
        fig.add_trace(
            go.Scatter(
                x=df["TimeStamp"],
                y=df[temp_col],
                mode="lines",
                line=dict(color=temp_color, width=3),
                name="Temperature (°C)",
                yaxis="y2",
            )
        )

    # Calculate time range
    time_range = (
        df["TimeStamp"].max() - df["TimeStamp"].min()
        if "TimeStamp" in df.columns
        else None
    )

    # Update layout
    layout = create_standard_layout(
        "Combined Process Parameters with Temperature", height=plot_height
    )
    layout.update(
        xaxis=create_time_axis(time_range, True),
        yaxis=dict(
            title=dict(text="Process Parameters (Various Units)", font=dict(size=18)),
            gridcolor="rgba(200, 200, 200, 0.3)",
            showgrid=True,
            fixedrange=False,
        ),
        yaxis2=dict(
            title=dict(text="Temperature (°C)", font=dict(size=16, color=temp_color)),
            overlaying="y",
            side="right",
            showgrid=False,
            title_font=dict(color=temp_color),
            tickfont=dict(color=temp_color),
            fixedrange=True,
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.15,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(0, 0, 0, 0.2)",
            borderwidth=1,
        ),
    )
    fig.update_layout(layout)

    # Save plot
    filepath = os.path.join(plots_dir, "combined_parameters.html")
    fig.write_html(
        filepath,
        config=get_plot_config(),
        include_plotlyjs="cdn",
    )

    logger.info("  ✓ Created: Combined Parameters")
    return filepath


def create_tmp_plot_with_forecast(
    df: pd.DataFrame,
    plots_dir: str,
    config: dict,
    forecast: dict = None,
    cycles: list = None,
    chemical_cleanings: list = None,
    fouling_rates: dict = None,
) -> str:
    """
    Create TMP forecast plot with irreversible fouling trend, dual forecasts,
    chemical cleaning markers, confidence cone, and 6-bar threshold.

    Args:
        df: Full DataFrame (not downsampled) for accurate slope
        plots_dir: Output directory
        config: Full configuration dictionary
        forecast: Forecast dictionary (linear or ML)
        cycles: Cycle information (list of dicts with tmp_start, start_time, etc.)
        chemical_cleanings: List of chemical cleaning timestamp strings
        fouling_rates: Dict with reversible/irreversible fouling rates

    Returns:
        Path to generated file
    """
    logger.info("Creating TMP forecast plot (redesigned)...")

    fig = go.Figure()

    # Use visualization data for background TMP traces
    df_viz_path = config["paths"]["processed_viz_data"]
    df_viz = load_parquet(df_viz_path)
    colors = config["visualization"]["colors"]

    # ── Trace 1: Raw TMP (background context) ───────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df_viz["TimeStamp"],
            y=df_viz["TMP"],
            mode="lines",
            line=dict(color=colors[0], width=1),
            name="TMP (Raw)",
            opacity=0.25,
        )
    )

    # ── Trace 2: TMP SMA (background context) ───────────────────────────
    if "TMP_SMA" in df_viz.columns:
        fig.add_trace(
            go.Scatter(
                x=df_viz["TimeStamp"],
                y=df_viz["TMP_SMA"],
                mode="lines",
                line=dict(color=colors[0], width=1.5),
                name="TMP (SMA)",
                opacity=0.5,
            )
        )

    # ── Extract cycle-start TMP values ───────────────────────────────────
    cycle_times = []
    cycle_tmp_starts = []

    if cycles:
        for c in cycles:
            # Support both dict and object access
            tmp_s = (
                c.get("tmp_start")
                if isinstance(c, dict)
                else getattr(c, "tmp_start", None)
            )
            t_s = (
                c.get("start_time")
                if isinstance(c, dict)
                else getattr(c, "start_time", None)
            )
            if tmp_s is not None and t_s is not None:
                cycle_times.append(pd.to_datetime(t_s))
                cycle_tmp_starts.append(float(tmp_s))

    # IQR outlier filtering on cycle-start TMP values
    valid_cycle_times = []
    valid_cycle_tmps = []
    if len(cycle_tmp_starts) >= 4:
        arr = np.array(cycle_tmp_starts)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for t, v in zip(cycle_times, cycle_tmp_starts):
            if lb <= v <= ub:
                valid_cycle_times.append(t)
                valid_cycle_tmps.append(v)
    else:
        valid_cycle_times = cycle_times
        valid_cycle_tmps = cycle_tmp_starts

    # ── Trace 3: Cycle-start TMP scatter (irreversible fouling envelope) ─
    if valid_cycle_times:
        fig.add_trace(
            go.Scatter(
                x=valid_cycle_times,
                y=valid_cycle_tmps,
                mode="markers",
                marker=dict(color="rgba(255, 140, 0, 0.6)", size=4, symbol="circle"),
                name="Cycle-start TMP",
            )
        )

    # ── Trace 4: Irreversible fouling trend (linear fit on cycle-starts) ─
    irrev_slope_info = {}
    if len(valid_cycle_times) > 2:
        t0 = min(valid_cycle_times)
        cs_seconds = np.array([(t - t0).total_seconds() for t in valid_cycle_times])
        cs_values = np.array(valid_cycle_tmps)
        coeffs = np.polyfit(cs_seconds, cs_values, 1)
        fit_vals = coeffs[0] * cs_seconds + coeffs[1]

        # R²
        y_mean = cs_values.mean()
        ss_tot = np.sum((cs_values - y_mean) ** 2)
        ss_res = np.sum((cs_values - fit_vals) ** 2)
        r_sq = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        irrev_slope_info = {
            "slope": coeffs[0],
            "intercept": coeffs[1],
            "r_squared": r_sq,
            "t0": t0,
        }

        fig.add_trace(
            go.Scatter(
                x=valid_cycle_times,
                y=fit_vals.tolist(),
                mode="lines",
                line=dict(color="black", width=2.5, dash="dash"),
                name="Irreversible Fouling Trend",
            )
        )

    # ── Trace 5: Linear forecast (extrapolation of irreversible trend) ───
    forecast_horizon = forecast.get("forecast_horizon_days", 7) if forecast else 7
    threshold_tmp = 6.0

    if irrev_slope_info and forecast:
        last_time = max(valid_cycle_times)
        pred_time = pd.Timestamp(forecast["prediction_date"])
        t0 = irrev_slope_info["t0"]

        last_sec = (last_time - t0).total_seconds()
        pred_sec = (pred_time - t0).total_seconds()

        fit_at_last = (
            irrev_slope_info["slope"] * last_sec + irrev_slope_info["intercept"]
        )
        fit_at_pred = (
            irrev_slope_info["slope"] * pred_sec + irrev_slope_info["intercept"]
        )

        # Generate intermediate points for smooth line
        n_pts = 20
        interp_secs = np.linspace(last_sec, pred_sec, n_pts)
        interp_times = [t0 + pd.Timedelta(seconds=float(s)) for s in interp_secs]
        interp_vals = (
            irrev_slope_info["slope"] * interp_secs + irrev_slope_info["intercept"]
        ).tolist()

        fig.add_trace(
            go.Scatter(
                x=interp_times,
                y=interp_vals,
                mode="lines",
                line=dict(color="#d62728", width=2.5, dash="dot"),
                name=f"Linear Forecast ({forecast_horizon}d)",
            )
        )

        # ── Trace 6: ML forecast line (if ML data available) ────────────
        # Check if the forecast dict contains ML-specific keys
        ml_forecast_value = forecast.get("predicted_value")
        model_type = forecast.get("model_type", "linear")
        if ml_forecast_value is not None and "ml_" in str(model_type):
            fig.add_trace(
                go.Scatter(
                    x=[last_time, pred_time],
                    y=[fit_at_last, ml_forecast_value],
                    mode="lines+markers",
                    line=dict(color="#9467bd", width=2.5, dash="dashdot"),
                    marker=dict(size=8, symbol="diamond"),
                    name=f"ML Forecast ({model_type.replace('ml_', '')})",
                )
            )

        # ── Trace 7: Confidence cone (shaded area) ──────────────────────
        if "lower_bound" in forecast and "upper_bound" in forecast:
            lower = forecast["lower_bound"]
            upper = forecast["upper_bound"]

            # Build widening cone from fit_at_last (zero width) to bounds at pred_time
            cone_x = []
            cone_y_upper = []
            cone_y_lower = []
            for i in range(n_pts):
                frac = i / (n_pts - 1)
                cone_x.append(interp_times[i])
                cone_y_upper.append(interp_vals[i] + frac * (upper - fit_at_pred))
                cone_y_lower.append(interp_vals[i] + frac * (lower - fit_at_pred))

            # Upper bound line
            fig.add_trace(
                go.Scatter(
                    x=cone_x,
                    y=cone_y_upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Lower bound line + fill between
            fig.add_trace(
                go.Scatter(
                    x=cone_x,
                    y=cone_y_lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(214, 39, 40, 0.12)",
                    name="95% Confidence",
                )
            )

    # ── Trace 8: 6-bar threshold line ────────────────────────────────────
    x_min = df["TimeStamp"].min()
    x_max = (
        pd.Timestamp(forecast["prediction_date"])
        if forecast and "prediction_date" in forecast
        else df["TimeStamp"].max()
    )
    fig.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[threshold_tmp, threshold_tmp],
            mode="lines",
            line=dict(color="rgba(220, 53, 69, 0.7)", width=2, dash="longdash"),
            name="Operational Limit (6 bar)",
        )
    )

    # ── Chemical cleaning vertical lines ─────────────────────────────────
    if chemical_cleanings:
        cleaning_times_dt = [pd.to_datetime(ts) for ts in chemical_cleanings]
        y_range_max = max(valid_cycle_tmps) * 1.3 if valid_cycle_tmps else 6.0
        y_range_min = min(valid_cycle_tmps) * 0.7 if valid_cycle_tmps else 0.0

        for i, ct in enumerate(cleaning_times_dt):
            fig.add_trace(
                go.Scatter(
                    x=[ct, ct],
                    y=[y_range_min, y_range_max],
                    mode="lines",
                    line=dict(color="rgba(40, 167, 69, 0.6)", width=1.5, dash="dash"),
                    name="Chemical Cleaning" if i == 0 else None,
                    showlegend=(i == 0),
                    hovertemplate=f"Chemical Cleaning<br>%{{x}}<extra></extra>",
                )
            )

    # ── Layout ───────────────────────────────────────────────────────────
    time_range = df["TimeStamp"].max() - df["TimeStamp"].min()
    layout = create_standard_layout(
        "TMP Forecast — Irreversible Fouling Trend", height=750
    )
    layout.update(xaxis=create_time_axis(time_range, True))

    # ── Annotation info box ──────────────────────────────────────────────
    annotations = []

    if irrev_slope_info:
        slope_per_hour = irrev_slope_info["slope"] * 3600
        slope_per_day = irrev_slope_info["slope"] * 86400
        r_sq = irrev_slope_info["r_squared"]

        # Fouling classification
        classification = "low"
        if fouling_rates and "reversible_fouling_rate_bar_per_day" in fouling_rates:
            rev_rate = fouling_rates["reversible_fouling_rate_bar_per_day"]
            irrev_rate = fouling_rates.get("irreversible_fouling_rate_bar_per_day")
        else:
            rev_rate = None
            irrev_rate = None

        # Classification colors
        class_colors = {
            "low": "#28a745",
            "medium": "#ffc107",
            "high": "#fd7e14",
            "critical": "#dc3545",
        }

        # Determine fouling classification from slope
        abs_slope_h = abs(slope_per_hour)
        if abs_slope_h < 0.001:
            classification = "low"
        elif abs_slope_h < 0.005:
            classification = "medium"
        elif abs_slope_h < 0.01:
            classification = "high"
        else:
            classification = "critical"

        cls_color = class_colors.get(classification, "#6c757d")

        # Build annotation text
        ann_lines = [
            f'<b style="color:{cls_color}">⬤ Fouling: {classification.upper()}</b>',
            f"<b>Irrev. slope:</b> {slope_per_day:.4f} bar/day (R²={r_sq:.3f})",
        ]
        if rev_rate is not None:
            ann_lines.append(f"<b>Rev. slope:</b> {rev_rate:.4f} bar/day")

        # Time to threshold
        if irrev_slope_info["slope"] > 0:
            current_fit = (
                irrev_slope_info["slope"]
                * (max(valid_cycle_times) - irrev_slope_info["t0"]).total_seconds()
                + irrev_slope_info["intercept"]
            )
            days_to_8 = (threshold_tmp - current_fit) / (
                irrev_slope_info["slope"] * 86400
            )
            if days_to_8 > 0:
                ann_lines.append(f"<b>Time to 6 bar:</b> {days_to_8:.0f} days")
                threshold_date = max(valid_cycle_times) + pd.Timedelta(days=days_to_8)
                ann_lines.append(
                    f"<b>Threshold date:</b> {threshold_date.strftime('%Y-%m-%d')}"
                )

        model_type = forecast.get("model_type", "linear") if forecast else "linear"
        ann_lines.append(f"<b>Model:</b> {model_type}")

        annotations.append(
            dict(
                x=0.98,
                y=0.98,
                xref="paper",
                yref="paper",
                text="<br>".join(ann_lines),
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.92)",
                bordercolor="rgba(0, 0, 0, 0.3)",
                borderwidth=1,
                borderpad=8,
                font=dict(size=11, family="Arial, sans-serif"),
                align="left",
                xanchor="right",
                yanchor="top",
            )
        )

    layout.update(annotations=annotations)
    fig.update_layout(layout)

    # ── Save with custom JavaScript for dynamic recalculation ────────────
    html_str = fig.to_html(config=get_plot_config(), include_plotlyjs="cdn")

    # Prepare cycle-start data for JavaScript dynamic recalculation
    cs_timestamps_js = [str(t) for t in valid_cycle_times]
    cs_tmps_js = valid_cycle_tmps
    cleaning_timestamps_js = chemical_cleanings if chemical_cleanings else []
    model_type_js = forecast.get("model_type", "linear") if forecast else "linear"

    custom_js = f"""
<script>
    // Cycle-start data for dynamic regression
    const csTimestamps = {cs_timestamps_js};
    const csTMPs = {cs_tmps_js};
    const forecastHorizonDays = {forecast_horizon};
    const originalModelType = "{model_type_js}";
    const thresholdTMP = {threshold_tmp};

    // Convert to numeric (milliseconds)
    const csTimeNumeric = csTimestamps.map(t => new Date(t).getTime());

    // Linear regression function
    function linearRegression(x, y) {{
        const n = x.length;
        if (n < 2) return null;
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        const denom = n * sumX2 - sumX * sumX;
        if (Math.abs(denom) < 1e-15) return null;
        const slope = (n * sumXY - sumX * sumY) / denom;
        const intercept = (sumY - slope * sumX) / n;
        const yMean = sumY / n;
        const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
        const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - (slope * x[i] + intercept), 2), 0);
        const r2 = ssTot > 0 ? 1 - (ssRes / ssTot) : 0;
        // Residual std
        const residuals = y.map((yi, i) => yi - (slope * x[i] + intercept));
        const resStd = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / Math.max(n - 2, 1));
        return {{slope, intercept, r2, resStd}};
    }}

    function updateForecast(xRange) {{
        let startTime, endTime;
        if (xRange && xRange['xaxis.range[0]'] && xRange['xaxis.range[1]']) {{
            startTime = new Date(xRange['xaxis.range[0]']).getTime();
            endTime = new Date(xRange['xaxis.range[1]']).getTime();
        }} else {{
            startTime = Math.min(...csTimeNumeric);
            endTime = Math.max(...csTimeNumeric);
        }}

        // Filter cycle-start data to visible range
        const vis = [];
        for (let i = 0; i < csTimeNumeric.length; i++) {{
            if (csTimeNumeric[i] >= startTime && csTimeNumeric[i] <= endTime) {{
                vis.push(i);
            }}
        }}
        if (vis.length < 2) return;

        const visTimes = vis.map(i => csTimeNumeric[i]);
        const visTMPs = vis.map(i => csTMPs[i]);

        const result = linearRegression(visTimes, visTMPs);
        if (!result) return;

        const {{slope, intercept, r2, resStd}} = result;
        const slopePerDay = slope * 86400000;

        // Compute fit line over full cycle-start range
        const fitX = csTimestamps;
        const fitY = csTimeNumeric.map(t => slope * t + intercept);

        // Compute forecast (continuation)
        const lastTime = csTimeNumeric[csTimeNumeric.length - 1];
        const fittedAtLast = slope * lastTime + intercept;
        const futureTime = lastTime + forecastHorizonDays * 86400000;
        const predictedTMP = slope * futureTime + intercept;
        const futureDate = new Date(futureTime);

        // Build confidence cone (20 points)
        const nPts = 20;
        const margin = 1.96 * resStd;
        const coneX_upper = [];
        const coneY_upper = [];
        const coneX_lower = [];
        const coneY_lower = [];
        for (let i = 0; i < nPts; i++) {{
            const frac = i / (nPts - 1);
            const t = lastTime + frac * (futureTime - lastTime);
            const v = slope * t + intercept;
            const d = new Date(t).toISOString();
            coneX_upper.push(d);
            coneY_upper.push(v + frac * margin);
            coneX_lower.push(d);
            coneY_lower.push(v - frac * margin);
        }}

        // Time to threshold
        let daysTo6 = null;
        if (slope > 0) {{
            daysTo6 = (thresholdTMP - fittedAtLast) / slopePerDay;
        }}

        const plotDiv = document.getElementsByClassName('plotly')[0];
        if (!plotDiv || !plotDiv.data) return;

        // Update Irreversible Fouling Trend trace
        const trendIdx = plotDiv.data.findIndex(t => t.name === 'Irreversible Fouling Trend');
        if (trendIdx >= 0) {{
            Plotly.restyle(plotDiv, {{ 'x': [fitX], 'y': [fitY] }}, [trendIdx]);
        }}

        // Update Linear Forecast trace
        const fcIdx = plotDiv.data.findIndex(t => t.name && t.name.startsWith('Linear Forecast'));
        if (fcIdx >= 0) {{
            const fcX = [];
            const fcY = [];
            for (let i = 0; i < nPts; i++) {{
                fcX.push(coneX_upper[i]);
                fcY.push(slope * (lastTime + (i / (nPts - 1)) * (futureTime - lastTime)) + intercept);
            }}
            Plotly.restyle(plotDiv, {{ 'x': [fcX], 'y': [fcY] }}, [fcIdx]);
        }}

        // Update confidence cone (upper bound trace, then lower bound + fill)
        const upperIdx = plotDiv.data.findIndex(t => t.name === '95% Confidence');
        if (upperIdx >= 0 && upperIdx > 0) {{
            // Upper is the trace before '95% Confidence'
            Plotly.restyle(plotDiv, {{ 'x': [coneX_upper], 'y': [coneY_upper] }}, [upperIdx - 1]);
            Plotly.restyle(plotDiv, {{ 'x': [coneX_lower], 'y': [coneY_lower] }}, [upperIdx]);
        }}

        // Determine if using full range
        const isFullRange = vis.length === csTimeNumeric.length;

        // Fouling classification
        const absSlopeH = Math.abs(slope * 3600000);
        let classification = 'low';
        let clsColor = '#28a745';
        if (absSlopeH >= 0.01) {{ classification = 'critical'; clsColor = '#dc3545'; }}
        else if (absSlopeH >= 0.005) {{ classification = 'high'; clsColor = '#fd7e14'; }}
        else if (absSlopeH >= 0.001) {{ classification = 'medium'; clsColor = '#ffc107'; }}

        // Build annotation
        let annText = `<b style="color:${{clsColor}}">⬤ Fouling: ${{classification.toUpperCase()}}</b>`;
        annText += `<br><b>Irrev. slope:</b> ${{slopePerDay.toFixed(4)}} bar/day (R²=${{r2.toFixed(3)}})`;
        if (daysTo6 !== null && daysTo6 > 0) {{
            annText += `<br><b>Time to 6 bar:</b> ${{Math.round(daysTo6)}} days`;
        }}
        const modelLabel = isFullRange ? originalModelType : 'dynamic (filtered range)';
        annText += `<br><b>Model:</b> ${{modelLabel}}`;

        Plotly.relayout(plotDiv, {{
            'annotations[0].text': annText
        }});
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        const plotDiv = document.getElementsByClassName('plotly')[0];
        if (plotDiv) {{
            plotDiv.on('plotly_relayout', function(eventData) {{
                updateForecast(eventData);
            }});
            updateForecast(null);
        }}
    }});
</script>
"""

    html_str = html_str.replace("</body>", f"{custom_js}</body>")

    filepath = os.path.join(plots_dir, "TMP_forecast.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_str)

    logger.info("  ✓ Created: TMP forecast plot (redesigned with irreversible trend)")

    return filepath


def create_kpi_dashboard(
    kpis: list, alerts: list, forecasts: dict, plots_dir: str, config: dict
) -> str:
    """
    Create comprehensive KPI dashboard with cards, alerts, and forecasts

    Args:
        kpis: List of KPI dictionaries
        alerts: List of alert dictionaries
        forecasts: Dictionary with 'tmp' and 'permeability' forecast dictionaries
        plots_dir: Output directory
        config: Configuration

    Returns:
        Path to generated HTML file
    """
    if not kpis:
        logger.warning("No KPIs available for dashboard")
        return None

    logger.info("Creating KPI dashboard...")

    # Status colors
    status_colors = {
        "good": "#28a745",
        "warning": "#ffc107",
        "critical": "#dc3545",
        None: "#6c757d",
    }
    status_icons = {"good": "✓", "warning": "⚠", "critical": "✗", None: "◯"}

    # Create HTML dashboard
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CapNF - KPI Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            color: #333;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            color: #666;
            font-size: 1.1em;
        }}
        .timestamp {{
            background: #f8f9fa;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
            margin-top: 15px;
            color: #666;
        }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 25px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
            border-left: 5px solid;
        }}
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }}
        .kpi-card.good {{ border-left-color: #28a745; }}
        .kpi-card.warning {{ border-left-color: #ffc107; }}
        .kpi-card.critical {{ border-left-color: #dc3545; }}
        .kpi-card.neutral {{ border-left-color: #6c757d; }}
        .kpi-name {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        .kpi-value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        .kpi-unit {{
            font-size: 1em;
            color: #666;
            margin-left: 5px;
        }}
        .kpi-target {{
            font-size: 0.9em;
            color: #666;
            margin-top: 8px;
        }}
        .kpi-status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin-top: 10px;
            color: white;
        }}
        .status-good {{ background: #28a745; }}
        .status-warning {{ background: #ffc107; color: #333; }}
        .status-critical {{ background: #dc3545; }}
        .status-neutral {{ background: #6c757d; }}
        .alerts-container {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        .alert-card {{
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .alert-card.critical {{
            border-left-color: #dc3545;
            background: #ffe5e5;
        }}
        .alert-card.warning {{
            border-left-color: #ffc107;
            background: #fff8e1;
        }}
        .alert-card.info {{
            border-left-color: #17a2b8;
            background: #e5f7fa;
        }}
        .alert-message {{
            flex: 1;
            font-size: 1.1em;
        }}
        .alert-severity {{
            padding: 5px 15px;
            border-radius: 5px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
        }}
        .severity-critical {{
            background: #dc3545;
            color: white;
        }}
        .severity-warning {{
            background: #ffc107;
            color: #333;
        }}
        .severity-info {{
            background: #17a2b8;
            color: white;
        }}
        .forecast-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        .forecast-title {{
            font-size: 1.5em;
            margin-bottom: 15px;
        }}
        .forecast-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        .forecast-detail {{
            font-size: 1em;
            margin: 5px 0;
            opacity: 0.9;
        }}
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }}
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .stat-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏭 CapNF System</h1>
            <p>Membrane Filtration Analytics Dashboard</p>
            <div class="timestamp">Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>

        <div class="section">
            <h2 class="section-title">📊 Key Performance Indicators</h2>
            <div class="kpi-grid">
"""

    # Add KPI cards
    for kpi in kpis:
        status = kpi.get("status") or "neutral"
        status_class = f"status-{status}" if status != "neutral" else "status-neutral"
        card_class = status if status else "neutral"

        value = kpi.get("value", 0)
        kpi_name = kpi.get("name", "")

        # Apply specific formatting based on KPI name
        if isinstance(value, (int, float)):
            if kpi_name == "Chemical Cleaning Frequency":
                value_str = f"{int(value)}"
            elif kpi_name in [
                "Average Specific Energy",
                "Average Cycle Duration",
                "Current TMP",
            ]:
                value_str = f"{value:.2f}"
            elif abs(value) >= 100:
                value_str = f"{value:.1f}"
            elif abs(value) >= 10:
                value_str = f"{value:.2f}"
            else:
                value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        target_text = ""
        if kpi.get("target") is not None:
            target_text = f"<div class='kpi-target'>Target: {kpi['target']} {kpi.get('unit', '')}</div>"

        status_icon = status_icons.get(status, "◯")
        status_text = status.capitalize() if status else "N/A"

        html_content += f"""
                <div class="kpi-card {card_class}">
                    <div class="kpi-name">{kpi.get("name", "Unknown")}</div>
                    <div class="kpi-value">{value_str}<span class="kpi-unit">{kpi.get("unit", "")}</span></div>
                    {target_text}
                    <span class="kpi-status {status_class}">{status_icon} {status_text}</span>
                </div>
"""

    html_content += """
            </div>
        </div>
"""

    # Add alerts section
    html_content += """
        <div class="section">
            <h2 class="section-title">⚠️ Alerts & Notifications</h2>
"""

    if alerts:
        html_content += '<div class="alerts-container">'
        for alert in alerts:
            severity = alert.get("severity", "info")
            html_content += f"""
                <div class="alert-card {severity}">
                    <div class="alert-message">{alert.get("message", "No message")}</div>
                    <span class="alert-severity severity-{severity}">{severity.upper()}</span>
                </div>
"""
        html_content += "</div>"
    else:
        html_content += (
            '<div class="no-data">✓ No active alerts - System operating normally</div>'
        )

    html_content += """
        </div>
"""

    # Add TMP forecast section
    tmp_forecast = forecasts.get("tmp") if forecasts else None
    if tmp_forecast:
        html_content += """
        <div class="section">
            <h2 class="section-title">🔮 TMP Forecast</h2>
            <div class="forecast-box">
                <div class="forecast-title">Transmembrane Pressure Prediction</div>
"""

        if "predicted_value" in tmp_forecast:
            pred_val = tmp_forecast["predicted_value"]
            html_content += f'<div class="forecast-value">{pred_val:.2f} bar</div>'

        if "prediction_date" in tmp_forecast:
            html_content += f'<div class="forecast-detail">📅 Forecast Date: {tmp_forecast["prediction_date"]}</div>'

        if (
            "time_to_threshold_days" in tmp_forecast
            and tmp_forecast["time_to_threshold_days"]
        ):
            days = tmp_forecast["time_to_threshold_days"]
            html_content += f'<div class="forecast-detail">⏱️ Time to 6 bar threshold: {days:.1f} days</div>'

        if "confidence" in tmp_forecast:
            html_content += f'<div class="forecast-detail">📊 Confidence Level: {tmp_forecast["confidence"] * 100:.0f}%</div>'

        if "model_type" in tmp_forecast:
            html_content += f'<div class="forecast-detail">🔧 Model: {tmp_forecast["model_type"].capitalize()}</div>'

        html_content += """
            </div>
        </div>
"""

    # Add Permeability TC forecast section
    perm_forecast = forecasts.get("permeability") if forecasts else None
    if perm_forecast:
        html_content += """
        <div class="section">
            <h2 class="section-title">🔮 Permeability TC Forecast</h2>
            <div class="forecast-box">
                <div class="forecast-title">Temperature-Corrected Permeability Prediction</div>
"""

        if "predicted_value" in perm_forecast:
            pred_val = perm_forecast["predicted_value"]
            html_content += f'<div class="forecast-value">{pred_val:.1f} LMH/bar</div>'

        if "prediction_date" in perm_forecast:
            html_content += f'<div class="forecast-detail">📅 Forecast Date: {perm_forecast["prediction_date"]}</div>'

        if (
            "time_to_threshold_days" in perm_forecast
            and perm_forecast["time_to_threshold_days"]
        ):
            days = perm_forecast["time_to_threshold_days"]
            threshold = perm_forecast.get("threshold_value", 200)
            html_content += f'<div class="forecast-detail">⏱️ Time to {threshold:.0f} LMH/bar threshold: {days:.1f} days</div>'

        if "confidence" in perm_forecast:
            html_content += f'<div class="forecast-detail">📊 Confidence Level: {perm_forecast["confidence"] * 100:.0f}%</div>'

        if "model_type" in perm_forecast:
            html_content += f'<div class="forecast-detail">🔧 Model: {perm_forecast["model_type"].capitalize()}</div>'

        html_content += """
            </div>
        </div>
"""

    # Add summary statistics
    html_content += """
        <div class="section">
            <h2 class="section-title">📈 Summary Statistics</h2>
            <div class="summary-stats">
"""

    total_kpis = len(kpis)
    good_kpis = sum(1 for k in kpis if k.get("status") == "good")
    warning_kpis = sum(1 for k in kpis if k.get("status") == "warning")
    critical_kpis = sum(1 for k in kpis if k.get("status") == "critical")
    total_alerts = len(alerts)
    critical_alerts = sum(1 for a in alerts if a.get("severity") == "critical")

    html_content += f"""
                <div class="stat-box">
                    <div class="stat-label">Total KPIs</div>
                    <div class="stat-value">{total_kpis}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">✓ Good Status</div>
                    <div class="stat-value" style="color: #28a745;">{good_kpis}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">⚠ Warnings</div>
                    <div class="stat-value" style="color: #ffc107;">{warning_kpis}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">✗ Critical</div>
                    <div class="stat-value" style="color: #dc3545;">{critical_kpis}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Active Alerts</div>
                    <div class="stat-value" style="color: {("#dc3545" if critical_alerts > 0 else "#17a2b8")};">{total_alerts}</div>
                </div>
"""

    html_content += """
            </div>
        </div>
    </div>
</body>
</html>
"""

    # Save HTML file
    filepath = os.path.join(plots_dir, "kpi_dashboard.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info("  ✓ Created: KPI Dashboard")
    return filepath


def create_cycle_comparison_plot(cycles: list, plots_dir: str, config: dict) -> str:
    """
    Create cycle comparison bar chart

    Args:
        cycles: List of cycle dictionaries
        plots_dir: Output directory
        config: Configuration

    Returns:
        Path to generated file
    """
    if not cycles:
        logger.warning("No cycles available for comparison plot")
        return None

    logger.info("Creating cycle comparison plot...")

    fig = go.Figure()

    cycle_ids = [c["cycle_id"] for c in cycles]
    tmp_slopes = [c["tmp_slope"] for c in cycles]
    durations = [c["duration_hours"] for c in cycles]

    # TMP slope bar chart
    fig.add_trace(
        go.Bar(
            x=cycle_ids,
            y=tmp_slopes,
            name="TMP Slope (bar/hour)",
            marker_color="indianred",
        )
    )

    # Duration on secondary axis
    fig.add_trace(
        go.Scatter(
            x=cycle_ids,
            y=durations,
            name="Duration (hours)",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="steelblue", width=2),
        )
    )

    fig.update_layout(
        title="<b>Cycle-by-Cycle Comparison</b>",
        xaxis=dict(title="Cycle ID"),
        yaxis=dict(title="TMP Slope (bar/hour)"),
        yaxis2=dict(
            title="Duration (hours)", overlaying="y", side="right", fixedrange=True
        ),
        height=600,
        showlegend=True,
    )

    filepath = os.path.join(plots_dir, "cycle_comparison.html")
    fig.write_html(filepath, config=get_plot_config(), include_plotlyjs="cdn")

    logger.info("  ✓ Created: Cycle comparison plot")

    return filepath


def create_unified_dashboard(
    kpis: list,
    alerts: list,
    forecasts: dict,
    df_viz: pd.DataFrame,
    df_full: pd.DataFrame,
    cycles: list,
    plots_dir: str,
    config: dict,
) -> str:
    """
    Create unified dashboard with navigation between KPI overview and all plots

    Args:
        kpis: List of KPI dictionaries
        alerts: List of alert dictionaries
        forecasts: Dictionary with 'tmp' and 'permeability' forecast dictionaries
        df_viz: Visualization DataFrame (downsampled)
        df_full: Full DataFrame for accurate calculations
        cycles: List of cycle dictionaries
        plots_dir: Output directory
        config: Configuration dictionary

    Returns:
        Path to unified dashboard HTML file
    """
    logger.info("Creating unified navigation dashboard...")

    # Get list of parameters for plots
    numeric_cols = [
        col
        for col in df_viz.columns
        if col != "TimeStamp"
        and pd.api.types.is_numeric_dtype(df_viz[col])
        and not col.endswith("_SMA")
        and not col.startswith("TMP_slope")
    ][:10]  # Limit to 10 main parameters

    # Status colors and icons
    status_colors = {
        "good": "#28a745",
        "warning": "#ffc107",
        "critical": "#dc3545",
        None: "#6c757d",
    }
    status_icons = {"good": "✓", "warning": "⚠", "critical": "✗", None: "○"}

    # Start HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <title>CapNF - Membrane Filtration Analytics</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            overflow-x: hidden;
        }}
        
        body.menu-open {{
            overflow: hidden;
            position: fixed;
            width: 100%;
        }}
        
        /* Sidebar Navigation */
        .sidebar {{
            width: 280px;
            background: white;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            z-index: 1000;
        }}
        
        .sidebar-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px 20px;
            text-align: center;
        }}
        
        .sidebar-header h1 {{
            font-size: 1.5em;
            margin-bottom: 5px;
        }}
        
        .sidebar-header p {{
            font-size: 0.85em;
            opacity: 0.9;
        }}
        
        .nav-menu {{
            list-style: none;
            padding: 20px 0;
        }}
        
        .nav-item {{
            padding: 15px 25px;
            cursor: pointer;
            transition: all 0.3s;
            border-left: 4px solid transparent;
            display: flex;
            align-items: center;
            gap: 10px;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
            user-select: none;
            -webkit-user-select: none;
        }}
        
        .nav-item:hover {{
            background: #f8f9fa;
            border-left-color: #667eea;
        }}
        
        .nav-item.active {{
            background: #e9ecef;
            border-left-color: #667eea;
            font-weight: 600;
            color: #667eea;
        }}
        
        .nav-icon {{
            font-size: 1.2em;
            width: 24px;
            text-align: center;
        }}
        
        .nav-section {{
            padding: 10px 25px;
            font-size: 0.8em;
            font-weight: 600;
            color: #999;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 10px;
        }}
        
        /* Main Content Area */
        .main-content {{
            margin-left: 280px;
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            height: 100vh;
        }}
        
        .content-section {{
            display: none;
            animation: fadeIn 0.3s;
        }}
        
        .content-section.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        /* KPI Dashboard Styles */
        .dashboard-header {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .dashboard-header h2 {{
            color: #667eea;
            font-size: 2em;
            margin-bottom: 10px;
        }}
        
        .timestamp {{
            background: #f8f9fa;
            padding: 10px 20px;
            border-radius: 5px;
            display: inline-block;
            margin-top: 10px;
            color: #666;
            font-size: 0.9em;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 20px;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 20px;
        }}
        
        .kpi-card {{
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s;
            border-left: 5px solid;
        }}
        
        .kpi-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        }}
        
        .kpi-card.good {{ border-left-color: #28a745; }}
        .kpi-card.warning {{ border-left-color: #ffc107; }}
        .kpi-card.critical {{ border-left-color: #dc3545; }}
        .kpi-card.neutral {{ border-left-color: #6c757d; }}
        
        .kpi-name {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 10px;
            text-transform: uppercase;
            font-weight: 600;
        }}
        
        .kpi-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .kpi-unit {{
            font-size: 0.5em;
            color: #666;
            margin-left: 5px;
        }}
        
        .kpi-target {{
            font-size: 0.85em;
            color: #666;
            margin-top: 8px;
        }}
        
        .kpi-status {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            margin-top: 10px;
            color: white;
        }}
        
        .status-good {{ background: #28a745; }}
        .status-warning {{ background: #ffc107; color: #333; }}
        .status-critical {{ background: #dc3545; }}
        .status-neutral {{ background: #6c757d; }}
        
        .alerts-container {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .alert-card {{
            padding: 20px;
            border-radius: 8px;
            border-left: 5px solid;
            background: #f8f9fa;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .alert-card.critical {{
            border-left-color: #dc3545;
            background: #ffe5e5;
        }}
        
        .alert-card.warning {{
            border-left-color: #ffc107;
            background: #fff8e1;
        }}
        
        .alert-card.info {{
            border-left-color: #17a2b8;
            background: #e5f7fa;
        }}
        
        .alert-severity {{
            padding: 5px 15px;
            border-radius: 5px;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8em;
        }}
        
        .severity-critical {{ background: #dc3545; color: white; }}
        .severity-warning {{ background: #ffc107; color: #333; }}
        .severity-info {{ background: #17a2b8; color: white; }}
        
        .forecast-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 10px;
        }}
        
        .forecast-title {{
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        
        .forecast-value {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .forecast-detail {{
            margin: 5px 0;
            opacity: 0.9;
        }}
        
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }}
        
        .summary-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        
        .stat-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .stat-label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .stat-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #667eea;
        }}
        
        /* Plot Container */
        .plot-container {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .plot-title {{
            font-size: 1.3em;
            color: #667eea;
            margin-bottom: 15px;
            font-weight: 600;
        }}
        
        /* Mobile Menu Button */
        .mobile-menu-btn {{
            display: none;
            position: fixed;
            top: 15px;
            left: 15px;
            z-index: 1100;
            background: white;
            border: none;
            padding: 12px 15px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
            cursor: pointer;
            font-size: 1.5em;
            -webkit-tap-highlight-color: transparent;
            touch-action: manipulation;
        }}
        
        .mobile-menu-btn:active {{
            background: #f0f0f0;
        }}
        
        .mobile-overlay {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.5);
            z-index: 999;
            pointer-events: none;
        }}
        
        .mobile-overlay.active {{
            pointer-events: auto;
        }}
        
        /* Plot responsiveness */
        .js-plotly-plot {{
            width: 100% !important;
        }}
        
        .plotly {{
            width: 100% !important;
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .mobile-menu-btn {{
                display: block;
            }}
            
            .sidebar {{
                width: 85vw;
                max-width: 320px;
                position: fixed;
                left: calc(-85vw);
                height: 100vh;
                height: 100dvh;
                transition: left 0.3s ease;
                z-index: 1000;
                -webkit-overflow-scrolling: touch;
                overscroll-behavior: contain;
            }}
            
            .sidebar.open {{
                left: 0;
                box-shadow: 2px 0 20px rgba(0,0,0,0.3);
            }}
            
            .mobile-overlay.active {{
                display: block;
            }}
            
            .nav-menu {{
                overflow-y: auto;
                overflow-x: hidden;
                max-height: calc(100vh - 120px);
                padding-bottom: 20px;
            }}
            
            .main-content {{
                margin-left: 0;
                padding: 60px 15px 15px 15px;
                width: 100%;
                max-width: 100vw;
                overflow-x: hidden;
            }}
            
            .kpi-grid {{
                grid-template-columns: 1fr;
                gap: 15px;
            }}
            
            .dashboard-header {{
                padding: 20px 15px;
            }}
            
            .dashboard-header h2 {{
                font-size: 1.5em;
            }}
            
            .section {{
                padding: 20px 15px;
            }}
            
            .section-title {{
                font-size: 1.2em;
            }}
            
            .kpi-card {{
                padding: 15px;
            }}
            
            .kpi-value {{
                font-size: 1.8em;
            }}
            
            .nav-item {{
                padding: 18px 25px;
                font-size: 1.05em;
            }}
            
            .summary-stats {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .forecast-box {{
                padding: 20px;
            }}
            
            .forecast-value {{
                font-size: 1.8em;
            }}
            
            .plot-container {{
                padding: 10px;
                margin-bottom: 20px;
            }}
        }}
        
        @media (max-width: 480px) {{
            .kpi-value {{
                font-size: 1.6em;
            }}
            
            .dashboard-header h2 {{
                font-size: 1.3em;
            }}
            
            .summary-stats {{
                grid-template-columns: 1fr;
            }}
        }}
        
        /* iPhone specific - safe area support */
        @supports (padding: max(0px)) {{
            .mobile-menu-btn {{
                top: max(15px, env(safe-area-inset-top));
                left: max(15px, env(safe-area-inset-left));
            }}
            
            @media (max-width: 768px) {{
                .main-content {{
                    padding: max(60px, calc(60px + env(safe-area-inset-top))) max(15px, env(safe-area-inset-right)) max(15px, env(safe-area-inset-bottom)) max(15px, env(safe-area-inset-left));
                }}
            }}
        }}
    </style>
</head>
<body>
    <!-- Mobile Menu Button -->
    <button class="mobile-menu-btn" onclick="toggleMobileMenu()">☰</button>
    
    <!-- Mobile Overlay -->
    <div class="mobile-overlay" onclick="closeMobileMenu()"></div>
    
    <!-- Sidebar Navigation -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>🏭 CapNF</h1>
            <p>Membrane Filtration Analytics</p>
        </div>
        
        <ul class="nav-menu">
            <li class="nav-item active" onclick="showSection('overview')">
                <span class="nav-icon">📊</span>
                <span>KPI Overview</span>
            </li>
            
            <li class="nav-section">Process Parameters</li>
"""

    # Add navigation items for each parameter plot
    # Define display names for navigation
    display_names_nav = {
        "Vcrossflow": "Cross flow velocity",
        "Mem. retention": "Membrane retention",
        "Sys. retention": "System retention",
        "01-TIT-01": "Temperature",
    }

    for i, col in enumerate(numeric_cols):
        icon = "📈" if i % 2 == 0 else "📉"
        display_col = display_names_nav.get(col, col)
        html += f"""            <li class="nav-item" onclick="showSection('{col}')">
                <span class="nav-icon">{icon}</span>
                <span>{display_col}</span>
            </li>
"""

    html += """            
            <li class="nav-section">Advanced Analytics</li>
            <li class="nav-item" onclick="showSection('combined-parameters')">
                <span class="nav-icon">📊</span>
                <span>Combined Parameters</span>
            </li>
            <li class="nav-item" onclick="showSection('tmp-forecast')">
                <span class="nav-icon">🔮</span>
                <span>TMP Forecast</span>
            </li>
"""

    if cycles:
        html += """            <li class="nav-item" onclick="showSection('cycle-comparison')">
                <span class="nav-icon">🔄</span>
                <span>Cycle Comparison</span>
            </li>
"""

    html += """        </ul>
    </div>
    
    <!-- Main Content -->
    <div class="main-content">
"""

    # KPI Overview Section
    html += """        <!-- KPI Overview Section -->
        <div id="overview" class="content-section active">
            <div class="dashboard-header">
                <h2>📊 System Performance Dashboard</h2>
                <div class="timestamp">Generated: """
    html += pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    html += """</div>
            </div>
            
            <div class="section">
                <h3 class="section-title">📈 Key Performance Indicators</h3>
                <div class="kpi-grid">
"""

    # Add KPI cards
    for kpi in kpis:
        status = kpi.get("status") or "neutral"
        status_class = f"status-{status}" if status != "neutral" else "status-neutral"
        card_class = status if status else "neutral"

        value = kpi.get("value", 0)
        kpi_name = kpi.get("name", "")

        # Apply specific formatting based on KPI name
        if isinstance(value, (int, float)):
            if kpi_name == "Chemical Cleaning Frequency":
                value_str = f"{int(value)}"
            elif kpi_name in [
                "Average Specific Energy",
                "Average Cycle Duration",
                "Current TMP",
            ]:
                value_str = f"{value:.2f}"
            elif abs(value) >= 100:
                value_str = f"{value:.1f}"
            elif abs(value) >= 10:
                value_str = f"{value:.2f}"
            else:
                value_str = f"{value:.3f}"
        else:
            value_str = str(value)

        target_text = ""
        if kpi.get("target") is not None:
            target_text = f"<div class='kpi-target'>Target: {kpi['target']} {kpi.get('unit', '')}</div>"

        status_icon = status_icons.get(status, "○")
        status_text = status.capitalize() if status else "N/A"

        html += f"""                    <div class="kpi-card {card_class}">
                        <div class="kpi-name">{kpi.get("name", "Unknown")}</div>
                        <div class="kpi-value">{value_str}<span class="kpi-unit">{kpi.get("unit", "")}</span></div>
                        {target_text}
                        <span class="kpi-status {status_class}">{status_icon} {status_text}</span>
                    </div>
"""

    html += """                </div>
            </div>
            
            <div class="section">
                <h3 class="section-title">⚠️ Alerts & Notifications</h3>
"""

    # Add alerts
    if alerts:
        html += '<div class="alerts-container">'
        for alert in alerts:
            severity = alert.get("severity", "info")
            html += f"""                    <div class="alert-card {severity}">
                        <div class="alert-message">{alert.get("message", "No message")}</div>
                        <span class="alert-severity severity-{severity}">{severity.upper()}</span>
                    </div>
"""
        html += "</div>"
    else:
        html += (
            '<div class="no-data">✓ No active alerts - System operating normally</div>'
        )

    html += """            </div>
"""

    # Add TMP forecast section
    tmp_forecast = forecasts.get("tmp") if forecasts else None
    if tmp_forecast:
        html += """            <div class="section">
                <h3 class="section-title">🔮 TMP Forecast</h3>
                <div class="forecast-box">
                    <div class="forecast-title">Transmembrane Pressure Prediction</div>
"""

        if "predicted_value" in tmp_forecast:
            pred_val = tmp_forecast["predicted_value"]
            html += f'                    <div class="forecast-value">{pred_val:.2f} bar</div>'

        if "prediction_date" in tmp_forecast:
            html += f'                    <div class="forecast-detail">📅 Forecast Date: {tmp_forecast["prediction_date"]}</div>'

        if (
            "time_to_threshold_days" in tmp_forecast
            and tmp_forecast["time_to_threshold_days"]
        ):
            days = tmp_forecast["time_to_threshold_days"]
            html += f'                    <div class="forecast-detail">⏱️ Time to 6 bar threshold: {days:.1f} days</div>'

        if "confidence" in tmp_forecast:
            html += f'                    <div class="forecast-detail">📊 Confidence Level: {tmp_forecast["confidence"] * 100:.0f}%</div>'

        if "model_type" in tmp_forecast:
            html += f'                    <div class="forecast-detail">🔧 Model: {tmp_forecast["model_type"].capitalize()}</div>'

        html += """                </div>
            </div>
"""

    # Add Permeability TC forecast section
    perm_forecast = forecasts.get("permeability") if forecasts else None
    if perm_forecast:
        html += """            <div class="section">
                <h3 class="section-title">🔮 Permeability TC Forecast</h3>
                <div class="forecast-box">
                    <div class="forecast-title">Temperature-Corrected Permeability Prediction</div>
"""

        if "predicted_value" in perm_forecast:
            pred_val = perm_forecast["predicted_value"]
            html += f'                    <div class="forecast-value">{pred_val:.1f} LMH/bar</div>'

        if "prediction_date" in perm_forecast:
            html += f'                    <div class="forecast-detail">📅 Forecast Date: {perm_forecast["prediction_date"]}</div>'

        if (
            "time_to_threshold_days" in perm_forecast
            and perm_forecast["time_to_threshold_days"]
        ):
            days = perm_forecast["time_to_threshold_days"]
            threshold = perm_forecast.get("threshold_value", 200)
            html += f'                    <div class="forecast-detail">⏱️ Time to {threshold:.0f} LMH/bar threshold: {days:.1f} days</div>'

        if "confidence" in perm_forecast:
            html += f'                    <div class="forecast-detail">📊 Confidence Level: {perm_forecast["confidence"] * 100:.0f}%</div>'

        if "model_type" in perm_forecast:
            html += f'                    <div class="forecast-detail">🔧 Model: {perm_forecast["model_type"].capitalize()}</div>'

        html += """                </div>
            </div>
"""

    # Add summary statistics
    total_kpis = len(kpis)
    good_kpis = sum(1 for k in kpis if k.get("status") == "good")
    warning_kpis = sum(1 for k in kpis if k.get("status") == "warning")
    critical_kpis = sum(1 for k in kpis if k.get("status") == "critical")
    total_alerts = len(alerts)
    critical_alerts = sum(1 for a in alerts if a.get("severity") == "critical")

    html += f"""            <div class="section">
                <h3 class="section-title">📊 Summary Statistics</h3>
                <div class="summary-stats">
                    <div class="stat-box">
                        <div class="stat-label">Total KPIs</div>
                        <div class="stat-value">{total_kpis}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">✓ Good Status</div>
                        <div class="stat-value" style="color: #28a745;">{good_kpis}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">⚠ Warnings</div>
                        <div class="stat-value" style="color: #ffc107;">{warning_kpis}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">✗ Critical</div>
                        <div class="stat-value" style="color: #dc3545;">{critical_kpis}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Active Alerts</div>
                        <div class="stat-value" style="color: {"#dc3545" if critical_alerts > 0 else "#17a2b8"};">{total_alerts}</div>
                    </div>
                </div>
            </div>
        </div>
"""

    # Create plots for each parameter
    colors = config["colors"]
    temp_color = config["temperature_color"]

    # Define display names for visual titles
    display_names = {
        "Vcrossflow": "Cross flow velocity",
        "Mem. retention": "Membrane retention",
        "Sys. retention": "System retention",
        "01-TIT-01": "Temperature",
    }

    for i, col in enumerate(numeric_cols):
        display_col = display_names.get(col, col)
        html += f"""        <!-- {display_col} Section -->
        <div id="{col}" class="content-section">
            <div class="plot-container">
                <h3 class="plot-title">{display_col} vs Time</h3>
                <div id="plot-{col}"></div>
            </div>
        </div>
"""

    # Combined Parameters Section
    html += """        <!-- Combined Parameters Section -->
        <div id="combined-parameters" class="content-section">
            <div class="plot-container">
                <h3 class="plot-title">Combined Process Parameters with Temperature</h3>
                <div id="plot-combined-parameters"></div>
            </div>
        </div>
"""

    # TMP Forecast Section
    html += """        <!-- TMP Forecast Section -->
        <div id="tmp-forecast" class="content-section">
            <div class="plot-container">
                <h3 class="plot-title">TMP Forecast — Irreversible Fouling Trend</h3>
                <div id="plot-tmp-forecast"></div>
            </div>
        </div>
"""

    # Cycle Comparison Section
    if cycles:
        html += """        <!-- Cycle Comparison Section -->
        <div id="cycle-comparison" class="content-section">
            <div class="plot-container">
                <h3 class="plot-title">Cycle-by-Cycle Comparison</h3>
                <div id="plot-cycle-comparison"></div>
            </div>
        </div>
"""

    # Add JavaScript for navigation and plotting
    html += """    </div>
    
    <script>
        // Mobile menu functions with improved touch handling
        function toggleMobileMenu() {
            const sidebar = document.querySelector('.sidebar');
            const overlay = document.querySelector('.mobile-overlay');
            const body = document.body;
            sidebar.classList.toggle('open');
            overlay.classList.toggle('active');
            body.classList.toggle('menu-open');
        }
        
        function closeMobileMenu() {
            const sidebar = document.querySelector('.sidebar');
            const overlay = document.querySelector('.mobile-overlay');
            const body = document.body;
            sidebar.classList.remove('open');
            overlay.classList.remove('active');
            body.classList.remove('menu-open');
        }
        
        // Prevent sidebar from closing when scrolling inside it
        document.addEventListener('DOMContentLoaded', function() {
            const sidebar = document.querySelector('.sidebar');
            const overlay = document.querySelector('.mobile-overlay');
            
            // Only close when clicking the overlay, not when interacting with sidebar
            if (overlay) {
                overlay.addEventListener('click', function(e) {
                    if (e.target === overlay) {
                        closeMobileMenu();
                    }
                });
            }
            
            // Prevent touch events on sidebar from bubbling to overlay
            if (sidebar) {
                sidebar.addEventListener('touchstart', function(e) {
                    e.stopPropagation();
                }, { passive: true });
                
                sidebar.addEventListener('touchmove', function(e) {
                    e.stopPropagation();
                }, { passive: true });
            }
        });
        
        // Navigation function
        function showSection(sectionId) {
            // Hide all sections
            const sections = document.querySelectorAll('.content-section');
            sections.forEach(section => section.classList.remove('active'));
            
            // Remove active class from all nav items
            const navItems = document.querySelectorAll('.nav-item');
            navItems.forEach(item => item.classList.remove('active'));
            
            // Show selected section
            const selectedSection = document.getElementById(sectionId);
            if (selectedSection) {
                selectedSection.classList.add('active');
            }
            
            // Add active class to clicked nav item
            event.currentTarget.classList.add('active');
            
            // Close mobile menu if open (with small delay to ensure click is registered)
            if (window.innerWidth <= 768) {
                setTimeout(function() {
                    closeMobileMenu();
                }, 150);
            }
            
            // Render plot if needed
            renderPlot(sectionId);
        }
        
        let renderedPlots = {};
        
        // Define display names for visual titles
        const displayNames = {
            'Vcrossflow': 'Cross flow velocity',
            'Mem. retention': 'Membrane retention',
            'Sys. retention': 'System retention',
            '01-TIT-01': 'Temperature'
        };
        
        function getDisplayName(col) {
            return displayNames[col] || col;
        }
        
        function renderPlot(sectionId) {
            if (renderedPlots[sectionId]) {
                return; // Already rendered
            }
            
            // Mark as rendered
            renderedPlots[sectionId] = true;
            
"""

    # Add plotting code for each parameter
    for i, col in enumerate(numeric_cols):
        html += f"""            if (sectionId === '{col}') {{
                var displayName = getDisplayName('{col}');
                var trace1 = {{
                    x: {df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()},
                    y: {df_viz[col].tolist()},
                    mode: 'lines',
                    name: displayName + ' (Raw)',
                    line: {{color: '{apply_color_palette(i, colors)}', width: 1}},
                    opacity: 0.4
                }};
                
                var traces = [trace1];
                
"""

        # Add SMA if available
        sma_col = f"{col}_SMA"
        if sma_col in df_viz.columns:
            html += f"""                var trace2 = {{
                    x: {df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()},
                    y: {df_viz[sma_col].tolist()},
                    mode: 'lines',
                    name: displayName + ' (SMA)',
                    line: {{color: '{apply_color_palette(i, colors)}', width: 2}}
                }};
                traces.push(trace2);
                
"""

        # Add temperature overlay if not temperature itself
        if "01-TIT-01" in df_viz.columns and col != "01-TIT-01":
            temp_col = (
                "01-TIT-01_SMA" if "01-TIT-01_SMA" in df_viz.columns else "01-TIT-01"
            )
            html += f"""                var trace3 = {{
                    x: {df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()},
                    y: {df_viz[temp_col].tolist()},
                    mode: 'lines',
                    name: 'Temperature (°C)',
                    line: {{color: '{temp_color}', width: 2}},
                    yaxis: 'y2',
                    opacity: 0.8
                }};
                traces.push(trace3);
                
"""

        html += f"""                var layout = {{
                    xaxis: {{
                        title: 'Time',
                        rangeselector: {{
                            buttons: [
                                {{count: 1, label: '1h', step: 'hour', stepmode: 'backward'}},
                                {{count: 12, label: '12h', step: 'hour', stepmode: 'backward'}},
                                {{count: 1, label: '1d', step: 'day', stepmode: 'backward'}},
                                {{count: 7, label: '1w', step: 'day', stepmode: 'backward'}},
                                {{count: 1, label: '1m', step: 'month', stepmode: 'backward'}},
                                {{count: 6, label: '6m', step: 'month', stepmode: 'backward'}},
                                {{step: 'all', label: 'All'}}
                            ]
                        }},
                        rangeslider: {{visible: true, thickness: 0.05}}
                    }},
                    yaxis: {{title: displayName, fixedrange: false}},
                    yaxis2: {{
                        title: 'Temperature (°C)',
                        overlaying: 'y',
                        side: 'right',
                        title: {{font: {{color: '{temp_color}'}}}},
                        tickfont: {{color: '{temp_color}'}},
                        fixedrange: true
                    }},
                    height: 650,
                    margin: {{t: 50, b: 100, l: 80, r: 80}},
                    hovermode: 'x unified',
                    dragmode: 'zoom'
                }};
                
                Plotly.newPlot('plot-{col}', traces, layout, {{responsive: true}});
            }}
            
"""

    # Add combined parameters plot
    # Get numeric columns (excluding temperature)
    numeric_cols_for_combined = [
        col
        for col in df_viz.columns
        if col != "TimeStamp"
        and col != "01-TIT-01"
        and pd.api.types.is_numeric_dtype(df_viz[col])
        and not col.endswith("_SMA")
        and not col.startswith("TMP_slope")
    ][:10]

    html += """            if (sectionId === 'combined-parameters') {
                var traces = [];
                
"""

    # Add each parameter
    for i, col in enumerate(numeric_cols_for_combined):
        sma_col = f"{col}_SMA"
        if sma_col in df_viz.columns:
            y_data = df_viz[sma_col].tolist()
            trace_name = col
        else:
            y_data = df_viz[col].tolist()
            trace_name = f"{col} (Raw)"

        visible = "true" if i < 5 else "'legendonly'"

        html += f"""                traces.push({{
                    x: {df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()},
                    y: {y_data},
                    mode: 'lines',
                    name: '{trace_name}',
                    line: {{color: '{apply_color_palette(i, colors)}', width: 2}},
                    visible: {visible}
                }});
                
"""

    # Add temperature on secondary axis
    if "01-TIT-01" in df_viz.columns:
        temp_col = "01-TIT-01_SMA" if "01-TIT-01_SMA" in df_viz.columns else "01-TIT-01"
        html += f"""                traces.push({{
                    x: {df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()},
                    y: {df_viz[temp_col].tolist()},
                    mode: 'lines',
                    name: 'Temperature (°C)',
                    line: {{color: '{temp_color}', width: 3}},
                    yaxis: 'y2'
                }});
                
"""

    html += """                var layout = {
                    xaxis: {
                        title: 'Time',
                        rangeselector: {
                            buttons: [
                                {count: 1, label: '1h', step: 'hour', stepmode: 'backward'},
                                {count: 12, label: '12h', step: 'hour', stepmode: 'backward'},
                                {count: 1, label: '1d', step: 'day', stepmode: 'backward'},
                                {count: 7, label: '1w', step: 'day', stepmode: 'backward'},
                                {count: 1, label: '1m', step: 'month', stepmode: 'backward'},
                                {count: 6, label: '6m', step: 'month', stepmode: 'backward'},
                                {step: 'all', label: 'All'}
                            ]
                        },
                        rangeslider: {visible: true, thickness: 0.05}
                    },
                    yaxis: {title: 'Process Parameters (Various Units)', fixedrange: false},
                    yaxis2: {
                        title: {text: 'Temperature (°C)', font: {color: '"""
    html += temp_color
    html += """'}},
                        overlaying: 'y',
                        side: 'right',
                        tickfont: {color: '"""
    html += temp_color
    html += """'},
                        fixedrange: true
                    },
                    height: 750,
                    margin: {t: 50, b: 100, l: 80, r: 120},
                    hovermode: 'x unified',
                    dragmode: 'zoom',
                    legend: {
                        orientation: 'v',
                        yanchor: 'top',
                        y: 1,
                        xanchor: 'left',
                        x: 1.08
                    }
                };
                
                Plotly.newPlot('plot-combined-parameters', traces, layout, {responsive: true});
            }
            
"""

    # Add TMP forecast plot - Generate inline to avoid CORS issues
    # Extract cycle-start data for embedding
    cycle_times = []
    cycle_tmp_starts = []
    if cycles:
        for c in cycles:
            tmp_s = (
                c.get("tmp_start")
                if isinstance(c, dict)
                else getattr(c, "tmp_start", None)
            )
            t_s = (
                c.get("start_time")
                if isinstance(c, dict)
                else getattr(c, "start_time", None)
            )
            if tmp_s is not None and t_s is not None:
                cycle_times.append(str(pd.to_datetime(t_s)))
                cycle_tmp_starts.append(float(tmp_s))

    # IQR filtering
    valid_cycle_times = []
    valid_cycle_tmps = []
    if len(cycle_tmp_starts) >= 4:
        arr = np.array(cycle_tmp_starts)
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        lb, ub = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        for t, v in zip(cycle_times, cycle_tmp_starts):
            if lb <= v <= ub:
                valid_cycle_times.append(t)
                valid_cycle_tmps.append(v)
    else:
        valid_cycle_times = cycle_times
        valid_cycle_tmps = cycle_tmp_starts

    # Get forecast data
    tmp_forecast = forecasts.get("tmp") if forecasts else {}
    forecast_horizon = tmp_forecast.get("forecast_horizon_days", 7)
    predicted_value = tmp_forecast.get("predicted_value")
    prediction_date = tmp_forecast.get("prediction_date")
    model_type = tmp_forecast.get("model_type", "linear")
    lower_bound = tmp_forecast.get("lower_bound")
    upper_bound = tmp_forecast.get("upper_bound")

    # Calculate irreversible fouling trend
    irrev_slope = 0
    irrev_intercept = 0
    r_squared = 0
    if len(valid_cycle_times) > 2:
        t0_dt = pd.to_datetime(min(valid_cycle_times))
        cs_seconds = np.array(
            [(pd.to_datetime(t) - t0_dt).total_seconds() for t in valid_cycle_times]
        )
        cs_values = np.array(valid_cycle_tmps)
        coeffs = np.polyfit(cs_seconds, cs_values, 1)
        fit_vals = coeffs[0] * cs_seconds + coeffs[1]
        y_mean = cs_values.mean()
        ss_tot = np.sum((cs_values - y_mean) ** 2)
        ss_res = np.sum((cs_values - fit_vals) ** 2)
        r_squared = float(1 - (ss_res / ss_tot) if ss_tot != 0 else 0)
        irrev_slope = float(coeffs[0])
        irrev_intercept = float(coeffs[1])

    # Generate TMP forecast plot inline - need TMP data arrays
    tmp_timestamps = df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist()
    tmp_raw_values = df_viz["TMP"].tolist()
    tmp_sma_values = df_viz["TMP_SMA"].tolist() if "TMP_SMA" in df_viz.columns else []

    html += f"""            if (sectionId === 'tmp-forecast') {{
                // TMP raw data arrays
                const timestamps = {tmp_timestamps};
                const tmp_values = {tmp_raw_values};
                const tmp_sma_values = {tmp_sma_values};
                
                // Embedded TMP forecast data
                const csTimestamps = {valid_cycle_times};
                const csTMPs = {valid_cycle_tmps};
                const irrevSlope = {irrev_slope};
                const irrevIntercept = {irrev_intercept};
                const rSquared = {r_squared};
                const forecastHorizonDays = {forecast_horizon};
                const predictedValue = {predicted_value if predicted_value else "null"};
                const predictionDate = "{prediction_date if prediction_date else ""}";
                const modelType = "{model_type}";
                const lowerBound = {lower_bound if lower_bound else "null"};
                const upperBound = {upper_bound if upper_bound else "null"};
                const thresholdTMP = 6.0;

                // Build plot traces
                const traces = [];

                // Trace 1: Raw TMP (background)
                traces.push({{
                    x: timestamps,
                    y: tmp_values,
                    mode: 'lines',
                    line: {{color: '{config["colors"][0]}', width: 1}},
                    name: 'TMP (Raw)',
                    opacity: 0.25
                }});

                // Trace 2: TMP SMA
                if (tmp_sma_values.length > 0) {{
                    traces.push({{
                        x: timestamps,
                        y: tmp_sma_values,
                        mode: 'lines',
                        line: {{color: '{config["colors"][0]}', width: 1.5}},
                        name: 'TMP (SMA)',
                        opacity: 0.5
                    }});
                }}

                // Trace 3: Cycle-start scatter
                if (csTimestamps.length > 0) {{
                    traces.push({{
                        x: csTimestamps,
                        y: csTMPs,
                        mode: 'markers',
                        marker: {{color: 'rgba(255, 140, 0, 0.6)', size: 4}},
                        name: 'Cycle-start TMP'
                    }});

                    // Trace 4: Irreversible fouling trend
                    const t0 = new Date(csTimestamps[0]).getTime();
                    const fitY = csTimestamps.map(t => {{
                        const sec = (new Date(t).getTime() - t0) / 1000;
                        return irrevSlope * sec + irrevIntercept;
                    }});
                    traces.push({{
                        x: csTimestamps,
                        y: fitY,
                        mode: 'lines',
                        line: {{color: 'black', width: 2.5, dash: 'dash'}},
                        name: 'Irreversible Fouling Trend'
                    }});

                    // Trace 5: Linear forecast
                    if (predictionDate) {{
                        const lastTime = new Date(csTimestamps[csTimestamps.length - 1]).getTime();
                        const predTime = new Date(predictionDate).getTime();
                        const lastSec = (lastTime - t0) / 1000;
                        const predSec = (predTime - t0) / 1000;
                        const fitAtLast = irrevSlope * lastSec + irrevIntercept;
                        const fitAtPred = irrevSlope * predSec + irrevIntercept;

                        const nPts = 20;
                        const fcX = [];
                        const fcY = [];
                        for (let i = 0; i < nPts; i++) {{
                            const frac = i / (nPts - 1);
                            const sec = lastSec + frac * (predSec - lastSec);
                            const t = new Date(t0 + sec * 1000);
                            fcX.push(t.toISOString());
                            fcY.push(irrevSlope * sec + irrevIntercept);
                        }}
                        traces.push({{
                            x: fcX,
                            y: fcY,
                            mode: 'lines',
                            line: {{color: '#d62728', width: 2.5, dash: 'dot'}},
                            name: `Linear Forecast (${{forecastHorizonDays}}d)`
                        }});

                        // Trace 6: Confidence cone
                        if (lowerBound !== null && upperBound !== null) {{
                            const coneXUpper = [];
                            const coneYUpper = [];
                            const coneXLower = [];
                            const coneYLower = [];
                            for (let i = 0; i < nPts; i++) {{
                                const frac = i / (nPts - 1);
                                const t = fcX[i];
                                const v = fcY[i];
                                coneXUpper.push(t);
                                coneYUpper.push(v + frac * (upperBound - fitAtPred));
                                coneXLower.push(t);
                                coneYLower.push(v + frac * (lowerBound - fitAtPred));
                            }}
                            traces.push({{
                                x: coneXUpper,
                                y: coneYUpper,
                                mode: 'lines',
                                line: {{width: 0}},
                                showlegend: false,
                                hoverinfo: 'skip'
                            }});
                            traces.push({{
                                x: coneXLower,
                                y: coneYLower,
                                mode: 'lines',
                                line: {{width: 0}},
                                fill: 'tonexty',
                                fillcolor: 'rgba(214, 39, 40, 0.12)',
                                name: '95% Confidence'
                            }});
                        }}
                    }}
                }}

                // Trace 7: 6-bar threshold
                const xMin = timestamps[0];
                const xMax = predictionDate || timestamps[timestamps.length - 1];
                traces.push({{
                    x: [xMin, xMax],
                    y: [thresholdTMP, thresholdTMP],
                    mode: 'lines',
                    line: {{color: 'rgba(220, 53, 69, 0.7)', width: 2, dash: 'longdash'}},
                    name: 'Operational Limit (6 bar)'
                }});

                // Layout with annotation
                const slopePerDay = irrevSlope * 86400;
                const absSlopeH = Math.abs(irrevSlope * 3600);
                let classification = 'low';
                let clsColor = '#28a745';
                if (absSlopeH >= 0.01) {{ classification = 'critical'; clsColor = '#dc3545'; }}
                else if (absSlopeH >= 0.005) {{ classification = 'high'; clsColor = '#fd7e14'; }}
                else if (absSlopeH >= 0.001) {{ classification = 'medium'; clsColor = '#ffc107'; }}

                let annText = `<b style="color:${{clsColor}}">⬤ Fouling: ${{classification.toUpperCase()}}</b>`;
                annText += `<br><b>Irrev. slope:</b> ${{slopePerDay.toFixed(4)}} bar/day (R²=${{rSquared.toFixed(3)}})`;
                annText += `<br><b>Model:</b> ${{modelType}}`;

                const layout = {{
                    title: '<b>TMP Forecast — Irreversible Fouling Trend</b>',
                    xaxis: {{
                        title: 'Time',
                        gridcolor: 'rgba(200, 200, 200, 0.3)',
                        showgrid: true,
                        rangeselector: {{
                            buttons: [
                                {{count: 1, label: '1h', step: 'hour', stepmode: 'backward'}},
                                {{count: 12, label: '12h', step: 'hour', stepmode: 'backward'}},
                                {{count: 1, label: '1d', step: 'day', stepmode: 'backward'}},
                                {{count: 7, label: '1w', step: 'day', stepmode: 'backward'}},
                                {{count: 1, label: '1m', step: 'month', stepmode: 'backward'}},
                                {{count: 6, label: '6m', step: 'month', stepmode: 'backward'}},
                                {{step: 'all', label: 'All'}}
                            ]
                        }},
                        rangeslider: {{visible: true, thickness: 0.05}}
                    }},
                    yaxis: {{
                        title: 'TMP (bar)',
                        gridcolor: 'rgba(200, 200, 200, 0.3)',
                        showgrid: true,
                        fixedrange: false
                    }},
                    height: 750,
                    margin: {{t: 80, b: 100, l: 80, r: 150}},
                    showlegend: true,
                    hovermode: 'x unified',
                    dragmode: 'zoom',
                    legend: {{
                        orientation: 'v',
                        yanchor: 'top',
                        y: 1,
                        xanchor: 'left',
                        x: 1.08
                    }},
                    annotations: [{{
                        x: 0.98,
                        y: 0.98,
                        xref: 'paper',
                        yref: 'paper',
                        text: annText,
                        showarrow: false,
                        bgcolor: 'rgba(255, 255, 255, 0.92)',
                        bordercolor: 'rgba(0, 0, 0, 0.3)',
                        borderwidth: 1,
                        borderpad: 8,
                        font: {{size: 11}},
                        align: 'left',
                        xanchor: 'right',
                        yanchor: 'top'
                    }}]
                }};

                Plotly.newPlot('plot-tmp-forecast', traces, layout, {{responsive: true}});
            }}
"""

    # Add cycle comparison plot
    if cycles:
        cycle_ids = [c["cycle_id"] for c in cycles]
        tmp_slopes = [c["tmp_slope"] for c in cycles]
        durations = [c["duration_hours"] for c in cycles]

        html += """            
            if (sectionId === 'cycle-comparison') {
                var trace1 = {
                    x: """
        html += str(cycle_ids)
        html += """,
                    y: """
        html += str(tmp_slopes)
        html += """,
                    type: 'bar',
                    name: 'TMP Slope (bar/hour)',
                    marker: {color: 'indianred'}
                };
                
                var trace2 = {
                    x: """
        html += str(cycle_ids)
        html += """,
                    y: """
        html += str(durations)
        html += """,
                    mode: 'lines+markers',
                    name: 'Duration (hours)',
                    yaxis: 'y2',
                    line: {color: 'steelblue', width: 2}
                };
                
                var layout = {
                    xaxis: {title: 'Cycle ID'},
                    yaxis: {title: 'TMP Slope (bar/hour)'},
                    yaxis2: {
                        title: 'Duration (hours)',
                        overlaying: 'y',
                        side: 'right',
                        fixedrange: true
                    },
                    height: 600,
                    margin: {t: 50, b: 80, l: 80, r: 80}
                };
                
                Plotly.newPlot('plot-cycle-comparison', [trace1, trace2], layout, {responsive: true});
            }
"""

    html += """        }
    </script>
</body>
</html>
"""

    # Save unified dashboard
    filepath = os.path.join(plots_dir, "index.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("  ✓ Created: Unified Navigation Dashboard")
    return filepath


def run_dashboard_app(config_path: str = "config.yaml") -> list:
    """
    Main dashboard generation pipeline

    Args:
        config_path: Path to configuration file

    Returns:
        List of generated file paths
    """
    logger.info("=" * 60)
    logger.info("📊 DASHBOARD APP STAGE")
    logger.info("=" * 60)

    # Load configuration
    config = load_config(config_path)
    viz_data_path = config["paths"]["processed_viz_data"]
    full_data_path = config["paths"]["cycles_data"]
    plots_dir = config["paths"]["plots_folder"]

    # Load data
    logger.info(f"Loading visualization data from: {viz_data_path}")
    df_viz = load_parquet(viz_data_path)

    logger.info(f"Loading full data from: {full_data_path}")
    df_full = load_parquet(full_data_path)

    # Load cycles
    try:
        cycle_summary_path = config["paths"]["cycle_summary_data"]
        cycle_df = load_parquet(cycle_summary_path)
        cycles = cycle_df.to_dict("records")
    except:
        cycles = []
        logger.warning("No cycle data available")

    # Load forecast
    try:
        with open(config["paths"]["forecast_json"], "r") as f:
            tmp_forecast = json.load(f)
    except:
        tmp_forecast = None
        logger.warning("No TMP forecast available")

    # Load permeability forecast
    try:
        with open(config["paths"]["permeability_forecast_json"], "r") as f:
            perm_forecast = json.load(f)
    except:
        perm_forecast = None
        logger.warning("No Permeability forecast available")

    # Combine forecasts
    forecasts = {"tmp": tmp_forecast, "permeability": perm_forecast}

    # Load KPIs
    try:
        with open(config["paths"]["kpis_json"], "r") as f:
            kpis = json.load(f)
    except:
        kpis = []
        logger.warning("No KPIs available")

    # Load alerts
    try:
        with open(config["paths"]["alerts_json"], "r") as f:
            alerts = json.load(f)
    except:
        alerts = []
        logger.warning("No alerts available")

    # Load chemical cleanings
    try:
        with open(
            os.path.join(config["paths"]["output_folder"], "chemical_cleanings.json"),
            "r",
        ) as f:
            cc_data = json.load(f)
            chemical_cleanings = cc_data.get("chemical_cleaning_timestamps", [])
    except:
        chemical_cleanings = []
        logger.warning("No chemical cleaning data available")

    # Load fouling rates
    try:
        with open(
            os.path.join(config["paths"]["output_folder"], "fouling_rates.json"), "r"
        ) as f:
            fouling_rates = json.load(f)
    except:
        fouling_rates = None
        logger.warning("No fouling rate data available")

    generated_files = []

    # Create Unified Dashboard (PRIMARY - All-in-one navigation app)
    unified_dashboard_file = create_unified_dashboard(
        kpis,
        alerts,
        forecasts,
        df_viz,
        df_full,
        cycles,
        plots_dir,
        config["visualization"],
    )
    if unified_dashboard_file:
        generated_files.append(unified_dashboard_file)
        logger.info(f"  🎯 Main dashboard: {unified_dashboard_file}")

    # Create standalone KPI Dashboard
    kpi_dashboard_file = create_kpi_dashboard(
        kpis, alerts, forecasts, plots_dir, config["visualization"]
    )
    if kpi_dashboard_file:
        generated_files.append(kpi_dashboard_file)

    # Create individual plots
    files = create_individual_plots(df_viz, plots_dir, config["visualization"], cycles)
    generated_files.extend(files)

    # Create combined parameters plot
    combined_file = create_combined_parameters_plot(
        df_viz, plots_dir, config["visualization"], cycles
    )
    if combined_file:
        generated_files.append(combined_file)

    # Create TMP plot with forecast
    tmp_file = create_tmp_plot_with_forecast(
        df_full,
        plots_dir,
        config,
        forecasts.get("tmp"),
        cycles,
        chemical_cleanings,
        fouling_rates,
    )
    if tmp_file:
        generated_files.append(tmp_file)

    # Create cycle comparison
    if cycles:
        cycle_file = create_cycle_comparison_plot(
            cycles, plots_dir, config["visualization"]
        )
        if cycle_file:
            generated_files.append(cycle_file)

    logger.info(f"\n✓ Generated {len(generated_files)} visualizations")
    logger.info(f"  Output directory: {plots_dir}")

    logger.info("=" * 60)
    logger.info("✓ DASHBOARD APP COMPLETE")
    logger.info("=" * 60)

    return generated_files


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run dashboard app
    files = run_dashboard_app()
    print(f"\n✓ Generated {len(files)} visualizations")
    print(f"✓ Open HTML files in your browser")
