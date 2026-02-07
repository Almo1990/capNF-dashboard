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
) -> str:
    """
    Create special TMP plot with linear fit and forecast

    Args:
        df: Full DataFrame (not downsampled) for accurate slope
        plots_dir: Output directory
        config: Visualization configuration
        forecast: Forecast dictionary
        cycles: Cycle information

    Returns:
        Path to generated file
    """
    logger.info("Creating TMP plot with forecast...")

    fig = go.Figure()

    # Use visualization data for plotting
    df_viz_path = config["paths"]["processed_viz_data"]
    df_viz = load_parquet(df_viz_path)

    colors = config["visualization"]["colors"]

    # Add raw TMP
    fig.add_trace(
        go.Scatter(
            x=df_viz["TimeStamp"],
            y=df_viz["TMP"],
            mode="lines",
            line=dict(color=colors[0], width=1),
            name="TMP (Raw)",
            opacity=0.4,
        )
    )

    # Add TMP SMA
    if "TMP_SMA" in df_viz.columns:
        fig.add_trace(
            go.Scatter(
                x=df_viz["TimeStamp"],
                y=df_viz["TMP_SMA"],
                mode="lines",
                line=dict(color=colors[0], width=2),
                name="TMP (SMA)",
            )
        )

    # Add linear fit
    df_tmp = df[["TimeStamp", "TMP"]].dropna()
    slope_info = {}
    if len(df_tmp) > 1:
        from .utils.time_utils import convert_to_seconds_since_start

        time_numeric = convert_to_seconds_since_start(df_tmp["TimeStamp"])
        coeffs = np.polyfit(time_numeric, df_tmp["TMP"], 1)
        fit_values = coeffs[0] * time_numeric + coeffs[1]

        # Calculate R²
        y_mean = df_tmp["TMP"].mean()
        ss_tot = np.sum((df_tmp["TMP"] - y_mean) ** 2)
        ss_res = np.sum((df_tmp["TMP"] - fit_values) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        slope_info = {
            "slope": coeffs[0],
            "intercept": coeffs[1],
            "r_squared": r_squared,
        }

        fig.add_trace(
            go.Scatter(
                x=df_tmp["TimeStamp"],
                y=fit_values,
                mode="lines",
                line=dict(color="black", width=2, dash="dash"),
                name="Linear Fit",
            )
        )

    # Add forecast if available
    if forecast and "predicted_value" in forecast:
        last_time = df["TimeStamp"].iloc[-1]
        pred_time = pd.Timestamp(forecast["prediction_date"])

        fig.add_trace(
            go.Scatter(
                x=[last_time, pred_time],
                y=[df["TMP"].iloc[-1], forecast["predicted_value"]],
                mode="lines+markers",
                line=dict(color="red", width=2, dash="dot"),
                name="Forecast",
            )
        )

        # Add confidence bounds if available
        if "lower_bound" in forecast and "upper_bound" in forecast:
            fig.add_trace(
                go.Scatter(
                    x=[pred_time, pred_time, pred_time],
                    y=[
                        forecast["lower_bound"],
                        forecast["predicted_value"],
                        forecast["upper_bound"],
                    ],
                    mode="markers",
                    marker=dict(color="red", size=8),
                    name="95% CI",
                    showlegend=False,
                )
            )

    # Layout
    time_range = df["TimeStamp"].max() - df["TimeStamp"].min()
    layout = create_standard_layout("TMP vs Time (with Forecast)", height=700)
    layout.update(xaxis=create_time_axis(time_range, True))

    # Calculate slope for annotation
    if slope_info:
        slope_per_hour = slope_info["slope"] * 3600
        slope_per_day = slope_info["slope"] * 86400
        r_squared = slope_info["r_squared"]

        # Add slope annotation with model info
        model_type = forecast.get("model_type", "unknown") if forecast else "linear"
        layout.update(
            annotations=[
                dict(
                    x=0.98,
                    y=0.15,
                    xref="paper",
                    yref="paper",
                    text=f"<b>Model:</b> {model_type} (default)<br><b>Slope:</b> {slope_per_hour:.6f} bar/h ({slope_per_day:.4f} bar/day)<br><b>R²:</b> {r_squared:.4f}",
                    showarrow=False,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=5,
                    font=dict(size=12),
                    align="right",
                    xanchor="right",
                    yanchor="bottom",
                )
            ]
        )

    fig.update_layout(layout)

    # Save with custom JavaScript for dynamic recalculation
    html_str = fig.to_html(config=get_plot_config(), include_plotlyjs="cdn")

    # Prepare raw data for JavaScript
    df_tmp_js = df_tmp.copy()
    timestamps_js = df_tmp_js["TimeStamp"].astype(str).tolist()
    tmp_values_js = df_tmp_js["TMP"].tolist()
    forecast_horizon = forecast.get("forecast_horizon_days", 7) if forecast else 7
    model_type_js = forecast.get("model_type", "unknown") if forecast else "linear"

    # Add custom JavaScript for dynamic calculations
    custom_js = f"""
<script>
    // Store raw data
    const rawTimestamps = {timestamps_js};
    const rawTMPValues = {tmp_values_js};
    const forecastHorizonDays = {forecast_horizon};
    const originalModelType = "{model_type_js}";
    
    // Convert timestamps to numeric (milliseconds)
    const rawTimeNumeric = rawTimestamps.map(t => new Date(t).getTime());
    
    // Function to calculate linear regression
    function linearRegression(x, y) {{
        const n = x.length;
        if (n < 2) return null;
        
        const sumX = x.reduce((a, b) => a + b, 0);
        const sumY = y.reduce((a, b) => a + b, 0);
        const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
        const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
        
        const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        const intercept = (sumY - slope * sumX) / n;
        
        // Calculate R²
        const yMean = sumY / n;
        const yFit = x.map(xi => slope * xi + intercept);
        const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
        const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - yFit[i], 2), 0);
        const r2 = 1 - (ssRes / ssTot);
        
        return {{slope, intercept, r2}};
    }}
    
    // Function to update plot based on visible range
    function updateForecast(xRange) {{
        let startTime, endTime;
        
        if (xRange && xRange['xaxis.range[0]'] && xRange['xaxis.range[1]']) {{
            startTime = new Date(xRange['xaxis.range[0]']).getTime();
            endTime = new Date(xRange['xaxis.range[1]']).getTime();
        }} else {{
            // Use full range
            startTime = Math.min(...rawTimeNumeric);
            endTime = Math.max(...rawTimeNumeric);
        }}
        
        // Filter data to visible range
        const visibleIndices = rawTimeNumeric.map((t, i) => (t >= startTime && t <= endTime) ? i : -1).filter(i => i >= 0);
        
        if (visibleIndices.length < 2) return;
        
        const visibleTimes = visibleIndices.map(i => rawTimeNumeric[i]);
        const visibleTMP = visibleIndices.map(i => rawTMPValues[i]);
        
        // Calculate regression on visible data
        const result = linearRegression(visibleTimes, visibleTMP);
        if (!result) return;
        
        const {{slope, intercept, r2}} = result;
        
        // Convert slope to bar/hour and bar/day
        const slopePerHour = slope * 3600000; // milliseconds to hours
        const slopePerDay = slope * 86400000; // milliseconds to days
        
        // Update linear fit trace
        const fitX = [rawTimestamps[0], rawTimestamps[rawTimestamps.length - 1]];
        const fitY = [
            slope * rawTimeNumeric[0] + intercept,
            slope * rawTimeNumeric[rawTimeNumeric.length - 1] + intercept
        ];
        
        // Calculate forecast
        const lastTime = rawTimeNumeric[rawTimeNumeric.length - 1];
        const lastTMP = rawTMPValues[rawTMPValues.length - 1];
        const futureTime = lastTime + (forecastHorizonDays * 86400000); // days to milliseconds
        const predictedTMP = slope * futureTime + intercept;
        const futureDate = new Date(futureTime);
        
        // Update traces
        const updatedData = {{}};
        
        // Find and update Linear Fit trace
        const plotDiv = document.getElementsByClassName('plotly')[0];
        if (plotDiv && plotDiv.data) {{
            const fitTraceIndex = plotDiv.data.findIndex(trace => trace.name === 'Linear Fit');
            if (fitTraceIndex >= 0) {{
                Plotly.restyle(plotDiv, {{
                    'x': [fitX],
                    'y': [fitY]
                }}, [fitTraceIndex]);
            }}
            
            // Find and update Forecast trace
            const forecastTraceIndex = plotDiv.data.findIndex(trace => trace.name === 'Forecast');
            if (forecastTraceIndex >= 0) {{
                Plotly.restyle(plotDiv, {{
                    'x': [[rawTimestamps[rawTimestamps.length - 1], futureDate.toISOString()]],
                    'y': [[lastTMP, predictedTMP]]
                }}, [forecastTraceIndex]);
            }}
            
            // Determine if using full range or filtered range
            const isFullRange = visibleIndices.length === rawTimeNumeric.length;
            const modelInfo = isFullRange ? 
                `<b>Model:</b> ${{originalModelType}} (default)` : 
                `<b>Model:</b> linear regression (dynamic - filtered range)`;
            
            // Update annotation with slope and model info
            const annotation = {{
                x: 0.98,
                y: 0.15,
                xref: 'paper',
                yref: 'paper',
                text: `${{modelInfo}}<br><b>Slope:</b> ${{slopePerHour.toFixed(6)}} bar/h (${{slopePerDay.toFixed(4)}} bar/day)<br><b>R²:</b> ${{r2.toFixed(4)}}`,
                showarrow: false,
                bgcolor: 'rgba(255, 255, 255, 0.8)',
                bordercolor: 'black',
                borderwidth: 1,
                borderpad: 5,
                font: {{size: 12}},
                align: 'right',
                xanchor: 'right',
                yanchor: 'bottom'
            }};
            
            Plotly.relayout(plotDiv, {{'annotations': [annotation]}});
        }}
    }}
    
    // Listen for range changes
    document.addEventListener('DOMContentLoaded', function() {{
        const plotDiv = document.getElementsByClassName('plotly')[0];
        if (plotDiv) {{
            plotDiv.on('plotly_relayout', function(eventData) {{
                updateForecast(eventData);
            }});
            
            // Initial calculation
            updateForecast(null);
        }}
    }});
</script>
"""

    # Insert custom JS before closing body tag
    html_str = html_str.replace("</body>", f"{custom_js}</body>")

    filepath = os.path.join(plots_dir, "TMP_forecast.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_str)

    logger.info("  ✓ Created: TMP forecast plot with dynamic slope calculation")

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
    <title>PWN CapNF - KPI Dashboard</title>
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
            <h1>🏭 PWN CapNF System</h1>
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
            html_content += f'<div class="forecast-detail">⏱️ Time to 8 bar threshold: {days:.1f} days</div>'

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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PWN CapNF - Membrane Filtration Analytics</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
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
        
        /* Responsive */
        @media (max-width: 768px) {{
            .sidebar {{
                width: 100%;
                height: auto;
                position: relative;
            }}
            .main-content {{
                margin-left: 0;
                padding: 15px;
            }}
            .kpi-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <!-- Sidebar Navigation -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1>🏭 PWN CapNF</h1>
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
            html += f'                    <div class="forecast-detail">⏱️ Time to 8 bar threshold: {days:.1f} days</div>'

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
                <h3 class="plot-title">TMP vs Time (with Forecast)</h3>
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

    # Add TMP forecast plot
    # Prepare raw data for dynamic calculations
    df_tmp_full = df_full[["TimeStamp", "TMP"]].dropna()
    timestamps_js = df_tmp_full["TimeStamp"].astype(str).tolist()
    tmp_values_js = df_tmp_full["TMP"].tolist()
    tmp_forecast = forecasts.get("tmp") if forecasts else None
    forecast_horizon = (
        tmp_forecast.get("forecast_horizon_days", 7) if tmp_forecast else 7
    )
    model_type_index = (
        tmp_forecast.get("model_type", "linear") if tmp_forecast else "linear"
    )

    html += f"""            if (sectionId === 'tmp-forecast') {{
                // Store raw data for dynamic calculations
                const rawTimestamps = {timestamps_js};
                const rawTMPValues = {tmp_values_js};
                const forecastHorizonDays = {forecast_horizon};
                
                // Convert timestamps to numeric (milliseconds)
                const rawTimeNumeric = rawTimestamps.map(t => new Date(t).getTime());
                const originalModelType = '{model_type_index}';
                
                var tmpTrace = {{
                    x: """
    html += str(df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist())
    html += """,
                    y: """
    html += str(df_viz["TMP"].tolist())
    html += """,
                    mode: 'lines',
                    name: 'TMP (Raw)',
                    line: {color: '"""
    html += colors[0]
    html += """', width: 1},
                    opacity: 0.4
                };
                
                var traces = [tmpTrace];
"""

    if "TMP_SMA" in df_viz.columns:
        html += """                
                var tmpSmaTrace = {
                    x: """
        html += str(df_viz["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist())
        html += """,
                    y: """
        html += str(df_viz["TMP_SMA"].tolist())
        html += """,
                    mode: 'lines',
                    name: 'TMP (SMA)',
                    line: {color: '"""
        html += colors[0]
        html += """', width: 2}
                };
                traces.push(tmpSmaTrace);
"""

    # Add linear fit
    if len(df_full) > 1:
        from .utils.time_utils import convert_to_seconds_since_start

        df_tmp = df_full[["TimeStamp", "TMP"]].dropna()
        time_numeric = convert_to_seconds_since_start(df_tmp["TimeStamp"])
        coeffs = np.polyfit(time_numeric, df_tmp["TMP"], 1)
        fit_values = coeffs[0] * time_numeric + coeffs[1]

        # Calculate R²
        y_mean = df_tmp["TMP"].mean()
        ss_tot = np.sum((df_tmp["TMP"] - y_mean) ** 2)
        ss_res = np.sum((df_tmp["TMP"] - fit_values) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        slope_per_hour = coeffs[0] * 3600
        slope_per_day = coeffs[0] * 86400

        html += """                
                var fitTrace = {
                    x: """
        html += str(df_tmp["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist())
        html += """,
                    y: """
        html += str(fit_values.tolist())
        html += """,
                    mode: 'lines',
                    name: 'Linear Fit',
                    line: {color: 'black', width: 2, dash: 'dash'}
                };
                traces.push(fitTrace);
"""

    # Add forecast if available
    if tmp_forecast and "predicted_value" in tmp_forecast:
        last_time = df_full["TimeStamp"].iloc[-1]
        pred_time = pd.Timestamp(tmp_forecast["prediction_date"])
        last_tmp = df_full["TMP"].iloc[-1]

        html += """                
                var forecastTrace = {
                    x: ['"""
        html += last_time.strftime("%Y-%m-%d %H:%M:%S")
        html += """', '"""
        html += pred_time.strftime("%Y-%m-%d %H:%M:%S")
        html += """'],
                    y: ["""
        html += str(last_tmp)
        html += """, """
        html += str(tmp_forecast["predicted_value"])
        html += """],
                    mode: 'lines+markers',
                    name: 'Forecast',
                    line: {color: 'red', width: 2, dash: 'dot'}
                };
                traces.push(forecastTrace);
"""

    # Add initial slope annotation
    initial_slope_text = ""
    if len(df_full) > 1:
        initial_slope_text = f"<b>Model:</b> {model_type_index} (default)<br><b>Slope:</b> {slope_per_hour:.6f} bar/h ({slope_per_day:.4f} bar/day)<br><b>R²:</b> {r_squared:.4f}"

    html += f"""                
                var layout = {{
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
                    yaxis: {{title: 'TMP (bar)', fixedrange: false}},
                    annotations: [{{
                        x: 0.98,
                        y: 0.15,
                        xref: 'paper',
                        yref: 'paper',
                        text: '{initial_slope_text}',
                        showarrow: false,
                        bgcolor: 'rgba(255, 255, 255, 0.8)',
                        bordercolor: 'black',
                        borderwidth: 1,
                        borderpad: 5,
                        font: {{size: 12}},
                        align: 'right',
                        xanchor: 'right',
                        yanchor: 'bottom'
                    }}],
                    height: 750,
                    margin: {{t: 50, b: 100, l: 80, r: 80}},
                    hovermode: 'x unified',
                    dragmode: 'zoom'
                }};
                
                Plotly.newPlot('plot-tmp-forecast', traces, layout, {{responsive: true}});
                
                // Function to calculate linear regression
                function linearRegression(x, y) {{
                    const n = x.length;
                    if (n < 2) return null;
                    
                    const sumX = x.reduce((a, b) => a + b, 0);
                    const sumY = y.reduce((a, b) => a + b, 0);
                    const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
                    const sumX2 = x.reduce((sum, xi) => sum + xi * xi, 0);
                    
                    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
                    const intercept = (sumY - slope * sumX) / n;
                    
                    // Calculate R²
                    const yMean = sumY / n;
                    const yFit = x.map(xi => slope * xi + intercept);
                    const ssTot = y.reduce((sum, yi) => sum + Math.pow(yi - yMean, 2), 0);
                    const ssRes = y.reduce((sum, yi, i) => sum + Math.pow(yi - yFit[i], 2), 0);
                    const r2 = 1 - (ssRes / ssTot);
                    
                    return {{slope, intercept, r2}};
                }}
                
                // Function to update plot based on visible range
                function updateTMPForecast(xRange) {{
                    let startTime, endTime;
                    
                    if (xRange && xRange['xaxis.range[0]'] && xRange['xaxis.range[1]']) {{
                        startTime = new Date(xRange['xaxis.range[0]']).getTime();
                        endTime = new Date(xRange['xaxis.range[1]']).getTime();
                    }} else if (xRange && xRange['xaxis.range']) {{
                        startTime = new Date(xRange['xaxis.range'][0]).getTime();
                        endTime = new Date(xRange['xaxis.range'][1]).getTime();
                    }} else {{
                        // Use full range
                        startTime = Math.min(...rawTimeNumeric);
                        endTime = Math.max(...rawTimeNumeric);
                    }}
                    
                    // Filter data to visible range
                    const visibleIndices = rawTimeNumeric.map((t, i) => (t >= startTime && t <= endTime) ? i : -1).filter(i => i >= 0);
                    
                    if (visibleIndices.length < 2) return;
                    
                    const visibleTimes = visibleIndices.map(i => rawTimeNumeric[i]);
                    const visibleTMP = visibleIndices.map(i => rawTMPValues[i]);
                    
                    // Calculate regression on visible data
                    const result = linearRegression(visibleTimes, visibleTMP);
                    if (!result) return;
                    
                    const {{slope, intercept, r2}} = result;
                    
                    // Convert slope to bar/hour and bar/day
                    const slopePerHour = slope * 3600000; // milliseconds to hours
                    const slopePerDay = slope * 86400000; // milliseconds to days
                    
                    // Update linear fit trace
                    const fitX = [rawTimestamps[0], rawTimestamps[rawTimestamps.length - 1]];
                    const fitY = [
                        slope * rawTimeNumeric[0] + intercept,
                        slope * rawTimeNumeric[rawTimeNumeric.length - 1] + intercept
                    ];
                    
                    // Calculate forecast
                    const lastTime = rawTimeNumeric[rawTimeNumeric.length - 1];
                    const lastTMP = rawTMPValues[rawTMPValues.length - 1];
                    const futureTime = lastTime + (forecastHorizonDays * 86400000); // days to milliseconds
                    const predictedTMP = slope * futureTime + intercept;
                    const futureDate = new Date(futureTime).toISOString();
                    
                    // Update traces
                    const plotDiv = document.getElementById('plot-tmp-forecast');
                    if (plotDiv && plotDiv.data) {{
                        // Find and update Linear Fit trace
                        const fitTraceIndex = plotDiv.data.findIndex(trace => trace.name === 'Linear Fit');
                        if (fitTraceIndex >= 0) {{
                            Plotly.restyle(plotDiv, {{
                                'x': [fitX],
                                'y': [fitY]
                            }}, [fitTraceIndex]);
                        }}
                        
                        // Find and update Forecast trace
                        const forecastTraceIndex = plotDiv.data.findIndex(trace => trace.name === 'Forecast');
                        if (forecastTraceIndex >= 0) {{
                            Plotly.restyle(plotDiv, {{
                                'x': [[rawTimestamps[rawTimestamps.length - 1], futureDate]],
                                'y': [[lastTMP, predictedTMP]]
                            }}, [forecastTraceIndex]);
                        }}
                        
                        // Determine if using full range or filtered range
                        const isFullRange = visibleIndices.length === rawTimeNumeric.length;
                        const modelInfo = isFullRange ?
                            `<b>Model:</b> ${{originalModelType}} (default)` :
                            `<b>Model:</b> linear regression (dynamic - filtered range)`;
                        
                        // Update annotation with slope and model info
                        const annotation = {{
                            x: 0.98,
                            y: 0.15,
                            xref: 'paper',
                            yref: 'paper',
                            text: `${{modelInfo}}<br><b>Slope:</b> ${{slopePerHour.toFixed(6)}} bar/h (${{slopePerDay.toFixed(4)}} bar/day)<br><b>R²:</b> ${{r2.toFixed(4)}}`,
                            showarrow: false,
                            bgcolor: 'rgba(255, 255, 255, 0.8)',
                            bordercolor: 'black',
                            borderwidth: 1,
                            borderpad: 5,
                            font: {{size: 12}},
                            align: 'right',
                            xanchor: 'right',
                            yanchor: 'bottom'
                        }};
                        
                        Plotly.relayout(plotDiv, {{'annotations': [annotation]}});
                    }}
                }}
                
                // Listen for range changes on TMP forecast plot
                const tmpPlotDiv = document.getElementById('plot-tmp-forecast');
                if (tmpPlotDiv) {{
                    tmpPlotDiv.on('plotly_relayout', function(eventData) {{
                        updateTMPForecast(eventData);
                    }});
                }}
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
        df_full, plots_dir, config, forecasts.get("tmp"), cycles
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
