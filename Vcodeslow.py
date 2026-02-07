import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os


def tsv_to_excel_with_plots():
    """
    Combines all TSV files in the folder and creates elegant time-axis plots
    """
    # Find all TSV files in the current directory
    tsv_files = [f for f in os.listdir(".") if f.endswith(".tsv")]

    if not tsv_files:
        print("Error: No TSV files found in the current directory")
        return

    print(f"Found {len(tsv_files)} TSV file(s):")
    for f in tsv_files:
        print(f"  - {f}")

    base_name = "combined_data"
    output_file = f"{base_name}_filtered.xlsx"

    try:
        # Read and combine all TSV files
        print("\nCombining all TSV files...")
        dfs = []
        for tsv_file in tsv_files:
            print(f"  Reading {tsv_file}...")
            temp_df = pd.read_csv(tsv_file, sep="\t", engine="python")
            dfs.append(temp_df)

        # Combine all dataframes
        df = pd.concat(dfs, ignore_index=True)
        print(f"✓ Combined {len(tsv_files)} file(s) into {len(df)} total rows")

        # Save combined data to CSV before filtering
        combined_csv = f"{base_name}_unfiltered.csv"
        df.to_csv(combined_csv, index=False)
        print(f"✓ Saved combined unfiltered data: {combined_csv}")

        # --- STEP 1: DEFINE USER BANDWIDTHS ---
        # Modify these values to set your operational ranges
        bandwidths = {
            "TMP": {"min": 0.1, "max": 8},
            "Permeability TC": {"min": 0, "max": 20},
            "Mem. retention": {"min": 0, "max": 50},
            "Sys. retention": {"min": 0, "max": 50},
            "Flux": {"min": 0, "max": 30},
        }

        # --- STEP 1: DEFINE USER BANDWIDTHS ---
        # Modify these values to set your operational ranges
        bandwidths = {
            "TMP": {"min": 0, "max": 8},
            "Permeability TC": {"min": 0, "max": 20},
            "Mem. retention": {"min": 0, "max": 50},
            "Sys. retention": {"min": 0, "max": 50},
            "Flux": {"min": 0, "max": 30},
        }

        # Select common columns
        common_columns = ["TimeStamp"] + list(bandwidths.keys())
        columns_to_keep = [col for col in common_columns if col in df.columns]

        df_filtered = df[columns_to_keep].copy()

        # --- STEP 2: APPLY THE FILTERING ---
        print("\nApplying bandwidth filters:")
        original_count = len(df_filtered)

        for param, bounds in bandwidths.items():
            if param in df_filtered.columns:
                # Remove rows outside the min/max range
                initial_count = len(df_filtered)
                df_filtered = df_filtered[
                    (df_filtered[param] >= bounds["min"])
                    & (df_filtered[param] <= bounds["max"])
                ]
                removed = initial_count - len(df_filtered)
                print(
                    f"  - {param}: Filtered [{bounds['min']} to {bounds['max']}]. Removed {removed} outliers."
                )

        final_count = len(df_filtered)
        print(f"✓ Total data points removed: {original_count - final_count}")

        # Convert TimeStamp if present
        time_converted = False
        if "TimeStamp" in df_filtered.columns:
            try:
                # 1. Ensure column is string type
                df_filtered["TimeStamp"] = df_filtered["TimeStamp"].astype(str)

                # 2. Fix format: Replace space before last digits with a dot
                # Turns "2026-01-28 03:30:04 48" into "2026-01-28 03:30:04.48"
                df_filtered["TimeStamp"] = df_filtered["TimeStamp"].str.replace(
                    r" (\d+)$", r".\1", regex=True
                )

                # 3. Convert to datetime with various format attempts
                try:
                    df_filtered["TimeStamp"] = pd.to_datetime(
                        df_filtered["TimeStamp"], format="%Y-%m-%d %H:%M:%S.%f"
                    )
                except Exception as e:
                    print(f"Note: Could not convert with specific format: {e}")
                    df_filtered["TimeStamp"] = pd.to_datetime(df_filtered["TimeStamp"])

                df_filtered = df_filtered.sort_values("TimeStamp")
                time_converted = True

                # Show time range info
                time_range = (
                    df_filtered["TimeStamp"].max() - df_filtered["TimeStamp"].min()
                )
                print("✓ TimeStamp converted to datetime format")
                print(f"  Time range: {time_range}")
                print(f"  Start: {df_filtered['TimeStamp'].min()}")
                print(f"  End: {df_filtered['TimeStamp'].max()}")
            except Exception as e:
                print(f"Note: Could not convert TimeStamp to datetime: {e}")
                time_converted = False

        # Write to Excel
        df_filtered.to_excel(output_file, index=False, sheet_name="Data")
        print(f"✓ Saved Excel file: {output_file}")

        # Create plots with elegant time axis
        create_elegant_plots(df_filtered, base_name, time_converted)

    except Exception as e:
        print(f"Error: {e}")


def create_elegant_plots(df, base_name, time_converted):
    """
    Create plots with elegant time axis formatting
    """
    # Create output directory
    plots_dir = f"{base_name}_plots"
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nCreating elegant plots in '{plots_dir}'...")

    # Create individual plots for each numeric column (excluding TimeStamp)
    numeric_cols = [
        col
        for col in df.columns
        if col != "TimeStamp" and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Professional color palette
    colors = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#d62728",  # red
        "#ff7f0e",  # orange
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow-green
        "#17becf",  # cyan
    ]

    # Calculate optimal SMA window based on data density
    if time_converted and len(df["TimeStamp"]) > 1:
        time_range_total = df["TimeStamp"].max() - df["TimeStamp"].min()

        if time_range_total.days > 7:
            sma_window = "1h"
        elif time_range_total.days > 1:
            sma_window = "30min"
        elif time_range_total.total_seconds() > 3600:
            sma_window = "15min"
        else:
            sma_window = "5min"

        print(f"  Individual plots using SMA window: {sma_window}")
    else:
        sma_window = 50
        print(f"  Individual plots using SMA window: {sma_window} points")

    for i, col in enumerate(numeric_cols):
        if "TimeStamp" in df.columns:
            fig = go.Figure()

            # Add raw data trace (lighter)
            fig.add_trace(
                go.Scatter(
                    x=df["TimeStamp"],
                    y=df[col],
                    mode="lines",
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2,
                        shape="spline",
                        smoothing=0.2,
                    ),
                    name=f"{col} (Raw)",
                    opacity=0.4,
                )
            )

            # Calculate and add SMA trace
            if time_converted and isinstance(sma_window, str):
                df_temp = df.set_index("TimeStamp")
                sma_values = df_temp[col].rolling(sma_window, min_periods=1).mean()
                sma_time = sma_values.index
            else:
                sma_values = df[col].rolling(window=sma_window, min_periods=1).mean()
                sma_time = df["TimeStamp"]

            fig.add_trace(
                go.Scatter(
                    x=sma_time,
                    y=sma_values,
                    mode="lines",
                    line=dict(
                        color=colors[i % len(colors)],
                        width=3,
                    ),
                    name=f"{col} (SMA)",
                    fill="tozeroy" if i == 0 else None,
                    fillcolor=f"rgba{tuple(int(colors[i % len(colors)][j : j + 2], 16) for j in (1, 3, 5)) + (0.2,)}"
                    if i == 0
                    else None,
                )
            )

            # Calculate time range for smart tick formatting
            time_range = None
            if time_converted and len(df["TimeStamp"]) > 1:
                time_range = df["TimeStamp"].max() - df["TimeStamp"].min()

            # Update layout with elegant time axis
            fig.update_layout(
                title=dict(
                    text=f"<b>{col} vs Time</b>",
                    font=dict(size=26, family="Arial, sans-serif"),
                    x=0.5,
                    y=0.95,
                ),
                xaxis=create_elegant_time_axis(time_range, time_converted),
                yaxis=dict(
                    title=dict(
                        text=col, font=dict(size=18, family="Arial, sans-serif")
                    ),
                    gridcolor="rgba(200, 200, 200, 0.3)",
                    gridwidth=1,
                    minor_gridcolor="rgba(220, 220, 220, 0.1)",
                    showgrid=True,
                    zeroline=True,
                    zerolinecolor="rgba(150, 150, 150, 0.5)",
                    zerolinewidth=1,
                    fixedrange=False,
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=700,
                autosize=True,
                margin=dict(l=80, r=50, t=150, b=100),
                font=dict(family="Arial, sans-serif", size=14),
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="white", font_size=14, font_family="Arial, sans-serif"
                ),
                showlegend=True,
                legend=dict(
                    x=1.02,
                    y=1,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(150, 150, 150, 0.5)",
                    borderwidth=1,
                ),
                dragmode="zoom",
            )

            # Add annotations for min/max values if data is reasonable
            if len(df) > 10 and df[col].notna().sum() > 0:
                min_val = df[col].min()
                max_val = df[col].max()
                min_time = df.loc[df[col].idxmin(), "TimeStamp"]
                max_time = df.loc[df[col].idxmax(), "TimeStamp"]

                fig.add_annotation(
                    x=min_time,
                    y=min_val,
                    text=f"Min: {min_val:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=40,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=colors[i % len(colors)],
                    borderwidth=1,
                )

                fig.add_annotation(
                    x=max_time,
                    y=max_val,
                    text=f"Max: {max_val:.3f}",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40,
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor=colors[i % len(colors)],
                    borderwidth=1,
                )

            # Save the plot
            filename = f"{col.replace(' ', '_').replace('.', '_')}.html"
            filepath = os.path.join(plots_dir, filename)

            fig.write_html(
                filepath,
                config={
                    "responsive": True,
                    "displayModeBar": True,
                    "displaylogo": False,
                    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
                    "scrollZoom": True,
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": f"{col}_plot",
                        "height": 700,
                        "width": 1200,
                        "scale": 2,
                    },
                },
                include_plotlyjs="cdn",
                full_html=True,
                auto_open=False,
            )

            print(f"✓ Created: {col} plot")

    # Create elegant dashboard
    if len(numeric_cols) >= 2 and "TimeStamp" in df.columns:
        create_elegant_dashboard(df, plots_dir, numeric_cols, time_converted)

    # Create full-page combined view
    if len(numeric_cols) >= 1 and "TimeStamp" in df.columns:
        create_fullpage_combined_view(df, plots_dir, numeric_cols, time_converted)

    print(f"\n🎨 All elegant plots saved to '{plots_dir}' folder")
    print(
        "📊 Open HTML files and use browser's full-screen mode (F11) for best experience"
    )


def create_elegant_time_axis(time_range, time_converted):
    """
    Create an elegant time axis configuration based on data time range
    """
    if not time_converted or time_range is None:
        # If time is not datetime or range unknown, use default
        return dict(
            title=dict(text="Time", font=dict(size=18, family="Arial, sans-serif")),
            gridcolor="rgba(200, 200, 200, 0.3)",
            showgrid=True,
            showline=True,
            linecolor="rgba(150, 150, 150, 0.5)",
            linewidth=2,
            mirror=True,
        )

    # Use tickformatstops for dynamic formatting based on zoom level
    # This ensures proper time display when zooming to 1h, 6h, 12h etc.
    # Note: dtickrange is based on tick INTERVAL, not visible range
    # When viewing 1 week with ~25 ticks, tick interval ≈ 7 hours
    tickformatstops = [
        # Less than 1 second - show milliseconds
        dict(dtickrange=[None, 1000], value="%H:%M:%S.%L"),
        # 1 second to 1 minute - show seconds
        dict(dtickrange=[1000, 60000], value="%H:%M:%S"),
        # 1 minute to 1 hour - show minutes and seconds
        dict(dtickrange=[60000, 3600000], value="%H:%M:%S"),
        # 1 hour to 4 hours - show hours and minutes with date (tight zoom like 12h view)
        dict(dtickrange=[3600000, 14400000], value="%H:%M\n%d %b"),
        # 4 hours to 1 week - show day only (no time) - covers 1w view with ~7h tick intervals
        dict(dtickrange=[14400000, 604800000], value="%a %d %b"),
        # 1 week to 1 month - show date
        dict(dtickrange=[604800000, "M1"], value="%d %b %Y"),
        # 1 month to 1 year - show month
        dict(dtickrange=["M1", "M12"], value="%b %Y"),
        # More than 1 year - show year
        dict(dtickrange=["M12", None], value="%Y"),
    ]

    return dict(
        title=dict(
            text="Time", font=dict(size=18, family="Arial, sans-serif"), standoff=15
        ),
        type="date",
        tickformatstops=tickformatstops,
        hoverformat="%d %b %Y, %H:%M:%S",
        tickangle=-45,
        nticks=10,
        automargin=True,
        gridcolor="rgba(200, 200, 200, 0.3)",
        gridwidth=1,
        showgrid=True,
        showline=True,
        linecolor="rgba(150, 150, 150, 0.5)",
        linewidth=2,
        mirror=True,
        rangeslider=dict(
            visible=True, thickness=0.05, bgcolor="rgba(240, 240, 240, 0.8)"
        ),
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1h", step="hour", stepmode="backward"),
                    dict(count=6, label="6h", step="hour", stepmode="backward"),
                    dict(count=12, label="12h", step="hour", stepmode="backward"),
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
            bgcolor="rgba(255, 255, 255, 0.9)",
            activecolor="rgba(100, 150, 255, 0.8)",
            bordercolor="rgba(150, 150, 150, 0.5)",
            borderwidth=1,
            font=dict(size=11, family="Arial, sans-serif"),
            x=0,
            xanchor="left",
            y=1.15,
            yanchor="top",
        ),
    )


def create_elegant_dashboard(df, plots_dir, numeric_cols, time_converted):
    """
    Create an elegant dashboard with synchronized time axis
    """
    # Determine grid layout
    n_plots = min(len(numeric_cols), 4)
    rows = 2 if n_plots > 2 else 1
    cols = 2 if n_plots > 1 else 1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"<b>{col}</b>" for col in numeric_cols[:n_plots]],
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
        shared_xaxes="all",  # Strongly link all x-axes
        x_title="Time" if rows == 2 else None,
    )

    colors = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#8c564b"]

    # Calculate optimal SMA window based on data density
    # Use time-based rolling window for better results with time series
    if time_converted and len(df["TimeStamp"]) > 1:
        time_range_total = df["TimeStamp"].max() - df["TimeStamp"].min()

        # Determine window size based on data span
        if time_range_total.days > 7:
            # For week+ data: 1 hour window
            sma_window = "1h"
        elif time_range_total.days > 1:
            # For multi-day data: 30 minute window
            sma_window = "30min"
        elif time_range_total.total_seconds() > 3600:
            # For hour+ data: 15 minute window
            sma_window = "15min"
        else:
            # For short data: 5 minute window
            sma_window = "5min"

        print(f"  Dashboard using SMA window: {sma_window}")
    else:
        # Fallback to point-based window if time conversion failed
        sma_window = 50  # 50 points
        print(f"  Dashboard using SMA window: {sma_window} points")

    for i, col in enumerate(numeric_cols[:n_plots]):
        row = (i // cols) + 1
        col_num = (i % cols) + 1

        # Add raw data trace (lighter, semi-transparent)
        fig.add_trace(
            go.Scatter(
                x=df["TimeStamp"],
                y=df[col],
                mode="lines",
                line=dict(
                    color=colors[i % len(colors)],
                    width=1.5,
                    shape="spline",
                    smoothing=0.2,
                ),
                name=f"{col} (Raw)",
                opacity=0.5,
                showlegend=False,
            ),
            row=row,
            col=col_num,
        )

        # Calculate and add SMA trace (bold trend line)
        if time_converted and isinstance(sma_window, str):
            # Time-based rolling window
            df_temp = df.set_index("TimeStamp")
            sma_values = df_temp[col].rolling(sma_window, min_periods=1).mean()
            sma_time = sma_values.index
        else:
            # Point-based rolling window
            sma_values = df[col].rolling(window=sma_window, min_periods=1).mean()
            sma_time = df["TimeStamp"]

        fig.add_trace(
            go.Scatter(
                x=sma_time,
                y=sma_values,
                mode="lines",
                line=dict(
                    color=colors[i % len(colors)],
                    width=3,
                ),
                name=f"{col} (SMA)",
                showlegend=False,
            ),
            row=row,
            col=col_num,
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text="<b>Performance Dashboard</b>",
            font=dict(size=28, family="Arial, sans-serif"),
            x=0.5,
            y=0.98,
        ),
        height=900,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", size=14),
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=80, r=50, t=150, b=120),
        dragmode="zoom",
    )

    # Configure time axis for bottom plots
    time_range = None
    if time_converted and len(df["TimeStamp"]) > 1:
        time_range = df["TimeStamp"].max() - df["TimeStamp"].min()

    # Get full time axis config and create versions with/without controls
    full_time_axis = create_elegant_time_axis(time_range, time_converted)

    # Version with rangeselector only (no rangeslider) - positioned for dashboard
    time_axis_selector_only = full_time_axis.copy()
    time_axis_selector_only.pop("rangeslider", None)
    # Reposition rangeselector for dashboard - left side, just above graphs
    if "rangeselector" in time_axis_selector_only:
        time_axis_selector_only["rangeselector"]["x"] = 0
        time_axis_selector_only["rangeselector"]["xanchor"] = "left"
        time_axis_selector_only["rangeselector"]["y"] = 1.12
        time_axis_selector_only["rangeselector"]["yanchor"] = "top"

    # Version with no controls at all
    time_axis_no_controls = full_time_axis.copy()
    time_axis_no_controls.pop("rangeslider", None)
    time_axis_no_controls.pop("rangeselector", None)

    for i in range(1, n_plots + 1):
        row = (i + cols - 1) // cols
        col_num = (i - 1) % cols + 1

        if row == rows:  # Bottom row gets x-axis
            if (
                col_num == 1
            ):  # Only first bottom plot gets rangeselector (no rangeslider)
                fig.update_xaxes(
                    **time_axis_selector_only,
                    row=row,
                    col=col_num,
                )
            else:  # Other bottom plots get time axis without any controls
                fig.update_xaxes(
                    **time_axis_no_controls,
                    row=row,
                    col=col_num,
                )
        else:
            fig.update_xaxes(
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.3)",
                row=row,
                col=col_num,
            )

        # Configure y-axis
        col_name = numeric_cols[i - 1]
        fig.update_yaxes(
            title_text=col_name if col_num == 1 else "",
            title_font=dict(size=14),
            gridcolor="rgba(200, 200, 200, 0.3)",
            fixedrange=False,
            row=row,
            col=col_num,
        )

    # Save dashboard
    dashboard_path = os.path.join(plots_dir, "elegant_dashboard.html")

    fig.write_html(
        dashboard_path,
        config={
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "scrollZoom": True,
        },
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )

    print("✓ Created: Elegant dashboard")


def create_fullpage_combined_view(df, plots_dir, numeric_cols, time_converted):
    """
    Create a beautiful full-page combined view of all parameters
    """
    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#2ca02c",
        "#d62728",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    # Calculate optimal SMA window
    if time_converted and len(df["TimeStamp"]) > 1:
        time_range_total = df["TimeStamp"].max() - df["TimeStamp"].min()

        if time_range_total.days > 7:
            sma_window = "1h"
        elif time_range_total.days > 1:
            sma_window = "30min"
        elif time_range_total.total_seconds() > 3600:
            sma_window = "15min"
        else:
            sma_window = "5min"

        print(f"  Combined view using SMA window: {sma_window}")
    else:
        sma_window = 50
        print(f"  Combined view using SMA window: {sma_window} points")

    for i, col in enumerate(numeric_cols[:8]):  # Limit to 8 parameters
        # Add raw data trace (lighter)
        fig.add_trace(
            go.Scatter(
                x=df["TimeStamp"],
                y=df[col],
                mode="lines",
                name=f"{col} (Raw)",
                line=dict(
                    color=colors[i % len(colors)],
                    width=1.5,
                    shape="spline",
                    smoothing=0.2,
                    dash="dot",
                ),
                opacity=0.4,
                visible=True if i < 3 else "legendonly",
            )
        )

        # Calculate and add SMA trace
        if time_converted and isinstance(sma_window, str):
            df_temp = df.set_index("TimeStamp")
            sma_values = df_temp[col].rolling(sma_window, min_periods=1).mean()
            sma_time = sma_values.index
        else:
            sma_values = df[col].rolling(window=sma_window, min_periods=1).mean()
            sma_time = df["TimeStamp"]

        fig.add_trace(
            go.Scatter(
                x=sma_time,
                y=sma_values,
                mode="lines",
                name=f"{col} (SMA)",
                line=dict(
                    color=colors[i % len(colors)],
                    width=2.5,
                ),
                visible=True if i < 3 else "legendonly",
                legendgroup=col,
            )
        )

    # Calculate time range for axis formatting
    time_range = None
    if time_converted and len(df["TimeStamp"]) > 1:
        time_range = df["TimeStamp"].max() - df["TimeStamp"].min()

    fig.update_layout(
        title=dict(
            text="<b>All Parameters - Combined View</b>",
            font=dict(size=30, family="Arial, sans-serif"),
            x=0.5,
            y=0.97,
        ),
        xaxis=create_elegant_time_axis(time_range, time_converted),
        yaxis=dict(
            title=dict(text="Value", font=dict(size=20, family="Arial, sans-serif")),
            gridcolor="rgba(200, 200, 200, 0.3)",
            showgrid=True,
            fixedrange=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=800,
        autosize=True,
        margin=dict(l=80, r=50, t=150, b=100),
        font=dict(family="Arial, sans-serif", size=14),
        hovermode="x unified",
        legend=dict(
            title=dict(text="<b>Parameters</b>"),
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="rgba(150, 150, 150, 0.5)",
            borderwidth=1,
            font=dict(size=12),
        ),
        dragmode="zoom",
    )

    # Save combined view
    combined_path = os.path.join(plots_dir, "combined_view.html")

    fig.write_html(
        combined_path,
        config={
            "responsive": True,
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["drawline", "eraseshape"],
            "scrollZoom": True,
        },
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=False,
    )

    print("✓ Created: Combined view")


if __name__ == "__main__":
    print("=" * 60)
    print("📈 ELEGANT TSV VISUALIZER")
    print("=" * 60)
    print("Creates professional plots with intelligent time axis formatting")
    print("=" * 60)
    print()

    tsv_to_excel_with_plots()
