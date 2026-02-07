"""
Plotting utilities for consistent visualization
"""

import plotly.graph_objects as go
from typing import Optional, List, Dict
import pandas as pd


def create_time_axis(
    time_range: Optional[pd.Timedelta], time_converted: bool = True
) -> dict:
    """
    Create an elegant time axis configuration based on data time range

    Args:
        time_range: Total time span of the data
        time_converted: Whether time is in datetime format

    Returns:
        Dictionary of x-axis configuration for Plotly
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
    tickformatstops = [
        # Less than 1 second - show milliseconds
        dict(dtickrange=[None, 1000], value="%H:%M:%S.%L"),
        # 1 second to 1 minute - show seconds
        dict(dtickrange=[1000, 60000], value="%H:%M:%S"),
        # 1 minute to 1 hour - show minutes and seconds
        dict(dtickrange=[60000, 3600000], value="%H:%M:%S"),
        # 1 hour to 4 hours - show hours and minutes with date
        dict(dtickrange=[3600000, 14400000], value="%H:%M\n%d %b"),
        # 4 hours to 1 week - show day only
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


def apply_color_palette(index: int, palette: Optional[List[str]] = None) -> str:
    """
    Get color from palette by index

    Args:
        index: Index into color palette
        palette: Optional custom color palette

    Returns:
        Color hex string
    """
    if palette is None:
        # Default professional palette
        palette = [
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

    return palette[index % len(palette)]


def get_plot_config(responsive: bool = True) -> dict:
    """
    Get standard Plotly plot configuration

    Args:
        responsive: Whether plot should be responsive

    Returns:
        Configuration dictionary for Plotly
    """
    return {
        "responsive": responsive,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToAdd": ["drawline", "drawopenpath", "eraseshape"],
        "scrollZoom": True,
        "toImageButtonOptions": {
            "format": "png",
            "height": 700,
            "width": 1200,
            "scale": 2,
        },
    }


def create_standard_layout(
    title: str,
    height: int = 700,
    show_legend: bool = True,
) -> dict:
    """
    Create standard plot layout

    Args:
        title: Plot title
        height: Plot height in pixels
        show_legend: Whether to show legend

    Returns:
        Layout dictionary for Plotly
    """
    return dict(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=26, family="Arial, sans-serif"),
            x=0.5,
            y=0.95,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=height,
        autosize=True,
        margin=dict(l=80, r=80, t=150, b=100),
        font=dict(family="Arial, sans-serif", size=14),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial, sans-serif"),
        showlegend=show_legend,
        legend=dict(
            x=1.02,
            y=1,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="rgba(150, 150, 150, 0.5)",
            borderwidth=1,
        )
        if show_legend
        else None,
        dragmode="zoom",
    )
