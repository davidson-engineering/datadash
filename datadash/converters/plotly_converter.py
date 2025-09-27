# Plotly Property Converter
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from typing import Dict, Any
import plotly.graph_objects as go

from ..standard.properties import (
    StandardTraceProperties,
    LineStyle,
    MarkerSymbol,
    TraceMode,
)


class PlotlyPropertyConverter:
    """Converts standard properties to Plotly-specific format."""

    # Mapping from standard line styles to Plotly line styles
    LINE_STYLE_MAP = {
        LineStyle.SOLID: "solid",
        LineStyle.DASHED: "dash",
        LineStyle.DOTTED: "dot",
        LineStyle.DASH_DOT: "dashdot",
    }

    # Mapping from standard marker symbols to Plotly symbols
    MARKER_SYMBOL_MAP = {
        MarkerSymbol.CIRCLE: "circle",
        MarkerSymbol.SQUARE: "square",
        MarkerSymbol.TRIANGLE: "triangle-up",
        MarkerSymbol.DIAMOND: "diamond",
        MarkerSymbol.CROSS: "cross",
        MarkerSymbol.PLUS: "cross",  # Plotly uses 'cross' for both
        MarkerSymbol.STAR: "star",
        MarkerSymbol.X: "x",
    }

    # Mapping from standard modes to Plotly modes
    MODE_MAP = {
        TraceMode.LINES: "lines",
        TraceMode.MARKERS: "markers",
        TraceMode.LINES_AND_MARKERS: "lines+markers",
        TraceMode.TEXT: "text",
    }

    @classmethod
    def convert_properties(cls, props: StandardTraceProperties) -> Dict[str, Any]:
        """
        Convert standard properties to Plotly trace properties.

        Args:
            props: StandardTraceProperties object

        Returns:
            Dictionary of Plotly trace properties
        """
        plotly_props = {}

        # Basic properties
        plotly_props["visible"] = props.visible
        plotly_props["opacity"] = props.opacity
        plotly_props["mode"] = cls.MODE_MAP[TraceMode(props.mode)]

        # Line properties
        if props.mode in (TraceMode.LINES, TraceMode.LINES_AND_MARKERS):
            plotly_props["line"] = {
                "color": props.line_color,
                "width": props.line_width,
                "dash": cls.LINE_STYLE_MAP[props.line_style],
            }

        # Marker properties
        if props.mode in (TraceMode.MARKERS, TraceMode.LINES_AND_MARKERS):
            marker_dict = {
                "color": props.marker_color,
                "size": props.marker_size,
                "symbol": cls.MARKER_SYMBOL_MAP[props.marker_symbol],
            }

            # Marker line properties
            if props.marker_line_width > 0:
                marker_dict["line"] = {
                    "color": props.marker_line_color,
                    "width": props.marker_line_width,
                }

            plotly_props["marker"] = marker_dict

        # Text properties
        if props.mode == TraceMode.TEXT:
            plotly_props["textfont"] = {
                "color": props.text_color,
                "size": props.text_size,
                "family": props.text_font,
            }

        # Fill properties
        if props.fill_color is not None:
            plotly_props["fillcolor"] = props.fill_color
            plotly_props["fill"] = "tonexty"  # Default fill mode
            plotly_props["fillcolor"] = (
                f"rgba{cls._hex_to_rgba(props.fill_color, props.fill_opacity)}"
            )

        # Hover properties
        if props.hover_text is not None:
            plotly_props["hovertext"] = props.hover_text

        if props.hover_template is not None:
            plotly_props["hovertemplate"] = props.hover_template

        # Legend properties
        plotly_props["showlegend"] = props.show_legend
        if props.legend_group is not None:
            plotly_props["legendgroup"] = props.legend_group

        # Custom properties (pass through if compatible)
        for key, value in props.custom.items():
            if cls._is_valid_plotly_property(key):
                plotly_props[key] = value

        return plotly_props

    @classmethod
    def _hex_to_rgba(cls, hex_color: str, opacity: float) -> str:
        """Convert hex color to RGBA string with opacity."""
        # Remove '#' if present
        hex_color = hex_color.lstrip("#")

        # Convert to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        return f"({r}, {g}, {b}, {opacity})"

    @classmethod
    def _is_valid_plotly_property(cls, prop_name: str) -> bool:
        """Check if a property name is valid for Plotly traces."""
        # List of common Plotly trace properties
        # This is a simplified check - in practice you might want a more comprehensive list
        valid_props = {
            "name",
            "text",
            "textposition",
            "texttemplate",
            "hoverinfo",
            "hoveron",
            "hovertemplate",
            "xaxis",
            "yaxis",
            "orientation",
            "groupnorm",
            "stackgroup",
            "connectgaps",
            "cliponaxis",
        }
        return prop_name in valid_props

    @classmethod
    def create_trace_from_standard(
        cls,
        trace_type: str,
        x_data,
        y_data,
        z_data=None,
        props: StandardTraceProperties = None,
        name: str = None,
    ) -> go.Scatter | go.Scatter3d:
        """
        Create a Plotly trace from standard format.

        Args:
            trace_type: "2d" or "3d"
            x_data: X coordinate data
            y_data: Y coordinate data
            z_data: Z coordinate data (for 3D traces)
            props: Standard properties object
            name: Trace name

        Returns:
            Plotly trace object
        """
        # Convert properties
        plotly_props = {}
        if props is not None:
            plotly_props = cls.convert_properties(props)

        # Add name if provided
        if name is not None:
            plotly_props["name"] = name

        # Create appropriate trace type
        if trace_type == "3d" and z_data is not None:
            return go.Scatter3d(x=x_data, y=y_data, z=z_data, **plotly_props)
        else:
            return go.Scatter(x=x_data, y=y_data, **plotly_props)


# Convenience function
def convert_standard_to_plotly(props: StandardTraceProperties) -> Dict[str, Any]:
    """
    Convenience function to convert standard properties to Plotly format.

    Args:
        props: StandardTraceProperties object

    Returns:
        Dictionary of Plotly trace properties
    """
    return PlotlyPropertyConverter.convert_properties(props)
