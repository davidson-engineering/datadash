# Unified Builder Classes for Visualization Components
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

import logging
from typing import Union
import numpy as np
from mergedeep import merge, Strategy

# Import core utilities
from .trace import get_plot_range, TraceBuilder, TraceConstructor, create_trace_constructor
from .figure import PlotFigure, PlotFigure3D, CombinedAxisFigure, SubplotsFigure
from ..themes.manager import get_theme_manager
from .layout import PlotLayoutBuilder
from plotly.colors import sample_colorscale

import numpy as np


def create_combined_traces(
    x: Union[np.ndarray, list],
    y: Union[np.ndarray, list],
    headers: list | None = None,
    hover_template: str | None = None,
):
    """
    Create multiple Plotly traces from multi-dimensional data for line plots.

    Args:
        x: Array-like, 1D or 2D x-axis data.
        y: Array-like, 2D y-axis data.
        headers: Iterable of trace names. Defaults to numeric indices if None.
        hover_template: Optional string for hover text formatting.

    Returns:
        dict: Dictionary of traces keyed by headers.

    Raises:
        ValueError: If x and y shapes are incompatible or headers length mismatches y's columns.
    """
    # Convert inputs to NumPy arrays for consistency
    x = np.asarray(x)
    y = np.asarray(y)

    # Validate input shapes
    if len(y.shape) != 2:
        raise ValueError("y must be 2D")
    if len(x.shape) > 2:
        raise ValueError("x must be 1D or 2D")

    n_cols = y.shape[1]
    # Set default headers if None
    headers = range(n_cols) if headers is None else list(headers)

    # Validate headers length
    if len(headers) != n_cols:
        raise ValueError(
            f"Length of headers ({len(headers)}) must match y's columns ({n_cols})"
        )

    # Handle 1D x by tiling to match y's shape
    if len(x.shape) == 1:
        x = np.tile(x[:, np.newaxis], (1, n_cols))
    elif x.shape[1] != n_cols:
        raise ValueError(
            f"x's columns ({x.shape[1]}) must match y's columns ({n_cols})"
        )

    # Create traces efficiently using list comprehension
    traces = {
        key: create_trace_constructor(
            name=key,
            data=np.column_stack([x[:, i], y[:, i]]),
            static=True
        )
        for i, key in enumerate(headers)
    }

    # Apply hover template if provided
    if hover_template:
        for trace in traces.values():
            props = (
                trace.properties.to_dict()
                if hasattr(trace.properties, "to_dict")
                else trace.properties if isinstance(trace.properties, dict) else {}
            )
            props["hovertemplate"] = hover_template
            trace.properties = props

    return traces


# def create_combined_traces(x, y, headers="xyz", hover_template=None):
#     """
#     Create multiple traces from multi-dimensional data.

#     Replacement for the old combined_trace_constructor function.
#     """
#     import numpy as np

#     trace_dict = {}
#     x = np.asarray(x)
#     y = np.asarray(y)
#     if len(x.shape) == 1:
#         # tile x to y.shape
#         x = np.tile(x, y.shape[1])
#     for key, x_data, y_data in zip(headers, x.T, y.T):
#         trace = create_trace_constructor(name=key, data=np.column_stack([x_data, y_data]), static=True)
#         if hover_template:
#             # Add hover template to the trace properties
#             # Ensure properties is a dict
#             if hasattr(trace.properties, "to_dict"):
#                 props_dict = trace.properties.to_dict()
#                 props_dict["hovertemplate"] = hover_template
#                 trace.properties = props_dict
#             elif isinstance(trace.properties, dict):
#                 trace.properties["hovertemplate"] = hover_template
#             else:
#                 trace.properties = {"hovertemplate": hover_template}
#         trace_dict[key] = trace
#     return trace_dict


def create_subplots_traces(
    x, y, rows=None, cols=None, hover_template=None, headers="xyz"
):
    """
    Create traces for subplots.

    Replacement for the old subplots_trace_constructor function.
    """
    import numpy as np

    if rows is None:
        rows = [1, 2, 3]
    if cols is None:
        cols = [1, 1, 1]

    data = []
    for axis, header in zip(np.asarray(y).T, headers):
        trace = create_trace_constructor(
            name=header,
            data=np.column_stack([x, axis]),
            static=True
        )
        if hover_template:
            trace.properties["hovertemplate"] = hover_template
        data.append(trace)

    return {
        "data": data,
        "rows": rows,
        "cols": cols,
    }


# =============================================================================
# PLOT CONTENT BUILDERS - For creating plots with data
# =============================================================================


class SpatialPlotBuilder:
    """Base class for building spatial plots with common functionality"""

    def __init__(self, margin=0.1):
        self.margin = margin
        self.layout_builder = PlotLayoutBuilder
        self.default_trace_properties = {
            "marker": {"colorscale": "Viridis"},
            "mode": "lines+markers",
        }

    def calculate_range(self, data):
        """Calculate range with margin for given data"""
        data_min = data.min()
        data_max = data.max()
        range_span = abs(data_max - data_min)
        margin_value = self.margin * range_span
        return (data_min - margin_value, data_max + margin_value)

    def extract_position_data(self, position, axes):
        """Extract position data for specified axes (e.g., [0, 2] for X,Z)"""
        return [position[:, axis] for axis in axes]

    def create_hover_template(self, x_title, y_title):
        """Create hover template for spatial plots"""
        return (
            f"{x_title}:"
            + "%{x:.2f} m<br>"
            + f"{y_title}:"
            + "%{y:.2f} m<br>"
            + "vel: %{customdata:.2f} m/s<br>"
        )

    def create_spatial_layout_overrides(
        self,
        x_data,
        y_data,
        x_title,
        y_title,
        units=None,
        xrange=None,
        yrange=None,
        **kwargs,
    ):
        """Create layout overrides specific to spatial plots"""

        margin = self.get_trace_margin()

        # Calculate ranges if not provided
        if xrange is None:
            xrange = get_plot_range(x_data, margins=[margin, margin])
        if yrange is None:
            yrange = get_plot_range(y_data, margins=[margin, margin])

        # For Z-axis plots (XZ, YZ), flip the Y-axis direction so positive Z goes down
        yaxis_config = {"constrain": "domain"}  # no scaleanchor here
        if y_title == "Z":
            if yrange is not None:
                yaxis_config["range"] = yrange[::-1]
            else:
                yaxis_config["autorange"] = "reversed"

        return {
            "scene": {},
            "xaxis": {"range": xrange, "constrain": "domain"},
            "yaxis": yaxis_config,
            "autosize": False,
            **kwargs,
        }

    def create_3d_layout_overrides(self, title="3D End Effector Trajectory"):
        """Create layout overrides for 3D spatial plots"""
        return {
            "scene": {
                "xaxis_title": "X [m]",
                "yaxis_title": "Y [m]",
                "zaxis_title": "Z [m]",
                "aspectmode": "cube",
            },
            "title": title,
            "autosize": False,
        }

    @classmethod
    def get_trace_margin(cls):
        """Get trace margin from theme manager"""
        theme = get_theme_manager()
        return theme.get_settings().get("trace_margin", 0)


class Spatial2DPlotBuilder(SpatialPlotBuilder):
    """Builder for 2D spatial plots (XY, XZ, YZ projections)"""

    def create_2d_plot(
        self, position, velocity_norm, axis_indices, axis_names, plot_name
    ):
        x_data, y_data = self.extract_position_data(position, axis_indices)
        x_range = self.calculate_range(x_data)
        y_range = self.calculate_range(y_data)

        marker_color = velocity_norm
        # return np.mean(velocity_norm) value of Viridis colorscale as
        line_color = sample_colorscale("Viridis", [np.mean(marker_color)])[0]

        trace_properties = merge(
            {},
            self.default_trace_properties,
            {
                "marker": {
                    "line": {
                        "color": marker_color,
                    },
                    "color": marker_color,
                },
                "line": {"color": line_color},
                "name": "traj.spatial.2d",
                "customdata": velocity_norm,
                "hovertemplate": self.create_hover_template(
                    axis_names[0], axis_names[1]
                ),
            },
        )

        # Create spatial-specific overrides
        spatial_overrides = self.create_spatial_layout_overrides(
            x_data,
            y_data,
            axis_names[0],
            axis_names[1],
            units="m",
            xrange=x_range,
            yrange=y_range,
            hovermode="x",
        )

        # Use PlotLayoutBuilder with overrides
        layout = self.layout_builder.create_plot_layout(
            mode="basic",
            title=f"End Effector Trajectory Position {plot_name}",
            x_title=axis_names[0],
            y_title=axis_names[1],
            x_units="m",
            y_units="m",
        )

        # Deep-merge the spatial overrides (preserve reversed Z)
        merge(layout, spatial_overrides)
        trace_properties["name"] = trace_properties.get("name", "trace_0")

        # Create standard trace constructor with line trace
        trace_name = trace_properties.pop("name")
        traces = {
            trace_name: create_trace_constructor(
                name=trace_name,
                data=np.column_stack([x_data, y_data]),
                static=True,
                properties=trace_properties
            )
        }

        return PlotFigure(
            trace_constructor=traces,
            layout=layout,
        ).figure


class Spatial3DPlotBuilder(SpatialPlotBuilder):
    """Builder for 3D spatial plots"""

    def __init__(self, margin=0.1):
        super().__init__(margin)

        self.default_3d_trace_properties = {
            "name": "traj.spatial.3d",
        }

    def create_3d_plot(self, position, velocity_norm):
        x_data, y_data, z_data = self.extract_position_data(position, [0, 1, 2])

        marker_color = velocity_norm
        # return np.mean(velocity_norm) value of Viridis colorscale as
        line_color = sample_colorscale("Viridis", [np.mean(marker_color)])[0]

        trace_properties = merge(
            {},
            self.default_trace_properties,
            {
                "marker": {
                    "line": {
                        "color": marker_color,
                    },
                    "color": marker_color,
                },
                "line": {"color": line_color},
                "name": "traj.spatial.3d",
                "customdata": velocity_norm,
            },
        )

        # Create standard 3D trace constructor
        trace_name = trace_properties.pop("name")
        traces = {
            trace_name: create_trace_constructor(
                name=trace_name,
                data=np.column_stack([x_data, y_data, z_data]),
                static=True,
                properties=trace_properties
            )
        }

        # Create 3D layout overrides
        layout_overrides = self.create_3d_layout_overrides()

        return PlotFigure3D(
            trace_constructor=traces,
            layout=layout_overrides,
        ).figure


# =============================================================================
# HIGH-LEVEL PLOT INTERFACES - Convenient APIs
# =============================================================================


class EndEffectorSpatialPlots:
    """High-level interface for creating end effector spatial plots"""

    def __init__(self, margin=0.1):
        self.plot_2d = Spatial2DPlotBuilder(margin)
        self.plot_3d = Spatial3DPlotBuilder(margin)

    def plot_xz(
        self,
        position,
        velocity_norm,
        trajectory_extents=None,
        xyrange=None,
        zrange=None,
    ):
        return self.plot_2d.create_2d_plot(
            position,
            velocity_norm,
            axis_indices=[0, 2],
            axis_names=["X", "Z"],
            plot_name="XZ",
        )

    def plot_yz(
        self,
        position,
        velocity_norm,
        trajectory_extents=None,
        xyrange=None,
        zrange=None,
    ):
        return self.plot_2d.create_2d_plot(
            position,
            velocity_norm,
            axis_indices=[1, 2],
            axis_names=["Y", "Z"],
            plot_name="YZ",
        )

    def plot_xy(self, position, velocity_norm, trajectory_extents=None, xyrange=None):
        return self.plot_2d.create_2d_plot(
            position,
            velocity_norm,
            axis_indices=[0, 1],
            axis_names=["X", "Y"],
            plot_name="XY",
        )

    def plot_overview(self, position, velocity_norm, xyrange=None, zrange=None):
        return self.plot_3d.create_3d_plot(position, velocity_norm)


# =============================================================================
# COORDINATED PLOT BUILDERS - Simplify plot function patterns
# =============================================================================


class BasePlotBuilder:
    """Base class for plot builders that coordinate Figure, trace constructor, and layout"""

    def __init__(self):
        self.layout_builder = PlotLayoutBuilder

    def create_plot(
        self,
        x,
        y,
        title,
        x_title="Time",
        y_title="",
        x_units="s",
        y_units="",
        secondary_axis=None,
        hover_precision=3,
        headers="xyz",
        hovermode="x unified",
        palette: str = "primary",
        **kwargs,
    ):
        """Create a complete plot by coordinating all components"""
        self.palette = palette
        # Get the appropriate trace constructor and figure class
        traces = self._create_traces(
            x,
            y,
            hover_precision=hover_precision,
            x_title=x_title,
            x_units=x_units,
            y_units=y_units,
            headers=headers,
            **kwargs,
        )
        # Get trace names and apply themes through theme manager
        traces = self._apply_themes_via_theme_manager(traces, palette)
        # Create layout
        layout = self._create_layout(
            x,
            y,
            title,
            x_title,
            y_title,
            x_units,
            y_units,
            hovermode=hovermode,
            **kwargs,
        )
        # Create figure instance with already-themed traces
        figure_instance = self._create_figure_with_themed_traces(
            traces, layout, **kwargs
        )
        if secondary_axis:
            figure_instance.add_secondary_axis(**secondary_axis)
        return figure_instance.figure

    def _create_traces(self, x, y, headers="xyz", **kwargs):
        """Create traces - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _create_traces")

    def _create_layout(self, x, y, title, x_title, y_title, x_units, y_units, **kwargs):
        """Create layout using PlotLayoutBuilder"""
        return self.layout_builder.create_plot_layout(
            mode=self._get_mode(),
            x=x,
            y=y,
            title=title,
            x_title=x_title,
            y_title=y_title,
            x_units=x_units,
            y_units=y_units,
            **kwargs,
        )

    def _create_figure(self, traces, layout, **kwargs):
        """Create figure instance - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _create_figure")

    def _create_figure_with_themed_traces(self, traces, layout, **kwargs):
        """Create figure instance with TraceConstructor objects.

        Args:
            traces: Dictionary of TraceConstructor objects
            layout: Layout configuration
        """
        # Create figure - traces will be built on demand with theme resolution
        figure_instance = self._create_figure(traces, layout, **kwargs)
        return figure_instance

    def _get_mode(self):
        """Get the layout mode for this plot type"""
        raise NotImplementedError("Subclasses must implement _get_mode")

    def _apply_themes_via_theme_manager(self, traces, palette_level):
        """Apply themes using the centralized theme manager approach.

        Args:
            traces: Dictionary or list of plain trace constructors (dicts with x, y, etc.)
            palette_level: Palette level (primary, secondary, tertiary)

        Returns:
            Dictionary of themed TraceConstructor objects ready for TraceBuilder
        """
        theme_manager = get_theme_manager()

        # Convert traces to consistent format
        if isinstance(traces, dict):
            trace_dict = traces
        else:
            # Convert list to dict with generated keys
            trace_dict = {f"trace_{i}": trace for i, trace in enumerate(traces)}

        # Extract trace names for theme assignment
        trace_names = list(trace_dict.keys())

        # Get all themed properties from theme manager (includes palette colors)
        themed_properties = theme_manager.assign_palette_colors_to_traces(
            trace_names, palette_level
        )

        # Convert plain trace constructors to TraceConstructor objects with themes
        themed_constructors = {}

        for trace_key, trace_data in trace_dict.items():
            # Get the themed properties for this trace
            theme_props = themed_properties.get(trace_key, {})

            # Convert plain dict constructor to TraceConstructor
            if isinstance(trace_data, dict) and "x" in trace_data and "y" in trace_data:
                x_data = trace_data["x"]
                y_data = trace_data["y"]
                z_data = trace_data.get("z", None)

                if z_data is not None:
                    points = np.column_stack((x_data, y_data, z_data))
                else:
                    points = np.column_stack((x_data, y_data))

                # Extract constructor properties (non-data properties)
                constructor_props = {
                    k: v
                    for k, v in trace_data.items()
                    if k not in ["x", "y", "z", "trace_type", "name"]
                }

                # Merge constructor properties with theme properties
                final_properties = constructor_props.copy()
                final_properties.update(theme_props)

                themed_constructors[trace_key] = TraceConstructor(
                    name=trace_data.get("name", trace_key),
                    data=points,
                    properties=final_properties,
                )
            else:
                # Handle other formats or fallback
                # Get data from various possible key names for compatibility
                data_array = trace_data.get("data")
                if data_array is None:
                    data_array = trace_data.get("points")
                if data_array is None:
                    data_array = []

                # Convert to numpy array and check if empty
                data_array = np.asarray(data_array)
                if data_array.size == 0:
                    # Skip empty traces to prevent errors
                    continue

                themed_constructors[trace_key] = TraceConstructor(
                    name=trace_data.get("name", trace_key),
                    data=data_array,
                    properties=theme_props,
                )

        return themed_constructors

    def _get_palette_colors(self, count):
        """Helper method to get palette colors."""

        theme_manager = get_theme_manager()
        return theme_manager.get_palette_colors(self.palette, count)


class BasicPlotBuilder(BasePlotBuilder):
    """Builder for basic single plots (PlotFigure)"""

    def _create_traces(self, x, y, **kwargs):
        hover_template = None

        return create_combined_traces(x, y, hover_template=hover_template)

    def _create_figure(self, traces, layout, **kwargs):
        return PlotFigure(trace_constructor=traces, layout=layout)

    def _get_mode(self):
        return "basic"


class CombinedPlotBuilder(BasePlotBuilder):
    """Builder for combined axis plots (CombinedAxisFigure)"""

    def _create_traces(self, x, y, headers="123", **kwargs):
        hover_template = None

        return create_combined_traces(
            x, y, hover_template=hover_template, headers=headers
        )

    def _create_figure(self, traces, layout, **kwargs):
        return CombinedAxisFigure(trace_constructor=traces, layout=layout)

    def _get_mode(self):
        return "combined"


class SubplotsPlotBuilder(BasePlotBuilder):
    """Builder for subplot plots (SubplotsFigure)"""

    def _create_traces(self, x, y, headers="xyz", **kwargs):
        hover_template = None

        result = create_subplots_traces(
            x, y, hover_template=hover_template, headers=headers
        )

        # Don't apply theming here - it will be handled in the base class
        return result

    def _apply_themes_via_theme_manager(self, traces, palette_level):
        """Override to handle subplots structure with data/rows/cols."""
        if isinstance(traces, dict) and "data" in traces:
            # Apply theming to the data array only
            themed_data = super()._apply_themes_via_theme_manager(
                traces["data"], palette_level
            )
            # Reconstruct the subplots structure
            return {
                "data": themed_data,
                "rows": traces["rows"],
                "cols": traces["cols"],
                "properties": traces.get("properties", None),
            }
        else:
            # Fallback to base implementation
            return super()._apply_themes_via_theme_manager(traces, palette_level)

    def _create_figure_with_themed_traces(self, traces, layout, **kwargs):
        """Override to handle subplots with themed traces."""
        # For subplots, traces has the structure {'data': {...}, 'rows': [...], 'cols': [...]}
        if isinstance(traces, dict) and "data" in traces:
            # Build final plotly traces from the themed constructors
            builder = TraceBuilder()
            final_trace_data = []

            for constructor in traces["data"].values():
                final_trace_data.append(builder.build_trace(constructor))

            # Create the subplots figure
            subplots_constructor = kwargs.get("subplots_constructor", {})
            figure_instance = SubplotsFigure(
                trace_constructor=traces,  # Keep original structure for compatibility
                layout=layout,
                subplots_contructor=subplots_constructor,
            )

            return figure_instance
        else:
            # Fallback to base implementation
            return super()._create_figure_with_themed_traces(traces, layout, **kwargs)

    def _create_figure(self, traces, layout, **kwargs):
        subplots_constructor = kwargs.get("subplots_constructor", {})
        return SubplotsFigure(
            trace_constructor=traces,
            layout=layout,
            subplots_contructor=subplots_constructor,
        )

    def _get_mode(self):
        return "subplots"
