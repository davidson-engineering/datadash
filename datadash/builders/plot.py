# Unified Builder Classes for Visualization Components
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

import logging
from mergedeep import merge, Strategy

# Import core utilities
from .trace import get_plot_range, trace_constructor
from .figure import PlotFigure, PlotFigure3D
from ..themes.manager import get_theme_manager
from .layout import PlotLayoutBuilder


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
        from .trace import get_plot_range

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

        trace_properties = merge(
            {},
            self.default_trace_properties,
            {
                "marker": {"color": velocity_norm},
                "name": plot_name,
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

        traces = trace_constructor(x_data, y_data, properties=trace_properties)

        return PlotFigure(
            trace_constructor=traces,
            layout=layout,
        ).figure


class Spatial3DPlotBuilder(SpatialPlotBuilder):
    """Builder for 3D spatial plots"""

    def __init__(self, margin=0.1):
        super().__init__(margin)

        self.default_3d_trace_properties = {
            "mode": "lines+markers",
            "marker": {
                "size": 3,
                "colorscale": "Viridis",
                "colorbar": {"title": "Velocity [m/s]"},
            },
            "line": {"width": 4},
            "name": "Trajectory",
        }

    def create_3d_plot(self, position, velocity_norm):
        x_data, y_data, z_data = self.extract_position_data(position, [0, 1, 2])

        trace_properties = merge(
            {}, self.default_3d_trace_properties, {"marker": {"color": velocity_norm}}
        )

        traces = trace_constructor(
            x=x_data, y=y_data, z=z_data, properties=trace_properties
        )

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

    def __init__(self, palette: str = 'primary'):
        self.palette = palette
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
        **kwargs,
    ):
        """Create a complete plot by coordinating all components"""
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
        figure_instance = self._create_figure(traces, layout, **kwargs)
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

    def _get_mode(self):
        """Get the layout mode for this plot type"""
        raise NotImplementedError("Subclasses must implement _get_mode")

    def _apply_palette_colors(self, traces):
        """Apply palette colors to traces based on the selected palette.

        Args:
            traces: Dictionary of trace data or list of traces

        Returns:
            Updated traces with palette colors applied
        """
        from ..themes.manager import get_theme_manager

        if isinstance(traces, dict):
            trace_list = list(traces.values())
            trace_keys = list(traces.keys())
        else:
            trace_list = traces
            trace_keys = None

        # Get colors from the theme manager
        theme_manager = get_theme_manager()
        palette_colors = theme_manager.get_palette_colors(self.palette, len(trace_list))

        # Apply colors to traces
        for i, trace_data in enumerate(trace_list):
            color = palette_colors[i]

            # Apply color to both line and marker properties
            if 'line' not in trace_data:
                trace_data['line'] = {}
            if 'marker' not in trace_data:
                trace_data['marker'] = {}

            trace_data['line']['color'] = color
            trace_data['marker']['color'] = color

        return traces

    def _get_palette_colors(self, count):
        """Helper method to get palette colors."""
        from ..themes.manager import get_theme_manager
        theme_manager = get_theme_manager()
        return theme_manager.get_palette_colors(self.palette, count)


class CombinedPlotBuilder(BasePlotBuilder):
    """Builder for combined axis plots (CombinedAxisFigure)"""

    def __init__(self, palette: str = 'primary'):
        super().__init__(palette)

    def _create_traces(self, x, y, headers="123", **kwargs):
        from .trace import combined_trace_constructor

        hover_template = None

        traces = combined_trace_constructor(
            x, y, hover_template=hover_template, headers=headers
        )

        # Apply palette colors
        return self._apply_palette_colors(traces)

    def _create_figure(self, traces, layout, **kwargs):
        from .figure import CombinedAxisFigure

        return CombinedAxisFigure(trace_constructor=traces, layout=layout)

    def _get_mode(self):
        return "combined"


class SubplotsPlotBuilder(BasePlotBuilder):
    """Builder for subplot plots (SubplotsFigure)"""

    def __init__(self, palette: str = 'primary'):
        super().__init__(palette)

    def _create_traces(self, x, y, headers="xyz", **kwargs):
        from .trace import subplots_trace_constructor

        hover_template = None

        result = subplots_trace_constructor(
            x, y, hover_template=hover_template, headers=headers
        )

        # Apply palette colors to the data array
        if isinstance(result, dict) and 'data' in result:
            palette_colors = self._get_palette_colors(len(result['data']))
            for i, trace_data in enumerate(result['data']):
                color = palette_colors[i]
                if 'line' not in trace_data:
                    trace_data['line'] = {}
                if 'marker' not in trace_data:
                    trace_data['marker'] = {}
                trace_data['line']['color'] = color
                trace_data['marker']['color'] = color

        return result

    def _create_figure(self, traces, layout, **kwargs):
        from .figure import SubplotsFigure

        subplots_constructor = kwargs.get("subplots_constructor", {})
        return SubplotsFigure(
            trace_constructor=traces,
            layout=layout,
            subplots_contructor=subplots_constructor,
        )

    def _get_mode(self):
        return "subplots"


class BasicPlotBuilder(BasePlotBuilder):
    """Builder for basic single plots (PlotFigure)"""

    def __init__(self, palette: str = 'primary'):
        super().__init__(palette)

    def _create_traces(self, x, y, **kwargs):
        from .trace import combined_trace_constructor

        hover_template = None

        traces = combined_trace_constructor(x, y, hover_template=hover_template)

        # Apply palette colors
        return self._apply_palette_colors(traces)

    def _create_figure(self, traces, layout, **kwargs):
        return PlotFigure(trace_constructor=traces, layout=layout)

    def _get_mode(self):
        return "basic"
