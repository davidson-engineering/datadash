# Core Figure Classes
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from __future__ import annotations
from abc import ABC
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import numpy.typing as npt
from mergedeep import merge

from ..themes.manager import get_theme_manager
from .trace import TraceConstructor
from ..utils import tile


def compute_ranges(*data: list[np.ndarray], margins=None):
    """
    Compute min/max ranges for each axis given arrays.
    Optionally apply margins.
    """

    def get_range(axis, margin):
        axis_min = np.min(axis)
        axis_max = np.max(axis)
        margin = np.max(np.abs((axis_min, axis_max))) * margin
        return (float(axis_min - margin), float(axis_max + margin))

    if margins is None:
        return tuple(get_range(axis, 0.0) for axis in data)
    return tuple(get_range(axis, margin) for axis, margin in zip(data, margins))


def compute_scene_ranges(data, margins=None):
    """
    Extract x, y, z arrays from frames and compute ranges.
    """
    xs = np.concatenate([np.array(trace.x) for trace in data])
    ys = np.concatenate([np.array(trace.y) for trace in data])
    zs = np.concatenate([np.array(trace.z) for trace in data])

    return compute_ranges(xs, ys, zs, margins=margins)


def compute_scene_ranges_animation(frames, margins=None):
    """
    Extract x, y, z arrays from frames and compute ranges.
    """
    xs = np.concatenate([np.array(trace.x) for f in frames for trace in f.data])
    ys = np.concatenate([np.array(trace.y) for f in frames for trace in f.data])
    zs = np.concatenate([np.array(trace.z) for f in frames for trace in f.data])

    return compute_ranges(xs, ys, zs, margins=margins)


def compute_overall_range(*data: list[np.ndarray], margin=0.0):
    """
    Compute a single global (min, max) across all provided arrays.
    """
    combined = np.concatenate([np.ravel(axis) for axis in data])
    axis_min = np.min(combined)
    axis_max = np.max(combined)
    margin = np.max(np.abs((axis_min, axis_max))) * margin
    return (float(axis_min - margin), float(axis_max + margin))


def compute_plot_extents(data, margins: int | float | tuple[int | float] | None = None):
    if margins is None:
        margins = (0.0, 0.0, 0.0)
    if margins is not None and isinstance(margins, (int, float)):
        margins = (margins, margins, margins)
    # Compute global axis ranges
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = compute_scene_ranges(
        data, margins=margins
    )
    # ensure that dimensions are cubic
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    z_center = (zmin + zmax) / 2
    x_range = (x_center - max_range / 2, x_center + max_range / 2)
    y_range = (y_center - max_range / 2, y_center + max_range / 2)
    z_range = (
        z_center - max_range / 2,
        z_center + max_range / 2,
    )
    return x_range, y_range, z_range


def compute_plot_extents_animation(
    frames, margins: int | float | tuple[int | float] | None = None
):
    if margins is None:
        margins = (0.0, 0.0, 0.0)
    if margins is not None and isinstance(margins, (int, float)):
        margins = (margins, margins, margins)
    # Compute global axis ranges
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = compute_scene_ranges_animation(
        frames, margins=margins
    )
    # ensure that dimensions are cubic
    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    z_center = (zmin + zmax) / 2
    x_range = (x_center - max_range / 2, x_center + max_range / 2)
    y_range = (y_center - max_range / 2, y_center + max_range / 2)
    z_range = (
        z_center - max_range / 2,
        z_center + max_range / 2,
    )
    return x_range, y_range, z_range


class PlotFigure:
    def __init__(
        self,
        layout=None,
        trace_constructor=None,
        settings=None,
    ):
        self.trace_constructor = trace_constructor

        # Get theme manager
        theme = get_theme_manager()

        # Get settings from theme - they are the base settings
        self.settings = theme.get_settings().copy()

        # Apply any custom settings overrides
        if settings:
            self.settings.update(settings)

        # Get plotly layout from theme
        plotly_theme = theme.get_plotly_theme()
        theme_layout = plotly_theme.get("layout", {})

        # Merge layout with any overrides
        merged_layout = merge(
            {},
            theme_layout,
            layout or {},
        )

        # Apply initial patterning during initialization if enabled
        if self.settings["pattern_properties_enable"]:
            self.layout = self.pattern_properties(
                merged_layout, self.settings["pattern_properties_lookup"]
            )
        else:
            self.layout = merged_layout

        self._figure = None

    def _build_trace_without_theme(self, constructor):
        """Build trace without theme application - themes should be applied in plot builders.

        Converts legacy dict constructors to TraceConstructor objects and builds
        basic traces without theme styling.
        """
        if not isinstance(constructor, TraceConstructor):
            # Convert legacy dict constructor to new TraceConstructor
            constructor = self._convert_legacy_constructor(constructor)

        # Build basic trace without theme application
        return self._build_basic_trace(constructor)

    def _build_basic_trace(self, constructor):
        """Build a basic trace from TraceConstructor without theme styling."""
        import plotly.graph_objects as go

        # Extract basic properties
        points = np.asarray(constructor.points)
        properties = constructor.properties.copy() if constructor.properties else {}

        # Determine if 2D or 3D based on points shape
        if points.ndim >= 2 and points.shape[-1] >= 3:
            # 3D trace
            if points.ndim == 1:
                x, y, z = points[0], points[1], points[2]
            elif points.ndim == 2:
                x, y, z = points[:, 0], points[:, 1], points[:, 2]
            else:
                # Flatten for multi-segment
                flat = points.reshape(-1, points.shape[-1])
                x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]

            return go.Scatter3d(x=x, y=y, z=z, **properties)
        else:
            # 2D trace
            if points.ndim == 1:
                x, y = points[0], points[1]
            elif points.ndim == 2:
                x, y = points[:, 0], points[:, 1]
            else:
                # Flatten for multi-segment
                flat = points.reshape(-1, points.shape[-1])
                x, y = flat[:, 0], flat[:, 1]

            return go.Scatter(x=x, y=y, **properties)

    def _convert_legacy_constructor(self, legacy_dict):
        """Convert legacy dict-based constructor to TraceConstructor.

        Args:
            legacy_dict: Legacy dict with keys like 'points', 'properties', etc.

        Returns:
            TraceConstructor: New TraceConstructor instance
        """
        # Extract common fields from legacy dict
        name = legacy_dict.get("name", "trace")
        points = legacy_dict.get("points", [])
        points_time = legacy_dict.get("points_time", None)
        closed = legacy_dict.get("closed", False)
        static = legacy_dict.get("static", points_time is None)
        properties = legacy_dict.get("properties", {})

        # Capture hovertemplate from top-level dict and add to properties
        if "hovertemplate" in legacy_dict:
            properties = properties.copy()  # Don't modify original
            properties["hovertemplate"] = legacy_dict["hovertemplate"]

        # Handle alternative field names
        if (not hasattr(points, "__len__") or len(points) == 0) and "x" in legacy_dict:
            # Convert x, y, z to points format
            x = legacy_dict.get("x", [])
            y = legacy_dict.get("y", [])
            z = legacy_dict.get("z", [])
            if len(z) > 0:
                points = np.column_stack((x, y, z))
            else:
                points = np.column_stack((x, y))

        return TraceConstructor(
            name=name,
            points=points,
            points_time=points_time,
            closed=closed,
            static=static,
            properties=properties,
        )

    @property
    def traces(self, trace_constructor=None):
        # If we have pre-themed traces from plot builders, use them
        if hasattr(self, '_themed_traces') and self._themed_traces:
            return self._themed_traces

        self.trace_constructor = (
            self.trace_constructor if trace_constructor is None else trace_constructor
        )
        return {
            key: self._build_trace_without_theme(constructor)
            for key, constructor in self.trace_constructor.items()
        }

    @traces.setter
    def traces(self, trace_constructor):
        self.trace_constructor = trace_constructor

    @property
    def surfaces(self, surface_constructor=None):
        self.surface_constructor = (
            self.surface_constructor
            if surface_constructor is None
            else surface_constructor
        )
        return {
            key: self._build_trace_without_theme(constructor)
            for key, constructor in self.surface_constructor.items()
        }

    def pattern_properties(self, properties, lookups, lookin=None, exclusions=None):
        """Apply property patterning to layout properties.

        Args:
            properties: The properties to pattern
            lookups: List of base property names to pattern (e.g., ['xaxis', 'yaxis'])
            lookin: Source properties to copy from (defaults to properties)
            exclusions: Property suffix to exclude from patterning

        Returns:
            Properties with patterning applied
        """
        exclusions = "_title" if exclusions is None else exclusions
        lookin = properties if lookin is None else lookin
        properties = merge(
            {},
            lookin,
            properties,
        )
        patterned_properties = {}
        for lookup in lookups:
            for k, v in properties.items():
                if k.startswith(lookup) and not k.endswith(exclusions):
                    patterned_properties[k] = lookin[lookup]
        return merge(
            properties,
            patterned_properties,
        )

    def pattern_layout(self, force_pattern=False):
        """Manually apply patterning to the current layout.

        Args:
            force_pattern: If True, apply patterning even if disabled in settings

        Returns:
            Self for method chaining
        """
        should_pattern = force_pattern or self.settings.get(
            "pattern_properties_enable", False
        )

        if should_pattern:
            lookups = self.settings.get("pattern_properties_lookup", [])
            self.layout = self.pattern_properties(self.layout, lookups)

        return self

    def add_secondary_axis(self, units: str, scale: float = 1, title: str = None):
        """Add a secondary y-axis with different units and scale.

        Creates hidden traces on the secondary axis that can be toggled in the legend.

        Args:
            units: Label for the secondary axis (e.g., "rpm", "deg/s")
            scale: Conversion factor from primary to secondary units
        """
        # Convert existing figure to subplot with secondary y-axis
        fig = self.figure

        # Create new subplot figure with secondary y-axis
        subplot_fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add all existing traces to primary axis
        for trace in fig.data:
            subplot_fig.add_trace(trace, secondary_y=False)

        # Add secondary traces (scaled and hidden)
        for key, trace in self.traces.items():
            # Create a copy of the trace for secondary axis
            # Use copy.deepcopy for a robust solution that works with all trace types
            import copy

            try:
                # First try using the trace type constructor directly
                trace_type = type(trace)
                if trace_type == go.Scatter:
                    secondary_trace = go.Scatter(trace)
                elif trace_type == go.Bar:
                    secondary_trace = go.Bar(trace)
                elif trace_type == go.Scattergl:
                    secondary_trace = go.Scattergl(trace)
                else:
                    # Fallback to deep copy for other trace types
                    secondary_trace = copy.deepcopy(trace)
            except Exception:
                # If constructor fails, use deep copy as fallback
                secondary_trace = copy.deepcopy(trace)

            # Update properties for secondary axis
            secondary_trace.name = f"{trace.name}_{units}"
            secondary_trace.visible = False  # Hidden by default

            # Scale the y-data
            if hasattr(secondary_trace, "y") and secondary_trace.y is not None:
                secondary_trace.y = [y * scale for y in secondary_trace.y]

            subplot_fig.add_trace(secondary_trace, secondary_y=True)

        # Set secondary axis range using overall range computation
        # Get all secondary axis y-data (already scaled)
        secondary_y_data = []
        for trace in subplot_fig.data:
            if (
                hasattr(trace, "y")
                and trace.y is not None
                and getattr(trace, "yaxis", "y") == "y2"
            ):
                secondary_y_data.append(np.array(trace.y))

        # Apply layout to the subplot figure
        subplot_fig.update_layout(fig.layout)

        if secondary_y_data:
            # Get margin from theme settings
            theme = get_theme_manager()
            margin = theme.get_settings().get("trace_margin", 0)

            # Compute overall range for secondary axis data with margin
            secondary_range = compute_overall_range(*secondary_y_data, margin=margin)

            # Apply the computed range to secondary axis
            subplot_fig.update_yaxes(range=secondary_range, secondary_y=True)

        if title:
            axis_title = f"{title} [{units}]"
        else:
            axis_title = f"[{units}]"

        # Update axis labels and properties
        subplot_fig.update_yaxes(
            secondary_y=True,
            title_text=axis_title,
            anchor="free",
            overlaying="y",
            autoshift=True,
            showgrid=False,  # Disable gridlines for secondary axis
        )

        # Replace the figure property to return the subplot figure
        self.figure = subplot_fig

        return self

    @property
    def figure(self):
        # Return existing figure if it exists, otherwise return normal figure
        if self._figure:
            return self._figure
        else:
            fig = go.Figure()
            fig.add_traces(list(self.traces.values()))

        # Apply theme-specific layout
        from ..themes.manager import get_theme_manager

        theme = get_theme_manager()
        plotly_theme = theme.get_plotly_theme()

        # Merge theme layout with existing layout, preserving specific axis settings
        themed_layout = self.layout.copy()
        if theme_layout := plotly_theme.get("layout"):
            merge(themed_layout, theme_layout)

        fig.update_layout(themed_layout)
        self._figure = fig
        return fig

    @figure.setter
    def figure(self, fig):
        self._figure = fig


class PlotFigure3D(PlotFigure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Apply 3D-specific patterning during initialization if enabled
        if self.settings["pattern_properties_enable"]:
            self._apply_3d_patterning()

    def _apply_3d_patterning(self):
        """Apply 3D-specific patterning to scene properties."""
        if "scene" in self.layout:
            lookups = self.settings.get("pattern_properties_lookup", [])
            self.layout["scene"] = self.pattern_properties(
                self.layout.get("scene", {}),
                lookups,
                lookin=self.layout.get("scene", {}),
            )

    def pattern_layout(self, force_pattern=False):
        """Manually apply patterning to the current layout, including 3D scene patterning.

        Args:
            force_pattern: If True, apply patterning even if disabled in settings

        Returns:
            Self for method chaining
        """
        should_pattern = force_pattern or self.settings.get(
            "pattern_properties_enable", False
        )

        if should_pattern:
            # Apply general layout patterning
            lookups = self.settings.get("pattern_properties_lookup", [])
            self.layout = self.pattern_properties(self.layout, lookups)

            # Apply 3D-specific scene patterning
            self._apply_3d_patterning()

        return self

    def add_secondary_axis(self, units: str, scale: float = 1, title: str = None):
        """Secondary axis is not supported for 3D plots."""
        raise NotImplementedError("Secondary axis is not supported for 3D plots")

    @property
    def figure(self):
        fig = go.Figure()
        fig.add_traces(list(self.traces.values()))

        # Apply theme-specific layout
        from ..themes.manager import get_theme_manager

        theme = get_theme_manager()
        plotly_theme = theme.get_plotly_theme()
        current_layout = self.layout.copy()

        x_range, y_range, z_range = compute_plot_extents(fig.data, margins=0)
        z_range = z_range[::-1]  # Invert z-axis for better 3D view

        layout_overrides = {
            "scene": {
                "xaxis": {"range": x_range},
                "yaxis": {"range": y_range},
                "zaxis": {"range": z_range},
            }
        }

        # Merge in theme layout if defined
        if plotly_theme.get("layout"):
            themed_layout = merge({}, current_layout, plotly_theme["layout"])
        else:
            themed_layout = current_layout

        # Override layout with 3D animation specifics
        merge(themed_layout, layout_overrides)

        fig.update_layout(themed_layout)
        return fig


class PlotFigure3DAnimation(PlotFigure3D):
    def __init__(self, *args, plot_duration: float = 1.0, fps: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.fps = fps
        self.number_frames = int(plot_duration * fps)
        self.time = np.linspace(0, plot_duration, self.number_frames)

    @property
    def frames(self):
        from .frames import FramesConstructor

        frames_constructor = FramesConstructor(
            trace_constructors=self.trace_constructor, number_frames=self.number_frames
        )
        return frames_constructor.build_frames()

    @property
    def figure(self):
        from ..themes.manager import get_theme_manager

        theme = get_theme_manager()
        plotly_theme = theme.get_plotly_theme()

        frames = self.frames
        current_layout = self.layout.copy()

        x_range, y_range, z_range = compute_plot_extents_animation(frames, margins=0)
        z_range = z_range[::-1]  # Invert z-axis for better 3D view

        layout_overrides = {
            "scene": {
                "xaxis": {"range": x_range},
                "yaxis": {"range": y_range},
                "zaxis": {"range": z_range},
            }
        }

        # Merge in theme layout if defined
        if plotly_theme.get("layout"):
            themed_layout = merge({}, current_layout, plotly_theme["layout"])
        else:
            themed_layout = current_layout

        # Override layout with 3D animation specifics
        merge(themed_layout, layout_overrides)

        # Create a new figure
        figure = go.Figure(
            data=self.frames[0].data,
            layout=go.Layout(**themed_layout),
        )
        figure.update(frames=self.frames)

        return figure


class SubplotsFigure(PlotFigure):
    def __init__(
        self,
        *args,
        subplots_contructor=None,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )
        # Use the actual max row/col values from trace constructor
        self.rows = max(kwargs["trace_constructor"]["rows"])
        self.cols = max(kwargs["trace_constructor"]["cols"])
        self.subplots = subplots_contructor or {}

    @property
    def traces(self):
        # If we have pre-themed traces from plot builders, use them
        if hasattr(self, '_themed_traces') and self._themed_traces:
            return self._themed_traces

        # Build traces without theme application - themes are applied in plot builders
        traces_data = []
        for data in self.trace_constructor["data"]:
            # Convert to TraceConstructor if needed
            if not isinstance(data, TraceConstructor):
                constructor = self._convert_legacy_constructor(data)
            else:
                constructor = data

            # Check if layout has showlegend=False and apply to individual traces
            merged_properties = constructor.properties.copy()
            if hasattr(self, "layout") and self.layout.get("showlegend") is False:
                merged_properties["showlegend"] = False

            # Create updated constructor with merged properties
            updated_constructor = TraceConstructor(
                name=constructor.name,
                points=constructor.points,
                points_time=constructor.points_time,
                closed=constructor.closed,
                static=constructor.static,
                properties=merged_properties,
            )

            # Build trace without theme application
            traces_data.append(self._build_trace_without_theme(updated_constructor))

        return {
            "data": traces_data,
            "rows": self.trace_constructor["rows"],
            "cols": self.trace_constructor["cols"],
        }

    def add_secondary_axis(self, units: str, scale: float = 1, title: str = None):
        """Add secondary y-axis to all subplots with different units and scale.

        Creates secondary y-axes for each subplot that can display the same data
        with different units (e.g., N⋅m vs rpm).

        Args:
            units: Label for the secondary axis (e.g., "rpm", "deg/s")
            scale: Conversion factor from primary to secondary units
        """
        from plotly.subplots import make_subplots
        import copy

        # Get existing traces data
        traces_data = self.traces

        # Create new subplot figure with secondary y-axes for each subplot
        specs = []
        for i in range(self.rows):
            row_specs = []
            for j in range(self.cols):
                row_specs.append({"secondary_y": True})
            specs.append(row_specs)

        # Create new subplot figure with secondary y-axis support
        subplot_fig = make_subplots(
            rows=self.rows, cols=self.cols, specs=specs, **self.subplots
        )

        # Add all existing traces to primary axes with their original positioning
        for trace, row, col in zip(
            traces_data["data"], traces_data["rows"], traces_data["cols"]
        ):
            subplot_fig.add_trace(trace, row=row, col=col, secondary_y=False)

        # Add secondary traces (scaled and hidden) for each subplot
        for trace, row, col in zip(
            traces_data["data"], traces_data["rows"], traces_data["cols"]
        ):
            # Create a copy of the trace for secondary axis
            try:
                # First try using the trace type constructor directly
                trace_type = type(trace)
                if trace_type == go.Scatter:
                    secondary_trace = go.Scatter(trace)
                elif trace_type == go.Bar:
                    secondary_trace = go.Bar(trace)
                elif trace_type == go.Scattergl:
                    secondary_trace = go.Scattergl(trace)
                else:
                    # Fallback to deep copy for other trace types
                    secondary_trace = copy.deepcopy(trace)
            except Exception:
                # If constructor fails, use deep copy as fallback
                secondary_trace = copy.deepcopy(trace)

            # Update properties for secondary axis
            secondary_trace.name = f"{trace.name}_{units}"
            secondary_trace.visible = False  # Hidden by default
            secondary_trace.showlegend = False  # Show in legend for toggling

            # Scale the y-data
            if hasattr(secondary_trace, "y") and secondary_trace.y is not None:
                secondary_trace.y = [y * scale for y in secondary_trace.y]

            # Add to the same subplot position but on secondary y-axis
            subplot_fig.add_trace(secondary_trace, row=row, col=col, secondary_y=True)

        if title:
            axis_title = f"{title} [{units}]"
        else:
            axis_title = f"[{units}]"

        # Update secondary y-axis labels for each subplot
        for i in range(1, self.rows + 1):
            for j in range(1, self.cols + 1):
                subplot_fig.update_yaxes(
                    title_text=axis_title,
                    showgrid=False,  # Disable gridlines for secondary axis
                    row=i,
                    col=j,
                    secondary_y=True,
                )

        # Apply consistent grid settings to all subplots
        self._apply_consistent_grid_settings(subplot_fig)

        # Apply consistent grid settings to all subplots
        self._apply_consistent_grid_settings(subplot_fig)

        # Store the new figure
        self.fig = subplot_fig
        return self

    def _apply_consistent_grid_settings(self, figure):
        """Apply consistent grid markings and tick spacing across all subplots.

        Args:
            figure: The plotly figure to apply consistent grid settings to
        """
        import numpy as np
        from ..themes.manager import get_theme_manager

        # Get theme settings for grid configuration
        theme = get_theme_manager()
        settings = theme.get_settings()

        # Grid configuration parameters
        grid_settings = {
            "showgrid": True,
            "gridwidth": 1,
            "gridcolor": "rgba(128,128,128,0.2)",
            "zeroline": True,
            "zerolinewidth": 1,
            "zerolinecolor": "rgba(128,128,128,0.4)",
        }

        # Calculate consistent tick spacing for each axis type
        all_x_data = []
        all_y_data = []

        # Collect all data to determine global ranges
        for trace in figure.data:
            if hasattr(trace, "x") and trace.x is not None:
                all_x_data.extend(trace.x)
            if hasattr(trace, "y") and trace.y is not None:
                all_y_data.extend(trace.y)

        # Calculate consistent tick intervals
        if all_x_data:
            x_range = [min(all_x_data), max(all_x_data)]
            x_tick_interval = self._calculate_nice_tick_interval(x_range)
        else:
            x_tick_interval = None

        if all_y_data:
            y_range = [min(all_y_data), max(all_y_data)]
            y_tick_interval = self._calculate_nice_tick_interval(y_range)
        else:
            y_tick_interval = None

        # Apply consistent settings to all x and y axes
        for i in range(1, self.rows + 1):
            for j in range(1, self.cols + 1):
                # X-axis settings
                x_axis_settings = grid_settings.copy()
                if x_tick_interval:
                    x_axis_settings.update(
                        {
                            "dtick": x_tick_interval,
                            "tick0": 0,  # Start ticks at 0
                        }
                    )

                # Y-axis settings
                y_axis_settings = grid_settings.copy()
                if y_tick_interval:
                    y_axis_settings.update(
                        {
                            "dtick": y_tick_interval,
                            "tick0": 0,  # Start ticks at 0
                        }
                    )

                # Apply to primary axes
                figure.update_xaxes(**x_axis_settings, row=i, col=j)
                figure.update_yaxes(**y_axis_settings, row=i, col=j)

                # Apply minimal grid to secondary y-axes (if they exist)
                try:
                    figure.update_yaxes(
                        showgrid=False,  # Disable grid for secondary to avoid clutter
                        zeroline=False,
                        row=i,
                        col=j,
                        secondary_y=True,
                    )
                except:
                    pass  # No secondary axis exists

    def _calculate_nice_tick_interval(self, data_range):
        """Calculate a nice tick interval for the given data range.

        Args:
            data_range: [min_value, max_value] for the data

        Returns:
            A nice tick interval value
        """
        import math

        if len(data_range) != 2 or data_range[0] == data_range[1]:
            return 1

        span = data_range[1] - data_range[0]

        # Target approximately 5-10 ticks
        target_ticks = 7
        raw_interval = span / target_ticks

        # Find the magnitude (power of 10)
        magnitude = 10 ** math.floor(math.log10(raw_interval))

        # Normalize to [1, 10)
        normalized = raw_interval / magnitude

        # Choose nice intervals: 1, 2, 5, or 10
        if normalized <= 1:
            nice_interval = 1
        elif normalized <= 2:
            nice_interval = 2
        elif normalized <= 5:
            nice_interval = 5
        else:
            nice_interval = 10

        return nice_interval * magnitude

    def set_consistent_grid(
        self,
        x_tick_interval=None,
        y_tick_interval=None,
        grid_color="rgba(128,128,128,0.2)",
        grid_width=1,
    ):
        """Manually set consistent grid properties across all subplots.

        Args:
            x_tick_interval: Fixed interval for x-axis ticks (None for auto)
            y_tick_interval: Fixed interval for y-axis ticks (None for auto)
            grid_color: Color for grid lines
            grid_width: Width of grid lines

        Returns:
            Self for method chaining
        """
        if not hasattr(self, "fig") or self.fig is None:
            # Create figure first if it doesn't exist
            _ = self.figure

        # Custom grid settings
        grid_settings = {
            "showgrid": True,
            "gridwidth": grid_width,
            "gridcolor": grid_color,
            "zeroline": True,
            "zerolinewidth": grid_width,
            "zerolinecolor": grid_color.replace(
                "0.2", "0.4"
            ),  # Slightly more opaque for zero line
        }

        # Apply to all subplots
        for i in range(1, self.rows + 1):
            for j in range(1, self.cols + 1):
                x_settings = grid_settings.copy()
                y_settings = grid_settings.copy()

                if x_tick_interval:
                    x_settings.update({"dtick": x_tick_interval, "tick0": 0})
                if y_tick_interval:
                    y_settings.update({"dtick": y_tick_interval, "tick0": 0})

                self.fig.update_xaxes(**x_settings, row=i, col=j)
                self.fig.update_yaxes(**y_settings, row=i, col=j)

        return self

    @property
    def figure(self):
        # Return existing figure if it has been modified (e.g., by add_secondary_axis)
        if hasattr(self, "fig") and self.fig is not None:
            return self.fig

        # Otherwise create the standard subplot figure
        traces_data = self.traces
        # Use the actual subplot grid dimensions
        properties = {"rows": self.rows, "cols": self.cols}
        self.fig = make_subplots(**(self.subplots | properties))

        # Add traces with proper subplot positioning
        for trace, row, col in zip(
            traces_data["data"], traces_data["rows"], traces_data["cols"]
        ):
            self.fig.add_trace(trace, row=row, col=col)

        # Apply theme-specific layout like the base PlotFigure class
        from ..themes.manager import get_theme_manager

        theme = get_theme_manager()
        plotly_theme = theme.get_plotly_theme()

        # Merge theme layout with existing layout, preserving our custom settings
        if theme_layout := plotly_theme.get("layout"):
            themed_layout = merge({}, theme_layout, self.layout)
        else:
            themed_layout = self.layout.copy()

        # Apply the merged layout to the subplot figure
        self.fig.update_layout(themed_layout)

        return self.fig


class CombinedAxisFigure(PlotFigure):
    @property
    def traces(self):
        # If we have pre-themed traces from plot builders, use them
        if hasattr(self, '_themed_traces') and self._themed_traces:
            return self._themed_traces

        # Build traces without theme application - themes are applied in plot builders
        traces = {}
        for key, constructor in self.trace_constructor.items():
            # Convert to TraceConstructor if needed
            if not isinstance(constructor, TraceConstructor):
                constructor = self._convert_legacy_constructor(constructor)

            # Build trace without theme application
            traces[key] = self._build_trace_without_theme(constructor)

        return traces
