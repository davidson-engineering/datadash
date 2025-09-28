# Trace Construction Utilities
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

import logging
from dataclasses import dataclass
import plotly.graph_objects as go
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from typing import Any, Dict, List, Union
from mergedeep import merge

from ..themes.manager import get_theme_manager

# from ..utils import repeat, tile
from ..standard.properties import StandardTraceProperties

# from ..converters.plotly_converter import convert_standard_to_plotly

# Legacy TraceConstructor class removed - use src.visu.builders.trace_constructor.TraceConstructor


def points_to_xyz(points, closed=False):
    max_shape = max(tuple(el.shape for el in points))
    points = np.array([np.broadcast_to(el, max_shape) for el in points])
    if closed:
        points = np.concatenate((points, [points[0]]))
    return dict(zip("xyz", zip(*points)))


def get_plot_range(*data: list[np.ndarray], margins=None):
    def get_range(axis, margin):
        axis_min = np.min(axis)
        axis_max = np.max(axis)
        margin = np.max(np.abs((axis_min, axis_max))) * margin
        return (
            axis_min - margin,
            axis_max + margin,
        )

    if margins is None:
        return tuple(get_range(axis, 0) for axis in data)
    return tuple(get_range(axis, margin) for axis, margin in zip(*data, margins))


def points_to_xyz_arrays(points):
    """Convert points array to separate x,y,z arrays"""
    if hasattr(points, "shape") and len(points.shape) == 2:
        if points.shape[0] == 3:
            # (3, N) format - already correct
            return points[0], points[1], points[2]
        elif points.shape[1] == 3:
            # (N, 3) format - transpose
            return points[:, 0], points[:, 1], points[:, 2]
    return points[0], points[1], points[2]


@dataclass
class TraceConstructor:
    """A class for constructing trace data with hierarchical naming.

    This class handles the construction of trace data with support for:
    - Hierarchical naming (base.key1.key2 format)
    - Static vs animated trace differentiation
    - Minimal data structure for theme-independent construction

    Attributes:
        name (str): Trace name in hierarchical format (e.g., "center_gravity.proximal.0")
        data (npt.ArrayLike): Trace point data
        time (Optional[npt.ArrayLike]): Time data for animated traces
        closed (bool): Whether the trace should be closed (connect last to first point)
        static (bool): Whether this is a static trace (doesn't change with animation)
        properties (Optional[Dict[str, Any]]): Minimal properties (e.g., visibility)
        number_frames (Optional[int]): Target number of frames for animation resizing
    """

    name: str
    data: npt.ArrayLike
    time: Optional[npt.ArrayLike] = None
    closed: bool = False
    static: bool = False
    properties: Optional[Dict[str, Any]] = None
    number_frames: Optional[int] = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure data is numpy array
        self.data = np.asarray(self.data)

        # Ensure time is numpy array if provided
        if self.time is not None:
            self.time = np.asarray(self.time)

        # Initialize properties if None
        if self.properties is None:
            self.properties = {}

        # Validate name format
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Name must be a non-empty string")

    @property
    def is_animated(self) -> bool:
        """Check if this trace is animated (has time data and is not static)."""
        return self.time is not None and not self.static

    @property
    def base_name(self) -> str:
        """Get the base name (first part before first dot)."""
        return self.name.split(".")[0]

    @property
    def name_parts(self) -> list[str]:
        """Get all parts of the hierarchical name."""
        return self.name.split(".")

    def get_hierarchical_lookup_keys(self) -> list[str]:
        """Get hierarchical lookup keys for theme resolution.

        Returns keys in order of specificity:
        - Most specific: "center_gravity.proximal.0"
        - Less specific: "center_gravity.proximal"
        - Base: "center_gravity"

        Returns:
            list[str]: List of keys from most specific to base
        """
        parts = self.name_parts
        keys = []

        # Add progressively less specific keys
        for i in range(len(parts), 0, -1):
            keys.append(".".join(parts[:i]))

        return keys

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation for TraceBuilder."""
        # Handle properties conversion
        properties_dict = {}
        if self.properties:
            if isinstance(self.properties, StandardTraceProperties):
                properties_dict = self.properties.to_dict()
            else:
                properties_dict = self.properties.copy()

        return {
            "name": self.name,
            "data": self.data,
            "time": self.time,
            "closed": self.closed,
            "static": self.static,
            "properties": properties_dict,
        }

    def __iter__(self):
        """Make TraceConstructor unpackable with ** operator.

        This allows: points_to_traces(**constructor)
        """
        # Return items from to_dict() for unpacking
        return iter(self.to_dict().items())

    def __getitem__(self, key):
        """Make TraceConstructor subscriptable like a dictionary."""
        return self.to_dict()[key]

    def __contains__(self, key):
        """Support 'in' operator for dict-like behavior."""
        return key in self.to_dict()

    def get(self, key, default=None):
        """Get method like dict.get()."""
        return self.to_dict().get(key, default)

    def keys(self):
        """Return keys for dict-like behavior."""
        return self.to_dict().keys()

    def values(self):
        """Return values for dict-like behavior."""
        return self.to_dict().values()

    def items(self):
        """Return items for dict-like behavior."""
        return self.to_dict().items()

    @property
    def is_3d(self) -> bool:
        """Check if the trace is 3D (has 3 spatial dimensions)."""
        return self.data.ndim == 3 and self.data.shape[2] == 3


# ---------- Convenience Function ----------


def build_traces_from_constructors(
    constructors: Union[List[TraceConstructor], Dict[str, TraceConstructor]],
) -> List[Union[go.Scatter, go.Scatter3d]]:
    """
    Convenience function to build multiple Plotly traces from constructors.

    Args:
        constructors: List or dictionary of TraceConstructor objects

    Returns:
        List of Plotly traces
    """
    builder = TraceBuilder()
    if isinstance(constructors, dict):
        constructors = list(constructors.values())
    return builder.build_traces(constructors)


# New system integration functions


def build_traces_with_themes(
    constructors: Union[List[TraceConstructor], Dict[str, TraceConstructor]],
) -> List[go.Scatter3d]:
    """Build plotly traces from new TraceConstructor objects with theme support.

    Args:
        constructors: List or dict of new TraceConstructor instances

    Returns:
        List[go.Scatter3d]: List of plotly traces with theme styling applied
    """
    return build_traces_from_constructors(constructors)


def create_trace_constructor(
    name: str,
    data: npt.ArrayLike,
    time: Optional[npt.ArrayLike] = None,
    closed: bool = False,
    static: bool = False,
    properties: Optional[Union[Dict[str, Any], StandardTraceProperties]] = None,
    **kwargs,
) -> TraceConstructor:
    """Convenience function to create a TraceConstructor.

    Enforces strict dimensionality: i x n x m where:
    - i = number of points in trace
    - n = time dimension
    - m = spatial dimension (2D or 3D)

    Trace Types:
    1. Single point, static -> (1, m) after processing
    2. Multi-point, static -> (i, m) after processing
    3. Single point, animated -> (1, n, m)
    4. Multi-point, animated -> (i, n, m)

    Args:
        name: Trace name in hierarchical format
        data: Trace point data
        time: Time data for animated traces
        closed: Whether trace should be closed
        static: Whether this is a static trace
        properties: Trace properties (StandardTraceProperties or dict)
        **kwargs: Additional properties merged with properties

    Returns:
        TraceConstructor: New trace constructor instance
    """
    import logging
    import warnings

    # Handle properties parameter and kwargs
    if properties is None:
        properties = kwargs
    elif isinstance(properties, dict) and kwargs:
        # Merge properties dict with kwargs
        properties = {**properties, **kwargs}
    elif isinstance(properties, StandardTraceProperties) and kwargs:
        # Convert StandardTraceProperties to dict and merge with kwargs
        properties = {**properties.to_dict(), **kwargs}
    # If properties is StandardTraceProperties and no kwargs, use as-is

    # Handle display_name -> name conversion at source
    if isinstance(properties, dict) and "display_name" in properties:
        properties = properties.copy()
        properties["name"] = properties.pop("display_name")

    # Convert to numpy array for consistent processing
    data = np.asarray(data)

    # Validation: static traces with time
    if static and time is not None:
        warnings.warn(
            f"Static trace '{name}' has time defined - ignoring time",
            UserWarning,
        )
        time = None

    # Process data based on static/animated and dimensionality
    if static:
        # Static traces: reduce dimensionality by taking first time element
        if data.ndim == 3:
            # Shape: (i, n, m) -> (i, m) - take first time element
            data = data[:, 0, :]
        elif data.ndim == 2:
            # Already correct format (i, m) - no change needed
            pass
        elif data.ndim == 1:
            # Single point: (m,) -> (1, m)
            data = data.reshape(1, -1)
        else:
            raise ValueError(
                f"Unsupported static trace dimensionality for '{name}': {data.ndim}"
            )
    else:
        # Animated traces: ensure (i, n, m) format
        if data.ndim == 2:
            # Could be (i, m) -> need to add time dimension or (n, m) single point over time
            if time is not None:
                n_time = len(time)
                if data.shape[0] == n_time:
                    # Shape: (n, m) - single point over time -> (1, n, m)
                    data = data.reshape(1, data.shape[0], data.shape[1])
                else:
                    raise ValueError(
                        f"Animated trace '{name}': data shape {data.shape} doesn't match time length {n_time}"
                    )
            else:
                raise ValueError(f"Animated trace '{name}' missing time")
        elif data.ndim == 3:
            # Already in (i, n, m) format - validate time dimension
            if time is not None:
                n_time = len(time)
                if data.shape[1] != n_time:
                    raise ValueError(
                        f"Animated trace '{name}': time dimension {data.shape[1]} doesn't match time length {n_time}"
                    )
        elif data.ndim == 1:
            raise ValueError(
                f"Animated trace '{name}' cannot have 1D data without time structure"
            )
        else:
            raise ValueError(
                f"Unsupported animated trace dimensionality for '{name}': {data.ndim}"
            )

    return TraceConstructor(
        name=name,
        data=data,
        time=time,
        closed=closed,
        static=static,
        properties=properties,
    )


def unpack_constructors(
    constructors: Dict[str, TraceConstructor], points_to_traces_func
) -> Dict[str, Any]:
    """Helper function to unpack TraceConstructor objects for trace building.

    Args:
        constructors: Dict of TraceConstructor instances
        points_to_traces_func: Function that converts trace data to traces

    Returns:
        Dict[str, Any]: Dict of traces built from constructors

    Example:
        traces = unpack_constructors(
            self.trace_constructor,
            self.points_to_traces
        )
    """
    return {
        key: points_to_traces_func(**constructor)
        for key, constructor in constructors.items()
    }


def convert_constructors_to_dicts(
    constructors: Dict[str, TraceConstructor],
) -> Dict[str, Dict[str, Any]]:
    """Convert TraceConstructor objects to dictionaries.

    Args:
        constructors: Dict of TraceConstructor instances

    Returns:
        Dict[str, Dict[str, Any]]: Dict of trace dictionaries

    Example:
        trace_dicts = convert_constructors_to_dicts(self.trace_constructor)
        traces = {
            key: self.points_to_traces(**trace_dict)
            for key, trace_dict in trace_dicts.items()
        }
    """
    return {key: constructor.to_dict() for key, constructor in constructors.items()}


class TraceBuilder:
    """
    Builds Plotly traces from TraceConstructor objects with theme support.

    Features:
    - Theme-based property merging (theme takes priority, caching handled by theme manager).
    - Handles static and animated traces.
    - Supports 2D and 3D traces, including multi-segment and closed shapes.
    - Preserves axes convention: (N_points, time, space), where:
        N_points = number of points in a trace
        time = number of frames for animated traces
        space = 2 or 3 (x, y[, z])
    """

    def __init__(self, theme_manager=None):
        """
        Initialize TraceBuilder.

        Args:
            theme_manager: Optional theme manager instance; defaults to global.
        """
        self.theme_manager = theme_manager or get_theme_manager()
        self.logger = logging.getLogger(__name__)

    # ---------- Public API ----------

    def build_trace(
        self,
        constructor: TraceConstructor,
        number_frames: Optional[int] = None,
    ) -> Union[go.Scatter, go.Scatter3d]:
        """
        Build a single Plotly trace from a TraceConstructor object.

        Args:
            constructor: TraceConstructor containing points, properties, and metadata.
            number_frames: Optional override to resample animated traces to a fixed number of frames.

        Returns:
            go.Scatter or go.Scatter3d: A Plotly trace ready for plotting.
        """
        # Always use theme resolution
        props = self._resolve_properties(constructor)

        # Prepare point arrays for plotting, optionally resampling animated traces
        x, y, z, resized_points = self._prepare_xyzn(constructor, number_frames)

        # Create Plotly trace
        trace = self._create_trace(x, y, z, props)

        # Add metadata for animated traces
        if constructor.is_animated and not constructor.static:
            trace.update(meta={"animated": True, "static": False})

        return trace

    def build_traces(
        self, constructors: List[TraceConstructor]
    ) -> List[Union[go.Scatter, go.Scatter3d]]:
        """
        Build multiple Plotly traces from a list of TraceConstructor objects.

        Args:
            constructors: List of TraceConstructor objects.

        Returns:
            List of Plotly traces.
        """
        traces: List[Union[go.Scatter, go.Scatter3d]] = []
        for c in constructors:
            try:
                traces.append(self.build_trace(c))
            except Exception as e:
                self.logger.error(
                    f"Failed to build trace '{getattr(c, 'name', '?')}': {e}"
                )
        return traces

    # ---------- Theme / Properties ----------

    def _resolve_properties(self, constructor: TraceConstructor) -> Dict[str, Any]:
        """
        Merge constructor properties with theme properties (theme takes priority).

        Performs hierarchical theme lookup via theme manager; falls back to 'default' if none found.
        Theme properties override constructor properties to allow theme manager
        control over styling priorities. All caching is handled by the theme manager.

        Args:
            constructor: TraceConstructor object.

        Returns:
            Merged dictionary of properties ready for Plotly.
        """
        # Convert constructor properties to dict if it's StandardTraceProperties
        constructor_props = constructor.properties
        # if isinstance(constructor_props, StandardTraceProperties):
        #     constructor_props = convert_standard_to_plotly(constructor_props)
        # elif constructor_props is None:
        if constructor_props is None:
            constructor_props = {}

        # Get theme properties
        for key in constructor.get_hierarchical_lookup_keys():
            theme = self.theme_manager.get_trace_theme(key)
            if theme:
                # Convert theme properties if they're StandardTraceProperties
                # if isinstance(theme, StandardTraceProperties):
                #     theme = convert_standard_to_plotly(theme)
                props = merge({}, constructor_props, theme)
                break
        else:
            default_theme = self.theme_manager.get_trace_theme("default")
            # if isinstance(default_theme, StandardTraceProperties):
            #     default_theme = convert_standard_to_plotly(default_theme)
            props = merge({}, constructor_props, default_theme or {})

        # Inject human-readable name from constructor (trace identity)
        if constructor.name and constructor.name != "trace":
            props["name"] = constructor.name

        return props

    # ---------- Data Preparation ----------

    def _prepare_xyzn(
        self, constructor: TraceConstructor, number_frames: Optional[int]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare X, Y, Z arrays for plotting, and optionally return resized points.

        Maintains your axis convention:
            axis 0 = points
            axis 1 = time
            axis 2 = space (2 or 3)

        Args:
            constructor: TraceConstructor containing points and metadata.
            number_frames: Optional number of frames to resample animated traces.

        Returns:
            Tuple:
                x: 1D array of X coordinates (points along first frame)
                y: 1D array of Y coordinates
                z: 1D array of Z coordinates, or None if 2D
                resized_points: 3D array (N_points, time, space) for animated traces, else None
        """
        points = np.asarray(constructor.data)
        if points.ndim == 0:
            raise ValueError("data must not be a scalar")
        if points.size == 0:
            raise ValueError(f"Trace '{constructor.name}' has empty data array")

        resized_points: Optional[np.ndarray] = None

        if constructor.is_animated and points.ndim == 3:
            # Animated trace: resample along axis 1 (time)
            points = self._resample_time_axis(points, number_frames)
            resized_points = points

            # Extract first frame (time=0) for plotting
            frame0 = points[:, 0, :]  # shape: (N_points, space)
            x = frame0[:, 0]
            y = frame0[:, 1]
            z = frame0[:, 2] if frame0.shape[1] > 2 else None

        else:
            # Static traces
            if points.ndim == 1:
                # Single point
                if len(points) < 2:
                    raise ValueError(
                        f"Trace '{constructor.name}' needs at least 2 coordinates (x, y), got {len(points)}"
                    )
                x = points[0:1]  # Keep as array for consistency
                y = points[1:2]
                z = points[2:3] if len(points) > 2 else None
            elif points.ndim == 2:
                # Multiple points in 2D or 3D: (N_points, space)
                if points.shape[0] == 0:
                    raise ValueError(f"Trace '{constructor.name}' has no points")
                if points.shape[1] < 2:
                    raise ValueError(
                        f"Trace '{constructor.name}' needs at least 2 coordinates (x, y), got {points.shape[1]}"
                    )
                x = points[:, 0]
                y = points[:, 1]
                z = points[:, 2] if points.shape[1] > 2 else None
            elif points.ndim == 3:
                # Multi-segment static: flatten segments for plotting
                if points.shape[2] in (2, 3):
                    flat = points.reshape(-1, points.shape[2])
                    x = flat[:, 0]
                    y = flat[:, 1]
                    z = flat[:, 2] if flat.shape[1] > 2 else None
                elif points.shape[0] in (2, 3):
                    flat = np.moveaxis(points, 0, -1).reshape(-1, points.shape[0])
                    x = flat[:, 0]
                    y = flat[:, 1]
                    z = flat[:, 2] if flat.shape[1] > 2 else None
                else:
                    raise ValueError(
                        f"Unsupported static 3D array shape: {points.shape}"
                    )
            else:
                raise ValueError(f"Unsupported points dimensionality: {points.ndim}")

        # Close polygon if requested
        if getattr(constructor, "closed", False) and len(x) > 0:
            x = np.concatenate([x, x[:1]])
            y = np.concatenate([y, y[:1]])
            if z is not None:
                z = np.concatenate([z, z[:1]])

        return x, y, z, resized_points

    def _resample_time_axis(
        self, points: np.ndarray, target_frames: Optional[int]
    ) -> np.ndarray:
        """
        Resample animated traces along the time axis (axis=1) using linear interpolation.

        Args:
            points: Input points of shape (N_points, time, space)
            target_frames: Number of frames to interpolate to. If None, no resampling.

        Returns:
            points: Resampled points, same axes convention (N_points, target_frames, space)
        """
        N, T, C = points.shape
        if target_frames is None or T == target_frames:
            return points

        # Interpolation indices
        old_idx = np.linspace(0, T - 1, T)
        new_idx = np.linspace(0, T - 1, target_frames)
        i0 = np.floor(new_idx).astype(int)
        i1 = np.clip(i0 + 1, 0, T - 1)
        w = (new_idx - i0).reshape(
            (1, target_frames, 1)
        )  # broadcast over points and space

        # Vectorized linear interpolation
        a = points[:, i0, :]
        b = points[:, i1, :]
        return a * (1 - w) + b * w

    # ---------- Trace Creation ----------

    def _create_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: Optional[np.ndarray],
        props: Dict[str, Any],
    ) -> Union[go.Scatter, go.Scatter3d]:
        """
        Create a Plotly Scatter or Scatter3d trace.

        Args:
            x: X coordinates (1D array)
            y: Y coordinates (1D array)
            z: Z coordinates (1D array) or None for 2D
            props: Dictionary of Plotly properties

        Returns:
            go.Scatter or go.Scatter3d
        """
        if z is None:
            return go.Scatter(x=x, y=y, **props)
        return go.Scatter3d(x=x, y=y, z=z, **props)
