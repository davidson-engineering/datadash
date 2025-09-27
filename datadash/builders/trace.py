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
from ..utils import repeat, tile

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


def trace_constructor(
    x=None, y=None, z=None, points=None, properties=None, name="trace0"
):
    properties = {} if properties is None else properties

    if points is not None:
        # Convert points to x,y,z arrays
        x, y, z = points_to_xyz_arrays(points)

    # Determine trace type based on presence of z
    trace_type = go.Scatter3d if z is not None else go.Scatter
    trace_data = {"x": x, "y": y}
    if z is not None:
        trace_data["z"] = z

    return {name: dict(**trace_data, trace_type=trace_type, properties=properties)}


def combined_trace_constructor(x, y, headers: str = "xyz", hover_template=None):
    trace_dict = {}
    for key, axis in zip(headers, np.asarray(y).T):
        trace_data = dict(
            x=x,
            y=axis,
            trace_type=go.Scatter,
            name=key,  # Use the provided header as the trace name
        )
        if hover_template:
            trace_data["hovertemplate"] = hover_template
        trace_dict[key] = trace_data
    return trace_dict


def subplots_trace_constructor(
    x,
    y,
    rows=None,
    cols=None,
    trace_constructor=None,
    hover_template=None,
    headers="xyz",
):
    if rows is None:
        rows = [1, 2, 3]
    if cols is None:
        cols = [1, 1, 1]

    data = []
    for axis, header in zip(np.asarray(y).T, headers):
        trace_data = dict(
            x=x,
            y=axis,
            trace_type=go.Scatter,
            name=header,
        )
        if hover_template:
            trace_data["hovertemplate"] = hover_template
        data.append(trace_data)

    return dict(
        data=data,
        rows=rows,
        cols=cols,
        properties=trace_constructor,
    )


def subplots_trace_constructor_combined_joints_axes(
    x, y, rows=None, cols=None, hover_template=None, headers=None
):

    if rows is None:
        rows = tile([1, 2, 3], 3)
    if cols is None:
        cols = repeat([1, 2, 3], 3)

    # Ensure y is properly shaped for iteration over traces
    y_array = np.asarray(y)
    if y_array.ndim == 2:
        # If 2D, transpose so we iterate over columns (traces)
        y_traces = y_array.T
    else:
        # If 1D or already properly shaped, use as-is
        y_traces = y

    # Default headers if not provided
    if headers is None:
        headers = [f"trace_{i}" for i in range(len(y_traces))]

    data = []
    for yi, header in zip(y_traces, headers):
        trace_data = dict(
            x=x,
            y=yi,
            trace_type=go.Scatter,
            name=header,  # Use the provided header as the trace name
        )
        if hover_template:
            trace_data["hovertemplate"] = hover_template
        data.append(trace_data)

    return dict(
        data=data,
        rows=rows,
        cols=cols,
    )


@dataclass
class TraceConstructor:
    """A class for constructing trace data with hierarchical naming.

    This class handles the construction of trace data with support for:
    - Hierarchical naming (base.key1.key2 format)
    - Static vs animated trace differentiation
    - Minimal data structure for theme-independent construction

    Attributes:
        name (str): Trace name in hierarchical format (e.g., "center_gravity.proximal.0")
        points (npt.ArrayLike): Trace point data
        points_time (Optional[npt.ArrayLike]): Time data for animated traces
        closed (bool): Whether the trace should be closed (connect last to first point)
        static (bool): Whether this is a static trace (doesn't change with animation)
        properties (Optional[Dict[str, Any]]): Minimal properties (e.g., visibility)
        number_frames (Optional[int]): Target number of frames for animation resizing
    """

    name: str
    points: npt.ArrayLike
    points_time: Optional[npt.ArrayLike] = None
    closed: bool = False
    static: bool = False
    properties: Optional[Dict[str, Any]] = None
    number_frames: Optional[int] = None

    def __post_init__(self):
        """Post-initialization processing."""
        # Ensure points is numpy array
        self.points = np.asarray(self.points)

        # Ensure points_time is numpy array if provided
        if self.points_time is not None:
            self.points_time = np.asarray(self.points_time)

        # Initialize properties if None
        if self.properties is None:
            self.properties = {}

        # Validate name format
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Name must be a non-empty string")

    @property
    def is_animated(self) -> bool:
        """Check if this trace is animated (has time data and is not static)."""
        return self.points_time is not None and not self.static

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
        return {
            "name": self.name,
            "points": self.points,
            "points_time": self.points_time,
            "closed": self.closed,
            "static": self.static,
            "properties": self.properties.copy() if self.properties else {},
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
        return self.points.ndim == 3 and self.points.shape[2] == 3


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
    points: npt.ArrayLike,
    points_time: Optional[npt.ArrayLike] = None,
    closed: bool = False,
    static: bool = False,
    **properties,
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
        points: Trace point data
        points_time: Time data for animated traces
        closed: Whether trace should be closed
        static: Whether this is a static trace
        **properties: Additional properties

    Returns:
        TraceConstructor: New trace constructor instance
    """
    import logging
    import warnings

    # Handle display_name -> name conversion at source
    if "display_name" in properties:
        properties = properties.copy()
        properties["name"] = properties.pop("display_name")

    # Convert to numpy array for consistent processing
    points = np.asarray(points)

    # Validation: static traces with points_time
    if static and points_time is not None:
        warnings.warn(
            f"Static trace '{name}' has points_time defined - ignoring points_time",
            UserWarning,
        )
        points_time = None

    # Process points based on static/animated and dimensionality
    if static:
        # Static traces: reduce dimensionality by taking first time element
        if points.ndim == 3:
            # Shape: (i, n, m) -> (i, m) - take first time element
            points = points[:, 0, :]
        elif points.ndim == 2:
            # Already correct format (i, m) - no change needed
            pass
        elif points.ndim == 1:
            # Single point: (m,) -> (1, m)
            points = points.reshape(1, -1)
        else:
            raise ValueError(
                f"Unsupported static trace dimensionality for '{name}': {points.ndim}"
            )
    else:
        # Animated traces: ensure (i, n, m) format
        if points.ndim == 2:
            # Could be (i, m) -> need to add time dimension or (n, m) single point over time
            if points_time is not None:
                n_time = len(points_time)
                if points.shape[0] == n_time:
                    # Shape: (n, m) - single point over time -> (1, n, m)
                    points = points.reshape(1, points.shape[0], points.shape[1])
                else:
                    raise ValueError(
                        f"Animated trace '{name}': points shape {points.shape} doesn't match time length {n_time}"
                    )
            else:
                raise ValueError(f"Animated trace '{name}' missing points_time")
        elif points.ndim == 3:
            # Already in (i, n, m) format - validate time dimension
            if points_time is not None:
                n_time = len(points_time)
                if points.shape[1] != n_time:
                    raise ValueError(
                        f"Animated trace '{name}': time dimension {points.shape[1]} doesn't match points_time length {n_time}"
                    )
        elif points.ndim == 1:
            raise ValueError(
                f"Animated trace '{name}' cannot have 1D points without time structure"
            )
        else:
            raise ValueError(
                f"Unsupported animated trace dimensionality for '{name}': {points.ndim}"
            )

    return TraceConstructor(
        name=name,
        points=points,
        points_time=points_time,
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
    - Theme-based property merging with caching for performance.
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
        self._theme_cache: Dict[str, Dict[str, Any]] = {}  # Cache theme lookups

    # ---------- Public API ----------

    def build_trace(
        self, constructor: TraceConstructor, number_frames: Optional[int] = None, use_precomputed_themes: bool = True
    ) -> Union[go.Scatter, go.Scatter3d]:
        """
        Build a single Plotly trace from a TraceConstructor object.

        Args:
            constructor: TraceConstructor containing points, properties, and metadata.
            number_frames: Optional override to resample animated traces to a fixed number of frames.
            use_precomputed_themes: If True, assumes constructor.properties contains pre-computed themes
                                   and skips additional theme resolution.

        Returns:
            go.Scatter or go.Scatter3d: A Plotly trace ready for plotting.
        """
        # Get properties - either use pre-computed themes or resolve them
        if use_precomputed_themes:
            # Use properties as-is (already themed by theme manager)
            props = constructor.properties.copy()
            # Ensure name is set
            if constructor.name and constructor.name != "trace":
                props["name"] = constructor.name
        else:
            # Traditional theme resolution
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
        self, constructors: List[TraceConstructor], use_precomputed_themes: bool = True
    ) -> List[Union[go.Scatter, go.Scatter3d]]:
        """
        Build multiple Plotly traces from a list of TraceConstructor objects.

        Args:
            constructors: List of TraceConstructor objects.
            use_precomputed_themes: If True, assumes properties contain pre-computed themes.

        Returns:
            List of Plotly traces.
        """
        traces: List[Union[go.Scatter, go.Scatter3d]] = []
        for c in constructors:
            try:
                traces.append(self.build_trace(c, use_precomputed_themes=use_precomputed_themes))
            except Exception as e:
                self.logger.error(
                    f"Failed to build trace '{getattr(c, 'name', '?')}': {e}"
                )
        return traces

    # ---------- Theme / Properties ----------

    def _resolve_properties(self, constructor: TraceConstructor) -> Dict[str, Any]:
        """
        Merge theme properties with constructor properties.

        Performs hierarchical theme lookup; falls back to 'default' if none found.

        Args:
            constructor: TraceConstructor object.

        Returns:
            Merged dictionary of properties ready for Plotly.
        """
        for key in constructor.get_hierarchical_lookup_keys():
            theme = self._get_theme_cached(key)
            if theme:
                props = merge({}, theme, constructor.properties)
                break
        else:
            props = merge({}, self._get_theme_cached("default"), constructor.properties)

        # Inject human-readable name, giving priority to constructor over theme
        if constructor.name and constructor.name != "trace":
            props["name"] = constructor.name

        return props

    def _get_theme_cached(self, key: str) -> Dict[str, Any]:
        """
        Retrieve a cached theme dict for a key or query the theme manager.

        Args:
            key: Hierarchical theme key.

        Returns:
            Theme dictionary (may be empty if no theme found).
        """
        if key not in self._theme_cache:
            self._theme_cache[key] = self.theme_manager.get_trace_theme(key) or {}
        return self._theme_cache[key]

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
        points = np.asarray(constructor.points)
        if points.ndim == 0:
            raise ValueError("points must not be a scalar")

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
                x = points[0]
                y = points[1]
                z = points[2] if len(points) > 2 else None
            elif points.ndim == 2:
                # Multiple points in 2D or 3D: (N_points, space)
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


# Legacy conversion function removed - no longer needed
