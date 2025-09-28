# Standardized Library-Independent Trace Constructor
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from __future__ import annotations
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
import numpy as np
import numpy.typing as npt

from .properties import StandardTraceProperties

if TYPE_CHECKING:
    from ..builders.trace import TraceConstructor


def create_standard_trace_constructor(
    name: str,
    data: npt.ArrayLike,
    time: Optional[npt.ArrayLike] = None,
    closed: bool = False,
    static: bool = False,
    properties: Optional[Union[StandardTraceProperties, Dict[str, Any]]] = None,
) -> "TraceConstructor":
    """
    Create a standardized library-independent trace constructor.

    Returns a TraceConstructor object with StandardTraceProperties that can be
    used by any plotting library after appropriate conversion.

    Args:
        name: Hierarchical trace name (e.g., "robot.link_0.center_of_mass")
        data: Point data array with shape:
              - Static 2D: (n_points, 2) for x,y coordinates
              - Static 3D: (n_points, 3) for x,y,z coordinates
              - Animated: (n_points, n_frames, space) for animated traces
        time: Time array for animated traces (length n_frames)
        closed: Whether to connect the last point to the first point
        static: Whether the trace is static (doesn't change over time)
        properties: Trace styling properties (StandardTraceProperties or dict)

    Returns:
        TraceConstructor object with standardized properties
    """
    # Import here to avoid circular imports
    from ..builders.trace import TraceConstructor

    # Convert data to numpy array
    data_array = np.asarray(data)

    # Handle properties
    # if properties is None:
    #     properties = StandardTraceProperties()
    # elif isinstance(properties, dict):
    #     properties = StandardTraceProperties.from_dict(properties)
    properties = properties or {}

    # Validate data dimensions
    if data_array.ndim < 2:
        raise ValueError(f"Data must be at least 2D, got shape {data_array.shape}")

    # Validate time data for animated traces
    if not static and time is not None:
        time_array = np.asarray(time)
        if data_array.ndim == 3:
            expected_frames = data_array.shape[1]
            if len(time_array) != expected_frames:
                raise ValueError(
                    f"Time array length {len(time_array)} doesn't match "
                    f"data frames {expected_frames}"
                )

    # Create TraceConstructor with standardized properties
    return TraceConstructor(
        name=name,
        data=data_array,
        time=np.asarray(time) if time is not None else None,
        closed=closed,
        static=static,
        properties=properties,
    )


def create_line_trace(
    name: str,
    x_data: npt.ArrayLike,
    y_data: npt.ArrayLike,
    z_data: Optional[npt.ArrayLike] = None,
    color: str = "#007bff",
    width: float = 2.0,
    **kwargs,
) -> Dict[str, Any]:
    """
    Create a standardized line trace constructor.

    Args:
        name: Trace name
        x_data: X coordinates
        y_data: Y coordinates
        z_data: Z coordinates (optional, for 3D traces)
        color: Line color
        width: Line width
        **kwargs: Additional properties

    Returns:
        Standardized trace constructor dictionary
    """
    # Combine coordinates
    if z_data is not None:
        data = np.column_stack([x_data, y_data, z_data])
    else:
        data = np.column_stack([x_data, y_data])

    # Create properties
    # from .properties import StandardPropertyTemplates

    # properties = StandardPropertyTemplates.line_trace(color=color, width=width)

    # Apply additional properties
    # for key, value in kwargs.items():
    #     if hasattr(properties, key):
    #         setattr(properties, key, value)

    return create_standard_trace_constructor(
        name=name, data=data, static=True, properties=kwargs
    )


# def create_scatter_trace(
#     name: str,
#     x_data: npt.ArrayLike,
#     y_data: npt.ArrayLike,
#     z_data: Optional[npt.ArrayLike] = None,
#     color: str = "#007bff",
#     size: float = 6.0,
#     **kwargs
# ) -> Dict[str, Any]:
#     """
#     Create a standardized scatter trace constructor.

#     Args:
#         name: Trace name
#         x_data: X coordinates
#         y_data: Y coordinates
#         z_data: Z coordinates (optional, for 3D traces)
#         color: Marker color
#         size: Marker size
#         **kwargs: Additional properties

#     Returns:
#         Standardized trace constructor dictionary
#     """
#     # Combine coordinates
#     if z_data is not None:
#         data = np.column_stack([x_data, y_data, z_data])
#     else:
#         data = np.column_stack([x_data, y_data])

#     # Create properties
#     from .properties import StandardPropertyTemplates
#     properties = StandardPropertyTemplates.scatter_trace(color=color, size=size)

#     # Apply additional properties
#     for key, value in kwargs.items():
#         if hasattr(properties, key):
#             setattr(properties, key, value)

#     return create_standard_trace_constructor(
#         name=name,
#         data=data,
#         static=True,
#         properties=properties
#     )


# def create_animated_trace(
#     name: str,
#     data: npt.ArrayLike,
#     time: npt.ArrayLike,
#     properties: Optional[Union[StandardTraceProperties, Dict[str, Any]]] = None
# ) -> Dict[str, Any]:
#     """
#     Create a standardized animated trace constructor.

#     Args:
#         name: Trace name
#         data: Animation data with shape (n_points, n_frames, space)
#         time: Time array with length n_frames
#         properties: Trace properties

#     Returns:
#         Standardized trace constructor dictionary
#     """
#     return create_standard_trace_constructor(
#         name=name,
#         data=data,
#         time=time,
#         static=False,
#         properties=properties
#     )


# def create_robot_link_trace(
#     name: str,
#     start_point: npt.ArrayLike,
#     end_point: npt.ArrayLike,
#     color: str = "#ff0000",
#     width: float = 3.0
# ) -> Dict[str, Any]:
#     """
#     Create a standardized robot link trace constructor.

#     Args:
#         name: Link name (e.g., "robot.link_0")
#         start_point: Starting point coordinates
#         end_point: Ending point coordinates
#         color: Link color
#         width: Link width

#     Returns:
#         Standardized trace constructor dictionary
#     """
#     # Create line data between start and end points
#     data = np.array([start_point, end_point])

#     # Create robot link properties
#     from .properties import StandardPropertyTemplates
#     properties = StandardPropertyTemplates.robot_link(color=color, width=width)

#     return create_standard_trace_constructor(
#         name=name,
#         data=data,
#         static=True,
#         properties=properties
#     )


# def batch_create_traces_from_data(
#     data_dict: Dict[str, Dict[str, Any]]
# ) -> Dict[str, "TraceConstructor"]:
#     """
#     Create multiple standardized trace constructors from a data dictionary.

#     Args:
#         data_dict: Dictionary where keys are trace names and values are
#                   dictionaries containing trace data and parameters

#     Returns:
#         Dictionary of TraceConstructor objects

#     Example:
#         data = {
#             "trajectory": {
#                 "x": x_data,
#                 "y": y_data,
#                 "color": "#00ff00",
#                 "trace_type": "line"
#             },
#             "robot.joint_0": {
#                 "start": [0, 0, 0],
#                 "end": [1, 0, 0],
#                 "color": "#ff0000",
#                 "trace_type": "robot_link"
#             }
#         }
#         traces = batch_create_traces_from_data(data)
#     """
#     trace_constructors = {}

#     for name, params in data_dict.items():
#         trace_type = params.get("trace_type", "line")

#         if trace_type == "line":
#             trace_constructors[name] = create_line_trace(
#                 name=name,
#                 x_data=params["x"],
#                 y_data=params["y"],
#                 z_data=params.get("z"),
#                 color=params.get("color", "#007bff"),
#                 width=params.get("width", 2.0)
#             )
#         elif trace_type == "scatter":
#             trace_constructors[name] = create_scatter_trace(
#                 name=name,
#                 x_data=params["x"],
#                 y_data=params["y"],
#                 z_data=params.get("z"),
#                 color=params.get("color", "#007bff"),
#                 size=params.get("size", 6.0)
#             )
#         elif trace_type == "robot_link":
#             trace_constructors[name] = create_robot_link_trace(
#                 name=name,
#                 start_point=params["start"],
#                 end_point=params["end"],
#                 color=params.get("color", "#ff0000"),
#                 width=params.get("width", 3.0)
#             )
#         else:
#             raise ValueError(f"Unknown trace type: {trace_type}")

#     return trace_constructors
