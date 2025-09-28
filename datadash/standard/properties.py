# # Standard Library-Independent Trace Properties
# # Author : Matthew Davidson
# # 2023/01/23
# # Davidson Engineering Ltd. © 2023

# from __future__ import annotations
from dataclasses import dataclass, field

# from typing import Optional, Literal, Union
# from enum import Enum


# class LineStyle(Enum):
#     """Standard line styles supported across plotting libraries."""
#     SOLID = "solid"
#     DASHED = "dashed"
#     DOTTED = "dotted"
#     DASH_DOT = "dash_dot"


# class MarkerSymbol(Enum):
#     """Standard marker symbols supported across plotting libraries."""
#     CIRCLE = "circle"
#     SQUARE = "square"
#     TRIANGLE = "triangle"
#     DIAMOND = "diamond"
#     CROSS = "cross"
#     PLUS = "plus"
#     STAR = "star"
#     X = "x"


# class TraceMode(Enum):
#     """Standard trace display modes."""
#     LINES = "lines"
#     MARKERS = "markers"
#     LINES_AND_MARKERS = "lines+markers"
#     TEXT = "text"


@dataclass
class StandardTraceProperties:
    """
    Standard library-independent trace properties.

    These properties can be converted to any plotting library
    (Plotly, Matplotlib, etc.) using appropriate converters.
    """

    # Visibility
    visible: bool = True
    opacity: float = 1.0


#     # Display mode
#     mode: TraceMode = TraceMode.LINES_AND_MARKERS

#     # Line properties
#     line_color: str = "#007bff"  # Default blue
#     line_width: float = 2.0
#     line_style: LineStyle = LineStyle.SOLID

#     # Marker properties
#     marker_color: Optional[str] = None  # Defaults to line_color if None
#     marker_size: float = 6.0
#     marker_symbol: MarkerSymbol = MarkerSymbol.CIRCLE
#     marker_line_color: Optional[str] = None
#     marker_line_width: float = 0.0

#     # Text properties
#     text_color: str = "#333333"
#     text_size: float = 12.0
#     text_font: str = "Arial"

#     # Fill properties (for filled shapes)
#     fill_color: Optional[str] = None
#     fill_opacity: float = 0.5

#     # Hover properties
#     hover_text: Optional[str] = None
#     hover_template: Optional[str] = None

#     # Legend properties
#     show_legend: bool = True
#     legend_group: Optional[str] = None

#     # Custom properties for specific use cases
#     custom: dict = field(default_factory=dict)

#     def __post_init__(self):
#         """Post-initialization processing."""
#         # If marker_color is not set, use line_color
#         if self.marker_color is None:
#             self.marker_color = self.line_color

#         # If marker_line_color is not set, use line_color
#         if self.marker_line_color is None:
#             self.marker_line_color = self.line_color

#     def to_dict(self) -> dict:
#         """Convert to dictionary representation."""
#         result = {}
#         for key, value in self.__dict__.items():
#             if isinstance(value, Enum):
#                 result[key] = value.value
#             else:
#                 result[key] = value
#         return result

#     @classmethod
#     def from_dict(cls, data: dict) -> 'StandardTraceProperties':
#         """Create from dictionary representation."""
#         # Convert enum string values back to enums
#         processed_data = data.copy()

#         if 'line_style' in processed_data:
#             processed_data['line_style'] = LineStyle(processed_data['line_style'])

#         if 'marker_symbol' in processed_data:
#             processed_data['marker_symbol'] = MarkerSymbol(processed_data['marker_symbol'])

#         if 'mode' in processed_data:
#             processed_data['mode'] = TraceMode(processed_data['mode'])

#         return cls(**processed_data)

#     def merge_with(self, other: 'StandardTraceProperties') -> 'StandardTraceProperties':
#         """Merge with another properties object (other takes priority)."""
#         # Convert both to dicts
#         base_dict = self.to_dict()
#         other_dict = other.to_dict()

#         # Merge dictionaries (other takes priority)
#         merged_dict = {**base_dict, **other_dict}

#         # Handle custom properties specially (merge instead of replace)
#         if 'custom' in base_dict and 'custom' in other_dict:
#             merged_dict['custom'] = {**base_dict['custom'], **other_dict['custom']}

#         return StandardTraceProperties.from_dict(merged_dict)


# # Predefined property templates for common use cases
# class StandardPropertyTemplates:
#     """Predefined property templates for common trace types."""

#     @staticmethod
#     def line_trace(color: str = "#007bff", width: float = 2.0) -> StandardTraceProperties:
#         """Standard line trace properties."""
#         return StandardTraceProperties(
#             mode=TraceMode.LINES,
#             line_color=color,
#             line_width=width,
#             marker_size=0  # No markers for line-only traces
#         )

#     @staticmethod
#     def scatter_trace(color: str = "#007bff", size: float = 6.0) -> StandardTraceProperties:
#         """Standard scatter trace properties."""
#         return StandardTraceProperties(
#             mode=TraceMode.MARKERS,
#             marker_color=color,
#             marker_size=size,
#             line_width=0  # No lines for marker-only traces
#         )

#     @staticmethod
#     def line_and_marker_trace(color: str = "#007bff") -> StandardTraceProperties:
#         """Standard line + marker trace properties."""
#         return StandardTraceProperties(
#             mode=TraceMode.LINES_AND_MARKERS,
#             line_color=color,
#             marker_color=color,
#             line_width=2.0,
#             marker_size=6.0
#         )

#     @staticmethod
#     def dashed_line(color: str = "#007bff") -> StandardTraceProperties:
#         """Dashed line trace properties."""
#         return StandardTraceProperties(
#             mode=TraceMode.LINES,
#             line_color=color,
#             line_style=LineStyle.DASHED,
#             line_width=2.0
#         )

#     @staticmethod
#     def robot_link(color: str = "#ff0000", width: float = 3.0) -> StandardTraceProperties:
#         """Robot link trace properties (thick, solid line)."""
#         return StandardTraceProperties(
#             mode=TraceMode.LINES,
#             line_color=color,
#             line_width=width,
#             line_style=LineStyle.SOLID,
#             opacity=0.8
#         )

#     @staticmethod
#     def trajectory_path(color: str = "#00ff00") -> StandardTraceProperties:
#         """Trajectory path properties."""
#         return StandardTraceProperties(
#             mode=TraceMode.LINES,
#             line_color=color,
#             line_width=2.5,
#             line_style=LineStyle.SOLID,
#             opacity=0.7
#         )
