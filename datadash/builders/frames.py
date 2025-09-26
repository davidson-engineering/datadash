# Frames Constructor
# Author : Matthew Davidson
# 2025/01/05
# Davidson Engineering Ltd. © 2025

"""
FramesConstructor class for building Plotly animation frames from TraceConstructor objects.
This separates frame construction logic from the main figure building.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import plotly.graph_objects as go
import logging

from .trace import TraceConstructor
from .trace import TraceBuilder


class FramesConstructor:
    """Constructs animation frames from TraceConstructor objects.

    This class handles:
    - Separation of static vs animated traces
    - Pre-processing and slicing of animated trace data
    - Building frame data with proper trace indexing
    - Optimized frame generation for animations
    """

    def __init__(
        self, trace_constructors: Dict[str, TraceConstructor], number_frames: int
    ):
        """Initialize FramesConstructor.

        Args:
            trace_constructors: Dict of TraceConstructor instances
            number_frames: Target number of animation frames
        """
        self.trace_constructors = trace_constructors
        self.number_frames = number_frames
        self.logger = logging.getLogger(__name__)

        # Build trace index mapping
        self.trace_indices = {key: i for i, key in enumerate(trace_constructors.keys())}

        # Separate static and animated constructors
        self.animated_constructors = {
            key: constructor
            for key, constructor in trace_constructors.items()
            if constructor.is_animated
        }

        self.static_constructors = {
            key: constructor
            for key, constructor in trace_constructors.items()
            if not constructor.is_animated
        }

        self.animated_indices = [
            self.trace_indices[key] for key in self.animated_constructors.keys()
        ]

        # Initialize processing components
        self.builder = TraceBuilder()
        self.frame_templates = {}
        self.pre_sliced = {}
        self.static_frame_traces = []

    def build_frames(self) -> List[go.Frame]:
        """Build animation frames.

        Returns:
            List[go.Frame]: List of Plotly frame objects
        """
        if not self.animated_constructors:
            return []

        self._prepare_animated_traces()
        self._prepare_static_traces()
        return self._build_frame_objects()

    def _prepare_animated_traces(self):
        """Prepare animated traces with pre-slicing for performance."""
        for key, constructor in self.animated_constructors.items():
            # Create constructor with target frame count
            resized_constructor = TraceConstructor(
                name=constructor.name,
                points=constructor.points,
                points_time=constructor.points_time,
                closed=constructor.closed,
                static=constructor.static,
                properties=constructor.properties,
                number_frames=self.number_frames,
            )

            # Build template trace
            trace = self.builder.build_trace(resized_constructor)
            template = trace.to_plotly_json()

            # Get resized points in (i, n, m) format
            _, _, _, resized_points = self.builder._prepare_xyzn(
                resized_constructor, self.number_frames
            )

            # Store template info
            self.frame_templates[key] = {
                "template": template,
                "trace_index": self.trace_indices[key],
                "is_3d": resized_constructor.is_3d,
                "closed": constructor.closed,
            }

            # Pre-slice along time axis (n dimension) for performance
            # Each slice is (i, m) - all points with spatial coordinates for that frame
            self.pre_sliced[key] = [
                resized_points[:, f, :] for f in range(self.number_frames)
            ]

    def _prepare_static_traces(self):
        """Prepare static traces for the first frame."""
        for key, constructor in self.static_constructors.items():
            template = self.builder.build_trace(constructor).to_plotly_json()
            self.static_frame_traces.append(template)

    def _build_frame_objects(self) -> List[go.Frame]:
        """Build the actual frame objects."""
        frames = []

        for f in range(self.number_frames):
            if f == 0:
                # First frame: build all traces in correct order
                frame_traces = self._build_first_frame()
                frames.append(go.Frame(data=frame_traces))
            else:
                # Subsequent frames: only animated traces
                frame_traces = self._build_animated_frame(f)
                frames.append(go.Frame(data=frame_traces, traces=self.animated_indices))

        return frames

    def _build_first_frame(self) -> List[dict]:
        """Build the first frame with all traces in correct order."""
        # Create ordered list based on original trace indices
        ordered_traces = [None] * len(self.trace_constructors)

        # Place animated traces at frame 0
        for key, template_info in self.frame_templates.items():
            if key in self.animated_constructors:
                pts = self.pre_sliced[key][0]
                trace_dict = dict(template_info["template"])

                # Update trace data for frame 0
                if template_info["is_3d"]:
                    trace_dict["x"] = pts[:, 0]
                    trace_dict["y"] = pts[:, 1]
                    trace_dict["z"] = pts[:, 2]
                    if template_info["closed"] and len(pts) > 0:
                        trace_dict["x"] = np.append(trace_dict["x"], trace_dict["x"][0])
                        trace_dict["y"] = np.append(trace_dict["y"], trace_dict["y"][0])
                        trace_dict["z"] = np.append(trace_dict["z"], trace_dict["z"][0])
                else:
                    trace_dict["x"] = pts[:, 0]
                    trace_dict["y"] = pts[:, 1]
                    if template_info["closed"] and len(pts) > 0:
                        trace_dict["x"] = np.append(trace_dict["x"], trace_dict["x"][0])
                        trace_dict["y"] = np.append(trace_dict["y"], trace_dict["y"][0])

                ordered_traces[template_info["trace_index"]] = trace_dict

        # Place static traces
        static_idx = 0
        for key, constructor in self.static_constructors.items():
            trace_index = self.trace_indices[key]
            ordered_traces[trace_index] = self.static_frame_traces[static_idx]
            static_idx += 1

        return ordered_traces

    def _build_animated_frame(self, frame_idx: int) -> List[dict]:
        """Build frame with only animated traces."""
        frame_traces = []

        for key, template_info in self.frame_templates.items():
            if key in self.animated_constructors:
                pts = self.pre_sliced[key][frame_idx]
                trace_dict = dict(template_info["template"])

                # Update trace data for this frame
                if template_info["is_3d"]:
                    trace_dict["x"] = pts[:, 0]
                    trace_dict["y"] = pts[:, 1]
                    trace_dict["z"] = pts[:, 2]
                    if template_info["closed"] and len(pts) > 0:
                        trace_dict["x"] = np.append(trace_dict["x"], trace_dict["x"][0])
                        trace_dict["y"] = np.append(trace_dict["y"], trace_dict["y"][0])
                        trace_dict["z"] = np.append(trace_dict["z"], trace_dict["z"][0])
                else:
                    trace_dict["x"] = pts[:, 0]
                    trace_dict["y"] = pts[:, 1]
                    if template_info["closed"] and len(pts) > 0:
                        trace_dict["x"] = np.append(trace_dict["x"], trace_dict["x"][0])
                        trace_dict["y"] = np.append(trace_dict["y"], trace_dict["y"][0])

                frame_traces.append(trace_dict)

        return frame_traces
