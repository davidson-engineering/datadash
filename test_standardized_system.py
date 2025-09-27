#!/usr/bin/env python3
"""
Test script for the new standardized trace constructor system.

This script tests the integration of:
- StandardTraceProperties
- Standard trace constructors
- Property converters (standard → plotly)
- TraceBuilder with standard properties
"""

import numpy as np
import sys
import os

# Add the datadash package to Python path
sys.path.insert(0, os.path.dirname(__file__))

from datadash.standard.properties import (
    StandardTraceProperties,
    StandardPropertyTemplates,
    LineStyle,
    MarkerSymbol,
    TraceMode,
)

from datadash.standard.trace_constructor import (
    create_line_trace,
    create_scatter_trace,
    create_standard_trace_constructor,
    create_animated_trace,
    create_robot_link_trace,
)

from datadash.converters.plotly_converter import convert_standard_to_plotly
from datadash.builders.trace import TraceBuilder


def test_standard_properties():
    """Test StandardTraceProperties creation and conversion."""
    print("Testing StandardTraceProperties...")

    # Test default properties
    props = StandardTraceProperties()
    assert props.line_color == "#007bff"
    assert props.mode == TraceMode.LINES_AND_MARKERS

    # Test property templates
    line_props = StandardPropertyTemplates.line_trace(color="#ff0000", width=3.0)
    assert line_props.line_color == "#ff0000"
    assert line_props.line_width == 3.0
    assert line_props.mode == TraceMode.LINES

    # Test dict conversion
    props_dict = props.to_dict()
    assert "line_color" in props_dict
    assert "mode" in props_dict

    # Test from_dict
    new_props = StandardTraceProperties.from_dict(props_dict)
    assert new_props.line_color == props.line_color

    print("✓ StandardTraceProperties tests passed")


def test_property_conversion():
    """Test conversion from standard to Plotly format."""
    print("Testing property conversion...")

    # Create standard properties
    props = StandardTraceProperties(
        line_color="#ff0000",
        line_width=3.0,
        line_style=LineStyle.DASHED,
        marker_color="#00ff00",
        marker_size=8.0,
        marker_symbol=MarkerSymbol.SQUARE,
        mode=TraceMode.LINES_AND_MARKERS,
    )

    # Convert to Plotly format
    plotly_props = convert_standard_to_plotly(props)

    # Verify conversion
    assert plotly_props["mode"] == "lines+markers"
    assert plotly_props["line"]["color"] == "#ff0000"
    assert plotly_props["line"]["width"] == 3.0
    assert plotly_props["line"]["dash"] == "dash"
    assert plotly_props["marker"]["color"] == "#00ff00"
    assert plotly_props["marker"]["size"] == 8.0
    assert plotly_props["marker"]["symbol"] == "square"

    print("✓ Property conversion tests passed")


def test_standard_trace_constructors():
    """Test standard trace constructor functions."""
    print("Testing standard trace constructors...")

    # Test data
    x_data = np.array([0, 1, 2, 3, 4])
    y_data = np.array([0, 2, 1, 3, 2])
    z_data = np.array([0, 1, 0, 2, 1])

    # Test 2D line trace
    line_trace = create_line_trace(
        name="test_line",
        x_data=x_data,
        y_data=y_data,
        color="#ff0000",
        width=2.5
    )

    assert line_trace.name == "test_line"
    assert line_trace.static == True
    assert isinstance(line_trace.properties, StandardTraceProperties)
    assert line_trace.properties.line_color == "#ff0000"
    assert line_trace.properties.line_width == 2.5

    # Test 3D line trace
    line_trace_3d = create_line_trace(
        name="test_line_3d",
        x_data=x_data,
        y_data=y_data,
        z_data=z_data
    )

    assert line_trace_3d.data.shape == (5, 3)

    # Test scatter trace
    scatter_trace = create_scatter_trace(
        name="test_scatter",
        x_data=x_data,
        y_data=y_data,
        color="#00ff00",
        size=8.0
    )

    assert scatter_trace.name == "test_scatter"
    assert scatter_trace.properties.marker_color == "#00ff00"
    assert scatter_trace.properties.marker_size == 8.0

    # Test robot link trace
    robot_trace = create_robot_link_trace(
        name="robot.link_0",
        start_point=[0, 0, 0],
        end_point=[1, 1, 1],
        color="#ff0000",
        width=4.0
    )

    assert robot_trace.name == "robot.link_0"
    assert robot_trace.data.shape == (2, 3)

    print("✓ Standard trace constructor tests passed")


def test_trace_builder_integration():
    """Test TraceBuilder with standard properties."""
    print("Testing TraceBuilder integration...")

    # Create a trace constructor with standard properties
    trace_constructor = create_line_trace(
        name="test_integration",
        x_data=np.array([0, 1, 2, 3]),
        y_data=np.array([0, 2, 1, 3]),
        color="#0000ff",
        width=2.0
    )

    # Create TraceBuilder and build trace
    builder = TraceBuilder()
    plotly_trace = builder.build_trace(trace_constructor, use_precomputed_themes=False)

    # Verify the result is a proper Plotly trace
    assert hasattr(plotly_trace, 'x')
    assert hasattr(plotly_trace, 'y')
    assert plotly_trace.name == "test_integration"

    # Verify properties were converted correctly
    assert hasattr(plotly_trace, 'line')
    assert plotly_trace.line.color is not None  # Should have some color from theme
    assert plotly_trace.line.width == 2  # Should have the width we set

    print("✓ TraceBuilder integration tests passed")


def test_animated_traces():
    """Test animated trace functionality."""
    print("Testing animated traces...")

    # Create animated data: (n_points, n_frames, space)
    n_points = 3
    n_frames = 5
    animated_data = np.random.rand(n_points, n_frames, 2)
    time_data = np.linspace(0, 1, n_frames)

    # Create animated trace
    anim_trace = create_animated_trace(
        name="test_animation",
        data=animated_data,
        time=time_data,
        properties=StandardTraceProperties(line_color="#ff00ff")
    )

    assert anim_trace.name == "test_animation"
    assert anim_trace.static == False
    assert anim_trace.is_animated == True
    assert anim_trace.time is not None
    assert len(anim_trace.time) == n_frames

    print("✓ Animated trace tests passed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Standardized Trace Constructor System")
    print("=" * 60)

    try:
        test_standard_properties()
        test_property_conversion()
        test_standard_trace_constructors()
        test_trace_builder_integration()
        test_animated_traces()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("The standardized system is working correctly.")
        print("=" * 60)

    except Exception as e:
        print("=" * 60)
        print("❌ TEST FAILED:")
        print(f"Error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())