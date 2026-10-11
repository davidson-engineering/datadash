"""Trace constructors with faces are drawn as meshes, and animate like lines."""

import numpy as np
import plotly.graph_objects as go
import pytest

from datadash.builders.figure import PlotFigure3DAnimation
from datadash.builders.frames import FramesConstructor
from datadash.builders.trace import TraceBuilder, create_trace_constructor
from datadash.themes.manager import ThemeManager

# A tetrahedron: four vertices, four triangles
_VERTICES = np.array(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=float)
_FACES = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))


def _moving(frames=5):
    """The tetrahedron sliding along x, vertices of shape (4, frames, 3)."""
    shift = np.linspace(0, 1, frames)[:, np.newaxis] * (1, 0, 0)
    return _VERTICES[:, np.newaxis, :] + shift[np.newaxis]


def _builder():
    return TraceBuilder(theme_manager=ThemeManager("default"))


def test_a_static_mesh_is_drawn_as_a_mesh3d_of_its_vertices_and_faces():
    constructor = create_trace_constructor(
        name="part", data=_VERTICES, faces=_FACES, static=True, color="#123456"
    )

    trace = _builder().build_trace(constructor)

    assert isinstance(trace, go.Mesh3d)
    np.testing.assert_array_equal(np.column_stack((trace.x, trace.y, trace.z)), _VERTICES)
    np.testing.assert_array_equal(np.column_stack((trace.i, trace.j, trace.k)), _FACES)
    assert trace.color == "#123456"


def test_a_mesh_takes_the_mesh_defaults_not_the_line_defaults():
    """The default trace's markers, lines and mode are not properties of a Mesh3d."""
    constructor = create_trace_constructor(name="part", data=_VERTICES, faces=_FACES, static=True)

    trace = _builder().build_trace(constructor)

    mesh_defaults = ThemeManager("default").get_plotly_theme()["traces"]["mesh"]
    assert trace.lighting.ambient == mesh_defaults["lighting"]["ambient"]
    assert trace.flatshading is False


def test_a_mesh_s_theme_entry_is_merged_over_the_mesh_defaults():
    theme = ThemeManager("default").merge_custom_config(
        {"plotly": {"traces": {"part": {"color": "#abcdef"}}}}
    )
    constructor = create_trace_constructor(
        name="part.left", data=_VERTICES, faces=_FACES, static=True
    )

    trace = TraceBuilder(theme_manager=theme).build_trace(constructor)

    assert trace.color == "#abcdef"
    assert trace.lighting.ambient is not None


def test_a_closed_mesh_is_not_closed_like_a_line():
    constructor = create_trace_constructor(
        name="part", data=_VERTICES, faces=_FACES, static=True, closed=True
    )

    trace = _builder().build_trace(constructor)

    assert len(trace.x) == len(_VERTICES)


@pytest.mark.parametrize(
    "faces",
    [
        np.array(((0, 1, 4),)),  # no fifth vertex
        np.array(((0, 1, -1),)),
        np.array(((0.0, 1.0, 2.0),)),  # not indices
        np.array((0, 1, 2)),  # not a list of triangles
    ],
)
def test_faces_must_be_triangles_of_the_mesh_s_vertices(faces):
    with pytest.raises(ValueError, match="Mesh 'part'"):
        create_trace_constructor(name="part", data=_VERTICES, faces=faces, static=True)


def test_an_animated_mesh_moves_its_vertices_and_keeps_its_faces():
    frames = 5
    constructor = create_trace_constructor(
        name="part", data=_moving(frames), time=np.linspace(0, 1, frames), faces=_FACES
    )

    built = FramesConstructor({"part": constructor}, number_frames=frames).build_frames()

    first, last = built[0].data[0], built[-1].data[0]
    assert isinstance(first, go.Mesh3d)
    np.testing.assert_array_equal(np.column_stack((first.i, first.j, first.k)), _FACES)
    np.testing.assert_allclose(
        np.column_stack((last.x, last.y, last.z)), _VERTICES + (1, 0, 0), rtol=1e-6
    )


def test_after_the_first_frame_a_mesh_s_frames_carry_only_its_vertices():
    """Plotly merges them into the trace, which keeps its faces and styling."""
    frames = 4
    constructor = create_trace_constructor(
        name="part", data=_moving(frames), time=np.linspace(0, 1, frames), faces=_FACES
    )

    built = FramesConstructor({"part": constructor}, number_frames=frames).build_frames()

    for frame in built[1:]:
        mesh = frame.data[0].to_plotly_json()
        assert set(mesh) == {"type", "x", "y", "z"}


def test_an_animated_mesh_s_vertices_are_sent_in_single_precision():
    """As WebGL draws them: a mesh's vertices are most of a figure's frames."""
    frames = 3
    constructor = create_trace_constructor(
        name="part", data=_moving(frames), time=np.linspace(0, 1, frames), faces=_FACES
    )

    built = FramesConstructor({"part": constructor}, number_frames=frames).build_frames()

    for frame in built:
        assert np.asarray(frame.data[0].x).dtype == np.float32


def test_an_animated_figure_mixes_meshes_and_lines_and_spans_both():
    frames = 4
    time = np.linspace(0, 1, frames)
    path = np.stack((np.zeros((frames, 3)), np.full((frames, 3), -2.0)))  # a segment, (2, n, 3)
    constructors = {
        "part": create_trace_constructor(
            name="part", data=_moving(frames), time=time, faces=_FACES
        ),
        "line": create_trace_constructor(name="line", data=path, time=time),
    }

    figure = PlotFigure3DAnimation(
        trace_constructor=constructors, layout={}, fps=frames, plot_duration=1.0
    ).figure

    assert [type(trace) for trace in figure.data] == [go.Mesh3d, go.Scatter3d]
    # The scene spans the line's far end and the mesh's furthest vertex
    x_range = figure.layout.scene.xaxis.range
    assert min(x_range) <= -2 and max(x_range) >= 2


def test_a_builder_takes_the_theme_set_last(monkeypatch):
    """set_theme() replaces the global theme; builders made after it use the new one."""
    from datadash.themes import manager

    monkeypatch.setattr(manager, "_theme_manager", None)
    constructor = create_trace_constructor(name="part", data=_VERTICES, faces=_FACES, static=True)
    TraceBuilder().build_trace(constructor)

    manager.set_theme("default").merge_custom_config(
        {"plotly": {"traces": {"part": {"color": "#abcdef"}}}}
    )

    assert TraceBuilder().build_trace(constructor).color == "#abcdef"
