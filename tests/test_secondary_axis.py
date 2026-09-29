"""Secondary y-axis on single-axis plots."""

import numpy as np
from datadash.builders.plot import CombinedPlotBuilder


def build(secondary_axis):
    x = np.linspace(0, 1, 50)
    y = np.column_stack([np.sin(x), np.cos(x)])
    fig = CombinedPlotBuilder().create_plot(
        x, y, title="Torque", headers=["a", "b"], secondary_axis=secondary_axis
    )
    return fig, y


def test_secondary_axis_adds_hidden_scaled_copies_on_y2():
    fig, y = build({"units": "rpm", "scale": 10.0, "title": "Speed"})

    primary, secondary = fig.data[:2], fig.data[2:]
    assert [t.yaxis for t in primary] == ["y", "y"]
    assert [t.name for t in secondary] == ["a_rpm", "b_rpm"]
    for i, trace in enumerate(secondary):
        assert trace.yaxis == "y2" and trace.xaxis == "x"
        assert trace.visible is False and trace.showlegend is False
        np.testing.assert_allclose(trace.y, y[:, i] * 10.0)


def test_secondary_axis_layout_matches_a_secondary_y_subplot():
    fig, y = build({"units": "rpm", "scale": 10.0, "title": "Speed"})
    layout = fig.layout

    assert layout.yaxis2.overlaying == "y" and layout.yaxis2.side == "right"
    assert layout.yaxis2.title.text == "Speed [rpm]"
    assert layout.yaxis2.showgrid is False
    assert tuple(layout.xaxis.domain) == (0.0, 0.94)
    low, high = layout.yaxis2.range
    assert low <= (y * 10.0).min() and high >= (y * 10.0).max()
    # The theme's axis styling survives on the primary axes
    assert layout.yaxis.gridcolor is not None


def test_secondary_axis_title_defaults_to_units():
    fig, _ = build({"units": "deg/s", "scale": 2.0})
    assert fig.layout.yaxis2.title.text == "[deg/s]"


def test_plot_without_secondary_axis_has_one_trace_per_column():
    fig, _ = build(None)
    assert len(fig.data) == 2
    assert fig.layout.yaxis2.overlaying is None
