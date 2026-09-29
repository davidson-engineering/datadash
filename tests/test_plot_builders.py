"""Every plot builder honours headers and labels."""

import numpy as np
import pytest
from datadash.builders.plot import BasicPlotBuilder, CombinedPlotBuilder, SubplotsPlotBuilder

T = np.linspace(0, 1, 20)
Y2 = np.column_stack([np.sin(T), np.cos(T)])
Y3 = np.column_stack([T, T**2, T**3])


def _names(fig):
    return [trace.name for trace in fig.data]


def test_basic_uses_given_headers_and_labels():
    fig = BasicPlotBuilder().create_plot(
        T, Y2, title="T", headers=["a1", "a2"], labels=["Actuator 1", "Actuator 2"]
    )
    assert _names(fig) == ["Actuator 1", "Actuator 2"]


def test_labels_without_headers_follow_trace_order():
    fig = BasicPlotBuilder().create_plot(T, Y2, title="T", labels=["A", "B"])
    assert _names(fig) == ["A", "B"]


@pytest.mark.parametrize(
    "builder, expected",
    [
        (BasicPlotBuilder, ["trace_0", "trace_1", "trace_2"]),
        (CombinedPlotBuilder, ["1", "2", "3"]),
        (SubplotsPlotBuilder, ["x", "y", "z"]),
    ],
)
def test_each_builder_keeps_its_default_headers(builder, expected):
    assert _names(builder().create_plot(T, Y3, title="T")) == expected


def test_subplots_put_each_column_on_its_own_axes():
    fig = SubplotsPlotBuilder().create_plot(T, Y3, title="T")
    assert [(trace.xaxis, trace.yaxis) for trace in fig.data] == [
        ("x", "y"),
        ("x2", "y2"),
        ("x3", "y3"),
    ]
