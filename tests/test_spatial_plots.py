"""Equal-aspect spatial plots stay readable for any path."""

import numpy as np
from datadash.builders.plot import Spatial2DPlotBuilder

# A planar arc in XZ: seen from above (XY) or the side (YZ) it has no Y extent
_T = np.linspace(0, 1, 50)
_X = 0.08 * np.cos(np.pi * _T)
_Y = np.zeros_like(_T)
_Z = 0.2 + 0.08 * np.sin(np.pi * _T)


def _ranges(x, y, y_title="Y"):
    layout = Spatial2DPlotBuilder().create_spatial_layout_overrides(x, y, "X", y_title)
    return layout["xaxis"]["range"], layout["yaxis"]["range"]


def _span(axis_range):
    return abs(axis_range[1] - axis_range[0])


def test_flat_axis_gets_the_other_axis_span():
    xrange, yrange = _ranges(_X, _Y)
    assert _span(yrange) == _span(xrange)
    assert min(yrange) < 0 < max(yrange)  # centred on the data


def test_z_axis_is_reversed_and_keeps_its_range():
    _, zrange = _ranges(_X, _Z, y_title="Z")
    assert zrange[0] > zrange[1]
    assert min(zrange) < _Z.min() and max(zrange) > _Z.max()


def test_non_flat_axes_are_left_alone():
    xrange, yrange = _ranges(_X, _Z)
    assert _span(xrange) > _span(yrange)
