"""Theme layout precedence and table formatting."""

import pandas as pd
from datadash.builders.figure import PlotFigure
from datadash.dashboard.components import _column_alignment, construct_dash_table


def test_theme_supplies_defaults_but_never_overrides_the_figure():
    figure = PlotFigure(
        trace_constructor={}, layout={"showlegend": False, "xaxis": {"range": [0, 1]}}
    )
    target = {"showlegend": False, "xaxis": {"range": [0, 1]}}
    figure._fast_layout_merge(
        target,
        {"showlegend": True, "xaxis": {"gridcolor": "#ddd", "range": [5, 6]}, "font": {"size": 14}},
    )

    assert target["showlegend"] is False  # figure's own value kept
    assert target["xaxis"] == {"range": [0, 1], "gridcolor": "#ddd"}  # nested defaults filled in
    assert target["font"] == {"size": 14}  # missing keys added


def test_column_alignment_follows_the_data():
    rows = [{"name": "a", "value": 1.5}, {"name": "b", "value": 2}, {"name": "c", "value": "text"}]
    assert _column_alignment(rows, "name") == "left"
    assert _column_alignment(rows, "value") == "right"  # mostly numeric


def test_tables_get_unique_ids_and_align_by_type():
    table = construct_dash_table(
        pd.DataFrame({"Metric": ["x"], "Mean": [1.0]}), table_id="results-a"
    )
    assert table.id == "results-a"
    aligns = {c["if"]["column_id"]: c["textAlign"] for c in table.style_cell_conditional}
    assert aligns == {"Metric": "left", "Mean": "right"}


def test_float_noise_is_rounded_for_display():
    table = construct_dash_table(pd.DataFrame({"Value": [0.16499999999999998]}))
    assert table.data == [{"Value": 0.165}]
