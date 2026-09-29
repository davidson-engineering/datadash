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


def cells(table):
    """Header and body rows of a constructed table, as lists of cells."""
    head, body = table.children.children
    return head.children.children, [row.children for row in body.children]


def test_tables_get_unique_ids_and_align_by_type():
    table = construct_dash_table(
        pd.DataFrame({"Metric": ["x"], "Mean": [1.0]}), table_id="results-a"
    )
    assert table.id == "results-a"
    header, rows = cells(table)
    assert [th.children for th in header] == ["Metric", "Mean"]
    for row in [header, *rows]:
        assert [cell.style["textAlign"] for cell in row] == ["left", "right"]


def test_header_stays_pinned_while_the_table_scrolls():
    table = construct_dash_table(pd.DataFrame({"Metric": ["x"]}), max_width="760px")
    header, _ = cells(table)
    assert table.style["overflowY"] == "auto" and table.style["maxWidth"] == "760px"
    assert header[0].style["position"] == "sticky" and header[0].style["top"] == 0


def test_cell_style_applies_to_every_cell_over_the_theme():
    table = construct_dash_table(
        pd.DataFrame({"Metric": ["x"]}), cell_style={"whiteSpace": "nowrap"}
    )
    header, rows = cells(table)
    assert header[0].style["whiteSpace"] == rows[0][0].style["whiteSpace"] == "nowrap"


def test_float_noise_is_rounded_for_display():
    table = construct_dash_table(pd.DataFrame({"Value": [0.16499999999999998]}))
    _, rows = cells(table)
    assert rows[0][0].children == 0.165
