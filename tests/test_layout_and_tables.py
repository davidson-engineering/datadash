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


SECTIONED = pd.DataFrame(
    {
        "Category": ["Kinematics", "Kinematics", "Dynamics"],
        "Metric": ["Leg 1 proximal angular acceleration X", "Velocity", "Torque"],
        "Unit": ["rad·s⁻²", "rad·s⁻¹", "N·m"],
        "Mean": [1.5, 2.0, "n/a"],
    }
)


def sections(grid):
    """Each section of a sectioned table: its caption, header cells and body rows."""
    out = []
    for table in grid.children:
        caption, pane = table.children
        header, *body = pane.children
        out.append((caption.children, header.children, [row.children for row in body]))
    return out


def test_sections_follow_the_order_their_values_first_appear():
    grid = construct_dash_table(SECTIONED, table_id="results", group_by="Category")

    assert grid.id == "results"
    assert [(caption, len(body)) for caption, _, body in sections(grid)] == [
        ("Kinematics", 2),
        ("Dynamics", 1),
    ]
    for _, header, body in sections(grid):
        # Each section's own header row, without the column it is grouped by
        assert [th.children for th in header] == ["Metric", "Unit", "Mean"]
        assert all(len(row) == 3 for row in body)
    assert [td.children for td in sections(grid)[1][2][0]] == ["Torque", "N·m", "n/a"]


def test_sections_share_one_grid_of_columns():
    """Each section's pane is a subgrid of the one grid, so the columns of every
    section are as wide as the widest entry of any."""
    grid = construct_dash_table(SECTIONED, group_by="Category")

    assert grid.style["display"] == "grid"
    assert grid.style["gridTemplateColumns"] == "repeat(3, auto)"
    for table in grid.children:
        caption, pane = table.children
        assert table.style == {"display": "contents"}
        assert caption.style["gridColumn"] == pane.style["gridColumn"] == "1 / -1"
        assert pane.style["display"] == "grid"
        assert pane.style["gridTemplateColumns"] == "subgrid"
        for row in pane.children:
            assert row.style == {"display": "contents"}


def test_sections_align_each_column_by_all_of_its_values():
    """Mean is mostly numeric over the whole table, though not in Dynamics."""
    grid = construct_dash_table(SECTIONED, group_by="Category")

    for _, header, body in sections(grid):
        for row in [header, *body]:
            assert [cell.style["textAlign"] for cell in row] == ["left", "left", "right"]


def test_each_section_scrolls_down_on_its_own_and_all_scroll_sideways_together():
    grid = construct_dash_table(SECTIONED, max_width="100%", group_by="Category")

    assert grid.style["maxWidth"] == "100%" and grid.style["overflowX"] == "auto"
    assert "maxHeight" not in grid.style and "overflowY" not in grid.style
    for table in grid.children:
        caption, pane = table.children
        header, *_ = pane.children
        assert pane.style["overflowY"] == "auto" and pane.style["maxHeight"] == "70vh"
        assert "maxWidth" not in pane.style
        assert header.children[0].style["position"] == "sticky"
        # The caption stays in view as the sections scroll sideways
        assert caption.style["position"] == "sticky" and caption.style["left"] == 0


def test_sections_say_what_part_of_a_table_each_element_is():
    """The grid lays the cells out, so their elements no longer show it."""
    grid = construct_dash_table(SECTIONED, group_by="Category")

    for table in grid.children:
        _, pane = table.children
        header, *body = pane.children
        assert (table.role, pane.role, header.role) == ("table", "rowgroup", "row")
        assert {th.role for th in header.children} == {"columnheader"}
        assert {td.role for row in body for td in row.children} == {"cell"}
