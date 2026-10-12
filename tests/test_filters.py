"""Filter dropdowns, and the rows that size them from how many there are."""

import math
import re

import pandas as pd
import pytest

from datadash.dashboard.base import assets_path
from datadash.dashboard.components import (
    create_filter_dropdown,
    create_parameter_filter_dropdowns,
    filter_row_class_name,
)

WIDTH_PROPS = {"width", "xs", "sm", "md", "lg", "xl", "xxl"}


class _Sweep:
    """What the parameter filters read of a sweep analyzer: each column swept."""

    def __init__(self, df):
        self.df = df

    def get_swept_parameters(self):
        return list(self.df.columns)

    def get_parameter_values(self, param):
        return sorted(self.df[param].unique())


def _per_row(count, breakpoint):
    """Filters a row of ``count`` holds at ``breakpoint`` ("" for phones)."""
    prefix = f"row-cols-{breakpoint}-" if breakpoint else "row-cols-"
    (size,) = [
        name.removeprefix(prefix)
        for name in filter_row_class_name(count).split()
        if name.startswith(prefix) and name.removeprefix(prefix).isdigit()
    ]
    return int(size)


@pytest.mark.parametrize("count", range(1, 13))
def test_phones_tablets_and_laptops_hold_one_two_and_three_filters_a_row(count):
    assert [_per_row(count, b) for b in ("", "sm", "lg")] == [1, 2, 3]


@pytest.mark.parametrize(
    ("count", "per_row"),
    # Fewer than four take a third of the row each, as on laptops; seven make
    # rows of four and three rather than six and one
    [(1, 3), (2, 3), (3, 3), (4, 4), (5, 5), (6, 6), (7, 4), (8, 4), (9, 5), (11, 6), (13, 5)],
)
def test_wide_screens_size_filters_from_their_number(count, per_row):
    assert _per_row(count, "xxl") == per_row


@pytest.mark.parametrize("count", range(4, 40))
def test_wide_screens_spread_filters_over_as_few_rows_as_evenly_as_they_can(count):
    per_row = _per_row(count, "xxl")
    rows = math.ceil(count / per_row)
    assert per_row <= 6
    assert rows == math.ceil(count / 6)
    # No narrower than that many rows need, so the last row is as full as can be
    assert per_row == math.ceil(count / rows)


def test_rows_that_are_not_full_are_centred_with_the_usual_gutters():
    classes = filter_row_class_name(7).split()
    assert {"g-3", "justify-content-center"} <= set(classes)


@pytest.mark.parametrize("count", range(1, 13))
def test_every_class_of_a_filter_row_is_in_the_bundled_stylesheet(count):
    css = (assets_path / "bootstrap.min.css").read_text()
    for name in filter_row_class_name(count).split():
        assert re.search(rf"\.{re.escape(name)}(?![\w-])", css), name


def test_a_filter_shows_its_whole_label_as_a_tooltip():
    options = [{"label": "Delta", "value": "delta"}, {"label": "SCARA", "value": "scara"}]
    column = create_filter_dropdown("robot", "Platform joint spacing (deg)", options, "scara")

    label, dropdown = column.children
    assert label.children == label.title == "Platform joint spacing (deg)"
    assert label.style["textOverflow"] == "ellipsis"
    assert (dropdown.id, dropdown.options, dropdown.value) == ("robot", options, "scara")
    assert dropdown.clearable is False


def test_a_filter_is_sized_by_its_row_not_by_itself():
    column = create_filter_dropdown("robot", "Robot", [], None)
    assert not WIDTH_PROPS & column.to_plotly_json()["props"].keys()


def test_parameter_filters_are_sized_from_their_number():
    df = pd.DataFrame(
        {"radius": [150.0, 200.0], "spacing": [10.0, 20.0], "mass": [1.0, 5.0]},
        index=["j0", "j1"],
    )

    container = create_parameter_filter_dropdowns(_Sweep(df), current_job_id="j1")

    row = container.children
    # Three filters on a wide screen take a third of the row each, not a sixth
    assert row.className == filter_row_class_name(3)
    assert _per_row(3, "xxl") == 3
    for column in row.children:
        assert not WIDTH_PROPS & column.to_plotly_json()["props"].keys()
    dropdowns = [column.children[1] for column in row.children]
    assert [d.id["param"] for d in dropdowns] == ["radius", "spacing", "mass"]
    assert [d.value for d in dropdowns] == [200.0, 20.0, 5.0]
    labels = [column.children[0] for column in row.children]
    assert [label.title for label in labels] == ["radius", "spacing", "mass"]
