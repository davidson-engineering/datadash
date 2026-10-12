"""The tab bar: equal tabs, in rows sized from how many tabs there are."""

import re

import pytest
from dash import dcc, html

from datadash.dashboard.base import DashboardApp, assets_path
from datadash.dashboard.components import (
    _FEWEST_TABS_PER_ROW,
    _MOST_TABS_PER_ROW,
    _tabs_per_row,
    create_tabs,
    create_themed_tab,
)
from datadash.themes.manager import set_theme

FITS = range(_FEWEST_TABS_PER_ROW, _MOST_TABS_PER_ROW + 1)

# Inline styles beat the stylesheet, so a theme setting any of these would undo
# the bar's layout. (A theme's padding or margin on the bar only moves where
# its rows break a little.)
TAB_LAYOUT = {"flex", "flexBasis", "flexGrow", "flexShrink", "width", "minWidth", "maxWidth"}
BAR_LAYOUT = {"display", "flexWrap", "justifyContent", "gap", "overflowX"}


def rows(count, fit):
    """How many tabs each row of the bar holds, filled in order."""
    per_row = _tabs_per_row(count, fit)
    return [min(per_row, count - start) for start in range(0, count, per_row)]


def tabs(count):
    return [create_themed_tab(f"TAB {i}", f"tab-{i}") for i in range(count)]


def bar_props(*args, **kwargs):
    """The props of a tab bar from create_tabs."""
    return create_tabs(*args, **kwargs).to_plotly_json()["props"]


@pytest.fixture(autouse=True)
def default_theme():
    set_theme("default")


@pytest.fixture(scope="module")
def stylesheet():
    css = (assets_path / "dash-components.css").read_text()
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


@pytest.mark.parametrize(
    "fit, expected",
    [
        (3, [3, 3, 3]),
        (4, [3, 3, 3]),
        (5, [5, 4]),
        (8, [5, 4]),
        (9, [9]),
        (12, [9]),
    ],
)
def test_nine_tabs_make_even_rows(fit, expected):
    assert rows(9, fit) == expected


@pytest.mark.parametrize("count", range(1, 25))
@pytest.mark.parametrize("fit", FITS)
def test_tabs_take_as_few_rows_as_fit(count, fit):
    layout = rows(count, fit)

    assert sum(layout) == count
    assert max(layout) <= fit
    assert len(layout) == -(-count // fit)
    # Rows fill in order, the first as full as any
    assert layout == sorted(layout, reverse=True)


def test_a_bar_says_how_many_tabs_a_row_holds_for_each_fit():
    props = bar_props(tabs(9))

    assert {k: v for k, v in props["style"].items() if k.startswith("--")} == {
        f"--tabs-per-row-{fit}": str(_tabs_per_row(9, fit)) for fit in FITS
    }


def test_a_bar_is_laid_out_by_the_stylesheet():
    props = bar_props(tabs(2))

    assert props["className"] == "datadash-tabs"
    assert props["parent_className"] == "datadash-tabs-parent"
    assert not BAR_LAYOUT & props["style"].keys()
    assert props["style"]["--tabs-per-row-3"] == "2"


@pytest.mark.parametrize("count, scrolls", [(1, False), (2, False), (3, True), (9, True)])
def test_a_bar_of_three_tabs_or_more_scrolls_where_three_do_not_fit(count, scrolls):
    classes = bar_props(tabs(count))["className"].split()

    assert ("datadash-tabs--scrolls" in classes) == scrolls


def test_a_bar_keeps_the_theme_and_the_callers_style():
    theme = set_theme("default").get_component_style("tab_container")

    props = bar_props(tabs(3), style={"background": "red"})

    assert props["style"]["background"] == "red"
    assert props["style"]["border-bottom"] == theme["border-bottom"]


@pytest.mark.parametrize("theme", ["default", "futuristic"])
def test_themes_leave_the_layout_to_the_stylesheet(theme):
    manager = set_theme(theme)

    for name in ("tab", "tab_selected"):
        assert not TAB_LAYOUT & manager.get_component_style(name).keys()
    assert not BAR_LAYOUT & manager.get_component_style("tab_container").keys()


def declaration(stylesheet, selector, prop):
    """A property's value in the first rule with exactly this selector."""
    rule = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", stylesheet)
    assert rule, selector
    value = re.search(rf"(?:^|;|\s){re.escape(prop)}\s*:\s*([^;]+);", rule.group(1))
    assert value, f"{selector} {{ {prop} }}"
    return value.group(1).strip()


def rem(value):
    match = re.fullmatch(r"([\d.]+)rem", value)
    assert match, value
    return float(match.group(1))


def fit_queries(stylesheet):
    """The container queries that set how many tabs a row holds: {fit: min-width in rem}."""
    queries = re.findall(
        r"@container datadash-tabs \(min-width: ([\d.]+)rem\) \{\s*"
        r"[^{]+\{\s*--tabs-per-row: var\(--tabs-per-row-(\d+)\);",
        stylesheet,
    )
    return {int(fit): float(width) for width, fit in queries}


def test_each_fit_starts_where_that_many_tabs_fit(stylesheet):
    bar = ".datadash-tabs-parent > .datadash-tabs"
    tab_width = rem(declaration(stylesheet, f"{bar} > .tab", "flex").split()[-1])
    gap = rem(declaration(stylesheet, bar, "--tabs-gap"))
    padding = rem(declaration(stylesheet, bar, "padding").split()[1])

    queries = fit_queries(stylesheet)

    assert list(queries) == list(FITS)
    for fit, width in queries.items():
        assert width == pytest.approx(fit * tab_width + (fit - 1) * gap + 2 * padding)


def test_tabs_wrap_onto_rows_where_the_fewest_fit(stylesheet):
    wrap = re.search(
        r"@container datadash-tabs \(min-width: ([\d.]+)rem\) \{\s*"
        r"\.datadash-tabs-parent > \.datadash-tabs \{\s*flex-wrap: wrap;",
        stylesheet,
    )

    assert wrap
    assert float(wrap.group(1)) == fit_queries(stylesheet)[_FEWEST_TABS_PER_ROW]


def test_a_scrolling_bar_shows_two_and_a_half_tabs_where_they_fit(stylesheet):
    bar = ".datadash-tabs-parent > .datadash-tabs"
    tab_width = rem(declaration(stylesheet, f"{bar} > .tab", "flex").split()[-1])
    gap = rem(declaration(stylesheet, bar, "--tabs-gap"))
    padding = rem(declaration(stylesheet, bar, "padding").split()[1])
    half = re.search(
        r"@container datadash-tabs \(min-width: ([\d.]+)rem\) \{\s*"
        r"\.datadash-tabs-parent > \.datadash-tabs--scrolls > \.tab \{\s*"
        r"flex-basis: max\(([\d.]+)rem, calc\(\(100% - 2 \* var\(--tabs-gap\)\) / 2\.5\)\);",
        stylesheet,
    )

    assert half
    assert rem(half.group(2) + "rem") == tab_width
    assert float(half.group(1)) == pytest.approx(2.5 * tab_width + 2 * gap + 2 * padding)


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        self.app.layout = html.Div(create_tabs([dcc.Tab(label="A", value="a")]))


def test_every_dashboard_page_keeps_the_selected_tab_in_view():
    assert (assets_path / "show-selected-tab.js").is_file()
    dashboard = _Dashboard()
    dashboard.initialize()

    page = dashboard.app.server.test_client().get("/").get_data(as_text=True)

    assert "/assets/show-selected-tab.js" in page
