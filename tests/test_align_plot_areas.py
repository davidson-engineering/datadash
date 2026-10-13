"""Charts stacked in one column of a dashboard have the same plot area, and a
phone's have plot areas as wide as their labels allow
(assets/align-plot-areas.js), in a browser.

Each chart's own margins fit its tick labels and axis titles, so a chart with
wider labels has a narrower plot area; the page gives every chart in a column
the widest margins any of them needs. The theme's labels and margins are sized
for a desktop: a phone's chart, about 300px wide, kept little more than a
third of its width for its plot area, so the page draws a narrow chart's axes
smaller, over margins no wider than its labels need.
"""

import os
import threading

import numpy as np
import pytest
from dash import Input, Output, html
from werkzeug.serving import make_server

from datadash.builders.content import create_two_column_builder
from datadash.builders.plot import CombinedPlotBuilder
from datadash.dashboard.base import DashboardApp, assets_path
from datadash.dashboard.components import create_graph_component, create_main_container
from datadash.themes.manager import get_theme_manager

playwright = pytest.importorskip("playwright.sync_api")

# Time series whose tick labels differ in width on both y-axes: from 0 to 8 on
# the left and right ("4") or from -0.25 to 0.25 on the left and -0.025 to 0.025
# on the right ("-0.2", "-0.02")
SCALES = {"narrow": (4.0, 4.0, 1.0), "wide": (0.0, 0.25, 0.1)}

_THEME = get_theme_manager().get_plotly_theme()["layout"]
# The size of a tick label, the theme's font's, and a chart's base side margins
FONT = _THEME["font"]["size"]
MARGINS = {"l": _THEME["margin"]["l"], "r": _THEME["margin"]["r"]}


def _chart(name):
    offset, amplitude, secondary_scale = SCALES[name]
    x = np.linspace(0, 1, 50)
    y = offset + amplitude * np.column_stack([np.sin(6 * x), np.cos(6 * x)])
    return CombinedPlotBuilder().create_plot(
        x,
        y,
        title=name,
        y_title="Force",
        y_units="N",
        headers=["a", "b"],
        secondary_axis={"units": "kN", "scale": secondary_scale},
    )


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        # Two columns from 992px up, the narrow chart over the wide one in the
        # first and under it in the second; one column of four below that
        grid = [["narrow", "wide"], ["wide", "narrow"]]
        self.app.layout = html.Div(
            [
                html.Button("Replace", id="replace"),
                # In the page's gutters, as a dashboard's charts are
                create_main_container(
                    create_two_column_builder().create_layout(
                        [
                            [
                                create_graph_component(f"{name}-{i}-{j}", _chart(name))
                                for j, name in enumerate(row)
                            ]
                            for i, row in enumerate(grid)
                        ]
                    )
                ),
            ]
        )

        # A new figure for a chart, as a callback sends one
        @self.app.callback(
            Output("narrow-0-0", "figure"),
            Input("replace", "n_clicks"),
            prevent_initial_call=True,
        )
        def replace(_clicks):
            return _chart("narrow")


@pytest.fixture(scope="module")
def url():
    dashboard = _Dashboard()
    dashboard.initialize()
    server = make_server("127.0.0.1", 0, dashboard.app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


@pytest.fixture(scope="module")
def browser():
    with playwright.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except playwright.Error as error:
            # CI installs Chromium; elsewhere it takes `uv run playwright install chromium`
            if os.environ.get("CI"):
                raise
            pytest.skip(f"no Chromium for Playwright: {error}")
        yield browser
        browser.close()


# Each chart's plot area and column, in page pixels, the margins Plotly drew it
# with and those its own figure gave it, how far its y-axes' labels and titles
# stay inside it, its width, the font sizes of its axes' tick labels and
# titles, and its base side margins
_CHARTS = """() => [...document.querySelectorAll('.js-plotly-plot')].map(gd => {
  const box = gd.getBoundingClientRect(), full = gd._fullLayout, xaxis = full.xaxis;
  const labels = [...gd.querySelectorAll(
    '.ytick text, .y2tick text, .g-ytitle text, .g-y2title text'
  )].map(text => text.getBoundingClientRect());
  const axes = [full.xaxis, full.yaxis, full.yaxis2];
  return {
    id: gd.parentElement.id,
    column: Math.round(box.left),
    left: box.left + xaxis._offset,
    right: box.left + xaxis._offset + xaxis._length,
    need: gd._alignNeed && {l: gd._alignNeed.l, r: gd._alignNeed.r},
    inside: Math.min(...labels.map(
      label => Math.min(label.left - box.left, box.right - label.right)
    )),
    width: full.width,
    ticks: axes.map(axis => axis.tickfont.size),
    titles: axes.map(axis => axis.title.font.size),
    base: {l: full.margin.l, r: full.margin.r},
  };
})"""


# Whether every chart has drawn at its div's width and the page has measured it
# there: a resized window resizes the charts a moment later
_SETTLED = """() => {
  const gds = [...document.querySelectorAll('.js-plotly-plot')];
  return gds.length === 4 && gds.every(gd => gd._alignNeed && !gd._alignBusy
    && gd._alignNeed.width === gd._fullLayout.width
    && Math.abs(gd._fullLayout.width - gd.getBoundingClientRect().width) < 1);
}"""


def _settled_charts(page):
    """The charts once each has drawn and the page has aligned them."""
    page.wait_for_function(_SETTLED, timeout=10000)
    page.wait_for_timeout(200)  # for the margins the page gives them
    page.wait_for_function(_SETTLED, timeout=10000)
    return page.evaluate(_CHARTS)


def _charts(browser, url, width, script=True):
    """The charts on a page whose window is ``width`` px wide, by id, as the
    page draws them or, without the script, as their figures say."""
    page = browser.new_page(viewport={"width": width, "height": 1000})
    if not script:
        page.route("**/assets/align-plot-areas.js*", lambda route: route.abort())
    page.goto(url)
    if script:
        charts = _settled_charts(page)
    else:
        page.wait_for_function(
            "() => [...document.querySelectorAll('.js-plotly-plot')].filter("
            "gd => gd._fullLayout && gd._fullLayout._size).length === 4",
            timeout=10000,
        )
        page.wait_for_timeout(500)
        charts = page.evaluate(_CHARTS)
    page.close()
    return {chart["id"]: chart for chart in charts}


def _columns(charts):
    columns = {}
    for chart in charts:
        columns.setdefault(chart["column"], []).append(chart)
    return list(columns.values())


def _assert_aligned(charts, columns):
    assert len(_columns(charts)) == columns
    for column in _columns(charts):
        assert len({round(chart["left"], 1) for chart in column}) == 1, column
        assert len({round(chart["right"], 1) for chart in column}) == 1, column
        # The wide chart needs wider margins on both sides: lining them up took
        # the page, not the figures. (A chart's need can grow past what it
        # measured when the margins it is given change its tick labels, so a
        # column's needs can take more than two values.)
        for side in ("l", "r"):
            need = {
                kind: [chart["need"][side] for chart in column if chart["id"].startswith(kind)]
                for kind in ("narrow", "wide")
            }
            assert max(need["narrow"]) < min(need["wide"]), (side, column)
    # Wider margins than a chart's own never clip its labels
    assert min(chart["inside"] for chart in charts) >= 0


def test_the_aligning_script_ships_with_the_package():
    assert (assets_path / "align-plot-areas.js").is_file()


def _assert_drawn_as_their_figures_say(charts):
    for chart in charts:
        assert chart["ticks"] == [FONT] * 3, chart
        assert chart["base"]["l"] >= MARGINS["l"] and chart["base"]["r"] >= MARGINS["r"], chart


def _assert_compact(charts):
    for chart in charts:
        assert all(10 <= size < FONT for size in chart["ticks"]), chart
        assert chart["base"]["l"] < MARGINS["l"] and chart["base"]["r"] < MARGINS["r"], chart


# Two columns, one, and a phone's
@pytest.mark.parametrize(("width", "columns"), [(1600, 2), (800, 1), (375, 1)])
def test_charts_stacked_in_a_column_share_a_plot_area(browser, url, width, columns):
    page = browser.new_page(viewport={"width": width, "height": 1000})
    page.goto(url)

    _assert_aligned(_settled_charts(page), columns)
    page.close()


@pytest.mark.parametrize("width", [1600, 800])
def test_a_desktop_s_charts_are_drawn_as_their_figures_say(browser, url, width):
    _assert_drawn_as_their_figures_say(_charts(browser, url, width).values())


# The narrower phones, a common Android phone, and the widest
@pytest.mark.parametrize("width", [360, 375, 430])
def test_a_phone_s_charts_draw_their_axes_smaller_for_wider_plot_areas(browser, url, width):
    """Lined up at their figures' own style, the column's plot areas would be
    as wide as the narrowest of its charts' own: 181px of 351 at 375px (184px
    in CI's Chromium on Linux). Drawn compact, they are 227px (223px)."""
    charts = _charts(browser, url, width)
    own = _charts(browser, url, width, script=False)
    _assert_compact(charts.values())
    lined_up = min(chart["right"] - chart["left"] for chart in own.values())
    for name, chart in charts.items():
        assert all(a < b for a, b in zip(chart["titles"], own[name]["titles"], strict=True)), name
        plot = chart["right"] - chart["left"]
        assert plot >= lined_up + 30 and plot > 0.55 * chart["width"], (name, plot, lined_up)


def test_charts_line_up_again_when_the_window_resizes(browser, url):
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(url)
    _settled_charts(page)

    page.set_viewport_size({"width": 800, "height": 1000})
    _assert_aligned(_settled_charts(page), 1)
    page.set_viewport_size({"width": 375, "height": 1000})
    charts = _settled_charts(page)
    _assert_aligned(charts, 1)
    _assert_compact(charts)
    page.set_viewport_size({"width": 1200, "height": 1000})
    charts = _settled_charts(page)
    _assert_aligned(charts, 2)
    _assert_drawn_as_their_figures_say(charts)
    page.close()


def test_a_phone_s_charts_stay_compact_when_dash_draws_them_again(browser, url):
    """Dash draws a chart again after each relayout, from the figure it holds
    updated with the chart's layout, and the figure it holds can lag behind a
    relayout or two. A chart zoomed, or sent a new figure, is drawn as compact
    as before, in line with the others."""
    page = browser.new_page(viewport={"width": 375, "height": 1000})
    page.goto(url)
    before = {chart["id"]: chart for chart in _settled_charts(page)}

    page.evaluate(
        "() => window.Plotly.relayout("
        "document.querySelector('#wide-0-1 .js-plotly-plot'), {'xaxis.range': [0.2, 0.8]})"
    )
    page.wait_for_timeout(300)
    zoomed = _settled_charts(page)
    page.click("#replace")
    page.wait_for_timeout(300)
    replaced = _settled_charts(page)
    page.close()

    for charts in (zoomed, replaced):
        _assert_aligned(charts, 1)
        for chart in charts:
            assert chart["ticks"] == before[chart["id"]]["ticks"], chart
            assert chart["titles"] == before[chart["id"]]["titles"], chart
