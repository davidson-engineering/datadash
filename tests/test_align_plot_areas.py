"""Charts stacked in one column of a dashboard have the same plot area
(assets/align-plot-areas.js), in a browser.

Each chart's own margins fit its tick labels and axis titles, so a chart with
wider labels has a narrower plot area; the page gives every chart in a column
the widest margins any of them needs.
"""

import os
import threading

import numpy as np
import pytest
from dash import html
from werkzeug.serving import make_server

from datadash.builders.content import create_two_column_builder
from datadash.builders.plot import CombinedPlotBuilder
from datadash.dashboard.base import DashboardApp, assets_path
from datadash.dashboard.components import create_graph_component

playwright = pytest.importorskip("playwright.sync_api")

# Time series whose tick labels differ in width on both y-axes: from 0 to 8 on
# the left and right ("4") or from -0.25 to 0.25 on the left and -0.025 to 0.025
# on the right ("-0.2", "-0.02")
SCALES = {"narrow": (4.0, 4.0, 1.0), "wide": (0.0, 0.25, 0.1)}


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
            create_two_column_builder().create_layout(
                [
                    [
                        create_graph_component(f"{name}-{i}-{j}", _chart(name))
                        for j, name in enumerate(row)
                    ]
                    for i, row in enumerate(grid)
                ]
            )
        )


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
# with and those its own figure gave it, and how far its y-axes' labels and
# titles stay inside it
_CHARTS = """() => [...document.querySelectorAll('.js-plotly-plot')].map(gd => {
  const box = gd.getBoundingClientRect(), xaxis = gd._fullLayout.xaxis;
  const labels = [...gd.querySelectorAll(
    '.ytick text, .y2tick text, .g-ytitle text, .g-y2title text'
  )].map(text => text.getBoundingClientRect());
  return {
    id: gd.parentElement.id,
    column: Math.round(box.left),
    left: box.left + xaxis._offset,
    right: box.left + xaxis._offset + xaxis._length,
    need: gd._alignNeed && {l: gd._alignNeed.l, r: gd._alignNeed.r},
    inside: Math.min(...labels.map(
      label => Math.min(label.left - box.left, box.right - label.right)
    )),
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
    return page.evaluate(_CHARTS)


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
        # The wide chart's own margins are wider on both sides: lining them up
        # took the page, not the figures
        assert len({chart["need"]["l"] for chart in column}) == 2, column
        assert len({chart["need"]["r"] for chart in column}) == 2, column
    # Wider margins than a chart's own never clip its labels
    assert min(chart["inside"] for chart in charts) >= 0


def test_the_aligning_script_ships_with_the_package():
    assert (assets_path / "align-plot-areas.js").is_file()


@pytest.mark.parametrize(("width", "columns"), [(1600, 2), (800, 1)])
def test_charts_stacked_in_a_column_share_a_plot_area(browser, url, width, columns):
    page = browser.new_page(viewport={"width": width, "height": 1000})
    page.goto(url)

    _assert_aligned(_settled_charts(page), columns)
    page.close()


def test_charts_line_up_again_when_the_window_resizes(browser, url):
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(url)
    _settled_charts(page)

    page.set_viewport_size({"width": 800, "height": 1000})
    _assert_aligned(_settled_charts(page), 1)
    page.set_viewport_size({"width": 1200, "height": 1000})
    _assert_aligned(_settled_charts(page), 2)
    page.close()
