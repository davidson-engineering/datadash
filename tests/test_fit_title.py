"""A chart's title fits the chart, in a browser, however narrow the chart.

Plotly draws a title on one line, centred on the chart, and clips whatever runs
past the chart's edges: on a phone, whose charts are about 300px wide, a title
of more than about 23 characters at the theme's 20px lost both its ends.
assets/fit-title.js draws a title (and a subtitle) wider than its chart
smaller, keeping the depth of the band it reserves, so the legend and plot area
stay in line with those of the charts beside it.
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
from datadash.dashboard.components import create_graph_component, create_main_container

playwright = pytest.importorskip("playwright.sync_api")

_T = np.linspace(0, 1, 50)

# The room the script leaves between a title and each side of its chart, in px
_GAP = 8


def _time_series(title):
    return CombinedPlotBuilder().create_plot(
        _T,
        np.sin(6 * _T)[:, None],
        title=title,
        y_title="Torque",
        y_units="N·m",
        headers=["0"],
        labels=["Actuator 1"],
    )


def _placed_by_its_figure():
    """A chart that places its title itself, under a top margin of its own, with
    a subtitle: its title reserves no band."""
    figure = _time_series("Shoulder position, std dev (rad)")
    return figure.update_layout(
        title={
            "automargin": False,
            "y": 1,
            "pad": {"t": 12, "b": 0},
            "subtitle": {
                "text": "Proximal length 125 mm · Distal 0.125 kg",
                "font": {"size": 13},
            },
        },
        margin_t=120,
    )


# A title of 33 characters beside one of 15, each over a legend of one row, and
# a chart that places its own title
_CHARTS = {
    "long": _time_series("End Effector Angular Acceleration"),
    "short": _time_series("Actuator Torque"),
    "own": _placed_by_its_figure(),
}


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        graph = lambda name: create_graph_component(name, _CHARTS[name])  # noqa: E731
        # Two columns from 992px up, one below that
        # In the page's gutters, as a dashboard's charts are
        self.app.layout = create_main_container(
            create_two_column_builder().create_layout(
                [[graph("long"), graph("short")], [graph("own"), html.Div()]]
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


# Each chart's title and subtitle, in px from the chart's top left, their font
# sizes, and the tops of its legend and plot area
_BOXES = """() => Object.fromEntries([...document.querySelectorAll('.js-plotly-plot')].map(gd => {
  const chart = gd.getBoundingClientRect();
  const box = node => {
    if (!node) return null;
    const rect = node.getBoundingClientRect();
    return {left: rect.left - chart.left, right: rect.right - chart.left,
            top: rect.top - chart.top, bottom: rect.bottom - chart.top,
            size: parseFloat(node.style.fontSize)};
  };
  return [gd.parentElement.id, {
    width: chart.width,
    title: box(gd.querySelector('.gtitle')),
    subtitle: box(gd.querySelector('.gtitle-subtitle')),
    legend: box(gd.querySelector('.legend .bg')).top,
    plot: gd._fullLayout._size.t,
  }];
}))"""

# Whether every chart has drawn at its div's width, and fitted its title to it
_SETTLED = """() => {
  const gds = [...document.querySelectorAll('.js-plotly-plot')];
  return gds.length === 3 && gds.every(gd => gd._fullLayout && gd._fullLayout._size
    && Math.abs(gd._fullLayout.width - gd.getBoundingClientRect().width) < 1
    && !gd._fittingTitle);
}"""


def _settle(page):
    page.wait_for_function(_SETTLED, timeout=10000)
    page.wait_for_timeout(500)  # for the redraws the scripts make
    page.wait_for_function(_SETTLED, timeout=10000)


def _boxes(browser, url, width, fit=True):
    """Each chart's boxes, on a page whose window is ``width`` px wide."""
    page = browser.new_page(viewport={"width": width, "height": 1000})
    if not fit:
        page.route("**/assets/fit-title.js*", lambda route: route.abort())
    page.goto(url)
    _settle(page)
    boxes = page.evaluate(_BOXES)
    page.close()
    return boxes


def test_the_title_fitting_script_ships_with_the_package():
    assert (assets_path / "fit-title.js").is_file()


def test_without_the_script_a_phone_s_chart_clips_a_long_title(browser, url):
    """If this fails, Plotly fits titles itself, and the script can go."""
    chart = _boxes(browser, url, 375, fit=False)["long"]
    assert chart["title"]["left"] < 0 and chart["title"]["right"] > chart["width"]


# The narrower phones, a common Android phone, a small tablet
@pytest.mark.parametrize("width", [360, 375, 600])
def test_every_title_and_subtitle_fits_its_chart(browser, url, width):
    for name, chart in _boxes(browser, url, width).items():
        for part in ("title", "subtitle"):
            if chart[part]:
                assert chart[part]["left"] >= _GAP - 1, (name, part, chart)
                assert chart[part]["right"] <= chart["width"] - _GAP + 1, (name, part, chart)


def test_only_a_title_too_wide_is_drawn_smaller(browser, url):
    boxes = _boxes(browser, url, 375)
    assert 10 <= boxes["long"]["title"]["size"] < 20
    assert boxes["short"]["title"]["size"] == 20
    # Its subtitle is drawn smaller only as far as it is too wide itself
    own = boxes["own"]
    assert own["title"]["size"] < 20 and own["subtitle"]["size"] < 13


def test_a_title_drawn_smaller_keeps_its_band_and_its_middle(browser, url):
    """The legend and plot area stay where they were, in line with those of
    the charts beside it, and the title in the middle of its band."""
    boxes = _boxes(browser, url, 375)
    long, short = boxes["long"], boxes["short"]
    assert long["plot"] == pytest.approx(short["plot"], abs=0.5)
    assert long["legend"] == pytest.approx(short["legend"], abs=0.5)
    middle = lambda chart: (chart["title"]["top"] + chart["title"]["bottom"]) / 2  # noqa: E731
    assert middle(long) == pytest.approx(middle(short), abs=1)


def test_a_title_that_places_itself_keeps_its_top(browser, url):
    """A title that reserves no band stays where its figure put it."""
    unfitted = _boxes(browser, url, 375, fit=False)["own"]
    fitted = _boxes(browser, url, 375)["own"]
    assert fitted["title"]["top"] == pytest.approx(unfitted["title"]["top"], abs=1)
    assert fitted["plot"] == unfitted["plot"]


def test_on_a_wide_page_no_title_changes(browser, url):
    fitted, unfitted = _boxes(browser, url, 1600), _boxes(browser, url, 1600, fit=False)
    assert fitted == unfitted


def test_a_title_gets_its_own_size_back_when_its_chart_widens(browser, url):
    page = browser.new_page(viewport={"width": 375, "height": 1000})
    page.goto(url)
    _settle(page)
    assert page.evaluate(_BOXES)["long"]["title"]["size"] < 20
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.wait_for_function(
        "() => [...document.querySelectorAll('.js-plotly-plot')].every("
        "gd => gd._fullLayout.width > 600)",
        timeout=10000,
    )
    _settle(page)
    widened = page.evaluate(_BOXES)
    page.close()
    assert widened == _boxes(browser, url, 1600)
