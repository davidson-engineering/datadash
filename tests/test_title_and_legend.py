"""A chart's title, its legend and its plot area each keep clear of the one
above, in a browser, however many rows the legend wraps onto.

The title sits at the top of the figure, the legend under it and the plot area
under that. A legend over the plot area pushes the top margin out by its
height, and so it used to grow upwards from the plot area: on a phone, where a
legend of three entries wraps to three rows, it was drawn over the title. The
title now reserves its own band of the top margin (the theme's automargin), and
the legend is pushed out below it. On a chart's first draw Plotly places the
legend before the band is reserved and leaves it there, over the title, unless
something else redraws the chart; assets/place-legend.js places it again.
"""

import os
import threading

import numpy as np
import pytest
from werkzeug.serving import make_server

from datadash.builders.content import create_two_column_builder
from datadash.builders.plot import CombinedPlotBuilder, EndEffectorSpatialPlots
from datadash.dashboard.base import DashboardApp, assets_path
from datadash.dashboard.components import create_graph_component, create_main_container
from datadash.themes.manager import ThemeManager

playwright = pytest.importorskip("playwright.sync_api")

_T = np.linspace(0, 1, 50)
_PATH = np.column_stack([0.1 * np.cos(np.pi * _T), 0.1 * np.sin(np.pi * _T), 0.2 + 0.05 * _T])


def _time_series(title, entries):
    y = np.column_stack([np.sin(6 * _T + i) for i in range(entries)])
    return CombinedPlotBuilder().create_plot(
        _T,
        y,
        title=title,
        y_title="Torque",
        y_units="N·m",
        headers=[str(i) for i in range(entries)],
        labels=[f"Actuator {i + 1}" for i in range(entries)],
        secondary_axis={"units": "oz·in", "scale": 141.6, "title": "Torque"},
    )


# The chart with a legend of one entry, laid out as the theme laid it out before
# its title reserved a band: the title a fraction of the figure's height from its
# top, over a margin deep enough for a legend of one row
_BEFORE = _time_series("Linear Actuator Force", 1).update_layout(
    title={"y": 0.98, "pad": {"t": 0, "b": 0}, "automargin": False}, margin_t=76
)

# Charts with a legend of three entries (three rows on a phone), of one, and of
# none, and a 3D chart, which sets margins of its own. The wide chart's side and
# bottom margins already hold its labels, so once its title has reserved its
# band nothing moves its margins again: Plotly draws it once, and leaves its
# legend where it placed it before the band (see place-legend.js).
_CHARTS = {
    "three": _time_series("Actuator Torque", 3),
    "one": _time_series("Linear Actuator Force", 1),
    "none": EndEffectorSpatialPlots().plot_xz(_PATH, np.linalg.norm(_PATH, axis=1)),
    "scene": EndEffectorSpatialPlots().plot_overview(_PATH, np.linalg.norm(_PATH, axis=1)),
    "wide": _time_series("Actuator Torque", 3).update_layout(margin={"l": 100, "r": 100, "b": 100}),
}


def _graph(name):
    return create_graph_component(name, _CHARTS[name])


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        # Two columns from 992px up, one below that, in the page's gutters, as a
        # dashboard's charts are
        self.app.layout = create_main_container(
            create_two_column_builder().create_layout(
                [
                    [_graph("three"), _graph("one")],
                    [_graph("none"), _graph("scene")],
                    [_graph("wide"), create_graph_component("before", _BEFORE)],
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


# Each chart's title, legend and plot area, in px from the chart's top, and how
# many rows its legend has
_BOXES = """() => Object.fromEntries([...document.querySelectorAll('.js-plotly-plot')].map(gd => {
  const top = gd.getBoundingClientRect().top;
  const box = node => {
    const rect = node.getBoundingClientRect();
    return {top: rect.top - top, bottom: rect.bottom - top};
  };
  const legend = gd._fullLayout.showlegend && gd.querySelector('.legend .bg');
  const rows = new Set([...gd.querySelectorAll('.legend .traces')].map(
    trace => Math.round(trace.getBoundingClientRect().top)
  ));
  return [gd.parentElement.id, {
    title: box(gd.querySelector('.g-gtitle')),
    legend: legend ? {...box(legend), rows: rows.size} : null,
    plot: {top: gd._fullLayout._size.t},
    width: gd._fullLayout.width,
  }];
}))"""

# Whether every chart has drawn at its div's width: a resized window redraws
# the charts a moment later
_SETTLED = """() => {
  const gds = [...document.querySelectorAll('.js-plotly-plot')];
  return gds.length === 6 && gds.every(gd => gd._fullLayout && gd._fullLayout._size
    && Math.abs(gd._fullLayout.width - gd.getBoundingClientRect().width) < 1);
}"""


def _boxes(browser, url, width, scripts=("place-legend.js",)):
    """Each chart's boxes, on a page that loads, of the scripts that redraw its
    charts, only ``scripts``: lining up plot areas redraws charts, which would
    place their legends again whatever place-legend.js does."""
    page = browser.new_page(viewport={"width": width, "height": 1000})
    for script in {"place-legend.js", "align-plot-areas.js"} - set(scripts):
        page.route(f"**/assets/{script}*", lambda route: route.abort())
    page.goto(url)
    page.wait_for_function(_SETTLED, timeout=10000)
    page.wait_for_timeout(300)  # for the margins the title and legend push out
    boxes = page.evaluate(_BOXES)
    page.close()
    return boxes


def _charts(browser, url, width):
    """The charts laid out by the theme, by name."""
    return {name: box for name, box in _boxes(browser, url, width).items() if name in _CHARTS}


def test_the_legend_placing_script_ships_with_the_package():
    assert (assets_path / "place-legend.js").is_file()


@pytest.mark.parametrize("width", [375, 1600])
def test_plotly_leaves_a_legend_over_the_title_that_the_script_places_under_it(browser, url, width):
    """Without place-legend.js Plotly leaves the wide chart's legend where it
    placed it before the title reserved its band (if this fails, Plotly places
    it again itself, and the script can go)."""
    unplaced = _boxes(browser, url, width, scripts=())["wide"]
    assert unplaced["legend"]["top"] < unplaced["title"]["bottom"]
    placed = _boxes(browser, url, width)["wide"]
    assert placed["legend"]["top"] >= placed["title"]["bottom"]
    assert placed["plot"]["top"] == unplaced["plot"]["top"]


def test_the_legend_stays_placed_when_the_plot_areas_line_up(browser, url):
    """The page loads both scripts that redraw charts."""
    boxes = _boxes(browser, url, 375, scripts=("place-legend.js", "align-plot-areas.js"))
    for name in _CHARTS:
        legend = boxes[name]["legend"]
        assert legend is None or legend["top"] >= boxes[name]["title"]["bottom"], name


@pytest.mark.parametrize("width", [375, 768, 1600])
def test_title_legend_and_plot_area_keep_clear_of_each_other(browser, url, width):
    boxes = _charts(browser, url, width)
    for name, chart in boxes.items():
        title, legend, plot = chart["title"], chart["legend"], chart["plot"]
        assert title["top"] >= 0, name  # not lifted off the figure's top
        if legend:
            assert legend["top"] >= title["bottom"], name
            assert plot["top"] >= legend["bottom"], name
        else:
            assert plot["top"] >= title["bottom"], name
    if width == 375:
        # A phone's chart wraps the legend of three entries onto rows of their own
        assert boxes["three"]["legend"]["rows"] == 3
    assert boxes["one"]["legend"]["rows"] == 1


@pytest.mark.parametrize("width", [375, 1600])
def test_titles_line_up_and_legends_start_one_gap_below_them(browser, url, width):
    """Plotly draws a title it reserves a band for lower than one it does not, so
    every chart's title takes the band, whatever its margins and its legend."""
    boxes = _charts(browser, url, width)
    assert len({round(chart["title"]["top"]) for chart in boxes.values()}) == 1, boxes
    legends = [chart["legend"]["top"] for chart in boxes.values() if chart["legend"]]
    assert max(legends) - min(legends) < 1, boxes


@pytest.mark.parametrize("theme", ["default", "futuristic"])
def test_every_theme_s_title_reserves_its_band_over_the_legend(theme):
    """Plotly reserves a title's band of the top margin, and pushes a legend
    over the plot area out below it, only when the band is deeper than
    margin.t: at least the title's font size and its pads. A theme's margin.t
    is the 12px Plotly pads a pushed margin with, as a smaller one would pad
    every axis's labels less."""
    layout = ThemeManager(theme).get_plotly_theme()["layout"]
    title, legend = layout["title"], layout["legend"]
    assert title["automargin"] is True
    assert (title["yref"], title["y"], title["yanchor"]) == ("container", 1, "top")
    assert layout["margin"]["t"] == 12
    assert layout["margin"]["t"] < layout["font"]["size"] + title["pad"]["t"] + title["pad"]["b"]
    # Over the plot area, on the paper, so it pushes the margin out
    assert legend.get("yref", "paper") == "paper"
    assert (legend["y"], legend["yanchor"]) == (1.02, "bottom")


@pytest.mark.parametrize("width", [375, 1600])
def test_a_legend_of_one_row_keeps_its_place(browser, url, width):
    """The title, a legend of one row and the plot area are where they were
    before the title reserved its band, to within a pixel and a half: the band
    is as deep as the title's line, which the browser's font rendering sets."""
    boxes = _boxes(browser, url, width)
    now, before = boxes["one"], boxes["before"]
    assert now["legend"]["rows"] == before["legend"]["rows"] == 1
    for part in ("title", "legend", "plot"):
        for edge in ("top", "bottom") if part != "plot" else ("top",):
            assert abs(now[part][edge] - before[part][edge]) < 1.5, (part, edge, now, before)
