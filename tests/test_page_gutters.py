"""A page's side gutters (assets/dash-components.css), in a browser.

A desktop's content sits 1.5rem in from the window's sides, the theme's
main_container and dropdown-container padding, and its rows of charts half
a gutter further, Bootstrap's column padding. A phone has no width to spare
for that: 36px a side left a 375px phone's charts 303px. Its gutters are
0.75rem, and its charts line up with everything else on the page.
"""

import os
import threading

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pytest
from dash import html
from werkzeug.serving import make_server

from datadash.builders.content import (
    PADDED_ROW,
    create_single_column_builder,
    create_split_window_builder,
    create_three_column_builder,
    create_two_column_builder,
)
from datadash.builders.plot import CombinedPlotBuilder
from datadash.dashboard.base import DashboardApp
from datadash.dashboard.components import (
    construct_dash_table,
    create_body_container,
    create_filter_dropdown,
    create_graph_component,
    create_main_container,
    filter_row_class_name,
    parameter_filter_container_style,
)

playwright = pytest.importorskip("playwright.sync_api")


def _chart(name):
    x = np.linspace(0, 1, 50)
    return CombinedPlotBuilder().create_plot(
        x, np.column_stack([np.sin(6 * x), np.cos(6 * x)]), title=name, headers=["a", "b"]
    )


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        options = [{"label": str(n), "value": n} for n in range(3)]
        # As many filters as a laptop's row holds, so the first row is full at any width
        filters = [create_filter_dropdown(f"filter-{n}", "Filter", options, 0) for n in range(3)]
        table = pd.DataFrame({"Parameter": ["Mass", "Length"], "Value": [1.5, 250.0]})
        self.app.layout = create_body_container(
            [
                html.Div(
                    dbc.Row(filters, className=filter_row_class_name(len(filters))),
                    style=parameter_filter_container_style(),
                ),
                create_main_container(
                    [
                        # A heading over charts, indented to line up with them
                        html.H2(
                            "Charts",
                            id="heading",
                            style={"paddingLeft": "var(--datadash-chart-inset)"},
                        ),
                        # Two columns from 992px up, one below that
                        create_two_column_builder().create_layout(
                            [[create_graph_component(name, _chart(name)) for name in "ab"]]
                        ),
                        create_single_column_builder().create_layout(
                            [[construct_dash_table(table, table_id="table")]]
                        ),
                    ]
                ),
            ]
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


# How far in from the window's left and right sides the page draws each kind of
# content, to the pixel: the outermost of its kind, and for the heading, its text
_EDGES = """() => {
  const edges = boxes => [
    Math.round(Math.min(...boxes.map(box => box.left))),
    Math.round(innerWidth - Math.max(...boxes.map(box => box.right))),
  ];
  const all = selector => [...document.querySelectorAll(selector)].map(
    element => element.getBoundingClientRect()
  );
  const text = document.createRange();
  text.selectNodeContents(document.getElementById('heading'));
  return {
    filters: edges(all('.dash-dropdown-wrapper')),
    charts: edges(all('.js-plotly-plot')),
    table: edges(all('#table')),
    heading: Math.round(text.getBoundingClientRect().left),
    overflow: document.documentElement.scrollWidth - innerWidth,
  };
}"""


def _edges(browser, url, width):
    page = browser.new_page(viewport={"width": width, "height": 1000})
    page.goto(url)
    page.wait_for_function(
        "() => document.querySelectorAll('.js-plotly-plot').length === 2"
        " && document.querySelectorAll('.dash-dropdown-wrapper').length === 3"
    )
    edges = page.evaluate(_EDGES)
    page.close()
    return edges


# The narrower phones, a common one, the widest, and the widest window below
# Bootstrap's sm breakpoint
@pytest.mark.parametrize("width", [360, 375, 430, 575])
def test_a_phone_s_content_lines_up_in_narrow_gutters(browser, url, width):
    edges = _edges(browser, url, width)

    for kind in ("filters", "charts", "table"):
        assert edges[kind] == [12, 12], (kind, edges)
    assert edges["heading"] == 12
    assert edges["overflow"] == 0


# The narrowest windows with a desktop's gutters, a tablet's, and two columns
@pytest.mark.parametrize("width", [576, 768, 1024])
def test_a_desktop_s_charts_sit_half_a_gutter_inside_the_rest_of_the_page(browser, url, width):
    edges = _edges(browser, url, width)

    for kind in ("filters", "table"):
        assert edges[kind] == [24, 24], (kind, edges)
    assert edges["charts"] == [36, 36], edges
    assert edges["heading"] == 36
    assert edges["overflow"] == 0


@pytest.mark.parametrize(
    "factory",
    [create_two_column_builder, create_three_column_builder, create_split_window_builder],
)
def test_rows_of_charts_are_padded_rows_beside_any_class_of_their_own(factory):
    assert factory().row_properties["className"] == PADDED_ROW
    builder = factory(row_properties={"className": "g-2"})
    assert builder.row_properties["className"] == f"{PADDED_ROW} g-2"
