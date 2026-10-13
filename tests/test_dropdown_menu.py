"""A filter's dropdown shows its options as soon as its menu does, in a menu as
wide as the dropdown (assets/dash-components.css), in a browser.

Dash 4.4 puts the options in an open menu by an effect that runs after the menu
is drawn, so the first time a menu opened, it was drawn once with only its
search box. And it widens a menu to its longest option, which on a phone moved
the menu left of its dropdown and flush with the screen's right edge.
"""

import os
import threading

import pytest
from dash import html
from werkzeug.serving import make_server

from datadash.dashboard.base import DashboardApp
from datadash.dashboard.components import create_filter_dropdown

playwright = pytest.importorskip("playwright.sync_api")

OPTIONS = [{"label": f"Option {i}", "value": i} for i in range(1, 4)]
# Options longer than a phone's dropdown is wide, as a dashboard's metrics are
LONG = [
    {"label": f"End effector acceleration {axis}, std dev (m·s⁻²)", "value": axis} for axis in "XYZ"
]


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        self.app.layout = html.Div(
            [
                create_filter_dropdown("filter", "Filter", OPTIONS, 1),
                create_filter_dropdown("empty", "Empty", [], None),
                # Inset as a dashboard's page insets it
                html.Div(
                    create_filter_dropdown("long", "Long", LONG, "X"),
                    style={"padding": "0 24px"},
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


# The menu of the dropdown ``id`` in each frame the page draws from now on:
# whether it is shown, and the options it shows
_WATCH = """id => {
  window.frames_drawn = [];
  const menu = () => document.getElementById(id)
    .closest('.dash-dropdown-wrapper').querySelector('.dash-dropdown-content');
  const frame = () => {
    const content = menu();
    window.frames_drawn.push(content && {
      shown: getComputedStyle(content).visibility === 'visible',
      options: [...content.querySelectorAll('.dash-dropdown-option')]
        .map(option => option.textContent.trim()),
    });
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}"""


def _menus_drawn(page, dropdown_id, open_menu):
    """The menus drawn, in order, from opening ``dropdown_id``'s by ``open_menu``."""
    page.evaluate(_WATCH, dropdown_id)
    open_menu()
    page.wait_for_timeout(500)
    return [menu for menu in page.evaluate("window.frames_drawn") if menu is not None]


def _wait_for_focus(page, label):
    """Wait until the option ``label`` has the keyboard's focus."""
    page.wait_for_function(
        """label => document.activeElement.closest('.dash-dropdown-option')
          ?.textContent.trim() === label""",
        arg=label,
        timeout=2000,
    )


@pytest.fixture
def page(browser, url):
    page = browser.new_page()
    page.goto(url)
    page.wait_for_selector("#filter")
    yield page
    page.close()


def test_a_menu_shows_its_options_the_first_time_it_opens(page):
    menus = _menus_drawn(page, "filter", lambda: page.click("#filter"))

    labels = [option["label"] for option in OPTIONS]
    shown = [menu["options"] for menu in menus if menu["shown"]]
    assert shown
    assert shown == [labels] * len(shown)


def test_a_menu_opened_from_the_keyboard_focuses_its_selected_option(page):
    """Dash focuses it once the options are in, when the menu shows: an element
    hidden from view takes no focus."""
    page.focus("#filter")
    page.keyboard.press("ArrowDown")
    _wait_for_focus(page, "Option 1")

    page.keyboard.press("ArrowDown")
    _wait_for_focus(page, "Option 2")
    page.keyboard.press("Space")
    playwright.expect(page.locator("#filter")).to_have_text("Option 2")


def test_a_search_that_matches_no_option_says_so(page):
    page.click("#filter")
    page.locator(".dash-dropdown-search").fill("nothing like it")
    menu = page.locator(".dash-dropdown-content")
    assert menu.is_visible()
    assert menu.locator(".dash-dropdown-option").inner_text() == "No options found"


def test_the_menu_of_a_dropdown_with_no_options_stays_hidden(page):
    menus = _menus_drawn(page, "empty", lambda: page.click("#empty"))

    assert menus
    assert not [menu for menu in menus if menu["shown"]]


# A phone, and a desktop
@pytest.mark.parametrize("width", [375, 1600])
def test_a_menu_is_as_wide_as_its_dropdown_and_under_it(browser, url, width):
    """Its longer options wrap, inside it."""
    page = browser.new_page(viewport={"width": width, "height": 800})
    page.goto(url)
    page.click("#long")
    menu = page.locator(".dash-dropdown-content")
    playwright.expect(menu.locator(".dash-dropdown-option")).to_have_count(3)
    dropdown, box = page.locator("#long").bounding_box(), menu.bounding_box()
    options = [option.bounding_box() for option in menu.locator(".dash-dropdown-option").all()]
    page.close()

    assert dropdown and box
    assert box["x"] == pytest.approx(dropdown["x"], abs=1)
    assert box["width"] == pytest.approx(dropdown["width"], abs=1)
    for option in options:
        assert option and option["x"] + option["width"] <= box["x"] + box["width"] + 0.5
