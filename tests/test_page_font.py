"""A dashboard page loads nothing from another host, and text no theme style
reaches is in the theme's body font rather than Bootstrap's."""

import re

import pytest
from dash import html

from datadash.dashboard.base import DashboardApp, assets_path
from datadash.themes.manager import reset_theme_manager, set_theme


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        self.app.layout = html.Div()


@pytest.fixture
def page():
    """The page a dashboard serves, under a theme"""

    def serve(theme):
        set_theme(theme)
        dashboard = _Dashboard()
        dashboard.initialize()
        return dashboard.app.server.test_client().get("/").get_data(as_text=True)

    yield serve
    reset_theme_manager()


@pytest.mark.parametrize("stylesheet", sorted(assets_path.rglob("*.css")), ids=lambda p: p.name)
def test_no_stylesheet_loads_from_another_host(stylesheet):
    css = stylesheet.read_text()
    assert not re.search(r"@import\s+(url\(|['\"])", css)
    assert not re.search(r"url\(\s*['\"]?(https?:)?//", css)


def test_bootstraps_sans_serif_names_no_font_the_page_does_not_load():
    css = (assets_path / "bootstrap.min.css").read_text()
    stack = re.search(r"--bs-font-sans-serif:([^;]*);", css)
    assert stack
    assert "Nunito" not in stack.group(1)


@pytest.mark.parametrize("theme", ["default", "futuristic"])
def test_the_page_takes_the_themes_body_font(page, theme):
    body_font = set_theme(theme).fonts["body"]

    served = page(theme)

    rule = f"--bs-body-font-family: {body_font}; --bs-font-sans-serif: {body_font};"
    assert rule in served
    # After Bootstrap's own :root rule, so it takes precedence
    assert served.index("bootstrap.min.css") < served.index(rule)
