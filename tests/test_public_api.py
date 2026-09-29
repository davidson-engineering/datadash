"""The top-level names are the contract with dashboards built on datadash."""

import subprocess
import sys

import pytest
from dash import dcc

import datadash


@pytest.mark.parametrize("name", datadash.__all__)
def test_public_name_resolves(name):
    assert getattr(datadash, name) is not None


def test_import_does_not_load_dash():
    code = "import sys, datadash; print('dash' in sys.modules, 'plotly' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False", "False"]


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no_such_name"):
        datadash.no_such_name  # noqa: B018


def test_tabs_start_on_first_tab_without_a_value():
    tabs = datadash.create_tabs([dcc.Tab(label="A", value="a"), dcc.Tab(label="B", value="b")])
    # dcc.Tabs only falls back to its first tab when "value" is absent entirely
    assert "value" not in tabs.to_plotly_json()["props"]


def test_tabs_keep_an_explicit_value():
    tabs = datadash.create_tabs([dcc.Tab(label="A", value="a")], value="a")
    assert tabs.to_plotly_json()["props"]["value"] == "a"


def test_header_defaults_to_theme_title():
    theme = datadash.set_theme("default")
    assert datadash.create_dashboard_header().children == theme.get_dashboard_title()


def test_dashboard_assets_ship_with_the_package():
    from datadash.dashboard.base import assets_path

    assert (assets_path / "bootstrap.min.css").is_file()
