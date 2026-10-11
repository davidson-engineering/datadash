"""Every dashboard page loads the script that keeps a 3D scene's camera when an
animation plays (assets/keep-3d-camera.js)."""

from dash import html

from datadash.dashboard.base import DashboardApp, assets_path


class _Dashboard(DashboardApp):
    def initialize(self):
        self.app = self._create_app()
        self.app.layout = html.Div()
        return self


def test_the_camera_keeping_script_ships_with_the_package():
    assert (assets_path / "keep-3d-camera.js").is_file()


def test_every_dashboard_page_loads_the_camera_keeping_script():
    app = _Dashboard().initialize().app

    page = app.server.test_client().get("/").get_data(as_text=True)

    assert "/assets/keep-3d-camera.js" in page
