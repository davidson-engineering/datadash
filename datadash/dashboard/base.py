# Base Dashboard Classes
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from abc import ABC, abstractmethod
import dash
from pathlib import Path
import dash_bootstrap_components as dbc


assets_path = Path("../assets/")


class DashboardApp(ABC):

    def run(self, debug=False, port=8050, host="127.0.0.1"):
        """Run the dashboard server.

        Args:
            debug: Enable debug mode
            port: Port to run on (default: 8050)
            host: Host to bind to (default: 127.0.0.1)
        """
        self.app.run(debug=debug, port=port, host=host)
        return self

    @abstractmethod
    def initialize(self):
        pass

    def _create_app(self):
        from ..themes.manager import get_theme_manager

        theme = get_theme_manager()

        # Add external stylesheets
        external_stylesheets = [dbc.themes.BOOTSTRAP]

        # # Add theme-specific font CSS
        # font_css = theme.get_font_css()
        # if font_css and theme.theme_name == "futuristic":
        #     # For futuristic theme, inject font CSS directly
        #     pass  # Will be handled by inline styles in dashboard layout

        return dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets,
            assets_folder=str(assets_path),
            title=theme.get_dashboard_title(),
        )
