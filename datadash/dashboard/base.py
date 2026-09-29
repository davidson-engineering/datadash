# Base Dashboard Classes
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from abc import ABC, abstractmethod
from pathlib import Path

import dash

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

        # No external Bootstrap: assets/bootstrap.min.css (Bootswatch Lux) is a
        # complete Bootstrap build and is loaded automatically from the assets
        # folder. Adding the CDN copy as well loaded two Bootstraps that fought
        # over the same selectors.
        external_stylesheets = []

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
            suppress_callback_exceptions=True,
        )
