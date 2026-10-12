# Base Dashboard Classes
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from abc import ABC, abstractmethod
from pathlib import Path

import dash

# Bootstrap build and bundled fonts, served by Dash from the package itself
assets_path = Path(__file__).resolve().parent.parent / "assets"


class DashboardApp(ABC):
    # The Dash app, which a subclass makes (as _create_app() does) in initialize()
    app: dash.Dash

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
    def initialize(self) -> None:
        """Make the Dash app, self.app, and its layout and callbacks."""

    def _create_app(self):
        from ..themes.manager import get_theme_manager

        theme = get_theme_manager()

        # No external Bootstrap: assets/bootstrap.min.css (Bootswatch Lux) is a
        # complete Bootstrap build and is loaded automatically from the assets
        # folder. Adding the CDN copy as well loaded two Bootstraps that fought
        # over the same selectors.
        external_stylesheets = []

        app = dash.Dash(
            __name__,
            external_stylesheets=external_stylesheets,
            assets_folder=str(assets_path),
            title=theme.get_dashboard_title(),
            suppress_callback_exceptions=True,
        )

        # Text that no theme style reaches, such as Dash's "Loading..." before
        # the layout arrives and Bootstrap's tooltips and popovers, takes the
        # theme's body font rather than Bootstrap's sans-serif. The rule comes
        # after the assets' stylesheets, so it wins over bootstrap.min.css.
        body_font = theme.fonts.get("body")
        if body_font:
            app.index_string = app.index_string.replace(
                "{%css%}",
                "{%css%}\n        <style>:root { "
                f"--bs-body-font-family: {body_font}; --bs-font-sans-serif: {body_font}; "
                "}</style>",
                1,
            )
        return app
