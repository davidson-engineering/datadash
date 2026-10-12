"""Themed Plotly figure builders and Dash dashboard components.

The names in ``__all__`` are the public API. Each is imported on first access,
so ``import datadash`` stays cheap and does not load Dash or Plotly.
"""

from importlib import import_module
from importlib.metadata import version
from typing import TYPE_CHECKING

__version__ = version("datadash")

# What type checkers see of the names imported on first access below
if TYPE_CHECKING:
    from datadash.builders.content import (
        ContentBuilder,
        create_single_column_builder,
        create_split_window_builder,
        create_table_builder,
        create_three_column_builder,
        create_two_column_builder,
    )
    from datadash.builders.figure import PlotFigure3DAnimation
    from datadash.builders.plot import (
        BasicPlotBuilder,
        CombinedPlotBuilder,
        SubplotsPlotBuilder,
    )
    from datadash.builders.trace import (
        create_trace_constructor,
        get_plot_range,
    )
    from datadash.dashboard.base import DashboardApp
    from datadash.dashboard.components import (
        construct_dash_table,
        create_body_container,
        create_dashboard_footer,
        create_dashboard_header,
        create_filter_dropdown,
        create_graph_component,
        create_job_selector_dropdown,
        create_main_container,
        create_parameter_filter_dropdowns,
        create_tab_layout,
        create_tabs,
        create_themed_tab,
        filter_row_class_name,
        parameter_filter_container_style,
    )
    from datadash.plots.registry import (
        PlotMetadata,
        PlotRegistry,
        register_plot,
    )
    from datadash.themes.manager import (
        get_theme_manager,
        set_theme,
    )

_EXPORTS = {
    # Themes
    "get_theme_manager": "datadash.themes.manager",
    "set_theme": "datadash.themes.manager",
    # Plot registry
    "PlotMetadata": "datadash.plots.registry",
    "PlotRegistry": "datadash.plots.registry",
    "register_plot": "datadash.plots.registry",
    # Figures and traces
    "BasicPlotBuilder": "datadash.builders.plot",
    "CombinedPlotBuilder": "datadash.builders.plot",
    "SubplotsPlotBuilder": "datadash.builders.plot",
    "PlotFigure3DAnimation": "datadash.builders.figure",
    "create_trace_constructor": "datadash.builders.trace",
    "get_plot_range": "datadash.builders.trace",
    # Page layout
    "ContentBuilder": "datadash.builders.content",
    "create_single_column_builder": "datadash.builders.content",
    "create_split_window_builder": "datadash.builders.content",
    "create_table_builder": "datadash.builders.content",
    "create_three_column_builder": "datadash.builders.content",
    "create_two_column_builder": "datadash.builders.content",
    # Dashboard
    "DashboardApp": "datadash.dashboard.base",
    "construct_dash_table": "datadash.dashboard.components",
    "create_body_container": "datadash.dashboard.components",
    "create_dashboard_footer": "datadash.dashboard.components",
    "create_dashboard_header": "datadash.dashboard.components",
    "create_filter_dropdown": "datadash.dashboard.components",
    "create_graph_component": "datadash.dashboard.components",
    "create_job_selector_dropdown": "datadash.dashboard.components",
    "create_main_container": "datadash.dashboard.components",
    "create_parameter_filter_dropdowns": "datadash.dashboard.components",
    "create_tab_layout": "datadash.dashboard.components",
    "create_tabs": "datadash.dashboard.components",
    "create_themed_tab": "datadash.dashboard.components",
    "filter_row_class_name": "datadash.dashboard.components",
    "parameter_filter_container_style": "datadash.dashboard.components",
}

__all__ = [
    "__version__",
    "get_theme_manager",
    "set_theme",
    "PlotMetadata",
    "PlotRegistry",
    "register_plot",
    "BasicPlotBuilder",
    "CombinedPlotBuilder",
    "SubplotsPlotBuilder",
    "PlotFigure3DAnimation",
    "create_trace_constructor",
    "get_plot_range",
    "ContentBuilder",
    "create_single_column_builder",
    "create_split_window_builder",
    "create_table_builder",
    "create_three_column_builder",
    "create_two_column_builder",
    "DashboardApp",
    "construct_dash_table",
    "create_body_container",
    "create_dashboard_footer",
    "create_dashboard_header",
    "create_filter_dropdown",
    "create_graph_component",
    "create_job_selector_dropdown",
    "create_main_container",
    "create_parameter_filter_dropdowns",
    "create_tab_layout",
    "create_tabs",
    "create_themed_tab",
    "filter_row_class_name",
    "parameter_filter_container_style",
]


def __getattr__(name):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module 'datadash' has no attribute {name!r}") from None
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
