# Dashboard Components
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

import datetime
import dash
from dash import html, dash_table, dcc
import dash_bootstrap_components as dbc
import pandas as pd
from mergedeep import merge

from ..themes.manager import get_theme_manager

CURRENT_YEAR = datetime.datetime.now().year


def construct_dash_table(table):

    theme = get_theme_manager()

    # Get base style but override problematic sizing properties
    default_style = theme.get_component_style("table_container").copy()
    # Remove problematic flex and overflow properties that can cause resize loops
    default_style.pop("flex", None)
    default_style.pop("overflow-y", None)
    # Set explicit height instead of flexible sizing
    if "height" not in default_style:
        default_style["height"] = "600px"

    if isinstance(table, pd.DataFrame):
        # Create proper column definitions for DataTable
        columns = [{"name": col, "id": col} for col in table.columns]
        data = table.to_dict("records")
    else:
        try:
            columns = [{"name": col, "id": col} for col in table.columns]
            data = (
                table.to_dict("records") if hasattr(table, "to_dict") else table.values
            )
        except AttributeError as e:
            raise e

    return dash_table.DataTable(
        id="table",
        columns=columns,
        data=data,
        style_cell_conditional=[
            {"if": {"column_id": "parameter"}, "textAlign": "left"},
            {"if": {"column_id": "metric"}, "textAlign": "left"},
            {"if": {"column_id": "unit"}, "textAlign": "center"},
            {"if": {"column_id": "value"}, "textAlign": "center"},
        ],
        style_header={"backgroundColor": "rgb(230, 230, 230)", "fontWeight": "bold"},
        style_table={**default_style, "overflowY": "auto"},
        page_action="none",
    )


def create_tabs(children, id="dashboard", value="actuator-dynamics", style=None):

    theme = get_theme_manager()

    default_style = theme.get_component_style("tab_container").copy()
    if style:
        merge(default_style, style)
    style = default_style

    return dcc.Tabs(
        id=id,
        value=value,
        children=children,
        style=style,
    )


def create_graph_component(graph_id, figure, width="auto", style=None, config=None):

    theme = get_theme_manager()

    graph_style = theme.get_component_style("graph_container").copy()
    graph_config = {
        "displayModeBar": False,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
    }
    if style:
        graph_style.update(style)
    if config:
        graph_config.update(config)

    return dbc.Col(
        html.Div(
            [
                dcc.Graph(
                    id=graph_id,
                    figure=figure,
                    config=graph_config,
                    style=graph_style,
                )
            ],
            # style=graph_style,
        ),
        width=width,
    )


def create_tab_layout(children, style=None):
    default_style = {}
    layout_style = merge({}, default_style, style or {})

    return html.Div(
        style=layout_style,
        children=children,
    )


def create_main_container(children=None, style=None):

    theme = get_theme_manager()

    default_style = theme.get_component_style("main_container").copy()
    if style:
        default_style.update(style)

    return html.Div(
        id="dashboard-display-area-primary",
        style=default_style,
        children=children,
    )


def create_dashboard_header(title="Robot Simulator"):

    theme = get_theme_manager()

    return html.H1(title, style=theme.get_component_style("header"))


def create_dashboard_footer(text=f"Davidson Engineering Ltd. © {CURRENT_YEAR}"):

    theme = get_theme_manager()

    return html.Footer(f"{text}", style=theme.get_component_style("footer"))


def create_themed_tab(label, value):

    theme = get_theme_manager()

    icon = theme.get_tab_icon(value)
    tab_label = f"{icon}  {label}" if icon else label

    return dcc.Tab(
        label=tab_label,
        value=value,
        style=theme.get_component_style("tab"),
        selected_style=theme.get_component_style("tab_selected"),
    )


def create_body_container(children, style=None):

    theme = get_theme_manager()

    default_style = theme.get_component_style("body_container")
    if style:
        default_style.update(style)

    return html.Div(
        children=children,
        style=default_style,
    )


def create_job_selector_dropdown(job_options, current_job_id=None):

    theme = get_theme_manager()

    # icon = theme.get_tab_icon("job-selector") or "󰮔"
    dropdown_container_style = theme.get_component_style("dropdown-container")
    dropdown_style = theme.get_component_style("dropdown") or {}

    return html.Div(
        id="job-selector-container",
        children=[
            dcc.Dropdown(
                id="job-selector",
                options=job_options,
                value=current_job_id,
                style=dropdown_style,
                clearable=False,
            ),
        ],
        style=dropdown_container_style,
    )
