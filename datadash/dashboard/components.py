# Dashboard Components
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
from mergedeep import merge

from ..themes.manager import get_theme_manager

# Enough precision for any engineering value, few enough digits to drop binary
# float noise (0.16499999999999998 -> 0.165) that would otherwise be displayed
TABLE_SIGNIFICANT_DIGITS = 10


def _clean_float(value):
    """Round floats to TABLE_SIGNIFICANT_DIGITS; leave every other value untouched."""
    if isinstance(value, float):
        return float(f"{value:.{TABLE_SIGNIFICANT_DIGITS}g}")
    return value


def _is_numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _column_alignment(rows, column):
    """Right-align mostly-numeric columns, left-align text columns.

    Decided by majority, so a numeric column with the odd text entry (a model
    name among dimensions, say) still lines its numbers up.
    """
    values = [row.get(column) for row in rows if row.get(column) not in (None, "")]
    numeric = sum(_is_numeric(v) for v in values)
    return "right" if values and numeric * 2 >= len(values) else "left"


def construct_dash_table(table, table_id="table", max_width=None):
    """Render a DataFrame as a themed DataTable.

    Numeric columns are right-aligned and text columns left-aligned, decided from
    the data rather than from column names. The table scrolls inside a container
    capped at the theme's max height, with the header row pinned while scrolling.

    Args:
        table: DataFrame (or DataFrame-like with ``columns`` and ``to_dict``)
        table_id: Component id; must be unique within the page
        max_width: Optional CSS max-width, to keep narrow tables (few columns)
            from stretching their values far away from their labels
    """
    theme = get_theme_manager()

    columns = [{"name": col, "id": col} for col in table.columns]
    data = table.to_dict("records")
    data = [{k: _clean_float(v) for k, v in row.items()} for row in data]

    header_style = theme.get_component_style("table_header")
    return dash_table.DataTable(
        id=table_id,
        columns=columns,
        data=data,
        style_cell=theme.get_component_style("table_cell"),
        style_cell_conditional=[
            {"if": {"column_id": col["id"]}, "textAlign": _column_alignment(data, col["id"])}
            for col in columns
        ],
        # Sticky within the table's own scroll container
        style_header={**header_style, "position": "sticky", "top": 0, "zIndex": 1},
        style_table={
            **theme.get_component_style("table_container"),
            **({"maxWidth": max_width} if max_width else {}),
        },
        page_action="none",
    )


def create_tabs(children, id="dashboard", value=None, style=None):
    """Themed tab bar. With no ``value``, the first tab starts selected."""

    theme = get_theme_manager()

    default_style = theme.get_component_style("tab_container").copy()
    if style:
        merge(default_style, style)
    style = default_style

    # dcc.Tabs falls back to its first tab only when value is absent: an
    # explicit None is serialized and leaves no tab selected.
    selection = {} if value is None else {"value": value}

    return dcc.Tabs(
        id=id,
        **selection,
        children=children,
        style=style,
        # dcc.Tabs stacks tabs vertically below 800px by default, which overflowed
        # the header. The theme's tab bar wraps onto extra rows instead.
        mobile_breakpoint=0,
    )


def create_graph_component(graph_id, figure, width=12, style=None, config=None):

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


def create_dashboard_header(title=None):
    """Page title; defaults to the theme's ``dashboard_title``."""

    theme = get_theme_manager()

    if title is None:
        title = theme.get_dashboard_title()
    return html.H1(title, style=theme.get_component_style("header"))


def create_dashboard_footer(text=""):

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


def parameter_filter_container_style():
    """Style for the parameter filter container.

    Shared by the initial render and any callback that toggles its visibility, so
    showing it again restores exactly the same layout.
    """
    return get_theme_manager().get_component_style("dropdown-container")


def _default_parameter_format(param):
    """Show the raw parameter name, and values to 3 decimal places."""
    return param, lambda value: f"{value:.3f}" if _is_numeric(value) else str(value)


def create_parameter_filter_dropdowns(sweep_analyzer, current_job_id=None, formatter=None):
    """Create filter dropdowns for each swept parameter to narrow down job selection.

    Args:
        sweep_analyzer: SweepAnalyzer instance with parameter sweep data
        current_job_id: Currently selected job ID
        formatter: Optional ``param -> (label, format_value)``, where
            ``format_value(value) -> str`` renders an option. Option values stay
            raw, so callbacks receive the original data.

    Returns:
        Container with one dropdown per swept parameter, laid out responsively
    """
    theme = get_theme_manager()
    formatter = formatter or _default_parameter_format
    dropdown_style = theme.get_component_style("dropdown")
    label_style = theme.get_component_style("dropdown-label")

    if not sweep_analyzer or sweep_analyzer.df.empty:
        return html.Div(id="job-selector-container", style={"display": "none"})

    swept_params = sweep_analyzer.get_swept_parameters()
    if not swept_params:
        return html.Div(id="job-selector-container", style={"display": "none"})

    current_values = {}
    if current_job_id and current_job_id in sweep_analyzer.df.index:
        for param in swept_params:
            current_values[param] = sweep_analyzer.df.loc[current_job_id, param]

    columns = []
    for param in swept_params:
        param_values = sweep_analyzer.get_parameter_values(param)
        label, format_value = formatter(param)
        options = [{"label": format_value(val), "value": val} for val in param_values]
        value = current_values.get(param, param_values[0] if param_values else None)

        columns.append(
            dbc.Col(
                [
                    html.Label(label, title=label, style=label_style),
                    dcc.Dropdown(
                        id={"type": "param-filter", "param": param},
                        options=options,
                        value=value,
                        style=dropdown_style,
                        clearable=False,
                    ),
                ],
                # One per row on phones, then 2, 3, and all six in one row on wide screens
                xs=12,
                sm=6,
                lg=4,
                xxl=2,
            )
        )

    return html.Div(
        id="job-selector-container",
        children=dbc.Row(columns, className="g-3", justify="center"),
        style=parameter_filter_container_style(),
    )
