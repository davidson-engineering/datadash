# datadash

Themed Plotly figure builders and Dash dashboard components. It is the
presentation layer of
[robot-dashboard](https://github.com/davidson-engineering/robot-dashboard),
where it sits beside [robot-simulator](https://github.com/davidson-engineering/robot-simulator)
as a submodule in one uv workspace.

## Installation

Inside the robot-dashboard workspace, `uv sync` at the workspace root installs
it. On its own:

```bash
pip install git+https://github.com/davidson-engineering/datadash.git
```

Requires Python 3.13.

## Themes

Every builder and component reads the global `ThemeManager`
(`datadash/themes/manager.py`), so pick the theme before building anything:

```python
from datadash.themes.manager import get_theme_manager, set_theme

theme = set_theme("futuristic")  # or "default"
theme.merge_custom_config("my_app/theme.yaml")  # optional app overrides
```

Themes are YAML files in `datadash/themes/config/`: `default.yaml`,
`futuristic.yaml` (layered over the default), and `palettes.yaml` for trace
color palettes. `example_overrides.yaml` documents what an override file can
set. Overrides take precedence over everything the theme defines.

`set_theme` replaces the global manager, discarding earlier overrides, so call
it once and then apply overrides. `get_theme_manager()` returns the current
manager, creating the default theme on first use.

The dashboard stylesheet is `datadash/assets/bootstrap.min.css` (Bootswatch
Lux), and `assets/fonts.css` declares the bundled Nerd Fonts. Dash serves the
`assets/` folder automatically.

## Figures

```python
import numpy as np
from datadash.builders.plot import BasicPlotBuilder

t = np.linspace(0, 1, 100)
torque = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])

fig = BasicPlotBuilder().create_plot(
    t, torque,
    title="Actuator Torque",
    y_title="Torque", y_units="N·m",
    headers=["a1", "a2"],                 # trace keys, matched by theme trace styles
    labels=["Actuator 1", "Actuator 2"],  # legend and hover names
)
```

- `BasicPlotBuilder`, `CombinedPlotBuilder`, `SubplotsPlotBuilder`
  (`builders/plot.py`) build line plots, plots sharing axes, and subplot grids.
- `Spatial2DPlotBuilder` and `Spatial3DPlotBuilder` plot paths with equal axis
  scaling; `EndEffectorSpatialPlots` produces the XZ, YZ, XY, and 3D views.
- `PlotRegistry` (`plots/registry.py`) registers plot functions by id with a
  decorator, so an app can look plots up by name.

## Dashboard components

`datadash/dashboard/components.py` returns themed Dash components:

| Function | Returns |
| --- | --- |
| `create_dashboard_header(title)`, `create_dashboard_footer(text)` | Page header and footer |
| `create_tabs(children, id, value)`, `create_themed_tab(label, value)` | Tab bar and tabs (with the theme's tab icons) |
| `create_main_container(children)`, `create_body_container(children)`, `create_tab_layout(children)` | Page structure |
| `create_graph_component(graph_id, figure, width)` | A `dcc.Graph` in a themed card |
| `construct_dash_table(table, table_id, max_width)` | A `DataTable` from a DataFrame, numbers right-aligned and floats rounded, with a sticky header |
| `create_job_selector_dropdown(job_options, current_job_id)` | A single job dropdown |
| `create_parameter_filter_dropdowns(sweep_analyzer, current_job_id, formatter)` | One dropdown per swept parameter; `formatter(param)` returns `(label, unit)` for display |
| `parameter_filter_container_style()` | The filter bar style, for callbacks that hide and show it |

`DashboardApp` (`dashboard/base.py`) is the base class for an app:
implement `initialize()` to build `self.app` (use `_create_app()`, which wires
the assets folder and page title), then call `run(debug, port, host)`.

## Development

From the robot-dashboard workspace root:

```bash
uv run pytest src/datadash/tests
```
