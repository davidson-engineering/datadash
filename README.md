# datadash

Themed Plotly figure builders and Dash dashboard components, for building
results dashboards on top of any computation.

## Installation

datadash is released as git tags. Pin one in your project:

```bash
uv add "datadash @ git+https://github.com/davidson-engineering/datadash@v0.2.0"
```

or in `pyproject.toml`:

```toml
[project]
dependencies = ["datadash"]

[tool.uv.sources]
datadash = { git = "https://github.com/davidson-engineering/datadash", tag = "v0.2.0" }
```

Requires Python 3.13, Dash 4.4.1 or later and Plotly 7.1 or later.

**Upgrading from v0.1.0.** v0.2.0 targets Dash 4 and Plotly 7.
`construct_dash_table` now returns an `html.Div` holding an HTML table instead
of a `dash_table.DataTable` (deprecated in Dash 4): pass cell styles through its
`cell_style` argument rather than setting `style_cell` on the result. Apps still
on Dash 3 and Plotly 6 can stay on the `v0.1.0` tag.

## Public API

The names exported from the top-level package (`datadash.__all__`) are the
public API, e.g. `from datadash import CombinedPlotBuilder, set_theme`. They are
loaded on first use, so `import datadash` is cheap. Everything else is internal
and may change between minor versions while datadash is at 0.x.

## Themes

Every builder and component reads the global `ThemeManager`
(`datadash/themes/manager.py`), so pick the theme before building anything:

```python
from datadash import set_theme

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
Lux), and `assets/fonts.css` declares the bundled Nerd Fonts.
`assets/dash-components.css` lets Dash 4's dropdowns take the theme's
`dropdown` style: that style sets the `--Dash-Fill-Interactive-Strong` (accent),
`--Dash-Stroke-Strong` (borders) and `--Dash-Text-Strong` (option text)
variables Dash 4 colors them with, so an override file recolors dropdowns by
setting those keys. Dash serves the `assets/` folder automatically.

## Figures

```python
import numpy as np
from datadash import BasicPlotBuilder

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
  They are imported from `datadash.builders.plot` and are not in `__all__`
  yet, so they may still change between minor versions.
- `PlotRegistry` (`plots/registry.py`) registers plot functions by id with a
  decorator, so an app can look plots up by name.

## Dashboard components

`datadash/dashboard/components.py` returns themed Dash components:

| Function | Returns |
| --- | --- |
| `create_dashboard_header(title)`, `create_dashboard_footer(text)` | Page header (defaults to the theme's `dashboard_title`) and footer |
| `create_tabs(children, id, value)`, `create_themed_tab(label, value)` | Tab bar (first tab selected unless `value` is given) and tabs (with the theme's tab icons) |
| `create_main_container(children)`, `create_body_container(children)`, `create_tab_layout(children)` | Page structure |
| `create_graph_component(graph_id, figure, width)` | A themed `dcc.Graph` in a `dbc.Col` of the given width |
| `construct_dash_table(table, table_id, max_width, cell_style)` | A read-only HTML table from a DataFrame, numbers right-aligned and floats rounded, with a sticky header |
| `create_job_selector_dropdown(job_options, current_job_id)` | A single job dropdown |
| `create_parameter_filter_dropdowns(sweep_analyzer, current_job_id, formatter)` | One dropdown per swept parameter; `formatter(param)` returns `(label, format_value)`, where `format_value(value)` renders an option's label (option values stay raw) |
| `parameter_filter_container_style()` | The filter bar style, for callbacks that hide and show it |

`DashboardApp` (`dashboard/base.py`) is the base class for an app:
implement `initialize()` to build `self.app` (use `_create_app()`, which wires
the assets folder and page title), then call `run(debug, port, host)`.

## Development

```bash
uv sync            # installs datadash and the dev tools from uv.lock
uv run pytest
uv run ruff check .
```

CI (`.github/workflows/ci.yml`) runs both against `uv.lock` on pushes to `main`
and on every pull request. Weekly, or when started by hand, it also runs the
tests (not ruff) against the newest dependency versions `pyproject.toml`
allows.

To work on datadash alongside an app that uses it, install your checkout into
the app's environment for the session: `uv pip install -e ../datadash` from the
app. The app's next `uv sync` restores its pinned tag.

## Releasing

Bump `version` in `pyproject.toml` and run `uv lock` (the lock file records
the version, and CI installs with `uv sync --locked`). Merge to `main`, then tag
the merge commit `v<version>` and push the tag. Apps upgrade by changing the
tag they pin.
