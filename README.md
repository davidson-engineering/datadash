# datadash

Themed Plotly figure builders and Dash dashboard components, for building
results dashboards on top of any computation.

## Installation

datadash is released as git tags. Pin one in your project:

```bash
uv add "datadash @ git+https://github.com/davidson-engineering/datadash@v0.6.0"
```

or in `pyproject.toml`:

```toml
[project]
dependencies = ["datadash"]

[tool.uv.sources]
datadash = { git = "https://github.com/davidson-engineering/datadash", tag = "v0.6.0" }
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
manager, creating the default theme on first use; trace builders take the
manager current when they are made.

The dashboard stylesheet is `datadash/assets/bootstrap.min.css` (Bootswatch
Lux), and `assets/fonts.css` declares the bundled Nerd Fonts. A page fetches
nothing from another host: Lux's Nunito Sans import from Google Fonts is cut
out, and `_create_app()` sets Bootstrap's body font to the theme's `fonts.body`,
so text no theme style reaches (Dash's "Loading..." placeholder, Bootstrap
tooltips) is in the theme's font too.
`assets/dash-components.css` lets Dash 4's dropdowns take the theme's
`dropdown` style: that style sets the `--Dash-Fill-Interactive-Strong` (accent),
`--Dash-Stroke-Strong` (borders) and `--Dash-Text-Strong` (option text)
variables Dash 4 colors them with, so an override file recolors dropdowns by
setting those keys. `assets/keep-3d-camera.js` keeps a 3D scene's camera where
the user left it when an animation plays: Plotly 4.1 saves a dragged camera
into a layout that `Plotly.react` (which `dcc.Graph` calls) has since replaced,
so each animation frame jumped back to an earlier camera.
`assets/align-plot-areas.js` lines up the plot areas of charts stacked in one
column of the page: Plotly fits each chart's margins to its own tick labels,
so it gives every chart in a column the widest margins any of them needs
(automargin still widens a margin a chart's labels need, so none is clipped).
Dash serves the `assets/` folder, scripts included, automatically.

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
  A `secondary_axis` shows the same data in other units on a y-axis on the
  plot area's right edge, its labels in the right margin.
- A figure's title sits at its top, its legend under the title and its plot
  area under the legend: the title reserves its band of the top margin
  (`title.automargin`) and the legend pushes the margin out below it, so a
  legend that wraps onto more rows on a narrow chart grows down, never over
  the title. Plotly reserves the band only while it is deeper than
  `margin.t`, which the themes keep to 12px: a figure that sets a deeper top
  margin has its title drawn higher than other charts', and its legend placed
  regardless of its title.
- `Spatial2DPlotBuilder` and `Spatial3DPlotBuilder` plot paths with equal axis
  scaling; `EndEffectorSpatialPlots` produces the XZ, YZ, XY, and 3D views.
  They are imported from `datadash.builders.plot` and are not in `__all__`
  yet, so they may still change between minor versions.
- `PlotFigure3DAnimation` (`builders/figure.py`) animates trace constructors
  (`create_trace_constructor`, `builders/trace.py`), whose points, of shape
  (points, time, space), move from frame to frame. Given `faces`, triangles as
  indices into its points of shape (F, 3), a constructor is a mesh: it is drawn
  as a `Mesh3d`, styled from the theme's `mesh` trace defaults (its lighting)
  rather than the line and marker `default`, and after the first frame its
  frames carry only its vertices, in single precision.
- `PlotRegistry` (`plots/registry.py`) registers plot functions by id with a
  decorator, so an app can look plots up by name.

## Dashboard components

`datadash/dashboard/components.py` returns themed Dash components:

| Function | Returns |
| --- | --- |
| `create_dashboard_header(title)`, `create_dashboard_footer(text)` | Page header (defaults to the theme's `dashboard_title`) and footer |
| `create_tabs(children, id, value)`, `create_themed_tab(label, value)` | Tab bar (first tab selected unless `value` is given) and tabs (with the theme's tab icons). The tabs share each row equally, never narrower than a word of 14 characters, and wrap onto as few rows as they fit in, as evenly as they can (nine tabs: five and four); where fewer than three fit side by side, they keep to one row that scrolls, with the selected tab in view. The layout is `assets/dash-components.css`'s, so a theme's `tab` and `tab_container` styles leave out widths, `display`, `gap` and wrapping |
| `create_main_container(children)`, `create_body_container(children)`, `create_tab_layout(children)` | Page structure |
| `create_graph_component(graph_id, figure, width)` | A themed `dcc.Graph` in a `dbc.Col` of the given width |
| `construct_dash_table(table, table_id, max_width, cell_style, group_by)` | A read-only HTML table from a DataFrame, numbers right-aligned and floats rounded, with a sticky header. With `group_by`, a table per value of that column, captioned with it (theme style `table_section`), each scrolling on its own, all on one grid of columns (CSS subgrid), so the columns line up across them |
| `create_job_selector_dropdown(job_options, current_job_id)` | A single job dropdown |
| `create_parameter_filter_dropdowns(sweep_analyzer, current_job_id, formatter)` | One filter per swept parameter, in a row sized by `filter_row_class_name`; `formatter(param)` returns `(label, format_value)`, where `format_value(value)` renders an option's label (option values stay raw) |
| `create_filter_dropdown(dropdown_id, label, options, value)` | A filter: its label (ending in an ellipsis if the column is narrower, and in full as a tooltip) over a dropdown, as a `dbc.Col` its row sizes |
| `filter_row_class_name(count)` | The classes of a `dbc.Row` of `count` filters, which size its columns: one a row on phones, two on tablets, three on laptops, and on wide screens up to six, over as few rows as evenly as possible (fewer than four take a third of the row each). For rows of your own filters, and callbacks that hide some of them |
| `parameter_filter_container_style()` | The filter bar style, for callbacks that hide and show it |

`DashboardApp` (`dashboard/base.py`) is the base class for an app:
implement `initialize()` to build `self.app` (use `_create_app()`, which wires
the assets folder and page title), then call `run(debug, port, host)`.

## Development

```bash
uv sync                               # installs datadash and the dev tools from uv.lock
uv run playwright install chromium    # once: the browser the page scripts' tests drive
uv run pytest
uv run ruff check .
uv run pyright
```

Pyright, pinned in the `dev` group, type-checks `datadash/` and `tests/` in
standard mode, configured under `[tool.pyright]` in `pyproject.toml`. Plotly's
trace properties and Dash's component props are not annotated, so tests read
them from `to_plotly_json()`. Where a library's own annotations are wrong, the
line carries a `# pyright: ignore[<rule>]` with the reason.

The tests of the scripts in `assets/` serve a small dashboard and measure it in
headless Chromium; without Chromium they are skipped, except in CI.

CI (`.github/workflows/ci.yml`) runs all three against `uv.lock` on pushes to
`main` and on every pull request. Weekly, or when started by hand, it also runs
the tests (not ruff or pyright) against the newest dependency versions
`pyproject.toml` allows.

To work on datadash alongside an app that uses it, install your checkout into
the app's environment for the session: `uv pip install -e ../datadash` from the
app. The app's next `uv sync` restores its pinned tag.

## Releasing

Bump `version` in `pyproject.toml` and run `uv lock` (the lock file records
the version, and CI installs with `uv sync --locked`). Merge to `main`, then tag
the merge commit `v<version>` and push the tag. Apps upgrade by changing the
tag they pin.
