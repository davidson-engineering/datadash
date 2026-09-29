# Datadash Figure Caching

Plot builders can cache complete Plotly figures in Redis. When a figure with
the same `plot_id` is requested again, the cached figure is restored and only
its trace data is replaced, skipping layout, styling, and theme work.

## Quick Start

```python
import numpy as np
from datadash.builders.plot import BasicPlotBuilder

builder = BasicPlotBuilder()  # connects to Redis at localhost:6379

x = np.linspace(0, 10, 100)
fig = builder.create_plot(x, np.sin(x), title="Plot", plot_id="my_plot")      # builds and caches
fig = builder.create_plot(x, 2 * np.sin(x), title="Plot", plot_id="my_plot")  # restores, swaps data
```

Caching only happens when a `plot_id` is given. Without Redis the builder logs
two warnings (`Failed to connect to Redis`, `Caching disabled`) and builds every
figure normally.

## Configuration

Every `BasePlotBuilder` creates its own `FigureCacheManager` with default
settings. To use other settings, replace it:

```python
from datadash.builders.figure_cache import FigureCacheManager

builder = BasicPlotBuilder()
builder.cache_manager = FigureCacheManager(
    host="redis.example.com",
    port=6380,
    db=0,
    password=None,
    ttl=3600,               # seconds, default 7200
    namespace="my_plots",   # default "datadash_figures"
)
```

To skip the cache for one call, pass `use_cache=False` to `create_plot`.

Keys are `<namespace>:v<FIGURE_CACHE_VERSION>:<plot_id>`.
`FIGURE_CACHE_VERSION` (in `builders/figure_cache.py`) is bumped whenever
figure layout or styling changes, so figures cached by older code are never
served.

## Is it faster?

Measure for your figures. A hit still unpickles and decompresses the whole
figure and fetches it over the network, so for small figures it can be slower
than rebuilding: with a 100-point line plot and Redis on localhost, a hit took
about 20 ms and a warm rebuild without Redis about 6 ms. Check before relying
on it for speed.

## Cache lifetime

Importing `datadash.builders.plot` calls `FigureCacheManager().clear_all()`,
which deletes every figure in the default namespace. The cache therefore only
pays off within one process: repeated `create_plot` calls with the same
`plot_id`, such as animation frames or a dashboard redrawing a plot for new
data. A fresh process always starts empty.

## How It Works

1. First call with a `plot_id`: builds the complete figure and caches it
2. Later calls with the same `plot_id`: restores it from the cache and
   replaces only the trace x/y data (and the axis ranges that depend on it)
3. Layout, theme, and trace styling come from the cached figure

The cache key is the `plot_id` alone, not the data or the other arguments, so
the data structure must stay the same between calls:

- Same number of traces
- Same headers
- Only the x/y values change

If the structure changes, invalidate the entry first (a mismatch in trace count
is logged as a warning):

```python
builder.cache_manager.invalidate("my_plot")
```

## Cache Operations

```python
cache = builder.cache_manager

cache.invalidate("plot_id")   # remove one figure; returns True if it existed
cache.clear_all()             # remove every figure in this namespace; returns the count
cache.enabled                 # False when Redis was unreachable at construction
```

## Requirements

- A Redis server (optional; see Quick Start)
- The `redis` Python package

The robot-dashboard repository runs a Redis container with
`docker compose -f docker-compose.redis.yml up -d`, and its
`examples/datadash_simple_cache.py` demonstrates the cache.
