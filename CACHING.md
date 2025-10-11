# Datadash Figure Caching

Simple Redis-based caching for `BasePlotBuilder` to speed up plot rendering.

## Quick Start

```python
import numpy as np
from datadash.builders.plot import BasicPlotBuilder
from datadash.builders.figure_cache import FigureCacheManager

# Auto-connects to localhost:6379
cache = FigureCacheManager()
builder = BasicPlotBuilder(cache_manager=cache)

# First call: builds and caches (~500ms)
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = builder.create_plot(x, y, title="Plot", plot_id="my_plot")

# Second call: updates from cache (~50ms - 10x faster!)
y_new = np.sin(x) * 2
fig = builder.create_plot(x, y_new, title="Plot", plot_id="my_plot")
```

**That's it!** Layout, styling, and themes are preserved. Only data is updated.

## Configuration

```python
# Default (localhost:6379)
cache = FigureCacheManager()

# Custom Redis server
cache = FigureCacheManager(host='redis.example.com', port=6380)

# Custom TTL and namespace
cache = FigureCacheManager(ttl=3600, namespace="my_plots")

# Explicitly disable caching
cache = FigureCacheManager(redis_client=None)

# Use existing Redis client
import redis
my_client = redis.Redis(...)
cache = FigureCacheManager(redis_client=my_client)
```

## How It Works

1. **First call** with a `plot_id`: Builds complete figure and caches it
2. **Subsequent calls** with same `plot_id`: Restores from cache, updates only x/y/z data
3. **Preserved**: Layout, themes, styling, trace properties
4. **Updated**: Only trace data arrays (x, y, z)

## Important

**Data structure must remain constant**:
- Same number of traces
- Same headers/names
- Only x/y/z values change

If structure changes, invalidate cache first:
```python
cache.invalidate("my_plot")
```

## Cache Operations

```python
# Invalidate specific plot
cache.invalidate("plot_id")

# Clear all cached figures
cache.clear_all()

# Check if enabled
if cache.enabled:
    print("Caching is active")
```

## Without Caching

Just don't provide a `cache_manager`:

```python
builder = BasicPlotBuilder()  # No caching
fig = builder.create_plot(x, y, title="Plot")
```

## Performance

- **Without cache**: ~500ms per render
- **First call** (cache miss): ~500ms (builds + caches)
- **Update** (cache hit): **~50ms** (10x faster!)

Best for:
- Real-time data updates
- Animations
- Streaming sensors
- Any scenario where plot structure is fixed but data changes

## Requirements

- Redis server running (auto-connects to localhost:6379)
- `redis` Python package
- Gracefully falls back if Redis unavailable

## Example

See [examples/datadash_simple_cache.py](../../examples/datadash_simple_cache.py)
