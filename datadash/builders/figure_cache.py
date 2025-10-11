# Figure Cache Manager for BasePlotBuilder
# Author : Matthew Davidson
# 2025/01/23
# Davidson Engineering Ltd. © 2025

"""Simple Redis-based figure caching for datadash plot builders.

This is a standalone datadash component with no external dependencies.
Provides caching for complete Plotly figure objects, enabling efficient
data-only updates without rebuilding styling, layout, or themes.

Example:
    >>> import redis
    >>> from datadash.builders.figure_cache import FigureCacheManager
    >>> from datadash.builders.plot import BasicPlotBuilder
    >>>
    >>> cache = FigureCacheManager.from_redis()  # Auto-connect to localhost
    >>> builder = BasicPlotBuilder(cache_manager=cache)
    >>>
    >>> fig = builder.create_plot(x, y, plot_id="my_plot")  # Fast updates!
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict, Any
import logging
import pickle
import gzip

if TYPE_CHECKING:
    from plotly.graph_objs import Figure
    import redis

logger = logging.getLogger(__name__)


class FigureCacheManager:
    """Manages Redis caching for plot figures.

    Stores complete Plotly figure objects with all styling, layout, and theme
    information applied. Enables efficient updates by restoring figures from
    cache and updating only the trace data (in-place).

    Automatically connects to Redis on localhost by default. If connection fails,
    caching is gracefully disabled.

    Attributes:
        redis_client: Redis client instance
        enabled: Whether caching is enabled
        ttl: Default time-to-live for cached figures (seconds)
        namespace: Key namespace prefix
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        ttl: int = 7200,
        namespace: str = "datadash_figures",
    ):
        """Initialize figure cache manager.

        By default, connects to Redis at localhost:6379. If connection fails,
        caching is disabled gracefully (no errors raised).

        Args:
            host: Redis host (default: localhost)
            port: Redis port (default: 6379)
            db: Redis database number (default: 0)
            password: Redis password (default: None)
            ttl: Time-to-live in seconds (default: 7200 = 2 hours)
            namespace: Key namespace prefix (default: datadash_figures)


        """
        self.ttl = ttl
        self.namespace = namespace

        try:
            import redis as redis_module

            self.redis_client = redis_module.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            logger.info(f"FigureCacheManager: Connected to Redis at {host}:{port}")

        except Exception as e:
            logger.warning(f"FigureCacheManager: Failed to connect to Redis: {e}")
            logger.warning("FigureCacheManager: Caching disabled")
            self.redis_client = None
            self.enabled = False

    def _generate_key(
        self, plot_id: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate Redis key for a figure.

        Args:
            plot_id: Plot identifier
            params: Optional parameters to include in key for cache variants

        Returns:
            Redis key string
        """
        key = f"{self.namespace}:{plot_id}"

        if params:
            # Create deterministic hash of params for cache key
            import hashlib
            import json

            params_str = json.dumps(params, sort_keys=True)
            param_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
            key += f":{param_hash}"

        return key

    def get(
        self, plot_id: str, params: Optional[Dict[str, Any]] = None
    ) -> Optional["Figure"]:
        """Retrieve a cached figure.

        Args:
            plot_id: Plot identifier
            params: Optional parameters used to generate cache key

        Returns:
            Plotly Figure object if cached, None otherwise
        """
        if not self.enabled:
            return None

        try:
            key = self._generate_key(plot_id, params)
            cached_data = self.redis_client.get(key)

            if cached_data:
                # Decompress and unpickle the figure
                decompressed = gzip.decompress(cached_data)
                fig = pickle.loads(decompressed)
                logger.debug(f"FigureCacheManager: Cache hit for '{plot_id}'")
                return fig
            else:
                logger.debug(f"FigureCacheManager: Cache miss for '{plot_id}'")
                return None

        except Exception as e:
            logger.warning(f"FigureCacheManager: Failed to retrieve '{plot_id}': {e}")
            return None

    def set(
        self,
        plot_id: str,
        figure: "Figure",
        params: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store a figure in cache.

        Args:
            plot_id: Plot identifier
            figure: Plotly Figure object to cache
            params: Optional parameters to include in cache key
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            return False

        try:
            key = self._generate_key(plot_id, params)

            # Pickle and compress the figure for efficient storage
            pickled = pickle.dumps(figure)
            compressed = gzip.compress(pickled)

            # Store with TTL
            effective_ttl = ttl if ttl is not None else self.ttl
            self.redis_client.set(key, compressed, ex=effective_ttl)

            logger.debug(
                f"FigureCacheManager: Cached '{plot_id}' (size: {len(compressed)} bytes)"
            )
            return True

        except Exception as e:
            logger.warning(f"FigureCacheManager: Failed to cache '{plot_id}': {e}")
            return False

    def invalidate(self, plot_id: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """Invalidate a cached figure.

        Args:
            plot_id: Plot identifier
            params: Optional parameters used in cache key

        Returns:
            True if deleted, False otherwise
        """
        if not self.enabled:
            return False

        try:
            key = self._generate_key(plot_id, params)
            deleted = self.redis_client.delete(key)

            if deleted:
                logger.debug(f"FigureCacheManager: Invalidated cache for '{plot_id}'")
            return bool(deleted)

        except Exception as e:
            logger.warning(f"FigureCacheManager: Failed to invalidate '{plot_id}': {e}")
            return False

    def clear_all(self) -> int:
        """Clear all cached figures in this namespace.

        Returns:
            Number of keys deleted
        """
        if not self.enabled:
            return 0

        try:
            pattern = f"{self.namespace}:*"
            keys = self.redis_client.keys(pattern)

            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"FigureCacheManager: Cleared {deleted} cached figures")
                return deleted
            return 0

        except Exception as e:
            logger.warning(f"FigureCacheManager: Failed to clear cache: {e}")
            return 0
