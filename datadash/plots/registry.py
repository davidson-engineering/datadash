# Datadash Plots
# Author : Matthew Davidson
# 2025/01/23
# Davidson Engineering Ltd. © 2025

"""Plot registry for declarative plot definitions."""

from __future__ import annotations
from typing import Callable, Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PlotMetadata:
    """Metadata for a registered plot."""
    id: str
    title: str
    category: str
    description: Optional[str] = None
    requires: List[str] = None

    def __post_init__(self):
        if self.requires is None:
            self.requires = []


class PlotRegistry:
    """Registry for plot functions using decorator pattern."""

    _plots: Dict[str, Callable] = {}
    _metadata: Dict[str, PlotMetadata] = {}

    @classmethod
    def register(
        cls,
        id: str,
        title: str,
        category: str,
        description: Optional[str] = None,
        requires: Optional[List[str]] = None
    ) -> Callable:
        """Decorator to register a plot function.

        Args:
            id: Unique plot identifier
            title: Human-readable plot title
            category: Plot category (e.g., "dynamics", "trajectory")
            description: Optional plot description
            requires: Optional list of required data fields

        Returns:
            Decorator function

        Example:
            @PlotRegistry.register(
                id="actuator_torque",
                title="Actuator Torque",
                category="dynamics",
                requires=["dynamics.torque"]
            )
            def plot_actuator_torque(data: PlotData):
                return create_plot(...)
        """
        def decorator(func: Callable) -> Callable:
            if id in cls._plots:
                logger.warning(f"Overwriting existing plot registration: {id}")

            cls._plots[id] = func
            cls._metadata[id] = PlotMetadata(
                id=id,
                title=title,
                category=category,
                description=description or func.__doc__,
                requires=requires or []
            )

            logger.debug(f"Registered plot: {id} ({category})")
            return func

        return decorator

    @classmethod
    def get(cls, plot_id: str) -> Callable:
        """Get a registered plot function by ID.

        Args:
            plot_id: Plot identifier

        Returns:
            Plot function

        Raises:
            KeyError: If plot_id not registered
        """
        if plot_id not in cls._plots:
            raise KeyError(f"Plot not registered: {plot_id}")
        return cls._plots[plot_id]

    @classmethod
    def get_metadata(cls, plot_id: str) -> PlotMetadata:
        """Get metadata for a registered plot.

        Args:
            plot_id: Plot identifier

        Returns:
            Plot metadata

        Raises:
            KeyError: If plot_id not registered
        """
        if plot_id not in cls._metadata:
            raise KeyError(f"Plot not registered: {plot_id}")
        return cls._metadata[plot_id]

    @classmethod
    def list(cls) -> List[str]:
        """List all registered plot IDs.

        Returns:
            List of plot IDs
        """
        return list(cls._plots.keys())

    @classmethod
    def list_by_category(cls, category: str) -> List[str]:
        """List all plots in a category.

        Args:
            category: Category name

        Returns:
            List of plot IDs in the category
        """
        return [
            plot_id for plot_id, meta in cls._metadata.items()
            if meta.category == category
        ]

    @classmethod
    def categories(cls) -> List[str]:
        """Get all registered categories.

        Returns:
            List of unique category names
        """
        return list(set(meta.category for meta in cls._metadata.values()))

    @classmethod
    def clear(cls) -> None:
        """Clear all registered plots (primarily for testing)."""
        cls._plots.clear()
        cls._metadata.clear()


# Convenience decorator
register_plot = PlotRegistry.register
