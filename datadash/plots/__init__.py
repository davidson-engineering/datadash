# Datadash Plots
# Author : Matthew Davidson
# 2025/01/23
# Davidson Engineering Ltd. © 2025

"""Plot registry module for datadash.

This module provides a registry system for declarative plot definitions.
"""

from .registry import PlotMetadata, PlotRegistry, register_plot

__all__ = [
    "PlotRegistry",
    "register_plot",
    "PlotMetadata",
]
