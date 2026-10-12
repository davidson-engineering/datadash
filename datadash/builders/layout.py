# Unified Builder Classes for Visualization Components
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

import logging
from typing import Any

from ..themes.manager import get_theme_manager
from .trace import get_plot_range

# =============================================================================
# PLOT LAYOUT BUILDERS - For individual plot layouts
# =============================================================================


class PlotLayoutBuilder:
    """Builder that supports multiple plot layout modes: basic, combined, subplots"""

    _excluded_keys = {
        "title",
        "x",
        "y",
        "x_title",
        "y_title",
        "x_units",
        "y_units",
        "x_range",
        "y_range",
        "yrange",
        "units",
        "mode",
        "hover_mode",
    }

    @classmethod
    def get_trace_margin(cls):
        """Get trace margin from theme manager"""
        theme = get_theme_manager()
        return theme.get_settings().get("trace_margin", 0)

    @classmethod
    def create_plot_layout(cls, **kwargs):
        core = {k: kwargs.pop(k, None) for k in cls._excluded_keys}
        core["title"] = core.get("title", "")
        core["mode"] = core.get("mode", "basic")  # default
        core["hover_mode"] = core.get("hover_mode", "closest")
        layout = cls._create_plot_layout(**core)
        return cls._update_layout(layout, kwargs)

    @classmethod
    def _update_layout(cls, layout, kwargs):
        """Update layout with additional kwargs and log changes"""
        for key, value in kwargs.items():
            if key in cls._excluded_keys:
                logging.debug(f"{cls.__name__}: Ignored excluded key '{key}'")
                continue
            if key in layout:
                logging.debug(
                    f"{cls.__name__}: Changed '{key}': {layout[key]} -> {value}"
                )
            else:
                logging.debug(f"{cls.__name__}: Added '{key}': {value}")
            layout[key] = value
        return layout

    @staticmethod
    def _data_ranges(x, y, margin):
        """The x and y data's ranges, y's widened by the margin."""
        if x is None or y is None:
            raise ValueError("x and y are needed to compute the plot's ranges")
        return get_plot_range([x, y], margins=[0, margin])

    @classmethod
    def _create_plot_layout(
        cls,
        *,
        mode="basic",
        x=None,
        y=None,
        title="",
        x_title="",
        y_title="",
        x_units=None,
        y_units=None,
        x_range=None,
        y_range=None,
        yrange=None,
        units=None,
        hover_mode="closest",
        **_,
    ):
        margin = cls.get_trace_margin()

        # === BASIC ===
        if mode == "basic":
            layout: dict[str, Any] = {
                "title_text": f"{title} [{y_units}]" if y_units else title,
                "xaxis_title": f"{x_title} [{x_units}]" if x_units else x_title,
                "yaxis_title": f"{y_title} [{y_units}]" if y_units else y_title,
            }
            if x_range is not None:
                layout["xaxis"] = {"range": x_range}
            if y_range is not None:
                layout["yaxis"] = {"range": y_range}

            layout["hovermode"] = hover_mode
            return layout

        # === COMBINED ===
        if mode == "combined":
            if x_range is None or y_range is None:
                _x_range, _y_range = cls._data_ranges(x, y, margin)
                if x_range is None:
                    x_range = _x_range
                if y_range is None:
                    y_range = _y_range
            layout = {
                "title_text": f"{title}",
                # "title_text": f"{title} [{y_units}]" if y_units else title,
                "xaxis": {"range": x_range, "zeroline": False},
                "xaxis_title": f"{x_title} [{x_units}]" if x_units else x_title,
                "yaxis": {"range": y_range},
                "yaxis_title": f"{y_title} [{y_units}]" if y_units else y_title,
            }

            layout["hovermode"] = hover_mode
            return layout

        # === SUBPLOTS ===
        if mode == "subplots":
            if yrange is None:
                _, yrange = cls._data_ranges(x, y, margin)
            layout = {
                "title_text": f"{title} [{units}]" if units else title,
                "xaxis3_title": "Time [s]",
                "xaxis": {"zeroline": False},
                "yaxis": {"range": yrange},
                "yaxis2": {"range": yrange},
                "yaxis3": {"range": yrange},
            }

            layout["hovermode"] = hover_mode if hover_mode != "closest" else "x"
            return layout

        raise ValueError(
            f"Unknown mode '{mode}', expected 'basic', 'combined' or 'subplots'"
        )
