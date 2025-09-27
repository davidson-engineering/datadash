# Theme Management System
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from __future__ import annotations
from typing import Dict, Any, Tuple
from pathlib import Path
from mergedeep import merge
import yaml

# Import palette system
from .palettes import get_palette_registry


class ThemeManager:
    def __init__(
        self, theme_name: str = "default", custom_config: Dict[str, Any] = None
    ):
        self.theme_name = theme_name
        self.palette_registry = get_palette_registry()
        self._load_palettes()
        self.theme_config = self._load_theme_config(theme_name)
        if custom_config:
            self.merge_custom_config(custom_config)

    def _load_theme_config(self, theme_name: str) -> Dict[str, Any]:
        theme_file = Path(__file__).parent / f"{theme_name}.yaml"
        if not theme_file.exists():
            theme_file = Path(__file__).parent / "default.yaml"

        with open(theme_file, "r") as f:
            return yaml.safe_load(f)

    def merge_custom_config(
        self, custom_config: Dict[str, Any] | Path | str
    ) -> ThemeManager:
        """Merge a custom configuration dictionary into the current theme configuration"""
        if isinstance(custom_config, (Path, str)):
            with open(custom_config, "r") as f:
                custom_config = yaml.safe_load(f)
        if isinstance(custom_config, dict):
            self.theme_config = merge({}, self.theme_config, custom_config)
            return self
        else:
            raise ValueError(
                "custom_config must be a dictionary or a Path to a YAML file"
            )

    @property
    def colors(self) -> Dict[str, str]:
        return self.theme_config.get("colors", {})

    @property
    def fonts(self) -> Dict[str, str]:
        return self.theme_config.get("fonts", {})

    @property
    def icons(self) -> Dict[str, str]:
        return self.theme_config.get("icons", {})

    @property
    def styles(self) -> Dict[str, Dict[str, Any]]:
        return self.theme_config.get("styles", {})

    def get_tab_icon(self, tab_value: str) -> str:
        return self.icons.get("tabs", {}).get(tab_value, "")

    def get_component_style(self, component: str) -> Dict[str, Any]:
        return self.styles.get(component, {})

    def get_plotly_theme(self) -> Dict[str, Any]:
        """Get Plotly-specific theme configuration with auto-generated numbered axes"""
        plotly_theme = self.theme_config.get("plotly", {}).copy()

        # Auto-generate numbered axis styles for subplots
        if "layout" in plotly_theme:
            layout = plotly_theme["layout"]
            layout_copy = layout.copy()

            # Generate numbered axis styles based on base xaxis/yaxis
            base_xaxis = layout.get("xaxis", {})
            base_yaxis = layout.get("yaxis", {})

            # Generate up to 10 numbered axes (should cover most use cases)
            for i in range(2, 11):
                if base_xaxis:
                    layout[f"xaxis{i}"] = base_xaxis.copy()
                if base_yaxis:
                    layout[f"yaxis{i}"] = base_yaxis.copy()

            # override layout with any additional customizations for x/y axes
            for key, value in layout_copy.items():
                if key.startswith("xaxis") or key.startswith("yaxis"):
                    layout[key].update(value)

        return plotly_theme

    def get_trace_theme(self, trace_name: str) -> Dict[str, Any]:
        """Get trace-specific theme configuration"""
        trace_themes = self.get_plotly_theme().get("traces", {})
        trace_theme = trace_themes.get(trace_name, None)
        if trace_theme:
            defaults = trace_themes.get("default", {})
            return merge({}, defaults, trace_theme)
        return None

    def get_settings(self) -> Dict[str, Any]:
        """Get settings from theme configuration"""
        return self.theme_config.get("settings", {})

    def get_css_file(self) -> str:
        return f"/assets/{self.theme_name}.css"

    def get_dashboard_title(self) -> str:
        return self.theme_config.get("dashboard_title", "Robot Simulator")

    def _load_palettes(self):
        """Load color palettes from palettes.yaml"""
        palette_file = Path(__file__).parent / "palettes.yaml"
        if palette_file.exists():
            with open(palette_file, "r") as f:
                palette_config = yaml.safe_load(f)
                if palette_config:
                    self.palette_registry.load_from_config(palette_config)

    def get_palette_colors(self, palette_level: str, count: int) -> Tuple[str, ...]:
        """Get colors for a specific palette level (primary, secondary, tertiary).

        Args:
            palette_level: 'primary', 'secondary', or 'tertiary'
            count: Number of colors needed

        Returns:
            Tuple of hex color strings
        """
        # Get palette assignments from theme config
        palette_assignments = self.theme_config.get("palettes", {})
        palette_name = palette_assignments.get(palette_level)

        if palette_name is None:
            # Fallback to default behavior
            return ("#007bff",) * count

        return self.palette_registry.generate_colors(palette_name, count)

    def get_themed_trace_properties(self, trace_name: str, palette_level: str, trace_index: int, total_traces: int) -> Dict[str, Any]:
        """Get complete themed properties for a trace including palette colors.

        Args:
            trace_name: Name/identifier of the trace for theme lookup
            palette_level: Palette level (primary, secondary, tertiary)
            trace_index: Index of this trace in the group (0-based)
            total_traces: Total number of traces in the group

        Returns:
            Dictionary of complete themed properties including palette colors
        """
        # Get base trace theme
        base_theme = self.get_trace_theme(trace_name) or {}

        # Get palette colors for the full group
        palette_colors = self.get_palette_colors(palette_level, total_traces)

        # Apply the specific color for this trace
        if trace_index < len(palette_colors):
            color = palette_colors[trace_index]

            # Merge palette color with base theme
            themed_properties = base_theme.copy()

            # Apply color to line and marker properties
            if "line" not in themed_properties:
                themed_properties["line"] = {}
            if "marker" not in themed_properties:
                themed_properties["marker"] = {}

            themed_properties["line"]["color"] = color
            themed_properties["marker"]["color"] = color

            return themed_properties

        # Fallback if index is out of range
        return base_theme

    def assign_palette_colors_to_traces(self, trace_names: list[str], palette_level: str) -> Dict[str, Dict[str, Any]]:
        """Assign palette colors to a group of traces and return themed properties.

        Args:
            trace_names: List of trace names/identifiers
            palette_level: Palette level to use for color assignment

        Returns:
            Dictionary mapping trace names to their complete themed properties
        """
        themed_traces = {}
        total_traces = len(trace_names)

        for i, trace_name in enumerate(trace_names):
            themed_traces[trace_name] = self.get_themed_trace_properties(
                trace_name, palette_level, i, total_traces
            )

        return themed_traces

    def list_available_palettes(self) -> Dict[str, str]:
        """Get the current palette assignments for this theme.

        Returns:
            Dictionary mapping palette levels to palette names
        """
        return self.theme_config.get("palettes", {})


# Global theme manager instance
_theme_manager = None


def get_theme_manager(theme_name: str = None) -> ThemeManager:
    global _theme_manager
    if _theme_manager is None or (
        theme_name and theme_name != _theme_manager.theme_name
    ):
        theme_name = theme_name or "default"
        _theme_manager = ThemeManager(theme_name)
    return _theme_manager


def set_theme(theme_name: str):
    """Switch to a different theme"""
    global _theme_manager
    _theme_manager = ThemeManager(theme_name)
    return _theme_manager
