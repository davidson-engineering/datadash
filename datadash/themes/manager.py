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

# Global cache for loaded YAML files to avoid repeated file I/O
_yaml_file_cache = {}


class ThemeManager:
    def __init__(
        self,
        theme_name: str = "default",
        overrides_config: Dict[str, Any] = None,
        custom_config: Dict[str, Any] = None,  # Deprecated - use overrides_config
    ):
        self.theme_name = theme_name
        self.palette_registry = get_palette_registry()
        self._load_palettes()

        # Load theme configuration using three-level hierarchy
        self.theme_config = self._load_hierarchical_theme_config(
            theme_name, overrides_config or custom_config
        )

        # Cache for expensive operations
        self._plotly_theme_cache = None
        self._palette_color_cache = {}

    def _load_hierarchical_theme_config(
        self, theme_name: str, overrides_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Load theme configuration using three-level hierarchy.

        Priority order (highest to lowest):
        3. Overrides: User-specified !important overrides
        2. Custom: User-selected theme (e.g., futuristic.yaml)
        1. Default: Base theme configuration (default.yaml)

        Palette colors have the lowest priority and are overridden by any color
        specified in these configuration levels.
        """
        # Level 1: Load default theme (always loaded as base)
        default_config = self._load_theme_file("default")

        # Level 2: Load custom theme (user-selected theme)
        custom_config = {}
        if theme_name != "default":
            custom_config = self._load_theme_file(theme_name)

        # Level 3: Apply overrides (user-specified !important overrides)
        overrides = overrides_config or {}

        # Merge configurations with proper priority
        # Default -> Custom -> Overrides
        merged_config = merge({}, default_config, custom_config)

        # Apply overrides with special handling for !important markers
        if overrides:
            merged_config = self._apply_overrides(merged_config, overrides)

        return merged_config

    def _load_theme_file(self, theme_name: str) -> Dict[str, Any]:
        """Load a theme configuration file with caching."""
        global _yaml_file_cache

        # Check cache first
        if theme_name in _yaml_file_cache:
            return _yaml_file_cache[theme_name]

        theme_file = Path(__file__).parent / "config" / f"{theme_name}.yaml"
        if not theme_file.exists():
            # Fallback to legacy location
            theme_file = Path(__file__).parent / f"{theme_name}.yaml"
            if not theme_file.exists():
                if theme_name == "default":
                    raise FileNotFoundError(
                        f"Default theme file not found: {theme_file}"
                    )
                # Cache empty result to avoid repeated file checks
                _yaml_file_cache[theme_name] = {}
                return {}

        with open(theme_file, "r") as f:
            config = yaml.safe_load(f) or {}
            # Cache the loaded configuration
            _yaml_file_cache[theme_name] = config
            return config

    def _apply_overrides(
        self, base_config: Dict[str, Any], overrides: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply override configuration.

        All override values take absolute priority and cannot be overridden by
        palette colors or other theme settings.
        """
        result_config = base_config.copy()

        # Process overrides recursively - all overrides are important
        for key, value in overrides.items():
            if isinstance(value, dict):
                if key not in result_config:
                    result_config[key] = {}
                if isinstance(result_config[key], dict):
                    result_config[key] = self._apply_overrides(
                        result_config[key], value
                    )
                else:
                    # Override non-dict with dict
                    result_config[key] = self._apply_overrides({}, value)
            else:
                # All override values are important by definition
                result_config[key] = value

        return result_config

    def _load_theme_config(self, theme_name: str) -> Dict[str, Any]:
        """Legacy method - kept for backward compatibility."""
        return self._load_theme_file(theme_name)

    def merge_custom_config(
        self, custom_config: Dict[str, Any] | Path | str
    ) -> ThemeManager:
        """Merge a custom configuration as overrides into the current theme configuration.

        This method treats the provided config as Level 3 overrides.
        """
        if isinstance(custom_config, (Path, str)):
            with open(custom_config, "r") as f:
                custom_config = yaml.safe_load(f)
        if isinstance(custom_config, dict):
            # Apply as overrides using the new system
            self.theme_config = self._apply_overrides(self.theme_config, custom_config)
            return self
        else:
            raise ValueError(
                "custom_config must be a dictionary or a Path to a YAML file"
            )

    def add_overrides(self, overrides: Dict[str, Any]) -> ThemeManager:
        """Add override configuration to the theme.

        Args:
            overrides: Dictionary of override values. All values take highest priority
                      and cannot be overridden by palette colors or other theme settings.

        Returns:
            Self for method chaining
        """
        self.theme_config = self._apply_overrides(self.theme_config, overrides)
        return self

    def has_override_for_property(self, property_path: str) -> bool:
        """Check if a property has been overridden at any level.

        Args:
            property_path: Dot-separated path to the property (e.g., 'plotly.traces.x.line.color')

        Returns:
            True if the property has been explicitly set (not using palette defaults)
        """
        path_parts = property_path.split(".")
        current = self.theme_config

        for part in path_parts:
            if part in current and isinstance(current, dict):
                current = current[part]
            else:
                return False

        # Property exists and has a value
        return current is not None

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
        return self.styles.get(component, {}).copy()

    def get_plotly_theme(self) -> Dict[str, Any]:
        """Get Plotly-specific theme configuration with auto-generated numbered axes (cached)"""
        if self._plotly_theme_cache is not None:
            return self._plotly_theme_cache

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

        # Cache the result
        self._plotly_theme_cache = plotly_theme
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
        """Load color palettes from palettes.yaml with caching"""
        global _yaml_file_cache

        palette_key = "_palettes"
        if palette_key in _yaml_file_cache:
            # Use cached palette config
            palette_config = _yaml_file_cache[palette_key]
            if palette_config:
                self.palette_registry.load_from_config(palette_config)
            return

        palette_file = Path(__file__).parent / "config" / "palettes.yaml"
        if not palette_file.exists():
            # Fallback to legacy location
            palette_file = Path(__file__).parent / "palettes.yaml"

        if palette_file.exists():
            with open(palette_file, "r") as f:
                palette_config = yaml.safe_load(f)
                # Cache the palette config
                _yaml_file_cache[palette_key] = palette_config
                if palette_config:
                    self.palette_registry.load_from_config(palette_config)
        else:
            # Cache empty result
            _yaml_file_cache[palette_key] = None

    def get_palette_colors(self, palette_level: str, count: int) -> Tuple[str, ...]:
        """Get colors for a specific palette level (primary, secondary, tertiary) with caching.

        Args:
            palette_level: 'primary', 'secondary', or 'tertiary'
            count: Number of colors needed

        Returns:
            Tuple of hex color strings
        """
        # Check cache first
        cache_key = (palette_level, count)
        if cache_key in self._palette_color_cache:
            return self._palette_color_cache[cache_key]

        # Get palette assignments from theme config
        palette_assignments = self.theme_config.get("palettes", {})
        palette_name = palette_assignments.get(palette_level)

        if palette_name is None:
            # Fallback to default behavior
            colors = ("#007bff",) * count
        else:
            colors = self.palette_registry.generate_colors(palette_name, count)

        # Cache the result
        self._palette_color_cache[cache_key] = colors
        return colors

    def get_themed_trace_properties(
        self, trace_name: str, palette_level: str, trace_index: int, total_traces: int
    ) -> Dict[str, Any]:
        """Get complete themed properties for a trace including palette colors.

        Respects the priority hierarchy:
        1. Overrides (highest priority - cannot be overridden)
        2. Custom theme configuration colors
        3. Default theme configuration colors
        4. Palette colors (lowest priority - only applied if no color is specified)

        Args:
            trace_name: Name/identifier of the trace for theme lookup
            palette_level: Palette level (primary, secondary, tertiary)
            trace_index: Index of this trace in the group (0-based)
            total_traces: Total number of traces in the group

        Returns:
            Dictionary of complete themed properties including palette colors
        """
        # Get base trace theme (includes all three levels: default -> custom -> overrides)
        base_theme = self.get_trace_theme(trace_name) or {}

        # Only apply palette colors if no colors are explicitly defined in any theme level
        needs_line_color = (
            "line" not in base_theme
            or "color" not in base_theme.get("line", {})
            or base_theme.get("line", {}).get("color") is None
        )

        needs_marker_color = (
            "marker" not in base_theme
            or "color" not in base_theme.get("marker", {})
            or base_theme.get("marker", {}).get("color") is None
        )

        if needs_line_color or needs_marker_color:
            # Get palette colors for the full group
            palette_colors = self.get_palette_colors(palette_level, total_traces)

            # Apply the specific color for this trace
            if trace_index < len(palette_colors):
                color = palette_colors[trace_index]

                # Merge palette color with base theme
                themed_properties = base_theme.copy()

                # Apply color to line properties if needed
                if needs_line_color:
                    if "line" not in themed_properties:
                        themed_properties["line"] = {}
                    themed_properties["line"]["color"] = color

                # Apply color to marker properties if needed
                if needs_marker_color:
                    if "marker" not in themed_properties:
                        themed_properties["marker"] = {}
                    themed_properties["marker"]["color"] = color
                    themed_properties["marker"]["line_color"] = color
                return themed_properties

        # Return base theme without palette colors if they're not needed or are already specified
        return base_theme

    def assign_palette_colors_to_traces(
        self, trace_names: list[str], palette_level: str
    ) -> Dict[str, Dict[str, Any]]:
        """Assign palette colors to a group of traces and return themed properties.

        Args:
            trace_names: List of trace names/identifiers
            palette_level: Palette level to use for color assignment

        Returns:
            Dictionary mapping trace names to their complete themed properties
        """
        themed_traces = {}
        total_traces = len(trace_names)

        # Pre-generate all palette colors once for better performance
        palette_colors = self.get_palette_colors(palette_level, total_traces)

        for i, trace_name in enumerate(trace_names):
            themed_traces[trace_name] = self._get_themed_properties_with_precomputed_color(
                trace_name, palette_colors[i] if i < len(palette_colors) else palette_colors[0]
            )

        return themed_traces

    def _get_themed_properties_with_precomputed_color(
        self, trace_name: str, color: str
    ) -> Dict[str, Any]:
        """Get themed properties with a precomputed palette color.

        This is an optimized version that skips palette color generation.

        Args:
            trace_name: Name/identifier of the trace for theme lookup
            color: Precomputed palette color to use

        Returns:
            Dictionary of themed properties with the given color applied
        """
        # Get base trace theme (includes all three levels: default -> custom -> overrides)
        base_theme = self.get_trace_theme(trace_name) or {}

        # Only apply palette colors if no colors are explicitly defined in any theme level
        needs_line_color = (
            "line" not in base_theme
            or "color" not in base_theme.get("line", {})
            or base_theme.get("line", {}).get("color") is None
        )

        needs_marker_color = (
            "marker" not in base_theme
            or "color" not in base_theme.get("marker", {})
            or base_theme.get("marker", {}).get("color") is None
        )

        if needs_line_color or needs_marker_color:
            # Use the precomputed color
            themed_properties = base_theme.copy()

            # Apply color to line properties if needed
            if needs_line_color:
                if "line" not in themed_properties:
                    themed_properties["line"] = {}
                themed_properties["line"]["color"] = color

            # Apply color to marker properties if needed
            if needs_marker_color:
                if "marker" not in themed_properties:
                    themed_properties["marker"] = {}
                themed_properties["marker"]["color"] = color
                themed_properties["marker"]["line_color"] = color
            return themed_properties

        # Return base theme without palette colors if they're not needed or are already specified
        return base_theme

    def list_available_palettes(self) -> Dict[str, str]:
        """Get the current palette assignments for this theme.

        Returns:
            Dictionary mapping palette levels to palette names
        """
        return self.theme_config.get("palettes", {})

    def get_theme_hierarchy_info(self) -> Dict[str, Any]:
        """Get information about the theme hierarchy and configuration sources.

        Returns:
            Dictionary with information about theme sources and priority levels
        """
        return {
            "theme_name": self.theme_name,
            "hierarchy": [
                "1. Default (lowest priority)",
                "2. Custom theme",
                "3. Overrides (highest priority)",
                "4. Palette colors (applied only when no explicit colors are set)",
            ],
            "available_themes": self._list_available_themes(),
            "palette_assignments": self.list_available_palettes(),
            "override_info": "All override values take absolute priority over theme and palette colors",
        }

    def _list_available_themes(self) -> list[str]:
        """List available theme files."""
        config_dir = Path(__file__).parent / "config"
        themes = []

        if config_dir.exists():
            for theme_file in config_dir.glob("*.yaml"):
                themes.append(theme_file.stem)

        # Also check legacy location
        legacy_dir = Path(__file__).parent
        for theme_file in legacy_dir.glob("*.yaml"):
            if theme_file.name not in ["palettes.yaml"]:
                themes.append(theme_file.stem)

        return sorted(set(themes))

    def create_theme_override_template(self) -> Dict[str, Any]:
        """Create a template for theme overrides with examples.

        Returns:
            Dictionary template showing how to structure override configurations
        """
        return {
            "_info": "Theme Override Configuration Template",
            "_priority": "All values in this configuration have the highest priority and override theme settings",
            "_palette_info": "Palette colors are only applied when no explicit colors are set at any theme level",
            "plotly": {
                "layout": {
                    "paper_bgcolor": "#ffffff",
                    "plot_bgcolor": "#f8f9fa",
                    "font": {"size": 14, "color": "#212529"},
                },
                "traces": {
                    "x": {
                        "line": {
                            "color": "#dc3545",  # Overrides any theme or palette color
                            "width": 3,
                        }
                    },
                    "y": {
                        "line": {
                            "color": "#28a745",  # Overrides any theme or palette color
                            "width": 3,
                        }
                    },
                    "trajectory": {
                        "line": {
                            "width": 4  # Width override, color will come from theme/palette
                        }
                    },
                },
            },
            "colors": {"primary_bg": "#ffffff", "accent": "#007bff"},
            "settings": {"trace_margin": 0.2, "animation_frame_rate": 30},
        }


# Global theme manager instance
_theme_manager = None


def get_theme_manager(
    theme_name: str = None, overrides_config: Dict[str, Any] = None
) -> ThemeManager:
    """Get the global theme manager instance with optional overrides.

    Args:
        theme_name: Name of the theme to use (default: "default")
        overrides_config: Optional override configuration dictionary

    Returns:
        ThemeManager instance
    """
    global _theme_manager
    if _theme_manager is None or (
        theme_name and theme_name != _theme_manager.theme_name
    ):
        theme_name = theme_name or "default"
        _theme_manager = ThemeManager(theme_name, overrides_config)
    elif overrides_config:
        # Apply new overrides to existing theme manager
        _theme_manager.add_overrides(overrides_config)
    return _theme_manager


def set_theme(theme_name: str, overrides_config: Dict[str, Any] = None):
    """Switch to a different theme with optional overrides.

    Args:
        theme_name: Name of the theme to switch to
        overrides_config: Optional override configuration dictionary

    Returns:
        ThemeManager instance
    """
    global _theme_manager
    _theme_manager = ThemeManager(theme_name, overrides_config)
    return _theme_manager


def add_theme_overrides(overrides: Dict[str, Any]) -> ThemeManager:
    """Add overrides to the current theme manager.

    Args:
        overrides: Dictionary of override values

    Returns:
        Current ThemeManager instance with overrides applied
    """
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager("default", overrides)
    else:
        _theme_manager.add_overrides(overrides)
    return _theme_manager


def reset_theme_manager():
    """Reset the global theme manager to None, forcing reinitialization."""
    global _theme_manager
    _theme_manager = None
