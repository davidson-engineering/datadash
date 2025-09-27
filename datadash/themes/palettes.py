# Color Palette System for Trace Coloring
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from __future__ import annotations
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.colors as mcolors


class ColorPalette(ABC):
    """Abstract base class for color palettes."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_colors(self, count: int) -> Tuple[str, ...]:
        """Generate exactly count colors as a tuple.

        Args:
            count: Number of colors needed

        Returns:
            Tuple of color strings (hex format)
        """
        pass


class StaticPalette(ColorPalette):
    """A static palette with predefined colors that uses HSV interpolation."""

    def __init__(self, name: str, colors: List[str]):
        super().__init__(name)
        self.colors = colors

    def generate_colors(self, count: int) -> Tuple[str, ...]:
        """Generate colors by interpolating across all defined palette colors."""
        if count == 0:
            return tuple()

        if not self.colors:
            return ("#000000",) * count

        if count == 1:
            return (self.colors[0],)

        if len(self.colors) == 1:
            return (self.colors[0],) * count

        # If we have exactly the same number of colors as requested, return them directly
        if count == len(self.colors):
            return tuple(self.colors)

        # If we have fewer colors than requested, interpolate between all palette colors
        if count > len(self.colors):
            return self._interpolate_across_palette(count)

        # If we have more colors than requested, sample evenly from the palette
        if count < len(self.colors):
            return self._sample_from_palette(count)

    def _interpolate_hsv(self, start_color: str, end_color: str, count: int) -> Tuple[str, ...]:
        """Interpolate between start and end colors in HSV space."""
        if count == 1:
            return (start_color,)

        # Convert to RGB first, then to HSV
        start_rgb = mcolors.to_rgb(start_color)
        end_rgb = mcolors.to_rgb(end_color)

        start_hsv = mcolors.rgb_to_hsv(start_rgb)
        end_hsv = mcolors.rgb_to_hsv(end_rgb)

        colors = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0

            # Interpolate hue with wraparound consideration
            h_interpolated = self._interpolate_hue(start_hsv[0], end_hsv[0], t)

            # Linear interpolation for saturation and value
            s_interpolated = start_hsv[1] + (end_hsv[1] - start_hsv[1]) * t
            v_interpolated = start_hsv[2] + (end_hsv[2] - start_hsv[2]) * t

            # Convert back to RGB then hex
            interpolated_hsv = np.array([h_interpolated, s_interpolated, v_interpolated])
            interpolated_rgb = mcolors.hsv_to_rgb(interpolated_hsv)
            hex_color = mcolors.to_hex(interpolated_rgb)
            colors.append(hex_color)

        return tuple(colors)

    def _interpolate_hue(self, start_hue: float, end_hue: float, t: float) -> float:
        """Interpolate hue values taking the shortest path around the color wheel."""
        # Ensure hues are in [0, 1) range (matplotlib uses 0-1 for HSV)
        start_hue = start_hue % 1.0
        end_hue = end_hue % 1.0

        # Calculate both possible paths around the color wheel
        direct_diff = end_hue - start_hue
        wraparound_diff = direct_diff - 1.0 if direct_diff > 0 else direct_diff + 1.0

        # Choose the shorter path
        if abs(direct_diff) <= abs(wraparound_diff):
            interpolated = start_hue + direct_diff * t
        else:
            interpolated = start_hue + wraparound_diff * t

        # Ensure result is in [0, 1) range
        return interpolated % 1.0

    def _interpolate_across_palette(self, count: int) -> Tuple[str, ...]:
        """Interpolate across all colors in the palette to generate count colors."""
        if count <= len(self.colors):
            return tuple(self.colors[:count])

        # Create segments between consecutive palette colors
        segments = len(self.colors) - 1

        result_colors = []

        for i in range(count):
            # Calculate which segment this color falls into
            position = i / (count - 1) * segments  # position in [0, segments]
            segment_idx = int(position)  # which segment (0 to segments-1)
            segment_pos = position - segment_idx  # position within segment (0 to 1)

            # Handle edge case for the last color
            if segment_idx >= segments:
                result_colors.append(self.colors[-1])
                continue

            # Interpolate between the two colors in this segment
            start_color = self.colors[segment_idx]
            end_color = self.colors[segment_idx + 1]

            if segment_pos == 0.0:
                # Exact match to start color
                result_colors.append(start_color)
            else:
                # Use continuous interpolation within the segment
                interpolated_color = self._interpolate_hsv_continuous(
                    start_color, end_color, segment_pos
                )
                result_colors.append(interpolated_color)

        return tuple(result_colors)

    def _sample_from_palette(self, count: int) -> Tuple[str, ...]:
        """Sample evenly spaced colors from the palette."""
        if count >= len(self.colors):
            return tuple(self.colors)

        # Sample evenly across the palette
        indices = []
        for i in range(count):
            # Map i from [0, count-1] to [0, len(self.colors)-1]
            idx = i * (len(self.colors) - 1) / (count - 1)
            indices.append(int(round(idx)))

        return tuple(self.colors[i] for i in indices)

    def _interpolate_hsv_continuous(self, start_color: str, end_color: str, position: float) -> str:
        """Interpolate between two colors at a specific position (0.0 to 1.0)."""
        if position <= 0.0:
            return start_color
        if position >= 1.0:
            return end_color

        # Convert to RGB first, then to HSV
        start_rgb = mcolors.to_rgb(start_color)
        end_rgb = mcolors.to_rgb(end_color)

        start_hsv = mcolors.rgb_to_hsv(start_rgb)
        end_hsv = mcolors.rgb_to_hsv(end_rgb)

        # Interpolate hue with wraparound consideration
        h_interpolated = self._interpolate_hue(start_hsv[0], end_hsv[0], position)

        # Linear interpolation for saturation and value
        s_interpolated = start_hsv[1] + (end_hsv[1] - start_hsv[1]) * position
        v_interpolated = start_hsv[2] + (end_hsv[2] - start_hsv[2]) * position

        # Convert back to RGB then hex
        interpolated_hsv = np.array([h_interpolated, s_interpolated, v_interpolated])
        interpolated_rgb = mcolors.hsv_to_rgb(interpolated_hsv)
        return mcolors.to_hex(interpolated_rgb)


class GeneratedPalette(ColorPalette):
    """A palette that generates colors algorithmically using HSV space."""

    def __init__(self, name: str, generator_type: str = "categorical", **params):
        super().__init__(name)
        self.generator_type = generator_type
        self.params = params

    def generate_colors(self, count: int) -> Tuple[str, ...]:
        """Generate colors based on the generator type."""
        if count == 0:
            return tuple()

        if self.generator_type == "categorical":
            return self._generate_categorical(count)
        elif self.generator_type == "sequential":
            return self._generate_sequential(count)
        elif self.generator_type == "diverging":
            return self._generate_diverging(count)
        else:
            # Fallback to categorical
            return self._generate_categorical(count)

    def _generate_categorical(self, count: int) -> Tuple[str, ...]:
        """Generate categorical colors using golden angle for maximum distinction."""
        golden_angle = 137.508  # degrees
        base_hue = self.params.get("base_hue", 0) / 360.0  # Convert to [0,1]
        saturation = self.params.get("saturation", 75) / 100.0  # Convert to [0,1]
        value = self.params.get("value", 70) / 100.0  # Convert to [0,1]

        colors = []
        for i in range(count):
            hue = (base_hue + (i * golden_angle / 360.0)) % 1.0
            hsv = np.array([hue, saturation, value])
            rgb = mcolors.hsv_to_rgb(hsv)
            hex_color = mcolors.to_hex(rgb)
            colors.append(hex_color)

        return tuple(colors)

    def _generate_sequential(self, count: int) -> Tuple[str, ...]:
        """Generate sequential colors from light to dark (or custom range)."""
        base_hue = self.params.get("base_hue", 210) / 360.0  # Convert to [0,1], default blue
        saturation = self.params.get("saturation", 80) / 100.0  # Convert to [0,1]
        value_start = self.params.get("value_start", 90) / 100.0  # Convert to [0,1]
        value_end = self.params.get("value_end", 20) / 100.0  # Convert to [0,1]

        colors = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0
            value = value_start + (value_end - value_start) * t
            hsv = np.array([base_hue, saturation, value])
            rgb = mcolors.hsv_to_rgb(hsv)
            hex_color = mcolors.to_hex(rgb)
            colors.append(hex_color)

        return tuple(colors)

    def _generate_diverging(self, count: int) -> Tuple[str, ...]:
        """Generate diverging colors (two hues meeting in the middle)."""
        hue_start = self.params.get("hue_start", 0) / 360.0    # Red
        hue_end = self.params.get("hue_end", 240) / 360.0      # Blue
        saturation = self.params.get("saturation", 70) / 100.0
        value = self.params.get("value", 70) / 100.0

        colors = []
        mid_point = count / 2

        for i in range(count):
            if count == 1:
                # Single color - use middle of the two hues
                hue = (hue_start + hue_end) / 2
                current_saturation = saturation
            elif i < mid_point:
                # First half: use start hue, reduce saturation towards middle
                hue = hue_start
                saturation_factor = 1.0 - (0.3 * (mid_point - i) / mid_point)
                current_saturation = saturation * saturation_factor
            else:
                # Second half: use end hue, increase saturation from middle
                hue = hue_end
                saturation_factor = 0.7 + (0.3 * (i - mid_point) / (count - mid_point))
                current_saturation = saturation * saturation_factor

            hsv = np.array([hue, current_saturation, value])
            rgb = mcolors.hsv_to_rgb(hsv)
            hex_color = mcolors.to_hex(rgb)
            colors.append(hex_color)

        return tuple(colors)


class PaletteRegistry:
    """Registry for managing color palettes loaded from YAML configuration."""

    def __init__(self):
        self.palettes: Dict[str, ColorPalette] = {}

    def register_palette(self, palette: ColorPalette):
        """Register a palette in the registry."""
        self.palettes[palette.name] = palette

    def get_palette(self, name: str) -> Optional[ColorPalette]:
        """Get a palette by name."""
        return self.palettes.get(name)

    def list_palettes(self) -> List[str]:
        """Get list of available palette names."""
        return list(self.palettes.keys())

    def load_from_config(self, palette_config: Dict[str, Any]):
        """Load palettes from configuration dictionary.

        Args:
            palette_config: Dictionary with palette definitions
        """
        for palette_name, config in palette_config.items():
            palette_type = config.get("type", "static")

            if palette_type == "static":
                colors = config.get("colors", [])
                palette = StaticPalette(palette_name, colors)
            else:
                # Generated palette
                params = config.get("params", {})
                palette = GeneratedPalette(palette_name, palette_type, **params)

            self.register_palette(palette)

    def generate_colors(self, palette_name: str, count: int) -> Tuple[str, ...]:
        """Generate colors from a specific palette.

        Args:
            palette_name: Name of the palette to use
            count: Number of colors to generate

        Returns:
            Tuple of hex color strings
        """
        palette = self.get_palette(palette_name)
        if palette is None:
            # Fallback to a simple default
            return ("#007bff",) * count

        return palette.generate_colors(count)


# Global palette registry instance
_palette_registry = None


def get_palette_registry() -> PaletteRegistry:
    """Get the global palette registry instance."""
    global _palette_registry
    if _palette_registry is None:
        _palette_registry = PaletteRegistry()
    return _palette_registry


def set_palette_registry(registry: PaletteRegistry):
    """Set a custom palette registry."""
    global _palette_registry
    _palette_registry = registry