from dash import html
import dash_bootstrap_components as dbc
from mergedeep import merge


class ContentBuilder:
    """Unified content builder that places content in configurable row/column structure"""

    def __init__(self, row_properties=None, column_properties=None):
        """
        Initialize ContentBuilder with optional row and column properties

        Args:
            row_properties: Dict or list of dicts for row properties (applied to all rows or per-row)
            column_properties: Dict or list of dicts for column properties (applied to all columns or per-column)
        """
        self.row_properties = row_properties or {}
        self.column_properties = column_properties or {}

    def create_layout(self, content_grid):
        """
        Create layout from n×m content grid

        Args:
            content_grid: List of lists containing dash components for each cell
                         Each cell should be a dash component (html.Div, dcc.Graph, etc.)
                         or None for empty cells

        Returns:
            html.Div containing the structured layout
        """
        rows = []

        for row_idx, row_content in enumerate(content_grid):
            columns = []

            # Get row properties (per-row or global)
            if isinstance(self.row_properties, list):
                current_row_props = (
                    self.row_properties[row_idx]
                    if row_idx < len(self.row_properties)
                    else {}
                )
            else:
                current_row_props = self.row_properties

            for col_idx, cell_content in enumerate(row_content):
                # Skip None/empty cells
                if cell_content is None:
                    continue

                # Get column properties (per-column or global)
                if isinstance(self.column_properties, list):
                    current_col_props = (
                        self.column_properties[col_idx]
                        if col_idx < len(self.column_properties)
                        else {}
                    )
                else:
                    current_col_props = self.column_properties.copy()

                # Cell content should already be a dash component
                columns.append(dbc.Col([cell_content], **current_col_props))

            # Create row if it has content
            if columns:
                rows.append(dbc.Row(columns, **current_row_props))

        return html.Div(rows)

    def create_single_row(self, content_list):
        """Convenience method for creating a single row layout"""
        return self.create_layout([content_list])

    def create_single_column(self, content_list):
        """Convenience method for creating a single column layout"""
        return self.create_layout([[item] for item in content_list])

    def create_grid(self, content_grid, rows, cols):
        """
        Convenience method for creating a regular grid layout

        Args:
            content_grid: Flat list of content items
            rows: Number of rows
            cols: Number of columns
        """
        grid = []
        for i in range(rows):
            row = []
            for j in range(cols):
                idx = i * cols + j
                if idx < len(content_grid):
                    row.append(content_grid[idx])
                else:
                    row.append(None)
            grid.append(row)
        return self.create_layout(grid)


# =============================================================================
# CONVENIENCE FACTORIES - Replicate old builder patterns with ContentBuilder
# =============================================================================


def merge_properties(default, overrides):
    """Merge two dictionaries, giving precedence to the overrides."""
    return merge(default, overrides)


def create_three_column_builder(width: int = 4, **kwargs):
    """Factory for three-column centered dashboard layouts"""
    default_column_props = {
        "width": width,  # Bootstrap component prop - each column takes one-third the width
    }
    # Center the entire three-column row on the page
    default_row_props = {
        "justify": "center",  # Bootstrap component prop - center the columns within the row
        "style": {
            "margin": "0 auto",
        },  # CSS for additional page centering
    }
    row_properties = merge_properties(
        default_row_props, kwargs.get("row_properties", {})
    )
    column_properties = merge_properties(
        default_column_props, kwargs.get("column_properties", {})
    )

    return ContentBuilder(
        row_properties=row_properties,
        column_properties=column_properties,
    )


def create_two_column_builder(**kwargs):
    """Factory for two-column centered dashboard layouts (replaces TwoColumnDashboardBuilder)"""
    default_column_props = {
        "width": 4,  # Bootstrap component prop - each column takes half the width
    }
    # Center the entire two-column row on the page
    default_row_props = {
        "justify": "center",  # Bootstrap component prop - center the columns within the row
        "style": {
            "margin": "0 auto",
        },  # CSS for additional page centering
    }
    row_properties = merge_properties(
        default_row_props, kwargs.get("row_properties", {})
    )
    column_properties = merge_properties(
        default_column_props, kwargs.get("column_properties", {})
    )

    return ContentBuilder(
        row_properties=row_properties,
        column_properties=column_properties,
    )


def create_single_column_builder(width=12, **kwargs):
    """Factory for single full-width column layouts (replaces SingleColumnDashboardBuilder)"""
    default_column_props = {
        "width": width,  # Full width column
        "style": {"text-align": "center"},
    }
    default_row_props = {"justify": "center", "style": {"margin-bottom": "5rem"}}

    row_properties = merge_properties(
        default_row_props, kwargs.get("row_properties", {})
    )
    column_properties = merge_properties(
        default_column_props, kwargs.get("column_properties", {})
    )

    return ContentBuilder(
        row_properties=row_properties,
        column_properties=column_properties,
    )


def create_centered_graph_builder(width=12, **kwargs):
    """Factory for centered single graph layouts (replaces CenteredGraphDashboardBuilder)"""
    default_column_props = {"width": width, "style": {"text-align": "center"}}
    default_row_props = {"style": {"justify-content": "center"}}

    row_properties = merge_properties(
        default_row_props, kwargs.get("row_properties", {})
    )
    column_properties = merge_properties(
        default_column_props, kwargs.get("column_properties", {})
    )

    return ContentBuilder(
        row_properties=row_properties,
        column_properties=column_properties,
    )


def create_table_builder(width=6, padding="10%", **kwargs):
    """Factory for table-based layouts (replaces TableDashboardBuilder)"""

    return create_single_column_builder()


def create_flex_builder(flex_sections=None, **kwargs):
    """
    Factory for flex-based layouts (replaces FlexDashboardBuilder)

    Args:
        flex_sections: List of flex-basis values for sections
    """
    if flex_sections:
        # Create flex-specific column properties
        flex_column_props = []
        for flex_basis in flex_sections:
            flex_column_props.append(
                {"style": {"flex-basis": flex_basis, "justify-content": "right"}}
            )
        column_properties = flex_column_props
    else:
        column_properties = kwargs.get("column_properties", {})

    return ContentBuilder(
        row_properties=kwargs.get("row_properties", {}),
        column_properties=column_properties,
    )


def create_split_window_builder(**kwargs):
    """
    Factory for split window layout: large plot on left, 2x2 grid on right

    Args:
        main_width: Width of the main (left) plot column (default: 8)
        grid_width: Width of the grid (right) column (default: 4)
        **kwargs: Additional customization options
    """
    main_width = kwargs.get("main_width", 8)
    grid_width = kwargs.get("grid_width", 4)

    # Properties for the main split row (large plot + grid column)
    default_row_props = {
        "justify": "center",  # Center the columns within the row
        "style": {
            "margin": "0 auto",  # Center the row on the page
            "margin-bottom": "1rem",
        },
    }

    # Column properties for the main split
    main_column_props = [
        {"width": main_width},  # Large plot column
        {"width": grid_width},  # Grid column
    ]

    row_properties = merge_properties(
        default_row_props, kwargs.get("row_properties", {})
    )

    return ContentBuilder(
        row_properties=row_properties,
        column_properties=main_column_props,
    )


def create_split_window_layout(main_plot, grid_plots, **kwargs):
    """
    Convenience function to create a split window layout with a main plot and 2x2 grid

    Args:
        main_plot: The main plot component for the left side
        grid_plots: List of up to 4 plot components for the 2x2 grid on the right
                   Should be ordered: [top-left, top-right, bottom-left, bottom-right]
        **kwargs: Additional customization options

    Returns:
        html.Div containing the split window layout
    """
    # Create the 2x2 grid from the grid plots
    grid_builder = ContentBuilder(
        row_properties={"style": {"margin-bottom": "0.5rem"}},
        column_properties={"width": 6},  # Each grid cell takes half the grid column
    )

    # Ensure we have exactly 4 plots, padding with None if necessary
    padded_grid_plots = list(grid_plots) + [None] * (4 - len(grid_plots))

    # Create 2x2 grid structure
    grid_content = [
        [padded_grid_plots[0], padded_grid_plots[1]],  # Top row
        [padded_grid_plots[2], padded_grid_plots[3]],  # Bottom row
    ]

    grid_layout = grid_builder.create_layout(grid_content)

    # Create the main split layout
    split_builder = create_split_window_builder(**kwargs)

    # Create the split layout with main plot and grid
    split_content = [[main_plot, grid_layout]]

    return split_builder.create_layout(split_content)
