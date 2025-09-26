# DeltaBot Dashboard Implementation
# Author : Matthew Davidson
# 2023/01/23
# Davidson Engineering Ltd. © 2023

from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from .simple import SimpleDashboard
from .content import create_robot_overview_tab
from .components import (
    create_dashboard_header,
    create_dashboard_footer,
    create_themed_tab,
    create_job_selector_dropdown,
    create_main_container,
    create_tabs,
    create_body_container,
)


class DeltaBotDashboard(SimpleDashboard):
    def __init__(
        self,
        plots=None,
        robot_geometry: dict = None,
        tables: dict = None,
        job_batch=None,
    ):
        super().__init__(plots, tables, job_batch)
        if job_batch:
            # In multi-job mode, get robot geometry from first job
            first_job_data = next(iter(job_batch.values()))
            self.robot_geometry = first_job_data.get("robot_geometry", {})
        else:
            self.robot_geometry = robot_geometry or {}

    def initialize(self):
        from ..themes.manager import get_theme_manager

        theme = get_theme_manager()

        self.app = self._create_app()
        self._setup_tabs()

        tab_children = [
            create_themed_tab("OVERVIEW", "robot-overview"),
            create_themed_tab("ACTUATOR DYNAMICS", "actuator-dynamics"),
            create_themed_tab("ACTUATOR LOADS", "actuator-loads"),
            create_themed_tab("TRAJECTORY SPATIAL", "end-affector-trajectory-spatial"),
            create_themed_tab("JOINT TRAJECTORIES", "joint-trajectories"),
            create_themed_tab("END AFFECTOR TRAJECTORY", "end-affector-trajectory"),
            create_themed_tab("SPECIFICATIONS", "summary-specifications"),
        ]

        layout_children = [
            # html.Link(href=theme.get_css_file(), rel="stylesheet"),
            # html.Link(href="/assets/fonts.css", rel="stylesheet"),  # Font declarations
            create_dashboard_header(theme.get_dashboard_title()),
            create_tabs(tab_children),
            create_main_container(),
            create_dashboard_footer(),
        ]

        self.app.layout = create_body_container(layout_children)

        self._init_callbacks()

        return self

    def _setup_tabs(self):
        super()._setup_tabs()
        self.tab_robot_overview = create_robot_overview_tab(self.plots)

    def _init_callbacks(self):
        @self.app.callback(
            Output("dashboard-display-area-primary", "children"),
            Input("dashboard", "value"),
        )
        def render_content(tab):
            tab_mapping = {
                "robot-overview": self.tab_robot_overview,
                "actuator-dynamics": self.tab_actuator_dynamics,
                "end-affector-trajectory-spatial": self.tab_end_effector_trajectory_spatial,
                "joint-trajectories": self.tab_joint_trajectories,
                "end-affector-trajectory": self.tab_end_effector_trajectory,
                "actuator-loads": self.tab_actuator_loads,
                "summary-specifications": self.tab_summary_specifications,
            }
            return tab_mapping.get(tab)
