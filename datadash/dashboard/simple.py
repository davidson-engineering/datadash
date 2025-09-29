# # Simple Dashboard Implementation
# # Author : Matthew Davidson
# # 2023/01/23
# # Davidson Engineering Ltd. © 2023

# import dash
# from dash import html, dcc
# from dash.dependencies import Input, Output
# import dash_bootstrap_components as dbc

# from .base import DashboardApp
# from .content import (
#     create_actuator_dynamics_tab,
#     create_end_effector_trajectory_spatial_tab,
#     create_joint_trajectories_tab,
#     create_end_effector_trajectory_tab,
#     create_actuator_loads_tab,
#     create_summary_specifications_tab,
#     create_results_table_tab,
# )
# from .components import (
#     create_dashboard_header,
#     create_dashboard_footer,
#     create_themed_tab,
#     create_job_selector_dropdown,
# )


# class SimpleDashboard(DashboardApp):
#     def initialize(self):
#         from ..themes.manager import get_theme_manager

#         theme = get_theme_manager()

#         self.app = self._create_app()

#         layout_children = [
#             html.Link(href=theme.get_css_file(), rel="stylesheet"),
#             html.Link(href="/assets/fonts.css", rel="stylesheet"),  # Font declarations
#         ]

#         layout_children.append(dbc.Col(create_dashboard_header()))

#         # Add job selector if we have multiple jobs
#         if self.job_batch:
#             job_options = [
#                 {
#                     "label": f"{job_id}: {job_data.get('description', job_id)}",
#                     "value": job_id,
#                 }
#                 for job_id, job_data in self.job_batch.items()
#             ]
#             layout_children.append(
#                 create_job_selector_dropdown(job_options, self.current_job_id)
#             )

#         layout_children.extend(
#             [
#                 dcc.Tabs(
#                     id="dashboard",
#                     value="actuator-dynamics",
#                     children=[
#                         create_themed_tab("ACTUATOR DYNAMICS", "actuator-dynamics"),
#                         create_themed_tab("ACTUATOR LOADS", "actuator-loads"),
#                         create_themed_tab(
#                             "TRAJECTORY SPATIAL", "end-affector-trajectory-spatial"
#                         ),
#                         create_themed_tab("JOINT TRAJECTORIES", "joint-trajectories"),
#                         create_themed_tab(
#                             "END AFFECTOR TRAJECTORY", "end-affector-trajectory"
#                         ),
#                         create_themed_tab("SPECIFICATIONS", "summary-specifications"),
#                         create_themed_tab("RESULTS", "results-table"),
#                     ],
#                     style=theme.get_component_style("tab_container"),
#                 ),
#                 html.Div(
#                     id="dashboard-display-area-primary",
#                     style=theme.get_component_style("content_area"),
#                 ),
#                 create_dashboard_footer(),
#             ]
#         )

#         self.app.layout = html.Div(
#             children=layout_children, style=theme.get_component_style("main_container")
#         )

#         self._setup_tabs()
#         self._init_callbacks()
#         return self

#     def _setup_tabs(self):
#         self.tab_actuator_dynamics = create_actuator_dynamics_tab(self.plots)
#         self.tab_end_effector_trajectory_spatial = (
#             create_end_effector_trajectory_spatial_tab(self.plots)
#         )
#         self.tab_joint_trajectories = create_joint_trajectories_tab(self.plots)
#         self.tab_end_effector_trajectory = create_end_effector_trajectory_tab(
#             self.plots
#         )
#         self.tab_actuator_loads = create_actuator_loads_tab(self.plots)
#         self.tab_summary_specifications = create_summary_specifications_tab(self.tables)
#         self.tab_results_table = create_results_table_tab(self.tables)

#     def _switch_job_data(self, job_id):
#         """Switch current job data and recreate tabs"""
#         if self.job_batch and job_id in self.job_batch:
#             self.current_job_id = job_id
#             self.plots = self.job_batch[job_id]["plots"]
#             self.tables = self.job_batch[job_id]["tables"]
#             self._setup_tabs()

#     def _init_callbacks(self):
#         # Always set up the main tab callback
#         @self.app.callback(
#             Output("dashboard-display-area-primary", "children"),
#             Input("dashboard", "value"),
#         )
#         def render_content(tab):
#             tab_mapping = {
#                 "actuator-dynamics": self.tab_actuator_dynamics,
#                 "end-affector-trajectory-spatial": self.tab_end_effector_trajectory_spatial,
#                 "joint-trajectories": self.tab_joint_trajectories,
#                 "end-affector-trajectory": self.tab_end_effector_trajectory,
#                 "actuator-loads": self.tab_actuator_loads,
#                 "summary-specifications": self.tab_summary_specifications,
#                 "results-table": self.tab_results_table,
#             }
#             return tab_mapping.get(tab)

#         # Add job selector callback if we have multiple jobs
#         if self.job_batch:

#             @self.app.callback(
#                 Output("dashboard", "value"),
#                 Input("job-selector", "value"),
#                 prevent_initial_call=True,
#             )
#             def handle_job_selection(selected_job_id):
#                 self._switch_job_data(selected_job_id)
#                 return "actuator-dynamics"  # Return to first tab when switching jobs
