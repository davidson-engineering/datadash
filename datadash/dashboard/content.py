# # Dashboard Layout Definitions
# # Author : Matthew Davidson
# # 2023/01/23
# # Davidson Engineering Ltd. © 2023

# from ..builders.content import (
#     create_two_column_builder,
#     create_flex_builder,
#     create_single_column_builder,
#     create_table_builder,
#     create_centered_graph_builder,
#     create_three_column_builder,
#     create_split_window_layout,
# )
# from .components import create_graph_component, construct_dash_table


# def create_actuator_dynamics_tab(plots):
#     # Create main plot (large plot on left)
#     main_plot = create_graph_component(
#         "joint-torque-plot-combined",
#         plots.plot_torque_combined_virtualwork(),
#         style={"height": "900px"},
#     )

#     # Create grid plots (2x2 grid on right)
#     grid_plots = [
#         create_graph_component(
#             "joint-position-plot-combined",
#             plots.plot_joint_trajectories_position_combined(),
#         ),
#         create_graph_component(
#             "joint-velocity-plot-combined",
#             plots.plot_joint_trajectories_velocity_combined(),
#         ),
#         create_graph_component(
#             "joint-acceleration-plot-combined",
#             plots.plot_joint_trajectories_acceleration_combined(),
#         ),
#         create_graph_component(
#             "joint-jerk-plot-combined",
#             plots.plot_joint_trajectories_jerk_combined(),
#         ),
#     ]

#     # Use the convenience function to create the split window layout
#     return create_split_window_layout(
#         main_plot,
#         grid_plots,
#         left_width=5,  # Large main plot
#         right_width=5,  # Smaller grid area
#     )


# def create_end_effector_trajectory_spatial_tab(plots):
#     builder = create_two_column_builder()

#     return builder.create_layout(
#         [
#             [  # First row
#                 create_graph_component(
#                     "end_effector_trajectory_3d",
#                     plots.plot_end_effector_trajectory_spatial_3d(),
#                 ),
#                 create_graph_component(
#                     "end_effector_trajectory_spatial_xz",
#                     plots.plot_end_effector_trajectory_spatial_xz(),
#                 ),
#             ],
#             [  # Second row
#                 create_graph_component(
#                     "end_effector_trajectory_spatial_yz",
#                     plots.plot_end_effector_trajectory_spatial_yz(),
#                 ),
#                 create_graph_component(
#                     "end_effector_trajectory_spatial_xy",
#                     plots.plot_end_effector_trajectory_spatial_xy(),
#                 ),
#             ],
#         ]
#     )


# def create_joint_trajectories_tab(plots):
#     builder = create_two_column_builder()

#     return builder.create_layout(
#         [
#             [  # First row
#                 create_graph_component(
#                     "plot_timeseries_trajectory_position",
#                     plots.plot_joint_trajectories_position_combined(),
#                 ),
#                 create_graph_component(
#                     "plot_timeseries_trajectory_velocity",
#                     plots.plot_joint_trajectories_velocity_combined(),
#                 ),
#             ],
#             [  # Second row
#                 create_graph_component(
#                     "plot_timeseries_trajectory_acceleration",
#                     plots.plot_joint_trajectories_acceleration_combined(),
#                 ),
#                 create_graph_component(
#                     "plot_timeseries_trajectory_jerk",
#                     plots.plot_joint_trajectories_jerk_combined(),
#                 ),
#             ],
#         ]
#     )


# def create_end_effector_trajectory_tab(plots):
#     builder = create_two_column_builder()

#     return builder.create_layout(
#         [
#             [  # First row
#                 create_graph_component(
#                     "plot_end_effector_trajectory_position",
#                     plots.plot_end_effector_trajectory_position_combined(),
#                 ),
#                 create_graph_component(
#                     "plot_end_effector_trajectory_velocity",
#                     plots.plot_end_effector_trajectory_velocity_combined(),
#                 ),
#             ],
#             [  # Second row
#                 create_graph_component(
#                     "plot_end_effector_trajectory_acceleration",
#                     plots.plot_end_effector_trajectory_acceleration_combined(),
#                 ),
#                 create_graph_component(
#                     "plot_end_effector_trajectory_jerk",
#                     plots.plot_end_effector_trajectory_jerk_combined(),
#                 ),
#             ],
#         ]
#     )


# def create_actuator_loads_tab(plots):
#     builder = create_three_column_builder(
#         width=3, row_properties={"style": {"margin-bottom": "0rem"}}
#     )

#     plot_M_1, plot_M_2, plot_M_3 = plots.plot_loads_moments_combined_newtoneuler()
#     plot_F_1, plot_F_2, plot_F_3 = plots.plot_loads_forces_combined_newtoneuler()

#     return builder.create_layout(
#         [
#             [
#                 create_graph_component(
#                     "plot_loads_moments_combined_newtoneuler_M_1",
#                     plot_M_1,
#                 ),
#                 create_graph_component(
#                     "plot_loads_moments_combined_newtoneuler_M_2",
#                     plot_M_2,
#                 ),
#                 create_graph_component(
#                     "plot_loads_moments_combined_newtoneuler_M_3",
#                     plot_M_3,
#                 ),
#             ],
#             [
#                 create_graph_component(
#                     "plot_loads_forces_combined_newtoneuler_F_1",
#                     plot_F_1,
#                 ),
#                 create_graph_component(
#                     "plot_loads_forces_combined_newtoneuler_F_2",
#                     plot_F_2,
#                 ),
#                 create_graph_component(
#                     "plot_loads_forces_combined_newtoneuler_F_3",
#                     plot_F_3,
#                 ),
#             ],
#         ]
#     )


# def create_summary_specifications_tab(tables):
#     builder = create_table_builder(width=6, padding="10%")
#     return builder.create_layout([[construct_dash_table(tables["robot_parameters"])]])


# def create_results_table_tab(tables):
#     builder = create_table_builder(width=6, padding="100px")
#     return builder.create_layout([[construct_dash_table(tables["results"])]])


# def create_robot_overview_tab(plots):
#     builder = create_centered_graph_builder(
#         row_properties={"justify": "center"},
#         column_properties={"width": 10, "style": {"text-align": "center"}},
#     )

#     graph_style = {
#         "margin": "0 auto",  # center horizontally
#         "width": "100%",  # responsive width
#         "max-width": "1400px",  # max width
#         "min-width": "600px",  # min width
#         "min-height": "800px",  # min height
#     }

#     return builder.create_overview_layout(
#         [
#             [
#                 create_graph_component(
#                     "robot-overview", plots.plot_robot_overview(), style=graph_style
#                 )
#             ]
#         ]
#     )
