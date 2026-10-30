from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    share = Path(get_package_share_directory('isaac_manipulation_bringup'))
    panda_share = Path(get_package_share_directory('moveit_resources_panda_moveit_config'))
    config = (
        MoveItConfigsBuilder('moveit_resources_panda')
        .robot_description(file_path='config/panda.urdf.xacro', mappings={
            'ros2_control_hardware_type': 'isaac',
            'initial_positions_file': str(panda_share / 'config/initial_positions.yaml'),
        })
        .trajectory_execution(file_path='config/gripper_moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs()
    )
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('rviz_config', default_value=str(share / 'config/combined.rviz')),
        Node(package='rviz2', executable='rviz2', name='combined_rviz', output='screen',
             arguments=['-d', LaunchConfiguration('rviz_config')],
             parameters=[config.to_dict(), {
                 'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool)}],
             remappings=[('joint_states', '/isaac/joint_states')]),
    ])
