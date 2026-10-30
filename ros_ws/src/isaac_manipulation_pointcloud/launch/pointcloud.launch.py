from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = Path(get_package_share_directory('isaac_manipulation_pointcloud'))
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/params.yaml')),
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
        Node(package='isaac_manipulation_pointcloud', executable='pointcloud', output='screen',
             parameters=[LaunchConfiguration('params_file')]),
        OpaqueFunction(function=launch_viewer),
    ])


def launch_viewer(context):
    if LaunchConfiguration('rviz').perform(context) != 'true':
        return []
    params = yaml.safe_load(Path(LaunchConfiguration('params_file').perform(context)).read_text())
    settings = params['pointcloud']['ros__parameters']
    share = Path(get_package_share_directory('isaac_manipulation_pointcloud'))
    return [Node(package='rviz2', executable='rviz2', name='pointcloud_rviz',
                 arguments=['-d', str(share / 'rviz/pointcloud.rviz'),
                            '-f', settings.get('output_frame', 'panda_link0')],
                 parameters=[{'use_sim_time': settings.get('use_sim_time', True)}])]
