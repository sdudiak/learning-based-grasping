import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from isaac_manipulation_segmentation.configuration import default_params_path, load_defaults, parse_overrides


def launch_node(context):
    values = {name: LaunchConfiguration(name).perform(context) for name in load_defaults()}
    overrides = parse_overrides(values)
    return [Node(
        package='isaac_manipulation_segmentation', executable='segmentation',
        name='segmentation', output='screen',
        parameters=[LaunchConfiguration('params_file').perform(context), overrides],
        prefix=[sys.executable],
    )]


def generate_launch_description():
    arguments = [DeclareLaunchArgument('params_file', default_value=str(default_params_path()),
                                       description='ROS parameter YAML file')]
    arguments.extend(DeclareLaunchArgument(name, default_value='', description=f'Override YAML {name} ({default})')
                     for name, default in load_defaults().items())
    return LaunchDescription([*arguments, OpaqueFunction(function=launch_node)])
