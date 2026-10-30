from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from isaac_manipulation_segmentation.configuration import load_defaults


def include(package, filename, arguments):
    share = Path(get_package_share_directory(package))
    return GroupAction(actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(share / 'launch' / filename)),
        launch_arguments=arguments.items())])


def launch_viewer(context):
    if LaunchConfiguration('rviz').perform(context) != 'true':
        return []
    params_file = LaunchConfiguration('pointcloud_params_file').perform(context)
    settings = yaml.safe_load(Path(params_file).read_text())['pointcloud']['ros__parameters']
    share = Path(get_package_share_directory('isaac_manipulation_segmentation'))
    return [Node(
        package='rviz2', executable='rviz2', name='perception_rviz', output='screen',
        arguments=['-d', str(share / 'rviz/pipeline.rviz'),
                   '-f', settings.get('output_frame', 'panda_link0')],
        parameters=[{'use_sim_time': settings.get('use_sim_time', True)}],
        remappings=[('/pointcloud/environment', settings.get('environment_topic', '/pointcloud/environment')),
                    ('/pointcloud/target', settings.get('target_topic', '/pointcloud/target')),
                    ('/pointcloud/target_centroid', settings.get('centroid_topic', '/pointcloud/target_centroid')),
                    ('/pointcloud/target_bbox', settings.get('bounding_box_topic', '/pointcloud/target_bbox'))])]


def generate_launch_description():
    segmentation = Path(get_package_share_directory('isaac_manipulation_segmentation'))
    pointcloud = Path(get_package_share_directory('isaac_manipulation_pointcloud'))
    arguments = [
        DeclareLaunchArgument('rviz', default_value='false', choices=['true', 'false']),
        DeclareLaunchArgument('segmentation_params_file', default_value=str(segmentation / 'config/params.yaml')),
        DeclareLaunchArgument('pointcloud_params_file', default_value=str(pointcloud / 'config/params.yaml')),
    ]
    forwarded = ('checkpoint', 'device', 'score_threshold', 'decimation')
    defaults = load_defaults()
    arguments.extend(DeclareLaunchArgument(name, default_value='', description=f'Override segmentation {name} ({defaults[name]})')
                     for name in forwarded)
    segmentation_args = {name: LaunchConfiguration(name) for name in forwarded}
    segmentation_args.update(rviz='false', publish_overlay_image='true',
                             params_file=LaunchConfiguration('segmentation_params_file'))
    return LaunchDescription([
        *arguments,
        include('isaac_manipulation_segmentation', 'segmentation.launch.py', segmentation_args),
        include('isaac_manipulation_pointcloud', 'pointcloud.launch.py',
                {'rviz': 'false', 'params_file': LaunchConfiguration('pointcloud_params_file')}),
        OpaqueFunction(function=launch_viewer),
    ])
