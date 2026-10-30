import shlex
import sys
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from isaac_manipulation_segmentation.configuration import default_params_path, load_defaults, load_parameter_file, parse_overrides


def launch_nodes(context):
    rviz = LaunchConfiguration('rviz').perform(context).lower()
    if rviz not in ('true', 'false'):
        raise ValueError('rviz must be true or false')
    values = {name: LaunchConfiguration(name).perform(context) for name in load_defaults()}
    if not values['image_topic']:
        values['image_topic'] = '/isaac/cameara/wrist/color/image_raw'
    if rviz == 'true' and not values['publish_overlay_image']:
        values['publish_overlay_image'] = 'true'
    overrides = parse_overrides(values)
    params_file = LaunchConfiguration('params_file').perform(context)
    node = Node(
        package='isaac_manipulation_segmentation', executable='segmentation',
        name='segmentation', output='screen',
        parameters=[params_file, overrides], prefix=[shlex.quote(sys.executable)],
    )
    if rviz == 'false':
        return [node]
    settings = load_defaults()
    settings.update(load_parameter_file(params_file))
    settings.update(overrides)
    overlay_topic = settings['overlay_image_topic']
    if overlay_topic.startswith('~/'):
        overlay_topic = '/segmentation/' + overlay_topic[2:]
    share = Path(get_package_share_directory('isaac_manipulation_segmentation'))
    viewer = Node(
        package='rviz2', executable='rviz2', name='segmentation_rviz', output='screen',
        arguments=['-d', str(share / 'rviz/segmentation.rviz'), '-f', 'wrist_color_optical_frame'],
        parameters=[{'use_sim_time': settings['use_sim_time']}],
        remappings=[('/segmentation_view/input_image', settings['image_topic']),
                    ('/segmentation_view/annotated_image', overlay_topic)],
    )
    shutdown = RegisterEventHandler(OnProcessExit(
        on_exit=[EmitEvent(event=Shutdown(reason='A segmentation viewer process exited'))]))
    return [shutdown, node, viewer]


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument('rviz', default_value='false', choices=('true', 'false')),
        DeclareLaunchArgument('params_file', default_value=str(default_params_path())),
    ]
    arguments.extend(DeclareLaunchArgument(name, default_value='', description=f'Override YAML {name} ({default})')
                     for name, default in load_defaults().items())
    share = Path(get_package_share_directory('isaac_manipulation_segmentation'))
    return LaunchDescription([
        *arguments,
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(share / 'launch/camera_tf.launch.py'))),
        OpaqueFunction(function=launch_nodes),
    ])
