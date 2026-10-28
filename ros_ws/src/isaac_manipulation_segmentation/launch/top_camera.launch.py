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
    share = Path(get_package_share_directory('isaac_manipulation_segmentation'))
    params_file = LaunchConfiguration('params_file').perform(context)
    arguments = {name: LaunchConfiguration(name).perform(context) for name in load_defaults()}
    settings = load_defaults()
    settings.update(load_parameter_file(params_file))
    settings.update(parse_overrides(arguments))
    overlay_topic = settings['overlay_image_topic']
    if overlay_topic.startswith('~/'):
        overlay_topic = '/segmentation/' + overlay_topic[2:]
    return [
        RegisterEventHandler(OnProcessExit(on_exit=[EmitEvent(event=Shutdown(reason='A viewer process exited'))])),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/segmentation.launch.py')),
            launch_arguments={**arguments, 'params_file': params_file}.items(),
        ),
        Node(package='rviz2', executable='rviz2', name='segmentation_rviz', output='screen',
             arguments=['-d', str(share / 'rviz/top_camera.rviz')],
             parameters=[{'use_sim_time': settings['use_sim_time']}],
             remappings=[('/top_camera_view/input_image', settings['image_topic']),
                         ('/top_camera_view/annotated_image', overlay_topic)]),
    ]


def generate_launch_description():
    defaults = {name: '' for name in load_defaults()}
    defaults.update(image_topic='/isaac/cameara/top/color/image_raw', publish_overlay_image='true')
    arguments = [DeclareLaunchArgument('params_file', default_value=str(default_params_path()))]
    arguments.extend(DeclareLaunchArgument(name, default_value=value) for name, value in defaults.items())
    return LaunchDescription([*arguments, OpaqueFunction(function=launch_nodes)])
