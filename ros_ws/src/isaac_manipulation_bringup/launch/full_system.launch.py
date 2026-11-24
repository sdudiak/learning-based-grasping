from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def include(package, filename, **arguments):
    path = Path(get_package_share_directory(package)) / 'launch' / filename
    return GroupAction(actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(path)), launch_arguments=arguments.items())])


def generate_launch_description():
    bringup = 'isaac_manipulation_bringup'
    return LaunchDescription([
        DeclareLaunchArgument('checkpoint'),
        DeclareLaunchArgument('device', default_value=''),
        DeclareLaunchArgument('score_threshold', default_value=''),
        DeclareLaunchArgument('decimation', default_value=''),
        include(bringup, 'moveit.launch.py', rviz='false'),
        include('isaac_manipulation_segmentation', 'pipeline.launch.py', rviz='false',
                **{name: LaunchConfiguration(name)
                   for name in ('checkpoint', 'device', 'score_threshold', 'decimation')}),
        include(bringup, 'rviz.launch.py'),
    ])
