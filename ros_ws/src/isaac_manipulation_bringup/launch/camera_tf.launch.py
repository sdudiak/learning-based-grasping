from launch import LaunchDescription
from launch_ros.actions import Node


# Mounts and optical offsets from custom_franka.usda and the Isaac Sim 5.1 D455 asset.
# Wrist mount uses the fixed joint pose; USD camera axes are converted to ROS optical axes.
TRANSFORMS = [
    ('panda_hand', 'wrist_camera_link', (0.0328466594, 0.0, 0.0117642377, 0.0, 0.7071067812, 0.0, 0.7071067812)),
    ('wrist_camera_link', 'wrist_color_optical_frame', (0.0, 0.0115, 0.0, -0.5, -0.5, 0.5, 0.5)),
    ('wrist_camera_link', 'wrist_depth_optical_frame', (0.0, 0.0, 0.0, 0.5, 0.5, -0.5, -0.5)),
]


def generate_launch_description():
    nodes = []
    for parent, child, transform in TRANSFORMS:
        arguments = ['--frame-id', parent, '--child-frame-id', child]
        for flag, value in zip(['--x', '--y', '--z', '--qx', '--qy', '--qz', '--qw'], transform):
            arguments.extend([flag, str(value)])
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=child + '_publisher', arguments=arguments,
            parameters=[{'use_sim_time': True}],
        ))
    return LaunchDescription(nodes)
