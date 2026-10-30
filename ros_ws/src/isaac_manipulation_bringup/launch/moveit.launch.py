from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    panda_share = Path(get_package_share_directory('moveit_resources_panda_moveit_config'))
    share = Path(get_package_share_directory('isaac_manipulation_bringup'))
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
    robot = ET.fromstring(config.robot_description['robot_description'])
    for hardware in robot.findall('ros2_control/hardware'):
        for param in hardware.findall('param'):
            if param.get('name') == 'joint_commands_topic':
                param.text = '/isaac/joint_command'
            elif param.get('name') == 'joint_states_topic':
                param.text = '/isaac/joint_states'
    for joint in robot.findall("ros2_control/joint[@name='panda_finger_joint2']"):
        joint.set('mimic', 'false')
        ET.SubElement(joint, 'command_interface', name='position')
        ET.SubElement(joint, 'param', name='mimic').text = 'panda_finger_joint1'
        ET.SubElement(joint, 'param', name='multiplier').text = '1'
    systems = robot.findall('ros2_control')
    system = systems[0]
    system.set('name', 'IsaacPandaSystem')
    for other in systems[1:]:
        for joint in other.findall('joint'):
            system.append(joint)
        robot.remove(other)
    config.robot_description['robot_description'] = ET.tostring(robot, encoding='unicode')
    sim_time = {'use_sim_time': True}
    feedback_remappings = [('joint_states', '/isaac/joint_states')]
    controllers = str(share / 'config/controllers.yaml')
    nodes = [
        DeclareLaunchArgument('rviz', default_value='false'),
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             parameters=[config.robot_description, sim_time],
             remappings=feedback_remappings),
        Node(package='tf2_ros', executable='static_transform_publisher',
             arguments=['--x', '-0.5', '--y', '0', '--z', '1',
                        '--frame-id', 'world', '--child-frame-id', 'panda_link0'],
             parameters=[sim_time]),
        Node(package='controller_manager', executable='ros2_control_node',
             parameters=[controllers, sim_time], output='screen',
             remappings=[('~/robot_description', '/robot_description')]),
        Node(package='moveit_ros_move_group', executable='move_group',
             parameters=[config.to_dict(), sim_time], output='screen',
             remappings=feedback_remappings),
        Node(package='rviz2', executable='rviz2',
             arguments=['-d', str(panda_share / 'launch/moveit.rviz')],
             parameters=[config.to_dict(), sim_time],
             remappings=feedback_remappings,
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ]
    for controller in ['joint_state_broadcaster', 'panda_arm_controller', 'panda_hand_controller']:
        arguments = [controller, '-c', '/controller_manager', '--param-file', controllers]
        if controller == 'joint_state_broadcaster':
            arguments += ['--controller-ros-args', '--remap /joint_states:=/controller_joint_states']
        nodes.append(Node(package='controller_manager', executable='spawner',
                          arguments=arguments, output='screen'))
    return LaunchDescription(nodes)
