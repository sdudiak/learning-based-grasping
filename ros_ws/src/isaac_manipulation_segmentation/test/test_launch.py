import importlib.util
import sys
from pathlib import Path

import pytest
from launch import LaunchContext
from launch.actions import DeclareLaunchArgument


@pytest.fixture
def launch_module(monkeypatch):
    path = Path(__file__).parents[1] / 'launch/segmentation.launch.py'
    spec = importlib.util.spec_from_file_location('segmentation_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    return module


def launch_context(module, **values):
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    context.launch_configurations.update(values)
    return context


def test_launch_overrides_preserve_types_and_custom_yaml(launch_module, tmp_path):
    params = tmp_path / 'custom.yaml'
    params.write_text('segmentation:\n  ros__parameters:\n    decimation: 7\n')
    context = launch_context(launch_module, rviz='false', params_file=str(params), checkpoint='/tmp/model.pt',
                             publish_overlay_image='false', score_threshold='0.75')
    node, = launch_module.launch_nodes(context)
    assert node['parameters'][0] == str(params)
    overrides = node['parameters'][1]
    assert overrides['publish_overlay_image'] is False
    assert overrides['score_threshold'] == 0.75
    assert 'decimation' not in overrides


def test_launch_rejects_fractional_decimation(launch_module):
    context = launch_context(launch_module, rviz='false', decimation='1.5')
    with pytest.raises(ValueError, match='decimation'):
        launch_module.launch_nodes(context)


def test_wrist_topic_matches_node_and_rviz(launch_module, tmp_path):
    params = tmp_path / 'params.yaml'
    params.write_text('segmentation:\n  ros__parameters:\n    overlay_image_topic: ~/custom_overlay\n'
                      '    use_sim_time: false\n')
    context = launch_context(launch_module, params_file=str(params), rviz='true')
    _, segmentation, rviz = launch_module.launch_nodes(context)
    expected_topic = '/isaac/cameara/wrist/color/image_raw'
    assert segmentation['parameters'][1]['image_topic'] == expected_topic
    assert segmentation['parameters'][1]['publish_overlay_image'] is True
    assert rviz['remappings'] == [('/segmentation_view/input_image', expected_topic),
                                  ('/segmentation_view/annotated_image', '/segmentation/custom_overlay')]
    assert rviz['parameters'] == [{'use_sim_time': False}]
    assert rviz['arguments'][-2:] == ['-f', 'wrist_color_optical_frame']


def test_custom_topic_overrides_camera_default(launch_module):
    node, = launch_module.launch_nodes(launch_context(
        launch_module, rviz='false', image_topic='/custom/rgb'))
    assert node['parameters'][1]['image_topic'] == '/custom/rgb'


def test_cli_forwards_launch_arguments(monkeypatch):
    root = Path(__file__).parents[4]
    spec = importlib.util.spec_from_file_location('ros_segmentation_cli', root / 'scripts/ros_segmentation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(root)
    checkpoint = 'checkpoint:=/tmp/model with spaces.pt'
    monkeypatch.setattr(sys, 'argv', ['ros-segmentation', 'score_threshold:=0.7', checkpoint])
    executed = []
    monkeypatch.setattr(module.os, 'execv', lambda executable, arguments: executed.append((executable, arguments)))
    module.main()
    executable, arguments = executed[0]
    assert executable == '/bin/bash'
    assert arguments[-2:] == ['score_threshold:=0.7', checkpoint]
    assert not any(argument.startswith('camera:=') for argument in arguments)


def test_pipeline_cli_builds_dependencies_and_preserves_inference_python(monkeypatch):
    root = Path(__file__).parents[4]
    spec = importlib.util.spec_from_file_location('pipeline_cli', root / 'scripts/ros_segmentation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, 'argv', ['ros-perception', 'decimation:=2', 'rviz:=false'])
    executed = []
    monkeypatch.setattr(module.os, 'execv', lambda executable, arguments: executed.append(arguments))
    module.main(pipeline=True)
    arguments, = executed
    assert '--packages-up-to isaac_manipulation_segmentation' in arguments[4]
    assert '-DPython3_EXECUTABLE=/usr/bin/python3' in arguments[4]
    assert arguments[6] == sys.executable
    assert 'pipeline.launch.py' in arguments
    assert arguments[-2:] == ['decimation:=2', 'rviz:=false']


def test_pipeline_launch_only_includes_perception_and_one_viewer(monkeypatch):
    from launch.actions import IncludeLaunchDescription
    path = Path(__file__).parents[1] / 'launch/pipeline.launch.py'
    spec = importlib.util.spec_from_file_location('pipeline_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    context = launch_context(module, rviz='true')
    includes = [child for entity in module.generate_launch_description().entities
                for child in entity.get_sub_entities()
                if isinstance(child, IncludeLaunchDescription)]
    assert len(includes) == 2
    assert all(entity.condition is None for entity in includes)
    assert 'bringup' not in context.launch_configurations
    from launch.utilities import perform_substitutions
    paths = [perform_substitutions(context, entity.launch_description_source.location)
             if isinstance(entity.launch_description_source.location, list)
             else entity.launch_description_source.location for entity in includes]
    assert all('moveit' not in path and 'bringup' not in path for path in paths)
    for entity in includes:
        assert dict(entity.launch_arguments)['rviz'] == 'false'
    viewer, = module.launch_viewer(context)
    assert viewer['name'] == 'perception_rviz'
    assert viewer['arguments'][-2:] == ['-f', 'panda_link0']
    context.launch_configurations['rviz'] = 'false'
    assert module.launch_viewer(context) == []


def test_segmentation_owns_camera_tf_independently_of_viewer(launch_module):
    from launch.actions import IncludeLaunchDescription
    includes = [entity for entity in launch_module.generate_launch_description().entities
                if isinstance(entity, IncludeLaunchDescription)]
    camera_tf, = includes
    assert camera_tf.condition is None
    description = camera_tf.launch_description_source.get_launch_description(LaunchContext())
    assert len(description.entities) == 3
    assert all(node.node_executable == 'static_transform_publisher' for node in description.entities)


def test_moveit_launch_has_no_camera_tf(monkeypatch):
    from launch.actions import IncludeLaunchDescription
    path = Path(__file__).parents[2] / 'isaac_manipulation_bringup/launch/moveit.launch.py'
    spec = importlib.util.spec_from_file_location('moveit_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    entities = module.generate_launch_description().entities
    assert not any(isinstance(entity, IncludeLaunchDescription) for entity in entities)
    assert not any(isinstance(entity, DeclareLaunchArgument) and entity.name == 'cameras' for entity in entities)
    static_publishers = [entity for entity in entities if isinstance(entity, dict)
                         and entity.get('executable') == 'static_transform_publisher']
    world_tf, = static_publishers
    assert world_tf['arguments'][-1] == 'panda_link0'


@pytest.mark.parametrize('filename', [
    'isaac_manipulation_bringup/launch/moveit.launch.py',
    'isaac_manipulation_segmentation/launch/segmentation.launch.py',
    'isaac_manipulation_segmentation/launch/pipeline.launch.py',
    'isaac_manipulation_pointcloud/launch/pointcloud.launch.py',
])
def test_ros_launches_default_to_no_rviz(filename, monkeypatch):
    path = Path(__file__).parents[2] / filename
    spec = importlib.util.spec_from_file_location('headless_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    context = launch_context(module)
    assert context.launch_configurations['rviz'] == 'false'


def test_combined_viewer_launch_only_starts_rviz(monkeypatch):
    import yaml
    path = Path(__file__).parents[2] / 'isaac_manipulation_bringup/launch/rviz.launch.py'
    spec = importlib.util.spec_from_file_location('combined_viewer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    nodes = [entity for entity in module.generate_launch_description().entities if isinstance(entity, dict)]
    viewer, = nodes
    assert viewer['executable'] == 'rviz2'
    assert 'robot_description' in viewer['parameters'][0]
    config = yaml.safe_load((path.parents[1] / 'config/combined.rviz').read_text())
    classes = {display['Class'] for display in config['Visualization Manager']['Displays']}
    assert {'moveit_rviz_plugin/MotionPlanning', 'rviz_default_plugins/PointCloud2',
            'rviz_default_plugins/Image', 'rviz_default_plugins/Marker'} <= classes
