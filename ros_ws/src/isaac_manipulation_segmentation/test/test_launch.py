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
    context = launch_context(launch_module, params_file=str(params))
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
