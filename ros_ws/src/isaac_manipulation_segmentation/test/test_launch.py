import importlib.util
from pathlib import Path

import pytest
from launch import LaunchContext


@pytest.fixture
def launch_module():
    path = Path(__file__).parents[1] / 'launch/segmentation.launch.py'
    spec = importlib.util.spec_from_file_location('segmentation_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_overrides_preserve_types_and_custom_yaml(launch_module, tmp_path, monkeypatch):
    monkeypatch.setattr(launch_module, 'Node', lambda **kwargs: kwargs)
    params = tmp_path / 'custom.yaml'
    params.write_text('segmentation:\n  ros__parameters:\n    decimation: 7\n')
    context = LaunchContext()
    context.launch_configurations.update({
        name: '' for name in launch_module.load_defaults()
    })
    context.launch_configurations.update(params_file=str(params), checkpoint='/tmp/model.pt',
                                         publish_overlay_image='false', score_threshold='0.75')
    node, = launch_module.launch_node(context)
    assert node['parameters'][0] == str(params)
    overrides = node['parameters'][1]
    assert overrides['publish_overlay_image'] is False
    assert overrides['score_threshold'] == 0.75
    assert 'decimation' not in overrides


def test_launch_rejects_fractional_decimation(launch_module):
    context = LaunchContext()
    context.launch_configurations.update({name: '' for name in launch_module.load_defaults()})
    context.launch_configurations['decimation'] = '1.5'
    with pytest.raises(ValueError, match='decimation'):
        launch_module.launch_node(context)


def test_top_camera_defaults_and_rviz_remapping(tmp_path, monkeypatch):
    from launch.actions import DeclareLaunchArgument

    path = Path(__file__).parents[1] / 'launch/top_camera.launch.py'
    spec = importlib.util.spec_from_file_location('top_camera_launch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'Node', lambda **kwargs: kwargs)
    monkeypatch.setattr(module, 'IncludeLaunchDescription', lambda source, **kwargs: kwargs)
    context = LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument):
            action.execute(context)
    params = tmp_path / 'params.yaml'
    params.write_text('segmentation:\n  ros__parameters:\n    overlay_image_topic: ~/custom_overlay\n'
                      '    use_sim_time: false\n')
    context.launch_configurations['params_file'] = str(params)
    _, segmentation, rviz = module.launch_nodes(context)
    arguments = dict(segmentation['launch_arguments'])
    assert arguments['image_topic'] == '/isaac/cameara/top/color/image_raw'
    assert arguments['publish_overlay_image'] == 'true'
    assert rviz['remappings'] == [('/top_camera_view/input_image', '/isaac/cameara/top/color/image_raw'),
                                  ('/top_camera_view/annotated_image', '/segmentation/custom_overlay')]
    assert rviz['parameters'] == [{'use_sim_time': False}]

