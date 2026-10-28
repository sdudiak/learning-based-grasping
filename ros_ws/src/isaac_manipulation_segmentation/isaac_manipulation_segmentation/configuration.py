from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def default_params_path():
    return Path(get_package_share_directory('isaac_manipulation_segmentation')) / 'config/params.yaml'


def load_defaults():
    return load_parameter_file(default_params_path())


def load_parameter_file(path):
    document = yaml.safe_load(Path(path).expanduser().read_text())
    if not isinstance(document, dict):
        raise ValueError(f'Expected a ROS parameter mapping in {path}')
    parameters = {}
    for key in ('/**', 'segmentation', '/segmentation'):
        if key in document:
            if not isinstance(document[key], dict):
                raise ValueError(f'Expected a node parameter mapping in {path}')
            values = document[key].get('ros__parameters', {})
            if not isinstance(values, dict):
                raise ValueError(f'Expected ros__parameters to be a mapping in {path}')
            parameters.update(values)
    return parameters


def parse_overrides(values):
    defaults = load_defaults()
    overrides = {}
    for name, value in values.items():
        if value == '':
            continue
        default = defaults[name]
        parsed = value if isinstance(default, str) else yaml.safe_load(value)
        if isinstance(default, float) and type(parsed) is int:
            parsed = float(parsed)
        if type(parsed) is not type(default):
            raise ValueError(f'{name} must have type {type(default).__name__}')
        overrides[name] = parsed
    return overrides


def validate_parameters(parameters):
    if not parameters['checkpoint']:
        raise ValueError('checkpoint must point to a trained Mask R-CNN checkpoint')
    if type(parameters['decimation']) is not int or parameters['decimation'] < 1:
        raise ValueError('decimation must be an integer >= 1')
    for name in ('score_threshold', 'mask_threshold'):
        if not 0.0 <= parameters[name] <= 1.0:
            raise ValueError(f'{name} must be between 0 and 1')
    if not 0 < parameters['tracking_iou_threshold'] <= 1:
        raise ValueError('tracking_iou_threshold must be greater than 0 and at most 1')
    missed = parameters['tracking_max_missed_frames']
    if type(missed) is not int or missed < 0:
        raise ValueError('tracking_max_missed_frames must be an integer >= 0')
    for name in ('image_topic', 'detections_topic', 'instance_mask_topic', 'overlay_image_topic'):
        if not parameters[name]:
            raise ValueError(f'{name} must not be empty')
