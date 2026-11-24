from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from data import validate_points, validate_predictions

ARM_JOINTS = [f'panda_joint{i}' for i in range(1, 8)]
FINGER_JOINTS = ['panda_finger_joint1', 'panda_finger_joint2']


def transform(xyz=(0, 0, 0), quaternion=(0, 0, 0, 1)):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion).as_matrix()
    result[:3, 3] = xyz
    return result


def graspgen_to_hand():
    result = np.eye(4)
    result[:3, :3] = Rotation.from_euler('z', np.pi / 2).as_matrix()
    return result


def load_result(directory):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    if metadata.get('gripper') != 'franka_panda':
        raise ValueError('Only pretrained franka_panda results are supported')
    observation = metadata['input']
    frame = observation.get('frame_id', '')
    if observation.get('units') != 'meters' or frame in ('', 'object', 'unspecified'):
        raise ValueError('Use a metric camera capture with a known ROS frame')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as data:
        grasps, scores = validate_predictions(data['grasps'], data['scores'])
        points = validate_points(data['points'])
        ranking_scores = data['ranking_scores'] if 'ranking_scores' in data else scores
        if ranking_scores.shape != scores.shape or not np.isfinite(ranking_scores).all():
            raise ValueError('Invalid candidate ranking scores')
    order = np.argsort(-ranking_scores, kind='stable')
    return grasps[order], scores[order], points, observation


def waypoint_poses(grasp, tool_offset, approach, lift):
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= approach * grasp[:3, 2]
    lifted = grasp.copy()
    lifted[2, 3] += lift
    return pregrasp @ tool_offset, grasp @ tool_offset, lifted @ tool_offset


def state_with_positions(state, names, positions):
    result = deepcopy(state)
    values = dict(zip(result.joint_state.name, result.joint_state.position))
    values.update(zip(names, positions))
    result.joint_state.name = list(values)
    result.joint_state.position = list(values.values())
    result.joint_state.velocity = []
    result.joint_state.effort = []
    result.is_diff = True
    return result


def trajectory_end(state, trajectory):
    joint = trajectory.joint_trajectory
    if len(joint.points) < 2 or not joint.joint_names:
        raise ValueError('Planner returned an empty or trivial trajectory')
    return state_with_positions(state, joint.joint_names, joint.points[-1].positions)


def validate_trajectory(trajectory, max_joint_step=None):
    joint = trajectory.joint_trajectory
    if set(joint.joint_names) != set(ARM_JOINTS) or len(joint.points) < 2:
        raise ValueError('Expected a nonempty seven-joint Panda trajectory')
    positions = np.asarray([point.positions for point in joint.points])
    times = np.asarray([p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 for p in joint.points])
    if positions.shape != (len(times), 7) or not np.isfinite(positions).all():
        raise ValueError('Malformed trajectory positions')
    if times[0] < 0 or times[-1] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Trajectory has invalid timing')
    if max_joint_step is not None and np.max(np.abs(np.diff(positions, axis=0))) > max_joint_step:
        raise ValueError('Cartesian trajectory contains a joint jump')


def observation_pose(center, azimuth, radius, height, tool_to_camera, roll=0):
    angle = np.deg2rad(azimuth)
    radial = np.array([np.cos(angle), np.sin(angle), 0.0])
    displacement = radius * radial + [0, 0, height]
    forward = -displacement / np.linalg.norm(displacement)
    right = np.array([-radial[1], radial[0], 0.0])
    down = np.cross(forward, right)
    camera = transform(center + displacement)
    camera[:3, :3] = np.column_stack([right, down, forward])
    camera[:3, :3] = camera[:3, :3] @ Rotation.from_euler('z', roll, degrees=True).as_matrix()
    return camera @ np.linalg.inv(tool_to_camera)
