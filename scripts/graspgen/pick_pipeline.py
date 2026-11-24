import json

import numpy as np
from rosidl_runtime_py.convert import message_to_ordereddict

from pick_geometry import (ARM_JOINTS, graspgen_to_hand,
                           trajectory_end, waypoint_poses)
from pick_moveit import PlanningFailure


def candidate_plan(node, scene, grasp, offset, start, args):
    pre, goal, lifted = waypoint_poses(grasp, offset, args.approach, args.lift)
    scene.reset()
    node.assert_valid(start)
    pre_plan = node.pregrasp_plan(pre, start)
    pre_end = trajectory_end(start, pre_plan)
    approach = node.cartesian_plan(goal, pre_end)
    grasp_end = trajectory_end(pre_end, approach)
    lift = node.cartesian_plan(lifted, grasp_end)
    return {'pregrasp': pre_plan, 'approach': approach, 'lift': lift}, [pre, goal, lifted]


def choose_candidate(node, scene, grasps, scores, offset, start, args, report):
    for rank, (grasp, score) in enumerate(zip(grasps[:args.max_candidates], scores)):
        try:
            plans, poses = candidate_plan(node, scene, grasp, offset, start, args)
        except PlanningFailure as error:
            node.get_logger().info(f'Candidate {rank}, score {score:.3f}: {error}')
            report['rejections'].append({'rank': rank, 'score': float(score), 'reason': str(error)})
            continue
        report.update(candidate_rank=rank, score=float(score),
                      tool_poses={name: pose.tolist() for name, pose in zip(plans, poses)})
        node.get_logger().info(f'Candidate {rank}, score {score:.3f}: pre-grasp, approach and lift planned')
        return plans, poses
    raise PlanningFailure(f'None of the first {min(args.max_candidates, len(grasps))} candidates has a complete plan')


def assert_arm_stationary(expected, actual, stage='during planning'):
    before = dict(zip(expected.joint_state.name, expected.joint_state.position))
    after = dict(zip(actual.joint_state.name, actual.joint_state.position))
    joint = max(ARM_JOINTS, key=lambda name: abs(before[name] - after[name]))
    error = abs(before[joint] - after[joint])
    if error > 0.01:
        raise RuntimeError(f'Arm position mismatch {stage}: {joint} differs by {error:.4f} rad '
                           '(limit 0.0100 rad).')


def save_plans(directory, plans):
    for name, trajectory in plans.items():
        (directory / f'{name}.json').write_text(json.dumps(message_to_ordereddict(trajectory), indent=2) + '\n')


def tool_offset(node):
    hand_to_tool = node.lookup('panda_hand', 'panda_link8')
    expected = np.eye(4)
    expected[:2, :2] = [[2**-.5, -2**-.5], [2**-.5, 2**-.5]]
    if not np.allclose(hand_to_tool, expected, atol=1e-5):
        raise ValueError('Unexpected Panda hand-to-link8 transform; verify the robot description')
    return graspgen_to_hand() @ hand_to_tool
