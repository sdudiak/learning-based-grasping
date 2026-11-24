import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from data import top_down_mask
from infer import add_sampling_arguments, validate_sampling_arguments

ROOT = Path(__file__).resolve().parents[2]


def add_motion_arguments(parser):
    parser.add_argument('--max-candidates', type=int, default=20)
    parser.add_argument('--approach', type=float, default=0.10)
    parser.add_argument('--lift', type=float, default=0.10)
    parser.add_argument('--speed', type=float, default=0.15)
    parser.add_argument('--planning-time', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=60.0)
    parser.add_argument('--cartesian-step', type=float, default=0.005)
    parser.add_argument('--max-joint-step', type=float, default=0.35)
    parser.add_argument('--open-position', type=float, default=0.04, help='Per-finger position in meters')
    parser.add_argument('--close-position', type=float, default=0.0)
    parser.add_argument('--gripper-effort', type=float, default=20.0)
    parser.add_argument('--table-height', type=float, default=0.0, help='Tabletop Z in panda_link0')
    parser.add_argument('--table-center', type=float, nargs=2, default=[0.5, 0.0], metavar=('X', 'Y'))
    parser.add_argument('--table-size', type=float, nargs=2, default=[1.5, 1.5], metavar=('X', 'Y'))
    parser.add_argument('--table-thickness', type=float, default=0.05)


def validate_motion_arguments(parser, args):
    positive = [args.approach, args.lift, args.planning_time, args.timeout, args.cartesian_step,
                args.max_joint_step, args.gripper_effort, args.table_thickness, *args.table_size]
    values = positive + [args.speed, args.open_position, args.close_position, args.table_height,
                         *args.table_center]
    if not np.isfinite(values).all() or min(positive) <= 0 or args.max_candidates < 1:
        parser.error('Distances, times, effort and candidate count must be finite and positive')
    if not 0 < args.speed <= 1 or not 0 <= args.close_position < args.open_position <= 0.04:
        parser.error('Speed must be in (0,1]; finger positions must satisfy 0 <= close < open <= 0.04')


def parse_args():
    parser = argparse.ArgumentParser(description='Interactive, four-view Isaac Sim grasp test.')
    add_motion_arguments(parser)
    add_sampling_arguments(parser)
    parser.set_defaults(max_candidates=60, speed=0.30, top_down_bonus=0.70,
                        moe_z_offsets_cm=[-1.2, -1.0, -0.75, -0.5])
    parser.add_argument('--radius', type=float, default=0.12, help='Camera orbit radius in meters')
    parser.add_argument('--height', type=float, default=0.34, help='Camera height above target in meters')
    parser.add_argument('--planner', choices=('diffusion', 'graspmoe'), default='graspmoe')
    parser.add_argument('--inference-timeout', type=float, default=300)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'outputs/graspgen/pipeline')
    args, ros_args = parser.parse_known_args()
    validate_motion_arguments(parser, args)
    validate_sampling_arguments(parser, args)
    values = [args.radius, args.height, args.inference_timeout]
    if not np.isfinite(values).all() or min(values) <= 0:
        parser.error('Radius, height and inference timeout must be positive and finite')
    return args, ros_args


def confirm(description):
    input(f'\nPress Enter to {description} (Ctrl+C to stop) ... ')


def infer(args, directory):
    output = directory / 'inference'
    subprocess.run([
        sys.executable, str(Path(__file__).with_name('run.py')), 'infer',
        str(directory / 'target_map.npz'), '--planner', args.planner,
        '--output-dir', str(output), '--topk', str(args.max_candidates),
        '--num-grasps', str(args.num_grasps), '--top-down-bonus', str(args.top_down_bonus),
        '--top-down-angle', str(args.top_down_angle), '--moe-num-yaws', str(args.moe_num_yaws),
        '--moe-obb-density', args.moe_obb_density, '--moe-z-offsets-cm',
        *map(str, args.moe_z_offsets_cm),
    ], check=True, timeout=args.inference_timeout)
    result, = output.glob(f'target_map/*/{args.planner}-00')
    return result


def home_joints():
    from ament_index_python.packages import get_package_share_directory
    path = Path(get_package_share_directory('moveit_resources_panda_moveit_config')) / 'config/panda.srdf'
    state = ET.parse(path).find("./group_state[@group='panda_arm'][@name='ready']")
    return {joint.attrib['name']: float(joint.attrib['value']) for joint in state}


def run(node, args, directory):
    from pick_geometry import load_result
    from pick_pipeline import assert_arm_stationary, tool_offset
    from pick_scene import TestScene
    from pipeline_test_ros import FRAME, ScanClient

    log = node.get_logger().info
    scan = ScanClient(node)
    confirm('scan the selected object')
    log('Checking arm and gripper controller activation')
    node.require_active_controllers()
    selection, center = scan.selected_target()
    log(f'Selected {selection} at {center.round(3).tolist()} m in {FRAME}')
    log(f'Writing test artifacts to {directory}')
    directory.mkdir(parents=True, exist_ok=False)
    scene = TestScene(node, FRAME, args)
    try:
        scene.reset()
        scan.map_target(selection, center, directory)

        confirm('ask GraspGen for candidates')
        scan.check_selection(selection)
        log(f'Running GraspGen {args.planner} on the fused object map')
        result = infer(args, directory)
        grasps, scores, _, observation = load_result(result)
        scan.check_selection(selection)
        if observation['frame_id'] != FRAME or observation.get('selection_id') != selection:
            raise RuntimeError('Inference result does not match the selected map')
        selected_count = min(len(grasps), args.max_candidates)
        scan.show_grasps(grasps[:selected_count], scores[:selected_count])
        top_down_count = int(top_down_mask(grasps[:selected_count], args.top_down_angle).sum())
        log(f'RViz shows all {selected_count} MoveIt candidates, including {top_down_count} '
            f'within {args.top_down_angle:g}° of straight down. Labels show raw model confidence.')

        confirm('plan and approach the object')
        scan.check_selection(selection)
        plans, poses, grasp_state = approach(node, scene, grasps, scores, tool_offset(node), args, directory)
        confirm('close the gripper')
        scan.check_selection(selection)
        now_ns = node.get_clock().now().nanoseconds
        assert_arm_stationary(grasp_state, node.current_state(after_ns=now_ns), 'before gripper closure')
        scan.mapping('reset')
        log('Closing the fingers and maintaining the close command')
        closed = node.gripper(args.close_position, allow_stall=True)
        log(f'Gripper position {closed.position:.4f} m, stalled={closed.stalled}, reached_goal={closed.reached_goal}')

        confirm('lift the object and return to the Panda ready pose')
        now_ns = node.get_clock().now().nanoseconds
        assert_arm_stationary(grasp_state, node.current_state(after_ns=now_ns), 'before lift')
        return_home(node, poses[2], plans, directory)
        log('Panda reached ready with the close command active. Check the object in Isaac Sim')
    finally:
        scan.clear_markers()
        scene.cleanup()


def approach(node, scene, grasps, scores, offset, args, directory):
    from pick_geometry import trajectory_end
    from pick_pipeline import assert_arm_stationary, choose_candidate, save_plans

    start = node.current_state(opening=args.open_position)
    report = {'rejections': []}
    plans, poses = choose_candidate(node, scene, grasps, scores, offset, start, args, report)
    (directory / 'candidate.json').write_text(json.dumps(report, indent=2) + '\n')
    log = node.get_logger().info
    log('Opening fingers and executing the pre-grasp trajectory')
    assert_arm_stationary(start, node.current_state())
    node.gripper(args.open_position)
    assert_arm_stationary(start, node.current_state())
    node.execute(plans['pregrasp'])
    log('Replanning and executing the Cartesian approach')
    approach_start = node.current_state(after_ns=node.get_clock().now().nanoseconds)
    plans['approach'] = node.cartesian_plan(poses[1], approach_start)
    node.execute(plans['approach'])
    save_plans(directory, plans)
    log('Waiting for post-approach joint feedback to settle')
    settle_ns = node.get_clock().now().nanoseconds + 200_000_000
    grasp_state = node.current_state(after_ns=settle_ns)
    assert_arm_stationary(trajectory_end(approach_start, plans['approach']), grasp_state,
                          'at the grasp pose')
    log('Reached and verified the grasp pose')
    return plans, poses, grasp_state


def return_home(node, lift_pose, plans, directory):
    from pick_pipeline import save_plans

    node.get_logger().info('Planning and executing the Cartesian lift')
    plans['lift'] = node.cartesian_plan(lift_pose, node.current_state())
    node.execute(plans['lift'])
    node.get_logger().info('Planning and executing the return to Panda ready')
    plans['home'] = node.joint_plan(home_joints(), node.current_state(), 'return-to-ready planning')
    save_plans(directory, plans)
    node.execute(plans['home'])


def main():
    args, ros_args = parse_args()
    import rclpy
    from rclpy.signals import SignalHandlerOptions
    from pick_moveit import MoveItClient

    if not sys.stdin.isatty():
        raise SystemExit('Run this test in a terminal so Enter can authorize each stage')
    if not (ROOT / 'third_party/GraspGen/.venv/bin/python').is_file():
        raise SystemExit('GraspGen environment is missing. Run pixi run graspgen-setup')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = MoveItClient(args)
    try:
        run(node, args, args.output_root.resolve() / run_id)
    except (KeyboardInterrupt, EOFError):
        node.get_logger().warning('Test stopped')
        raise SystemExit(130)
    except (RuntimeError, ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        node.get_logger().error(str(error))
        raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
