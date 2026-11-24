from copy import deepcopy
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from control_msgs.action import GripperCommand
from controller_manager_msgs.srv import ListControllers
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import Constraints, JointConstraint, RobotState, RobotTrajectory
from moveit_msgs.srv import (
    GetCartesianPath, GetMotionPlan, GetPositionIK, GetStateValidity)
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from pick_geometry import ARM_JOINTS, FINGER_JOINTS, state_with_positions, transform, validate_trajectory
from pick_scene import pose_message


class PlanningFailure(RuntimeError):
    pass


class MoveItClient(Node):
    def __init__(self, args):
        super().__init__('graspgen_pick', parameter_overrides=[Parameter('use_sim_time', value=True)])
        self.args = args
        self.latest = None
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.subscription = self.create_subscription(
            JointState, '/isaac/joint_states', self.receive_state, qos_profile_sensor_data)
        self.ik = self.create_client(GetPositionIK, '/compute_ik')
        self.plan = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.cartesian = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.validity = self.create_client(GetStateValidity, '/check_state_validity')
        self.arm = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.hand = ActionClient(self, GripperCommand, '/panda_hand_controller/gripper_cmd')

    def require_active_controllers(self):
        client = self.create_client(ListControllers, '/controller_manager/list_controllers')
        try:
            response = self.call(client, ListControllers.Request())
        finally:
            self.destroy_client(client)
        active = {controller.name for controller in response.controller if controller.state == 'active'}
        missing = {'panda_arm_controller', 'panda_hand_controller'} - active
        if missing:
            raise RuntimeError(f'Inactive or missing controllers: {", ".join(sorted(missing))}. '
                               'Keep Isaac Sim playing and check the ROS bringup logs.')

    def receive_state(self, message):
        if len(message.name) == len(message.position):
            self.latest = (message, time.monotonic())

    def wait(self, future):
        rclpy.spin_until_future_complete(self, future, timeout_sec=self.args.timeout)
        if not future.done():
            raise TimeoutError('ROS request timed out; check simulation playback and MoveIt')
        if future.cancelled() or future.result() is None:
            raise RuntimeError('ROS request failed')
        return future.result()

    def call(self, client, request):
        if not client.wait_for_service(timeout_sec=self.args.timeout):
            raise TimeoutError(f'Service unavailable: {client.srv_name}; start ros-moveit')
        return self.wait(client.call_async(request))

    def current_state(self, opening=None, *, after_ns=None):
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.latest is None or time.monotonic() - self.latest[1] > 2:
                continue
            message = self.latest[0]
            stamp_ns = Time.from_msg(message.header.stamp).nanoseconds
            if after_ns is not None and stamp_ns <= after_ns:
                continue
            age = (self.get_clock().now().nanoseconds - stamp_ns) * 1e-9
            if not set(ARM_JOINTS + FINGER_JOINTS).issubset(message.name) or not -0.1 <= age <= 1:
                continue
            if not np.isfinite(message.position).all():
                continue
            state = RobotState(joint_state=deepcopy(message), is_diff=True)
            return state if opening is None else state_with_positions(state, FINGER_JOINTS, [opening, opening])
        raise TimeoutError('No fresh complete /isaac/joint_states; press Play and check robot feedback')

    def lookup(self, target, source, stamp=None):
        when = Time.from_msg(stamp) if stamp is not None else Time()
        deadline = time.monotonic() + self.args.timeout
        while not self.buffer.can_transform(target, source, when):
            if time.monotonic() >= deadline:
                raise TimeoutError(f'Missing TF {source} -> {target} at the requested observation time')
            rclpy.spin_once(self, timeout_sec=0.05)
        value = self.buffer.lookup_transform(target, source, when).transform
        return transform([value.translation.x, value.translation.y, value.translation.z],
                         [value.rotation.x, value.rotation.y, value.rotation.z, value.rotation.w])

    def assert_valid(self, state):
        result = self.call(self.validity, GetStateValidity.Request(robot_state=state, group_name='panda_arm'))
        if not result.valid:
            pairs = [(contact.contact_body_1, contact.contact_body_2) for contact in result.contacts[:5]]
            raise PlanningFailure(f'Invalid robot state; contacts: {pairs}')

    def pregrasp_plan(self, pose, start):
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'panda_arm'
        request.ik_request.robot_state = start
        request.ik_request.ik_link_name = 'panda_link8'
        request.ik_request.pose_stamped = PoseStamped(pose=pose_message(pose))
        request.ik_request.pose_stamped.header.frame_id = 'panda_link0'
        request.ik_request.avoid_collisions = True
        request.ik_request.timeout.sec = 1
        result = self.call(self.ik, request)
        self.require_success(result.error_code, 'tool pose IK')
        goal = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        return self.joint_plan(goal, start, 'pose planning')

    def joint_plan(self, goal, start, stage='joint planning'):
        current = dict(zip(start.joint_state.name, start.joint_state.position))
        if all(abs(current[name] - goal[name]) < 0.001 for name in ARM_JOINTS):
            return self.hold_trajectory(current)
        constraints = [JointConstraint(joint_name=name, position=goal[name], tolerance_above=0.001,
                                       tolerance_below=0.001, weight=1.0) for name in ARM_JOINTS]
        request = GetMotionPlan.Request()
        motion = request.motion_plan_request
        motion.group_name = 'panda_arm'
        motion.pipeline_id = 'ompl'
        motion.start_state = start
        motion.goal_constraints = [Constraints(joint_constraints=constraints)]
        motion.num_planning_attempts = 1
        motion.allowed_planning_time = self.args.planning_time
        motion.max_velocity_scaling_factor = self.args.speed
        motion.max_acceleration_scaling_factor = self.args.speed
        result = self.call(self.plan, request).motion_plan_response
        self.require_success(result.error_code, stage)
        validate_trajectory(result.trajectory)
        return result.trajectory

    @staticmethod
    def hold_trajectory(positions):
        trajectory = RobotTrajectory()
        trajectory.joint_trajectory.joint_names = ARM_JOINTS
        values = [positions[name] for name in ARM_JOINTS]
        trajectory.joint_trajectory.points = [
            JointTrajectoryPoint(positions=values, velocities=[0.0] * 7, time_from_start=duration)
            for duration in (Duration(), Duration(nanosec=100_000_000))]
        return trajectory

    def cartesian_plan(self, pose, start):
        request = GetCartesianPath.Request()
        request.header.frame_id = 'panda_link0'
        request.group_name = 'panda_arm'
        request.link_name = 'panda_link8'
        request.start_state = start
        request.waypoints = [pose_message(pose)]
        request.max_step = self.args.cartesian_step
        request.revolute_jump_threshold = self.args.max_joint_step
        request.avoid_collisions = True
        request.max_velocity_scaling_factor = self.args.speed
        request.max_acceleration_scaling_factor = self.args.speed
        result = self.call(self.cartesian, request)
        self.require_success(result.error_code, 'Cartesian planning')
        if result.fraction < 1 - 1e-6:
            raise PlanningFailure(f'Incomplete Cartesian path: {result.fraction:.1%}')
        try:
            validate_trajectory(result.solution, self.args.max_joint_step)
        except ValueError as error:
            raise PlanningFailure(str(error)) from error
        return result.solution

    @staticmethod
    def require_success(error, stage):
        if error.val != 1:
            raise PlanningFailure(f'{stage} failed: MoveIt error {error.val}')

    def action(self, client, goal, stage):
        if not client.wait_for_server(timeout_sec=self.args.timeout):
            raise TimeoutError('Execution action unavailable; check ros-moveit and its controllers')
        pending = client.send_goal_async(goal)
        try:
            handle = self.wait(pending)
        except BaseException:
            pending.add_done_callback(lambda future: future.result().cancel_goal_async()
                                      if not future.cancelled() and future.result().accepted else None)
            raise
        if not handle.accepted:
            raise RuntimeError('Controller rejected the action goal')
        try:
            result = self.wait(handle.get_result_async())
        except BaseException:
            cancelled = handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancelled, timeout_sec=2)
            raise
        if result.status != GoalStatus.STATUS_SUCCEEDED:
            code = getattr(result.result, 'error_code', None)
            detail = f', MoveIt error {code.val}' if code is not None else ''
            raise RuntimeError(f'{stage} ended with status {result.status}{detail}. '
                               'See MoveIt and controller logs for details.')
        return result.result

    def execute(self, trajectory):
        result = self.action(self.arm, ExecuteTrajectory.Goal(
            trajectory=trajectory, controller_names=['panda_arm_controller']), 'Arm execution')
        self.require_success(result.error_code, 'trajectory execution')

    def gripper(self, position, allow_stall=False):
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = self.args.gripper_effort
        result = self.action(self.hand, goal, 'Gripper command')
        if not result.reached_goal and not (allow_stall and result.stalled):
            raise RuntimeError('Gripper did not reach the requested position')
        return result
