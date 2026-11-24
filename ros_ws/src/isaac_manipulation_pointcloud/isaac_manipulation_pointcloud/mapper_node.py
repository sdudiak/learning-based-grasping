import json

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.clock import Clock, ClockType, JumpThreshold
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.task import Future
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32, read_points_numpy
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .target_map import TargetMap


class TargetMapper(Node):
    def __init__(self):
        super().__init__('target_mapper')
        defaults = {
            'target_topic': '/pointcloud/target',
            'frame_id': 'panda_link0',
            'camera_frame': 'wrist_color_optical_frame',
            'voxel_resolution': 0.002,
            'view_angle_deg': 12.0,
            'minimum_overlap': 0.2,
            'maximum_observation_age': 3.0,
            'capture_timeout': 3.0,
            'save_path': 'target_map.npz',
        }
        self.declare_parameters('', list(defaults.items()))
        self.settings = {key: self.get_parameter(key).value for key in defaults}
        p = self.settings
        self.map = TargetMap(p['voxel_resolution'], p['view_angle_deg'], p['minimum_overlap'])
        for name in ('maximum_observation_age', 'capture_timeout'):
            if not np.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        self.max_age = int(p['maximum_observation_age'] * 1e9)
        self.state = 'idle'
        self.selection_id = ''
        self.capture_future = None
        self.capture_stamp = 0
        self.capture_rejection = None
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.last_status = None
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.cloud = self.create_publisher(PointCloud2, '/pointcloud/target_map', qos)
        self.status = self.create_publisher(String, '/pointcloud/mapping/status', qos)
        self.tf = Buffer(node=self)
        self.listener = TransformListener(self.tf, self)
        self.create_subscription(PointCloud2, p['target_topic'], self.receive, qos_profile_sensor_data)
        self.create_subscription(String, '/pointcloud/selection_id', self.selection_changed, qos)
        self.clock_jump = self.get_clock().create_jump_callback(
            JumpThreshold(min_forward=None, min_backward=Duration(nanoseconds=-1), on_clock_change=True),
            post_callback=lambda _: self.clear('Simulation clock changed'))
        for name in ('start', 'pause', 'reset', 'save'):
            self.create_service(Trigger, f'/pointcloud/mapping/{name}', getattr(self, name))
        self.create_service(Trigger, '/pointcloud/mapping/capture', self.capture,
                            callback_group=ReentrantCallbackGroup())
        self.publish_status('Waiting to start')

    def selection_changed(self, message):
        if self.selection_id != message.data:
            self.selection_id = message.data
            self.clear('Target selection changed')

    def publish_status(self, reason):
        status = json.dumps(dict(state=self.state, views=len(self.map.views), reason=reason))
        if status != self.last_status:
            self.status.publish(String(data=status))
            self.last_status = status

    def publish_cloud(self, stamp):
        header = Header(frame_id=self.settings['frame_id'], stamp=stamp)
        self.cloud.publish(create_cloud_xyz32(header, self.map.points()))

    def clear(self, reason):
        self.map.clear()
        self.state = 'idle'
        self.finish_capture(False, reason)
        self.publish_cloud(self.get_clock().now().to_msg())
        self.publish_status(reason)

    def start(self, request, response):
        if not self.selection_id:
            return Trigger.Response(success=False, message='Select a target first')
        if self.state != 'mapping':
            self.state = 'mapping'
            self.publish_status('Ready to capture')
        return Trigger.Response(success=True, message='Session open. Call capture at each viewpoint.')

    async def capture(self, request, response):
        if self.state != 'mapping':
            return Trigger.Response(success=False, message='Start mapping before capturing')
        if self.capture_future is not None:
            return Trigger.Response(success=False, message='A capture is already pending')
        future = Future()
        self.capture_future = future
        self.capture_stamp = self.get_clock().now().nanoseconds
        self.capture_rejection = None
        self.publish_status('Waiting for a fresh view')
        timer = self.create_timer(self.settings['capture_timeout'], self.capture_timed_out,
                                  clock=self.steady_clock)
        try:
            return await future
        finally:
            self.destroy_timer(timer)
            self.capture_future = None

    def capture_timed_out(self):
        reason = self.capture_rejection or 'No fresh target cloud received'
        self.finish_capture(False, f'Capture timed out: {reason}')

    def finish_capture(self, success, message):
        future = self.capture_future
        if future is not None and not future.done():
            self.publish_status(message)
            future.set_result(Trigger.Response(success=success, message=message))

    def pause(self, request, response):
        if self.state == 'idle':
            return Trigger.Response(success=False, message='No mapping session')
        self.state = 'paused'
        self.finish_capture(False, 'Mapping paused')
        self.publish_status('Map frozen')
        return Trigger.Response(success=True, message='Map frozen')

    def reset(self, request, response):
        self.clear('Map cleared')
        return Trigger.Response(success=True, message='Map cleared')

    def save(self, request, response):
        if self.state != 'paused':
            return Trigger.Response(success=False, message='Pause mapping before saving')
        path = self.get_parameter('save_path').value
        try:
            if not path.endswith('.npz'):
                raise ValueError('save_path must end in .npz')
            path = self.map.save(path, self.settings['frame_id'],
                                 self.settings['camera_frame'], self.selection_id)
        except (OSError, ValueError) as error:
            return Trigger.Response(success=False, message=str(error))
        return Trigger.Response(success=True, message=str(path))

    def receive(self, message):
        if self.capture_future is None or self.capture_future.done():
            return
        stamp = Time.from_msg(message.header.stamp)
        age = self.get_clock().now().nanoseconds - stamp.nanoseconds
        if stamp.nanoseconds <= self.capture_stamp or not 0 <= age <= self.max_age:
            return
        try:
            reason = self.add_view(message, stamp)
        except (TransformException, ValueError, AssertionError) as error:
            reason = str(error)
        if reason is not None:
            self.capture_rejection = reason
            self.publish_status(reason)
            return
        self.publish_cloud(message.header.stamp)
        self.finish_capture(True, f'Captured view {len(self.map.views)}')

    def add_view(self, message, stamp):
        frame = self.settings['frame_id']
        if message.header.frame_id != frame:
            return f'Expected target cloud in {frame}'
        transform = self.tf.lookup_transform(frame, self.settings['camera_frame'], stamp).transform
        t, q = transform.translation, transform.rotation
        pose = (t.x, t.y, t.z, q.x, q.y, q.z, q.w)
        points = read_points_numpy(message, field_names=('x', 'y', 'z'), skip_nans=True)
        return self.map.add(points, stamp.nanoseconds, pose)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = TargetMapper()
        try:
            rclpy.spin(node)
        finally:
            node.destroy_node()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        rclpy.try_shutdown()
