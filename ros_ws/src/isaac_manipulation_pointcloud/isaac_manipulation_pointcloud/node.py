from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker

from isaac_manipulation_pointcloud_interfaces.srv import SelectInstance, SelectTarget
from .processing import ObservationCache, bounds, candidates, partition, project_depth, transform_points
from .selection import Selection


def stamp_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


class PointCloudNode(Node):
    def __init__(self):
        super().__init__('pointcloud')
        path = Path(get_package_share_directory('isaac_manipulation_pointcloud')) / 'config/params.yaml'
        defaults = yaml.safe_load(path.read_text())['pointcloud']['ros__parameters']
        for name, value in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, value)
        self.settings = {name: self.get_parameter(name).value for name in defaults}
        p = self.settings
        for name in ('selection_distance', 'association_distance', 'maximum_observation_age',
                     'maximum_depth', 'minimum_points', 'cache_capacity'):
            if not np.isfinite(p[name]) or p[name] <= 0:
                raise ValueError(f'{name} must be positive and finite')
        for name in ('ambiguity_margin', 'synchronization_tolerance', 'minimum_depth', 'voxel_resolution'):
            if not np.isfinite(p[name]) or p[name] < 0:
                raise ValueError(f'{name} must be nonnegative and finite')
        if p['minimum_depth'] >= p['maximum_depth'] or not p['output_frame']:
            raise ValueError('Invalid depth range or output frame')
        if p['input_cloud_topic']:
            raise ValueError('This camera integration uses aligned depth; input_cloud_topic must be empty')
        self.age = int(p['maximum_observation_age'] * 1e9)
        def cache():
            return ObservationCache(p['cache_capacity'], self.age, int(p['synchronization_tolerance'] * 1e9))
        self.depths, self.infos, self.masks, self.observations = [cache() for _ in range(4)]
        self.pending_depths = set()
        self.latest = None
        self.last_stamp = None
        self.selection = Selection(p['selection_distance'], p['association_distance'], p['ambiguity_margin'])
        self.bridge = CvBridge()
        self.tf = Buffer(cache_time=Duration(seconds=p['maximum_observation_age'] + 2))
        self.listener = TransformListener(self.tf, self)
        self.environment = self.create_publisher(PointCloud2, p['environment_topic'], 1)
        self.target = self.create_publisher(PointCloud2, p['target_topic'], 1)
        self.centroid = self.create_publisher(PointStamped, p['centroid_topic'], 1)
        self.box = self.create_publisher(Marker, p['bounding_box_topic'], 1)
        self.create_subscription(Image, p['depth_topic'], self.depth_callback, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, p['camera_info_topic'], self.info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, p['mask_topic'], self.mask_callback, qos_profile_sensor_data)
        self.create_service(SelectTarget, '/pointcloud/select_target', self.select_callback)
        self.create_service(SelectInstance, '/pointcloud/select_instance', self.instance_callback)
        self.create_service(Trigger, '/pointcloud/clear_target', self.clear_callback)
        self.create_timer(0.02, self.process)

    def fresh(self, stamp):
        return 0 <= self.get_clock().now().nanoseconds - stamp <= self.age

    def depth_callback(self, message):
        stamp = stamp_ns(message.header.stamp)
        if self.last_stamp is not None and stamp < self.last_stamp:
            for cache in (self.depths, self.infos, self.masks, self.observations):
                cache.items.clear()
            self.pending_depths.clear()
            self.latest = None
            self.selection.clear()
            self.delete_marker(message.header)
        self.last_stamp = stamp
        self.depths.add(stamp, message)
        self.pending_depths.add(stamp)
        self.pending_depths.intersection_update(self.depths.items)

    def info_callback(self, message):
        self.infos.add(stamp_ns(message.header.stamp), message)

    def mask_callback(self, message):
        self.masks.add(stamp_ns(message.header.stamp), message)

    def transform(self, points, frame, stamp):
        if frame == self.settings['output_frame']:
            return points
        if not frame or stamp_ns(stamp) == 0:
            raise ValueError('Missing frame or acquisition timestamp')
        t = self.tf.lookup_transform(self.settings['output_frame'], frame, Time.from_msg(stamp)).transform
        return transform_points(points, [t.translation.x, t.translation.y, t.translation.z],
                                [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w])

    def scene(self, depth, info):
        if (depth.header.frame_id != info.header.frame_id or
                (depth.width, depth.height) != (info.width, info.height)):
            raise ValueError('Depth and CameraInfo must have identical projection and frame')
        if depth.encoding not in ('32FC1', '16UC1'):
            raise ValueError('Depth must be 32FC1 meters or 16UC1 millimeters')
        image = self.bridge.imgmsg_to_cv2(depth).astype(np.float32)
        if depth.encoding == '16UC1':
            image *= 0.001
        points, valid = project_depth(image, info.k, self.settings['minimum_depth'], self.settings['maximum_depth'])
        points = self.transform(points[valid], depth.header.frame_id, depth.header.stamp)
        header = deepcopy(depth.header)
        header.frame_id = self.settings['output_frame']
        return points, valid, header

    def process(self):
        for stamp in sorted(self.depths.items):
            if not self.fresh(stamp):
                self.pending_depths.discard(stamp)
                self.masks.items.pop(stamp, None)
                continue
            depth = self.depths.items[stamp]
            mask = self.masks.nearest(stamp)
            if stamp not in self.pending_depths and mask is None:
                continue
            info = self.infos.nearest(stamp)
            if info is None:
                continue
            try:
                points, valid, header = self.scene(depth, info)
                if stamp in self.pending_depths:
                    if self.selection.reference is None:
                        self.publish_cloud(self.environment, header, points)
                    self.pending_depths.discard(stamp)
                if mask is None:
                    continue
                if mask.encoding != '16UC1' or mask.header.frame_id != depth.header.frame_id:
                    raise ValueError('Mask must be 16UC1 in the aligned depth frame')
                labels = self.bridge.imgmsg_to_cv2(mask)
                if labels.shape != valid.shape:
                    raise ValueError('Mask and aligned depth dimensions differ')
                labels = labels[valid]
                resolution = self.settings['voxel_resolution']
                if resolution > 0:
                    # Include labels in the key so voxelization cannot merge objects.
                    keys = np.column_stack((np.floor(points / resolution), labels))
                    _, indices = np.unique(keys, axis=0, return_index=True)
                    points, labels = points[indices], labels[indices]
                centers = candidates(points, labels, self.settings['minimum_points'])
                observation = (points, labels, centers, header)
                self.observations.add(stamp_ns(mask.header.stamp), observation)
                if self.settings['debug']:
                    self.get_logger().info(f'Synchronized {stamp}: {len(points)} points, {len(centers)} candidates')
                if self.latest is None or stamp >= self.latest[0]:
                    self.latest = (stamp, observation)
                self.masks.items.pop(stamp_ns(mask.header.stamp), None)
                if self.selection.reference is not None:
                    self.publish_observation(observation)
            except (TransformException, ValueError) as exc:
                self.get_logger().warning(str(exc), throttle_duration_sec=5)

    def publish_cloud(self, publisher, header, points):
        publisher.publish(create_cloud_xyz32(header, np.asarray(points, dtype=np.float32)))

    def publish_observation(self, observation):
        points, labels, centers, header = observation
        label = self.selection.match(centers)
        target, environment = partition(points, labels, label)
        self.publish_cloud(self.environment, header, environment)
        if label is None:
            self.delete_marker(header)
            self.get_logger().warning('Selected object unavailable or ambiguous', throttle_duration_sec=5)
            return
        self.publish_cloud(self.target, header, target)
        center, box_center, dimensions = bounds(target)
        message = PointStamped(header=header)
        message.point.x, message.point.y, message.point.z = map(float, center)
        self.centroid.publish(message)
        if self.settings['visualization']:
            marker = Marker(header=header, ns='target', id=0, type=Marker.CUBE, action=Marker.ADD)
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, box_center)
            marker.pose.orientation.w = 1.0
            marker.scale.x, marker.scale.y, marker.scale.z = map(float, np.maximum(dimensions, 0.001))
            marker.color.r, marker.color.a = 1.0, 0.35
            marker.lifetime = Duration(seconds=0.5).to_msg()
            self.box.publish(marker)

    def delete_marker(self, header):
        self.box.publish(Marker(header=header, ns='target', id=0, action=Marker.DELETE))

    def select_callback(self, request, response):
        try:
            if self.latest is None or not self.fresh(self.latest[0]):
                raise ValueError('No fresh synchronized observation')
            point = request.reference_position
            xyz = np.array([point.point.x, point.point.y, point.point.z])
            if not np.isfinite(xyz).all():
                raise ValueError('Reference coordinates must be finite')
            xyz = self.transform(xyz, point.header.frame_id, point.header.stamp)
            label = self.selection.select(self.latest[1][2], xyz)
            if label is None:
                raise ValueError('No unique candidate within selection distance')
            response.success, response.message = True, f'Selected instance {label}'
            self.publish_observation(self.latest[1])
        except (TransformException, ValueError) as exc:
            response.success, response.message = False, str(exc)
        return response

    def instance_callback(self, request, response):
        stamp = stamp_ns(request.stamp)
        observation = self.observations.items.get(stamp)
        if observation is None or not self.fresh(stamp) or request.instance_id not in observation[2]:
            response.success, response.message = False, 'Exact fresh frame and valid instance required'
        else:
            center = observation[2][request.instance_id]
            label = self.selection.select(observation[2], center)
            response.success = label == request.instance_id
            response.message = 'Selected' if response.success else 'Ambiguous instance'
            if response.success:
                self.publish_observation(observation)
        return response

    def clear_callback(self, request, response):
        self.selection.clear()
        if self.latest is not None:
            points, _, _, header = self.latest[1]
            self.delete_marker(header)
            if self.fresh(self.latest[0]):
                self.publish_cloud(self.environment, header, points)
        response.success, response.message = True, 'Target cleared; IDLE'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
