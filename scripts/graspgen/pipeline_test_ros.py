from collections import deque
from pathlib import Path
import shutil
import time

from builtin_interfaces.msg import Time
from geometry_msgs.msg import PointStamped
import numpy as np
import rclpy
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time as RosTime
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py.point_cloud2 import read_points_numpy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from data import save_cloud
from pick_geometry import observation_pose
from pick_moveit import PlanningFailure
from rviz import candidate_markers

FRAME = 'panda_link0'
CAMERA = 'wrist_color_optical_frame'
AZIMUTHS = (0, 90, 180, 270)


def stamp_ns(message):
    return RosTime.from_msg(message.header.stamp).nanoseconds


class ScanClient:
    def __init__(self, node):
        self.node = node
        self.log = node.get_logger().info
        self.selection = ''
        self.centroid = None
        self.map_cloud = None
        self.environment = deque(maxlen=12)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(String, '/pointcloud/selection_id', self.receive_selection, latched)
        node.create_subscription(PointStamped, '/pointcloud/target_centroid',
                                 lambda msg: setattr(self, 'centroid', msg), qos_profile_sensor_data)
        node.create_subscription(PointCloud2, '/pointcloud/target_map',
                                 lambda msg: setattr(self, 'map_cloud', msg), latched)
        node.create_subscription(PointCloud2, '/pointcloud/environment',
                                 lambda msg: self.environment.append(msg), 5)
        self.services = {
            name: node.create_client(Trigger, f'/pointcloud/mapping/{name}')
            for name in ('reset', 'start', 'capture', 'pause', 'save')
        }
        self.markers = node.create_publisher(MarkerArray, '/graspgen/markers', latched)

    def receive_selection(self, message):
        if message.data != self.selection:
            self.selection = message.data
            self.centroid = None

    def wait_for(self, find, description, selection=None):
        deadline = time.monotonic() + self.node.args.timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            if selection is not None:
                self.require_selection(selection)
            result = find()
            if result is not None:
                return result
        raise TimeoutError(f'Timed out waiting for {description}')

    def selected_target(self):
        self.log('Waiting for a fresh centroid from the target selected in RViz')
        self.node.current_state()
        cutoff = self.node.get_clock().now().nanoseconds

        def current_centroid():
            if self.selection and self.centroid is not None and stamp_ns(self.centroid) > cutoff:
                return self.centroid

        message = self.wait_for(current_centroid, 'a selected target centroid')
        point = message.point
        center = np.array([point.x, point.y, point.z])
        if message.header.frame_id != FRAME or not np.isfinite(center).all():
            raise ValueError(f'Expected a finite target centroid in {FRAME}')
        return self.selection, center

    def require_selection(self, expected):
        if self.selection != expected:
            raise RuntimeError('RViz target selection changed during the test')

    def check_selection(self, expected):
        self.node.current_state()
        self.require_selection(expected)

    def mapping(self, action):
        self.log(f'Requesting mapper {action}')
        response = self.node.call(self.services[action], Trigger.Request())
        if not response.success:
            raise RuntimeError(f'Mapper {action} failed. {response.message}')
        self.log(response.message)
        return response.message

    def move_camera(self, center, angle, camera_transform, selection):
        args = self.node.args
        start = self.node.current_state()
        self.require_selection(selection)
        for roll in (0, 45, -45, 90, -90, 135, -135, 180):
            pose = observation_pose(center, angle, args.radius, args.height, camera_transform, roll)
            self.log(f'Planning view {angle}° with roll {roll}° at tool XYZ {pose[:3, 3].round(3).tolist()} m')
            try:
                plan = self.node.pregrasp_plan(pose, start)
            except PlanningFailure as error:
                self.log(f'View {angle}° roll {roll}° rejected. {error}')
                continue
            self.require_selection(selection)
            self.log(f'Executing view {angle}°')
            self.node.execute(plan)
            return
        raise PlanningFailure(f'No reachable collision-free observation pose for {angle}°')

    def capture_view(self, angle, selection):
        cutoff = self.node.get_clock().now().nanoseconds
        self.mapping('capture')

        def matching_environment():
            cloud = self.map_cloud
            if cloud is None or not cloud.width or stamp_ns(cloud) <= cutoff:
                return None
            return next((msg for msg in self.environment if msg.header == cloud.header), None)

        message = self.wait_for(matching_environment, f'environment cloud for view {angle}°', selection)
        points = read_points_numpy(message, field_names=('x', 'y', 'z'), skip_nans=True)
        return stamp_ns(message), np.asarray(points).reshape(-1, 3)

    def map_target(self, selection, center, directory):
        camera = self.node.lookup('panda_link8', CAMERA)
        self.mapping('reset')
        self.mapping('start')
        snapshots = {}
        try:
            for angle in AZIMUTHS:
                self.move_camera(center, angle, camera, selection)
                snapshots[angle] = self.capture_view(angle, selection)
        except BaseException:
            try:
                self.mapping('pause')
            except (RuntimeError, OSError) as error:
                self.node.get_logger().warning(str(error))
            raise
        self.mapping('pause')
        self.require_selection(selection)
        path = Path(self.mapping('save'))
        archive = directory / 'target_map.npz'
        shutil.copyfile(path, archive)
        self.save_scenes(archive, snapshots)

    def save_scenes(self, archive, snapshots):
        with np.load(archive, allow_pickle=False) as data:
            offsets = data['view_offsets']
            views = data['view_points']
            stamps = data['view_stamps_ns']
            self.log(f'Fused object map contains {len(data["points"])} points')
        if list(stamps) != [stamp for stamp, _ in snapshots.values()]:
            raise RuntimeError('Saved map timestamps do not match the four scene snapshots')
        for index, (angle, (stamp, environment)) in enumerate(snapshots.items()):
            target = views[offsets[index]:offsets[index + 1]]
            path = archive.parent / f'scene_{angle:03}.npz'
            save_cloud(path, np.concatenate((environment, target)),
                       dict(frame_id=FRAME, units='meters', stamp_ns=stamp, azimuth_deg=angle,
                            environment_point_count=len(environment)))
            self.log(f'Saved {len(environment) + len(target)} scene points for {angle}° to {path}')

    def show_grasps(self, grasps, scores):
        markers = [Marker(action=Marker.DELETEALL)]
        for rank, (grasp, score) in enumerate(zip(grasps, scores)):
            markers.extend(candidate_markers(grasp, score, rank, FRAME, Time(), self.node.args.approach))
        self.markers.publish(MarkerArray(markers=markers))

    def clear_markers(self):
        self.markers.publish(MarkerArray(markers=[Marker(action=Marker.DELETEALL)]))
