import json
from time import monotonic

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger

from isaac_manipulation_pointcloud.mapper_node import TargetMapper


def send_view(node, points, camera_x):
    header = Header(frame_id='panda_link0', stamp=node.get_clock().now().to_msg())
    transform = TransformStamped(header=header, child_frame_id='wrist_color_optical_frame')
    transform.transform.translation.x = camera_x
    transform.transform.translation.z = 0.3
    transform.transform.rotation.w = 1.0
    node.tf.set_transform(transform, 'test')
    node.receive(create_cloud_xyz32(header, points))


def test_mapping_lifecycle_and_export(tmp_path):
    path = tmp_path / 'map.npz'
    rclpy.init(args=['--ros-args', '-p', f'save_path:={path}', '-p', 'capture_timeout:=0.2'])
    node = TargetMapper()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    request = Trigger.Request()
    response = Trigger.Response()

    def request_capture():
        result = executor.create_task(node.capture, request, response)
        deadline = monotonic() + 1.0
        while node.capture_future is None and monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert node.capture_future is not None
        return result

    def capture_result(result):
        executor.spin_until_future_complete(result, timeout_sec=1.0)
        assert result.done()
        return result.result()

    try:
        assert not node.start(request, response).success
        node.selection_changed(String(data='target-1'))
        assert node.start(request, response).success
        x, y = np.meshgrid(np.arange(8) * 0.004, np.arange(8) * 0.004)
        seed = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        send_view(node, seed, 0.0)
        assert len(node.map.views) == 0

        result = request_capture()
        send_view(node, seed, 0.0)
        send_view(node, seed + [0.008, 0, 0], 0.1)
        assert capture_result(result).success
        assert len(node.map.views) == 1

        result = request_capture()
        send_view(node, seed, 0.0)
        duplicate = capture_result(result)
        assert not duplicate.success
        assert 'already captured' in duplicate.message

        result = request_capture()
        send_view(node, seed + [0.15, 0, 0], 0.1)
        assert len(node.map.views) == 1
        send_view(node, seed + [0.008, 0, 0], 0.1)
        assert capture_result(result).success
        assert len(node.map.views) == 2
        assert len(node.map.points()) > len(seed)
        assert not node.save(request, response).success

        result = request_capture()
        assert node.pause(request, response).success
        assert not capture_result(result).success
        send_view(node, seed, -0.1)
        assert node.save(request, response).success
        with np.load(path, allow_pickle=False) as data:
            np.testing.assert_array_equal(data['points'], node.map.points())
            np.testing.assert_array_equal(data['view_offsets'], [0, len(seed), 2 * len(seed)])
            assert set(data['voxel_view_ids']) == {0, 1}
            assert json.loads(data['metadata'].item())['frame_id'] == 'panda_link0'

        node.selection_changed(String(data='target-2'))
        assert node.state == 'idle'
        assert len(node.map.points()) == 0
        assert not node.save(request, response).success
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
