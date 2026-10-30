import time

import numpy as np
import rclpy
from sensor_msgs.msg import CameraInfo
from std_srvs.srv import Trigger
from isaac_manipulation_pointcloud_interfaces.srv import SelectTarget
from isaac_manipulation_pointcloud.node import PointCloudNode


class Recorder:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


def test_idle_selection_loss_clear_and_nonblocking_services():
    rclpy.init(args=['--ros-args', '-p', 'use_sim_time:=false', '-p', 'output_frame:=camera',
                     '-p', 'minimum_points:=1'])
    node = PointCloudNode()
    try:
        for name in ('environment', 'target', 'centroid', 'box'):
            setattr(node, name, Recorder())
        request = SelectTarget.Request()
        request.reference_position.header.frame_id = 'camera'
        request.reference_position.point.z = 1.0
        start = time.monotonic()
        assert not node.select_callback(request, SelectTarget.Response()).success
        assert time.monotonic() - start < 0.2
        stamp = node.get_clock().now().to_msg()
        depth = node.bridge.cv2_to_imgmsg(np.ones((2, 2), dtype=np.float32), encoding='32FC1')
        depth.header.frame_id, depth.header.stamp = 'camera', stamp
        info = CameraInfo(header=depth.header, width=2, height=2,
                          k=[100., 0., 0., 0., 100., 0., 0., 0., 1.])
        mask = node.bridge.cv2_to_imgmsg(np.array([[1, 1], [0, 2]], dtype=np.uint16), encoding='16UC1')
        mask.header = depth.header
        node.depth_callback(depth)
        node.process()
        assert not node.environment.messages
        node.info_callback(info)
        node.process()
        assert node.environment.messages[-1].width == 4
        assert not node.target.messages
        # A mask from a different source exposure must not be used.
        mask.header = type(depth.header)(frame_id='camera', stamp=type(stamp)(sec=stamp.sec, nanosec=stamp.nanosec + 1))
        node.mask_callback(mask)
        node.process()
        assert node.latest is None
        mask.header = depth.header
        node.mask_callback(mask)
        node.process()
        node.selection.ambiguity = 0.001
        assert node.select_callback(request, SelectTarget.Response()).success
        assert node.target.messages[-1].width == 2
        assert node.environment.messages[-1].width == 2
        assert node.centroid.messages[-1].header == depth.header
        observation = node.latest[1]
        node.publish_observation((observation[0], observation[1], {2: np.array([1., 0, 1])}, observation[3]))
        assert len(node.target.messages) == 1
        assert node.environment.messages[-1].width == 4
        assert node.clear_callback(Trigger.Request(), Trigger.Response()).success
        assert node.selection.reference is None
        assert node.environment.messages[-1].width == 4
        request.reference_position.header.frame_id = 'missing_frame'
        request.reference_position.header.stamp = stamp
        start = time.monotonic()
        assert not node.select_callback(request, SelectTarget.Response()).success
        assert time.monotonic() - start < 0.2
    finally:
        node.destroy_node()
        rclpy.shutdown()
