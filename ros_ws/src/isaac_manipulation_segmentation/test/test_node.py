import importlib
import sys
import time
from types import ModuleType

import numpy as np
import pytest
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


@pytest.mark.parametrize('overlay_enabled', [True, False])
def test_ros_image_pipeline(monkeypatch, overlay_enabled):
    class Segmenter:
        def __init__(self, *args):
            pass

        def predict(self, rgb):
            return {'masks': np.ones((1, *rgb.shape[:2]), dtype=bool),
                    'boxes': np.array([[0., 0., 4., 3.]]), 'scores': np.array([0.9]),
                    'labels': np.array([1])}

    inference = ModuleType('isaac_manipulation_segmentation.inference')
    inference.Segmenter = Segmenter
    monkeypatch.setitem(sys.modules, inference.__name__, inference)
    sys.modules.pop('isaac_manipulation_segmentation.node', None)
    module = importlib.import_module('isaac_manipulation_segmentation.node')
    rclpy.init(args=['--ros-args', '-p', 'checkpoint:=test.pt', '-p', 'use_sim_time:=false',
                    '-p', 'image_topic:=/test_segmentation/image', '-p',
                    f'publish_overlay_image:={str(overlay_enabled).lower()}'])
    node = peer = executor = None
    try:
        node = module.SegmentationNode()
        assert (node.overlay is not None) == overlay_enabled
        peer = Node('segmentation_test')
        masks, detections, overlays = [], [], []
        peer.create_subscription(Image, '/segmentation/instance_mask', masks.append, 1)
        peer.create_subscription(Detection2DArray, '/segmentation/detections', detections.append, 1)
        peer.create_subscription(Image, '/segmentation/overlay_image', overlays.append, 1)
        publisher = peer.create_publisher(Image, '/test_segmentation/image', 1)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(peer)
        bridge = CvBridge()
        image = bridge.cv2_to_imgmsg(np.zeros((3, 4, 3), dtype=np.uint8), encoding='rgb8')
        image.header.frame_id = 'test_camera'
        image.header.stamp.sec = 42
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            publisher.publish(image)
            executor.spin_once(timeout_sec=0.02)
            if masks and detections and (overlays or not overlay_enabled):
                break
        assert masks and detections
        assert masks[0].header == image.header
        assert masks[0].encoding == '16UC1'
        np.testing.assert_array_equal(bridge.imgmsg_to_cv2(masks[0]), np.ones((3, 4)))
        assert detections[0].detections[0].id == '1'
        detection = detections[0].detections[0]
        assert detection.bbox.center.position.x == 2.0
        assert detection.bbox.center.position.y == 1.5
        assert detection.results[0].hypothesis.score == 0.9
        assert detections[0].header == image.header
        empty = {'boxes': [], 'scores': [], 'labels': []}
        assert module.detections_message(empty, image.header).detections == []
        assert bool(overlays) == overlay_enabled
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if peer is not None:
            peer.destroy_node()
        rclpy.shutdown()
        sys.modules.pop('isaac_manipulation_segmentation.node', None)
