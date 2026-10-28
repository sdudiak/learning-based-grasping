import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .configuration import load_defaults, validate_parameters
from .inference import Segmenter
from .processing import FrameProcessor, OverlayTracker, instance_mask, render_overlay


def detections_message(result, header):
    message = Detection2DArray(header=header)
    for index, (box, score, label) in enumerate(zip(result['boxes'], result['scores'], result['labels']), start=1):
        detection = Detection2D(header=header, id=str(index))
        x1, y1, x2, y2 = map(float, box)
        detection.bbox.center.position.x = (x1 + x2) / 2
        detection.bbox.center.position.y = (y1 + y2) / 2
        detection.bbox.size_x = x2 - x1
        detection.bbox.size_y = y2 - y1
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = str(int(label))
        hypothesis.hypothesis.score = float(score)
        detection.results.append(hypothesis)
        message.detections.append(detection)
    return message


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation')
        defaults = load_defaults()
        for name, default in defaults.items():
            if not self.has_parameter(name):
                self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
        settings = {name: self.get_parameter(name).value for name in defaults}
        validate_parameters(settings)
        self.segmenter = Segmenter(settings['checkpoint'], settings['device'],
                                   settings['score_threshold'], settings['mask_threshold'])
        self.bridge = CvBridge()
        self.detections = self.create_publisher(Detection2DArray, settings['detections_topic'], 1)
        self.masks = self.create_publisher(Image, settings['instance_mask_topic'], 1)
        self.overlay = (self.create_publisher(Image, settings['overlay_image_topic'], 1)
                        if settings['publish_overlay_image'] else None)
        self.color_tracker = OverlayTracker(settings['tracking_iou_threshold'], settings['tracking_max_missed_frames'])
        self._last_frame = None
        self._last_stamp = None
        self.subscription = self.create_subscription(
            Image, settings['image_topic'], lambda message: self.processor.submit(message), qos_profile_sensor_data)
        self.processor = FrameProcessor(settings['decimation'], self.process_image)
        self.get_logger().info(f"Segmenting {settings['image_topic']} with decimation={settings['decimation']}")

    def process_image(self, message):
        try:
            rgb = self.bridge.imgmsg_to_cv2(message, desired_encoding='rgb8')
            result = self.segmenter.predict(rgb)
            if not self.context.ok():
                return
            instances = instance_mask(result['masks'], result['scores'])
            self.detections.publish(detections_message(result, message.header))
            self.publish_image(self.masks, instances, '16UC1', message.header)
            if self.overlay is not None:
                frame = (message.header.frame_id, message.width, message.height)
                stamp = (message.header.stamp.sec, message.header.stamp.nanosec)
                if frame != self._last_frame or (self._last_stamp is not None and stamp < self._last_stamp):
                    self.color_tracker.reset()
                self._last_frame, self._last_stamp = frame, stamp
                color_ids = self.color_tracker.update(result['boxes'], result['labels'])
                overlay = render_overlay(rgb, instances, result['boxes'], result['scores'], color_ids)
                self.publish_image(self.overlay, overlay, 'rgb8', message.header)
        except Exception as error:
            if self.context.ok():
                self.get_logger().error(f'Image segmentation failed: {error}')

    def publish_image(self, publisher, image, encoding, header):
        message = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        message.header = header
        publisher.publish(message)

    def destroy_node(self):
        self.processor.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SegmentationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
