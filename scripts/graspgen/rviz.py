from geometry_msgs.msg import Point
import numpy as np
from visualization_msgs.msg import Marker

from grasp_markers import approach_line, gripper_lines


def marker_points(points):
    return [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in points]


def base_marker(frame, stamp, index, kind):
    marker = Marker(ns='graspgen_candidates', id=index, type=kind, action=Marker.ADD)
    marker.header.frame_id = frame
    marker.header.stamp = stamp
    marker.pose.orientation.w = 1.0
    marker.color.a = 1.0
    return marker


def candidate_markers(grasp, score, rank, frame, stamp, approach):
    wire = base_marker(frame, stamp, rank * 3, Marker.LINE_LIST)
    wire.points = marker_points(gripper_lines(grasp))
    wire.scale.x = .002
    wire.color.r, wire.color.g, wire.color.b = float(1 - score), .3, float(score)
    arrow = base_marker(frame, stamp, rank * 3 + 1, Marker.ARROW)
    arrow.points = marker_points(approach_line(grasp, approach))
    arrow.scale.x, arrow.scale.y, arrow.scale.z = .003, .008, .012
    arrow.color.r, arrow.color.g = 1.0, .6
    label = base_marker(frame, stamp, rank * 3 + 2, Marker.TEXT_VIEW_FACING)
    label.pose.position = marker_points([grasp[:3, 3] + [0, 0, .025]])[0]
    label.scale.z = .015
    label.color.r = label.color.g = label.color.b = 1.0
    label.text = f'{rank}: {score:.3f}'
    return [wire, arrow, label]
