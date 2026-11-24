from copy import deepcopy

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    AllowedCollisionEntry, AttachedCollisionObject, CollisionObject, PlanningScene,
    PlanningSceneComponents)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
import numpy as np
from scipy.spatial.transform import Rotation
from shape_msgs.msg import SolidPrimitive

from pick_geometry import transform

TABLE_ID = 'graspgen_test_table'
TARGET_ID = 'graspgen_test_target'


def pose_message(matrix):
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = map(float, matrix[:3, 3])
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = map(float, quaternion)
    return pose


def box_object(name, frame, center, size):
    obj = CollisionObject()
    obj.header.frame_id = frame
    obj.id = name
    obj.pose = pose_message(transform(center))
    obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=list(map(float, size)))]
    obj.primitive_poses = [pose_message(np.eye(4))]
    obj.operation = CollisionObject.ADD
    return obj


def allow_pairs(matrix, name, partners):
    matrix = deepcopy(matrix)
    missing = [entry for entry in [name, *partners] if entry not in matrix.entry_names]
    missing = list(dict.fromkeys(missing))
    matrix.entry_names.extend(missing)
    size = len(matrix.entry_names)
    rows = [list(row.enabled) + [False] * len(missing) for row in matrix.entry_values]
    rows.extend([[False] * size for _ in missing])
    matrix.entry_values = [AllowedCollisionEntry(enabled=row) for row in rows]
    index = matrix.entry_names.index(name)
    for partner in partners:
        other = matrix.entry_names.index(partner)
        matrix.entry_values[index].enabled[other] = True
        matrix.entry_values[other].enabled[index] = True
    return matrix


class TestScene:
    def __init__(self, node, frame, args):
        self.node = node
        self.installed = False
        self.get = node.create_client(GetPlanningScene, '/get_planning_scene')
        self.apply = node.create_client(ApplyPlanningScene, '/apply_planning_scene')
        request = GetPlanningScene.Request()
        request.components.components = (PlanningSceneComponents.ALLOWED_COLLISION_MATRIX |
                                         PlanningSceneComponents.WORLD_OBJECT_NAMES |
                                         PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        previous = node.call(self.get, request).scene
        occupied = {obj.id for obj in previous.world.collision_objects}
        occupied.update(obj.object.id for obj in previous.robot_state.attached_collision_objects)
        if TABLE_ID in occupied:
            raise RuntimeError('A previous graspgen test scene still exists; remove its objects before retrying')
        self.remove_legacy_target(previous)
        self.original_acm = previous.allowed_collision_matrix
        self.base_acm = allow_pairs(self.original_acm, TABLE_ID, ['panda_link0'])
        self.table = box_object(TABLE_ID, frame,
                                [*args.table_center, args.table_height - args.table_thickness / 2],
                                [*args.table_size, args.table_thickness])

    def update(self, objects=(), attached=(), acm=None):
        scene = PlanningScene(is_diff=True)
        scene.robot_state.is_diff = True
        scene.world.collision_objects = list(objects)
        scene.robot_state.attached_collision_objects = list(attached)
        if acm is not None:
            scene.allowed_collision_matrix = acm
        if not self.node.call(self.apply, ApplyPlanningScene.Request(scene=scene)).success:
            raise RuntimeError('MoveIt rejected the planning-scene update')

    def remove_legacy_target(self, previous):
        attached = [obj for obj in previous.robot_state.attached_collision_objects
                    if obj.object.id == TARGET_ID]
        for obj in attached:
            removal = AttachedCollisionObject(link_name=obj.link_name)
            removal.object.id = TARGET_ID
            removal.object.operation = CollisionObject.REMOVE
            self.update(attached=[removal])
        if attached or any(obj.id == TARGET_ID for obj in previous.world.collision_objects):
            self.update(objects=[CollisionObject(id=TARGET_ID, operation=CollisionObject.REMOVE)])

    def reset(self):
        self.update(objects=[self.table], acm=self.base_acm)
        self.installed = True

    def cleanup(self):
        if self.installed:
            self.update(objects=[CollisionObject(id=TABLE_ID, operation=CollisionObject.REMOVE)],
                        acm=self.original_acm)
