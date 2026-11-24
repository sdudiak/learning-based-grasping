import numpy as np


def gripper_lines(grasp):
    vertices = np.array([[0, 0, 0], [0, 0, .0584],
                         [-.04, 0, .0584], [.04, 0, .0584],
                         [-.04, 0, .1053], [.04, 0, .1053]])
    vertices = vertices @ grasp[:3, :3].T + grasp[:3, 3]
    return vertices[[0, 1, 2, 3, 2, 4, 3, 5]]


def approach_line(grasp, distance):
    origin = grasp[:3, 3]
    return np.array([origin - distance * grasp[:3, 2], origin])
