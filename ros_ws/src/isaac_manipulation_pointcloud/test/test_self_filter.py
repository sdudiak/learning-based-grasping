import numpy as np
import pytest

from isaac_manipulation_pointcloud.processing import partition
from isaac_manipulation_pointcloud.self_filter import exclusion_mask, validate_geometry


def test_box_moves_with_tf_preserves_neighbor_and_partition():
    geometry = validate_geometry([dict(link='finger', shape='box', size=[.02, .02, .1])])
    points = np.array([[0, 0, 0], [.014, 0, 0], [.5, 0, 0]])
    excluded = exclusion_mask(points, geometry, {'finger': ([0, 0, 0], [0, 0, 0, 1])}, .002)
    np.testing.assert_array_equal(excluded, [True, False, False])
    target, environment = partition(points[~excluded], np.array([0, 1, 2])[~excluded], 1)
    np.testing.assert_array_equal(target, points[1:2])
    np.testing.assert_array_equal(environment, points[2:])
    moved = exclusion_mask(points, geometry, {'finger': ([-.5, 0, 0], [0, 0, 0, 1])})
    np.testing.assert_array_equal(moved, [False, False, True])


def test_rotated_geometry_and_primitives():
    geometry = validate_geometry([dict(link='hand', shape='box', size=[.2, .02, .02],
                                       quaternion_xyzw=[0, 0, 2**-.5, 2**-.5])])
    result = exclusion_mask(np.array([[0, .08, 0], [.08, 0, 0]]), geometry,
                            {'hand': ([0, 0, 0], [0, 0, 0, 1])})
    np.testing.assert_array_equal(result, [True, False])
    for shape, size in [('sphere', [.1]), ('cylinder', [.1, .2])]:
        geometry = validate_geometry([dict(link='hand', shape=shape, size=size)])
        result = exclusion_mask(np.array([[0, 0, 0], [.2, 0, 0]]), geometry,
                                {'hand': ([0, 0, 0], [0, 0, 0, 1])})
        np.testing.assert_array_equal(result, [True, False])


def test_invalid_geometry_and_missing_tf():
    with pytest.raises(ValueError):
        validate_geometry([dict(link='hand', shape='box', size=[-1, 1, 1])])
    geometry = validate_geometry([dict(link='hand', shape='sphere', size=[.1])])
    with pytest.raises(KeyError):
        exclusion_mask(np.zeros((1, 3)), geometry, {})
    with pytest.raises(ValueError):
        exclusion_mask(np.zeros((1, 3)), geometry, {}, -.1)


def test_mesh_surface_tolerance_preserves_nearby_external_points(tmp_path):
    path = tmp_path / 'tetra.stl'
    triangles = np.array([[[0, 0, 0], [0, 1, 0], [1, 0, 0]],
                          [[0, 0, 0], [1, 0, 0], [0, 0, 1]],
                          [[0, 0, 0], [0, 0, 1], [0, 1, 0]],
                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=float)
    lines = ['solid tetra']
    for triangle in triangles:
        lines.extend(['facet normal 0 0 1', 'outer loop'])
        lines.extend('vertex ' + ' '.join(map(str, vertex)) for vertex in triangle)
        lines.extend(['endloop', 'endfacet'])
    lines.append('endsolid tetra')
    path.write_text('\n'.join(lines))
    geometry = validate_geometry([dict(link='finger', shape='mesh', mesh=str(path))])
    points = np.array([[.1, .1, .1], [-.001, .1, .1], [-.004, .1, .1], [.8, .8, .8]])
    excluded = exclusion_mask(points, geometry, {'finger': ([0, 0, 0], [0, 0, 0, 1])}, .002)
    np.testing.assert_array_equal(excluded, [True, True, False, False])
    moved = exclusion_mask(points + [2, 0, 0], geometry, {'finger': ([-2, 0, 0], [0, 0, 0, 1])}, .002)
    np.testing.assert_array_equal(moved, excluded)
