import numpy as np
from isaac_manipulation_pointcloud.processing import (
    ObservationCache, bounds, candidates, partition, project_depth, transform_points)
from isaac_manipulation_pointcloud.selection import Selection


def scene():
    return np.array([[0., 0, 1], [0.02, 0, 1], [1, 0, 1], [1.02, 0, 1], [0, 0, 2]]), np.array([1, 1, 2, 2, 0])


def test_idle_and_clear():
    points, labels = scene()
    state = Selection(.1, .1, .02)
    state.select(candidates(points, labels, 2), points[0])
    state.clear()
    target, environment = partition(points, labels, state.match(candidates(points, labels, 2)))
    assert not len(target)
    np.testing.assert_array_equal(environment, points)


def test_partition_multiple_objects():
    points, labels = scene()
    for label in (1, 2):
        target, environment = partition(points, labels, label)
        assert len(target) == 2
        assert not set(map(tuple, target)) & set(map(tuple, environment))
        assert set(map(tuple, np.concatenate((target, environment)))) == set(map(tuple, points))
        np.testing.assert_array_equal(target, points[labels == label])


def test_centroid_and_bounds():
    centroid, center, dimensions = bounds(np.array([[0, 1, 2], [2, 5, 8], [1, 3, 5]]))
    np.testing.assert_allclose(centroid, [1, 3, 5])
    np.testing.assert_allclose(center, [1, 3, 5])
    np.testing.assert_allclose(dimensions, [2, 4, 6])


def test_depth():
    points, valid = project_depth(np.array([[1, 0, np.nan], [np.inf, -1, 11]]), np.eye(3))
    assert valid.sum() == 1
    np.testing.assert_allclose(points[valid], [[0, 0, 1]])


def test_ids_and_camera_motion():
    points, labels = scene()
    state = Selection(.1, .1, .02)
    assert state.select(candidates(points, labels, 2), points[0]) == 1
    moved = transform_points(points - [0.2, 0, 0], [.2, 0, 0], [0, 0, 0, 1])
    assert state.match(candidates(moved, labels * 3, 2)) == 3
    np.testing.assert_allclose(state.reference, [.01, 0, 1])
    assert state.match({2: np.array([1., 0, 1])}) is None
    assert state.reference is not None


def test_reject_distant_ambiguous_invalid():
    state = Selection(.1, .1, .02)
    assert state.select({1: np.zeros(3)}, np.ones(3)) is None
    assert state.select({1: np.zeros(3), 2: np.array([.01, 0, 0])}, np.zeros(3)) is None
    assert state.select({1: np.zeros(3)}, np.array([np.nan, 0, 0])) is None
    assert state.reference is None


def test_cache_sync_bounded_and_reset():
    cache = ObservationCache(2, 10, 0)
    assert cache.nearest(1) is None
    for stamp in (10, 11, 12):
        cache.add(stamp, str(stamp))
    assert len(cache.items) == 2
    assert cache.nearest(10) is None
    assert cache.nearest(11) == '11'
    assert cache.nearest(13) is None
    cache.add(0, 'reset')
    assert list(cache.items) == [0]


def test_rotation():
    points = transform_points(np.array([[1., 0, 0]]), [1, 2, 3], [0, 0, 2**-.5, 2**-.5])
    np.testing.assert_allclose(points, [[1, 3, 3]], atol=1e-12)


def test_zero_depth_invalid_even_with_zero_minimum():
    _, valid = project_depth(np.array([[0., 1.]]), np.eye(3), minimum=0)
    np.testing.assert_array_equal(valid, [[False, True]])
