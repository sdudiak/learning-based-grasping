from threading import Event

import numpy as np
import pytest

from isaac_manipulation_segmentation.configuration import load_parameter_file, parse_overrides, validate_parameters
from isaac_manipulation_segmentation.processing import FrameProcessor, OverlayTracker, instance_mask, render_overlay


def test_decimation_and_latest_pending_frame():
    started, release, finished = Event(), Event(), Event()
    processed = []

    def process(frame):
        processed.append(frame)
        if frame == 0:
            started.set()
            assert release.wait(5)
        else:
            finished.set()

    processor = FrameProcessor(2, process)
    try:
        processor.submit(0)
        assert started.wait(5)
        for frame in range(1, 6):
            processor.submit(frame)
        release.set()
        assert finished.wait(5)
    finally:
        release.set()
        processor.close()
    assert processed == [0, 4]


def test_masks_detections_and_overlay():
    result = {
        'masks': np.array([[[True, True], [False, False]], [[False, True], [False, True]]]),
        'scores': np.array([0.7, 0.9]),
        'boxes': np.array([[0., 0., 2., 1.], [1., 0., 2., 2.]]),
        'labels': np.array([1, 1]),
    }
    instances = instance_mask(result['masks'], result['scores'])
    np.testing.assert_array_equal(instances, [[1, 2], [0, 2]])
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = render_overlay(rgb, instances, result['boxes'], result['scores'])
    assert overlay.shape == rgb.shape
    assert overlay.any()
    assert not rgb.any()


def test_empty_predictions():
    result = {'masks': np.empty((0, 3, 4), dtype=bool), 'scores': np.empty(0),
              'boxes': np.empty((0, 4)), 'labels': np.empty(0, dtype=int)}
    assert not instance_mask(result['masks'], result['scores']).any()


@pytest.mark.parametrize('decimation', [0, -1, 1.5, True])
def test_invalid_decimation(decimation):
    with pytest.raises(ValueError, match='decimation'):
        validate_parameters({'checkpoint': 'model.pt', 'decimation': decimation})


def test_colors_survive_detection_reordering():
    tracker = OverlayTracker(0.3, 5)
    boxes = np.array([[10., 20., 60., 70.], [90., 20., 140., 70.]])
    masks = np.zeros((2, 80, 160), dtype=bool)
    masks[0, 20:70, 10:60] = True
    masks[1, 20:70, 90:140] = True
    scores = np.array([0.9, 0.8])
    labels = np.ones(2, dtype=int)
    rgb = np.zeros((80, 160, 3), dtype=np.uint8)
    ids = tracker.update(boxes, labels)
    before = render_overlay(rgb, instance_mask(masks, scores), boxes, scores, ids)
    reordered = tracker.update(boxes[::-1], labels)
    after = render_overlay(rgb, instance_mask(masks[::-1], scores[::-1]), boxes[::-1], scores[::-1], reordered)
    np.testing.assert_array_equal(reordered, ids[::-1])
    for y, x in ((45, 35), (45, 115), (50, 10), (50, 90)):
        np.testing.assert_array_equal(before[y, x], after[y, x])


def test_tracking_handles_brief_misses_and_expires_old_tracks():
    tracker = OverlayTracker(0.3, 2)
    boxes = np.array([[0., 0., 10., 10.]])
    original = tracker.update(boxes, [1])[0]
    tracker.update([], [])
    tracker.update([], [])
    assert tracker.update(boxes + 1, [1])[0] == original
    for _ in range(3):
        tracker.update([], [])
    assert not tracker.tracks
    assert tracker.update(boxes, [1])[0] != original


def test_tracking_matches_one_to_one_and_respects_labels():
    tracker = OverlayTracker(0.3, 2)
    boxes = np.array([[0., 0., 10., 10.]])
    original = tracker.update(boxes, [1])[0]
    ids = tracker.update(np.repeat(boxes, 2, axis=0), [1, 1])
    assert len(set(ids)) == 2
    assert original in ids
    assert tracker.update(boxes, [2])[0] not in ids


def test_tracker_reset_discards_previous_camera_history():
    tracker = OverlayTracker(0.3, 5)
    tracker.update([[0, 0, 10, 10]], [1])
    tracker.update([[20, 20, 30, 30]], [1])
    tracker.reset()
    assert not tracker.tracks
    assert tracker.update([[20, 20, 30, 30]], [1]).tolist() == [1]


def test_parameter_file_merges_wildcard_and_node_settings(tmp_path):
    path = tmp_path / 'params.yaml'
    path.write_text('/**:\n  ros__parameters:\n    decimation: 2\n    use_sim_time: true\n'
                    '/segmentation:\n  ros__parameters:\n    decimation: 3\n')
    assert load_parameter_file(path) == {'decimation': 3, 'use_sim_time': True}
    assert parse_overrides({'mask_threshold': '1', 'publish_overlay_image': 'false'}) == {
        'mask_threshold': 1.0, 'publish_overlay_image': False}
    with pytest.raises(ValueError, match='bool'):
        parse_overrides({'publish_overlay_image': '0'})
