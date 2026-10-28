import tempfile
import unittest
import hashlib
import json
from pathlib import Path

import numpy as np

from ml_ws.segmentation.metrics import aggregate, average_precision, box_iou, detection_stats, mask_iou
from ml_ws.scripts.evaluation.data import load_instances, load_prediction_run
from isaac_ws.dataset_generation.segmentation.capture import instance_annotations


class MetricsTests(unittest.TestCase):
    def test_integer_boxes_have_fractional_iou(self):
        actual = box_iou(np.array([[0, 0, 2, 2]]), np.array([[1, 0, 3, 2]]))
        self.assertAlmostEqual(actual[0, 0], 1 / 3)

    def test_rejects_incomplete_predictions_and_changed_ground_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / 'run/predictions'
            predictions.mkdir(parents=True)
            dataset = root / 'dataset'
            dataset.mkdir()
            manifest = dataset / 'manifest.json'
            manifest.write_text(json.dumps({'status': 'complete', 'frames': [{'id': 'one'}]}))
            (predictions / 'one.npz').touch()
            run = {'status': 'incomplete', 'dataset': str(dataset), 'frame_ids': ['one'],
                   'dataset_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
            run_path = predictions.parent / 'manifest.json'
            run_path.write_text(json.dumps(run))
            with self.assertRaisesRegex(ValueError, 'complete successfully'):
                load_prediction_run(predictions)
            run['status'] = 'complete'
            run_path.write_text(json.dumps(run))
            self.assertEqual(load_prediction_run(predictions)[1], [{'id': 'one'}])
            manifest.write_text(manifest.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'differs'):
                load_prediction_run(predictions)

    def test_evaluation_loads_masks_without_saved_boxes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'truth.npz'
            masks = np.zeros((1, 4, 5), dtype=bool)
            masks[0, 1:3, 2:5] = True
            np.savez_compressed(path, masks=masks, instance_ids=np.array([1]))
            truth = load_instances(path, (4, 5))
            np.testing.assert_array_equal(truth['boxes'], [[2, 1, 5, 3]])
            np.savez_compressed(path, masks=np.zeros((0, 4, 5), dtype=bool))
            self.assertEqual(load_instances(path, (4, 5))['boxes'].shape, (0, 4))

    def test_duplicate_detection_is_false_positive(self):
        iou = np.array([[1.0], [1.0], [0.0]])
        stats = detection_stats(iou, np.array([0.9, 0.8, 0.7]), 0.3, 0.5)
        self.assertEqual((stats['tp'], stats['fp'], stats['fn']), (1, 2, 0))
        self.assertEqual(len(stats['matches']), 1)

    def test_ap_penalizes_higher_confidence_false_positive(self):
        samples = [{'iou': np.array([[0.0], [1.0]]), 'scores': np.array([0.9, 0.8])}]
        self.assertAlmostEqual(average_precision(samples, 0.5), 0.5)

    def test_ap_matches_independently_per_image(self):
        samples = [{'iou': np.array([[1.0]]), 'scores': np.array([0.9])}] * 2
        self.assertEqual(average_precision(samples, 0.5), 1.0)

    def test_no_predictions_and_no_ground_truth(self):
        scores = np.zeros(0)
        result = aggregate([{'iou': np.zeros((0, 1)), 'scores': scores}], 0.3, 0.5)
        self.assertEqual(result['fn'], 1)
        self.assertEqual(result['ap50_95'], 0)
        empty = aggregate([{'iou': np.zeros((0, 0)), 'scores': scores}], 0.3, 0.5)
        self.assertIsNone(empty['ap50_95'])
        self.assertIsNone(empty['recall'])

    def test_pixel_conventions_and_mask_overlap(self):
        boxes = np.array([[0., 0., 2., 2.], [2., 0., 4., 2.]])
        np.testing.assert_array_equal(box_iou(boxes, boxes), np.eye(2))
        masks = np.array([[[1, 1], [0, 0]], [[0, 1], [0, 1]]], dtype=bool)
        self.assertAlmostEqual(mask_iou(masks, masks)[0, 1], 1 / 3)

    def test_score_filter_does_not_change_ap_input(self):
        iou = np.array([[1.]])
        stats = detection_stats(iou, np.array([0.1]), 0.3, 0.5)
        self.assertEqual((stats['tp'], stats['fp'], stats['fn']), (0, 0, 1))
        self.assertEqual(average_precision([{'iou': iou, 'scores': np.array([0.1])}], 0.5), 1.0)

    def test_multiple_render_parts_merge_into_one_object(self):
        objects = [{'id': 1, 'prim_path': '/World/Objects/one'},
                   {'id': 2, 'prim_path': '/World/Objects/one_more'},
                   {'id': 3, 'prim_path': '/World/Objects/hidden'}]
        data = {'data': np.array([[0, 10, 11], [0, 20, 0]]),
                'info': {'idToLabels': {'0': 'BACKGROUND', '10': '/World/Objects/one/mesh_a',
                                      '11': '/World/Objects/one/mesh_b', '20': '/World/Objects/one_more'}}}
        result = instance_annotations(data, objects)
        np.testing.assert_array_equal(result['instance_ids'], [1, 2])
        self.assertNotIn('boxes', result)
        np.testing.assert_array_equal(result['instance_map'], [[0, 1, 1], [0, 2, 0]])
        np.testing.assert_array_equal(result['render_instance_map'], data['data'])
        self.assertEqual(result['masks'][0].sum(), 2)


if __name__ == '__main__':
    unittest.main()
