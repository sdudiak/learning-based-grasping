import json
import io
import random
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from ml_ws.segmentation.dataset import InstanceDataset, PreparedDataset, load_splits, prepare_splits
from ml_ws.segmentation.engine import restore_rng, rng_state, seed_everything, validation_loss
from ml_ws.segmentation.model import build_model
from ml_ws.segmentation.scripts.prepare_dataset import prepare_dataset
from ml_ws.segmentation.scripts import train
from ml_ws.segmentation.transforms import make_transforms


CONFIG = tomllib.loads((Path(__file__).parents[1] / 'configs/maskrcnn.toml').read_text())


def fixture(root, piles=5):
    (root / 'images').mkdir()
    (root / 'ground_truth').mkdir()
    frames = []
    for pile in range(piles):
        for view in range(2):
            name = f'pile_{pile}_view_{view}'
            rgb = np.zeros((8, 10, 3), dtype=np.uint8)
            rgb[2, 3] = 255
            Image.fromarray(rgb).save(root / 'images' / f'{name}.png')
            mask = np.zeros((1, 8, 10), dtype=bool)
            mask[0, 2, 3] = True
            np.savez_compressed(root / 'ground_truth' / f'{name}.npz', masks=mask, instance_ids=np.array([1]))
            frames.append({'id': name, 'pile': pile, 'view': view, 'image': f'images/{name}.png',
                           'ground_truth': f'ground_truth/{name}.npz'})
    (root / 'manifest.json').write_text(json.dumps({'status': 'complete', 'frames': frames}))
    return root


class TrainingTests(unittest.TestCase):
    def test_resume_rejects_zero_steps_instead_of_reusing_checkpoint_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / 'checkpoints/last.pt'
            checkpoint.parent.mkdir()
            torch.save({'config': CONFIG, 'dataset': str(Path(temporary) / 'missing'),
                        'overfit_images': 0, 'steps_per_epoch': 4}, checkpoint)
            with patch.object(sys, 'argv', ['train', '--resume', str(checkpoint), '--steps-per-epoch', '0']), patch(
                'sys.stderr', new_callable=io.StringIO
            ) as error:
                with self.assertRaises(SystemExit) as result:
                    train.main()
            self.assertEqual(result.exception.code, 2)
            self.assertIn('steps must be positive', error.getvalue())

    def test_validation_loss_preserves_batch_norm_modes_rng_and_gradients(self):
        class Detector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.BatchNorm2d(3)

            def forward(self, images, targets):
                self.assert_training = self.training
                self.assert_grad_enabled = torch.is_grad_enabled()
                torch.rand(1)
                return {'loss_mask': self.norm(torch.stack(images)).mean()}

        model = Detector().eval()
        model.norm.train()
        before = model.norm.running_mean.clone()
        state = torch.get_rng_state()
        loader = [([torch.ones(3, 2, 2)] * 2, [{}, {}]),
                  ([torch.full((3, 2, 2), 4.0)], [{}])]
        losses = validation_loss(model, loader, torch.device('cpu'))
        self.assertAlmostEqual(losses['total'], 2.0, places=4)
        self.assertEqual(losses['total'], losses['loss_mask'])
        self.assertTrue(model.assert_training)
        self.assertFalse(model.assert_grad_enabled)
        self.assertFalse(model.training)
        self.assertTrue(model.norm.training)
        torch.testing.assert_close(model.norm.running_mean, before)
        torch.testing.assert_close(torch.get_rng_state(), state)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_offline_preparation_preserves_alignment_and_excludes_heldout_variants(self):
        with tempfile.TemporaryDirectory() as temp:
            root = fixture(Path(temp))
            output = root / 'prepared'
            settings = {**CONFIG['augmentation'], 'horizontal_flip': 1.0, 'vertical_flip': 0,
                        'brightness': 0, 'contrast': 0, 'saturation': 0, 'hue': 0, 'blur_probability': 0}
            prepare_dataset(root, output, {**CONFIG, 'augmentation': settings}, copies=1)
            prepare_splits(output)
            splits = load_splits(output)
            self.assertEqual(sum(len(frames) for frames in splits.values()), 16)
            for name in ('validation', 'test'):
                self.assertFalse(any(frame['augmented'] for frame in splits[name]))
            variants = [frame for frame in splits['train'] if frame['augmented']]
            data = PreparedDataset(output, variants)
            image, target = data[0]
            self.assertEqual(target['boxes'].tolist(), [[6.0, 2.0, 7.0, 3.0]])
            self.assertEqual(image[:, 2, 6].tolist(), [1.0, 1.0, 1.0])
            self.assertTrue(target['masks'][0, 2, 6])
            image2, target2 = data[0]
            torch.testing.assert_close(image, image2)
            torch.testing.assert_close(target['boxes'], target2['boxes'])
            with self.assertRaisesRegex(ValueError, 'prepare-segmentation-dataset'):
                PreparedDataset(root, json.loads((root / 'manifest.json').read_text())['frames'])

    def test_rng_state_restores_augmentation_and_sampling_generators(self):
        seed_everything(123)
        state = rng_state()
        expected = (random.random(), np.random.random(), torch.rand(3))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.rand(3), expected[2])

    def test_splits_keep_views_together_and_cover_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = fixture(Path(temp))
            counts = prepare_splits(root)
            self.assertEqual(sum(counts.values()), 5)
            splits = load_splits(root)
            groups = [{f['pile'] for f in frames} for frames in splits.values()]
            self.assertFalse(groups[0] & groups[1] or groups[1] & groups[2] or groups[0] & groups[2])
            self.assertEqual(sum(map(len, splits.values())), 10)
            for frames in splits.values():
                for pile in {f['pile'] for f in frames}:
                    self.assertEqual(sum(f['pile'] == pile for f in frames), 2)
            with self.assertRaisesRegex(ValueError, 'already exist'):
                prepare_splits(root)

    def test_rejects_modified_manifest_and_pile_leakage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = fixture(Path(temp))
            prepare_splits(root)
            path = root / 'splits/validation.json'
            train = json.loads((root / 'splits/train.json').read_text())
            validation = json.loads(path.read_text())
            validation['frame_ids'] = train['frame_ids'][:2]
            validation['pile_ids'] = train['pile_ids'][:1]
            path.write_text(json.dumps(validation))
            with self.assertRaisesRegex(ValueError, 'leakage'):
                load_splits(root)
            manifest = root / 'manifest.json'
            manifest.write_text(manifest.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                load_splits(root)

    def test_single_pixel_boxes_and_rgb_masks_remain_aligned_after_flip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = fixture(Path(temp))
            frames = json.loads((root / 'manifest.json').read_text())['frames']
            settings = {**CONFIG['augmentation'], 'horizontal_flip': 1.0, 'vertical_flip': 0,
                        'brightness': 0, 'contrast': 0, 'saturation': 0, 'hue': 0, 'blur_probability': 0}
            data = InstanceDataset(root, frames, make_transforms(settings, training=True))
            image, target = data[0]
            self.assertEqual(image.dtype, torch.float32)
            self.assertEqual(image.max().item(), 1.0)
            self.assertEqual(target['labels'].tolist(), [1])
            self.assertEqual(target['boxes'].tolist(), [[6.0, 2.0, 7.0, 3.0]])
            self.assertEqual(image[:, 2, 6].tolist(), [1.0, 1.0, 1.0])
            self.assertEqual(target['masks'][0, 2, 6].item(), 1)

    def test_rejects_invalid_masks_and_too_few_piles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = fixture(Path(temp), piles=2)
            with self.assertRaisesRegex(ValueError, 'three'):
                prepare_splits(root)
            frames = json.loads((root / 'manifest.json').read_text())['frames']
            np.savez_compressed(root / frames[0]['ground_truth'], masks=np.zeros((1, 8, 10), dtype=bool))
            data = InstanceDataset(root, frames, make_transforms(CONFIG['augmentation']))
            with self.assertRaisesRegex(ValueError, 'Empty instance'):
                data[0]

    def test_model_has_single_foreground_class_and_supports_no_pretraining(self):
        model = build_model({**CONFIG['model'], 'pretrained': False})
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, 2)
        self.assertEqual(model.roi_heads.mask_predictor.mask_fcn_logits.out_channels, 2)


if __name__ == '__main__':
    unittest.main()
