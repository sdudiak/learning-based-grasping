# Machine learning workspace

Class-agnostic instance segmentation using Mask R-CNN ResNet-50 FPN v2: background (`0`) and object (`1`), with separate masks for visible objects. Run commands from the repository root.

## Layout and requirements

- `segmentation/`: datasets, transforms, model, training engine, and metrics.
- `segmentation/scripts/`: preparation, splitting, training, and prediction.
- `scripts/evaluation/`: saved-prediction validation, reports, and comparisons.
- [`segmentation/configs/maskrcnn.toml`](segmentation/configs/maskrcnn.toml): model, optimization, and augmentation settings.
- [`../isaac_ws/dataset_generation/segmentation/`](../isaac_ws/dataset_generation/segmentation/): headless capture using the shared object-spawning backend.

Isaac Sim generation uses the default Python 3.11 environment and requires an NVIDIA GPU/driver. Offline ML uses the `ml` environment with Python 3.11, PyTorch 2.7.0, and torchvision 0.22.0. Live ROS inference uses Python 3.12; see the [ROS guide](../ros_ws/README.md).

```bash
pixi install -e default
pixi install -e ml
```

The default environment expects ROS Jazzy at `/opt/ros/jazzy`. Offline training/prediction default to `cuda:0`; `--device cpu` selects CPU execution.

## Data generation and preparation

Choose fresh dataset and experiment names when repeating this workflow; nonempty outputs are rejected.

```bash
pixi run download-gso-objects --count 50 --seed 42
pixi run convert-gso-objects --size-cm 10

pixi run generate-dataset \
  --name gso_raw_v1 --piles 100 --views-per-pile 5 \
  --min-objects 5 --max-objects 15 --spread-cm 3 \
  --resolution 640 480 --settle-timeout 30 --seed 42 \
  --save-masked-images

pixi run prepare-segmentation-dataset \
  --dataset ml_ws/datasets/segmentation/gso_raw_v1 \
  --name gso_prepared_v1 --augmented-copies 2 --seed 42

pixi run prepare-segmentation-splits \
  --dataset ml_ws/datasets/segmentation/gso_prepared_v1 \
  --validation-fraction 0.1 --test-fraction 0.1 --seed 42
```

- Assets are cached under `isaac_ws/assets/objects/gso/{raw,converted}/`. Conversion scales the longest dimension; use `--overwrite` to replace existing conversions. Generation samples all usable converted assets.
- Generation creates its own ground, fixed lighting, and randomized camera views. Objects drop sequentially at 120 Hz and must settle before capture. Use `--objects N` instead of minimum/maximum counts for fixed-size piles.
- Raw datasets contain RGB `images/`, mask NPZ and metadata JSON in `ground_truth/`, optional `masked_images/`, and `manifest.json`. Masks contain visible pixels only; fully hidden objects have no mask. Boxes are derived using exclusive upper pixel bounds.
- Continue only when the manifest status is `complete`. Failed settling produces an incomplete run; retry with a new name and adjusted settings.
- Preparation saves originals, fixed augmented variants, and model-ready `tensors/`. Defaults include flips, color jitter, and occasional blur. Augmentation is performed once, before training.
- Splits group all views by physical pile. Only training includes augmented variants; validation/test use originals. At least three piles are required. Assets can appear in every split, so this is not an unseen-object benchmark.

## Training

```bash
pixi run train-segmentation \
  --dataset ml_ws/datasets/segmentation/gso_prepared_v1 \
  --name maskrcnn-v1

pixi run tensorboard
```

Defaults: pretrained visual features with new single-class heads, 480/800-pixel resize limits, 20 epochs, batch size 2, SGD at learning rate 0.0025, momentum 0.9, weight decay 0.0005, gradient clipping at 5, and CUDA mixed precision. The learning rate drops by 0.1 every 8 epochs. Override `--epochs`, `--batch-size`, or `--config PATH`; changing augmentation requires preparing a new dataset.

Runs are saved under `ml_ws/outputs/segmentation/<name>/`: resolved configuration, manifests, per-epoch metrics, TensorBoard logs, and checkpoints. `best.pt` maximizes validation mask AP50–95; `last.pt` stores optimizer, scheduler, scaler, and RNG state for resume. The final summary re-evaluates the best checkpoint on validation, not test.

```bash
pixi run train-segmentation \
  --resume ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/last.pt \
  --epochs 40
```

Resume keeps the original experiment and verifies dataset/split fingerprints. Epochs specify the desired total; omit a new `--name` or custom configuration.

For an implementation diagnostic, train with `--overfit-images 4 --steps-per-epoch 100 --epochs 10` under a new name. This evaluates on the same training subset, so its metrics are not held-out performance.

## Validation and test evaluation

Tune the checkpoint and confidence threshold using validation. Once fixed, evaluate the reserved test split:

```bash
pixi run segment-dataset \
  --checkpoint ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt \
  --split test --name maskrcnn-v1-test

pixi run evaluate-segmentation \
  --predictions ml_ws/outputs/segmentation/maskrcnn-v1-test/predictions \
  --score-threshold 0.3 --iou-threshold 0.5

pixi run compare-segmentation \
  --predictions ml_ws/outputs/segmentation/maskrcnn-v1-test/predictions
```

Use `--split validation` during development. Prediction defaults to the checkpoint's dataset; override with `--dataset PATH`. It saves masks, boxes, scores, labels, latency, and a dataset fingerprint. Evaluation rejects incomplete runs or mismatched manifests.

Results appear in the prediction run's `evaluation/` directory: `metrics.json`, `per_image.json`, `overlays/`, and `comparisons/`. AP uses confidence-ranked matching, 101-point interpolation, IoU 0.50–0.95, and at most 100 predictions/image. It uses all saved scores above the model's score floor (default 0.01); the evaluation score threshold filters precision/recall and visualizations. These are class-agnostic metrics, without COCO class/area breakdowns.

## Checks and artifacts

```bash
pixi run -e ml python -m unittest discover -s ml_ws/segmentation/tests -v
pixi run -e ml python -m unittest discover -s ml_ws/scripts/evaluation/tests -v
python3 -m unittest discover -s scripts/tests -v
pixi run python isaac_ws/extensions/isaac_manipulation.object_spawner/test/test_spawning.py
```

The last command requires Isaac Sim and a GPU. Datasets, prepared tensors, experiment outputs, checkpoints, downloaded weights, and TensorBoard logs are ignored by Git.
