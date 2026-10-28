# Machine learning workspace

This workspace trains and evaluates **class-agnostic instance segmentation** for grasping scenes. The model is Mask R-CNN with a ResNet-50 FPN v2 backbone. It predicts a separate mask and bounding box for each visible object, with two classes: background (`0`) and object (`1`). It does not distinguish object categories or predict grasp poses.

Run the commands below from the **repository root**. Dataset names and experiment names in these examples are illustrative; choose new names when creating another run. Existing nonempty output directories are rejected.

## Code and environments

| Location | Responsibility |
| --- | --- |
| `segmentation/` | Dataset loading, transforms, model construction, training engine, and shared metrics |
| `segmentation/scripts/` | Dataset preparation, splitting, training, and offline prediction entrypoints |
| `segmentation/configs/maskrcnn.toml` | Model, optimization, and augmentation configuration |
| `scripts/evaluation/` | Saved-prediction validation, metric reports, overlays, and comparisons |
| `../isaac_ws/dataset_generation/segmentation/` | Headless scene generation and RGB/instance-mask capture |
| `../isaac_ws/extensions/isaac_manipulation.object_spawner/` | Shared object placement and settling code, plus the interactive spawner |

Pixi separates the simulator, offline ML, and live ROS runtimes:

| Environment | Python | Use |
| --- | --- | --- |
| `default` | 3.11 | Isaac Sim 5.1, asset conversion, and dataset generation |
| `ml` | 3.11 | Preparation, training, offline prediction, and evaluation |
| `segmentation` | 3.12 | Live inference with ROS 2 Jazzy and RViz |

The ML environments pin PyTorch 2.7.0 and torchvision 0.22.0 with CUDA 12.8 wheels. Isaac Sim capture requires a working NVIDIA GPU and driver. Offline training and prediction default to `cuda:0`; pass `--device cpu` to run those stages on CPU. The default environment also sources the project's ROS installation at `/opt/ros/jazzy/setup.bash`.

Install the environments and check GPU visibility:

```bash
pixi install -e default
pixi install -e ml
pixi run -e ml python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Pixi selects the environment containing each named task automatically. The Python distribution defined by `pyproject.toml` is `grasping-segmentation`; its import namespace is `ml_ws`. It is installed editably by the live `segmentation` environment. Optional package extras are `training` for TensorBoard and `evaluation` for OpenCV-based reports.

## 1. Download and convert objects

The downloader obtains Google Scanned Objects from the GoogleResearch Gazebo Fuel collection. Raw assets are cached under `isaac_ws/assets/objects/gso/raw/`.

```bash
pixi run download-gso-objects --list
pixi run download-gso-objects --count 50 --seed 42
```

Use `--objects NAME ...` for explicit selection, or `--all` for the collection. Cached downloads remain available across runs; selecting a smaller set does not remove older assets.

Convert the downloaded visual OBJ meshes into textured USD assets with rigid bodies and convex-decomposition collisions:

```bash
pixi run convert-gso-objects --size-cm 10
```

`--size-cm` sets each object's longest bounding-box dimension. Conversion writes `isaac_ws/assets/objects/gso/converted/<object>/<object>.usd`, associated meshes/textures, and `conversion_manifest.json`. Existing converted assets are skipped unless `--overwrite` is supplied. Replacement is staged so a failed conversion preserves the previous asset.

Generation samples from all usable assets in the converted directory.

## 2. Generate raw RGB and instance annotations

```bash
pixi run generate-dataset \
  --name gso_raw_v1 \
  --piles 100 \
  --views-per-pile 5 \
  --min-objects 5 \
  --max-objects 15 \
  --spread-cm 3 \
  --resolution 640 480 \
  --settle-timeout 30 \
  --seed 42 \
  --save-masked-images
```

This creates `ml_ws/datasets/segmentation/gso_raw_v1/`. The standalone generator creates its own ground, lighting, and camera; it does not use the robot/table scene.

For each pile, generation:

1. Samples an inclusive object count and a horizontal spread up to `--spread-cm` per axis. Use `--objects N` instead of the minimum/maximum arguments for a fixed count.
2. Samples object assets and orientations, then drops objects sequentially with 15 mm clearance above the current pile. Asset sampling avoids duplicates when enough assets are available.
3. Advances physics at 120 Hz, allowing 30 steps between drops. The shared settling check accepts low velocity or sufficiently small pose drift over a half-second window, after at least approximately one second.
4. Samples camera views around the settled pile. Focal length, azimuth, elevation, roll, distance, and target offset vary; lighting stays fixed.
5. Captures RGB and visible instance annotations at each view. Rendered mesh parts belonging to one object are merged into one instance.

Objects hidden entirely behind other objects have no visible mask. These are visible-instance annotations, rather than amodal object shapes. Failed settling skips that pile; missing expected frames make the run `incomplete`. Exceptions mark it `failed`. Preparation accepts only `complete` runs, so inspect the manifest before continuing after a generation failure. A new run name allows a retry with different count, spread, or timeout settings.

### Raw dataset layout

```text
ml_ws/datasets/segmentation/gso_raw_v1/
  manifest.json
  images/<frame>.png
  ground_truth/<frame>.npz
  ground_truth/<frame>.json
  masked_images/<frame>.png
```

`masked_images/` is optional. Previews can also be generated later:

```bash
pixi run save-masked-images --dataset ml_ws/datasets/segmentation/gso_raw_v1
```

Each NPZ contains boolean `masks` of shape `[instances, height, width]`, corresponding `instance_ids`, a `uint32` object `instance_map` with background `0`, and the original `render_instance_map`. JSON records preserve camera, lighting, object poses, and renderer-label mappings. Bounding boxes are derived from masks during loading, using `xyxy` pixel-edge coordinates with exclusive upper bounds.

## 3. Prepare fixed augmentations and training tensors

```bash
pixi run prepare-segmentation-dataset \
  --dataset ml_ws/datasets/segmentation/gso_raw_v1 \
  --name gso_prepared_v1 \
  --augmented-copies 2 \
  --seed 42
```

The prepared dataset contains an original and two augmented variants per source frame, plus `tensors/<frame>.pt` files holding uint8 RGB images and model targets. Masks, boxes, labels, and areas stay aligned with the transformed image.

Augmentations are generated **once before training**, not resampled each epoch. Defaults in [`segmentation/configs/maskrcnn.toml`](segmentation/configs/maskrcnn.toml) are horizontal/vertical flips, brightness/contrast/saturation/hue jitter, and occasional Gaussian blur. Pass `--config PATH` when preparing with another configuration; the manifest records the preparation settings and source-manifest fingerprint.

Each flip has probability 0.5. Color-jitter strengths are 0.15 for brightness and contrast, 0.1 for saturation, and 0.02 for hue; blur has probability 0.1. Augmentation is enabled by default.

## 4. Split by physical pile

```bash
pixi run prepare-segmentation-splits \
  --dataset ml_ws/datasets/segmentation/gso_prepared_v1 \
  --validation-fraction 0.1 \
  --test-fraction 0.1 \
  --seed 42
```

Splits are disjoint by **pile**, with at least three source piles required. All camera views of a pile stay together. Training includes originals and prepared augmentations from training piles; validation and test include only original views. Augmented files belonging to held-out piles remain on disk but are excluded from the splits.

Files under `splits/` record frame IDs, pile IDs, and the dataset-manifest hash. Loading checks for leakage, duplicate frames, incomplete coverage, and manifest changes. Existing splits are not overwritten.

This evaluates generalization to new piles and views. The same asset library can appear in every split; this is not an object-identity-disjoint benchmark. Keep the test split reserved until model/configuration choices have been made using validation.

## 5. Train and monitor

```bash
pixi run train-segmentation \
  --dataset ml_ws/datasets/segmentation/gso_prepared_v1 \
  --name maskrcnn-v1 \
  --device cuda:0
```

The model starts from torchvision's pretrained Mask R-CNN weights when `model.pretrained = true`; the first run may download them. Classification and mask heads are replaced for the single foreground class. Torchvision handles internal image resizing and normalization.

### Default configuration

| Setting | Default |
| --- | --- |
| Image resizing | Short edge 480 px; long edge at most 800 px |
| Model detection score floor | 0.01 |
| Maximum detections per image | 100 |
| Epochs / batch size / loader workers | 20 / 2 / 0 |
| Optimizer | SGD, learning rate 0.0025, momentum 0.9, weight decay 0.0005 |
| Learning-rate schedule | Multiply by 0.1 every 8 epochs |
| Gradient norm clipping | 5.0 |
| Mixed precision | Enabled on CUDA when configured |
| Seed | 42 |

Override `--epochs`, `--batch-size`, and `--device` from the command line. Use `--config PATH` for a new experiment with other settings.

Training first records an untrained-head baseline on validation. Each epoch records training losses, validation mask metrics, and validation losses. Prediction metrics use evaluation mode; validation-loss computation uses the detection model's loss-returning mode while freezing batch-normalization statistics and restoring random-generator state afterward.

Training validation reports operating-point precision/recall at confidence 0.3 and IoU 0.5. Checkpoint selection uses AP50–95 across all saved validation scores.

Monitor with:

```bash
pixi run tensorboard
```

Open `http://127.0.0.1:6006`. Logs include training losses, validation metrics/losses, learning rate, and epoch duration.

### Experiment outputs

```text
ml_ws/outputs/segmentation/maskrcnn-v1/
  checkpoints/last.pt
  checkpoints/best.pt
  config.toml
  resolved_config.json
  manifest.json
  training_metrics.jsonl
  final_validation.json
  tensorboard/
```

`last.pt` stores model, optimizer, scheduler, scaler, random-generator, and loader state for resuming. `best.pt` is selected by validation mask AP averaged over IoU thresholds 0.50–0.95. The final validation summary reloads that best checkpoint; it is still a validation result, not the held-out test result.

Resume in the original experiment directory:

```bash
pixi run train-segmentation \
  --resume ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/last.pt \
  --epochs 40
```

`--epochs` is the total desired epoch count. Resume uses the checkpoint configuration and verifies dataset/split consistency. Do not supply a new experiment name or custom configuration when resuming.

### Small-subset diagnostic

```bash
pixi run train-segmentation \
  --dataset ml_ws/datasets/segmentation/gso_prepared_v1 \
  --name maskrcnn-overfit-check \
  --overfit-images 4 \
  --steps-per-epoch 100 \
  --epochs 10
```

This uses original training images without prepared augmentations and evaluates on the same subset. It checks whether the model can fit a small sample; its metrics are diagnostic, not held-out validation/test performance.

## 6. Predict and evaluate the held-out test split

```bash
pixi run segment-dataset \
  --checkpoint ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt \
  --split test \
  --name maskrcnn-v1-test

pixi run evaluate-segmentation \
  --predictions ml_ws/outputs/segmentation/maskrcnn-v1-test/predictions \
  --score-threshold 0.3 \
  --iou-threshold 0.5

pixi run compare-segmentation \
  --predictions ml_ws/outputs/segmentation/maskrcnn-v1-test/predictions \
  --score-threshold 0.3 \
  --iou-threshold 0.5
```

Prediction takes the dataset from the checkpoint unless `--dataset PATH` is supplied. `--split validation` supports model-development checks; `--split all` explicitly includes all manifest frames, including prepared variants. Predictions save boolean masks thresholded at 0.5, boxes, scores, labels, and per-image inference latency. The run manifest records selected frames and the dataset fingerprint.

Evaluation rejects incomplete prediction runs, missing frames, and changed dataset manifests when a prediction fingerprint is available. Reports are written beside `predictions/`:

| Output | Contents |
| --- | --- |
| `evaluation/metrics.json` | Aggregate box/mask precision, recall, matched IoU, AP50, AP75, and AP50–95 |
| `evaluation/per_image.json` | Per-image counts, matches, and operating-point metrics |
| `evaluation/overlays/` | Ground truth, matches, misses, and unmatched predictions |
| `evaluation/comparisons/` | Ground truth and prediction panels; matched instances share colors |

AP uses confidence-ranked one-to-one matching, 101-point precision interpolation, IoU thresholds 0.50 through 0.95, and at most 100 predictions per image. It uses all saved scores, subject to the model's detection score floor. The evaluation `--score-threshold` changes precision/recall and visualization filtering, not AP. Metrics are class-agnostic and do not include COCO class/area/crowd breakdowns.

## Live inference and checks

See the [ROS workspace README](../ros_ws/README.md) for the top-camera node and side-by-side RViz view. Offline evaluation defaults to a 0.3 operating-point score threshold; the live node defaults to 0.5 and accepts command-line overrides.

```bash
pixi run -e ml python -m unittest discover -s ml_ws/segmentation/tests -v
pixi run -e ml python -m unittest discover -s ml_ws/scripts/evaluation/tests -v
python3 -m unittest discover -s scripts/tests -v
pixi run python isaac_ws/extensions/isaac_manipulation.object_spawner/test/test_spawning.py
```

The final command runs headless Isaac Sim integration checks and requires a GPU. Dataset directories, prepared tensors, experiment outputs, checkpoints, downloaded weights, TensorBoard logs, and asset caches are ignored by Git. Store or transfer these artifacts separately from source code.
