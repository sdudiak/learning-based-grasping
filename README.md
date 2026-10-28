## Google Scanned Objects (GSO)

The [Google Scanned Objects dataset](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) provides textured 3D meshes of household objects for robotics manipulation and grasping experiments. The downloader retrieves OBJ model ZIP packages from the [GoogleResearch collection on Gazebo Fuel](https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research). Downloads are source meshes; they need conversion and physics setup before use in Isaac Sim.

Run these commands from the project root through Pixi:

```bash
# List object names and show all options
pixi run download-gso-objects --list
pixi run download-gso-objects --help

# Download specific objects using exact names from --list
pixi run download-gso-objects --objects "OBJECT_NAME_1" "OBJECT_NAME_2"

# Download 20 random objects with a reproducible selection order
pixi run download-gso-objects --count 20 --seed 42

# Download all available objects
pixi run download-gso-objects --all

# Choose another output directory
pixi run download-gso-objects --count 50 --output /path/to/assets/gso
```

Assets go into `isaac_ws/assets/objects/gso/raw/<object>/` by default (override with `--output`). Completed downloads are cached, and `selection_gso.json` records successful objects from the current run. Unavailable archives are skipped; `--count` tries other objects until the requested count is reached. A run exits with an error if it cannot fulfill the selection. Without selection arguments, it attempts the full collection. Downloaded assets are excluded from Git and are not included in this repository.

Acknowledgement: Google Scanned Objects © Google LLC is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). These third-party assets retain that license; redistribution must credit Google Scanned Objects, link the license, and indicate modifications.

### Convert GSO objects to USD

Convert downloaded OBJ meshes with Isaac Sim 5.1’s native asset converter through Pixi:

```bash
# Scale each object's longest dimension to 15 cm and convert without a GUI
pixi run convert-gso-objects --size-cm 15

# Regenerate existing converted assets
pixi run convert-gso-objects --size-cm 15 --overwrite

pixi run convert-gso-objects --help
```

The converter always runs headlessly (without the Isaac Sim UI). It reads `isaac_ws/assets/objects/gso/raw` recursively and writes to `isaac_ws/assets/objects/gso/converted`. For example, `<object>/meshes/model.obj` becomes `<object>/<object>.usd`, with `mesh.usd` and the `textures/` directory beside it. `conversion_manifest.json` records converted assets, cached outputs, and failures. Generated files are excluded from Git.

`--size-cm` is required and sets the longest bounding-box dimension using uniform scaling, preserving proportions. Visual meshes retain their original complexity; Blender is not required. The converter skips auxiliary meshes whose filenames contain `collision` or `collider`, adds rigid bodies and convex-decomposition collisions, and skips existing USDs unless `--overwrite` is supplied. Use `--overwrite` when changing the target size. Conversion failures produce a nonzero exit status. Verify textures, dimensions, and physics on sample assets before using the full collection in simulation.

## Generate segmentation datasets

Run the generator from the project root after downloading and converting GSO objects. It reads `isaac_ws/assets/objects/gso/converted` and uses Isaac Sim headlessly; a working NVIDIA driver and GPU are required.

Generate 10 piles, each containing a random number of objects from 5 to 20, with three camera views per pile:

```bash
pixi run generate-dataset \
  --name randomized-test \
  --min-objects 5 \
  --max-objects 20 \
  --spread-cm 5 \
  --piles 10 \
  --views-per-pile 3 \
  --seed 42
```

This requests up to 30 RGB images with matching visible-instance annotations. Objects are dropped sequentially onto a ground plane and allowed to settle before capture. Each view samples a camera pose and framing; lighting remains fixed and neutral.

| Argument | Meaning | Default |
| --- | --- | --- |
| `--name` | Dataset folder name under `ml_ws/datasets/segmentation/` | UTC timestamp |
| `--min-objects` | Minimum object count per pile, inclusive | `5` |
| `--max-objects` | Maximum object count per pile, inclusive | `15` |
| `--objects` | Fixed count instead of randomized counts; cannot be combined with the min/max arguments | Unset |
| `--spread-cm` | Maximum horizontal drop offset per axis, in centimeters | `3` |
| `--piles` | Number of independent piles | `1` |
| `--views-per-pile` | Camera views captured for each settled pile | `1` |
| `--seed` | Seed for object selection, counts, spread, rotations, and camera sampling | `42` |
| `--resolution WIDTH HEIGHT` | Image dimensions in pixels | `640 480` |
| `--settle-timeout` | Maximum final settling time per pile, in simulated seconds | `30` |
| `--save-masked-images` | Also save color-coded RGB mask previews with black backgrounds | Disabled |

For each pile, the generator samples an object count uniformly between the selected bounds and a horizontal spread uniformly between zero and `--spread-cm`. Each object's horizontal drop offsets are then sampled within ±that pile's spread on both axes. The parameter controls initial placement; collisions, sliding, and rolling determine the final footprint. Setting `--spread-cm 0` centers every drop. If the requested count exceeds the number of available assets, assets can repeat as separate instances.

Camera azimuth covers a full circle, elevation varies from 10–88° above the floor, roll varies within ±35°, and focal length varies from 16–40 mm. Framing fits the pile's projected bounds. Sampled camera settings, actual object counts, and spread are recorded in the manifest. A seed reproduces the sampled choices, but identical physics/rendering results across hardware are not guaranteed.

For a fixed number of objects and optional mask previews:

```bash
pixi run generate-dataset \
  --name fixed-count-test \
  --objects 20 \
  --piles 10 \
  --views-per-pile 1 \
  --save-masked-images
```

Output structure:

```text
ml_ws/datasets/segmentation/randomized-test/
├── images/          # RGB PNG images
├── ground_truth/    # NPZ instance annotations and per-frame JSON metadata
├── masked_images/   # Optional previews, created only when requested
└── manifest.json    # Settings, assets, pile outcomes, and captured frame list
```

Ground-truth NPZ files contain binary masks (`masks`), corresponding logical object IDs (`instance_ids`), a logical instance ID image (`instance_map`, zero means background), and the original renderer ID image (`render_instance_map`). Per-frame JSON files include the renderer ID mapping. Renderer IDs are frame-specific; logical IDs identify spawned objects within a pile. Multi-mesh objects are merged into one logical instance. Fully hidden objects have no binary mask; partially visible objects include only their visible pixels. Bounding boxes are not saved by the generator.

The output folder must be empty; choose a new `--name` for another dataset. Piles that fail to settle are skipped and recorded in `manifest.json`. The command returns a nonzero exit status if it cannot capture all requested views. Check the manifest's `status` and `frames` before using the dataset.

Mask previews can also be generated afterward without rerunning Isaac Sim:

```bash
pixi run save-masked-images --dataset ml_ws/datasets/segmentation/randomized-test
pixi run generate-dataset --help
```

## Train an object instance-segmentation model

The first training milestone uses [Torchvision's Mask R-CNN ResNet-50 FPN v2](https://docs.pytorch.org/vision/0.22/models/generated/torchvision.models.detection.maskrcnn_resnet50_fpn_v2.html) with one foreground class, `object`. Different objects remain separate instances even though they share a class. PyTorch and Torchvision run in the Pixi `ml` environment; Isaac Sim runs separately in the default environment. The training code uses the existing PyTorch/Torchvision environment directly.

### 1. Generate fresh training data

If assets are not prepared yet, download and convert them first:

```bash
pixi run download-gso-objects --count 20 --seed 42
pixi run convert-gso-objects --size-cm 15
```

Create a new dataset rather than training on an existing evaluation benchmark:

```bash
pixi run generate-dataset \
  --name segmentation-v1-raw \
  --min-objects 5 \
  --max-objects 15 \
  --spread-cm 5 \
  --piles 100 \
  --views-per-pile 3 \
  --seed 1001
```

This requests 300 images. The generator options and annotation format are documented above. Add `--save-masked-images` to inspect ground-truth previews. Inspect the images and masks before training, and confirm `manifest.json` has `status: "complete"`. The training pipeline rejects incomplete datasets. Fix a generation failure and generate into a new folder before continuing.

These data measure performance on new arrangements of the downloaded assets. They do not establish generalization to unseen object assets or real camera images. The current generator samples from the whole converted collection; asset-held-out generation is not implemented.

### 2. Prepare augmented images and annotations

```bash
pixi run -e ml prepare-segmentation-dataset \
  --dataset ml_ws/datasets/segmentation/segmentation-v1-raw \
  --name segmentation-v1 \
  --augmented-copies 2 \
  --seed 42
```

This separate CPU step saves the original plus two fixed augmented versions of each image in `ml_ws/datasets/segmentation/segmentation-v1/`. It applies coordinated horizontal/vertical flips to RGB and masks, mild color jitter, and occasional blur, using `[augmentation]` in `ml_ws/segmentation/configs/maskrcnn.toml`. Use `--config PATH` to change preparation settings, or `--augmented-copies 0` to prepare only originals. Augmentations are generated once rather than changing each epoch.

The prepared dataset contains `images/`, `ground_truth/` mask NPZ files, and `tensors/` with uint8 RGB and model-ready masks, boxes, labels, areas, and crowd flags. Boxes are calculated from transformed masks here. The manifest records the source, augmentation settings, seed, and each variant's original pile. Preparation uses additional disk space and does not change the generated source dataset. Interrupted preparation is marked incomplete; rerun into a new name.

### 3. Prepare train/validation/test splits

```bash
pixi run -e ml prepare-segmentation-splits \
  --dataset ml_ws/datasets/segmentation/segmentation-v1 \
  --validation-fraction 0.1 \
  --test-fraction 0.1 \
  --seed 42
```

Splits are made by pile, not by image: every view of a pile stays together. For 100 captured piles, the defaults produce 80 training piles, 10 validation piles, and 10 test piles. At least three captured piles are required; small datasets reserve at least one pile for each split. Files are saved as `splits/train.json`, `splits/validation.json`, and `splits/test.json` inside the dataset. Images are not copied during splitting. Augmented variants stay with their source pile and are included only in the training split. Validation and test contain only originals; their augmented copies remain on disk but are excluded from all splits.

Split files contain frame IDs, pile IDs, and the dataset manifest hash. Loading checks for overlap, omitted eligible frames, and changes to the prepared manifest. Existing split files are not overwritten. Keep the splits fixed across model experiments; do not use test results for checkpoint selection or threshold tuning.

Training loads saved image tensors and targets without augmentation, box calculation, or annotation generation. Conversion of stored uint8 RGB to floating point, batch loading, device transfer, and the model's normal resize/normalization still occur during training. Foreground label `1` denotes every object instance; `0` is background. Fully hidden objects have no target mask; tiny visible fragments are retained. Raw generated datasets must be prepared and split before training.

### 4. Verify that the model can overfit a few images

Before a larger experiment, deliberately train on four training images repeatedly:

```bash
pixi run -e ml train-segmentation \
  --dataset ml_ws/datasets/segmentation/segmentation-v1 \
  --name maskrcnn-overfit-v1 \
  --overfit-images 4 \
  --steps-per-epoch 25 \
  --epochs 10
```

Overfit mode selects only prepared original training images and evaluates on those same images. Its records are explicitly marked `overfit_train`: this is a memorization/implementation diagnostic, not held-out validation performance. Expect training loss to decrease and training-image mask AP to improve. There is no hard pass threshold because image complexity varies. If neither improves, inspect the data and optimization before scaling up.

The first pretrained run downloads official Torchvision weights into `ml_ws/segmentation/weights/`. A working internet connection is required for that first download. Checkpoints and downloaded weights are excluded from Git. Use new experiment names for new runs; nonempty experiment folders are not overwritten.

### 5. Train using held-out validation

```bash
pixi run -e ml train-segmentation \
  --dataset ml_ws/datasets/segmentation/segmentation-v1 \
  --name maskrcnn-v1
```

Defaults are configured in `ml_ws/segmentation/configs/maskrcnn.toml`: 20 epochs, batch size 2, SGD with momentum, a learning-rate schedule, CUDA mixed precision, and input resizing with a 480-pixel short side and an 800-pixel maximum long side. The model uses pretrained visual features with new single-class box and mask heads. Every epoch evaluates mask AP on the validation split; the test split is not used during training.

Use `--config path/to/config.toml` for another configuration, `--epochs N` or `--batch-size N` to override those settings, and `--device cpu` for explicit CPU execution (much slower). Defaults use `cuda:0`. If GPU memory is insufficient, lower the batch size or resize limits in the configuration. `--steps-per-epoch` repeats batches to a selected update count; omit it for normal full passes over the training split.

All random augmentation and annotation preparation happen in step 2. Changing augmentation settings at training time has no effect on the prepared dataset. To change them, prepare a new dataset and create new splits. Validation and prediction do not use random augmentation. Aggressive crops, mosaic, copy-paste, and lighting randomization in Isaac Sim are not enabled.

For a true random-weight experiment, copy the configuration and set `[model] pretrained = false`. This disables both full-model and backbone pretrained weights. Fine-tuning is the default and should be the first baseline.

Outputs:

```text
ml_ws/outputs/segmentation/maskrcnn-v1/
├── config.toml              # Original configuration
├── resolved_config.json     # Configuration including CLI overrides
├── manifest.json            # Dataset, split hashes, model, and training subset
├── training_metrics.jsonl   # Baseline and per-epoch losses/AP
├── final_validation.json    # Best checkpoint re-evaluated after training
├── tensorboard/             # TensorBoard event logs
└── checkpoints/
    ├── best.pt              # Best validation mask AP (or overfit AP in diagnostic mode)
    └── last.pt              # Most recent epoch, for resuming
```

Checkpoints include model, optimizer, scheduler, mixed-precision scaler, epoch, random-generator states, and split fingerprints. Resume in the original experiment directory, with `--epochs` giving the new total epoch count:

```bash
pixi run -e ml train-segmentation \
  --resume ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/last.pt \
  --epochs 30
```

Resume keeps the saved dataset, configuration, and subset; do not provide `--name`. Checkpoints use Python serialization and should only be loaded from trusted sources. Resume is supported at epoch boundaries; identical results across different hardware/software are not guaranteed.

After training finishes, `best.pt` is loaded again and evaluated on the validation split. `final_validation.json` records its epoch, image count, and mask metrics. In overfit mode this uses the training subset and is marked `diagnostic_only`; it is not held-out validation. The test split is reserved for the explicit final evaluation below.

TensorBoard logs are written automatically for each experiment. Start the viewer in another terminal:

```bash
pixi run -e ml tensorboard
```

Open http://localhost:6006 to compare runs. Scalars include component/total training losses, validation mask AP (AP50, AP75, AP50–95), precision, recall, matched IoU, learning rate, and epoch duration. Overfit metrics appear under `overfit_train`, separately from `validation`. Resuming appends to the same experiment logs. Logs from older runs are not backfilled.

### 6. Predict and evaluate

During model development, use validation images:

```bash
pixi run -e ml segment-dataset \
  --checkpoint ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt \
  --split validation \
  --name maskrcnn-v1-validation

pixi run -e ml evaluate-segmentation \
  --predictions ml_ws/outputs/segmentation/maskrcnn-v1-validation/predictions \
  --score-threshold 0.3
```

Choose confidence thresholds using validation results. Once the checkpoint and thresholds are fixed, run the held-out test split:

```bash
pixi run -e ml segment-dataset \
  --checkpoint ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt \
  --split test \
  --name maskrcnn-v1-test

pixi run -e ml evaluate-segmentation \
  --predictions ml_ws/outputs/segmentation/maskrcnn-v1-test/predictions \
  --score-threshold 0.3
```

Prediction defaults to the checkpoint's dataset and the test split. Supply `--dataset PATH` to process another dataset; use `--split all` when processing a complete generated dataset without training split files. Predictions are saved in the new run's `predictions/` folder as NPZ masks/boxes/scores/labels and per-image JSON metadata. Inference does not download pretrained weights when loading a trained checkpoint.

The run manifest records selected frame IDs. Evaluation reads only those frames, rather than requiring predictions for other splits. Outputs go into the prediction run's sibling `evaluation/` directory:

```text
ml_ws/outputs/segmentation/maskrcnn-v1-test/
├── manifest.json
├── predictions/
└── evaluation/
    ├── metrics.json
    ├── per_image.json
    └── overlays/
```

Evaluation reports class-agnostic visible-instance box/mask precision, recall, matched IoU, and AP. AP uses confidence-ranked matching, 101 recall points, IoU thresholds 0.50–0.95, and at most 100 predictions per image; it is not the full COCO benchmark with class/area breakdowns. Precision/recall default to confidence 0.3 and IoU 0.5. The model saves predictions above its configured score floor (default 0.01); thresholds below that floor cannot recover discarded predictions.

Overlay colors: green is matched ground truth, yellow is missed ground truth, blue is a matched prediction, and red is an unmatched prediction. Matches in overlays use mask IoU. Repeating evaluation on the same prediction directory overwrites metrics and overlays; use a new prediction run name to preserve separate comparisons.

### Code layout and checks

```text
ml_ws/segmentation/
├── configs/maskrcnn.toml
├── dataset.py               # Manifest validation, pile splits, mask-derived targets
├── transforms.py            # Coordinated RGB/mask augmentation
├── model.py                 # Single-class Mask R-CNN construction
├── engine.py                # Training and mask-AP validation
├── scripts/
│   ├── prepare_dataset.py    # Offline augmentation and saved targets
│   ├── prepare_splits.py
│   ├── train.py
│   └── predict.py
└── tests/
```

Model-independent metrics and visualization remain in `ml_ws/scripts/evaluation/`. Run the checks through Pixi:

```bash
pixi run -e ml python -m unittest discover -s ml_ws/segmentation/tests -v
pixi run -e ml env PYTHONPATH=ml_ws/scripts/evaluation python -m unittest discover -s ml_ws/scripts/evaluation/tests -v
```

Existing raw datasets and experiments are preserved. Older checkpoints tied to unprepared datasets cannot resume with this training loader; use a freshly prepared dataset and a new experiment name. Prediction from older checkpoints remains supported.
