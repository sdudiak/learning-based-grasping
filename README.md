# Learning-based grasping

Learning-based robotic grasping with a Franka Panda in Isaac Sim, ROS 2 Jazzy/MoveIt, and Mask R-CNN instance segmentation. Workspace guides: [ML](ml_ws/README.md) and [ROS](ros_ws/README.md).

Run tasks from the repository root with `pixi run <task>`.

## Simulation and assets

| Task | Purpose |
| --- | --- |
| `isaacsim` | Open Isaac Sim |
| `isaacsim-default-scene` | Open the default Panda scene |
| `isaacsim-random-objects-scene` | Open the empty-table scene and spawn random objects |
| `download-gso-objects` | Download Google Scanned Objects |
| `convert-gso-objects` | Convert objects to physics-enabled USD |
| `generate-dataset` | Generate RGB images and instance annotations |
| `save-masked-images` | Create ground-truth mask previews |

## Training and evaluation

These tasks use the `ml` environment automatically.

| Task | Purpose |
| --- | --- |
| `prepare-segmentation-dataset` | Prepare fixed augmentations and training tensors |
| `prepare-segmentation-splits` | Create pile-disjoint train/validation/test splits |
| `train-segmentation` | Train or resume Mask R-CNN |
| `tensorboard` | View training metrics at `http://127.0.0.1:6006` |
| `segment-dataset` | Save checkpoint predictions |
| `evaluate-segmentation` | Compute box and mask metrics |
| `compare-segmentation` | Render ground-truth/prediction comparisons |

## ROS and live inference

| Task | Purpose |
| --- | --- |
| `ros-build` | Build Panda bringup and its dependencies |
| `ros-moveit` | Launch controllers, MoveIt, camera TF, and robot RViz |
| `ros-segmentation-top-camera` | Run top-camera inference and side-by-side raw/annotated RViz views |
