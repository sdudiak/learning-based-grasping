# Learning-based grasping

Learning-based robotic grasping with a Franka Panda in Isaac Sim, ROS 2 Jazzy/MoveIt, Mask R-CNN instance segmentation, and selection of visible object point clouds. Workspace guides: [ML](ml_ws/README.md) and [ROS](ros_ws/README.md).

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

## ROS and perception

| Task | Purpose |
| --- | --- |
| `ros-build` | Build robot bringup, point-cloud processing, interfaces, and RViz plugin |
| `ros-moveit` | Launch robot TF, controllers, and MoveIt |
| `ros-segmentation` | Run wrist-camera inference and camera mounting TF |
| `ros-pointcloud` | Run point-cloud separation using aligned depth and segmentation masks |
| `ros-perception` | Launch segmentation, point-cloud processing, and camera mounting TF together |
| `rviz` | Open a combined MoveIt/perception viewer with clickable object selection |

Start the full setup in separate terminals:

```bash
pixi run isaacsim-random-objects-scene --enable isaacsim.ros2.bridge
pixi run ros-moveit
pixi run ros-perception
pixi run rviz
```

The simulator needs converted GSO assets, and inference needs a trained checkpoint at `ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt`. Override it with `pixi run ros-perception checkpoint:=/absolute/path/to/best.pt`.

In RViz, click an object in **Annotated Image**. The viewer resolves the pixel's instance ID against the exact displayed mask frame and requests target selection. The target cloud appears orange, with a centroid and bounding box; the remaining scene stays gray. Results appear beneath the image. Clear the selection from a terminal:

```bash
pixi shell
ros2 service call /pointcloud/clear_target std_srvs/srv/Trigger '{}'
```

ROS processing tasks default to `rviz:=false`; pass `rviz:=true` for an individual viewer. `ros-perception` starts no MoveIt nodes or controllers. The default output frame is `panda_link0`, so the robot's moving TF chain must be available from `ros-moveit` or another source. Camera mounting TF belongs to segmentation.

`pixi shell` automatically sources ROS Jazzy and the built ROS workspace. Isaac Sim's Python 3.11 bridge library path is isolated to simulator tasks; ROS CLI commands use Jazzy's system Python 3.12. Re-enter the shell after the first build to discover newly installed packages. See the [ROS guide](ros_ws/README.md) for interfaces, configuration, tests, and troubleshooting.
