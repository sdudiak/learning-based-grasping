# ROS 2 workspace

ROS 2 Jazzy integration for the Isaac Sim Panda and live segmentation. Run Pixi commands from the repository root; see the [ML guide](../ml_ws/README.md) for data and training.

## Packages and setup

| Package | Purpose |
| --- | --- |
| `isaac_manipulation_bringup` | Robot description, controllers, MoveIt, camera TF, and robot RViz |
| `isaac_manipulation_segmentation` | RGB inference, detections/masks, and segmentation RViz |
| `moveit_resources` submodule | Panda description and MoveIt configuration |
| `topic_based_ros2_control` submodule | Joint state/command hardware interface |

Requires ROS Jazzy at `/opt/ros/jazzy`, colcon, MoveIt, RViz, ros2_control, cv_bridge, and message dependencies. Isaac Sim uses Python 3.11; the Pixi `segmentation` environment uses Python 3.12 for Jazzy bindings and PyTorch.

```bash
git submodule update --init --recursive
pixi install -e default
pixi install -e segmentation
pixi run ros-build
```

On a new ROS installation, initialize rosdep if necessary, then run `rosdep update` and `rosdep install --from-paths ros_ws/src --ignore-src --rosdistro jazzy -r -y`. `ros-build` builds bringup dependencies; the segmentation task builds its own package.

## Simulator and segmentation

Run these commands in separate terminals:

```bash
pixi run isaacsim-random-objects-scene --enable isaacsim.ros2.bridge
pixi run segmentation-top-camera
```

The simulator needs converted GSO assets and automatically spawns objects on the empty table. The segmentation task opens side-by-side raw/annotated image panes in RViz. For the original scene, use `pixi run isaacsim-default-scene --enable isaacsim.ros2.bridge` and press **Play**.

The default checkpoint is `ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt`. Train/copy it locally, or override it:

```bash
pixi run segmentation-top-camera \
  checkpoint:=/absolute/path/to/best.pt \
  score_threshold:=0.7 decimation:=2
```

Application settings live in [`config/params.yaml`](src/isaac_manipulation_segmentation/config/params.yaml). Override them with `name:=value` or `params_file:=/absolute/path/to/params.yaml`, then restart the node. Common defaults: `device=cuda:0`, `score_threshold=0.5`, `mask_threshold=0.5`, `decimation=1`, and `publish_overlay_image=true`.

The top-camera launch selects `/isaac/cameara/top/color/image_raw`. Override `image_topic` to use `/isaac/cameara/wrist/color/image_raw` or another RGB stream. The spelling `cameara` matches the current USD graphs. The generic `segmentation.launch.py` defaults to the wrist camera and launches inference without RViz.

| Output topic | Contents |
| --- | --- |
| `/segmentation/detections` | `Detection2DArray`: boxes, labels, confidence, frame-local IDs |
| `/segmentation/instance_mask` | `16UC1` image: background 0, labels matching detection IDs |
| `/segmentation/overlay_image` | Annotated `rgb8` image |

Outputs retain the source timestamp/frame. Decimation selects every Nth frame; under overload, only the latest selected pending frame is retained. Overlay colors use box-overlap matching, tolerate five missed processed frames by default, and can change after occlusion or ambiguous matches.

Close RViz or press **Ctrl+C** to stop inference and its viewer. Isaac Sim runs independently.

## Robot control and checks

```bash
pixi run ros-moveit
```

This launches controllers, MoveIt, camera TF, and robot RViz using `/isaac/joint_states` and `/isaac/joint_command`. Add `rviz:=false` to omit robot RViz. The `cameras` argument controls TF publishers, not simulator camera streams; segmentation does not require MoveIt.

After building, run tests from a clean ROS terminal:

```bash
source /opt/ros/jazzy/setup.bash
cd ros_ws
source install/setup.bash
ROS_DOMAIN_ID=91 colcon test --packages-select isaac_manipulation_segmentation --event-handlers console_direct+
colcon test-result --verbose
```

For missing images, check simulation playback, bridge/camera enablement, matching `ROS_DOMAIN_ID`, the raw topic, and checkpoint/device settings. Missing Panda configuration usually means uninitialized submodules or an unsourced build. Generated build/install/log directories and model artifacts are ignored by Git.
