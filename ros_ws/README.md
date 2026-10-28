# ROS 2 workspace

This workspace connects the Isaac Sim Panda scene to ROS 2 Jazzy and runs live instance segmentation. Run the Pixi commands below from the **repository root**.

## Packages

| Package | Responsibility |
| --- | --- |
| `isaac_manipulation_bringup` | Panda robot description, topic-based `ros2_control`, controllers, MoveIt, robot RViz, and camera transforms |
| `isaac_manipulation_segmentation` | Camera subscription, Mask R-CNN inference, detection/mask publication, and the segmentation RViz viewer |
| `moveit_resources` submodule | Upstream robot descriptions/configurations; bringup uses the Panda packages |
| `topic_based_ros2_control` submodule | Hardware-interface plugin exchanging joint state/command topics with Isaac Sim |

The segmentation package keeps ROS message/publisher logic in `node.py`, model inference in `inference.py`, frame selection and overlay processing in `processing.py`, and parameter loading in `configuration.py`. It imports the shared model code from `ml_ws`.

Training, dataset preparation, and offline evaluation are documented in the [ML workspace README](../ml_ws/README.md).

## Prerequisites and build

The task commands assume ROS 2 Jazzy is installed at `/opt/ros/jazzy`, with colcon, rosdep, MoveIt, RViz, `ros2_control`, `cv_bridge`, and the ROS message dependencies available. A working NVIDIA GPU/driver is needed for Isaac Sim and the default CUDA inference device.

Initialize the pinned submodules before building:

```bash
git submodule update --init --recursive
```

On a newly configured ROS installation, resolve workspace dependencies:

```bash
source /opt/ros/jazzy/setup.bash
rosdep update
rosdep install --from-paths ros_ws/src --ignore-src --rosdistro jazzy -r -y
```

rosdep must already be initialized on the machine. Build the bringup and its source dependencies with:

```bash
pixi run ros-build
```

This builds packages up to `isaac_manipulation_bringup`. The segmentation Pixi task builds its own package before launching it. Missing `moveit_resources_panda_moveit_config` usually means the submodules were not initialized or the build/install setup has not been sourced.

## Runtime environments

Isaac Sim and offline training use Python 3.11. The live segmentation task uses the separate Pixi `segmentation` environment with Python 3.12, matching the installed Jazzy Python bindings. It includes the editable ML package, CUDA PyTorch/torchvision, and image/launch dependencies.

```bash
pixi install -e default
pixi install -e segmentation
```

The Pixi tasks isolate the simulator's libraries from the system ROS build environment. Use the supplied tasks for the normal workflow.

## Start the simulator

In one terminal:

```bash
pixi run isaacsim-random-objects-scene --enable isaacsim.ros2.bridge
```

This opens the empty-table scene, automatically spawns and settles GSO objects, and hides the spawner panel. It requires converted assets; see the [asset and generation setup](../ml_ws/README.md#1-download-and-convert-objects).

For the original scene:

```bash
pixi run isaacsim-default-scene --enable isaacsim.ros2.bridge
```

Press **Play** for the original scene. Camera images and `/clock` are published while simulation advances. The random-object scene advances simulation automatically during spawning and continues running afterward.

The USD camera graphs currently use the spelling `cameara` in their topic namespaces:

| Camera | RGB topic |
| --- | --- |
| Top | `/isaac/cameara/top/color/image_raw` |
| Wrist | `/isaac/cameara/wrist/color/image_raw` |

Both graphs also publish depth images and camera information. The segmentation model consumes RGB only.

## Run top-camera segmentation and RViz

In another terminal:

```bash
pixi run segmentation-top-camera
```

The task builds `isaac_manipulation_segmentation`, starts the node on the top RGB topic, enables overlay publication, and opens RViz with **Raw Image** and **Annotated Image** docked side by side. It does not start Isaac Sim.

Its default checkpoint is:

```text
ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt
```

Weights are ignored by Git, so train or copy the checkpoint locally, or override it:

```bash
pixi run segmentation-top-camera \
  checkpoint:=/absolute/path/to/best.pt \
  score_threshold:=0.7 \
  decimation:=2
```

To use another RGB topic with the same viewer:

```bash
pixi run segmentation-top-camera \
  image_topic:=/isaac/cameara/wrist/color/image_raw
```

Close RViz or press **Ctrl+C** to stop the viewer launch and inference node together. Isaac Sim runs independently.

## Parameters and outputs

Application defaults are declared in [`src/isaac_manipulation_segmentation/config/params.yaml`](src/isaac_manipulation_segmentation/config/params.yaml). The generic launch defaults to the wrist camera; the top-camera launch overrides the RGB topic and enables overlays.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `checkpoint` | Empty in YAML | Trained checkpoint; the Pixi task supplies the default path above |
| `device` | `cuda:0` | Inference device; use `cpu` explicitly when needed |
| `image_topic` | Wrist RGB in YAML | RGB input; top-camera launch selects top RGB |
| `use_sim_time` | `true` | Use Isaac Sim's `/clock` |
| `publish_overlay_image` | `true` | Enable annotated-image rendering and publication |
| `decimation` | `1` | Select every Nth received frame, starting with the first |
| `score_threshold` | `0.5` | Minimum detection confidence |
| `mask_threshold` | `0.5` | Pixel probability cutoff when forming binary masks |
| `tracking_iou_threshold` | `0.3` | Bounding-box overlap needed to reuse an overlay color |
| `tracking_max_missed_frames` | `5` | Processed frames a color track can remain unmatched |
| `detections_topic` | `~/detections` | Detection output |
| `instance_mask_topic` | `~/instance_mask` | Instance-label image output |
| `overlay_image_topic` | `~/overlay_image` | Annotated RGB output |

Parameters can be overridden as `name:=value` launch arguments or through `params_file:=/absolute/path/to/params.yaml`. Application parameters are set at startup; restart the node after changing them.

Decimation happens before image conversion and inference. When processing falls behind, only the latest selected pending frame is retained, preventing a growing backlog. A value of `1` selects every frame but does not guarantee processing every frame under overload.

The default node name is `segmentation`, giving these outputs:

| Topic | Type / contents |
| --- | --- |
| `/segmentation/detections` | `vision_msgs/Detection2DArray`: boxes, class labels, confidence scores, and frame-local IDs |
| `/segmentation/instance_mask` | `sensor_msgs/Image`, `16UC1`: background `0`, object labels matching detection IDs |
| `/segmentation/overlay_image` | `sensor_msgs/Image`, `rgb8`: colored masks, boxes, and scores |

Outputs retain the input timestamp and camera frame. Overlapping mask pixels go to the higher-confidence detection. Detection IDs are frame-local; overlay colors are associated across frames by bounding-box overlap. Colors can change after long occlusions, large motion, or ambiguous overlaps. Tracking history resets when the camera frame/resolution changes or timestamps move backward.

## Launch the node without RViz

After building the segmentation package, a manual launch can be run inside the Python 3.12 Pixi environment. From a clean terminal at the repository root:

```bash
pixi shell -e segmentation
unset PYTHONPATH OLD_PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
cd ros_ws
colcon build --symlink-install --packages-select isaac_manipulation_segmentation
source install/setup.bash
python /opt/ros/jazzy/bin/ros2 launch isaac_manipulation_segmentation segmentation.launch.py \
  checkpoint:=/absolute/path/to/best.pt \
  image_topic:=/isaac/cameara/top/color/image_raw
```

The explicit `python` invocation runs launch and inference with the Pixi interpreter containing PyTorch, while using the installed Jazzy bindings.

## MoveIt and robot control

With Isaac Sim running:

```bash
pixi run ros-moveit
```

This builds the bringup dependencies and launches the robot description, controllers, MoveIt, camera transforms, and robot RViz. It exchanges `/isaac/joint_states` and `/isaac/joint_command` with the simulator.

Use `pixi run ros-moveit rviz:=false` when only the segmentation RViz window is wanted. The `cameras` launch argument controls camera TF publishers; it does not start the simulator's camera streams. Image comparison in the segmentation viewer does not require MoveIt to be running.

## Checks and troubleshooting

Run the ROS package tests after building, from a clean system-ROS terminal:

```bash
source /opt/ros/jazzy/setup.bash
cd ros_ws
source install/setup.bash
ROS_DOMAIN_ID=91 colcon test --packages-select isaac_manipulation_segmentation --event-handlers console_direct+
colcon test-result --verbose
```

The example uses a separate ROS domain so test publishers do not mix with a running simulator or inference node.

For missing images, check that simulation is playing, the ROS bridge is enabled, camera variants are enabled, and both processes use the same `ROS_DOMAIN_ID`. Check the raw topic first; then check the node's checkpoint/device and annotated output. A higher confidence threshold can produce an overlay with no detections.

Generated `build/`, `install/`, and `log/` directories are ignored by Git. Model checkpoints and training/evaluation artifacts live under the ignored ML artifact directories.
