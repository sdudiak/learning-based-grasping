# ROS 2 workspace

ROS 2 Jazzy integration for the Isaac Sim Panda, Mask R-CNN inference, and target/environment point-cloud separation. Run Pixi tasks from the repository root. Training and datasets are covered in the [ML guide](../ml_ws/README.md).

## Setup

Requires ROS Jazzy at `/opt/ros/jazzy`, colcon, MoveIt, RViz, ros2_control, cv_bridge, Qt development libraries, and ROS message dependencies. Initialize rosdep if necessary, then install workspace dependencies:

```bash
git submodule update --init --recursive
rosdep update
rosdep install --from-paths ros_ws/src --ignore-src --rosdistro jazzy -r -y
pixi install -e default
pixi install -e segmentation
pixi run ros-build
```

Isaac Sim uses Python 3.11. Inference uses the Pixi `segmentation` environment with Python 3.12 and PyTorch; ROS builds and the point-cloud node use system Jazzy Python. The tasks handle this isolation automatically. `pixi shell` sources clean Jazzy settings and the workspace overlay when built. Simulator tasks add the Isaac bridge library path only for their own process. Re-enter an existing shell after building new packages.

| Package | Purpose |
| --- | --- |
| `isaac_manipulation_bringup` | Robot TF, controllers, MoveIt, and combined viewer launch |
| `isaac_manipulation_segmentation` | RGB inference, masks, camera mounting TF, and perception pipeline launch |
| `isaac_manipulation_pointcloud` | Depth projection, scene partition, target association, and selection services |
| `isaac_manipulation_pointcloud_interfaces` | Position and timestamped-instance selection services |
| `isaac_manipulation_rviz` | Clickable Annotated Image display |
| `moveit_resources` submodule | Panda description and MoveIt configuration |
| `topic_based_ros2_control` submodule | Joint state/command hardware interface |

## Run the full pipeline

Run each command in a separate terminal:

```bash
pixi run isaacsim-random-objects-scene --enable isaacsim.ros2.bridge
pixi run ros-moveit
pixi run ros-perception
pixi run rviz
```

The simulator needs converted GSO assets. For the original scene, use `pixi run isaacsim-default-scene --enable isaacsim.ros2.bridge` and press **Play**. Restart the scene after changing camera graphs.

`ros-perception` launches only segmentation, point-cloud processing, and camera mounting TF. It never launches MoveIt or controllers and has no `bringup` argument. `ros-moveit` provides the robot's moving TF chain and controls through `/isaac/joint_states` and `/isaac/joint_command`; it does not publish camera mounting TF. Another source may provide robot TF instead of MoveIt. Without the complete camera-to-`panda_link0` chain, point-cloud output cannot be transformed.

All ROS processing launches default to **`rviz:=false`**. Pass `rviz:=true` to enable an individual viewer. `pixi run rviz` launches only the combined viewer; closing it leaves the processing nodes running. Stop processing tasks with **Ctrl+C**.

The inference checkpoint defaults to `ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt`. Common pipeline overrides:

```bash
pixi run ros-perception checkpoint:=/absolute/path/to/best.pt score_threshold:=0.7 decimation:=2
pixi run ros-perception segmentation_params_file:=/absolute/segmentation.yaml pointcloud_params_file:=/absolute/pointcloud.yaml
```

For individual components, use `ros-segmentation` or `ros-pointcloud`, each accepting `params_file:=/absolute/path.yaml`. `ros-segmentation` also accepts segmentation parameters as `name:=value`. The pipeline forwards `checkpoint`, `device`, `score_threshold`, and `decimation`; use its parameter files for other settings. Avoid running standalone segmentation and the perception pipeline together, since both publish masks and camera TF.

`ros-build` builds robot bringup and point-cloud dependencies, including the RViz plugin. Segmentation/perception tasks build their own ROS dependencies before launching. Direct launches are `segmentation.launch.py` and `pipeline.launch.py` in the segmentation package, `pointcloud.launch.py` in the point-cloud package, and `moveit.launch.py` / `rviz.launch.py` in bringup. Inference needs the segmentation Python environment.

## Select an object in RViz

The combined viewer uses `panda_link0` as its fixed frame and includes:

| Display | Purpose |
| --- | --- |
| MotionPlanning | Panda model, planning scene, interactive planning, and planned trajectories |
| Raw Image | Wrist-camera RGB |
| Annotated Image | Segmentation overlay with direct click selection |
| Environment | Gray scene points |
| Target | Orange selected-object points |
| Target centroid | Yellow mean of visible target points |
| Target bounding box | Translucent axis-aligned cube |
| TF (debug) | Optional frame inspection; disabled by default |

Click inside an object's mask in **Annotated Image**. The custom `isaac_manipulation_rviz/SelectableImage` display pairs overlay and mask messages by exact timestamp and frame, converts the click through image scaling/letterboxing, and sends the pixel's instance ID and mask timestamp to `/pointcloud/select_instance`. No `/clicked_point` topic or toolbar selection tool is involved. IDs are frame-local; the server resolves the clicked ID into a persistent spatial reference. Background and letterbox clicks leave the selection unchanged.

The label below the image reports acceptance, rejection, missing service, or timeout. The display retains at most ten messages per image/mask input and allows one asynchronous request at a time, with a three-second response timeout. It supports the pipeline's `rgb8` overlay and `16UC1` mask, including padded rows and either mask byte order. An old displayed frame can be rejected when it has expired from the server cache. A timed-out request may still complete on the server.

For custom topics, set **Image Topic** and **Mask Topic** on the Annotated Image display and the corresponding ROS configuration. The standalone combined viewer uses the default perception topics; custom pipeline configurations may require updating its other displays too. `pixi run rviz rviz_config:=/absolute/custom.rviz` loads another layout; `use_sim_time:=false` selects wall time.

## Architecture and interfaces

The simulator camera graph publishes RGB, CameraInfo, and depth, but no native PointCloud2. `AlignedDepthImage` renders depth from the same color render product as RGB, so mask pixels and projected points correspond exactly. The separate physical depth camera is not used for this projection.

```text
RGB -> Mask R-CNN -> annotated image + instance mask -> clickable RViz display
Color-aligned depth + CameraInfo + mask + timestamped TF -> scene partition
RViz click -> select_instance(mask timestamp, ID) -> fixed base-frame reference
```

| Input topic | Type / purpose |
| --- | --- |
| `/isaac/cameara/wrist/color/image_raw` | Image: inference input |
| `/isaac/cameara/wrist/aligned_depth_to_color/image_raw` | Image: `32FC1` meters or `16UC1` millimeters |
| `/isaac/cameara/wrist/color/camera_info` | CameraInfo: color intrinsics |
| `/segmentation/instance_mask` | `16UC1` Image: 0 background, nonzero instance IDs |
| `/tf`, `/tf_static` | Camera/robot transforms at acquisition time |

The spelling `cameara` matches the simulator USD graph. Relevant frames are `panda_link0`, `panda_hand`, `wrist_camera_link`, `wrist_color_optical_frame`, and `wrist_depth_optical_frame`. Projection uses `wrist_color_optical_frame`.

| Published topic | Type / contents |
| --- | --- |
| `/segmentation/detections` | Detection2DArray: boxes, classes, confidence, frame-local IDs |
| `/segmentation/instance_mask` | Image: frame-local labels |
| `/segmentation/overlay_image` | Image: annotated `rgb8` |
| `/pointcloud/environment` | PointCloud2: full valid scene in IDLE, complementary scene after selection |
| `/pointcloud/target` | PointCloud2: matched target's visible XYZ points |
| `/pointcloud/target_centroid` | PointStamped: mean of target points |
| `/pointcloud/target_bbox` | Marker: axis-aligned cube in output frame |

Segmentation retains source headers. Decimation processes every Nth input frame; under overload, only the latest pending selected frame is retained. Overlay colors use box-overlap association and are unrelated to persistent target identity.

The point-cloud node starts in IDLE, publishing the full valid scene at depth rate independently of inference. It uses bounded timestamp caches and nonblocking acquisition-time TF lookups. By default, masks, depth, and intrinsics require identical source timestamps. Missing or stale data is skipped. Sensor subscriptions use best-effort QoS; output queues have depth one. Nonfinite, zero, negative, and out-of-range depths are discarded.

After selection, each synchronized observation is split into disjoint target/environment clouds. The target is associated conservatively against its initial centroid in `panda_link0`, without drifting the reference. If association fails or is ambiguous, the full synchronized scene becomes environment and no new target/centroid is emitted. The reference remains available for reacquisition. Clearing returns to IDLE. Boxes are deleted on clear/loss and expire after 0.5 seconds. Target displays expire shortly afterward; the centroid display retains its last observation, so downstream consumers must check timestamps.

## Selection services

Enter `pixi shell` from the repository root for ROS CLI access. All services return `success` and `message`.

| Service | Type / behavior |
| --- | --- |
| `/pointcloud/select_target` | `isaac_manipulation_pointcloud_interfaces/srv/SelectTarget`: select nearest unambiguous centroid to a PointStamped reference |
| `/pointcloud/select_instance` | `isaac_manipulation_pointcloud_interfaces/srv/SelectInstance`: resolve ID against an exact cached mask timestamp |
| `/pointcloud/clear_target` | `std_srvs/srv/Trigger`: clear reference and restore IDLE |

```bash
ros2 service call /pointcloud/select_target isaac_manipulation_pointcloud_interfaces/srv/SelectTarget   '{reference_position: {header: {frame_id: panda_link0}, point: {x: 0.5, y: 0.0, z: 0.1}}}'
ros2 service call /pointcloud/select_instance isaac_manipulation_pointcloud_interfaces/srv/SelectInstance   '{stamp: {sec: 42, nanosec: 0}, instance_id: 1}'
ros2 service call /pointcloud/clear_target std_srvs/srv/Trigger '{}'
```

Replace example values with actual observations. Position coordinates are meters. References already in the output frame need no transform; other frames require a nonzero acquisition timestamp and a valid TF transform at that time. Instance selection requires a fresh cached frame, not an RViz click timestamp. The Annotated Image display supplies this automatically.

## Configuration and limitations

Inference settings: [`segmentation/config/params.yaml`](src/isaac_manipulation_segmentation/config/params.yaml). Defaults include `device=cuda:0`, `score_threshold=0.99`, `mask_threshold=0.5`, `decimation=1`, and `publish_overlay_image=true`. The perception pipeline enables the overlay for image selection.

Point-cloud settings: [`pointcloud/config/params.yaml`](src/isaac_manipulation_pointcloud/config/params.yaml). Parameters are read at startup:

| Parameters | Meaning/default |
| --- | --- |
| `use_sim_time` | `true`; simulator clock |
| `depth_topic`, `camera_info_topic`, `mask_topic` | Inputs listed above; depth must share RGB projection |
| `input_cloud_topic` | Empty; this integration uses aligned depth, and rejects a nonempty value |
| `target_topic`, `environment_topic`, `centroid_topic`, `bounding_box_topic` | Outputs listed above |
| `output_frame` | `panda_link0`; geometry and association frame |
| `selection_distance`, `association_distance` | Maximum centroid distance, 0.10 / 0.08 m |
| `ambiguity_margin` | Reject when runner-up distance is within 0.02 m of nearest |
| `minimum_points` | 20 valid points per candidate |
| `synchronization_tolerance` | 0 seconds; only relax if source timestamps still identify the same exposure |
| `maximum_observation_age`, `cache_capacity` | 3 seconds and 90 messages per cache; size for inference latency and available memory |
| `minimum_depth`, `maximum_depth` | 0.01–10 m; invalid depths are discarded |
| `voxel_resolution` | 0 disables; positive meters samples one point per voxel and label on synchronized observations |
| `visualization` | `true`; publish bounding-box cubes |
| `debug` | Log synchronized point/candidate counts; warnings are throttled independently |

Partial surface centroids shift with viewpoint. Position gating is an initial association method, not guaranteed tracking: objects at the same or nearby positions can remain indistinguishable. It cannot identify a replacement object occupying the same location. XYZ is reconstructed without RGB. With voxelization disabled, the partition reconstructs all valid depth geometry; enabled voxelization partitions a sampled scene. No grasp estimation or execution is included.

## Tests and troubleshooting

Build with tests enabled and use a clean system ROS terminal:

```bash
source /opt/ros/jazzy/setup.bash
cd ros_ws
colcon build --symlink-install --packages-up-to isaac_manipulation_segmentation isaac_manipulation_bringup --cmake-args -DBUILD_TESTING=ON -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
ROS_DOMAIN_ID=91 colcon test --packages-select isaac_manipulation_rviz isaac_manipulation_pointcloud isaac_manipulation_segmentation --event-handlers console_direct+
colcon test-result --verbose
```

Tests cover numerical projection/partition/association, service callback behavior, launch defaults and isolation, pixel scaling/letterboxing, mask byte order, and exact image/mask matching. Live validation also needs simulator playback, a checkpoint, and robot/camera TF. The full pipeline and image-click selection have been exercised in the simulator.

- **No images:** check playback, camera/bridge enablement, `ROS_DOMAIN_ID`, checkpoint/device settings, and input topics.
- **No clouds or stale observations:** check aligned depth, CameraInfo, matching timestamps, TF at acquisition time, and inference latency against cache age/capacity.
- **Selection rejected:** use an object's interior mask pixels; the frame may be stale, too sparse, or spatially ambiguous. Review the message beneath Annotated Image.
- **Python/type-support errors:** use a fresh `pixi shell`; avoid adding Isaac bridge library paths to interactive ROS commands.
- **Missing RViz display/plugin:** rebuild with the Pixi tasks and restart RViz. Add the display manually as `isaac_manipulation_rviz/SelectableImage` if using an older saved layout.
- **Missing Panda configuration:** initialize submodules and source the workspace overlay.

Generated build/install/log directories, Python caches, and model artifacts are ignored by Git.
