# isaac-manipulation

An example of running MoveIt 2 and cameras in Isaac Sim 5.1 with ROS 2 Jazzy.
MoveIt plans and executes motions for a simulated Franka Panda arm and gripper,
while wrist-mounted and overhead cameras publish RGB images, depth images, and
camera information over ROS 2.

## Components

- `isaac_ws/assets/`: USD assets for the Panda, table, cube, and default scene
  (`scenes/default_secene.usda`). OmniGraphs bridge joint commands, joint states,
  simulation time, and camera streams to ROS 2. The scene's `cameras` variant
  enables or disables both cameras.
- `isaac_ws/extensions/isaac_manipulation.scene_loader/`: Isaac Sim extension
  that opens a USD scene supplied through its command-line setting.
- `ros_ws/src/isaac_manipulation_bringup/`: Launch files and controller
  configuration for MoveIt, RViz, robot transforms, camera transforms, and
  ros2_control arm/gripper controllers. Joint commands and feedback use
  `/isaac/joint_command` and `/isaac/joint_states`.
- `ros_ws/src/moveit_resources/`: Submodule providing the Panda robot description
  and MoveIt configuration.
- `ros_ws/src/topic_based_ros2_control/`: Submodule providing the topic-based
  hardware interface between ros2_control and Isaac Sim.
- `pixi.toml` and `pixi.lock`: Environment definition and locked dependencies,
  including Python 3.11, Isaac Sim 5.1, and PyTorch.

## Running

The setup uses Linux, Pixi, a GPU supported by Isaac Sim 5.1, and a system ROS 2
Jazzy installation at `/opt/ros/jazzy` with colcon and the ROS dependencies
declared in the bringup package's `package.xml`.

Initialize the source dependencies and build the ROS workspace:

```bash
git submodule update --init --recursive
pixi run ros-build
```

Start the default scene:

```bash
pixi run isaacsim-default-scene
```

Press **Play** in Isaac Sim to start simulation and ROS publishing. In another
terminal, start MoveIt and use RViz to plan and execute motions:

```bash
pixi run ros-moveit
```

Run tasks with `pixi run <task>`:

| Task | Description |
| --- | --- |
| `isaacsim` | Launch Isaac Sim with the project's extension directory available. |
| `isaacsim-default-scene` | Launch Isaac Sim and open the default Panda scene with cameras enabled. |
| `ros-build` | Build the bringup package and its source dependencies using system ROS 2 Jazzy. |
| `ros-moveit` | Build the ROS workspace, then launch MoveIt, RViz, robot and camera transforms, and ros2_control controllers. |

The MoveIt launch accepts `rviz:=false` and `cameras:=false`, for example:

```bash
pixi run ros-moveit rviz:=false cameras:=false
```

`cameras:=false` disables camera transform publishers; camera rendering is
controlled separately by the scene's `cameras` variant.
