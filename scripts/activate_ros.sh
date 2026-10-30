unset PYTHONPATH OLD_PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
if [[ -f "${PIXI_PROJECT_ROOT}/ros_ws/install/local_setup.bash" ]]; then
    source "${PIXI_PROJECT_ROOT}/ros_ws/install/local_setup.bash"
fi
