unset PYTHONPATH OLD_PYTHONPATH
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$@"
