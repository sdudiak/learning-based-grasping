import os
import sys
from pathlib import Path


def main(pipeline=False):
    root = Path(__file__).resolve().parents[1]
    os.chdir(root / 'ros_ws')
    build = (
        'PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin '
        'colcon build --symlink-install --packages-up-to isaac_manipulation_segmentation '
        '--cmake-args -DBUILD_TESTING=OFF -DPython3_EXECUTABLE=/usr/bin/python3')
    setup = f'''
unset PYTHONPATH OLD_PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
{build}
source install/setup.bash
exec "$@"
'''
    launch_file = 'pipeline.launch.py' if pipeline else 'segmentation.launch.py'
    command = [sys.executable, '/opt/ros/jazzy/bin/ros2', 'launch',
               'isaac_manipulation_segmentation', launch_file,
               f'checkpoint:={root / "ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt"}',
               *sys.argv[1:]]
    os.execv('/bin/bash', ['bash', '--noprofile', '--norc', '-ec', setup, '--', *command])


if __name__ == '__main__':
    main()
