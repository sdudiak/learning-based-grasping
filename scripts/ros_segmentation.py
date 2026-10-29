import os
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root / 'ros_ws')
    setup = '''
unset PYTHONPATH OLD_PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select isaac_manipulation_segmentation
source install/setup.bash
exec "$@"
'''
    command = [sys.executable, '/opt/ros/jazzy/bin/ros2', 'launch',
               'isaac_manipulation_segmentation', 'segmentation.launch.py',
               f'checkpoint:={root / "ml_ws/outputs/segmentation/maskrcnn-v1/checkpoints/best.pt"}',
               *sys.argv[1:]]
    os.execv('/bin/bash', ['bash', '--noprofile', '--norc', '-ec', setup, '--', *command])


if __name__ == '__main__':
    main()
