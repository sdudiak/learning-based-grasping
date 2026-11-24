import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(add_help=False,
                                     description='Run GraspGen outside the Isaac/ROS Python environment.')
    parser.add_argument('command', choices=['infer', 'pipeline_test_cli'])
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'third_party/GraspGen')
    args, extra = parser.parse_known_args()
    source = args.source_dir.resolve()
    env = os.environ.copy()
    for key in ('PYTHONPATH', 'OLD_PYTHONPATH', 'AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH',
                'COLCON_PREFIX_PATH', 'LD_LIBRARY_PATH', 'VIRTUAL_ENV'):
        env.pop(key, None)
    if args.command == 'pipeline_test_cli':
        command = ['/usr/bin/python3', str(Path(__file__).with_name(args.command + '.py')), *extra]
        setup = 'source /opt/ros/jazzy/setup.bash\nexec "$@"'
        os.execve('/bin/bash', ['bash', '--noprofile', '--norc', '-ec', setup, '--', *command], env)
    python = source / '.venv/bin/python'
    if not python.exists():
        parser.error('GraspGen environment is missing; run pixi run graspgen-setup')
    script = Path(__file__).with_name('infer.py')
    os.execve(python, [str(python), str(script), *extra], env)


if __name__ == '__main__':
    main()
