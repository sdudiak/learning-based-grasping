import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE_REVISION = '2dd8852e1be60f5f9d277fafcc621835cdf59110'
MODEL_REVISION = 'ec1ccbb5eec0680db669246ac312a3636f16ee43'
MODEL_REPO = 'adithyamurali/GraspGenModels'
VANILLA_REVISION = '44eb1d77d6d27bc97ee361e3afb717a2115e30a9'
VANILLA_SHA256 = 'bbf501ce93382bcc0502f1749cf35a798bf5ff25b8c836401852fca5afaf24c9'


def run(*command, cwd=ROOT):
    subprocess.run([str(arg) for arg in command], cwd=cwd, check=True)


def download_models(directory):
    directory.mkdir(parents=True, exist_ok=True)
    info = subprocess.check_output([
        'curl', '-fsSL', '--retry', '3',
        f'https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REVISION}'])
    files = [entry['rfilename'] for entry in json.loads(info)['siblings']]
    selected = [name for name in files if name.startswith('checkpoints/graspgen_franka_panda') or name == 'LICENSE']
    for name in selected:
        destination = directory / name
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + '.partial')
        run('curl', '-fsSL', '--retry', '3', '--continue-at', '-',
            f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}',
            '-o', partial)
        partial.replace(destination)
    (directory / 'revision.json').write_text(json.dumps({
        'repository': MODEL_REPO, 'revision': MODEL_REVISION, 'files': selected}, indent=2) + '\n')


def install(source, uv):
    if not source.exists():
        run('git', 'clone', '--no-checkout', 'https://github.com/NVlabs/GraspGen.git', source)
        run('git', 'checkout', '--detach', SOURCE_REVISION, cwd=source)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError(f'{source} is at {revision}; expected {SOURCE_REVISION}')
    cache = ROOT / '.cache/graspgen/uv'
    python = source / '.venv/bin/python'
    if not python.exists():
        run(uv, 'venv', '--cache-dir', cache, '--python', '3.10', source / '.venv')
    pip = [uv, '--no-config', 'pip', '--cache-dir', cache, 'install', '--python', python,
           '--default-index', 'https://pypi.org/simple']
    run(*pip, '-r', Path(__file__).with_name('requirements.lock'),
        '--index-strategy', 'first-index')
    install_compatibility(source)
    site_packages = subprocess.check_output([
        str(python), '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip()
    (Path(site_packages) / 'graspgen_source.pth').write_text(str(source) + '\n')
    frozen = subprocess.check_output([
        str(uv), 'pip', 'freeze', '--python', str(python)], text=True)
    (source / 'installed-requirements.txt').write_text(frozen)


def install_compatibility(source):
    patch = Path(__file__).with_name('blackwell.patch')
    applied = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)],
                             cwd=source, capture_output=True).returncode == 0
    if not applied:
        run('git', 'apply', '--check', patch, cwd=source)
        run('git', 'apply', patch, cwd=source)
    destination = source / 'grasp_gen/models/ptv3/ptv3_vanilla.py'
    run('curl', '-fsSL', '--retry', '3',
        f'https://raw.githubusercontent.com/NVlabs/GraspGenX/{VANILLA_REVISION}/graspgenx/models/ptv3/ptv3_vanilla.py',
        '-o', destination)
    if hashlib.sha256(destination.read_bytes()).hexdigest() != VANILLA_SHA256:
        raise RuntimeError('NVIDIA PTv3 source checksum mismatch')
    shutil.copyfile(Path(__file__).with_name('ptv3_compat.py'), destination.with_name('compat.py'))


def main():
    parser = argparse.ArgumentParser(description='Set up isolated pretrained Panda inference.')
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'third_party/GraspGen')
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'models/graspgen')
    parser.add_argument('--uv', default=shutil.which('uv') or 'uv')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--download-only', action='store_true')
    mode.add_argument('--install-only', action='store_true')
    args = parser.parse_args()
    if not args.download_only:
        install(args.source_dir.resolve(), args.uv)
    if not args.install_only:
        download_models(args.models_dir.resolve())


if __name__ == '__main__':
    main()
