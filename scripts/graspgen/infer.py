import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import random
import subprocess
import time

import numpy as np

from data import load_input, top_down_mask, validate_predictions

ROOT = Path(__file__).resolve().parents[2]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(config):
    import yaml
    settings = yaml.safe_load(config.read_text())
    paths = [config, config.parent / settings['eval']['checkpoint'],
             config.parent / settings['discriminator']['checkpoint']]
    return [{'file': path.name, 'sha256': file_sha256(path)} for path in paths]


def initialize(config):
    import torch
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable to the GraspGen process')
    torch.cuda.synchronize()
    start = time.perf_counter()
    sampler = GraspGenSampler(load_grasp_cfg(str(config)))
    torch.cuda.synchronize()
    return sampler, time.perf_counter() - start


def predict(points, sampler, planner, args):
    import torch
    from grasp_gen.samplers import run_planner_on_object
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        grasps, scores, branches, _ = run_planner_on_object(
            points.copy(), sampler, planner=planner, num_grasps=args.num_grasps,
            grasp_threshold=args.threshold, topk_num_grasps=-1 if args.top_down_bonus else args.topk,
            moe_num_yaws=args.moe_num_yaws, moe_z_offsets_cm=args.moe_z_offsets_cm,
            moe_obb_density=args.moe_obb_density)
    torch.cuda.synchronize()
    metrics = {'inference_seconds': time.perf_counter() - start,
               'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_gpu_reserved_bytes': torch.cuda.max_memory_reserved()}
    grasps, scores = validate_predictions(grasps, scores)
    top_down = top_down_mask(grasps, args.top_down_angle)
    ranking_scores = scores + args.top_down_bonus * top_down
    order = np.argsort(-ranking_scores, kind='stable')[:args.topk]
    metrics['returned_before_topk'] = len(scores)
    metrics['top_down_before_shortlist'] = int(top_down.sum())
    metrics['top_down_shortlisted'] = int(top_down[order].sum())
    return grasps[order], scores[order], np.asarray(branches)[order], ranking_scores[order], metrics


def save_result(directory, points, metadata, prediction):
    grasps, scores, branches, ranking_scores, metrics = prediction
    directory.mkdir(parents=True, exist_ok=False)
    metadata.update(metrics, returned_grasps=len(scores),
                    confidence={'min': float(scores.min()), 'max': float(scores.max()),
                                'mean': float(scores.mean()),
                                'quantiles': np.quantile(scores, [0, .25, .5, .75, 1]).tolist()})
    np.savez_compressed(directory / 'predictions.npz', points=points, grasps=grasps,
                        scores=scores, branches=branches, ranking_scores=ranking_scores)
    (directory / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{directory}: {len(scores)} grasps, {metrics["inference_seconds"]:.3f}s', flush=True)


def add_sampling_arguments(parser):
    parser.add_argument('--num-grasps', type=int, default=200)
    parser.add_argument('--top-down-bonus', type=float, default=0.0,
                        help='Ranking bonus for near-vertical downward approaches')
    parser.add_argument('--top-down-angle', type=float, default=30.0,
                        help='Maximum deviation from straight down in degrees')
    parser.add_argument('--moe-num-yaws', type=int, default=36)
    parser.add_argument('--moe-z-offsets-cm', type=float, nargs='+', default=[-8, -6, -4, -2, 0])
    parser.add_argument('--moe-obb-density', choices=['sparse', 'dense', 'dense-topandside'], default='sparse')


def validate_sampling_arguments(parser, args):
    if not np.isfinite([args.top_down_bonus, *args.moe_z_offsets_cm]).all() or args.top_down_bonus < 0:
        parser.error('Ranking bonus and OBB offsets must be finite, with a nonnegative bonus')
    if not 0 < args.top_down_angle < 90 or min(args.num_grasps, args.moe_num_yaws) < 1:
        parser.error('Top-down angle must be in (0, 90) and sample counts must be positive')


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate pretrained Panda grasps on offline inputs.')
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'models/graspgen/checkpoints/graspgen_franka_panda.yml')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--planner', choices=['diffusion', 'graspmoe', 'both'], default='both')
    add_sampling_arguments(parser)
    parser.add_argument('--num-points', type=int, default=2048)
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--threshold', type=float, default=-1.0)
    parser.add_argument('--mesh-scale', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    validate_sampling_arguments(parser, args)
    if min(args.topk, args.repeats) < 1 or args.mesh_scale <= 0:
        parser.error('Counts and mesh scale must be positive')
    if args.num_points < 32:
        parser.error('--num-points must be at least 32')
    if not np.isfinite(args.mesh_scale) or not (-1 <= args.threshold <= 1):
        parser.error('Invalid scale or confidence threshold')
    return args


def main():
    args = parse_args()
    import torch
    import grasp_gen
    source = Path(grasp_gen.__file__).resolve().parents[1]
    identities = checkpoint_identity(args.config)
    sampler, initialization_seconds = initialize(args.config)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       cwd=source, text=True).strip()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    planners = ['diffusion', 'graspmoe'] if args.planner == 'both' else [args.planner]
    base = {'checkpoints': identities, 'source_revision': revision,
            'ptv3_backend': 'NVIDIA GraspGenX pure PyTorch',
            'compatibility_patch_sha256': file_sha256(Path(__file__).with_name('blackwell.patch')),
            'compatibility_adapter_sha256': file_sha256(Path(__file__).with_name('ptv3_compat.py')),
            'ptv3_source_sha256': file_sha256(source / 'grasp_gen/models/ptv3/ptv3_vanilla.py'),
            'initialization_seconds': initialization_seconds,
            'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'seed': args.seed,
            'num_diffusion_samples': args.num_grasps, 'topk': args.topk,
            'threshold': args.threshold, 'gripper': 'franka_panda',
            'top_down_bonus': args.top_down_bonus, 'top_down_angle_deg': args.top_down_angle,
            'moe_num_yaws': args.moe_num_yaws, 'moe_z_offsets_cm': args.moe_z_offsets_cm,
            'moe_obb_density': args.moe_obb_density,
            'pose_convention': 'T_input_graspgen; +Z approach, X finger closing; meters'}
    for path in args.inputs:
        evaluate_input(path, sampler, planners, args, base, run_id)


def evaluate_input(path, sampler, planners, args, base, run_id):
    points, input_metadata = load_input(path, scale=args.mesh_scale,
                                       num_points=args.num_points, seed=args.seed)
    for planner, repeat in product(planners, range(args.repeats)):
        metadata = dict(base, input=input_metadata, planner=planner, repeat=repeat,
                        timing_scope='model call, including filtering and CPU/GPU transfers')
        prediction = predict(points, sampler, planner, args)
        directory = args.output_dir / path.stem / run_id / f'{planner}-{repeat:02d}'
        save_result(directory, points, metadata, prediction)


if __name__ == '__main__':
    main()
