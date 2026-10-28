import argparse
import ctypes
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main():
    parser = argparse.ArgumentParser(description='Generate settled GSO piles, RGB images and visible instance ground truth headlessly.')
    parser.add_argument('--save-masked-images', action='store_true', help='Save color-coded masked RGB previews (default: disabled)')
    parser.add_argument('--objects', type=int, help='Use a fixed count instead of --min-objects/--max-objects')
    parser.add_argument('--min-objects', type=int, help='Minimum objects per pile, inclusive (default: 5)')
    parser.add_argument('--max-objects', type=int, help='Maximum objects per pile, inclusive (default: 15)')
    parser.add_argument('--spread-cm', type=float, default=3.0, help='Maximum horizontal drop offset per axis in cm; spread is sampled per pile (default: 3)')
    parser.add_argument('--piles', type=int, default=1)
    parser.add_argument('--views-per-pile', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resolution', type=int, nargs=2, default=[640, 480], metavar=('WIDTH', 'HEIGHT'))
    parser.add_argument('--settle-timeout', type=float, default=30)
    parser.add_argument('--name', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'), help='Dataset folder name under ml_ws/datasets/segmentation')
    args = parser.parse_args()
    if not args.name or args.name in ('.', '..') or '/' in args.name or '\\' in args.name:
        parser.error('--name must be a single folder name')
    if args.objects is not None:
        if args.min_objects is not None or args.max_objects is not None:
            parser.error('--objects cannot be combined with --min-objects/--max-objects')
        args.min_objects = args.max_objects = args.objects
    else:
        args.min_objects = args.min_objects if args.min_objects is not None else 5
        args.max_objects = args.max_objects if args.max_objects is not None else 15
    if not 1 <= args.min_objects <= args.max_objects:
        parser.error('Object counts must satisfy 1 <= min-objects <= max-objects')
    if not math.isfinite(args.spread_cm) or args.spread_cm < 0:
        parser.error('--spread-cm must be finite and nonnegative')
    if args.seed < 0 or min(args.min_objects, args.piles, args.views_per_pile, *args.resolution) < 1 or not math.isfinite(args.settle_timeout) or args.settle_timeout < 1:
        parser.error('Seed must be nonnegative, counts/resolution positive, and settling timeout finite and >= 1 second')
    assets_root = ROOT / 'isaac_ws/assets/objects/gso/converted'
    assets = sorted(p / f'{p.name}.usd' for p in assets_root.iterdir() if p.is_dir() and (p / f'{p.name}.usd').is_file()) if assets_root.is_dir() else []
    if not assets:
        parser.error(f'No converted GSO assets in {assets_root}')
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        count = ctypes.c_int()
        if cuda.cuInit(0) != 0 or cuda.cuDeviceGetCount(ctypes.byref(count)) != 0 or count.value < 1:
            raise RuntimeError('No usable CUDA device')
    except (OSError, RuntimeError) as error:
        parser.error(f'Isaac Sim image capture requires a working NVIDIA driver and GPU: {error}')
    output = ROOT / 'ml_ws/datasets/segmentation' / args.name
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error(f'Output directory must be empty: {output}')
    for name in ('images', 'ground_truth'):
        (output / name).mkdir()
    manifest = {'schema_version': 2, 'status': 'running', 'evaluation': 'class_agnostic_visible_instances',
                'annotation_format': 'binary instance masks and uint32 instance ID maps; zero is background', 'settings': {**vars(args), 'output': str(output)},
                'simulation': {'physics_dt': 1 / 120, 'stabilization': True, 'drop_clearance_m': 0.015,
                               'spawn_xy_half_width_max_m': args.spread_cm / 100, 'steps_between_drops': 30, 'linear_velocity_threshold_m_s': 0.01,
                               'angular_velocity_threshold_rad_s': 0.15, 'stability_window_steps': 60,
                               'stability_rule': 'velocity window or pose drift below 2 mm and rotation below 1 degree',
                               'position_drift_threshold_m': 0.002, 'rotation_drift_threshold_deg': 1, 'minimum_steps': 120,
                               'solver_position_iterations': 16, 'solver_velocity_iterations': 4,
                               'sleep_threshold': 0.001, 'contact_offset_m': 0.002, 'rest_offset_m': 0,
                               'ground_static_friction': 0.7, 'ground_dynamic_friction': 0.6, 'ground_restitution': 0},
                'assets': [str(p) for p in assets], 'piles': [], 'frames': []}
    def save():
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    app = None
    capture = None
    failed = False
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({'headless': True, 'renderer': 'RayTracedLighting'})
        import numpy as np
        sys.path.insert(0, str(ROOT))
        sys.path.insert(0, str(ROOT / 'isaac_ws/extensions/isaac_manipulation.object_spawner'))
        from isaac_ws.dataset_generation.segmentation.scene import Scene
        from isaac_ws.dataset_generation.segmentation.capture import Capture
        scene = Scene(np.random.default_rng(args.seed))
        capture = Capture(tuple(args.resolution), save_masked_images=args.save_masked_images)
        for pile in range(args.piles):
            count = int(scene.rng.integers(args.min_objects, args.max_objects + 1))
            spread = float(scene.rng.uniform(0, args.spread_cm / 100))
            objects = scene.spawn(assets, count, spread=spread)
            settling = scene.settle(args.settle_timeout)
            manifest['piles'].append({'id': pile, 'object_count': count, 'spread_m': spread, 'objects': objects, **settling})
            save()
            if not settling['settled']:
                print(f'Pile {pile}: {settling["reason"]}; skipped', flush=True)
                continue
            for view in range(args.views_per_pile):
                metadata = scene.randomize_view(tuple(args.resolution))
                frame_id = f'pile_{pile:04d}_view_{view:04d}'
                frame = capture.save(output, frame_id, objects, {'pile': pile, 'view': view, **metadata})
                manifest['frames'].append(frame)
                save()
                print(f'Saved {frame_id}', flush=True)
        failed = len(manifest['frames']) != args.piles * args.views_per_pile
        manifest['status'] = 'incomplete' if failed else 'complete'
    except (Exception, KeyboardInterrupt) as error:
        failed = True
        manifest.update(status='failed', error=str(error))
        traceback.print_exc()
    finally:
        try:
            if capture is not None:
                capture.close()
        except (Exception, KeyboardInterrupt) as error:
            failed = True
            manifest.update(status='failed', error=str(error))
            traceback.print_exc()
        try:
            save()
        finally:
            if app is not None:
                import omni.kit.app

                omni.kit.app.get_app().post_quit(1 if failed or sys.exc_info()[0] is not None else 0)
                app.close(skip_cleanup=False)
    if failed:
        sys.exit(1)
    print(f'Dataset: {output}', flush=True)


if __name__ == '__main__':
    main()
