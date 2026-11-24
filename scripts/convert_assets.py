#!/usr/bin/env python3
"""Convert downloaded GSO OBJ assets to physics-enabled USD using Isaac Sim."""
import argparse
import asyncio
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'isaac_ws/assets/objects/gso/raw'
OUTPUT = ROOT / 'isaac_ws/assets/objects/gso/converted'


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sizing = p.add_mutually_exclusive_group(required=True)
    sizing.add_argument('--size-cm', type=float, help='Target longest bounding-box dimension in centimeters')
    sizing.add_argument('--max-diagonal-cm', type=float, help='Maximum bounding-box diagonal in centimeters')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing converted assets')
    return p.parse_args()


def discover(root: Path) -> list[Path]:
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() == '.obj')
    return [p for p in paths if not any(t in p.stem.lower() for t in ('collision', 'collider'))]


def convert_asset(src: Path, directory: Path, size_cm: float | None, *, max_diagonal_cm: float | None = None):
    import omni.kit.asset_converter as converter
    from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

    directory.mkdir(parents=True, exist_ok=True)
    context = converter.AssetConverterContext()
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.use_meter_as_world_unit = True
    context.convert_stage_up_z = True
    context.use_double_precision_to_usd_transform_op = True
    # GSO MTL files reference textures as siblings of the OBJ.
    with tempfile.TemporaryDirectory(prefix='gso_convert_') as temp:
        staged = Path(temp) / 'meshes'
        shutil.copytree(src.parent, staged)
        textures = src.parent.parent / 'materials' / 'textures'
        if textures.is_dir():
            shutil.copytree(textures, staged, dirs_exist_ok=True)
        task = converter.get_instance().create_converter_task(
            str(staged / src.name), str(directory / 'mesh.usd'), None, context
        )
        if not asyncio.get_event_loop().run_until_complete(task.wait_until_finished()):
            raise RuntimeError(f'Asset conversion failed: {task.get_error_message()}')

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/Object')
    stage.SetDefaultPrim(root.GetPrim())
    geometry = UsdGeom.Xform.Define(stage, '/Object/geometry')
    geometry.GetPrim().GetReferences().AddReference(str(directory / 'mesh.usd'))
    meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise ValueError(f'Converted asset contains no meshes: {src}')
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    extent = bounds.ComputeWorldBound(root.GetPrim()).ComputeAlignedRange().GetSize()
    longest = max(extent)
    if not all(math.isfinite(value) for value in extent) or longest <= 0:
        raise ValueError(f'Cannot determine converted mesh bounds: {src}')
    if max_diagonal_cm is None:
        factor = (size_cm / 100.0) / longest
        target = {'target_longest_cm': size_cm}
    else:
        factor = min(1.0, (max_diagonal_cm / 100.0) / math.hypot(*extent))
        target = {'max_diagonal_cm': max_diagonal_cm}
    root.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(factor))
    bounds.Clear()
    dimensions = list(bounds.ComputeWorldBound(root.GetPrim()).ComputeAlignedRange().GetSize())
    diagonal = math.hypot(*dimensions)
    if max_diagonal_cm is not None and diagonal > max_diagonal_cm / 100.0 + 1e-9:
        raise ValueError(f'Converted asset exceeds the {max_diagonal_cm:g} cm diagonal limit: {src}')
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(True)
    for prim in meshes:
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('convexDecomposition')
        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
    output_layer = Sdf.Layer.CreateAnonymous()
    output_layer.TransferContent(stage.GetRootLayer())
    output_layer.UpdateExternalReference(str(directory / 'mesh.usd'), './mesh.usd')
    if not output_layer.Export(str(directory / f'{directory.name}.usd')):
        raise RuntimeError(f'Could not save USD: {directory}')
    return {**target, 'scale_factor': factor, 'dimensions_m': dimensions, 'diagonal_m': diagonal}


def main():
    args = arguments()
    if not INPUT.is_dir():
        raise SystemExit(f'Input directory not found: {INPUT}')
    option, value = ('--size-cm', args.size_cm) if args.max_diagonal_cm is None else (
        '--max-diagonal-cm', args.max_diagonal_cm)
    if not math.isfinite(value) or value <= 0:
        raise SystemExit(f'{option} must be finite and positive')
    sources = discover(INPUT)
    if not sources:
        raise SystemExit(f'No OBJ files under {INPUT}')

    object_names = [src.relative_to(INPUT).parts[0] for src in sources]
    if len(set(object_names)) != len(object_names):
        raise SystemExit('Expected one visual OBJ per GSO object; found duplicate object names')

    # Start SimulationApp before importing Omniverse conversion and USD APIs.
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension('omni.kit.asset_converter')
        OUTPUT.mkdir(parents=True, exist_ok=True)
        manifest = OUTPUT / 'conversion_manifest.json'
        previous = json.loads(manifest.read_text(encoding='utf-8')) if manifest.exists() else []
        previous_by_usd = {entry['usd']: entry for entry in previous if 'usd' in entry}
        report = []
        print(f'Found {len(sources)} source OBJ files')
        for i, src in enumerate(sources, start=1):
            object_name = src.relative_to(INPUT).parts[0]
            directory = OUTPUT / object_name
            final_usd = directory / f'{directory.name}.usd'
            usd = str(final_usd.relative_to(OUTPUT))
            entry = previous_by_usd.get(usd, {})
            is_current = final_usd.exists() and (
                args.max_diagonal_cm is None or (
                    entry.get('max_diagonal_cm') == args.max_diagonal_cm
                    and 'diagonal_m' in entry
                    and final_usd.stat().st_mtime_ns <= manifest.stat().st_mtime_ns
                )
            )
            if is_current and not args.overwrite:
                print(f'[{i}/{len(sources)}] SKIP {final_usd}')
                report.append({**entry, 'source': str(src.relative_to(INPUT)), 'usd': usd, 'cached': True})
                continue
            print(f'[{i}/{len(sources)}] {src.relative_to(INPUT)}', flush=True)
            try:
                with tempfile.TemporaryDirectory(prefix='.convert_', dir=OUTPUT) as temp:
                    staging = Path(temp) / object_name
                    sizing = convert_asset(src, staging, args.size_cm, max_diagonal_cm=args.max_diagonal_cm)
                    backup = Path(temp) / 'previous'
                    if directory.exists():
                        directory.rename(backup)
                    try:
                        staging.rename(directory)
                    except OSError:
                        if backup.exists():
                            backup.rename(directory)
                        raise
                report.append({'source': str(src.relative_to(INPUT)), 'usd': usd, **sizing})
                print(f"  -> {usd} (factor {sizing['scale_factor']:.6g})")
            except Exception as e:
                print(f'  ERROR: {e}', file=sys.stderr)
                report.append({'source': str(src.relative_to(INPUT)), 'error': str(e)})
        manifest.write_text(json.dumps(report, indent=2), encoding='utf-8')
        success = sum('usd' in item for item in report)
        failures = sum('error' in item for item in report)
        print(f'Finished: {success} converted, {failures} failed; manifest: {manifest}')
        if failures:
            raise SystemExit(1)
    finally:
        import omni.kit.app

        omni.kit.app.get_app().post_quit(1 if sys.exc_info()[0] is not None else 0)
        app.close()


if __name__ == '__main__':
    main()
