import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[4]
EXTENSION = ROOT / 'isaac_ws/extensions/isaac_manipulation.object_spawner'


def run():
    from isaacsim import SimulationApp

    app = SimulationApp({'headless': True, 'extra_args': ['--ext-folder', str(EXTENSION.parent)]})
    try:
        import carb.settings
        import numpy as np
        import omni.kit.app
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdGeom, UsdPhysics
        from isaacsim.core.prims import SingleRigidPrim

        sys.path.insert(0, str(EXTENSION))
        sys.path.insert(0, str(ROOT / 'isaac_ws/dataset_generation/segmentation'))
        from isaac_manipulation.object_spawner.spawning import ROOT_PATH, SettlingTracker, clear_objects
        from isaac_manipulation.object_spawner.extension import ObjectSpawnerExtension

        settings = carb.settings.get_settings()
        prefix = '/exts/isaac_manipulation.object_spawner'
        manager = omni.kit.app.get_app().get_extension_manager()
        temporary = tempfile.TemporaryDirectory(prefix='object-spawner-test-')
        assets = Path(temporary.name)
        for name in ('a', 'b'):
            directory = assets / name
            directory.mkdir()
            stage = Usd.Stage.CreateNew(str(directory / f'{name}.usd'))
            root = UsdGeom.Xform.Define(stage, '/Object')
            cube = UsdGeom.Cube.Define(stage, '/Object/Cube')
            cube.CreateSizeAttr(0.05)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
            UsdPhysics.MassAPI.Apply(root.GetPrim()).CreateMassAttr(0.1)
            stage.SetDefaultPrim(root.GetPrim())
            stage.GetRootLayer().Save()

        def updates_until(predicate, timeout=40):
            end = time.monotonic() + timeout
            while not predicate() and time.monotonic() < end:
                app.update()
            if not predicate():
                raise AssertionError(f'Timed out; status={settings.get(prefix + "/status")}, error={settings.get(prefix + "/last_error")}')

        class SpawnerTests(unittest.TestCase):
            def test_extension_and_dataset(self):
                context = omni.usd.get_context()
                context.new_stage()
                stage = context.get_stage()
                UsdGeom.SetStageMetersPerUnit(stage, 1.0)
                UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
                UsdPhysics.Scene.Define(stage, '/World/PhysicsScene')
                table = UsdGeom.Cube.Define(stage, '/World/Table')
                table.CreateSizeAttr(1.0)
                table.AddTranslateOp().Set((0.5, 0.0, 0.45))
                table.AddScaleOp().Set((0.8, 0.8, 0.1))
                UsdPhysics.CollisionAPI.Apply(table.GetPrim())
                UsdGeom.Xform.Define(stage, '/World/KeepMe')
                settings.set(prefix + '/assets_directory', str(assets))
                settings.set(prefix + '/target_surface', '/World/Table')
                settings.set(prefix + '/count', 2)
                settings.set(prefix + '/auto_spawn', True)
                settings.set(prefix + '/show_ui', True)
                instances = []
                original_startup = ObjectSpawnerExtension.on_startup

                def startup(instance, extension_id):
                    original_startup(instance, extension_id)
                    instances.append(instance)

                with patch.object(ObjectSpawnerExtension, 'on_startup', startup):
                    manager.set_extension_enabled_immediate('isaac_manipulation.object_spawner', True)
                extension, = instances
                updates_until(lambda: settings.get(prefix + '/status') in ('ready', 'failed'))
                self.assertEqual(settings.get(prefix + '/status'), 'ready', settings.get(prefix + '/last_error'))
                self.assertIsNone(extension._window)
                self.assertEqual(len(extension.objects), 2)
                self.assertTrue(omni.timeline.get_timeline_interface().is_playing())
                self.assertTrue(stage.GetPrimAtPath('/World/KeepMe'))
                positions = np.array([body.get_world_pose()[0] for body in extension.bodies])
                self.assertTrue(np.all(positions[:, 2] > 0.49))
                self.assertTrue(np.all(np.abs(positions[:, 0] - 0.5) < 0.4))
                marker = UsdGeom.Cube.Define(stage, '/World/KeepRigid')
                marker.CreateSizeAttr(0.05)
                UsdPhysics.CollisionAPI.Apply(marker.GetPrim())
                UsdPhysics.RigidBodyAPI.Apply(marker.GetPrim()).CreateKinematicEnabledAttr(True)
                persistent_body = SingleRigidPrim('/World/KeepRigid', position=np.array([-1., 0., 1.]))
                app.update()
                persistent_body.initialize()
                settings.set(prefix + '/count', 1)
                extension.spawn()
                updates_until(lambda: extension._task.done())
                self.assertEqual(settings.get(prefix + '/status'), 'ready', settings.get(prefix + '/last_error'))
                self.assertEqual(len(extension.objects), 1)
                self.assertTrue(np.isfinite(persistent_body.get_linear_velocity()).all())
                np.testing.assert_allclose(persistent_body.get_world_pose()[0], [-1., 0., 1.])
                extension.spawn()
                updates_until(lambda: settings.get(prefix + '/status') == 'settling' or extension._task.done())
                extension.clear()
                updates_until(lambda: extension._task.done())
                self.assertEqual(settings.get(prefix + '/status'), 'idle')
                self.assertFalse(stage.GetPrimAtPath(ROOT_PATH))
                self.assertTrue(np.isfinite(persistent_body.get_linear_velocity()).all())
                table.GetPrim().GetAttribute('xformOp:scale').Set((0.01, 0.01, 0.1))
                extension.spawn()
                updates_until(lambda: extension._task.done())
                self.assertEqual(settings.get(prefix + '/status'), 'failed')
                self.assertFalse(stage.GetPrimAtPath(ROOT_PATH))
                app.update()
                self.assertTrue(omni.timeline.get_timeline_interface().is_playing())
                table.GetPrim().GetAttribute('xformOp:scale').Set((0.8, 0.8, 0.1))
                gso = ROOT / 'isaac_ws/assets/objects/gso/converted'
                if gso.is_dir():
                    settings.set(prefix + '/assets_directory', str(gso))
                    settings.set(prefix + '/count', 2)
                    extension.spawn()
                    updates_until(lambda: extension._task.done())
                    self.assertEqual(settings.get(prefix + '/status'), 'ready', settings.get(prefix + '/last_error'))
                    self.assertEqual(len(extension.objects), 2)
                extension._build_ui()
                self.assertIsNotNone(extension._window)
                extension.clear()
                updates_until(lambda: extension._task.done())
                self.assertFalse(stage.GetPrimAtPath(ROOT_PATH))
                self.assertTrue(stage.GetPrimAtPath('/World/KeepMe'))
                unowned = UsdGeom.Xform.Define(stage, ROOT_PATH)
                with self.assertRaisesRegex(ValueError, 'not owned'):
                    clear_objects(stage)
                stage.RemovePrim(unowned.GetPath())
                settings.set(prefix + '/target_surface', '/World/Missing')
                extension.spawn()
                updates_until(lambda: extension._task.done())
                self.assertEqual(settings.get(prefix + '/status'), 'failed')
                self.assertFalse(stage.GetPrimAtPath(ROOT_PATH))
                manager.set_extension_enabled_immediate('isaac_manipulation.object_spawner', False)
                omni.timeline.get_timeline_interface().stop()
                app.update()
                context.new_stage()
                from scene import Scene
                scene = Scene(np.random.default_rng(42))
                objects = scene.spawn(sorted(assets.glob('*/*.usd')), 2)
                self.assertEqual(len(objects), 2)
                self.assertTrue(scene.settle(10)['settled'])
                self.assertIn('position', objects[0])

            def test_nonfinite_velocity_is_not_reported_as_settled(self):
                body = SimpleNamespace(get_world_pose=lambda: (np.zeros(3), np.array([1., 0., 0., 0.])),
                                       get_linear_velocity=lambda: np.array([np.nan, 0., 0.]),
                                       get_angular_velocity=lambda: np.zeros(3))
                tracker = SettlingTracker(1 / 60)
                self.assertEqual(tracker.sample([body]), {'settled': False, 'reason': 'invalid object velocity'})

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SpawnerTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        app._app.post_quit(0 if result.wasSuccessful() else 1)
        temporary.cleanup()
        return result.wasSuccessful()
    finally:
        app.close()


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
