import asyncio
import time
from pathlib import Path

import carb
import carb.settings
import numpy as np
import omni.ext
import omni.kit.app
import omni.physx
import omni.physics.tensors
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

from .spawning import ROOT_PATH, OWNER, SettlingTracker, SpawnBody, bounds, clear_objects, drop_object, find_assets, owned_root, prepare_object


SETTINGS = '/exts/isaac_manipulation.object_spawner'


class ObjectSpawnerExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._settings = carb.settings.get_settings()
        self._context = omni.usd.get_context()
        self._timeline = omni.timeline.get_timeline_interface()
        self._app = omni.kit.app.get_app()
        self._task = None
        self._window = None
        self._status_label = None
        self._physics_subscription = None
        self._tracker = None
        self._simulation_view = None
        self.bodies = []
        self.objects = []
        self._auto_stage = None
        extension_path = Path(self._app.get_extension_manager().get_extension_path(ext_id))
        self._default_assets = extension_path.parents[1] / 'assets/objects/gso/converted'
        self._stage_subscription = self._context.get_stage_event_stream().create_subscription_to_pop(self._on_stage_event)
        self._set_status('idle')
        if self._settings.get(SETTINGS + '/show_ui') and self._settings.get('/app/window/enabled'):
            self._build_ui()
        self._auto_spawn()

    def _set_status(self, status, detail=''):
        self._settings.set(SETTINGS + '/status', status)
        self._settings.set(SETTINGS + '/last_error', detail if status == 'failed' else '')
        if self._status_label is not None:
            self._status_label.text = f'{status}: {detail}' if detail else status
        carb.log_info(f'Object spawner: {status} {detail}')

    def _on_stage_event(self, event):
        if event.type == int(omni.usd.StageEventType.CLOSING):
            self._cancel()
            self._release_bodies()
            self.objects = []
            self._auto_stage = None
            self._set_status('idle')
        elif event.type == int(omni.usd.StageEventType.OPENED):
            self._auto_spawn(require_target=False)

    def _auto_spawn(self, require_target=True):
        if not self._settings.get(SETTINGS + '/auto_spawn'):
            return
        stage = self._context.get_stage()
        if stage is None or self._auto_stage == self._context.get_stage_id():
            return
        if require_target and not stage.GetPrimAtPath(self._settings.get(SETTINGS + '/target_surface')):
            return
        self._auto_stage = self._context.get_stage_id()
        self.spawn()

    def spawn(self):
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self._spawn())

    def _cancel(self):
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._tracker = None
        self._physics_subscription = None
        self._task = None

    def clear(self):
        previous = self._task
        stage_id = self._context.get_stage_id()
        self._cancel()
        self._task = asyncio.ensure_future(self._clear(previous, stage_id))

    async def _clear(self, previous, stage_id):
        if previous is not None and not previous.done():
            try:
                await previous
            except asyncio.CancelledError:
                pass
        if self._context.get_stage_id() != stage_id:
            return
        stage = self._context.get_stage()
        if stage is None:
            return
        try:
            was_playing = self._timeline.is_playing()
            self._timeline.pause()
            await self._app.next_update_async()
            if self._context.get_stage_id() != stage_id:
                return
            self._remove_objects(stage)
            self._set_status('idle')
        except Exception as error:
            self._set_status('failed', str(error))
            carb.log_error(str(error))
        finally:
            if was_playing and self._context.get_stage_id() == stage_id:
                self._timeline.play()
                await self._app.next_update_async()

    def _release_bodies(self):
        # Release tensor views before removing their physics actors.
        if self._simulation_view is not None:
            self._simulation_view.invalidate()
            self._simulation_view = None
        for body in self.bodies:
            body.destroy()
        self.bodies = []

    def _remove_objects(self, stage):
        owned_root(stage)
        self._release_bodies()
        clear_objects(stage)
        self.objects = []

    def _check_stage(self):
        if self._context.get_stage_id() != self._stage_id:
            raise RuntimeError('The scene changed during spawning')
        if not self._timeline.is_playing():
            raise RuntimeError('Simulation was paused or stopped during spawning')
        if time.monotonic() - self._last_physics_update > 30:
            raise RuntimeError('Physics stopped advancing during spawning')

    async def _advance(self, seconds):
        end = self._elapsed + seconds
        while self._elapsed < end:
            self._check_stage()
            await self._app.next_update_async()

    def _on_physics_step(self, dt):
        if not np.isfinite(dt) or dt <= 0:
            return
        self._last_physics_update = time.monotonic()
        self._elapsed += dt
        self._physics_dt = dt
        if self._tracker is not None and self._result is None:
            try:
                result = self._tracker.sample(self.bodies)
                if result and (not result['settled'] or self._elapsed - self._settle_start >= 1.0):
                    self._result = result
            except Exception as error:
                self._result = {'settled': False, 'reason': str(error)}

    async def _spawn(self):
        stage = None
        stage_id = self._context.get_stage_id()
        created = False
        prepared_all = False
        completed = False
        was_playing = self._timeline.is_playing()
        keep_playing = was_playing or self._settings.get(SETTINGS + '/auto_spawn')
        try:
            self._set_status('waiting')
            await self._app.next_update_async()
            while not self._app.is_app_ready() or self._context.get_stage_loading_status()[2] > 0:
                if self._context.get_stage_id() != stage_id:
                    raise asyncio.CancelledError()
                await self._app.next_update_async()
            stage = self._context.get_stage()
            if stage is None:
                raise ValueError('Open a scene before spawning objects')
            self._stage_id = self._context.get_stage_id()
            if self._stage_id != stage_id:
                raise asyncio.CancelledError()
            settings = {name: self._settings.get(SETTINGS + '/' + name) for name in
                        ('assets_directory', 'target_surface', 'count', 'seed', 'spread', 'settle_timeout')}
            if type(settings['count']) is not int or type(settings['seed']) is not int or settings['count'] < 1 or settings['seed'] < 0:
                raise ValueError('Count must be positive and seed must be nonnegative')
            if not np.isfinite(settings['spread']) or settings['spread'] < 0:
                raise ValueError('Spread must be finite and nonnegative')
            if not np.isfinite(settings['settle_timeout']) or settings['settle_timeout'] < 1:
                raise ValueError('Settling timeout must be finite and at least one second')
            if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.0):
                raise ValueError('The scene must use meters and a Z up axis')
            surface = stage.GetPrimAtPath(settings['target_surface'])
            if not surface:
                raise ValueError(f"Target surface does not exist: {settings['target_surface']}")
            if str(surface.GetPath()) == ROOT_PATH or str(surface.GetPath()).startswith(ROOT_PATH + '/'):
                raise ValueError('Select a target surface outside ' + ROOT_PATH)
            if not any(prim.HasAPI(UsdPhysics.CollisionAPI) for prim in Usd.PrimRange(surface)):
                raise ValueError('The target surface must have collision geometry')
            surface_bounds = bounds(surface)
            assets = find_assets(settings['assets_directory'] or self._default_assets)
            self._timeline.pause()
            await self._app.next_update_async()
            if self._context.get_stage_id() != stage_id:
                raise asyncio.CancelledError()
            self._remove_objects(stage)
            root = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
            root.SetCustomDataByKey('objectSpawnerOwner', OWNER)
            created = True
            self._set_status('spawning')
            rng = np.random.default_rng(settings['seed'])
            selected = rng.choice(len(assets), size=settings['count'], replace=settings['count'] > len(assets))
            prepared = []
            for index, asset_index in enumerate(selected):
                body, pose, metadata = prepare_object(
                    stage, assets[int(asset_index)], f'{ROOT_PATH}/instance_{index:04d}',
                    index, rng, settings['spread'], surface_bounds, body_factory=SpawnBody)
                self.bodies.append(body)
                self.objects.append(metadata)
                prepared.append(pose)
            prepared_all = True
            await self._app.next_update_async()
            while self._context.get_stage_loading_status()[2] > 0:
                if self._context.get_stage_id() != stage_id:
                    raise asyncio.CancelledError()
                await self._app.next_update_async()
            if self._context.get_stage_id() != stage_id:
                raise asyncio.CancelledError()
            self._elapsed, self._physics_dt = 0.0, 1 / 60
            self._last_physics_update = time.monotonic()
            self._physics_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_physics_step)
            self._timeline.play()
            await self._app.next_update_async()
            await self._advance(0.05)
            self._simulation_view = omni.physics.tensors.create_simulation_view('numpy')
            self._simulation_view.set_subspace_roots('/')
            view = self._simulation_view.create_rigid_body_view(ROOT_PATH + '/instance_*')
            for body in self.bodies:
                body.physics_view = view
                body.index = view.prim_paths.index(body.prim_path)
            for index, (body, pose) in enumerate(zip(self.bodies, prepared)):
                drop_object(stage, body, pose, self.objects[:index], self.objects[index], surface_bounds[1][2])
                await self._advance(0.25)
            self._set_status('settling')
            self._result = None
            self._settle_start = self._elapsed
            self._tracker = SettlingTracker(self._physics_dt, minimum_height=surface_bounds[1][2] - 0.5)
            while self._result is None and self._elapsed - self._settle_start < settings['settle_timeout']:
                self._check_stage()
                await self._app.next_update_async()
            if self._result is None or not self._result['settled']:
                raise RuntimeError(self._result['reason'] if self._result else 'Settling timeout')
            for body, metadata in zip(self.bodies, self.objects):
                position, orientation = body.get_world_pose()
                body.set_default_state(position, orientation)
                metadata.update(position=position.tolist(), orientation_wxyz=orientation.tolist())
            self._set_status('ready', f'{len(self.objects)} objects settled')
            completed = True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if created and not prepared_all and self._context.get_stage_id() == stage_id:
                self._remove_objects(stage)
            self._set_status('failed', str(error))
            carb.log_error(f'Object spawning failed: {error}')
        finally:
            if self._task is None or self._task is asyncio.current_task():
                self._tracker = None
                self._physics_subscription = None
                if stage is not None and self._context.get_stage_id() == stage_id:
                    play = keep_playing if completed else was_playing
                    if play != self._timeline.is_playing():
                        if play:
                            self._timeline.play()
                        else:
                            self._timeline.pause()
                        await self._app.next_update_async()

    def _build_ui(self):
        import omni.ui as ui

        self._window = ui.Window('Object Spawner', width=420, height=310)
        with self._window.frame:
            with ui.VStack(spacing=6):
                for name, label, field in (
                    ('assets_directory', 'Asset directory', ui.StringField),
                    ('target_surface', 'Target surface prim', ui.StringField),
                    ('count', 'Object count', ui.IntField),
                    ('seed', 'Random seed', ui.IntField),
                    ('spread', 'Spread per axis (m)', ui.FloatField),
                    ('settle_timeout', 'Settling timeout (s)', ui.FloatField),
                ):
                    with ui.HStack():
                        ui.Label(label, width=170)
                        widget = field()
                        value = self._settings.get(SETTINGS + '/' + name)
                        if name == 'assets_directory' and not value:
                            value = str(self._default_assets)
                        widget.model.set_value(value)
                        getter = 'as_string' if field == ui.StringField else ('as_int' if field == ui.IntField else 'as_float')
                        widget.model.add_value_changed_fn(
                            lambda model, name=name, getter=getter: self._settings.set(SETTINGS + '/' + name, getattr(model, getter)))
                with ui.HStack(spacing=6):
                    ui.Button('Spawn / Replace', clicked_fn=self.spawn)
                    ui.Button('Clear', clicked_fn=self.clear)
                self._status_label = ui.Label(self._settings.get(SETTINGS + '/status'), word_wrap=True)

    def on_shutdown(self):
        running = self._task is not None and not self._task.done()
        self._cancel()
        self._release_bodies()
        stage = self._context.get_stage()
        root = stage.GetPrimAtPath(ROOT_PATH) if stage is not None else None
        if running and root and root.GetCustomDataByKey('objectSpawnerOwner') == OWNER:
            clear_objects(stage)
        self._stage_subscription = None
        if self._window is not None:
            self._window.destroy()
            self._window = None
        self._status_label = None
        self.bodies, self.objects = [], []
