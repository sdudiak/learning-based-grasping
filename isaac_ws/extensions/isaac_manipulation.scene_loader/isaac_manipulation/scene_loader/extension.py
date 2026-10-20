import asyncio
import logging
from pathlib import Path

import carb.settings
import omni.ext
import omni.kit.app
import omni.usd

LOGGER = logging.getLogger(__name__)
SETTING = "/exts/isaac_manipulation.scene_loader/scene"


class SceneLoaderExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        self._task = None
        scene = carb.settings.get_settings().get(SETTING)
        if scene:
            self._task = asyncio.ensure_future(self._open_scene(scene))

    async def _open_scene(self, scene):
        path = Path(scene).expanduser().resolve()
        if not path.is_file():
            LOGGER.error("Scene not found: %s", path)
            return

        app = omni.kit.app.get_app()
        context = omni.usd.get_context()
        await app.next_update_async()
        while not app.is_app_ready() or not context.can_open_stage():
            await app.next_update_async()
        try:
            success, error = await context.open_stage_async(str(path))
        except Exception:
            LOGGER.exception("Failed to open scene: %s", path)
            return
        if not success:
            LOGGER.error("Failed to open scene %s: %s", path, error)
            return
        LOGGER.info("Opened scene: %s", path)

    def on_shutdown(self):
        if self._task and not self._task.done():
            self._task.cancel()
