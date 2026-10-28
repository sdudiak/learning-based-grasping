import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from scripts import convert_assets, download_gso_objects


class AssetTests(unittest.TestCase):
    def test_empty_asset_name_cannot_replace_the_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / 'keep').write_text('existing assets')
            with self.assertRaises(ValueError):
                download_gso_objects.fetch_one('', output)
            self.assertEqual((output / 'keep').read_text(), 'existing assets')

    def test_archive_rejects_traversal_and_extracts_regular_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / 'asset.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('../outside.obj', 'invalid')
            with self.assertRaises(ValueError):
                download_gso_objects.safe_extract(archive, root / 'extracted')
            self.assertFalse((root / 'outside.obj').exists())
            with zipfile.ZipFile(archive, 'w') as stream:
                stream.writestr('meshes/model.obj', 'mesh')
            download_gso_objects.safe_extract(archive, root / 'extracted')
            self.assertEqual((root / 'extracted/meshes/model.obj').read_text(), 'mesh')

    def test_failed_overwrite_preserves_previous_asset(self):
        self.check_conversion(fail=True)

    def test_successful_overwrite_replaces_complete_asset(self):
        self.check_conversion(fail=False)

    def check_conversion(self, fail):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'raw/object/model.obj'
            source.parent.mkdir(parents=True)
            source.write_text('mesh')
            destination = root / 'converted/object'
            destination.mkdir(parents=True)
            (destination / 'object.usd').write_text('old asset')
            (destination / 'mesh.usd').write_text('old mesh')
            quit_codes = []
            modules = {name: types.ModuleType(name) for name in (
                'isaacsim', 'isaacsim.core', 'isaacsim.core.utils', 'isaacsim.core.utils.extensions',
                'omni', 'omni.kit', 'omni.kit.app')}
            modules['isaacsim'].SimulationApp = lambda config: types.SimpleNamespace(close=lambda: None)
            modules['isaacsim.core.utils.extensions'].enable_extension = lambda name: None
            modules['omni'].kit = modules['omni.kit']
            modules['omni.kit'].app = modules['omni.kit.app']
            modules['omni.kit.app'].get_app = lambda: types.SimpleNamespace(post_quit=quit_codes.append)

            def convert(src, staging, size):
                staging.mkdir()
                (staging / 'mesh.usd').write_text('new mesh')
                if fail:
                    raise RuntimeError('conversion failed')
                (staging / 'object.usd').write_text('new asset')
                return 1.0

            arguments = types.SimpleNamespace(size_cm=10.0, overwrite=True)
            with patch.dict(sys.modules, modules), patch.multiple(
                convert_assets, INPUT=root / 'raw', OUTPUT=root / 'converted'
            ), patch.object(convert_assets, 'arguments', return_value=arguments), patch.object(
                convert_assets, 'convert_asset', side_effect=convert
            ):
                if fail:
                    with self.assertRaises(SystemExit) as error:
                        convert_assets.main()
                    self.assertEqual(error.exception.code, 1)
                else:
                    convert_assets.main()
            expected = 'old' if fail else 'new'
            self.assertEqual((destination / 'object.usd').read_text(), expected + ' asset')
            self.assertEqual((destination / 'mesh.usd').read_text(), expected + ' mesh')
            self.assertEqual(quit_codes, [1 if fail else 0])
            self.assertEqual(sorted(path.name for path in destination.parent.iterdir()), ['conversion_manifest.json', 'object'])


if __name__ == '__main__':
    unittest.main()
