#!/usr/bin/env python3
"""Download selected Google Scanned Objects (GSO) from the Gazebo Fuel collection.

Original data: Google Scanned Objects, CC BY 4.0.
Source: https://app.gazebosim.org/GoogleResearch/fuel/collections/Scanned%20Objects%20by%20Google%20Research

Run from the project root. Python 3.10+; no third-party packages required.
"""
import argparse
import json
import random
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE = 'https://fuel.gazebosim.org/1.0'
OWNER = 'GoogleResearch'
COLLECTION = 'Scanned Objects by Google Research'
PER_PAGE = 100


def read_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'gso-asset-downloader/1.0'})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def read_index():
    """Read the whole paginated collection, keeping only GoogleResearch models."""
    result = set()
    page = 1
    while True:
        params = urllib.parse.urlencode({
            'page': page,
            'per_page': PER_PAGE,
            'q': f'collections:{COLLECTION}',
        })
        url = f'{BASE}/models?{params}'
        data = read_json(url)
        if not isinstance(data, list):
            raise ValueError(f'Unexpected Fuel API response at page {page}: expected list')
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get('name'), str):
                continue
            owner = item.get('owner', item.get('owner_name', OWNER))
            if isinstance(owner, dict):
                owner = owner.get('name', owner.get('username'))
            # Collection occasionally includes contributions from other owners.
            if owner is not None and str(owner).lower() != OWNER.lower():
                continue
            name = item['name']
            if not name or name in ('.', '..') or '/' in name or '\\' in name:
                continue
            result.add(name)
        if len(data) < PER_PAGE:
            break
        page += 1
        if page > 100:
            raise ValueError('Too many Fuel API pages; refusing to continue')
    if not result:
        raise ValueError('No objects found; check Fuel collection availability / API response')
    return sorted(result)


def url_for(name):
    return f'{BASE}/{OWNER}/models/{urllib.parse.quote(name, safe="")}.zip'


def download(url, destination):
    request = urllib.request.Request(url, headers={'User-Agent': 'gso-asset-downloader/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, destination.open('wb') as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)


def safe_extract(archive, destination):
    """Reject path traversal, symlinks, special files and absolute paths."""
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            path = item.filename.replace('\\', '/')
            if not path or path.startswith('/') or '\x00' in path:
                raise ValueError(f'Unsafe ZIP path: {item.filename!r}')
            parts = path.split('/')
            if '..' in parts or any(p.endswith(':') for p in parts):
                raise ValueError(f'Unsafe ZIP path: {item.filename!r}')
            target = (root / path).resolve()
            if not target.is_relative_to(root):
                raise ValueError(f'Unsafe ZIP path: {item.filename!r}')
            mode = (item.external_attr >> 16) & 0o170000
            if mode not in (0, 0o100000, 0o040000):
                raise ValueError(f'Unsupported ZIP member: {item.filename!r}')
        zf.extractall(root)


def fetch_one(name, output):
    if not name or name in ('.', '..') or '/' in name or '\\' in name:
        raise ValueError(f'Invalid GSO model name: {name!r}')
    target = output / name
    if (target / '.complete').is_file():
        print(f'[cached] {name}')
        return True
    url = url_for(name)
    with tempfile.TemporaryDirectory(prefix='gso_', dir=output) as tmp:
        tmp = Path(tmp)
        archive = tmp / 'asset.zip'
        try:
            print(f'[download] {name}')
            download(url, archive)
        except urllib.error.HTTPError as err:
            if err.code == 404:
                print(f'[unavailable] {name}: no ZIP archive')
                return False
            raise
        extracted = tmp / 'extracted'
        extracted.mkdir()
        safe_extract(archive, extracted)
        if not any(extracted.rglob('*.obj')):
            raise ValueError('ZIP contains no OBJ mesh')
        (extracted / '.complete').write_text(url + '\n', encoding='utf-8')
        if target.exists():
            shutil.rmtree(target)
        extracted.rename(target)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--objects', nargs='+', metavar='NAME', help='Exact GSO model names (use --list)')
    selection.add_argument('--count', type=int, metavar='N', help='Randomly select N available models')
    selection.add_argument('--all', action='store_true', help='Download all models in the collection')
    parser.add_argument('--list', action='store_true', help='List available GSO model names')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--output', type=Path, default=Path('isaac_ws/assets/objects/gso/raw'))
    args = parser.parse_args()

    try:
        names = read_index()
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as err:
        parser.exit(1, f'Could not read GSO model list: {err}\n')
    if args.list:
        print('\n'.join(names))
        return
    if args.count is not None and not 1 <= args.count <= len(names):
        parser.error(f'--count must be between 1 and {len(names)}')
    if args.objects:
        invalid = set(args.objects) - set(names)
        if invalid:
            parser.error(f'Unknown object(s): {sorted(invalid)}. Run --list.')
        selected = list(dict.fromkeys(args.objects))
        wanted = len(selected)
    elif args.count is not None:
        selected = random.Random(args.seed).sample(names, len(names))
        wanted = args.count
    else:
        selected = names
        wanted = len(names)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    successful = []
    for name in selected:
        try:
            if fetch_one(name, output):
                successful.append(name)
        except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile) as err:
            print(f'[failed] {name}: {err}')
        if len(successful) >= wanted:
            break
    manifest = output / 'selection_gso.json'
    manifest.write_text(json.dumps({
        'source': 'Gazebo Fuel / GoogleResearch / Scanned Objects by Google Research',
        'objects': successful,
    }, indent=2) + '\n', encoding='utf-8')
    print(f'\nDownloaded/available: {len(successful)}/{wanted}. Manifest: {manifest}')
    if len(successful) != wanted:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
