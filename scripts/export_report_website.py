"""Publish the same report UI on the existing static website; Modal serves its API."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil


def export(destination: Path, api_origin: str, site_root: str) -> None:
    from urllib.parse import urlsplit
    for value in (api_origin, site_root):
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment or url.username or url.password:
            raise ValueError("Expected a deployment URL without credentials or query")
    source = Path(__file__).resolve().parents[1] / "ehs_spatial/static"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination / "workspace-assets", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('workspace.html', '__pycache__', '*.pyc'))
    (destination / "workspace-assets/workspace.html").unlink(missing_ok=True)
    page = (source / "workspace.html").read_text().replace('"/workspace-assets/', '"workspace-assets/')
    page = page.replace('href="/reports"', 'href="reports.html"')
    (destination / "reports.html").write_text(page)
    config = {
        "apiOrigin": api_origin.rstrip('/'), "siteRoot": site_root.rstrip('/') + '/',
        "workspaceRoot": site_root.rstrip('/') + '/reports.html',
        "workspaceAssets": site_root.rstrip('/') + '/workspace-assets/',
    }
    script = 'window.panoptesSiteConfig = ' + json.dumps(config, indent=2) + ';\n'
    for directory in (destination, destination / 'workspace-assets'):
        (directory / 'site-config.js').write_text(script)
        shutil.copyfile(source / 'site-client.js', directory / 'site-client.js')
    # Entry pages must not combine this release with an older cached loader.
    for name in ('index.html', 'reports.html', 'viewer.html', 'report-artifact.html'):
        entry = destination / name
        if not entry.is_file():
            continue
        def version(match):
            asset = destination / match[2]
            if not asset.is_file() or not asset.resolve().is_relative_to(destination.resolve()):
                return match[0]
            digest = hashlib.sha256(asset.read_bytes()).hexdigest()[:12]
            return f'{match[1]}="{match[2]}?v={digest}"'
        entry.write_text(re.sub(r'(src|href)="([^"?]+\.(?:js|css))(?:\?[^"\n]*)?"', version, entry.read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--api-origin', required=True)
    parser.add_argument('--site-root', required=True)
    args = parser.parse_args()
    export(args.destination, args.api_origin, args.site_root)
