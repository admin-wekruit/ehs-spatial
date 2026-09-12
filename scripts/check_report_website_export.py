"""Run with Python: check one-source export and content-based browser cache versions."""
import hashlib
from pathlib import Path
import re
import tempfile

from export_report_website import export


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / 'viewer.html').write_text('<script src="site-client.js?v=old"></script>')
    export(root, 'https://api.example', 'https://site.example/reports/')
    assert not (root / 'workspace-assets/workspace.html').exists()
    assert (root / 'site-client.js').read_bytes() == (root / 'workspace-assets/site-client.js').read_bytes()
    for name in ['reports.html', 'viewer.html']:
        page = (root / name).read_text()
        for asset, version in re.findall(r'(?:src|href)="([^"?]+)\?v=([a-f0-9]+)"', page):
            assert hashlib.sha256((root / asset).read_bytes()).hexdigest().startswith(version)
        assert '"/workspace-assets/' not in page
    previous = (root / 'viewer.html').read_bytes()
    export(root, 'https://api.example', 'https://site.example/reports/')
    assert (root / 'viewer.html').read_bytes() == previous
    try:
        export(root, 'https://secret@api.example', 'https://site.example/')
    except ValueError:
        pass
    else:
        raise AssertionError('Credentials must not enter a public config')
print('Website export and cache versions passed')
