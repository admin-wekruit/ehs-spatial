"""Wrap every exported .glb as <name>.glb.json ({"b64": ...}) - claude.ai artifacts do not serve .glb - and write
files.json, the published-path -> local-path map to pass as the Artifact tool's `files` for <video>/index.html."""
import base64, json, os, sys

R = os.path.dirname(os.path.abspath(__file__))
for key in (sys.argv[1:] or ('me340', 'samsclub', 'walmart')):
    d = f'{R}/{key}'; files = {}
    for f in sorted(os.listdir(d)):
        p = f'{d}/{f}'
        if f.endswith('.glb'):
            out = p + '.json'
            json.dump({'b64': base64.b64encode(open(p, 'rb').read()).decode()}, open(out, 'w'))
            assert os.path.getsize(out) < 15.9e6, f'{out} is over the 16 MB text-file limit: lower the budgets in export.py'
            files[f + '.json'] = out
        elif f in ('data.json', 'video.mp4'):
            files[f] = p
    total = sum(os.path.getsize(v) for v in files.values())
    assert total < 63e6, f'{key}: {total / 1e6:.1f} MB is over the 64 MB per-version limit'
    json.dump(files, open(f'{d}/files.json', 'w'), indent=0)
    print(key, f'{total / 1e6:.1f} MB in {len(files)} files')
