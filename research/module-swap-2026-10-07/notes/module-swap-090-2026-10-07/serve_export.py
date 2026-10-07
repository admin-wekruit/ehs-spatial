"""Serve a locally exported (unpublished) publication to the cloud checks: the platform export catalog
(.platform/publication-catalog/<id>: bundle.json + blobs/<sha>) laid out as modal_apps/workcell_layer_trial.py --served-dir expects
(api/api/assets/<id> = {"url": "/blobs/<sha>"}, api/blobs/<sha> = the bytes), plus the --photo argument for the checks.

    python serve_export.py VARIANT    -> swap-runs/<VARIANT>-served/, prints the photo map (imageId=file,...)
"""
import json
import os
from pathlib import Path
import shutil
import sys

SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
PLATFORM = Path(os.environ.get('PANOPTES_PLATFORM', '/Users/adam/Desktop/Tesla/panoptes-platform'))


def main(variant):
    result = json.loads((PLATFORM / '.platform/swap-20261007' / variant / 'result.json').read_text())
    catalog = PLATFORM / result['export']
    result['view'] = str(Path(result['view']) if Path(result['view']).is_absolute() else PLATFORM / result['view'])
    bundle = json.loads((catalog / 'bundle.json').read_text())
    served = SP / 'swap-runs' / f'{variant}-served'
    (served / 'api/api/assets').mkdir(parents=True, exist_ok=True)
    (served / 'api/blobs').mkdir(parents=True, exist_ok=True)
    n = 0
    for route, body in bundle['responses'].items():
        if route.startswith('/api/assets/') and isinstance(body, dict) and body.get('sha256'):
            blob = catalog / 'blobs' / body['sha256']
            if not blob.exists():
                continue
            (served / 'api/api/assets' / body['id']).write_text(json.dumps({'url': '/blobs/' + body['sha256']}))
            dst = served / 'api/blobs' / body['sha256']
            if not dst.exists():
                shutil.copyfile(blob, dst)
            n += 1
    view = json.loads(Path(result['view']).read_bytes())
    doc = view['publication']['snapshot']['revision']['document']
    run = Path(result['run'])
    manifest = json.loads((run / 'manifest.json').read_text())
    by_frame = {f['frame_id']: f['input'].split('/')[-1] for f in manifest['frames']}
    photos = []
    for cam in doc['cameras']:
        frame = cam.get('frameId') or cam.get('frame_id') or next((f for f in by_frame if f in json.dumps(cam)), None)
        if frame is None:  # match by image sha / name
            frame = next((f['frame_id'] for f in manifest['frames'] if f.get('decoded_rgb_sha256') == cam.get('imageSha256')), None)
        photos.append(f"{cam['imageId']}={by_frame[frame]}")
    out = {'served': str(served), 'assets': n, 'photos': ','.join(photos), 'photosDir': str(run / 'input'), 'view': result['view'],
           'publicationId': result['publicationId'], 'revisionId': result['revisionId'], 'nativeToMeters': result['nativeToMeters']}
    (served.parent / f'{variant}-served.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == '__main__':
    main(sys.argv[1])
