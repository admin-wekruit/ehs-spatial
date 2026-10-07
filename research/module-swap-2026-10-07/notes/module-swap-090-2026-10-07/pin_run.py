"""The platform importer's pins on a materialised run directory (what swap-runs/mvs-sam3d carries beyond the raw geometry export):
every object / floor view's sha256 gets mask.png + canonical_mask.npy (import_report_mask_source_unpinned), manifest.evidence
objects_sha256 / floor_sha256 are re-pinned, and geometry/point_cloud.glb = the content-valid points of every frame with their
canonical colours as one trimesh PointCloud node (import_geometry_path_invalid without it). Idempotent.

    python pin_run.py RUN_DIR
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import trimesh


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(run):
    run = Path(run)
    for name in ('objects', 'floor'):
        p = run / f'evidence/{name}.json'; d = json.loads(p.read_text())
        views = [v for o in d['objects'] for v in o['views']] if name == 'objects' else d['views']
        for v in views:
            v['sha256'] = dict(v.get('sha256') or {}, **{Path(v[k]).name: digest(run / v[k]) for k in ('mask_path', 'canonical_mask_path')})
            v['sha256'].pop('mask_path', None); v['sha256'].pop('canonical_mask_path', None)   # keys of an earlier, wrong pin pass
            v['observed_only'] = True
        p.write_text(json.dumps(d, indent=2) + '\n')
    man = json.loads((run / 'manifest.json').read_text())
    man['evidence'].update(objects_sha256=digest(run / 'evidence/objects.json'), floor_sha256=digest(run / 'evidence/floor.json'))
    glb = run / 'geometry/point_cloud.glb'
    if not glb.exists():
        P, C = [], []
        for f in man['frames']:
            g = run / 'geometry/frames' / f['frame_id']
            m = np.load(g / 'content_valid_mask.npy').astype(bool)
            P.append(np.load(g / 'pts3d.npy')[m]); C.append(np.asarray(Image.open(g / 'canonical.png').convert('RGB'))[m])
        trimesh.PointCloud(np.concatenate(P), colors=np.concatenate(C)).export(glb)
    (run / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    print(run, 'pinned; glb points', len(trimesh.load(glb, force='scene').geometry['geometry_0'].vertices))


if __name__ == '__main__':
    main(sys.argv[1])
