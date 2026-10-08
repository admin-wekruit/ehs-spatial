"""Materialize the selected SAM3D candidates in the capture-run generation contract."""
from argus import ROOT
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import trimesh

import argus.pipeline.assemble_lucida_scene as als

HERE = Path(os.environ['PANOPTES_DATA_ROOT']) / 'pipeline'
SP = Path(os.environ['PANOPTES_DATA_ROOT'])
LUCIDA = Path(os.environ['PANOPTES_RUNS']) / 'lucida-replica-01'
VARIANTS = {f'{c}/mvs-fill-sam3d': dict(src=SP / f'checks/bbab-export-{c}-mvs-fill', cands=SP / f'swap-runs/{c}/mvs-fill-ab',
            sel=HERE / f'cmp-{c}-mvs-fill/results.json', objects=json.loads((ROOT / 'argus/pipeline/cells' / f'{c}.json').read_text())['objects'])
            for c in ('090', '030')}
VIEW_FILES = ['canonical_mask.npy', 'mask.png', 'points.npy', 'colors.npy']
P3D = np.diag([-1., -1, 1, 1])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copyfile(src, dst)


def main(name):
    v = VARIANTS[name]
    src, dest = v['src'], SP / 'swap-runs' / name
    objects = json.loads((src / 'evidence/objects.json').read_text())['objects']
    sel = json.loads(v['sel'].read_text())
    missing = set(v['objects']) - sel.keys()
    if missing:
        raise RuntimeError(f'Missing SAM3D selections for required objects: {sorted(missing)}')
    copy(src / 'manifest.json', dest / 'manifest.json')
    for p in (src / 'input').iterdir():
        copy(p, dest / 'input' / p.name)
    for p in (src / 'geometry/frames').rglob('*'):
        if p.is_file():
            copy(p, dest / p.relative_to(src))
    for n in ('objects.json', 'floor.json', 'floor_points.npy', 'floor_colors.npy'):
        if (src / 'evidence' / n).exists():
            copy(src / 'evidence' / n, dest / 'evidence' / n)
    for p in (src / 'evidence/canonical').iterdir():  # frozen canonical frames + alpha (report build and platform import read them)
        if p.is_file():
            copy(p, dest / 'evidence/canonical' / p.name)
    if (src / 'geometry/point_cloud.glb').exists():  # the geometry stage's observed point cloud (platform import)
        copy(src / 'geometry/point_cloud.glb', dest / 'geometry/point_cloud.glb')
    for d in (src / 'evidence/objects').iterdir():  # every evidence object incl. observed_floor (assemble reads its canonical masks)
        for p in d.rglob('*'):
            if p.is_file() and p.name in VIEW_FILES:
                copy(p, dest / p.relative_to(src))
    record = {}
    for o in objects:
        oid = o['object_id']
        if oid not in sel:
            continue
        variant = sel[oid]['variant']
        out = dest / 'generation' / oid
        selection = {'candidate': variant, 'decision': sel[oid]['decision'], 'rule': 'select_sam3d uniform (no box gates)'}
        if (out / 'output.json').exists():
            output = json.loads((out / 'output.json').read_text())
            if output['selection']['candidate'] != variant:
                raise RuntimeError(f'{oid}: selected candidate changed; recompute this workcell from S4c through S8 before publishing')
            selection['generationPhoto'] = output['anchor_frame']
            output['selection'] = selection
            (out / 'output.json').write_text(json.dumps(output, indent=2) + '\n')
            record[oid] = selection; continue
        frame = variant.split('-', 1)[1] if variant.startswith('sam3d-') else None
        spec = next(x for x in o['views'] if x['frame_id'] == frame) if frame else max(o['views'], key=lambda x: x['mask_pixels'])
        anchor = als.native_view(dest, spec)
        data = np.load(v['cands'] / variant / f'{oid}.npz')
        V, F = data['vertices'].astype(np.float64), data['faces'].astype(np.int64)
        colors = data['colors'] if 'colors' in data else None
        object_to_camera = P3D @ data['object_to_camera_p3d'].astype(np.float64)
        als.decompose(anchor['c2w'] @ object_to_camera)  # validates a rigid pose
        out.mkdir(parents=True)
        trimesh.Trimesh(V, F, vertex_colors=colors, process=False).export(out / 'object.ply')
        trimesh.Trimesh(als.transformed(V, object_to_camera), F, vertex_colors=colors, process=False).export(out / 'posed-object.ply')
        rec = json.loads((v['cands'] / variant / 'record.json').read_text())['objects'][oid]
        selection['generationPhoto'] = spec['frame_id']
        output = {'schema_version': 1, 'status': 'complete', 'object_id': oid, 'model_id': 'facebook/sam-3d-objects',
                  'model_revision': rec.get('pins', {}).get('modelRevision'), 'code_revision': rec.get('pins', {}).get('codeRevision'),
                  'licenses': {'sam3d_code': 'SAM License (Meta)', 'sam3d_weights': 'SAM License (Meta)'},
                  'pose_from_model': True, 'coordinate_space': 'native object; posed mesh in anchor OpenCV camera coordinates',
                  'anchor_frame': anchor['frame_id'], 'object_to_camera': object_to_camera.tolist(),
                  'vertices': int(len(V)), 'faces': int(len(F)), 'inference_seconds': rec.get('seconds'), 'hardware': {'gpu': rec.get('gpu')},
                  'paths': {'mesh': 'object.ply', 'posed_mesh': 'posed-object.ply'},
                  'output_sha256': {n: digest(out / n) for n in ('object.ply', 'posed-object.ply')}, 'selection': selection}
        (out / 'output.json').write_text(json.dumps(output, indent=2) + '\n')
        record[oid] = selection
        print(oid, variant, len(F), 'faces', flush=True)
    (dest / 'generation' / 'swap-record.json').write_text(json.dumps({'variant': name, 'source_run': str(src), 'objects': record}, indent=1, ensure_ascii=False))
    print(name, 'done:', len(record), 'objects ->', dest)


if __name__ == '__main__':
    main(sys.argv[1])
