"""Materialise a module-swapped run directory in the lucida-replica-01 layout, so the ORIGINAL downstream stages run unchanged:
the source run's inputs / geometry / evidence (real files, no symlinks) plus generation/<id>/{object.ply, posed-object.ply,
output.json} written from the chosen SAM 3D candidate of each object (the completion module's output contract, exactly what
generate_lucida_assets.py / generate_sam3d_assets.py write). The chosen candidate = compare.py's uniform selection (cmp-*/results.json).

    nice python swap_generation.py pi3x-sam3d | mvs-sam3d
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.environ.get('PANOPTES_SERVING', '/Users/adam/Desktop/panoptes-public/panoptes-serving') + '/scripts/research')
import assemble_lucida_scene as als  # noqa: E402

HERE = Path(__file__).resolve().parent
SP = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
LUCIDA = Path(os.environ.get('PANOPTES_RUNS', '/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation')) / 'lucida-replica-01'
VARIANTS = {
    'pi3x-sam3d': dict(src=LUCIDA, cands=SP / 'checks/completionAB-090', sel=HERE / 'cmp-pi3x/results.json'),
    'mvs-sam3d': dict(src=SP / 'checks/bbab-export-090-mvs-scipyba', cands=SP / 'swap-runs/mvs-recgen-ab', sel=HERE / 'cmp-mvs/results.json'),
    # geometry module v2: the same MVS run with MoGe-3 in-mask hole fill (fill_geometry.py); SAM 3D regenerated on it (its pointmap input changed)
    'mvs-fill-sam3d': dict(src=SP / 'checks/bbab-export-090-mvs-fill', cands=SP / 'swap-runs/mvs-fill-ab', sel=HERE / 'cmp-mvs-fill/results.json'),
    # cell 030 (the same scripts; run dirs under swap-runs/030): the published Pi3X run bor1-030-01 and the filled MVS export
    '030/pi3x-sam3d': dict(src=LUCIDA.parent / 'bor1-030-01', cands=SP / 'swap-runs/030/pi3x-ab', sel=HERE / 'cmp-030-pi3x/results.json'),
    '030/mvs-fill-sam3d': dict(src=SP / 'checks/bbab-export-030-mvs-fill', cands=SP / 'swap-runs/030/mvs-fill-ab', sel=HERE / 'cmp-030-mvs-fill/results.json'),
}
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
    sel = json.loads(v['sel'].read_text())
    record = {}
    for o in objects:
        oid = o['object_id']
        if oid not in sel:
            continue
        variant = sel[oid]['variant']
        out = dest / 'generation' / oid
        if (out / 'output.json').exists():
            record[oid] = json.loads((out / 'output.json').read_text())['selection']; continue
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
        selection = {'candidate': variant, 'generationPhoto': spec['frame_id'], 'decision': sel[oid]['decision'], 'rule': 'compare.py uniform (no box gates)'}
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
