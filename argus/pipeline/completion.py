"""SAM3D candidates with the existing pointmap inputs and assembly parameters."""
from argus import ROOT
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import time

import modal


def env(key):  # env.template section 6, set by env.sh; a Modal container has none of them and uses nothing read through here
    return os.environ[key] if modal.is_local() else os.environ.get(key, '')


HERE = Path(__file__).resolve().parent
WT = ROOT
SERVING = ROOT
CELL = os.environ.get('AB_CELL', '030')  # AB_CELL=090: the September cell (run lucida-replica-01, all nine RecGen objects)
# AB_RUN=DIR AB_OUT=DIR [AB_OBJECTS=a,b,...]: any run in the lucida-replica-01 layout (e.g. a rebuilt geometry), its own output
# directory, and the objects to complete (default: every object of its evidence/objects.json); the 030 / 090 defaults are unchanged
RUN = Path(os.environ['AB_RUN']) if os.environ.get('AB_RUN') else Path(os.environ.get('PANOPTES_RUNS', Path(env('PANOPTES_DATA_ROOT')) / 'runs')) / {'030': 'bor1-030-01', '090': 'lucida-replica-01'}[CELL]
OUT = Path(os.environ['AB_OUT']) if os.environ.get('AB_OUT') else Path(
    env('PANOPTES_DATA_ROOT') + '/checks/'
    + {'030': 'completionAB-out', '090': 'completionAB-090'}[CELL])
OBJECTS = (os.environ['AB_OBJECTS'].split(',') if os.environ.get('AB_OBJECTS') else
           [o['object_id'] for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects']] if os.environ.get('AB_RUN') else
           {'030': ['cart', 'guard', 'left_light_curtain', 'left_post', 'right_light_curtain', 'robot'],
            '090': ['left_light_curtain', 'right_light_curtain', 'left_fence', 'right_fence', 'left_post', 'right_post', 'robot',
                    'cart', 'guard']}[CELL])  # field-checked objects first

A100 = .000694 + 4 * .0000131 + 32 * .00000222
CPU8 = 8 * .0000131 + 16 * .00000222
app = modal.App('panoptes-completion')
if modal.is_local():
    import argus.providers.sam3d_modal as sam3d_research
    app.include(sam3d_research.app)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 23), b''):
            h.update(block)
    return h.hexdigest()






def best_view(obj, frame_id=None):
    """The generation photo: the largest mask, or the given frame (variant 'sam3d-<frame_id>': SAM 3D from another photo)."""
    if frame_id:
        return next(v for v in obj['views'] if v['frame_id'] == frame_id)
    return max(obj['views'], key=lambda v: v['mask_pixels'])


def variant_frame(variant):
    return variant.split('-', 1)[1] if variant.startswith('sam3d-') else None


def sam3d_inputs(obj, manifest, factor=2, frame_id=None):
    """The whole photo (SAM 3D crops around the mask itself and infers a centred-principal-point K from the pointmap) at
    1/factor, the accepted original-resolution mask, and our Pi3X depth (nearest canonical pixel) as its pointmap:
    PyTorch3D camera (OpenCV x, y negated), NaN = no depth (as complete_video_objects.sam3d_pointmap)."""
    import cv2
    import numpy as np
    from PIL import Image
    from argus.pipeline.assemble_lucida_scene import camera_depth
    view = best_view(obj, frame_id)
    frame = next(f for f in manifest['frames'] if f['frame_id'] == view['frame_id'])
    assert digest(RUN / frame['input']) == frame['sha256']
    photo = Image.open(RUN / frame['input']).convert('RGB')
    W, H = photo.size
    w, h = W // factor, H // factor
    rgb = np.asarray(photo.resize((w, h), Image.Resampling.BOX))
    mask = cv2.resize((np.asarray(Image.open(RUN / view['mask_path'])) > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= .5
    points = np.load(RUN / view['pointmap_path'])
    c2w = np.load(RUN / view['c2w_path'])
    K = np.load(RUN / view['K_path']).astype(np.float64)
    conf = np.load(RUN / view['conf_path'])
    valid = np.load(RUN / view['content_valid_path']).astype(bool) & np.isfinite(conf) & (conf >= .1)
    depth, _ = camera_depth(points, c2w, valid)
    A = np.asarray(frame['input_to_canonical_pixel_centres'], float)
    v, u = np.indices((h, w), dtype=np.float64)
    uf, vf = factor * u + (factor - 1) / 2, factor * v + (factor - 1) / 2  # full-resolution pixel centres
    cx = np.floor(A[0, 0] * uf + A[0, 2] + .5).astype(int)
    cy = np.floor(A[1, 1] * vf + A[1, 2] + .5).astype(int)
    inside = (cx >= 0) & (cy >= 0) & (cx < depth.shape[1]) & (cy < depth.shape[0])
    z = np.full((h, w), np.nan, np.float32)
    z[inside] = depth[cy[inside], cx[inside]]
    z[z <= 0] = np.nan
    Kf = np.linalg.inv(A) @ K  # full-resolution K
    Ks = np.diag([1 / factor, 1 / factor, 1.]) @ Kf
    Ks[:2, 2] = (Kf[:2, 2] - (factor - 1) / 2) / factor
    pointmap = np.stack([-(u - Ks[0, 2]) / Ks[0, 0] * z, -(v - Ks[1, 2]) / Ks[1, 1] * z, z], -1).astype(np.float32)
    meta = {'frame_id': view['frame_id'], 'grid_hw': [h, w], 'factor': factor, 'K': Ks.tolist(), 'mask_pixels': int(mask.sum()),
            'mask_depth_pixels': int((mask & np.isfinite(z)).sum())}
    return rgb, mask, pointmap, meta


def assemble_variants(archive: bytes, variants: list, iterations: int = 100, eval_size: int = 288):
    import shutil
    import numpy as np
    import trimesh
    import argus.pipeline.assemble_lucida_scene as als
    started = time.monotonic()
    work = Path(tempfile.mkdtemp(prefix='completion-ab-'))
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as bundle:
        bundle.extractall(work, filter='data')
    shared = work / 'shared'
    objects = {o['object_id']: o for o in json.loads((shared / 'evidence/objects.json').read_text())['objects']}
    up = np.asarray(json.loads((shared / 'evidence/floor.json').read_text())['up_native'], float)
    results = {}
    for variant in variants:
        root = Path(tempfile.mkdtemp(prefix='completion-ab-runs-')) / variant
        (root / 'generation').mkdir(parents=True)
        for name in ['manifest.json', 'evidence', 'geometry']:
            (root / name).symlink_to(shared / name)
        inits = {}
        for gen in sorted((work / 'gen' / variant).iterdir()):
            oid, out = gen.name, root / 'generation' / gen.name
            obj = objects[oid]
            data = np.load(gen / 'mesh.npz')
            V, F = data['vertices'].astype(np.float64), data['faces'].astype(np.int64)
            anchor = als.native_view(root, best_view(obj, variant_frame(variant)))
            object_to_camera = np.diag([-1., -1, 1, 1]) @ data['object_to_camera_p3d'].astype(np.float64)
            info = {'pose': 'SAM 3D from our pointmap'}
            als.decompose(anchor['c2w'] @ object_to_camera)
            out.mkdir()
            trimesh.Trimesh(V, F, process=False).export(out / 'object.ply')
            trimesh.Trimesh(als.transformed(V, object_to_camera), F, process=False).export(out / 'posed-object.ply')
            model = 'facebook/sam-3d-objects'
            record = {'status': 'complete', 'object_id': oid, 'model_id': model, 'anchor_frame': anchor['frame_id'],
                      'object_to_camera': object_to_camera.tolist(), 'init': info,
                      'paths': {'mesh': 'object.ply', 'posed_mesh': 'posed-object.ply'},
                      'output_sha256': {n: digest(out / n) for n in ['object.ply', 'posed-object.ply']}}
            (out / 'output.json').write_text(json.dumps(record, indent=1))
            inits[oid] = info
        t = time.monotonic()
        als.assemble(root, iterations, eval_size)
        comparisons = json.loads((root / 'result/comparisons.json').read_text())
        silhouettes = {}
        for c in comparisons['objects']:
            c['refinement'].pop('trajectory', None)
            obj, rec = objects[c['object_id']], json.loads((root / 'generation' / c['object_id'] / 'output.json').read_text())
            mesh = trimesh.load(root / 'generation' / c['object_id'] / rec['paths']['mesh'], force='mesh', process=False)
            for spec in obj['views']:
                view = als.load_view(root, spec, c['evaluation_grid_height'])
                for key in ['initial_object_to_world', 'final_object_to_world']:
                    hit = als.cast_depth(mesh, np.asarray(c[key]), view['rays'])
                    silhouettes[f"{c['object_id']}/{spec['frame_id']}/{key[:-16]}"] = np.packbits(np.isfinite(hit) & (hit > 0)).tobytes()
                silhouettes[f"{c['object_id']}/{spec['frame_id']}/target"] = np.packbits(view['target']).tobytes()
                silhouettes[f"{c['object_id']}/{spec['frame_id']}/shape"] = json.dumps(view['target'].shape).encode()
        results[variant] = {'comparisons': comparisons, 'inits': inits, 'assembly_seconds': time.monotonic() - t,
                            'silhouettes': silhouettes}
    return {'results': results, 'container_seconds': time.monotonic() - started}




def stage_sam3d(frame_id=None):
    import numpy as np
    manifest = json.loads((RUN / 'manifest.json').read_text())
    if frame_id == 'all':  # AB_FRAME=all: the largest-mask photo, then every other photo, in one app run (one warm container)
        for f in [None] + [f['frame_id'] for f in manifest['frames']]:
            stage_sam3d(f)
        from argus.pipeline.cli import missing_candidates
        missing = missing_candidates(OUT, OBJECTS, ['sam3d'] + ['sam3d-' + f['frame_id'] for f in manifest['frames']])
        if missing:
            raise RuntimeError('SAM3D produced no candidate for configured objects: ' + ', '.join(missing))
        return
    objects = {o['object_id']: o for o in json.loads((RUN / 'evidence/objects.json').read_text())['objects']}
    dest = OUT / ('sam3d-' + frame_id if frame_id else 'sam3d')
    dest.mkdir(parents=True, exist_ok=True)
    backend = os.environ.get('SAM3D_BACKEND', 'modal')
    from argus.providers import sam3d as sam3d_provider
    log = json.loads((dest / 'record.json').read_text()) if (dest / 'record.json').exists() else {'objects': {}}
    for oid in OBJECTS:
        if (dest / f'{oid}.npz').exists():
            continue
        if frame_id and (frame_id not in [v['frame_id'] for v in objects[oid]['views']] or best_view(objects[oid])['frame_id'] == frame_id):
            continue  # no such photo, or it is the default generation photo (already in sam3d/)
        rgb, mask, pointmap, meta = sam3d_inputs(objects[oid], manifest, frame_id=frame_id)
        t = time.monotonic()
        out = sam3d_provider.generate(rgb, mask, pointmap, 42)
        meta['call_seconds'] = time.monotonic() - t
        if 'error' in out:
            meta.update(error=out['error'][-3000:], seconds=out['seconds'])
            print(oid, 'error', out['error'][-800:])
        else:
            np.savez_compressed(dest / f'{oid}.npz', vertices=out['vertices'], faces=out['faces'].astype(np.int32), colors=out['colors'],
                                object_to_camera_p3d=out['object_to_camera_p3d'])
            meta.update(seconds=out['seconds'], gpu=out['gpu'], pins=out['pins'], vertices=len(out['vertices']), faces=len(out['faces']))
            if backend:
                meta.update(backend=backend, model_info=out.get('model_info'))
            print(oid, len(out['faces']), 'faces', round(out['seconds'], 1), 's')
        log['objects'][oid] = meta
        (dest / 'record.json').write_text(json.dumps(log, indent=1))


def stage_assemble(variants):
    import numpy as np
    buffer = io.BytesIO()
    objects = json.loads((RUN / 'evidence/objects.json').read_text())['objects']
    # dereference: a run may hold symlinks (e.g. masks linked to the source run); the container's extractall(filter='data')
    # refuses absolute links, so the archive carries the files themselves
    with tarfile.open(fileobj=buffer, mode='w:gz', dereference=True) as bundle:
        for rel in ['manifest.json', 'evidence/objects.json', 'evidence/floor.json', 'geometry/frames']:
            bundle.add(RUN / rel, arcname=f'shared/{rel}')
        for p in sorted((RUN / 'evidence/objects').glob('*/*/canonical_mask.npy')):
            bundle.add(p, arcname=f'shared/{p.relative_to(RUN)}')
        for variant in variants:
            for oid in OBJECTS:
                src = OUT / variant / f'{oid}.npz'
                if src.exists():
                    if variant.startswith('sam3d'):  # mesh.npz with the keys assemble_variants reads
                        d = np.load(src)
                        one = io.BytesIO()
                        np.savez(one, vertices=d['vertices'], faces=d['faces'], object_to_camera_p3d=d['object_to_camera_p3d'])
                        info = tarfile.TarInfo(f'gen/{variant}/{oid}/mesh.npz')
                        info.size = one.getbuffer().nbytes
                        one.seek(0)
                        bundle.addfile(info, one)
                    else:
                        bundle.add(src, arcname=f'gen/{variant}/{oid}/mesh.npz')
    print('archive MB', round(buffer.getbuffer().nbytes / 1e6, 1))
    t = time.monotonic()
    result = assemble_variants(buffer.getvalue(), variants)
    call = time.monotonic() - t
    for variant, r in result['results'].items():
        dest = OUT / 'assembly' / variant
        dest.mkdir(parents=True, exist_ok=True)
        sil = r.pop('silhouettes')
        np.savez_compressed(dest / 'silhouettes.npz', **{k.replace('/', '__'): np.frombuffer(v, np.uint8) for k, v in sil.items()})
        (dest / 'comparisons.json').write_text(json.dumps(r, indent=1))
    ledger = {'functionSeconds': result['container_seconds'], 'callSeconds': call, 'estimateUsd': result['container_seconds'] * CPU8}
    (OUT / 'assembly' / f"ledger-{'-'.join(variants)}.json").write_text(json.dumps(ledger, indent=1))
    print(json.dumps(ledger))



@app.local_entrypoint()
def main(stage: str, variants: str = 'sam3d'):
    OUT.mkdir(parents=True, exist_ok=True)
    if stage == 'sam3d':
        stage_sam3d(os.environ.get('AB_FRAME'))
    elif stage == 'assemble':
        stage_assemble(variants.split(','))
    else:
        raise ValueError(stage)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Assemble SAM3D candidates on the pipeline machine')
    parser.add_argument('--stage', choices=['sam3d', 'assemble'], required=True)
    parser.add_argument('--variants', default='sam3d')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == 'sam3d':
        stage_sam3d(os.environ.get('AB_FRAME'))
    else:
        stage_assemble(args.variants.split(','))
