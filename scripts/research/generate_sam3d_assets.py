"""Per-object completion for the serving generation step, switchable: --completion recgen | sam3d.

  recgen  generate_lucida_assets.py unchanged (RecGen: TRI non-commercial code, CC-BY-NC-4.0 weights); every other flag is its own.
  sam3d   SAM 3D Objects (facebook/sam-3d-objects, SAM License), the licence-clean on-prem path. Runs INSIDE the on-prem image
          docker/sam3d.Dockerfile (worktree codex/workcell-photo-speed) through scripts/onprem/run_stage.py --weights, which runs
          modal_apps/sam3d_research.SAM3DObjects (mesh only, internal depth disabled) in this process: no Modal, no network.
            docker run --rm --gpus all --network none -v W:/weights -v RUNS:/data \
                -v SERVING/scripts/research:/serving/scripts/research:ro panoptes-sam3d \
                python /workcell/scripts/onprem/run_stage.py --weights /weights \
                /serving/scripts/research/generate_sam3d_assets.py --completion sam3d --root /data/RUN --object-ids a,b [--seed 42]
          (W = scripts/onprem/fetch_weights_sam3d.py --cache W). Outside run_stage.py it refuses to run.

sam3d writes RUN/generation/<id>/ as generate_lucida_assets.py does: object.ply / object.glb (native object frame, vertex colours),
posed-object.ply / .glb (in the anchor's OpenCV camera), output.json (status, anchor_frame, object_to_camera, paths,
output_sha256, pins, licences), source.json; so assemble_lucida_scene.py places it unchanged. A failed object gets
output.json status 'failed' (assembly lists it as unavailable); existing outputs are never overwritten.
Input (= research-notes/completion-licence-ab-2026-10-05 sam3d_inputs, the proven one): the view with the largest mask is the
anchor; the whole photo at 1/2 (BOX), the accepted original-resolution mask (INTER_AREA >= .5), and our geometry's depth
(nearest canonical pixel) as its pointmap in the PyTorch3D camera (OpenCV x, y negated; NaN = no depth). SAM 3D infers K from
that pointmap and returns the pose in the same camera, so object_to_camera (OpenCV) = diag(-1, -1, 1, 1) @ its pose.
--self-check: synthetic pointmap / pose / file round trip (no model).
"""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_lucida_assets import camera_depth, digest  # noqa: E402

P3D = np.diag([-1., -1, 1, 1])  # OpenCV camera <-> PyTorch3D camera (x, y negated); its own inverse
LICENCES = {'sam3d_code': 'SAM License (Meta)', 'sam3d_weights': 'SAM License (Meta)', 'dinov2': 'Apache-2.0'}
SAFE = set('abcdefghijklmnopqrstuvwxyz0123456789_-')


def inside(root, rel):
    path = (root / rel).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f'Evidence path missing/outside run: {rel}')
    return path


def pointmap(z, K):
    """Camera depth z (NaN = none) on the grid of K -> H x W x 3 PyTorch3D-camera points."""
    v, u = np.indices(z.shape, dtype=np.float64)
    return np.stack([-(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, z], -1).astype(np.float32)


def sam3d_inputs(root, obj, manifest, factor=2):
    import cv2
    view = max(obj['views'], key=lambda v: v['mask_pixels'])
    frame = next(f for f in manifest['frames'] if f['frame_id'] == view['frame_id'])
    photo_path, mask_path = inside(root, frame['input']), inside(root, view['mask_path'])
    if digest(photo_path) != frame['sha256'] or digest(mask_path) != view['sha256']['mask.png']:
        raise ValueError('Photo or accepted mask hash changed')
    photo = Image.open(photo_path).convert('RGB')
    W, H = photo.size
    w, h = W // factor, H // factor
    rgb = np.asarray(photo.resize((w, h), Image.Resampling.BOX))
    mask = cv2.resize((np.asarray(Image.open(mask_path)) > 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) >= .5
    load = lambda key: np.load(inside(root, view[key]), allow_pickle=False)
    conf = load('conf_path')
    depth, _ = camera_depth(load('pointmap_path'), load('c2w_path'), load('content_valid_path').astype(bool) & np.isfinite(conf) & (conf >= .1))
    A = np.asarray(frame['input_to_canonical_pixel_centres'], float)
    v, u = np.indices((h, w), dtype=np.float64)
    cx = np.floor(A[0, 0] * (factor * u + (factor - 1) / 2) + A[0, 2] + .5).astype(int)  # full-resolution pixel centres
    cy = np.floor(A[1, 1] * (factor * v + (factor - 1) / 2) + A[1, 2] + .5).astype(int)  # -> nearest canonical pixel
    ok = (cx >= 0) & (cy >= 0) & (cx < depth.shape[1]) & (cy < depth.shape[0])
    z = np.full((h, w), np.nan, np.float32)
    z[ok] = depth[cy[ok], cx[ok]]
    z[z <= 0] = np.nan
    Kf = np.linalg.inv(A) @ load('K_path').astype(np.float64)  # full-resolution K
    Ks = np.diag([1 / factor, 1 / factor, 1.]) @ Kf
    Ks[:2, 2] = (Kf[:2, 2] - (factor - 1) / 2) / factor
    meta = {'frame_id': view['frame_id'], 'grid_hw': [h, w], 'factor': factor, 'K': Ks.tolist(), 'mask_pixels': int(mask.sum()),
            'mask_depth_pixels': int((mask & np.isfinite(z)).sum()), 'photo': frame['input'], 'photo_sha256': frame['sha256'],
            'mask': view['mask_path'], 'mask_sha256': view['sha256']['mask.png'],
            'depth_derivation': 'z of inv(native camera_to_world) @ native pointmap; content_valid, finite confidence>=0.1, z>0; '
                                'nearest canonical pixel per half-resolution pixel centre; NaN = no depth'}
    return rgb, mask, pointmap(z, Ks), meta


def write_outputs(out, r, extra):
    """SAM 3D result -> the generate_lucida_assets.py file set; verified as assemble_lucida_scene.py reads it."""
    import trimesh
    V, F = np.asarray(r['vertices'], np.float32).astype(np.float64), np.asarray(r['faces'], np.int64)
    M = P3D @ np.asarray(r['objectToCamera'], np.float64)
    if not np.isfinite(V).all() or not len(F) or np.linalg.det(M[:3, :3]) <= 0:
        raise ValueError('Invalid SAM 3D mesh or pose')
    colors = np.asarray(r['colors'], np.uint8)
    raw = trimesh.Trimesh(V, F, vertex_colors=colors, process=False)
    posed = trimesh.Trimesh(V @ M[:3, :3].T + M[:3, 3], F, vertex_colors=colors, process=False)
    for mesh, stem in ((raw, 'object'), (posed, 'posed-object')):
        mesh.export(out / f'{stem}.ply')
        mesh.export(out / f'{stem}.glb')
    back = trimesh.load(out / 'object.ply', force='mesh', process=False)
    residual = float(np.abs(back.vertices @ M[:3, :3].T + M[:3, 3] - trimesh.load(out / 'posed-object.ply', force='mesh', process=False).vertices).max())
    if residual > 2e-5 or len(back.faces) != len(F):
        raise ValueError(f'Raw/posed asset disagreement {residual}')
    paths = {'mesh': 'object.ply', 'glb': 'object.glb', 'posed_mesh': 'posed-object.ply', 'posed_glb': 'posed-object.glb'}
    record = {**extra, 'status': 'complete', 'object_to_camera': M.tolist(), 'object_to_camera_pytorch3d': np.asarray(r['objectToCamera']).tolist(),
              'pose_vertex_max_abs_residual': residual, 'vertices': len(V), 'faces': len(F),
              'bounds': [V.min(0).tolist(), V.max(0).tolist()], 'watertight': bool(raw.is_watertight),
              'paths': paths, 'output_sha256': {p: digest(out / p) for p in paths.values()}}
    (out / 'output.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def run_sam3d(args):
    if os.environ.get('PANOPTES_ONPREM') != '1':
        raise SystemExit('sam3d runs inside docker/sam3d.Dockerfile through scripts/onprem/run_stage.py --weights (see --help)')
    import torch
    root = args.root.resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    objects = {o['object_id']: o for o in json.loads((root / 'evidence/objects.json').read_text())['objects']}
    ids = args.object_ids.split(',') if args.object_ids else []
    if not ids or len(set(ids)) != len(ids) or any(not i or set(i) - SAFE or i not in objects for i in ids):
        raise ValueError('Provide explicit unique known --object-ids')
    generation = root / 'generation'
    generation.mkdir(exist_ok=True)
    for oid in ids:
        if (generation / oid).exists():
            raise FileExistsError('Preserve existing generation: ' + str(generation / oid))
    sys.path.insert(0, str(args.workcell / 'modal_apps'))
    import sam3d_research as s3
    model, failed = s3.SAM3DObjects(), []
    for oid in ids:
        out = generation / oid
        rgb, mask, pm, meta = sam3d_inputs(root, objects[oid], manifest)
        out.mkdir(parents=True)
        (out / 'source.json').write_text(json.dumps(meta, indent=2) + '\n')
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        r = model.run.remote(rgb, mask, pm, args.seed)  # in-process under run_stage.py; the first call also loads the weights
        extra = {'schema_version': 1, 'completion': 'sam3d', 'object_id': oid, 'seed': args.seed, 'model_id': 'facebook/sam-3d-objects',
                 'model_revision': s3.MODEL_REVISION, 'code_revision': s3.CODE_REVISION, 'pose_from_model': True, 'metric_scale_known': False,
                 'coordinate_space': 'native object; posed mesh in anchor OpenCV camera coordinates',
                 'texture': 'native generated vertex colors; no texture baking or input photo projection',
                 'licenses': LICENCES, 'anchor_frame': meta['frame_id'], 'source_json_sha256': digest(out / 'source.json'),
                 'inference_seconds': r.get('seconds'), 'call_wall_seconds': time.monotonic() - begin,
                 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'hardware': {'gpu': r.get('gpu'), 'torch': torch.__version__}}
        if 'error' in r:
            (out / 'output.json').write_text(json.dumps({**extra, 'status': 'failed', 'error': r['error'][-3000:]}, indent=2) + '\n')
            failed.append(oid)
            print(oid, 'failed', r['error'][-600:], flush=True)
            continue
        rec = write_outputs(out, r, extra)
        print(oid, 'complete', rec['vertices'], rec['faces'], round(r['seconds'], 1), 's', flush=True)
    if failed:
        raise SystemExit(f'SAM 3D failed for {failed}')


def self_check():
    K = np.array([[200., 0, 63.5], [0, 200, 47.5], [0, 0, 1]])
    z = np.full((96, 128), 2.)
    pm = pointmap(z, K)
    # an object at pixel (u, v) = (90, 30), depth 2: SAM 3D reports its pose in the pointmap's (PyTorch3D) camera
    target = np.linalg.inv(K) @ [90, 30, 1] * 2
    assert np.allclose(P3D[:3, :3] @ target, pm[30, 90])
    pose_cv = np.eye(4)
    pose_cv[:3, :3] = .5 * np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    pose_cv[:3, 3] = target
    r = {'vertices': np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32), 'faces': np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], np.uint32),
         'colors': np.full((4, 3), 128, np.uint8), 'objectToCamera': P3D @ pose_cv}
    with tempfile.TemporaryDirectory() as tmp:
        rec = write_outputs(Path(tmp), r, {'object_id': 'synthetic', 'anchor_frame': 'frame_0001'})
        M = np.asarray(rec['object_to_camera'])
        assert np.allclose(M, pose_cv) and rec['pose_vertex_max_abs_residual'] < 2e-5
        uvw = K @ M[:3, 3]
        assert np.allclose(uvw[:2] / uvw[2], [90, 30])  # the object's origin lands on its pixel in the OpenCV camera
        assert all(digest(Path(tmp) / p) == rec['output_sha256'][p] for p in rec['paths'].values())
    A = np.array([[.25, 0, 3.], [0, .25, -1], [0, 0, 1]])  # full-res -> half-res K keeps pixel centres: (2u + .5, 2v + .5) <-> (u, v)
    Kf = np.linalg.inv(A) @ K
    Ks = np.diag([.5, .5, 1.]) @ Kf
    Ks[:2, 2] = (Kf[:2, 2] - .5) / 2
    p = Kf @ target; q = Ks @ target
    assert np.allclose((p[:2] / p[2] - .5) / 2, q[:2] / q[2])
    print('self-check passed: PyTorch3D pointmap signs, SAM 3D pose -> OpenCV object_to_camera, posed/raw files and hashes as assembly reads them, half-resolution K')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--completion', choices=['sam3d', 'recgen'])
    args, rest = parser.parse_known_args()
    if args.completion == 'recgen':  # everything else is generate_lucida_assets.py's own (its --self-check too)
        import generate_lucida_assets
        sys.argv = [generate_lucida_assets.__file__, *rest]
        return generate_lucida_assets.main()
    if rest == ['--self-check']:
        return self_check()
    if args.completion != 'sam3d':
        parser.error('--completion sam3d|recgen is required')
    sam = argparse.ArgumentParser(prog='generate_sam3d_assets.py --completion sam3d')
    sam.add_argument('--root', type=Path, required=True)
    sam.add_argument('--object-ids', required=True)
    sam.add_argument('--seed', type=int, default=42)
    sam.add_argument('--workcell', type=Path, default=Path(os.environ.get('PANOPTES_WORKCELL', '/workcell')),
                     help='the workcell repository (modal_apps/sam3d_research.py); /workcell in the image')
    run_sam3d(sam.parse_args(rest))


if __name__ == '__main__':
    main()
