"""Validate, evaluate and export the selected MVS geometry on the pipeline machine."""
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
from argus import ROOT
SCR = Path(os.environ['PANOPTES_DATA_ROOT'])
RUNS = Path(os.environ.get('PANOPTES_RUNS', SCR / 'runs'))
GEOM, OUT = SCR / 'checks/bbab-geom', SCR / 'checks/bbab-analyse'

def harness():
    """The fair harness module (local side only): frames_for, geometry_tar, analyse + its app, CELLS, rates."""
    import argus.pipeline.field_evaluator as fair_ab_modal
    return fair_ab_modal

def analyse_local(names):
    """The fair harness's analyse function and app, unchanged; the contract check runs on each geometry first."""
    import argus.pipeline.geometry_contract as bb
    fair = harness(); jobs = []
    for d in sorted(GEOM.glob('*-padded')):
        cell, backbone = d.name.split('-', 1)[0], d.name.split('-', 1)[1].rsplit('-', 1)[0]
        if names and d.name not in names:
            continue
        bb.check_geometry(d / 'geometry', RUNS / fair.CELLS[cell]['run'])
        jobs.append(dict(cell=cell, backbone=backbone, variant='padded', geometry=fair.geometry_tar(d / 'geometry')))
    print('contract passed:', [f"{j['cell']}-{j['backbone']}" for j in jobs])
    OUT.mkdir(parents=True, exist_ok=True); t = time.monotonic(); total = 0.
    for job in jobs:
        r = fair.analyse(job)
        name = f"{job['cell']}-{job['backbone']}-{job['variant']}"
        if isinstance(r, Exception):
            print(name, 'FAILED', repr(r)[:800]); continue
        (OUT / f'{name}.json').write_text(json.dumps(r, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)) + '\n')
        total += r['containerSeconds']; print(name, f"{r['containerSeconds']:.0f}s")
    row = dict(stage='analyse', mode='local CPU', hardware='8 CPU, 16 GiB per job', jobs=len(jobs),
               functionSeconds=total, callSeconds=time.monotonic() - t, estimateUsd=0.0, rateSource='https://modal.com/pricing')
    (OUT / f'spend-ledger-{int(time.time())}.json').write_text(json.dumps(row, indent=2) + '\n'); print(json.dumps(row))

def export_run(backbone: str, cell: str = '090'):
    """The run directory the downstream completion A/B reads (lucida-replica-01 layout), with this backbone's geometry:
    geometry/frames/<id>/ (+ content_valid_mask.npy by prepare_capture_evidence.RULE), evidence/objects/<obj>/<id>/points.npy +
    colors.npy re-derived (mask & content, as prepare_lucida_evidence), objects.json centroids (median), floor.json = the fair
    A/B primary floor (maxInlier) on the frozen floor masks + this backbone's e-stop scale. Image-space files (input photos,
    masks, rgba crops) are symlinked read-only; canonical frames copied. Nothing in the source run is written."""
    import hashlib
    import os
    import shutil
    import numpy as np
    from PIL import Image
    import argus.pipeline.geometry_contract as bb
    fair = harness(); src = RUNS / fair.CELLS[cell]['run']; geom = GEOM / f'{cell}-{backbone}-padded/geometry'
    dst = SCR / f'checks/bbab-export-{cell}-{backbone}'
    if dst.exists():
        raise ValueError(f'{dst} exists')
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    man = json.loads((src / 'manifest.json').read_text()); bb.check_geometry(geom, src)
    (dst / 'evidence').mkdir(parents=True); os.symlink(src / 'input', dst / 'input')
    shutil.copytree(src / 'evidence/canonical', dst / 'evidence/canonical')
    P, content, rgb, records = {}, {}, {}, []
    for f in man['frames']:
        fid = f['frame_id']; g = dst / 'geometry/frames' / fid; g.mkdir(parents=True)
        for name in ('pts3d.npy', 'conf.npy', 'valid_mask.npy', 'intrinsics.npy', 'camera_to_world.npy'):
            shutil.copy(geom / 'frames' / fid / name, g / name)
        shutil.copy(src / f['canonical'], g / 'canonical.png')  # the frozen bytes (check_geometry asserted the same pixels)
        P[fid] = np.load(g / 'pts3d.npy'); conf = np.load(g / 'conf.npy'); valid = np.load(g / 'valid_mask.npy').astype(bool)
        content[fid] = valid & np.load(src / f['alpha']) & np.isfinite(P[fid]).all(-1) & (np.linalg.norm(P[fid], axis=-1) > 1e-6) & (conf >= .1)
        np.save(g / 'content_valid_mask.npy', content[fid]); rgb[fid] = np.asarray(Image.open(g / 'canonical.png').convert('RGB'))
        records.append(dict(frame_id=fid, valid_content_points=int(content[fid].sum()), canonical_pixels=int(content[fid].size),
                            files={p.name: digest(p) for p in sorted(g.iterdir())}))
    shutil.copy(geom / 'candidate_manifest.json', dst / 'geometry/candidate_manifest.json')

    def rederive(v):
        d = dst / Path(v['canonical_mask_path']).parent; d.mkdir(parents=True, exist_ok=True)
        for p in (v['canonical_mask_path'], v['mask_path'], v['rgba_path']):
            os.symlink(src / p, dst / p)
        m = np.load(src / v['canonical_mask_path']).astype(bool) & content[v['frame_id']]
        np.save(dst / v['points_path'], P[v['frame_id']][m]); np.save(dst / v['colors_path'], rgb[v['frame_id']][m])
        v.update(partial_point_count=int(m.sum()), centroid_native=np.median(P[v['frame_id']][m], axis=0).tolist() if m.any() else None,
                 sha256={Path(p).name: digest(dst / p) for p in (v['points_path'], v['colors_path'])})
        return P[v['frame_id']][m], rgb[v['frame_id']][m]
    objs = json.loads((src / 'evidence/objects.json').read_text())
    for o in objs['objects']:
        for v in o['views']:
            rederive(v)
        c = [np.array(v['centroid_native']) for v in o['views'] if v['centroid_native'] is not None]
        o['physical_identity']['pairwise_visible_centroid_distances_native'] = [[float(np.linalg.norm(a - b)) for b in c] for a in c]
        o['physical_identity'].pop('two_view_centroid_distance_native', None)
    objs['coordinate_system'] = f'{backbone} native OpenCV world (geometry-backbone-ab export; not the published Pi3X world)'
    (dst / 'evidence/objects.json').write_text(json.dumps(objs, indent=2) + '\n')
    fl = json.loads((src / 'evidence/floor.json').read_text()); pts, cols = zip(*[rederive(v) for v in fl['views']])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    np.save(dst / 'evidence/floor_points.npy', pts); np.save(dst / 'evidence/floor_colors.npy', cols)
    a = json.loads((OUT / f'{cell}-{backbone}-padded.json').read_text())['floors']['maxInlier']; n, d = np.array(a['normal']), a['offset']
    res = np.abs(pts @ n + d)
    fl.update(plane_native=[*n.tolist(), d], up_native=n.tolist(), coordinate_system=objs['coordinate_system'],
              fit='geometry-licence-ab-fair maxInlier (primary): largest-consensus plane, 20 deg gate, cameras above, two SVD refits',
              distance_threshold_native=None, total_observed_points=len(pts), all_point_residual_median_native=float(np.median(res)),
              all_point_residual_p95_native=float(np.quantile(res, .95)), inlier_points=None, inlier_residual_p95_native=None,
              camera_signed_heights_native=a['cameraHeightsNative'], estopNativeToMeters=a['estop']['nativeToMeters'],
              estopMaxDeviation=a['estop']['maxDeviation'], estopScaleSource='this backbone\'s own e-stop (fair harness); not a published scale')
    (dst / 'evidence/floor.json').write_text(json.dumps(fl, indent=2) + '\n')
    man['geometry'] = dict(status='complete', path='geometry', model=backbone, input_frames=[f['frame_id'] for f in man['frames']],
                           coordinate_system=objs['coordinate_system'], metric_scale_known=False, frames=records,
                           content_valid_rule='native valid & declared alpha & finite & nonzero points & conf>=0.1',
                           candidate=json.loads((geom / 'candidate_manifest.json').read_text()).get('model_id'))
    man['evidence'] = dict(objects='evidence/objects.json', objects_sha256=digest(dst / 'evidence/objects.json'), floor='evidence/floor.json',
                           floor_sha256=digest(dst / 'evidence/floor.json'), exportedFrom=str(src), exportedBy=str(Path(__file__).resolve()))
    man['experiment'] = dst.name
    (dst / 'manifest.json').write_text(json.dumps(man, indent=2) + '\n')
    bb.check_geometry(dst / 'geometry', src)
    print(dst, {o['object_id']: [v['partial_point_count'] for v in o['views']] for o in objs['objects']}, 'floor points', len(pts))

if __name__ == '__main__':
    if sys.argv[1:2] == ['analyse']:
        analyse_local(sys.argv[2:])
    elif sys.argv[1:2] == ['export']:
        export_run(*sys.argv[2:])
    else:
        raise SystemExit('analyse [geometry names] | export BACKBONE CELL')
