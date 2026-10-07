"""The GPU stages of serving/geometry_mvs_service.py, each run as a subprocess in the venv of its pinned on-prem image
(deploy/Dockerfile.geometry: /opt/geometry for RoMa + the numpy bundle adjustment, /opt/moge for MoGe-3; DA3-BASE runs
scripts/candidate_geometry_backend.py in /opt/da3 directly). Function bodies = ehs-spatial modal_apps/geometry_clean_ab.py
(roma_model, roma_pair, refine) and moge3_app.py, imported unchanged against the on-prem modal stub (scripts/onprem/run_stage.py).

  geometry_mvs_stage.py roma-refine --workcell W --weights DIR --start GEOM --canonicals f1.png ... --alphas a1.npy ...
        --frame-ids frame_0001 ... --x0 63 --x1 455 --out-gpu checks/clean-gpu/CELL --out-geom checks/clean-geom/CELL-da3-base-ba-f/geometry
        [--device auto|cuda|cpu]
      RoMa v1 outdoor on the content crops [x0:x1] of every frame pair (roma-I-J.npz: uvA, uvB, certainty on the 518 grid;
      dense-I-J.npz: uvAB, certA, uvBA, certB = the full warp at every crop pixel, fp16 autocast on CUDA), then
      geometry_clean_ab.refine(focal=True) from the DA3-BASE start geometry -> the BA-f frames (run_stage.py geometry's first
      two steps; its MVS triangulation stays on the CPU route, mvs_route.py).
  geometry_mvs_stage.py moge --weights DIR --model Ruicheng/moge-3-vitl --revision REV --crops c1.png ... --names frame_0001 ... --out DIR
      MoGe-3 per content crop: moge-NAME.npz (points, mask, intrinsicsNormalised), as geometry_clean_ab.moge.
"""
from __future__ import annotations

import argparse
import hashlib
import io
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time

FRAME_FILES = ('pts3d', 'conf', 'valid_mask', 'intrinsics', 'camera_to_world')


def npz(**arrays) -> bytes:
    import numpy as np
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def roma_refine(a) -> dict:
    sys.path.insert(0, str(Path(a.workcell) / 'scripts/onprem'))
    import run_stage
    run_stage.use_stub()
    run_stage.pin_torch_hub()
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PANOPTES_ONPREM='1')
    sys.path[:0] = [str(Path(a.workcell) / 'modal_apps')]
    import numpy as np
    from PIL import Image
    import torch
    import fetch_weights_geometry as fwg
    import geometry_clean_ab as gc
    t0 = time.monotonic()
    frames = gc.read_geometry(Path(a.start))
    if len(frames) != len(a.canonicals) or len(a.alphas) != len(a.canonicals) or len(a.frame_ids) != len(a.canonicals):
        raise SystemExit('roma-refine: frames, canonicals, alphas and frame ids differ in count')
    H, W = frames[0]['pts3d'].shape[:2]
    x0, x1 = a.x0, a.x1
    contents = [fr['valid'] & np.load(p).astype(bool) & np.isfinite(fr['pts3d']).all(-1) & (np.linalg.norm(fr['pts3d'], axis=-1) > 1e-6) & (fr['conf'] >= .1)
                for fr, p in zip(frames, a.alphas)]  # prepare_capture_evidence.RULE (fair_ab.content_mask), as run_stage.py geometry
    device = a.device if a.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    model = gc.roma_model(device, a.weights)  # both RoMa files SHA-256 checked against the pins first
    t_load = time.monotonic() - t0
    crops = [Image.open(p).convert('RGB').crop((x0, 0, x1, H)) for p in a.canonicals]
    out_gpu = Path(a.out_gpu); out_gpu.mkdir(parents=True, exist_ok=True)
    matches, written = {}, {}
    for i, j in combinations(range(len(crops)), 2):
        r = gc.roma_pair(model, crops[i], crops[j], device, x0=x0)
        matches[(i, j)] = (r['sparse']['uvA'], r['sparse']['uvB'])
        for name, arrays in ((f'roma-{i}-{j}.npz', r['sparse']), (f'dense-{i}-{j}.npz', r['dense'])):
            data = npz(**arrays); (out_gpu / name).write_bytes(data); written[name] = hashlib.sha256(data).hexdigest()
    t_roma = time.monotonic() - t0 - t_load
    baf, rep = gc.refine(frames, contents, matches, focal=True, size=(H, W))
    if not rep['usable']:
        raise SystemExit('roma-refine: bundle adjustment not usable, nothing written: ' + json.dumps([p['report'] for p in rep['ba']]))
    out_geom = Path(a.out_geom)
    for fr, fid, png in zip(baf, a.frame_ids, a.canonicals):
        d = out_geom / 'frames' / fid; d.mkdir(parents=True, exist_ok=True)
        for name, v in zip(FRAME_FILES, (fr['pts3d'], fr['conf'], fr['valid'], fr['K'], fr['c2w'])):
            np.save(d / f'{name}.npy', v)
        (d / 'canonical.png').write_bytes(Path(png).read_bytes())
    start_manifest = Path(a.start) / 'candidate_manifest.json'
    summary = dict(base='da3-base', init=str(a.start), startModel=json.loads(start_manifest.read_text()).get('model_id') if start_manifest.exists() else None,
                   refinement='ba-f', ba=rep['ba_backend'], baConverged=rep['converged'], baUsable=rep['usable'],
                   roma=dict(device=device, code=fwg.ROMA_CODE, torch=str(torch.__version__), autocast='fp16' if device == 'cuda' else 'fp32 (romatch disables autocast off CUDA)',
                             weightsSha256={rel: f[1] for n in ('roma_outdoor', 'roma_dinov2') for rel, f in fwg.FILES[n]['files'].items()}, outputsSha256=written),
                   contentRect=[x0, 0, x1, H], gpu=torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu',
                   seconds=dict(romaLoad=round(t_load, 1), roma=round(t_roma, 1), refine=round(time.monotonic() - t0 - t_load - t_roma, 1)),
                   created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    (out_geom / 'candidate_manifest.json').write_text(json.dumps(dict(summary, report=rep), indent=1, default=float) + '\n')
    return summary


def moge(a) -> dict:
    os.environ.update(HF_HOME=str(Path(a.weights) / 'hf'), HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1')
    import numpy as np
    from PIL import Image
    import torch
    from moge.model.v3 import MoGeModel
    t0 = time.monotonic()
    torch.manual_seed(0)
    model = MoGeModel.from_pretrained(a.model, revision=a.revision).to('cuda').eval()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name, crop in zip(a.names, a.crops):
        im = np.asarray(Image.open(crop).convert('RGB'))
        with torch.inference_mode():
            o = model.infer(torch.from_numpy(im.copy()).float().permute(2, 0, 1).cuda() / 255, use_fp16=True)
        (out / f'moge-{name}.npz').write_bytes(npz(points=o['points'].float().cpu().numpy(), mask=o['mask'].cpu().numpy().astype(bool),
                                                   intrinsicsNormalised=o['intrinsics'].float().cpu().numpy()))
    return dict(model=a.model, revision=a.revision, gpu=torch.cuda.get_device_name(0), frames=len(a.names), seconds=round(time.monotonic() - t0, 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='stage', required=True)
    r = sub.add_parser('roma-refine')
    r.add_argument('--workcell', required=True); r.add_argument('--weights', required=True); r.add_argument('--start', required=True)
    r.add_argument('--canonicals', nargs='+', required=True); r.add_argument('--alphas', nargs='+', required=True)
    r.add_argument('--frame-ids', nargs='+', required=True); r.add_argument('--x0', type=int, required=True); r.add_argument('--x1', type=int, required=True)
    r.add_argument('--out-gpu', required=True); r.add_argument('--out-geom', required=True); r.add_argument('--device', default='auto')
    m = sub.add_parser('moge')
    m.add_argument('--weights', required=True); m.add_argument('--model', required=True); m.add_argument('--revision', required=True)
    m.add_argument('--crops', nargs='+', required=True); m.add_argument('--names', nargs='+', required=True); m.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    print(a.stage, json.dumps({'roma-refine': roma_refine, 'moge': moge}[a.stage](a), default=str))


if __name__ == '__main__':
    main()
