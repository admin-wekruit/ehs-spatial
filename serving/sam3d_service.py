"""SAM 3D Objects service — docs/BACKENDS-v1.md `sam3d` (port 8805): one masked photo + our pointmap in, one posed mesh out.

    POST /v1/sam3d        {"input_sha256", "seed", "image_b64" | "image_uri", "mask_b64" | "mask_uri",
                           "pointmap_npz_b64" | "pointmap_npz_uri"}            synchronous (one candidate, ~10 s on an A100)
    ->                    {"mesh_npz_b64": <npz vertices float32 Nx3, faces int32 Mx3, colors uint8 Nx3, object_to_camera_p3d
                           float64 4x4>, "pins": {...}, "vertices": N, "faces": M, "seconds", "gpu", "model_info"}
    POST /v1/sam3d/jobs   same body -> 202 {"job_id", "input_sha256", "cached"}; GET /v1/jobs/{id}
    + /healthz, /v1/info, X-API-Key, errors: serving/v1_common.py

The model is modal_apps/sam3d_research.SAM3DObjects of the ehs-spatial checkout at PANOPTES_WORKCELL (/workcell in the image),
function body unchanged, imported against the on-prem modal stub exactly as scripts/onprem/run_stage.py does (use_stub,
pin_torch_hub, link_weights: no modal client, no torch.hub download, weights from PANOPTES_WEIGHTS_DIR). Weights are SHA-256
checked against serving/registry.yaml before the model loads. PANOPTES_FAKE_MODEL=1: a deterministic cube, no torch.

Run:  cd serving && uvicorn sam3d_service:app --host 0.0.0.0 --port 8805
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from pydantic import BaseModel

import v1_common as v1

service = v1.Service('sam3d', loaded=False)
app = service.app


class Sam3dRequest(BaseModel):
    input_sha256: str
    seed: int = 42
    image_b64: str | None = None
    image_uri: str | None = None
    mask_b64: str | None = None
    mask_uri: str | None = None
    pointmap_npz_b64: str | None = None
    pointmap_npz_uri: str | None = None


# ---------------------------------------------------------------- the model
def load_model():
    """SAM3DObjects through the run_stage stub; returns the stub-wrapped instance, loaded now (not on the first request).
    PANOPTES_WEIGHTS_MODE=manifest (default): the fetch_weights_sam3d.py cache at WEIGHTS, SHA-256 verified against registry.yaml,
    linked into HF_HOME / torch hub by run_stage. PANOPTES_WEIGHTS_MODE=hf-cache: HF_HOME and the torch hub checkpoint directory are
    already in place (a baked image, a pre-mounted cache, a GPU host that already serves the other models); nothing is linked or
    verified here, the model's own snapshot_download(revision=...) still pins the revision."""
    mode = os.environ.get('PANOPTES_WEIGHTS_MODE', 'manifest')
    if mode == 'manifest':
        v1.verify_weights(service.entry)  # registry.yaml pins; a mismatch refuses to start
    workcell = Path(os.environ.get('PANOPTES_WORKCELL', '/workcell'))
    sys.path.insert(0, str(workcell / 'scripts/onprem'))
    import run_stage  # the on-prem runner: modal stub, torch.hub pin, weights overlay
    run_stage.use_stub()
    os.environ.update(HF_HUB_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', PANOPTES_ONPREM='1')
    if mode == 'manifest':
        run_stage.pin_torch_hub()
        run_stage.link_weights(v1.weights_dir())  # HF_HOME -> the cache, /opt/torch-hub/hub/checkpoints -> the DINOv2 file
    elif mode != 'hf-cache':
        raise SystemExit(f'PANOPTES_WEIGHTS_MODE={mode!r}: manifest | hf-cache')
    sys.path.insert(0, str(workcell / 'modal_apps'))
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # registry seed_policy
    import sam3d_research
    model = sam3d_research.SAM3DObjects()
    model.load  # noqa: B018  the stub runs the @enter hook (the real load) on the first attribute access: eager, so /healthz is honest
    return model


def fake_run(rgb, mask, pointmap, seed, input_sha256) -> dict:
    """A unit cube at the mask's median depth: deterministic, shaped like the real result, with the request's sha in pins."""
    z = float(np.nanmedian(pointmap[mask][:, 2])) if mask.any() and np.isfinite(pointmap[mask][:, 2]).any() else 1.0
    v = np.array([[x, y, w] for x in (-.5, .5) for y in (-.5, .5) for w in (-.5, .5)], np.float32)
    faces = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.uint32)
    colors = np.tile(np.array([[200, 60, 40]], np.uint8), (8, 1))
    pose = np.eye(4); pose[2, 3] = z
    return dict(vertices=v, faces=faces, colors=colors, objectToCamera=pose, gpu='fake', seconds=0.,
                pins={'model': 'fake', 'input_sha256': input_sha256, 'seed': int(seed)})


MODEL = None if v1.FAKE else load_model()
service.loaded = True


# ---------------------------------------------------------------- the call
def decode(body: dict) -> tuple:
    try:
        rgb = np.asarray(Image.open(io.BytesIO(v1.read_binary(body, 'image'))).convert('RGB'))
        mask = np.asarray(Image.open(io.BytesIO(v1.read_binary(body, 'mask'))).convert('L')) > 127
    except (OSError, ValueError) as error:
        raise v1.bad_input(f'image / mask: not a decodable image ({error})') from None
    arrays = v1.load_npz(v1.read_binary(body, 'pointmap_npz'), 'pointmap_npz')
    if 'pointmap' not in arrays:
        raise v1.bad_input("pointmap_npz: no 'pointmap' array")
    pointmap = np.asarray(arrays['pointmap'], np.float32)
    if rgb.shape[:2] != mask.shape or pointmap.shape != rgb.shape[:2] + (3,):
        raise v1.bad_input(f'image {rgb.shape[:2]}, mask {mask.shape} and pointmap {pointmap.shape} grids differ')
    return rgb, mask, pointmap


def run(rgb, mask, pointmap, seed: int, sha: str) -> dict:
    out = fake_run(rgb, mask, pointmap, seed, sha) if MODEL is None else MODEL.run.remote(rgb, mask, pointmap, seed)
    if 'error' in out:  # SAM3DObjects.run returns the traceback as data
        raise v1.model_error(out['error'][-2000:])
    mesh = v1.npz(vertices=np.asarray(out['vertices'], np.float32), faces=np.asarray(out['faces'], np.int32),
                  colors=np.asarray(out['colors'], np.uint8), object_to_camera_p3d=np.asarray(out['objectToCamera'], np.float64))
    return dict(mesh_npz_b64=base64.b64encode(mesh).decode('ascii'), pins=out['pins'],
                vertices=int(len(out['vertices'])), faces=int(len(out['faces'])))


def submit(request: Sam3dRequest):
    """Inputs are decoded (and s3:// ones read) in the request, so bad_input is a 400 now, not an errored job."""
    body = request.model_dump()
    rgb, mask, pointmap = decode(body)
    return service.submit(body['input_sha256'], lambda: run(rgb, mask, pointmap, int(body['seed']), body['input_sha256']))


@app.post('/v1/sam3d')
def sam3d(request: Sam3dRequest) -> dict:
    job, cached = submit(request)
    return job.result if cached else service.wait(job)


@app.post('/v1/sam3d/jobs')
def sam3d_job(request: Sam3dRequest):
    return service.accepted(*submit(request))
