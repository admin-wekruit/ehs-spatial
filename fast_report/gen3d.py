"""r5 (models): pose-free image-to-3D generators for the model bench, each resident in its own venv and process
(fast_report.sam3d.Pool). Their meshes come in their own canonical frame (z up, unit-ish size) and are never posed here: the
bench aligns them to the object's observed points and gates them on a held-out view. Commercial-licence candidates only:
  TRELLIS-image-large  MIT code and weights (microsoft/TRELLIS @ TRELLIS_CODE; the weights on panoptes-lucida-weights, its
                       mesh-only pipeline). Only the mesh decoder runs (its FlexiCubes fork: Apache-2.0); nvdiffrast and the
                       Gaussian rasterisers (NVIDIA / Inria, non-commercial) are never imported, colours come from our frames.
                       Several views: run_multi_image (stochastic).
  TripoSR              MIT (VAST-AI-Research/TripoSR @ TRIPOSR_CODE, stabilityai/TripoSR weights, facebook/dino-vitb16:
                       Apache-2.0); marching cubes by PyMCubes (BSD) in place of torchmcubes; vertex colours from its colour field.
Messages: {"views": [RGBA crops (uint8, mask as alpha)], "seed"} -> {"vertices", "faces", "colors" or None, "seconds"}.

    python -m fast_report.gen3d --self-check    # CPU: the crop recipe and the marching-cubes stand-in
"""
import os
import sys
import time
import types

import numpy as np

TRELLIS_PY, TRELLIS_DIR, TRELLIS_CODE = "/opt/trellis-venv/bin/python", "/opt/trellis", "442aa1e1afb9014e80681d3bf604e8d728a86ee7"
TRELLIS_PIPE, DINO_WEIGHTS, DINO_DIR = "/cache/weights/trellis-mesh", "/cache/weights/dinov2_vitl14_reg4_pretrain.pth", "/opt/dinov2"
TRELLIS_STEPS = 12  # sparse structure and latent flow steps each (default 25)
TRIPOSR_PY, TRIPOSR_DIR, TRIPOSR_CODE = "/opt/triposr-venv/bin/python", "/opt/triposr", "107cefdc244c39106fa830359024f6a2f1c78871"
TRIPOSR_WEIGHTS, HF_HOME, TRIPOSR_MC = "/v/r5/hf/triposr", "/v/r5/hf", 192
CU121 = "https://download.pytorch.org/whl/cu121"


def cluster(V, F, cells=128):
    """Vertex clustering on a cells^3 grid over the mesh's longest side (numpy): TRELLIS returns ~2.2 M faces at 256^3, a 504x280 view
    resolves far less, and every face crosses a pipe; ~0.2 s. -> (vertices, faces)."""
    V, F = np.asarray(V, np.float64), np.asarray(F, np.int64)
    if len(F) == 0:
        return V.astype(np.float32), F
    lo = V.min(0)
    q = np.floor((V - lo) / (max(float(np.ptp(V, 0).max()), 1e-9) / cells)).astype(np.int64)
    _, inv, cnt = np.unique((q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2], return_inverse=True, return_counts=True)
    inv = inv.ravel()
    Vc = np.stack([np.bincount(inv, V[:, j], len(cnt)) for j in range(3)], 1) / cnt[:, None]
    Fc = inv[F]
    Fc = Fc[(Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])]
    keep = np.sort(np.unique(np.sort(Fc, 1), axis=0, return_index=True)[1])  # one face per vertex triple
    return Vc.astype(np.float32), Fc[keep]


def rgba_crop(rgb, mask, side=518, pad=1.2):
    """Square crop around the mask (pad x its longer side), mask as alpha, resized: the generators' input."""
    import cv2
    ys, xs = np.nonzero(mask)
    cx, cy, half = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, max(xs.max() - xs.min(), ys.max() - ys.min(), 8) * pad / 2
    x0, y0, s = int(np.floor(cx - half)), int(np.floor(cy - half)), int(np.ceil(2 * half))
    out = np.zeros((s, s, 4), np.uint8)
    a0, b0, a1, b1 = max(y0, 0), max(x0, 0), min(y0 + s, rgb.shape[0]), min(x0 + s, rgb.shape[1])
    out[a0 - y0:a1 - y0, b0 - x0:b1 - x0, :3] = rgb[a0:a1, b0:b1]
    out[a0 - y0:a1 - y0, b0 - x0:b1 - x0, 3] = mask[a0:a1, b0:b1].astype(np.uint8) * 255
    return cv2.resize(out, (side, side), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- TRELLIS
def _stubs():
    """Modules TRELLIS imports but never needs on the mesh path: rembg (preprocess_image only calls it without an alpha channel),
    kaolin (FlexiCubes' input shape assertions, check_tensor)."""
    sys.modules.setdefault("rembg", types.ModuleType("rembg"))
    for name in ("kaolin", "kaolin.utils", "kaolin.utils.testing"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["kaolin.utils.testing"].check_tensor = lambda *a, **k: True


def load_trellis():
    import torch
    _stubs()
    for name in ("trellis", "trellis.pipelines"):  # their __init__s import the renderers and the text pipeline: never loaded
        m = types.ModuleType(name)
        m.__path__ = [os.path.join(TRELLIS_DIR, *name.split("."))]
        sys.modules[name] = m
    original = torch.hub.load

    def dino(repo, model, *a, **k):
        assert repo == "facebookresearch/dinov2" and model == "dinov2_vitl14_reg", (repo, model)
        net = original(DINO_DIR, model, source="local", pretrained=False)
        net.load_state_dict(torch.load(DINO_WEIGHTS, map_location="cpu"))
        return net
    torch.hub.load = dino
    try:
        from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline
        pipe = TrellisImageTo3DPipeline.from_pretrained(TRELLIS_PIPE)
    finally:
        torch.hub.load = original
    pipe.cuda()
    return pipe


def run_trellis(pipe, views, seed, steps=TRELLIS_STEPS):
    import torch
    from PIL import Image
    t = time.monotonic()
    images = [Image.fromarray(v, "RGBA") for v in views]
    kw = dict(seed=seed, formats=["mesh"], sparse_structure_sampler_params={"steps": steps}, slat_sampler_params={"steps": steps})
    out = pipe.run(images[0], **kw) if len(images) == 1 else pipe.run_multi_image(images, mode="stochastic", **kw)
    torch.cuda.synchronize()
    m = out["mesh"][0]
    V, F = cluster(m.vertices.float().cpu().numpy(), m.faces.cpu().numpy())
    return {"vertices": V, "faces": F, "colors": None, "views": len(images), "seconds": time.monotonic() - t, "faces_raw": int(len(m.faces))}


# ---------------------------------------------------------------- TripoSR
def _mcubes_stand_in():
    """torchmcubes.marching_cubes(level, threshold) -> (vertices, faces) torch tensors, by PyMCubes on the CPU."""
    import mcubes
    import torch
    mod = types.ModuleType("torchmcubes")

    def marching_cubes(level, threshold):
        v, f = mcubes.marching_cubes(level.detach().float().cpu().numpy(), float(threshold))
        return torch.from_numpy(v.astype(np.float32)), torch.from_numpy(f.astype(np.int64))
    mod.marching_cubes = marching_cubes
    return mod


def load_triposr():
    sys.modules.setdefault("torchmcubes", _mcubes_stand_in())
    sys.modules.setdefault("rembg", types.ModuleType("rembg"))
    sys.path.insert(0, TRIPOSR_DIR)
    from tsr.system import TSR
    model = TSR.from_pretrained(TRIPOSR_WEIGHTS, config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(131072)
    return model.to("cuda")


def triposr_input(rgba, ratio=.85, side=512):
    """TripoSR's run.py recipe on an RGBA crop: the foreground resized to `ratio` of the square, on 50 % grey."""
    import cv2
    a = rgba[..., 3] > 0
    ys, xs = np.nonzero(a)
    fg = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    s = int(max(fg.shape[:2]) / ratio)
    canvas = np.zeros((s, s, 4), np.uint8)
    y0, x0 = (s - fg.shape[0]) // 2, (s - fg.shape[1]) // 2
    canvas[y0:y0 + fg.shape[0], x0:x0 + fg.shape[1]] = fg
    canvas = cv2.resize(canvas, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    return (canvas[..., :3] * canvas[..., 3:] + (1 - canvas[..., 3:]) * .5) * 255


def run_triposr(model, views, seed):
    import torch
    from PIL import Image
    t = time.monotonic()
    torch.manual_seed(seed)
    img = Image.fromarray(triposr_input(views[0]).astype(np.uint8))
    with torch.no_grad():
        codes = model([img], device="cuda")
    mesh = model.extract_mesh(codes, True, resolution=TRIPOSR_MC)[0]
    torch.cuda.synchronize()
    return {"vertices": np.asarray(mesh.vertices, np.float32), "faces": np.asarray(mesh.faces, np.int64),
            "colors": (np.asarray(mesh.visual.vertex_colors)[:, :3]).astype(np.uint8), "views": 1, "seconds": time.monotonic() - t}


# ---------------------------------------------------------------- resident processes
def worker(kind):
    import torch
    from fast_report.sam3d import serve
    load, run = {"trellis": (load_trellis, run_trellis), "triposr": (load_triposr, run_triposr)}[kind]
    state = {}

    def boot():
        t = time.time()
        state["m"] = load()
        loaded = time.time() - t
        t, err = time.time(), None
        rgb = np.full((300, 300, 3), 90, np.uint8)
        rgb[80:220, 100:200] = (200, 60, 40)
        mask = np.zeros((300, 300), bool)
        mask[80:220, 100:200] = True
        try:  # kernels and allocator at the real shapes; a synthetic failure is recorded, not fatal
            run(state["m"], [rgba_crop(rgb, mask)] * (2 if kind == "trellis" else 1), 42)
        except Exception as e:  # noqa: BLE001
            err = repr(e)[-500:]
        warm = torch.cuda.max_memory_reserved()
        torch.cuda.empty_cache()
        return {"ready": True, "load_s": round(loaded, 1), "warm_s": round(time.time() - t, 1), "warm_error": err,
                "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "reserved_after_warm_gib": round(warm / 2 ** 30, 2)}

    def handle(message, _):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        out = run(state["m"], message["views"], message.get("seed", 42))
        return {**out, "start_unix": start, "end_unix": time.time(), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "max_reserved_gib": round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)}
    serve(handle, boot)


def with_generators(image):
    """Both venvs on RecGen's TRELLIS-era lock (Python 3.10, torch 2.4.0 cu121, xformers 0.0.27.post2, spconv-cu120 2.3.6), code at
    its pinned commits; TripoSR's weights and its DINO tokenizer go to the panoptes-r5-models volume (setup())."""
    tv, uv = "uv pip install --python " + TRELLIS_PY, "uv pip install --python " + TRIPOSR_PY
    return image.run_commands(
        "uv venv --python 3.10 /opt/trellis-venv",
        f"{tv} torch==2.4.0 torchvision==0.19.0 --index-url {CU121}",
        f"{tv} numpy==1.26.4 pillow==11.3.0 scipy==1.15.3 easydict==1.13 tqdm==4.67.1 safetensors==0.6.2 huggingface_hub==0.36.0 "
        "spconv-cu120==2.3.6 trimesh==4.7.4 plyfile==1.1.2 einops==0.8.1 opencv-python-headless==4.11.0.86",
        f"{tv} xformers==0.0.27.post2 --index-url {CU121}",
        f"{tv} git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8",
        f"git clone https://github.com/microsoft/TRELLIS.git {TRELLIS_DIR} && git -C {TRELLIS_DIR} checkout --detach {TRELLIS_CODE} "
        f"&& git -C {TRELLIS_DIR} submodule update --init trellis/representations/mesh/flexicubes",
        f"test -d {DINO_DIR} || (git clone https://github.com/facebookresearch/dinov2.git {DINO_DIR} && git -C {DINO_DIR} checkout --detach 7764ea0f912e53c92e82eb78a2a1631e92725fc8)",
        "uv venv --python 3.10 /opt/triposr-venv",
        f"{uv} torch==2.4.0 torchvision==0.19.0 --index-url {CU121}",
        f"{uv} numpy==1.26.4 omegaconf==2.3.0 pillow==10.4.0 einops==0.7.0 transformers==4.35.0 trimesh==4.0.5 huggingface_hub==0.17.3 "
        "imageio==2.34.2 pymcubes==0.1.6 opencv-python-headless==4.11.0.86",
        f"git clone https://github.com/VAST-AI-Research/TripoSR.git {TRIPOSR_DIR} && git -C {TRIPOSR_DIR} checkout --detach {TRIPOSR_CODE}",
    ).env({"ATTN_BACKEND": "xformers", "SPCONV_ALGO": "native"})


def self_check():
    rgb = np.zeros((720, 1280, 3), np.uint8)
    rgb[300:400, 600:700] = 200
    mask = np.zeros((720, 1280), bool)
    mask[300:400, 600:700] = True
    c = rgba_crop(rgb, mask)
    assert c.shape == (518, 518, 4) and 0.6 < (c[..., 3] > 127).mean() / (1 / 1.2 ** 2) < 1.1, (c[..., 3] > 127).mean()
    t = triposr_input(c)
    assert t.shape == (512, 512, 3) and abs(t[0, 0, 0] - 127.5) < 1 and t[256, 256, 0] > 190
    th, ph = np.meshgrid(np.linspace(0, np.pi, 200), np.linspace(0, 2 * np.pi, 400), indexing="ij")
    V = np.c_[np.sin(th).ravel() * np.cos(ph).ravel(), np.sin(th).ravel() * np.sin(ph).ravel(), np.cos(th).ravel()]
    i = np.arange(199 * 399).reshape(199, 399) + np.arange(199)[:, None]
    F = np.r_[np.c_[i.ravel(), i.ravel() + 1, i.ravel() + 400], np.c_[i.ravel() + 1, i.ravel() + 401, i.ravel() + 400]]
    Vc, Fc = cluster(V, F, cells=32)
    assert len(Fc) < len(F) / 10 and abs(np.linalg.norm(Vc, axis=1).mean() - 1) < .05, (len(Fc), len(F))
    try:
        import mcubes  # noqa: F401
    except ImportError:
        print("gen3d self-check ok: crops (PyMCubes not installed here: the marching-cubes stand-in is checked in the container)")
        return
    import torch
    g = np.linalg.norm(np.mgrid[:20, :20, :20].T - 9.5, axis=-1) - 6
    Vc, Fc = cluster(*[np.asarray(x) for x in _mcubes_stand_in().marching_cubes(torch.tensor(g), 0.)], cells=8)
    assert 0 < len(Fc) < 2000 and abs(np.linalg.norm(Vc - 9.5, axis=1).mean() - 6) < .8
    v, f = _mcubes_stand_in().marching_cubes(torch.tensor(g), 0.)
    assert len(f) > 100 and abs(np.linalg.norm(v.numpy() - 9.5, axis=1).mean() - 6) < .3
    print("gen3d self-check ok: crops, TripoSR input, marching-cubes stand-in")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
