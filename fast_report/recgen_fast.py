"""Fast RecGen (recgen/fast): where one RecGen call spends its time, and settings that cut it, on X7's saved inputs.

RecGen = TRELLIS-image-large fine-tuned (TRI non-commercial licence: internal use only). One call, as
recgen_inference.generate / generate_multiview make it (X7's x7.run_recgen):
  preprocess   crops -> 518^2 image, normalised pointmap, eroded mask (CPU)
  cond_ss      DINOv2 ViT-L + point/mask embedders, per (anchor, view_i) pair
  ss_flow      dense 16^3 transformer + pose tokens, Euler steps; per step one forward per pair, plus one negative forward per
               pair inside the CFG interval (N pairs: 2N forwards)
  ss_decode    occupancy decoder -> 64^3 voxels
  cond_slat    the anchor + first second view (RecGen's default SLAT regime)
  slat_flow    sparse 64^3 transformer, Euler steps with CFG
  mesh_decode  sparse decoder -> FlexiCubes 256^3 -> mesh with vertex colours
  gs_decode    Gaussian decoder (RecGen always runs it; the mesh does not need it)
  mesh_out     the posed trimesh with vertex colours (GPU -> CPU, pose, camera frame)
run() makes exactly those calls with CUDA-synced stage timers and a setting (DEFAULT = RecGen as X7 calls it):
  ss_steps / slat_steps   Euler steps (None: pipeline.json)
  ss_cfg / slat_cfg       CFG strength (None: pipeline.json; 0: off, the negative forward never runs)
  formats                 decoders: ("mesh", "gaussian") as RecGen, ("mesh",) for mesh + vertex colours only
  fusion                  multidiffusion    RecGen's own pair fusion (above)
                          stochastic        RecGen's own round robin: one pair per step
                          fused             multidiffusion's maths in ONE batched forward per step: the N pair conditions and
                                            ONE negative (zeros, same x_t, pose and t for every pair, so the N negatives RecGen
                                            computes are one), 2N forwards -> N + 1 rows; single-condition steps (1-2 views,
                                            SLAT) batch condition + negative
                          fused_stochastic  round robin, batched with the negative
  max_views               the first n generation views (anchor first)
  cache_cond              DINOv2 features of each image once per call (RecGen encodes the anchor again for every pair and
                          for SLAT; exact: a copy of the same tensor)
  fast_out                the posed mesh straight from the decoder's tensors (the pose applied on the GPU) instead of RecGen's
                          trimesh round trip (which merges duplicate vertices on 0.3-1 M faces: 0.2-1.3 s); the same surface
Process level (worker env): ATTN_BACKEND xformers | sdpa | flash_attn, SPARSE_ATTN_BACKEND, COMPILE_SS=1 (torch.compile of
the SS flow model's forward; needs triton + setuptools in the venv: X7's venv lacks setuptools, so triton cannot import).

FAST is the recommendation of runs/recgen-fast-002..005 (19 X7 objects, 3 videos, one A100-80GB): the SS stage keeps CFG at 12
steps, SLAT 8 steps without CFG, plus the lossless items (fused, mesh only, cache_cond, fast_out). 2.53-2.57 s/object vs 8.20-8.38
(3.3x, PCIe); SXM4 X7 view mix 1.96 s; held-out IoU -0.010 +/- 0.010 against the default over 3 seeds each, acceptance 9.3 vs
10.0 of 19, Chamfer to the default 1.1-2.0 % of the diagonal (another default seed: 1.8 %). COMPILE_SS=1 adds 1.17x (208 s compile
at process start).

judge() is X7's gate_model on one mesh (anchor camera -> world, light copy, bounded refine on X7's generation views, the
held-out gate unchanged) plus what the bench needs: the held-out tile always, and the Chamfer distance to a reference mesh
(the default arm's, surface samples in world metres, before placement so it measures the generator only).

  python -m fast_report.recgen_fast --self-check      # CPU: settings, surface sampling, Chamfer
"""
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "scripts", ROOT / "modal_apps"):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

DEFAULT = {"ss_steps": None, "ss_cfg": None, "slat_steps": None, "slat_cfg": None, "formats": ("mesh", "gaussian"),
           "fusion": "multidiffusion", "max_views": 4, "cache_cond": False, "fast_out": False}
FUSIONS = ("multidiffusion", "stochastic", "fused", "fused_stochastic")
FAST = {"ss_steps": 12, "slat_steps": 8, "slat_cfg": 0, "formats": ("mesh",), "fusion": "fused", "cache_cond": True, "fast_out": True}
SAMPLES = 20000  # surface points per mesh for the Chamfer distance
GATE_KEYS = ("accepted_source_consistency", "silhouette_iou", "relative_depth_median", "relative_depth_p95", "supported_pixels",
             "min_supported_pixels")


def setting(**change):
    unknown = set(change) - set(DEFAULT)
    assert not unknown, f"unknown setting keys {unknown}"
    s = {**DEFAULT, **change}
    assert s["fusion"] in FUSIONS and set(s["formats"]) <= {"mesh", "gaussian"} and "mesh" in s["formats"], s
    return s


def sampler_params(steps, cfg):
    """Overrides of the pipeline.json sampler params; cfg 0 empties the guidance interval (no negative forward at all)."""
    out = {} if steps is None else {"steps": int(steps)}
    if cfg is not None:
        out["cfg_strength"] = float(cfg)
        if cfg == 0:
            out["cfg_interval"] = (2., 2.)
    return out


# ---------------------------------------------------------------- GPU side (/opt/recgen-venv)
class Stages:
    """CUDA-synced seconds per stage name, summed when a stage runs more than once in a call."""

    def __init__(self):
        self.s = {}

    @contextmanager
    def __call__(self, name):
        import torch
        torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.s[name] = self.s.get(name, 0.) + time.perf_counter() - t


def instrument(pipeline):
    """Timers around the pipeline's stages, once per process; pipeline._stages collects the current call's seconds."""
    pipeline._stages = Stages()

    def wrap(obj, attr, name):
        fn = getattr(obj, attr)

        def timed(*a, **k):
            with pipeline._stages(name(k) if callable(name) else name):
                return fn(*a, **k)
        setattr(obj, attr, timed)
    cond = lambda k: "cond_ss" if "sparse" in k.get("model_key", "") else "cond_slat"  # noqa: E731
    wrap(pipeline, "get_cond", cond)
    wrap(pipeline, "get_cond_multiview", cond)
    wrap(pipeline, "sample_sparse_structure_pose", "ss_flow")  # includes ss_decode: subtracted in run()
    wrap(pipeline.models["sparse_structure_decoder"], "forward", "ss_decode")
    wrap(pipeline, "sample_slat", "slat_flow")
    for key, name in (("slat_decoder_mesh", "mesh_decode"), ("slat_decoder_gs", "gs_decode"), ("slat_decoder_rf", "rf_decode")):
        if key in pipeline.models:
            wrap(pipeline.models[key], "forward", name)


def dense_step(conds=None, stochastic=False):
    """The SS sampler's _inference_model as ONE forward (module docstring 'fused'): conds = the pair conditions (None: the one
    the sampler passes); inside the CFG interval one shared negative row. Same maths as RecGen's per-pair CFG average."""
    import torch
    step = [0]

    def infer(model, x_t, t, cond, neg_cond=None, cfg_strength=0., cfg_interval=(0., 1.), pose=None, **_):
        cs = [cond] if conds is None else list(conds)
        if stochastic:
            cs = [cs[step[0] % len(cs)]]
            step[0] += 1
        guided = neg_cond is not None and cfg_strength > 0 and cfg_interval[0] <= t <= cfg_interval[1]
        rows = torch.cat(cs + [neg_cond] * guided)
        b, n = len(rows), len(cs)
        rep = lambda z: z.expand(b, *z.shape[1:]).contiguous()  # noqa: E731
        tt = torch.full((b,), 1000. * t, device=x_t.device, dtype=torch.float32)
        v, pv = (model(rep(x_t), tt, rows), None) if pose is None else model(rep(x_t), rep(pose), tt, rows)

        def mix(z):
            if z is None:
                return None
            m = z[:n].mean(0, keepdim=True)
            return (1 + cfg_strength) * m - cfg_strength * z[n:] if guided else m
        return mix(v), mix(pv)
    return infer


def sparse_step():
    """The SLAT sampler's _inference_model with condition + negative as one batch of 2 (the same coords twice)."""
    import torch
    from recgen_inference.recgen_modules.modules import sparse as sp

    def infer(model, x_t, t, cond, neg_cond=None, cfg_strength=0., cfg_interval=(0., 1.), **_):
        guided = neg_cond is not None and cfg_strength > 0 and cfg_interval[0] <= t <= cfg_interval[1]
        if not guided:
            return model(x_t, torch.full((1,), 1000. * t, device=cond.device, dtype=torch.float32), cond), None
        out = model(sp.sparse_cat([x_t, x_t]), torch.full((2,), 1000. * t, device=cond.device, dtype=torch.float32), torch.cat([cond, neg_cond]))
        a, b = out.layout
        assert torch.equal(out.coords[a, 1:], x_t.coords[:, 1:]), "batched SLAT output lost the input's voxel order"
        return x_t.replace((1 + cfg_strength) * out.feats[a] - cfg_strength * out.feats[b]), None
    return infer


@contextmanager
def fusion(pipeline, mode):
    """'fused*': batched CFG in both samplers and the batched pair fusion for the call; otherwise RecGen's own code."""
    if not mode.startswith("fused"):
        yield
        return
    from recgen_inference.recgen_modules.utils import multi_view_utils as mvu
    own = mvu.multidiffusion_sampling
    ss, sl = pipeline.sparse_structure_sampler, pipeline.slat_sampler

    @contextmanager
    def pairs(sampler, view_conds, num_steps=50, mode="multidiffusion"):  # run_pointmap_many_views imports it at call time
        before = sampler._inference_model
        sampler._inference_model = dense_step(view_conds, stochastic=mode == "stochastic")
        try:
            yield
        finally:
            sampler._inference_model = before
    mvu.multidiffusion_sampling = pairs
    ss._inference_model, sl._inference_model = dense_step(), sparse_step()
    try:
        yield
    finally:
        mvu.multidiffusion_sampling = own
        for s in (ss, sl):
            s.__dict__.pop("_inference_model", None)


@contextmanager
def cached_images(pipeline, on):
    """cache_cond: encode_image once per image object for the call; a clone out, since get_cond adds to it in place."""
    if not on:
        yield
        return
    own, cache = pipeline.encode_image, {}

    def encode_image(image):
        key = tuple(map(id, image)) if isinstance(image, list) else id(image)
        if key not in cache:
            cache[key] = own(image)
        return cache[key].clone()
    pipeline.encode_image = encode_image
    try:
        yield
    finally:
        pipeline.__dict__.pop("encode_image", None)


def posed_arrays(out, cam2ncam):
    """_build_result_from_outputs' posed mesh without trimesh: v -> (R s v + t - cam2ncam_t) / cam2ncam_s, colours as
    mesh_from_result makes them (clip, x255, truncate)."""
    import torch
    from recgen_inference.utils import parse_pose
    pose, _, _ = parse_pose(out)
    m = out["mesh"][0]
    c = np.asarray(cam2ncam, np.float64)
    a = torch.as_tensor(pose[:3, :3], dtype=torch.float64, device=m.vertices.device)
    b = torch.as_tensor(pose[:3, 3] - c[:3, 3], dtype=torch.float64, device=m.vertices.device)
    v = (m.vertices.double() @ a.T + b) / float(c[0, 0])
    colors = ((m.vertex_attrs[:, :3].clamp(0, 1) * 255).to(torch.uint8) if m.vertex_attrs is not None and m.vertex_attrs.shape[1] >= 3
              else torch.full((len(v), 3), 160, dtype=torch.uint8, device=v.device))
    return {"vertices": v.float().cpu().numpy(), "faces": m.faces.to(torch.int32).cpu().numpy().astype(np.uint32), "colors": colors.cpu().numpy()}


def same_surface(a, b, faces_a, faces_b, tol=1e-5):
    """fast_out's check: both vertex sets, duplicates merged as trimesh merges them, match point for point (KD-tree, both ways)."""
    import trimesh
    from scipy.spatial import cKDTree
    va = np.asarray(trimesh.Trimesh(a, faces_a, process=True).vertices)
    vb = np.asarray(trimesh.Trimesh(b, faces_b, process=True).vertices)
    d = max(cKDTree(vb).query(va)[0].max(), cKDTree(va).query(vb)[0].max())
    return {"vertices": [len(va), len(vb)], "max_distance": float(d), "same": bool(len(va) == len(vb) and d <= tol)}


def run(pipeline, views, seed, s, export_dir=None, check_out=False):
    """One RecGen call with setting s (module docstring). -> mesh arrays (the anchor's OpenCV camera frame, metres as given),
    per-stage seconds, total seconds (preprocess -> numpy mesh), voxels; export_dir: also RecGenResult.save() there, timed apart."""
    import torch
    from recgen_inference.inference import _build_result_from_outputs
    from recgen_inference.preprocessing import normalize_depth, preprocess_view
    st = pipeline._stages = Stages()
    views = list(views)[:s["max_views"]]
    started = time.perf_counter()
    with st("preprocess"):
        procs = [preprocess_view(v["rgb"], normalize_depth(v["depth"]), v["mask"], v["camera_intrinsics"], quantile_drop_threshold=.05,
                                 clamp_range=(-2., 3.), mask_erosion_enabled=True, mask_erosion_params=None) for v in views]
    kw = dict(seed=seed, sparse_structure_sampler_params=sampler_params(s["ss_steps"], s["ss_cfg"]),
              slat_sampler_params=sampler_params(s["slat_steps"], s["slat_cfg"]), formats=list(s["formats"]))
    p0, rest = procs[0], procs[1:]
    with fusion(pipeline, s["fusion"]), cached_images(pipeline, s["cache_cond"]):
        if not rest:
            out = pipeline.run_pointmap(p0["image"], pointmap=p0["pointmap"], mask=p0["mask"], **kw)
        elif len(rest) == 1:
            out = pipeline.run_pointmap_multiview(images=[p["image"] for p in procs], pointmaps=[p["pointmap"] for p in procs],
                                                  masks=[p["mask"] for p in procs], **kw)
        else:
            out = pipeline.run_pointmap_many_views(view1_image=p0["image"], views2_images=[p["image"] for p in rest], view1_pointmap=p0["pointmap"],
                                                   views2_pointmaps=[p["pointmap"] for p in rest], view1_mask=p0["mask"],
                                                   views2_masks=[p["mask"] for p in rest],
                                                   multidiffusion_mode="stochastic" if "stochastic" in s["fusion"] else "multidiffusion", **kw)
    with st("mesh_out"):
        if s["fast_out"] and not export_dir:
            mesh = posed_arrays(out, p0["cam2ncam"])
            if check_out:  # the same decoder output through RecGen's own path (outside the timer's meaning: a check call)
                ref = _build_result_from_outputs(out, p0["cam2ncam"], rgb=views[0]["rgb"], intrinsics=views[0]["camera_intrinsics"]).mesh
                pipeline._out_check = same_surface(np.asarray(ref.vertices), mesh["vertices"], np.asarray(ref.faces), mesh["faces"])
        else:
            result = _build_result_from_outputs(out, p0["cam2ncam"], rgb=views[0]["rgb"], intrinsics=views[0]["camera_intrinsics"])
            posed = result.mesh
            colors = np.asarray(posed.visual.vertex_colors)[:, :3] if hasattr(posed.visual, "vertex_colors") else np.full((len(posed.vertices), 3), 160)
            mesh = {"vertices": np.asarray(posed.vertices, np.float32), "faces": np.asarray(posed.faces, np.uint32), "colors": colors.astype(np.uint8)}
    seconds = time.perf_counter() - started
    sec = dict(st.s)
    sec["ss_flow"] = sec.get("ss_flow", 0.) - sec.get("ss_decode", 0.)
    pipeline._last_coords = out["coords"]
    rec = {**mesh, "views": len(views), "seconds": seconds, "stages": {k: round(v, 4) for k, v in sec.items()},
           "voxels": int(len(out["coords"])), "faces_n": int(len(mesh["faces"]))}
    if check_out and s["fast_out"]:
        rec["fast_out_check"] = pipeline.__dict__.pop("_out_check", None)
    if export_dir:  # RecGen's own outputs (OBJ x2, overlay, PLY x2 when Gaussians exist; turntable needs nvdiffrast: not here)
        t = time.perf_counter()
        try:
            result.save(export_dir, save_splat="gaussian" in out)
        except Exception as error:  # noqa: BLE001  the timing is a side measurement: never lose the call
            rec["export_error"] = repr(error)[-300:]
        torch.cuda.synchronize()
        rec["export_s"] = round(time.perf_counter() - t, 3)
        rec["export_files"] = sorted(p.name for p in Path(export_dir).iterdir()) if Path(export_dir).exists() else []
    return rec


def fused_check(pipeline, pairs=3, t=.9):
    """One SS step with RecGen's own per-pair CFG average vs dense_step's one batch, same random inputs: the largest
    difference relative to the largest value (fp16 blocks: small, not zero)."""
    import torch
    from recgen_inference.recgen_modules.utils.multi_view_utils import multidiffusion_sampling
    m, s = pipeline.models["sparse_structure_pose_flow_model"], pipeline.sparse_structure_sampler
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(1, m.in_channels, *[m.resolution] * 3, device="cuda", generator=g)
    pose = torch.randn(1, *((m.num_pose_tokens,) if m.num_pose_tokens > 1 else ()), m.pose_channels, device="cuda", generator=g)
    conds = [torch.randn(1, 2 * 1369, m.cond_channels, device="cuda", generator=g) for _ in range(pairs)]
    kw = {"neg_cond": torch.zeros_like(conds[0]), "cfg_strength": 7.5, "cfg_interval": (0., 1.), "pose": pose}
    with torch.no_grad():
        with multidiffusion_sampling(s, conds, 25, "multidiffusion"):
            a, ap = s._inference_model(m, x, t, conds[0], **kw)
        b, bp = dense_step(conds)(m, x, t, conds[0], **kw)
    rel = lambda u, v: float((u - v).abs().max() / u.abs().max().clamp_min(1e-6))  # noqa: E731
    out = {"velocity_rel_max_diff": rel(a, b), "pose_rel_max_diff": rel(ap, bp)}
    assert out["velocity_rel_max_diff"] < .02 and out["pose_rel_max_diff"] < .02, out
    return out


def micro(pipeline, batches=(1, 2, 4, 8), reps=5):
    """Forward seconds of the SS and SLAT flow models at batch B (inputs repeated; SLAT on the last call's voxels): what
    batching rows (CFG, pairs, objects) into one forward buys, with no sampler around it."""
    import torch
    from recgen_inference.recgen_modules.modules import sparse as sp
    out = {}
    m = pipeline.models["sparse_structure_pose_flow_model"]
    sl = pipeline.models["slat_flow_model"]
    coords = pipeline._last_coords
    with torch.no_grad():
        for b in batches:
            x = torch.randn(b, m.in_channels, *[m.resolution] * 3, device="cuda")
            pose = torch.randn(b, *((m.num_pose_tokens,) if m.num_pose_tokens > 1 else ()), m.pose_channels, device="cuda")
            cond = torch.randn(b, 2 * 1369, m.cond_channels, device="cuda")
            t = torch.full((b,), 500., device="cuda")
            xs = sp.sparse_cat([sp.SparseTensor(feats=torch.randn(len(coords), sl.in_channels, device="cuda"), coords=coords)] * b)
            cs = torch.randn(b, 2 * 1369 + 1, cond.shape[-1], device="cuda")
            for name, fn in (("ss", lambda: m(x, pose, t, cond)), ("slat", lambda: sl(xs, t, cs))):
                fn()
                times = []
                for _ in range(reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    times.append(time.perf_counter() - t0)
                out.setdefault(name, {})[str(b)] = round(float(np.median(times)), 4)
    out["slat_voxels"] = int(len(coords))
    return out


def recgen_worker():
    """A RecGen process (/opt/recgen-venv): loads once (x7.load_recgen), times its stages, self-checks the fused step, warms
    up; then {'op': 'run', views, seed, setting[, export]} or {'op': 'micro'}."""
    import torch
    from fast_report import x7
    from fast_report.sam3d import serve
    state = {}

    def place(device):  # r5b integrate: the whole pipeline's modules (p.models, DINOv2 included) to a device
        for m in state["p"].models.values():
            if hasattr(m, "to"):
                m.to(device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def boot():
        t = time.time()
        p = state["p"] = x7.load_recgen()
        torch.cuda.synchronize()
        if os.environ.get("RECGEN_PARK") == "1":  # r5b integrate: GPU 1's pair waits in host memory until SAM 3 has left GPU 1 (its
            instrument(p)  # first job wakes it: models_job dispatches only after the facts), no warm-up beside GPU 1's boot
            place("cpu")
            state["parked"] = True
            return {"ready": True, "parked": True, "load_s": round(time.time() - t, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES")}
        info = {"load_s": round(time.time() - t, 1), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "attn": {k: os.environ.get(k) for k in ("ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "COMPILE_SS")},
                "samplers": [type(p.sparse_structure_sampler).__name__, type(p.slat_sampler).__name__],
                "ss_params": p.sparse_structure_sampler_params, "slat_params": p.slat_sampler_params,
                "fp16": {k: getattr(m, "use_fp16", None) for k, m in p.models.items()}, "slat_use_pose": getattr(p, "slat_use_pose", None),
                "ss_model": {k: getattr(p.models["sparse_structure_pose_flow_model"], k, None) for k in
                             ("resolution", "model_channels", "num_blocks", "num_pose_tokens", "use_asymmetric_mask", "use_frame_token_embedder")}}
        instrument(p)
        info["fused_check"] = fused_check(p)
        if os.environ.get("COMPILE_SS") == "1":
            m = p.models["sparse_structure_pose_flow_model"]
            m.forward = torch.compile(m.forward)
        t, compiled = time.time(), os.environ.get("COMPILE_SS") == "1"
        for w in [setting(), setting(fusion="fused", formats=("mesh",))] if compiled else [setting()]:  # compile: every batch shape
            for n in (1, 2, 3, 4) if compiled else (2,):
                run(p, x7.synthetic_views(n), x7.SEED, w)
        info["warm_s"] = round(time.time() - t, 1)
        torch.cuda.empty_cache()
        return {"ready": True, **info}

    def handle(message, _):
        start, woke = time.time(), None
        if state.get("parked"):
            place("cuda")
            state["parked"], woke = False, round(time.time() - start, 2)
        torch.cuda.reset_peak_memory_stats()
        if message["op"] == "micro":
            return {"micro": micro(state["p"]), "start_unix": start, "end_unix": time.time()}
        if message["op"] == "x7":  # X7's own call (recgen_inference.generate / generate_multiview): run()'s default must match it
            out = x7.run_recgen(state["p"], message["views"], message["seed"])
        else:
            out = run(state["p"], message["views"], message["seed"], message["setting"], message.get("export_dir"), message.get("check_out", False))
        peak = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
        torch.cuda.empty_cache()  # r5b (804a4ca's fix): the call's cache back to the device, beside the report's core on the same GPU
        return {**out, "start_unix": start, "end_unix": time.time(), "max_reserved_gib": peak, "woke_s": woke}
    serve(handle, boot)


# ---------------------------------------------------------------- CPU side (/opt/gate venv)
def surface_points(vertices, faces, n=SAMPLES, seed=0):
    """n points uniformly on the mesh surface (area-weighted faces, uniform barycentrics)."""
    v, f = np.asarray(vertices, np.float64), np.asarray(faces, np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    i = rng.choice(len(f), n, p=area / area.sum())
    u, w = rng.random((2, n, 1))
    flip = u + w > 1
    u, w = np.where(flip, 1 - u, u), np.where(flip, 1 - w, w)
    return a[i] + u * (b[i] - a[i]) + w * (c[i] - a[i])


def to_surface(points, vertices, faces):
    """Unsigned distance of each point to the nearest triangle of the mesh (exact, open3d BVH)."""
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene.compute_distance(o3d.core.Tensor(np.asarray(points, np.float32))).numpy()


def chamfer(a, b, n=SAMPLES):
    """Symmetric Chamfer distance between meshes a and b, each (vertices, faces): n surface samples of each, their mean
    distance to the other's surface, the two means averaged (no sampling floor: one mesh gives 0)."""
    return float((to_surface(surface_points(*a, n), *b).mean() + to_surface(surface_points(*b, n, seed=1), *a).mean()) / 2)


def judge(src, msg):
    """msg: key, gen (X7's generation frames: placement), held, crop, anchor (the frame whose camera the mesh is posed in),
    mesh, ref (the reference mesh in the same anchor camera, or None). -> held-out gate, placement, Chamfer, tile."""
    from fast_report import x7
    t0 = time.time()
    g = msg["mesh"]
    out = {}
    if msg.get("ref") is not None:  # both meshes in the anchor's camera: distances as in the world
        ref = (msg["ref"]["vertices"], msg["ref"]["faces"])
        d = chamfer((g["vertices"], g["faces"]), ref)
        out.update(chamfer_cm=round(100 * d, 3), chamfer_rel=round(d / float(np.linalg.norm(np.ptp(ref[0], 0))), 5))
    c2w = np.asarray(src.rows[msg["anchor"]]["c2w"], float)
    if g.get("objectToCamera") is not None:  # SAM 3D: its pose puts the mesh in the pointmap's PyTorch3D camera (x7.gate_model)
        import complete_video_objects as cvo
        vertices = cvo.sam3d_to_world(g["vertices"], g["objectToCamera"], c2w)
    else:
        vertices = x7.transformed(g["vertices"], c2w)
    vertices, faces, colors = x7.light(vertices, np.asarray(g["faces"], np.int64), np.asarray(g["colors"], np.float64))
    views = []
    for f in msg["gen"]:
        depth, mask, k, c2w = x7.view_data(src, msg["key"], f, .5)
        views.append({"rays": x7.rays(k, c2w, depth.shape[1], depth.shape[0]), "target": mask, "depth": depth})
    moved, placement = x7.refine(vertices, faces, views)
    vertices = x7.transformed(vertices, moved)
    gate = x7.held_gate(src, msg["key"], msg["held"], vertices, faces)
    out.update(gate={k: gate.get(k) for k in GATE_KEYS}, placement_s=placement["seconds"], placement_moved=placement["moved"],
               faces_light=int(len(faces)), tile=x7.render_tile(src, msg["held"], msg["crop"], vertices, faces, colors, np.ones(len(vertices), bool)))
    return {**out, "judge_s": round(time.time() - t0, 3)}


def cpu_worker():
    """One CPU process: X7's 'select' and this module's 'judge' on the staged hand-off, sources cached per shot."""
    os.nice(5)
    import json
    import complete_video_objects as cvo
    from fast_report import sam3d, x7
    sources = {}

    def source(spec, shot):
        key = (json.dumps(spec, sort_keys=True), shot)
        if key not in sources:
            cvo.VOXEL, cvo.OCCLUSION = sam3d.VOXEL_M, sam3d.VOXEL_M / 2  # X7's gate tolerances in estimated metres
            cvo.FIT_GATE["max_fit_median_native"] = sam3d.VOXEL_M
            sources[key] = sam3d.FastSource(spec, shot)
        return sources[key]

    def handle(m, _):
        start = time.time()
        src = source(m["src"], m["shot"])
        out = x7.select(src, m["key"], m["obj"], m["min_sep"]) if m["op"] == "select" else judge(src, m)
        return {**out, "start_unix": start, "end_unix": time.time()}

    def boot():
        import cv2
        cv2.setNumThreads(1)
        import open3d  # noqa: F401
        import scipy.optimize  # noqa: F401
        import build_lingbot_object_model  # noqa: F401
        from ehs_spatial.platform import recgen  # noqa: F401
        return {"ready": True, "pid": os.getpid()}
    sam3d.serve(handle, boot)


# ---------------------------------------------------------------- self-check (local, CPU)
def self_check():
    import trimesh
    assert setting(**FAST)["ss_cfg"] is None  # FAST keeps the SS stage's CFG
    s = setting(ss_steps=12, slat_cfg=0, formats=("mesh",), fusion="fused", cache_cond=True)
    assert s["max_views"] == 4 and s["ss_cfg"] is None and s["formats"] == ("mesh",) and s["cache_cond"] and not DEFAULT["cache_cond"]
    assert sampler_params(None, None) == {} and sampler_params(12, 0) == {"steps": 12, "cfg_strength": 0., "cfg_interval": (2., 2.)}
    assert sampler_params(8, 3.) == {"steps": 8, "cfg_strength": 3.}
    for bad in ({"steps": 3}, {"fusion": "avg"}, {"formats": ("gaussian",)}):
        try:
            setting(**bad)
        except AssertionError:
            continue
        raise AssertionError(f"accepted {bad}")
    box = trimesh.creation.box(extents=[1., 2., .5])
    p = surface_points(box.vertices, box.faces)
    assert p.shape == (SAMPLES, 3) and np.allclose(np.ptp(p, 0), [1, 2, .5], atol=.01)
    assert np.isclose(np.abs(p), [.5, 1, .25], atol=1e-9).any(1).all()  # every point on a face
    mesh = (box.vertices, box.faces)
    assert chamfer(mesh, mesh) < 1e-5  # one surface: no sampling floor
    d = chamfer(mesh, (box.vertices + [.1, 0, 0], box.faces))
    assert .027 < d < .032, d  # 10 cm shift along x: (1 x .1 + 1 x .08 + 5 x .005) / 7 m^2 = 0.029 m by hand
    v, f = np.asarray(box.vertices), np.asarray(box.faces)
    assert same_surface(v, v + 1e-7, f, f)["same"] and not same_surface(v, v + [.01, 0, 0], f, f)["same"]  # fast_out's check
    print("fast_report.recgen_fast self-check passed: settings, sampler params, surface sampling, Chamfer, same surface")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
