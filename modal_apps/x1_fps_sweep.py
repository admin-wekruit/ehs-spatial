"""X1 frame-rate sweep: the fast core processes keyframes, not every frame; does a higher rate change the result?

One Modal container per video (2 x A100-80GB), fb/a-core's own pieces, everything but the rate fixed:
  - SAM 3 (bf16, 80 pairs per forward; vocabulary floor .3, no cap, flood dedupe; person/floor .4 top 12) runs ONCE on
    every frame. SAM 3 image mode treats frames independently, so each rate is a subset of the same per-frame masks.
    Timed in nested phases (current object frames, then the rest of 5 fps, 10 fps, 15 fps, every frame).
  - The vocabulary is fixed per site: the Qwen v1 list of runs/fb-d-harness-gaps-001 (no VLM run-to-run spread).
  - Geometry: DA3-GIANT-1.1 any-view per shot at 5 fps (production: 'G5'), ~10 fps in one forward ('G10'), every
    frame three ways: consecutive 150-view chunks with evenly spread overlaps chained by a Sim3 fitted on the shared
    frames' depth points ('G30c', the naive chunking), interleaved chunks that each hold ALL the shot's 5 fps frames plus
    every m-th other frame, each put onto G5 by the Sim3 of those shared frames' depth ('G30i': G5's own frames kept as
    they are), and one forward over every frame of the longest shot ('G30single', feasibility); plus 5 fps and extra
    views where the camera moves fast ('Gmotion', the 10 fps view budget).
  - Objects and people above 5 fps take their in-between frames' depth and pose from G30i, so the 5 fps frames keep the
    production geometry exactly and every rate lifts into the same world.
  - Held-out frames (about 1 per second, frames that have the delivered DA3-posed depth) are never used by any
    configuration; they score the geometry.

Scale everywhere is 'estimated' (floor plane + an assumed 1.6 m camera height); nothing here is a metric claim.

  modal run modal_apps/x1_fps_sweep.py --out RUNS/fx-x1-fps-NNN [--sites me340,samsclub-a2,walmart] [--frames A-B]
  python modal_apps/x1_fps_sweep.py --evaluate RUNS/fx-x1-fps-NNN     # fb/d-harness rows -> results.json
  python modal_apps/x1_fps_sweep.py --self-check                      # numpy parts; the torch parts run in the container
"""
import copy
import json
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "scripts"), str(HERE.parent)]
import fast_report_app as fr  # noqa: E402  the a-core image, DA3 / SAM 3 pins, weight volumes

PHASE2 = fr.PHASE2
CPU, MEM_GIB = 16, 96
SITES = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}  # fast_report_eval.CLIPS
# the delivered DA3-posed depth (DROID-native units, same metres_per_native_unit as the harness reference): E1 used ME340's
MONO = {"me340": "da3-posed-me340-223-shotc/mono", "samsclub-a2": "da3-posed-samsclub-a2-281/mono", "walmart": "da3-posed-walmart-251-shot383/mono"}
CHUNK_VIEWS, MIN_OVERLAP = 150, 30  # every-frame DA3: views per forward (E1's 150-view timing), least shared frames
INTERLEAVED_VIEWS = 300  # G30i: views per forward (5 fps anchors + every m-th other frame); ~G10's size
EVAL_FRAMES, SAM_CHUNK, MIN_REF_PX = 60, 8, 184  # outline frames per video; SAM 3 frames per task; 1200 px at 1280x720 on the DA3 grid
SMALL_M, THIN_RATIO, THIN_MIN_M = .3, 3., .3  # small: largest extent < 0.3 m; thin/long: longest >= 3 x middle and >= 0.3 m (estimated)
NEW_SHARE = .1  # an object is 'new' when under 10 % of its voxels lie in any object of the current rate
PROD_MASK_BYTES = 280 * 504 + 288 * 288 * 2  # a-core keeps every vocabulary mask (DA3 grid bool) and its 288^2 fp16 logits on GPU 0
app = modal.App("panoptes-fx-x1-fps")
OUT = modal.Volume.from_name("panoptes-fx-x1-fps", create_if_missing=True)
VOLUMES = {"/v/da3": fr.VOLUMES["/v/da3"], "/v/sam3": fr.VOLUMES["/v/sam3"], "/v/out": OUT}
image = fr.image.add_local_python_source("fast_report_app")


# ---------- frame sets (numpy) ----------

def keyset(sharp, lo, hi, block, held):
    """a-core's rule over [lo, hi): the sharpest frame of each `block`-frame block, blocks from frame 0; held-out never."""
    keys = []
    for b in range(lo - lo % block, hi, block):
        cands = [f for f in range(max(b, lo), min(b + block, hi)) if f not in held]
        if cands:
            keys.append(max(cands, key=lambda f: sharp[f]))
    return keys


def allocate(scores, room, budget):
    """Split `budget` extra frames over intervals in proportion to score (largest remainder), at most room[i] each."""
    s = np.maximum(np.asarray(scores, float), 0) + 1e-9
    room, n = np.asarray(room, int), np.zeros(len(s), int)
    left = int(budget)
    while left > 0 and (n < room).any():
        act = n < room
        share = np.where(act, s, 0) / np.where(act, s, 0).sum() * left
        add = np.minimum(np.floor(share).astype(int), room - n)
        if add.sum() == 0:
            add = np.zeros_like(n)
            add[np.argmax(np.where(act, share - np.floor(share), -1))] = 1
        n += add
        left -= int(add.sum())
    return n


def extras(a, b, k, held):
    """k frames strictly inside (a, b), nearest to k evenly spaced positions."""
    cands = [f for f in range(a + 1, b) if f not in held]
    out = []
    for j in range(k):
        want = a + (j + 1) * (b - a) / (k + 1)
        left = [f for f in cands if f not in out]
        if left:
            out.append(min(left, key=lambda f: (abs(f - want), f)))
    return sorted(out)


def adaptive(base, scores, budget, held):
    """base keyframes (sorted, one shot) plus `budget` frames placed by interval score."""
    room = [len([f for f in range(a + 1, b) if f not in held]) for a, b in zip(base, base[1:])]
    n = allocate(scores, room, budget) if len(room) else []
    return sorted(set(base) | {f for (a, b), k in zip(zip(base, base[1:]), n) for f in extras(a, b, int(k), held)})


def chunk_starts(n, size=CHUNK_VIEWS, overlap=MIN_OVERLAP):
    """Evenly spread starts of `size`-view windows over n views, neighbours sharing >= overlap views."""
    if n <= size:
        return [0]
    k = int(np.ceil((n - overlap) / (size - overlap)))
    return [int(round(i * (n - size) / (k - 1))) for i in range(k)]


def robust_extents(codes, voxel):
    """An object's voxel codes (segment.lift's packing) -> extents along its principal axes, longest first: the 10-90 %
    range / 0.8 (a uniform box's length) + one voxel, so a few flying-depth voxels do not stretch it. Metres, estimated."""
    B, OFF = 1 << 21, 1 << 20
    c = np.stack([codes // (B * B) - OFF, (codes // B) % B - OFF, codes % B - OFF], 1).astype(float) * voxel
    if len(c) < 3:
        return np.full(3, voxel)
    x = c - c.mean(0)
    p = x @ np.linalg.eigh(x.T @ x)[1][:, ::-1]
    lo, hi = np.percentile(p, [10, 90], axis=0)
    return (hi - lo) / .8 + voxel


def interleaved(anchors, others, most=INTERLEAVED_VIEWS):
    """Chunks that each hold every anchor plus every m-th other frame, m the fewest that keeps chunks <= most views."""
    m = max(1, int(np.ceil(len(others) / max(1, most - len(anchors)))))
    return [sorted(anchors + others[j::m]) for j in range(m)]


def sim3_points(src, dst, iters=3, keep=.8):
    """dst ~ s R src + t (Umeyama, E1's), refitted on the best `keep` share of residuals `iters` times."""
    import m3_exp_geometry as geo
    idx = np.arange(len(src))
    for _ in range(iters):
        s, R, t = geo.umeyama(src[idx], dst[idx])
        r = np.linalg.norm((s * (R @ src.T)).T + t - dst, axis=1)
        idx = np.flatnonzero(r <= np.quantile(r, keep))
    return s, R, t, r


def self_check():
    sharp = {f: float((f * 7) % 5) for f in range(40)}
    assert keyset(sharp, 0, 12, 6, set()) == [2, 7] and keyset(sharp, 0, 12, 6, {2}) == [4, 7]
    assert keyset(sharp, 8, 12, 6, set()) == [9], "blocks stay aligned to frame 0"
    assert allocate([1, 3], [5, 5], 4).tolist() == [1, 3] and allocate([0, 1], [5, 2], 4).tolist() == [2, 2]
    assert allocate([1, 1, 1], [1, 1, 1], 9).tolist() == [1, 1, 1]
    assert extras(0, 6, 1, set()) == [3] and extras(0, 6, 2, {2}) == [1, 4] and extras(0, 2, 3, set()) == [1]
    assert adaptive([0, 6, 12], [0, 1], 2, set()) == [0, 6, 8, 10, 12]
    assert chunk_starts(100) == [0] and chunk_starts(651) == [0, 100, 200, 301, 401, 501]
    st = chunk_starts(725)
    assert all(b - a <= CHUNK_VIEWS - MIN_OVERLAP for a, b in zip(st, st[1:])) and st[-1] + CHUNK_VIEWS == 725
    rng = np.random.default_rng(0)
    src = rng.normal(size=(500, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    dst = 2.5 * src @ q.T + [1, 2, 3]
    dst[:50] += 5  # outliers the trimmed refit must ignore
    s, R, t, _ = sim3_points(src, dst)
    assert abs(s - 2.5) < 1e-6 and np.allclose(R, q, atol=1e-6) and np.allclose(t, [1, 2, 3], atol=1e-5)
    B, OFF = 1 << 21, 1 << 20
    ijk = np.array([[i, j, k] for i in range(20) for j in range(2) for k in range(2)]) + OFF
    e = robust_extents((ijk[:, 0] * B + ijk[:, 1]) * B + ijk[:, 2], .05)
    assert abs(e[0] - 1.0) < .03 and .09 < e[1] < .15 and .09 < e[2] < .15, e  # the 2 x 2 cross-section: its axes are free to turn
    ch = interleaved(list(range(0, 60, 6)), [f for f in range(60) if f % 6], 30)
    assert len(ch) == 3 and all(len(c) <= 30 and set(range(0, 60, 6)) <= set(c) for c in ch) and set().union(*ch) == set(range(60))
    print("x1 self-check ok (numpy): keysets, allocation, extras, chunks, interleaved chunks, robust extents, trimmed Sim3")


# ---------- torch helpers (checked in the container: gpu_self_check) ----------

def pack(m):
    """(n,H,W) bool -> (n,H,W/8) uint8 numpy, np.packbits bit order."""
    import torch
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=m.device)
    return (m.reshape(m.shape[0], m.shape[1], m.shape[2] // 8, 8).to(torch.uint8) * w).sum(-1, dtype=torch.uint8).cpu().numpy()


def unpack(p, dev):
    import torch
    x = torch.from_numpy(np.ascontiguousarray(p)).to(dev)
    w = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=dev)
    return ((x[..., None] & w) > 0).reshape(x.shape[0], x.shape[1], x.shape[2] * 8)


def lift_big(m2, frame_of, depth_m, K, c2w_m, dyn2, rows=4096, batch=16384):
    """fast_report.segment.lift (E7/E9's rule, unchanged) for masks and people already at stride 2, with the voxel
    overlap product in row blocks (one product over 30 fps masks does not fit) -> (comp, arrays, stats, extra) where
    extra: per lifted mask its index, 3D centroid and own extents (edge pixels out), per component its voxel codes."""
    import torch
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from fast_report import segment as sg
    dev, n, stride = depth_m.device, len(depth_m), 2
    comp = np.full(len(m2), -1)
    valid2 = depth_m[:, ::stride, ::stride] > 0
    idx, mids, vys, vxs, moving, off = [], [], [], [], 0, 0
    for b0 in range(0, len(m2), batch):
        mb, fb = m2[b0:b0 + batch], frame_of[b0:b0 + batch]
        dy = dyn2[fb]
        mv = (mb & dy).sum((1, 2)) >= sg.MOVING * mb.sum((1, 2)).clamp(min=1)
        mm = mb & valid2[fb] & ~dy
        keep = ~mv & (mm.sum((1, 2)) >= sg.MIN_PIXELS)
        moving += int(mv.sum())
        k = torch.nonzero(keep).squeeze(1)
        mid, vy, vx = torch.nonzero(mm[k], as_tuple=True)
        idx.append(k + b0)
        mids.append(mid + off)
        vys.append(vy)
        vxs.append(vx)
        off += len(k)
    idx = torch.cat(idx)
    stats = {"masks_in": len(m2), "masks_dropped_moving": moving, "masks_lifted": len(idx)}
    if len(idx) < 2:
        return comp, None, stats, None
    mid, vy, vx = torch.cat(mids), torch.cat(vys), torch.cat(vxs)
    fr_ = frame_of[idx]
    world = sg.backproject(depth_m, K, c2w_m, fr_[mid], vy, vx, stride)
    ijk = torch.floor(world / sg.LIFT_VOXEL).long()
    B, OFF = 1 << 21, 1 << 20
    vox, vid = torch.unique(((ijk[:, 0] + OFF) * B + (ijk[:, 1] + OFF)) * B + (ijk[:, 2] + OFF), return_inverse=True)
    L, nv = len(idx), len(vox)
    pair = torch.unique(mid * nv + vid)
    pm, pv = pair // nv, pair % nv
    size = torch.bincount(pm, minlength=L).float()
    At = torch.sparse_coo_tensor(torch.stack([pv, pm]), torch.ones_like(pm, dtype=torch.float32), (nv, L)).coalesce()
    bounds = torch.searchsorted(pm, torch.arange(0, L + rows, rows, device=dev)).tolist()
    ea, eb, overlap_pairs = [], [], 0
    for i, r0 in enumerate(range(0, L, rows)):
        s, e = bounds[i], bounds[i + 1]
        if s == e:
            continue
        Ab = torch.sparse_coo_tensor(torch.stack([pm[s:e] - r0, pv[s:e]]), torch.ones(e - s, device=dev), (min(rows, L - r0), nv)).coalesce()
        inter = torch.sparse.mm(Ab, At).coalesce()
        (a, b), c = inter.indices(), inter.values()
        a = a + r0
        overlap_pairs += len(c)
        ok = (a < b) & (fr_[a] != fr_[b]) & (c >= sg.MATCH_MIN * torch.minimum(size[a], size[b])) & (c >= sg.MATCH_MAX * torch.maximum(size[a], size[b]))
        ea.append(a[ok].cpu().numpy())
        eb.append(b[ok].cpu().numpy())
    a, b = np.concatenate(ea), np.concatenate(eb)
    ncomp, label = connected_components(coo_matrix((np.ones(len(a)), (a, b)), shape=(L, L)), directed=False)
    lab = torch.from_numpy(label.astype(np.int64)).to(dev)  # scipy gives int32: lab * voxels overflows at 30 fps (segment.lift has the same limit)
    ov = torch.unique(lab[pm] * nv + pv)
    oc, ovid = ov // nv, ov % nv
    centre = torch.stack([vox // (B * B) - OFF, (vox // B) % B - OFF, vox % B - OFF], 1).float()[ovid] * sg.LIFT_VOXEL + sg.LIFT_VOXEL / 2
    nvox = torch.bincount(oc, minlength=ncomp).float()
    cent = torch.zeros(ncomp, 3, device=dev).index_add_(0, oc, centre) / nvox.clamp(min=1)[:, None]
    lo = torch.full((ncomp, 3), 1e9, device=dev).scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amin")
    hi = torch.full((ncomp, 3), -1e9, device=dev).scatter_reduce(0, oc[:, None].expand(-1, 3), centre, "amax")
    nframes = torch.bincount(torch.unique(lab * n + fr_) // n, minlength=ncomp)
    comp[idx.cpu().numpy()] = label
    arrays = {k: x.cpu().numpy() for k, x in (("frames", nframes), ("voxels", nvox), ("centroid", cent), ("lo", lo), ("hi", hi))}
    stats.update(points=int(len(mid)), voxels=int(nv), edges=int(len(a)), components=int(ncomp), overlap_pairs=int(overlap_pairs))
    # extras for the sweep's measures (not part of the lift rule)
    cnt = torch.bincount(mid, minlength=L).float().clamp(min=1)
    mask_cent = torch.zeros(L, 3, device=dev).index_add_(0, mid, world) / cnt[:, None]
    # per lifted mask, its own 3D extents (one view: no cross-view depth disagreement), flying depth-edge pixels out
    import m3_exp_geometry as geo
    pad = torch.nn.functional.pad(depth_m[:, None], (1, 1, 1, 1), mode="replicate")[:, 0]
    nb = torch.stack([pad[:, 1:-1, :-2], pad[:, 1:-1, 2:], pad[:, :-2, 1:-1], pad[:, 2:, 1:-1]])
    edge = ((nb - depth_m).abs().amax(0) > geo.EDGE_JUMP * depth_m)[:, ::stride, ::stride]
    ok = ~edge[fr_[mid], vy, vx]
    m_, w_ = mid[ok], world[ok]
    n_ = torch.bincount(m_, minlength=L).float()
    mu = torch.zeros(L, 3, device=dev).index_add_(0, m_, w_) / n_.clamp(min=1)[:, None]
    dm = w_ - mu[m_]
    cov = torch.zeros(L, 9, device=dev).index_add_(0, m_, (dm[:, :, None] * dm[:, None, :]).reshape(-1, 9)).reshape(L, 3, 3) / n_.clamp(min=1)[:, None, None]
    view_ext = torch.sqrt(12 * torch.linalg.eigvalsh(cov.double().cpu()).clamp(min=0)).flip(1).float()  # CPU: cusolver's batched syevd failed on 180k matrices
    view_ext[n_ < 8] = float("nan")
    counts = torch.bincount(oc, minlength=ncomp).cpu().numpy()
    codes = np.split(vox[ovid].cpu().numpy(), np.cumsum(counts)[:-1])
    extra = {"lifted": idx.cpu().numpy(), "mask_centroid": mask_cent.cpu().numpy(), "codes": codes, "view_extent": view_ext.cpu().numpy()}
    return comp, arrays, stats, extra


def per_object_ids(per, ids, dev):
    """(k,H,W) bool per object -> (H,W) long map, 0 = nothing, ids[j] + 1 = object j (segment.paint: smaller wins)."""
    import torch
    from fast_report import segment as sg
    return torch.cat([torch.zeros(1, dtype=torch.long, device=dev), torch.tensor(ids, device=dev) + 1])[sg.paint(per)]


def project_to(sources, ids_of, geo_rows, target):
    """E6b 'pair' with explicit frames: sources in order (nearer first), the second fills what the first cannot see.
    geo_rows(f) -> (depth_m (H,W), K, c2w_m). -> (H,W) long, -1 = nothing, 0 = background, i + 1 = object i."""
    import torch
    from fast_report import segment as sg
    _, Kt, ct = geo_rows(target)
    w2c = torch.linalg.inv(ct)
    key = None
    for k in sources:
        z, Kk, ck = geo_rows(k)
        v, u = torch.nonzero(z > 0, as_tuple=True)
        world = sg.backproject(z[None], Kk[None], ck[None], torch.zeros_like(v), v, u, 1)
        s = sg.splat(world, ids_of(k)[v, u], Kt, w2c, sg.DA3_HW)
        key = s if key is None else torch.where(key != sg.EMPTY, key, s)
    return sg.finish(key, sg.DA3_HW)


def gpu_self_check():
    """lift_big == segment.lift on a synthetic three-view scene (row blocks of 2, batches of 3 exercise the blocking);
    pack/unpack round trip."""
    import torch
    from fast_report import segment as sg
    g = torch.Generator().manual_seed(0)
    m = torch.rand((5, 280, 504), generator=g) > .5
    assert np.array_equal(pack(m), np.packbits(m.numpy(), axis=-1)) and torch.equal(unpack(pack(m), "cpu"), m)
    n, H, W = 3, *sg.DA3_HW
    depth = torch.full((n, H, W), 2.)
    K = torch.tensor([[300., 0, 252], [0, 300, 140], [0, 0, 1]]).repeat(n, 1, 1)
    c2w = torch.eye(4).repeat(n, 1, 1)
    c2w[:, 0, 3] = torch.tensor([0., .1, .2])
    masks, frame_of = [], []
    for f in range(n):
        for (y0, x0, h, w) in [(40, 60, 60, 80), (150, 300, 50, 50), (20, 400, 30, 90)]:
            mk = torch.zeros((H, W), dtype=torch.bool)
            dx = int(round(f * .1 * 300 / 2))  # the same 3D patch in every view
            mk[y0:y0 + h, x0 - dx:x0 - dx + w] = True
            masks.append(mk)
            frame_of.append(f)
    masks, frame_of = torch.stack(masks), torch.tensor(frame_of)
    dyn = torch.zeros((n, H, W), dtype=torch.bool)
    comp, arr, _ = sg.lift(masks, frame_of, depth, K, c2w, dyn)
    comp2, arr2, _, extra = lift_big(masks[:, ::2, ::2], frame_of, depth, K, c2w, dyn[:, ::2, ::2], rows=2, batch=3)
    same = lambda a, b: all(len(set(b[a == c])) == 1 for c in set(a)) and all(len(set(a[b == c])) == 1 for c in set(b))  # noqa: E731
    assert same(comp, comp2), (comp, comp2)
    order = [int(comp2[np.flatnonzero(comp == c)[0]]) for c in range(len(arr["frames"]))]
    for k in ("frames", "voxels", "centroid", "lo", "hi"):
        assert np.allclose(arr[k], arr2[k][order]), k
    assert len(extra["codes"]) == len(arr2["frames"])
    print("x1 gpu self-check ok: lift_big == segment.lift, pack/unpack")


# ---------- the container ----------

@app.function(image=image, gpu="A100-80GB:2", cpu=CPU, memory=MEM_GIB * 1024, volumes=VOLUMES, timeout=3000, retries=0)
def sweep(p: dict):
    import torch
    t_enter = time.time()
    gpu_self_check()
    self_check()
    boot = load_models()
    boot["enter_to_ready_s"] = round(time.time() - t_enter, 1)
    from fast_report.instrument import Clock, Vram, usd_per_s
    vram = Vram([0, 1]).start()
    for d in (0, 1):
        torch.cuda.reset_peak_memory_stats(d)
    clock = Clock()  # t0: models warm, the MP4 bytes in the container
    rec, jpgs = {"site": p["site"], "boot": boot}, {}
    try:
        analyse(p, boot.pop("_models"), clock, rec, jpgs)
        rec["error"] = None
    except Exception:  # noqa: BLE001  the paid numbers so far come back with the error
        rec["error"] = traceback.format_exc()[-5000:]
    vram.stop()
    rec["run"] = clock.report(vram, price_per_s=usd_per_s(2, CPU, MEM_GIB), site=p["site"])
    rec["run"]["torch_reserved_peak_gib"] = [round(torch.cuda.max_memory_reserved(d) / 2 ** 30, 2) for d in (0, 1)]
    rec["container_s"] = round(time.time() - t_enter, 1)
    rec["usd_container_estimate"] = round(rec["container_s"] * usd_per_s(2, CPU, MEM_GIB), 3)
    OUT.commit()
    return {"record": json.loads(json.dumps(rec, default=plain)), "jpgs": jpgs}


def plain(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o.item() if hasattr(o, "item") else str(o)


def load_models():
    import torch
    import transformers
    from depth_anything_3.api import DepthAnything3
    from transformers import Sam3Model, Sam3Processor
    import sam3_app
    from fast_report import core, segment
    t, b = time.perf_counter(), {}
    d0, d1 = torch.device("cuda:0"), torch.device("cuda:1")
    da3 = DepthAnything3.from_pretrained(fr.DA3_MODEL, revision=fr.DA3_REV, cache_dir="/v/da3/huggingface/hub").eval()
    da3s = {d1: core.Da3(copy.deepcopy(da3).to(d1), d1), d0: core.Da3(da3.to(d0), d0)}
    proc = Sam3Processor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
    sam = Sam3Model.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub", torch_dtype=torch.bfloat16).eval()
    sams = {d0: segment.Sam3(copy.deepcopy(sam).to(d0), proc, d0), d1: segment.Sam3(sam.to(d1), proc, d1)}
    b["load_s"] = round(time.perf_counter() - t, 1)
    with torch.inference_mode():  # warm kernels at the run's shapes
        for d in (d0, d1):
            with torch.cuda.device(d):
                da3s[d].shot(torch.randint(0, 255, (16, 720, 1280, 3), dtype=torch.uint8, device=d))
                noise = torch.randint(0, 255, (SAM_CHUNK, 720, 1280, 3), dtype=torch.uint8, device=d)
                v = sams[d].vision(noise)
                sams[d].detect(v, SAM_CHUNK, ("person", "floor"), segment.PERSON_SCORE, top=segment.PERSON_TOP)
                sams[d].detect(v, 2, [f"word {i}" for i in range(57)], segment.VOCAB_SCORE)
                torch.cuda.synchronize(d)
                torch.cuda.empty_cache()
    import m3_exp_geometry as geo  # Open3D CUDA's first integrate is slow (a-core warms it at boot too)
    geo.fuse(torch.full((2, 280, 504), 2., device=d0), np.repeat(np.array([[[300., 0, 252], [0, 300, 140], [0, 0, 1]]]), 2, 0),
             np.repeat(np.eye(4)[None], 2, 0), torch.full((2, 280, 504, 3), .5, device=d0))
    b.update(ready_s=round(time.perf_counter() - t, 1), gpus=fr.gpu_listing(), torch=str(torch.__version__), transformers=transformers.__version__,
             _models={"da3": da3s, "sam": sams, "d0": d0, "d1": d1})
    return b


class Dense:
    """SAM 3 on a list of frames, both GPUs taking 8-frame tasks from one queue; phases end on a barrier so the time
    of each nested rate is measured. Vocabulary masks are deduplicated per frame at once (the a-core flood rule) and
    kept bit-packed on the CPU; person/floor detections stay on GPU 0 as in a-core."""

    def __init__(self, sams, words, frames, clock, dev_out):
        self.sams, self.words, self.frames, self.clock, self.dev_out = sams, list(words), frames, clock, dev_out
        self.q, self.lock, self.error = queue.Queue(), threading.Lock(), None
        self.person, self.vocab, self.masks_in = [], [], {}

    def run(self, phases):
        threads = [threading.Thread(target=self.worker, args=(d,), daemon=True) for d in self.sams]
        for t in threads:
            t.start()
        rows = []
        for name, fl in phases:
            start = self.clock.now()
            for i in range(0, len(fl), SAM_CHUNK):
                self.q.put(fl[i:i + SAM_CHUNK])
            self.q.join()
            if self.error is not None:
                raise RuntimeError("SAM 3 worker failed") from self.error
            rows.append({"phase": name, "frames": len(fl), "start_s": start, "end_s": self.clock.now(), "s": round(self.clock.now() - start, 3)})
        for _ in threads:
            self.q.put(None)
        for t in threads:
            t.join()
        return rows

    def worker(self, dev):
        import torch
        with torch.cuda.device(dev), torch.inference_mode():
            while True:
                fl = self.q.get()
                try:
                    if fl is None:
                        return
                    if self.error is None:
                        self.task(dev, fl)
                except BaseException as e:  # noqa: BLE001  re-raised by run()
                    self.error = e
                finally:
                    self.q.task_done()

    def task(self, dev, fl):
        import torch
        from fast_report import segment as sg
        sam, gpu, real = self.sams[dev], dev.index, len(fl)
        x = torch.from_numpy(np.stack([self.frames[f] for f in fl])).to(dev)
        if real < SAM_CHUNK:  # fixed shapes for the allocator, as a-core
            x = torch.cat([x, x[-1:].expand(SAM_CHUNK - real, -1, -1, -1)])
        ft = torch.tensor(fl, device=dev)
        with self.clock.stage(f"sam3.person@gpu{gpu}", gpu=gpu, n={"frames": real}, sync=True):
            v = sam.vision(x)
            r = sam.detect(v, SAM_CHUNK, ("person", "floor"), sg.PERSON_SCORE, top=sg.PERSON_TOP)
            r = {k: t[r["frame"] < real] for k, t in r.items()}
            r["frame"] = ft[r["frame"]]
        with self.clock.stage(f"sam3.vocab.wave1@gpu{gpu}", gpu=gpu, n={"frames": real, "words": len(self.words)}, sync=True):
            vr = v if real == SAM_CHUNK else sam.pick(v, list(range(real)))  # detect() takes the whole batch when n_frames fits one forward
            vv = sam.detect(vr, real, self.words, sg.VOCAB_SCORE)
        with self.clock.stage(f"dedupe@gpu{gpu}", gpu=gpu, n={"masks": int(len(vv["frame"]))}, sync=True):
            kept, votes = sg.dedupe(vv["frame"], vv["word"], vv["score"], vv["mask"])
            kt = torch.from_numpy(kept).to(dev)
            km = vv["mask"][kt]
            pos = {int(k): i for i, k in enumerate(kept)}
            out = {"frame": ft[vv["frame"][kt]].cpu().numpy(), "word": vv["word"][kt].cpu().numpy().astype(np.int16),
                   "score": vv["score"][kt].float().cpu().numpy(), "area": km.sum((1, 2)).cpu().numpy().astype(np.int32), "packed": pack(km),
                   "votes": [(pos[int(a)], int(w), float(s)) for a, w, s in votes]}
        counts = np.bincount(vv["frame"].cpu().numpy(), minlength=real)
        with self.lock:
            self.person.append({k: t.to(self.dev_out) for k, t in r.items()})
            self.vocab.append(out)
            for j, f in enumerate(fl):
                self.masks_in[f] = int(counts[j])

    def gathered(self):
        import torch
        P = {k: torch.cat([r[k] for r in self.person]) for k in self.person[0]}
        n = np.cumsum([0] + [len(r["frame"]) for r in self.vocab])
        V = {k: np.concatenate([r[k] for r in self.vocab]) for k in ("frame", "word", "score", "area", "packed")}
        V["votes"] = {}
        for off, r in zip(n, self.vocab):
            for a, w, s in r["votes"]:
                V["votes"].setdefault(off + a, []).append((w, s))
        return P, V


def analyse(p, M, clock, rec, jpgs):
    import cv2
    import torch
    import torch.nn.functional as F
    import m3_exp_geometry as geo
    from fast_report import core, segment as sg
    d0, d1, da3s, sams = M["d0"], M["d1"], M["da3"], M["sam"]
    site, words, held = p["site"], list(p["words"]), set(p["held"])
    run_dir = Path("/v/out") / p["run"] / site
    rec["errors"] = {}

    class guard:  # one failed measure is recorded and the rest still run
        def __init__(self, name):
            self.name = name

        def __enter__(self):
            return self

        def __exit__(self, kind, value, tb):
            if kind is not None and issubclass(kind, Exception):
                rec["errors"][self.name] = "".join(traceback.format_exception(kind, value, tb))[-3000:]
                return True
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------- decode, sharpness, cuts (a-core's rules) ----------
    with clock.stage("decode"):
        Path("/tmp/in.mp4").write_bytes(p["mp4"])
        cap = cv2.VideoCapture("/tmp/in.mp4")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
        cap.release()
    n_all = len(frames)
    lo, hi = p.get("range") or (0, n_all)
    with clock.stage("cuts"), ThreadPoolExecutor(CPU) as pool:
        gs = list(pool.map(core.gray_sharp, frames[lo:hi]))
        gray, sharp = [g for g, _ in gs], {lo + i: s for i, (_, s) in enumerate(gs)}
        n, ch, span = len(gray), 64, core.dsc.SPAN
        parts = list(pool.map(lambda a: core.measure_chunk(gray[max(0, a - 2):min(n, a + ch + span + 1)], a, min(a + ch, n), max(0, a - 2),
                                                         min(n, a + ch + span + 1)), range(0, n, ch)))
        cuts = core.cuts_from(core.stitch(parts), n)
    shots = [(a + lo, b + lo) for a, b in cuts["segments"] if b - a + 1 >= core.MIN_SHOT]
    rec.update(fps=fps, frames=n_all, range=[lo, hi], cuts={k: cuts[k] for k in ("cuts", "fades", "segments")}, shots=shots, held=sorted(held),
               words=words)
    in_shot = lambda f: next((si for si, (a, b) in enumerate(shots) if a <= f <= b), None)  # noqa: E731

    # ---------- frame sets ----------
    b6 = keyset(sharp, lo, hi, 6, held)
    sets = {"b6x3": b6[::3], "b6": b6, "b3": keyset(sharp, lo, hi, 3, held), "b2": keyset(sharp, lo, hi, 2, held),
            "b1": [f for f in range(lo, hi) if f not in held]}
    dur = (hi - lo) / fps

    # ---------- SAM 3 on every frame, nested phases ----------
    seen = set()
    phases = []
    for name in ("b6x3", "b6", "b3", "b2", "b1"):
        fl = [f for f in sets[name] if f not in seen]
        seen |= set(fl)
        phases.append((f"{name} (+{len(fl)})", fl))
    rest = [f for f in range(lo, hi) if f not in seen]  # the held-out frames: people masks for their depth check only
    phases.append(("held-out", rest))
    dense = Dense(sams, words, frames, clock, d0)
    rec["sam3_phases"] = dense.run(phases)
    P, V = dense.gathered()
    rec["sam3_masks"] = {"vocab_in": int(sum(dense.masks_in.values())), "vocab_kept": int(len(V["frame"])),
                         "vocab_in_per_frame_mean": round(float(np.mean(list(dense.masks_in.values()))), 1),
                         "kept_per_frame_mean": round(len(V["frame"]) / max(len(dense.masks_in), 1), 1),
                         "person_floor_detections": int(len(P["frame"])), "packed_cpu_gb": round(V["packed"].nbytes / 1e9, 3)}
    del dense.vocab
    with torch.inference_mode():
        is_p = P["word"] == 0
        dyn = F.max_pool2d(core.union_by_frame(P["frame"][is_p], P["mask"][is_p], n_all)[:, None].float(), 5, 1, 2)[:, 0] > 0
        floor = core.union_by_frame(P["frame"][~is_p], P["mask"][~is_p], n_all)
    for d in (d0, d1):
        with torch.cuda.device(d):
            torch.cuda.empty_cache()

    # ---------- geometry ----------
    geo_out = {}  # (config, shot) -> {frames, depth, K, c2w, colors} on GPU 0, DA3 units

    def da3_run(dev, cfg, si, fl, tag=None):
        with torch.cuda.device(dev), clock.stage(f"da3.shot{si}@gpu{dev.index}", gpu=dev.index, n={"views": len(fl), "config": tag or cfg}, sync=True):
            kf = torch.from_numpy(np.stack([frames[f] for f in fl])).to(dev)
            with torch.inference_mode():
                g = da3s[dev].shot(kf)
            del kf
        return {"frames": list(fl), **{k: v.to(d0) for k, v in g.items()}}

    shot_frames = {si: [f for f in sets["b1"] if a <= f <= b] for si, (a, b) in enumerate(shots)}
    main = max(range(len(shots)), key=lambda si: len(set(range(shots[si][0], shots[si][1] + 1)) & set(p["ref_frames"])))
    chunks = [("G30c", si, ci, shot_frames[si][s:s + CHUNK_VIEWS]) for si in shot_frames for ci, s in enumerate(chunk_starts(len(shot_frames[si])))]
    inter = {si: interleaved([f for f in sets["b6"] if f in set(fl)], [f for f in fl if f not in set(sets["b6"])]) for si, fl in shot_frames.items()}
    chunks += [("G30i", si, ci, fl) for si in inter for ci, fl in enumerate(inter[si])]
    cq = queue.Queue()
    for c in chunks:
        cq.put(c)
    chunk_out, geo_err = {}, []

    def motion_scores(g, base):
        """per consecutive pair of base frames (G5 anchors): rotation (deg) + translation over median depth (deg)."""
        row = {f: i for i, f in enumerate(g["frames"])}
        out = []
        for a, b in zip(base, base[1:]):
            ca, cb = g["c2w"][row[a]].double().cpu().numpy(), g["c2w"][row[b]].double().cpu().numpy()
            z = float(g["depth"][row[a]][g["depth"][row[a]] > 0].median())
            out.append(geo.angle_deg(ca[:3, :3].T @ cb[:3, :3]) + np.degrees(np.linalg.norm(ca[:3, 3] - cb[:3, 3]) / max(z, 1e-6)))
        return out

    def gpu_worker(dev, jobs):
        try:
            for job in jobs:
                job()
            while True:
                try:
                    kind, si, ci, fl = cq.get_nowait()
                except queue.Empty:
                    return
                chunk_out[(kind, si, ci)] = da3_run(dev, kind, si, fl, f"{kind} chunk {ci}")
        except BaseException:  # noqa: BLE001
            geo_err.append(traceback.format_exc())

    def g5_then_motion():
        for si, fl in shot_frames.items():
            geo_out[("G5", si)] = da3_run(d0, "G5", si, [f for f in sets["b6"] if f in set(fl)])
        for si in shot_frames:
            g = geo_out[("G5", si)]
            base = g["frames"]
            budget = len([f for f in sets["b3"] if in_shot(f) == si]) - len(base)
            fl = adaptive(base, motion_scores(g, base), budget, held)
            geo_out[("Gmotion", si)] = da3_run(d0, "Gmotion", si, fl)

    def g10():
        for si, fl in shot_frames.items():
            geo_out[("G10", si)] = da3_run(d1, "G10", si, [f for f in sets["b3"] if f in set(fl)])

    def g30_single():  # feasibility: every frame of the longest shot in ONE forward (GPU 1, before its share of chunks)
        if not p.get("g30_single"):
            return
        si = main  # the reference shot (Walmart's longest shot is not the one DROID covers)
        try:
            geo_out[("G30single", si)] = da3_run(d1, "G30single", si, shot_frames[si])
            rec["g30_single"] = {"shot": si, "views": len(shot_frames[si]), "ok": True}
        except torch.cuda.OutOfMemoryError as e:
            rec["g30_single"] = {"shot": si, "views": len(shot_frames[si]), "ok": False, "error": repr(e)[:300]}
            with torch.cuda.device(d1):
                torch.cuda.empty_cache()

    ths = [threading.Thread(target=gpu_worker, args=(d0, [g5_then_motion])), threading.Thread(target=gpu_worker, args=(d1, [g10, g30_single]))]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if geo_err:
        raise RuntimeError(geo_err[0])

    # chunks -> G30c (consecutive, chained on shared frames) and G30i (interleaved, each onto G5 by its anchors)
    def pts(g, rows, step=8):
        """world points of rows on a step grid (DA3 units), and a validity mask; people and depth edges out."""
        dd = g["depth"][rows].clone()
        geo.edge_filter(dd)
        fr_ = torch.tensor([g["frames"][r] for r in rows], device=d0)
        dd[dyn[fr_]] = 0
        sub = dd[:, step // 2::step, step // 2::step]
        f, vy, vx = torch.nonzero(sub > 0, as_tuple=True)
        vy, vx = vy * step + step // 2, vx * step + step // 2
        world = sg.backproject(g["depth"][rows], g["K"][rows], g["c2w"][rows], f, vy, vx, 1)
        key = (f * 10 ** 6 + vy * 1000 + vx).cpu().numpy()
        return world.double().cpu().numpy(), key

    def align_onto(g, ref):
        """Sim3 (depth points of the shared frames) that brings chunk g onto ref; returns the moved chunk and fit stats."""
        shared = sorted(set(g["frames"]) & set(ref["frames"]))
        rg = [g["frames"].index(f) for f in shared]
        rr = [ref["frames"].index(f) for f in shared]
        a, ka = pts(g, rg)
        b, kb = pts(ref, rr)
        common, ia, ib = np.intersect1d(ka, kb, return_indices=True)
        s, R, t, res = sim3_points(a[ia], b[ib])
        zb = np.linalg.norm(b[ib] - ref["c2w"][rr].double().cpu().numpy()[:, :3, 3].mean(0), axis=1)
        T = torch.eye(4, dtype=torch.float64, device=d0)
        T[:3, :3], T[:3, 3] = torch.from_numpy(s * R), torch.from_numpy(t)
        c2w = T[None] @ g["c2w"].double()
        c2w[:, :3, :3] /= s
        moved = {**g, "c2w": c2w.float(), "depth": g["depth"] * s}
        return moved, {"shared_frames": len(shared), "points": int(len(common)), "scale": round(float(s), 4),
                       "residual_median_rel": round(float(np.median(res / np.maximum(zb, 1e-6))), 4)}

    rec["chunks"] = {}
    for si in shot_frames:
        cs = [chunk_out[("G30c", si, ci)] for ci in range(len(chunk_starts(len(shot_frames[si]))))]
        chained, fits_c, fits_a = [cs[0]], [], []
        for g in cs[1:]:
            ref = {"frames": [f for c in chained for f in c["frames"]], **{k: torch.cat([c[k] for c in chained]) for k in ("depth", "K", "c2w")}}
            moved, fit = align_onto(g, ref)
            chained.append(moved)
            fits_c.append(fit)
        g5 = geo_out[("G5", si)]
        anchored = []
        for ci in range(len(inter[si])):
            moved, fit = align_onto(chunk_out[("G30i", si, ci)], g5)
            anchored.append(moved)
            fits_a.append(fit)
        for name, parts in (("G30c", chained), ("G30i", anchored)):
            rows, taken = {}, set()
            for ci, c in enumerate(parts):
                for r, f in enumerate(c["frames"]):
                    if name == "G30i" and f in g5["frames"]:
                        rows[f] = ("g5", g5["frames"].index(f))
                    elif f not in taken:
                        rows[f] = (ci, r)
                    taken.add(f)
            fl = sorted(rows)
            src = lambda key, f: (g5 if rows[f][0] == "g5" else parts[rows[f][0]])[key][rows[f][1]]  # noqa: E731
            geo_out[(name, si)] = {"frames": fl, **{k: torch.stack([src(k, f) for f in fl]) for k in ("depth", "K", "c2w", "colors")}}
        rec["chunks"][si] = {"consecutive_starts": chunk_starts(len(shot_frames[si])), "consecutive_views": [len(c["frames"]) for c in cs],
                             "consecutive_chained_fits": fits_c, "interleaved_views": [len(c) for c in inter[si]], "interleaved_onto_g5_fits": fits_a}
    chunk_out.clear()

    # per config and shot: floor-plane scale (a-core), metric cameras, TSDF
    configs_geo = sorted({c for c, _ in geo_out})
    ref_K = np.array(p["raster_K"], float)
    K_ref = np.array([[ref_K[0] * 1.5, 0, 160 + (ref_K[2] + .5) * 1.5 - .5], [0, ref_K[1] * 1.5, (ref_K[3] + .5) * 1.5 - .5], [0, 0, 1]])
    cams, geo_rec, meshes, scale = {}, {}, {}, {}
    rec["main_shot"] = main
    for cfg in configs_geo:
        cams[cfg], geo_rec[cfg] = [], {}
        for si in range(len(shots)):
            g = geo_out.get((cfg, si))
            if g is None:
                continue
            fi = torch.tensor(g["frames"], device=d0)
            with torch.inference_mode():
                plane = core.floor_plane(g["depth"], g["K"], g["c2w"], floor[fi])
            mpu = core.CAMERA_HEIGHT_M / plane["camera_height_units"] if plane else 1.
            scale[(cfg, si)] = (plane, mpu)
            c2w_m = g["c2w"].clone()
            c2w_m[:, :3, 3] *= mpu
            cams[cfg].append({"index": si, "frames": list(shots[si]), "keyframes": g["frames"], "c2w_m": c2w_m.cpu().numpy().round(5).tolist(),
                              "scale_status": "estimated" if plane else "uncalibrated"})
            with guard(f"tsdf {cfg} shot {si}"):
                with torch.inference_mode(), clock.stage(f"tsdf.shot{si}", gpu=0, n={"views": len(g["frames"]), "config": cfg}, sync=True):
                    d = g["depth"] * mpu
                    d[dyn[fi]] = 0
                    geo.edge_filter(d)
                    ts = geo.fuse(d, g["K"].cpu().numpy(), c2w_m.cpu().numpy().astype(np.float64), g["colors"])
                geo_rec[cfg][si] = {"views": len(g["frames"]), "tsdf_points": ts["n_points"], "tsdf_triangles": ts["n_triangles"], "tsdf_timing": ts["timing"],
                                    "metres_per_unit": mpu, "floor": {k: v for k, v in (plane or {}).items() if k not in ("normal", "point")}}
                if si == main:
                    meshes[cfg] = ts["mesh"]
                    path = run_dir / f"tsdf-{cfg}-shot{si}.ply"
                    import open3d as o3d
                    o3d.io.write_triangle_mesh(str(path), ts["mesh"])
                    geo_rec[cfg][si]["mesh_volume_path"] = f"panoptes-fx-x1-fps:/{p['run']}/{site}/{path.name}"
    rec["cameras"], rec["geometry"] = cams, geo_rec

    # held-out depth: our mesh rendered from DROID's camera (moved into our frame by this config's own Sim3) vs the
    # delivered DA3-posed depth there (E1's render_depth_eval rule, raster grid 320x240 of the 4:3 centre)
    droid = np.asarray(p["droid"], np.float64)
    ref_frames = set(p["ref_frames"])
    held_main = [h for h in p["held"] if shots[main][0] <= h <= shots[main][1]]
    gx, gy = np.meshgrid(geo.GX, geo.GY)
    pix = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)])
    u_grid = np.clip((gx.ravel() * 504 / 1280).astype(int), 0, 503)
    v_grid = np.clip((gy.ravel() * 280 / 720).astype(int), 0, 279)

    def sim3_to_droid(cfg):
        cam = next(c for c in cams[cfg] if c["index"] == main)
        keys = [k for k in cam["keyframes"] if k in ref_frames]
        ours = np.array([c for k, c in zip(cam["keyframes"], cam["c2w_m"]) if k in ref_frames], np.float64)
        target = droid[keys].copy()
        target[:, :3, 3] *= p["mpn"]
        return geo.align_sim3(ours, target)

    def droid_cam(h, s, R, t):
        c = np.eye(4)
        c[:3, :3] = R.T @ droid[h][:3, :3]
        c[:3, 3] = R.T @ (droid[h][:3, 3] * p["mpn"] - t) / s
        return c

    def scene_of(mesh):
        import open3d as o3d
        import open3d.core as o3c
        sc = o3d.t.geometry.RaycastingScene()
        sc.add_triangles(o3c.Tensor(np.asarray(mesh.vertices, np.float32)), o3c.Tensor(np.asarray(mesh.triangles, np.uint32)))
        return sc

    def cast(sc, c2w, K, pixels):
        import open3d.core as o3c
        d = c2w[:3, :3] @ (np.linalg.inv(K) @ pixels)
        rays = np.concatenate([np.repeat(c2w[:3, 3][None], d.shape[1], 0), d.T], 1).astype(np.float32)
        return sc.cast_rays(o3c.Tensor(rays))

    ref_depth = np.asarray(p["ref_depth"], np.float32)
    ref_valid = np.asarray(p["ref_valid"], bool)
    held_rec, sims = {}, {}
    with guard("held-out depth"), clock.stage("heldout.eval"):
        for cfg, mesh in meshes.items():
            s, R, t = sim3_to_droid(cfg)
            sims[cfg] = (s, R, t)
            sc = scene_of(mesh)
            rel, cover, total = [], 0, 0
            for h in held_main:
                z = cast(sc, droid_cam(h, s, R, t), K_ref, pix)["t_hit"].numpy() * s / p["mpn"]  # our metres -> DROID native
                j = p["held"].index(h)
                ref = ref_depth[j].reshape(-1)
                person = dyn[h].cpu().numpy()[v_grid, u_grid]
                ok = (ref > 0) & ref_valid[j].reshape(-1) & ~person
                hit = ok & np.isfinite(z)
                total, cover = total + ok.sum(), cover + hit.sum()
                rel.append(np.abs(z[hit] - ref[hit]) / ref[hit])
            rel = np.concatenate(rel) if rel else np.zeros(0)
            held_rec[cfg] = {"frames": len(held_main), "coverage": round(float(cover / max(total, 1)), 4), "hole_fraction": round(1 - float(cover / max(total, 1)), 4),
                             "absrel_median": round(float(np.median(rel)), 4) if len(rel) else None,
                             "within_5pct": round(float((rel < .05).mean()), 4) if len(rel) else None,
                             "within_10pct": round(float((rel < .10).mean()), 4) if len(rel) else None, "sim3_scale_ours_to_droid_m": round(float(s), 4)}
    rec["held_out_depth"] = held_rec

    # ---------- dense geometry for objects / people: G5 frames as they are, the rest from G30i (G5's world) ----------
    dense_geo = {si: geo_out[("G30i", si)] for si in shot_frames}
    drow = {si: {f: i for i, f in enumerate(g["frames"])} for si, g in dense_geo.items()}
    g5scale = {si: scale[("G5", si)] for si in shot_frames}

    def geo_rows(si, dev):
        g, mpu = dense_geo[si], g5scale[si][1]
        cache = {}

        def get(f):
            if f not in cache:
                r = drow[si][f]
                c = g["c2w"][r].to(dev).clone()
                c[:3, 3] *= mpu
                cache[f] = (g["depth"][r].to(dev) * mpu, g["K"][r].to(dev), c)
            return cache[f]
        return get

    # ---------- adaptive frame sets (scores from G5 only) ----------
    def novelty(si, base):
        g = geo_out[("G5", si)]
        row = {f: i for i, f in enumerate(g["frames"])}
        get = lambda f: (g["depth"][row[f]], g["K"][row[f]], g["c2w"][row[f]])  # noqa: E731
        out = []
        for a, b in zip(base, base[1:]):
            share = []
            for x, y in ((a, b), (b, a)):
                zx, Kx, cx = get(x)
                zy, Ky, cy = get(y)
                v, u = torch.nonzero(zx > 0, as_tuple=True)
                world = sg.backproject(zx[None], Kx[None], cx[None], torch.zeros_like(v), v, u, 1)
                key = sg.splat(world, torch.ones_like(v), Ky, torch.linalg.inv(cy), sg.DA3_HW)
                seen = sg.finish(key, sg.DA3_HW) >= 0
                share.append(1 - float((seen & (zy > 0)).sum()) / max(float((zy > 0).sum()), 1))
            out.append(max(share))
        return out

    with torch.inference_mode(), clock.stage("adaptive.sets"):
        for name, base_name, target_name in (("a1", "b6x3", "b6"), ("a2", "b6", "b3")):
            for kind in ("motion", "novel"):
                fl = []
                for si in shot_frames:
                    base = [f for f in sets[base_name] if in_shot(f) == si]
                    budget = len([f for f in sets[target_name] if in_shot(f) == si]) - len(base)
                    sc_ = motion_scores(geo_out[("G5", si)], base) if kind == "motion" else novelty(si, base)
                    fl += adaptive(base, sc_, budget, held)
                sets[f"{name}-{kind}"] = fl
        persons_at = set(P["frame"][P["word"] == 0].cpu().numpy().tolist())
        fl = []
        for si in shot_frames:
            base = [f for f in sets["b6"] if in_shot(f) == si]
            b2 = [f for f in sets["b2"] if in_shot(f) == si]
            fl += sorted(set(base) | {f for a, b in zip(base, base[1:]) if a in persons_at or b in persons_at for f in b2 if a < f < b})
        sets["p-adapt"] = fl
    rec["sets"] = {k: {"frames": len(v), "frames_in_geometry_shots": len([f for f in v if in_shot(f) is not None]), "fps_equivalent": round(len(v) / dur, 2)}
                   for k, v in sets.items()}
    rec["geometry_sets"] = {cfg: sum(len(geo_out[(cfg, si)]["frames"]) for si in shot_frames if (cfg, si) in geo_out) for cfg in configs_geo}

    # ---------- objects per configuration (lift on GPU 1) ----------
    obj_cfgs = ["b6x3", "b6", "b3", "b2", "b1", "a1-motion", "a1-novel", "a2-motion", "a2-novel"]
    objs, members_of, codes_of = {}, {}, {}
    rec["objects"] = {}
    for cfg in obj_cfgs:
        objs[cfg], members_of[cfg], codes_of[cfg] = [], [], []
        with guard(f"objects {cfg}"):
            fset = set(sets[cfg])
            found, members, codes = [], [], []
            stats_all, lift_s = [], 0.
            for si in shot_frames:
                fl = sorted(f for f in fset if f in drow[si])
                if len(fl) < 2:
                    continue
                local = {f: j for j, f in enumerate(fl)}
                sel = np.flatnonzero(np.isin(V["frame"], fl))
                with torch.inference_mode(), clock.stage("lift", gpu=1, n={"config": cfg, "shot": si, "masks": int(len(sel)), "frames": len(fl)}, sync=True):
                    m2 = torch.cat([unpack(V["packed"][sel[i:i + 8192]], d1)[:, ::2, ::2] for i in range(0, len(sel), 8192)]) if len(sel) else \
                        torch.zeros((0, 140, 252), dtype=torch.bool, device=d1)
                    get = geo_rows(si, d1)
                    rows = [get(f) for f in fl]
                    depth_m, K, c2w_m = (torch.stack([r[k] for r in rows]) for k in range(3))
                    frame_of = torch.tensor([local[int(f)] for f in V["frame"][sel]], device=d1, dtype=torch.long)
                    dyn2 = dyn[torch.tensor(fl, device=d0)][:, ::2, ::2].to(d1)
                    torch.cuda.synchronize(d1)
                    t_l = time.perf_counter()  # the lift itself (a-core's 'lift' stage); unpacking and gathering are this sweep's own
                    comp, arr, st, ex = lift_big(m2, frame_of, depth_m, K, c2w_m, dyn2)
                    lift_s += time.perf_counter() - t_l
                    del m2, depth_m, rows
                stats_all.append({"shot": si, **st})
                if arr is None:
                    continue
                area = V["area"][sel]
                lifted_pos = {int(g): i for i, g in enumerate(ex["lifted"])}
                for c in np.flatnonzero(arr["frames"] >= sg.CONFIRMED):
                    mem = np.flatnonzero(comp == c)
                    gi = sel[mem]
                    vv = {}
                    for x in gi:
                        for w_, s_ in V["votes"].get(int(x), []):
                            vv[words[w_]] = vv.get(words[w_], 0.) + s_
                    best = gi[np.argmax(V["score"][gi] * np.sqrt(area[mem]))]
                    ve = ex["view_extent"][[lifted_pos[int(m)] for m in mem]]
                    eu = robust_extents(ex["codes"][c], sg.LIFT_VOXEL)
                    e = np.nanmedian(ve, 0) if np.isfinite(ve).all(1).any() else eu  # the median view's extents; the union's when no view has 8 points
                    mc = ex["mask_centroid"][[lifted_pos[int(m)] for m in mem]]
                    dev_ = np.linalg.norm(mc - np.median(mc, 0), axis=1)
                    nfr = len(set(V["frame"][gi].tolist()))
                    found.append({"id": f"{cfg}-{si}-{len(found)}", "shot": si, "word": sg.name(vv, words), "frames": int(arr["frames"][c]),
                                  "masks": int(len(mem)), "voxels": int(arr["voxels"][c]), "centroid_m": arr["centroid"][c].round(3).tolist(),
                                  "extent_m": e.round(3).tolist(), "extent_union_m": eu.round(3).tolist(), "aabb_m": [(arr["lo"][c] - sg.LIFT_VOXEL / 2).round(3).tolist(), (arr["hi"][c] + sg.LIFT_VOXEL / 2).round(3).tolist()],
                                  "best": [int(V["frame"][best]), int(best)], "first_frame": int(V["frame"][gi].min()),
                                  "view_centroid_spread_m": round(float(np.median(dev_)), 4) if nfr >= 3 else None,
                                  "small": bool(e[0] < SMALL_M), "thin_long": bool(e[0] >= THIN_MIN_M and e[0] >= THIN_RATIO * e[1])})
                    members.append(gi)
                    codes.append(ex["codes"][c])
            objs[cfg], members_of[cfg], codes_of[cfg] = found, members, codes
            spread = [o["view_centroid_spread_m"] for o in found if o["view_centroid_spread_m"] is not None]
            rel_spread = [o["view_centroid_spread_m"] / max(o["extent_m"][0], 1e-6) for o in found if o["view_centroid_spread_m"] is not None]
            rec["objects"][cfg] = {"objects": len(found), "small": sum(o["small"] for o in found), "thin_long": sum(o["thin_long"] for o in found),
                                   "views_mean": round(float(np.mean([o["frames"] for o in found])), 2) if found else None,
                                   "views_median": float(np.median([o["frames"] for o in found])) if found else None,
                                   "view_centroid_spread_m": {"median": round(float(np.median(spread)), 4), "p90": round(float(np.percentile(spread, 90)), 4),
                                                              "objects": len(spread)} if spread else None,
                                   "view_centroid_spread_over_extent_median": round(float(np.median(rel_spread)), 4) if rel_spread else None,
                                   "lift_s": round(lift_s, 3), "lift": stats_all, "segmented_frames": len([f for f in fset if in_shot(f) is not None])}
            torch.cuda.empty_cache()

    # new objects vs the current rate (voxel share), per configuration; stability of the objects every rate keeps
    with guard("new objects"):
        for cfg in obj_cfgs:
            cur = {si: np.unique(np.concatenate([c for o, c in zip(objs["b6x3"], codes_of["b6x3"]) if o["shot"] == si] or [np.zeros(0, np.int64)]))
                   for si in shot_frames}
            new = []
            for o, c in zip(objs[cfg], codes_of[cfg]):
                share = float(np.isin(c, cur[o["shot"]]).mean())
                o["share_in_current"] = round(share, 3)
                if share < NEW_SHARE:
                    new.append(o)
            kept_ = [o["view_centroid_spread_m"] for o in objs[cfg] if o["share_in_current"] >= .5 and o["view_centroid_spread_m"] is not None]
            words_new = {}
            for o in new:
                words_new[o["word"]] = words_new.get(o["word"], 0) + 1
            rec["objects"].setdefault(cfg, {}).update(new_vs_current=len(new), new_small=sum(o["small"] for o in new), new_thin_long=sum(o["thin_long"] for o in new),
                                       new_with_3plus_views=sum(o["frames"] >= 3 for o in new),
                                       new_words_top=sorted(words_new.items(), key=lambda x: -x[1])[:15],
                                       spread_of_objects_shared_with_current_m=round(float(np.median(kept_)), 4) if kept_ else None)
    rec["object_lists"] = {cfg: [{k: o[k] for k in ("id", "shot", "word", "centroid_m", "frames", "extent_m", "extent_union_m", "small", "thin_long", "first_frame")} | {"share_in_current": o.get("share_in_current")}
                                 for o in objs[cfg]] for cfg in obj_cfgs}

    # contact sheets: objects found at 30 fps (every frame) / 5 fps with no counterpart at the current rate
    def crop(o):
        f, gi = o["best"]
        img = frames[f].copy()
        mk = cv2.resize(unpack(V["packed"][gi:gi + 1], "cpu")[0].numpy().astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(mk)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        half = max(ys.max() - ys.min(), xs.max() - xs.min(), 64) * .75
        y0, y1, x0, x1 = int(max(0, cy - half)), int(min(img.shape[0], cy + half)), int(max(0, cx - half)), int(min(img.shape[1], cx + half))
        cv2.drawContours(img, cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
        tile = cv2.resize(img[y0:y1, x0:x1], (180, 180), interpolation=cv2.INTER_AREA)
        tile = cv2.copyMakeBorder(tile, 0, 34, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(tile, (o["word"] or "?")[:22], (3, 194), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(tile, f"{o['frames']}v {o['extent_m'][0]:.2f}m est f{f}", (3, 209), cv2.FONT_HERSHEY_SIMPLEX, .38, (60, 60, 60), 1, cv2.LINE_AA)
        return tile

    for cfg in ("b1", "b6"):
        with guard(f"contact sheet {cfg}"):
            new = sorted([o for o in objs[cfg] if o["share_in_current"] < NEW_SHARE], key=lambda o: (-o["frames"], -o["voxels"]))[:30]
            if new:
                tiles = [crop(o) for o in new]
                tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % 6)
                sheet = np.concatenate([np.concatenate(tiles[i:i + 6], 1) for i in range(0, len(tiles), 6)], 0)
                jpgs[f"new-objects-{cfg}-vs-b6x3.jpg"] = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 78])[1].tobytes()

    # ---------- outlines at in-between frames: projected ('pair') and held (last keyframe), vs every-frame masks ----------
    used = set().union(*(set(v) for k, v in sets.items() if k != "b1"))
    cand = [f for si in shot_frames for f in dense_geo[si]["frames"] if f not in used]
    evalf = [cand[int((i + .5) * len(cand) / min(EVAL_FRAMES, len(cand)))] for i in range(min(EVAL_FRAMES, len(cand)))] if cand else []
    by_frame = {}
    for cfg in obj_cfgs:
        bf = {}
        for oi, gi in enumerate(members_of[cfg]):
            for x in gi:
                bf.setdefault(int(V["frame"][x]), {}).setdefault(oi, []).append(int(x))
        by_frame[cfg] = bf

    def idmap(cfg, q, dev):
        objs_q = by_frame[cfg].get(q)
        if not objs_q:
            return torch.zeros(sg.DA3_HW, dtype=torch.long, device=dev)
        ids = list(objs_q)
        allm = [x for o in ids for x in objs_q[o]]
        m = unpack(V["packed"][allm], dev)
        owner = torch.tensor([j for j, o in enumerate(ids) for _ in objs_q[o]], device=dev)
        per = torch.zeros((len(ids), *sg.DA3_HW), dtype=torch.int32, device=dev).index_add_(0, owner, m.int()) > 0
        return per_object_ids(per, ids, dev)

    def score(ref, pred, person):
        rid = torch.unique(ref)
        rid = rid[rid > 0]
        pid = torch.unique(pred)
        pid = pid[pid > 0]
        out = []
        if not len(rid):
            return out
        R = (ref.flatten()[None] == rid[:, None]).float()
        area = R.sum(1)
        on_p = (R @ person.flatten().float()) / area
        cov = (R @ (pred.flatten() > 0).float()) / area
        if len(pid):
            Pm = (pred.flatten()[None] == pid[:, None]).float()
            inter = R @ Pm.T
            best = (inter / (area[:, None] + Pm.sum(1)[None] - inter)).max(1).values
        else:
            best = torch.zeros_like(area)
        for a, o, c, b in zip(area.tolist(), on_p.tolist(), cov.tolist(), best.tolist()):
            if a >= MIN_REF_PX and o < .5:
                out.append((b, a, c))
        return out

    out_rows = {}
    with guard("outlines"), torch.inference_mode(), clock.stage("outlines", gpu=1, n={"frames": len(evalf)}):
        refs = {e: idmap("b1", e, d1) for e in evalf}
        for cfg in [c for c in obj_cfgs if c != "b1"]:
            rows, union = {"projected": [], "held": []}, {"projected": [0, 0], "held": [0, 0]}
            t_o = time.perf_counter()
            for e in evalf:
                si = in_shot(e)
                ks = [f for f in sets[cfg] if in_shot(f) == si]
                before, after = [k for k in ks if k < e][-1:], [k for k in ks if k > e][:1]
                if not before and not after:
                    continue
                srcs = sorted(before + after, key=lambda k: (abs(k - e), k))
                get = geo_rows(si, d1)
                lab = project_to(srcs, lambda k: idmap(cfg, k, d1), get, e)
                person = dyn[e].to(d1)
                lab[person] = -1
                held_lab = idmap(cfg, (before or after)[0], d1)
                for kind, pred in (("projected", lab.clamp(min=0)), ("held", held_lab)):
                    rows[kind] += score(refs[e], pred, person)
                    a_, b_ = (refs[e] > 0) & ~person, (pred > 0) & ~person
                    union[kind][0] += int((a_ & b_).sum())
                    union[kind][1] += int((a_ | b_).sum())
            res = {}
            for kind, r in rows.items():
                a = np.array(r, float).reshape(-1, 3)
                res[kind] = {"regions": len(a), "iou_mean": round(float(a[:, 0].mean()), 4) if len(a) else None,
                             "iou_area_weighted": round(float((a[:, 0] * a[:, 1]).sum() / a[:, 1].sum()), 4) if len(a) else None,
                             "coverage_area_weighted": round(float((a[:, 2] * a[:, 1]).sum() / a[:, 1].sum()), 4) if len(a) else None,
                             "object_pixels_iou": round(union[kind][0] / max(union[kind][1], 1), 4)}
            res["s"] = round(time.perf_counter() - t_o, 2)
            out_rows[cfg] = res
    rec["outlines"] = {"eval_frames": evalf, "reference": "every-frame (b1) objects' SAM 3 masks at the eval frame, static (< 50% on a person), >= 1200 px at 1280x720; "
                                                          "object_pixels_iou: all object pixels vs all object pixels (no identity, people out)",
                       "rows": out_rows}

    # ---------- people: PeopleLoop at 5 / 10 / 15 fps and person-adaptive, same G5 floor and scale ----------
    rec["people"] = {}
    for cfg in ("b6", "b3", "b2", "p-adapt"):
        with guard(f"people {cfg}"):
            tracks_all, findings_all, nrows, speeds = [], [], 0, []
            t_p = time.perf_counter()
            for si in shot_frames:
                fl = sorted(f for f in sets[cfg] if f in drow[si])
                if len(fl) < 2:
                    continue
                get = geo_rows(si, d0)
                rows = [get(f) for f in fl]
                depth_m, K, c2w_m = (torch.stack([r[k] for r in rows]) for k in range(3))
                g = {"colors": dense_geo[si]["colors"][[drow[si][f] for f in fl]], "K": K}
                plane, mpu = g5scale[si]
                with clock.stage("people", n={"config": cfg, "shot": si, "keyframes": len(fl)}):
                    pm = core.person_masks(P, {f: j for j, f in enumerate(fl)})
                    tracks, prow, findings, _ = core.people_shot(si, fl, fps, g, depth_m, c2w_m, pm, plane, mpu)
                tracks_all += [[{"frame": q["frame"], "xyz": q["xyz"]} for q in pts_] for pts_ in tracks.values()]
                findings_all += [{**f_, "shot": si} for f_ in findings]
                nrows += len(prow)
                speeds += [r_["speedMps"] for r_ in prow if r_.get("speedMps") is not None]
            rec["people"][cfg] = {"tracks": tracks_all, "rules": findings_all, "detections": nrows, "s": round(time.perf_counter() - t_p, 3),
                                  "keyframes": len([f for f in sets[cfg] if in_shot(f) is not None]),
                                  "speed_mps_estimated": {"n": len(speeds), "median": round(float(np.median(speeds)), 3) if speeds else None,
                                                          "p90": round(float(np.percentile(speeds, 90)), 3) if speeds else None}}

    # ---------- fine structures: every geometry config rendered from the same held-out camera, crops on thin objects ----------
    with guard("render compare"), clock.stage("render.compare"):
        thin = [o for o in objs["b1"] if o["thin_long"] and o["shot"] == main and o["frames"] >= 5]
        s5, R5, t5 = sims["G5"]
        picks = []
        for h in held_main:
            cam = droid_cam(h, s5, R5, t5)
            w2c = np.linalg.inv(cam)
            for o in thin:
                lo_, hi_ = np.array(o["aabb_m"][0]), np.array(o["aabb_m"][1])
                corners = np.array([[x, y, z] for x in (lo_[0], hi_[0]) for y in (lo_[1], hi_[1]) for z in (lo_[2], hi_[2])])
                cc = corners @ w2c[:3, :3].T + w2c[:3, 3]
                if (cc[:, 2] <= .2).any():
                    continue
                uv = (cc @ K_ref.T)[:, :2] / cc[:, 2:3]
                x0, y0 = uv.min(0)
                x1, y1 = uv.max(0)
                if x0 < 0 or y0 < 0 or x1 > 1280 or y1 > 720 or max(x1 - x0, y1 - y0) < 40 or x1 - x0 > 420 or y1 - y0 > 320:
                    continue
                picks.append((o["frames"], h, o, (x0, y0, x1, y1)))
        picks.sort(key=lambda x: -x[0])
        chosen, used_o = [], set()
        for _, h, o, box in picks:
            if o["id"] not in used_o and sum(c[0] == h for c in chosen) < 2:
                chosen.append((h, o, box))
                used_o.add(o["id"])
            if len(chosen) == 4:
                break
        order_g = [c for c in ("G5", "G10", "Gmotion", "G30i", "G30c", "G30single") if c in meshes]
        scenes = {cfg: scene_of(meshes[cfg]) for cfg in order_g}
        vcol = {cfg: np.asarray(meshes[cfg].vertex_colors) for cfg in order_g}
        tri = {cfg: np.asarray(meshes[cfg].triangles) for cfg in order_g}
        rows_img = []
        for h, o, (x0, y0, x1, y1) in chosen:
            cx, cy, side = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * .65 + 16
            X0, X1 = int(max(0, cx - side)), int(min(1280, cx + side))
            Y0, Y1 = int(max(0, cy - side)), int(min(720, cy + side))
            xs, ys = np.meshgrid(np.arange(X0, X1) + .0, np.arange(Y0, Y1) + .0)
            pixels = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)])
            tiles = [cv2.resize(frames[h][Y0:Y1, X0:X1], (220, 220), interpolation=cv2.INTER_AREA)]
            for cfg in order_g:
                s, R, t = sims[cfg]
                hit = cast(scenes[cfg], droid_cam(h, s, R, t), K_ref, pixels)
                ids, uvs = hit["primitive_ids"].numpy(), hit["primitive_uvs"].numpy()
                ok = ids != 0xFFFFFFFF
                col = np.full((len(ids), 3), .5)
                tr = tri[cfg][ids[ok]]
                u_, v_ = uvs[ok, 0:1], uvs[ok, 1:2]
                col[ok] = (1 - u_ - v_) * vcol[cfg][tr[:, 0]] + u_ * vcol[cfg][tr[:, 1]] + v_ * vcol[cfg][tr[:, 2]]
                img = (col.reshape(Y1 - Y0, X1 - X0, 3)[..., ::-1] * 255).clip(0, 255).astype(np.uint8)
                tiles.append(cv2.resize(img, (220, 220), interpolation=cv2.INTER_AREA))
            row = np.concatenate([cv2.copyMakeBorder(tl, 0, 22, 2, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for tl in tiles], 1)
            for k, name in enumerate(["photo (held out)"] + order_g):
                cv2.putText(row, name, (6 + k * 224, 236), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(row, f"f{h} {o['word']}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
            rows_img.append(row)
        if rows_img:
            jpgs["fine-structures-geometry.jpg"] = cv2.imencode(".jpg", np.concatenate(rows_img, 0), [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
        rec["render_compare"] = [{"held_frame": h, "object": o["id"], "word": o["word"], "box_px": [round(v, 1) for v in box]} for h, o, box in chosen]

    # a-core's layout at each rate (every vocabulary mask + logits on GPU 0), from the measured mask counts [M x n]
    rec["production_layout_gb"] = {cfg: round(sum(dense.masks_in.get(f, 0) for f in sets[cfg]) * PROD_MASK_BYTES / 2 ** 30, 2) for cfg in sets}


# ---------- local ----------

def payload(site, run, frame_range=None, g30_single=False):
    import fast_report_eval as ev
    ref = ev.reference(site)
    clip = SITES[site]
    fps = json.loads((PHASE2 / "data/clips" / clip / "source-full.json").read_text())["fps"]
    mono_dir = PHASE2 / "runs" / MONO[site]
    have = sorted(int(q.stem) for q in mono_dir.glob("*.npz"))
    seg = ref["segment"]
    lo_, hi_ = frame_range or (0, 10 ** 9)
    held = []
    k = 0
    while seg[0] + (k + .5) * fps <= seg[-1]:
        want = seg[0] + (k + .5) * fps
        near = [f for f in have if abs(f - want) <= 3 and seg[0] < f < seg[-1] and lo_ <= f < hi_]
        if near:
            held.append(min(near, key=lambda f: abs(f - want)))
        k += 1
    z = [np.load(mono_dir / f"{h:05d}.npz") for h in held]
    return {"site": site, "run": run, "mp4": (PHASE2 / "data/clips" / clip / "source-full.mp4").read_bytes(),
            "words": json.loads((PHASE2 / "runs/fb-d-harness-gaps-001/summary.json").read_text())["words"][site]["qwen-v1-5+core"],
            "held": held, "ref_frames": ref["frames"], "droid": np.load(ref["droid"])["poses_c2w"].astype(np.float64), "mpn": ref["mpn"],
            "raster_K": json.loads((PHASE2 / "data/clips" / clip / "clip.json").read_text())["K"],
            "ref_depth": np.stack([x["depth"][::2, ::2] for x in z]).astype(np.float16) if z else np.zeros((0, 240, 320), np.float16),
            "ref_valid": np.stack([x["mask"][::2, ::2] if "mask" in x.files else x["depth"][::2, ::2] > 0 for x in z]) if z else np.zeros((0, 240, 320), bool),
            "range": list(frame_range) if frame_range else None, "g30_single": g30_single, "mono_dir": str(mono_dir.relative_to(PHASE2))}


@app.local_entrypoint()
def main(out: str, sites: str = "me340,samsclub-a2,walmart", frames: str = "", g30_single: str = "me340,samsclub-a2,walmart"):
    import shutil
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run folder
    free = shutil.disk_usage("/System/Volumes/Data").free / 1e9
    assert free > 8, f"{free:.1f} GB free: stop"
    rng = tuple(int(x) for x in frames.split("-")) if frames else None
    calls = {}
    for site in sites.split(","):
        pl = payload(site, out.name, rng, site in g30_single.split(","))
        (out / f"payload-{site}.json").write_text(json.dumps({k: v for k, v in pl.items() if k not in ("mp4", "droid", "ref_depth", "ref_valid", "ref_frames")}, indent=1))
        calls[site] = (time.time(), sweep.spawn(pl))
        print(f"spawned {site}: held-out {len(pl['held'])}, words {len(pl['words'])}", flush=True)
    for site, (t0, call) in calls.items():
        try:
            res = call.get()
        except Exception:  # noqa: BLE001
            (out / f"error-{site}.txt").write_text(traceback.format_exc())
            print(f"{site}: FAILED", flush=True)
            continue
        res["record"]["client_wall_s"] = round(time.time() - t0, 1)
        (out / f"raw-{site}.json").write_text(json.dumps(res["record"]))
        for name, b in res["jpgs"].items():
            (out / f"{site}-{name}").write_bytes(b)
        r = res["record"]
        print(json.dumps({"site": site, "error": (r.get("error") or "")[-1500:], "container_s": r.get("container_s"), "usd": r.get("usd_container_estimate"),
                          "sam3_phases": r.get("sam3_phases"), "flags": [f for f in r["run"]["flags"] if "unknown stage" not in f]}, default=plain)[:4000], flush=True)


# ---------- evaluation (local numpy, fb/d-harness rows) ----------

def stage_rows(rec, pred):
    return [r for r in rec["run"]["stages"] if pred(r)]


def peak(rows):
    """per-GPU peak (GiB) over stage rows (each row's whole-device window peak)."""
    out = [None, None]
    for r in rows:
        for i, v in enumerate(r.get("peak_gb") or []):
            if v is not None:
                out[i] = max(out[i] or 0, v)
    return out


def id_switches(tracks, ref, align, fps):
    """Our tracks vs the delivered people (fast_report_eval's matching: floor distance <= 1 m to the delivered centroid
    interpolated to our frame): identity changes inside one of our tracks, and extra tracks per delivered person."""
    import fast_report_eval as ev
    s, R, t, shot = align
    lo, hi = min(shot["keyframes"]), max(shot["keyframes"])
    up = np.asarray(ref["up"], float)
    up /= np.linalg.norm(up)
    by_entity = {}
    refs = ev.reference_people(ref)
    for f in sorted(refs):
        for e, x in refs[f]:
            by_entity.setdefault(e, []).append((f, x))
    floor = lambda v: float(np.linalg.norm(v - (v @ up) * up))  # noqa: E731
    frames, switches, per_entity, used = set(ref["frames"]), 0, {}, 0
    for k, track in enumerate(tracks):
        seq = []
        for pt in track:
            if not lo <= pt["frame"] <= hi or pt["frame"] not in frames:
                continue
            x = s * (R @ np.asarray(pt["xyz"], float)) + t
            near = [(floor(y - x), e) for e, v in by_entity.items() for y in [ev.at_frame(v, pt["frame"], round(.45 * fps))] if y is not None]
            near = [z for z in near if z[0] <= ev.MATCH_M]
            if near:
                seq.append(min(near)[1])
        if seq:
            used += 1
        switches += sum(a != b for a, b in zip(seq, seq[1:]))
        for e in set(seq):
            per_entity.setdefault(e, set()).add(k)
    return {"our_tracks_matched": used, "id_switches_inside_our_tracks": switches, "delivered_people": len(by_entity),
            "delivered_people_matched": len(per_entity), "extra_tracks_per_delivered_person": sum(len(v) - 1 for v in per_entity.values())}


def object_recalls(objects, ref, align):
    """fast_report_eval.objects3d_row's pool (named-map objects seen >= 3 times in our shot, never in another), split by
    the delivered box size (native AABB x metres_per_native: estimated), and word-matched recall (same_name)."""
    import fast_report_eval as ev
    s, R, t, shot = align
    ents, ref_xyz = ev.reference_objects(ref, set(range(min(shot["keyframes"]), max(shot["keyframes"]) + 1)))
    names = json.loads(ref["names"].read_text())
    ours = [o for o in objects if o["shot"] == shot["index"]]
    if not ours or not len(ents):
        return {}
    xyz = (s * (R @ np.array([o["centroid_m"] for o in ours], float).T)).T + t
    d = np.linalg.norm(ref_xyz[:, None] - xyz[None], axis=2)
    ext = np.array([np.sort(np.diff(np.asarray(e["boundsNative"], float), axis=0)[0])[::-1] * ref["mpn"] for e in ents])
    groups = {"all": np.ones(len(ents), bool), "small_lt_0.3m": ext[:, 0] < SMALL_M,
              "thin_long": (ext[:, 0] >= THIN_MIN_M) & (ext[:, 0] >= THIN_RATIO * ext[:, 1]),
              "named_clear_partial": np.array([names.get(e["entityId"], {}).get("status") in ("clear", "partial") for e in ents])}
    out = {}
    for g, m in groups.items():
        if not m.any():
            out[g] = {"pool": 0}
            continue
        row = {"pool": int(m.sum())}
        for th in (.3, .5):
            row[f"recall_{th}m"] = round(float((d[m].min(1) < th).mean()), 3)
        out[g] = row
    named = [i for i, e in enumerate(ents) if names.get(e["entityId"], {}).get("status") in ("clear", "partial")]
    hit = [any(d[i, j] < .5 and ev.same_name(ours[j]["word"] or "", names[ents[i]["entityId"]]["category"]) for j in range(len(ours))) for i in named]
    out["word_matched_named_0.5m"] = {"pool": len(named), "recall": round(float(np.mean(hit)), 3) if named else None}
    return out


def evaluate(run_dir):
    import fast_report_eval as ev
    run_dir = Path(run_dir)
    result = {"schema": "fx-x1-fps-results-v1", "run_dir": str(run_dir), "sites": {}, "spend_usd_estimate": 0.}
    for raw in sorted(run_dir.glob("raw-*.json")):
        rec = json.loads(raw.read_text())
        site = rec["site"]
        ref = ev.reference(site)
        fps = rec["fps"]
        out = {"errors": rec.get("errors"), "container_error": rec.get("error"), "shots": rec.get("shots"), "main_shot": rec.get("main_shot"),
               "held_out_frames": len(rec.get("held") or []), "sets": rec.get("sets"), "words": len(rec["words"]),
               "hardware": rec["boot"].get("gpus"), "boot_s": rec["boot"].get("ready_s"), "container_s": rec.get("container_s"),
               "usd_container_estimate": rec.get("usd_container_estimate"), "analysis_s": rec["run"].get("elapsed_s"),
               "flags_over_90pct": [f for f in rec["run"]["flags"] if "unknown stage" not in f]}
        result["spend_usd_estimate"] += rec.get("usd_container_estimate") or 0
        # SAM 3: nested phases [M]; per configuration GPU seconds = frames x measured s/frame [M x n]
        ph = rec.get("sam3_phases") or []
        per = {k: stage_rows(rec, lambda r, k=k: r["stage"].startswith(k)) for k in ("sam3.person", "sam3.vocab", "dedupe")}
        frames_done = sum(r["n"].get("frames", 0) for r in per["sam3.person"])
        s_frame = {k: sum(r["s"] for r in v) / max(frames_done, 1) for k, v in per.items()}
        out["sam3"] = {"phases": ph, "frames": frames_done, "wall_s_all_phases": round(sum(r["s"] for r in ph), 2),
                       "gpu_s_per_frame": {k: round(v, 4) for k, v in s_frame.items()},
                       "peak_gib_per_gpu_during_sam3": peak(per["sam3.person"] + per["sam3.vocab"]), "masks": rec.get("sam3_masks"),
                       "production_layout_gib": rec.get("production_layout_gb"),
                       "note": "one pass over every frame; a configuration's SAM 3 time = its frames x the measured GPU s/frame / 2 GPUs [M x n]"}
        sets = rec.get("sets") or {}

        def sam_s(cfg, vocab=True):
            n = sets.get(cfg, {}).get("frames", 0)
            return round(n * (s_frame["sam3.person"] + (s_frame["sam3.vocab"] if vocab else 0)) / 2, 2)
        # geometry
        geo_rows_, align = {}, {}
        for cfg, shots in (rec.get("cameras") or {}).items():
            try:
                cam, scale, al = ev.camera_rows({"cameras": {"shots": shots}}, ref)
            except Exception as e:  # noqa: BLE001
                geo_rows_[cfg] = {"error": repr(e)}
                continue
            align[cfg] = al
            g5keys = {k for sh in rec["cameras"].get("G5", []) for k in sh["keyframes"]}
            sub = [{**sh, "keyframes": [k for k in sh["keyframes"] if k in g5keys],
                    "c2w_m": [c for k, c in zip(sh["keyframes"], sh["c2w_m"]) if k in g5keys]} for sh in shots]
            try:
                cam5 = ev.camera_rows({"cameras": {"shots": [x for x in sub if x["keyframes"]]}}, ref)[0]
            except Exception as e:  # noqa: BLE001
                cam5 = {"error": repr(e)}
            da3 = stage_rows(rec, lambda r, c=cfg: r["stage"].startswith("da3.") and (r["n"].get("config") == c or str(r["n"].get("config", "")).startswith(f"{c} chunk")))
            tsdf = stage_rows(rec, lambda r, c=cfg: r["stage"].startswith("tsdf.") and r["n"].get("config") == c)
            g = (rec.get("geometry") or {}).get(cfg, {}).get(str(rec["main_shot"]), {})
            geo_rows_[cfg] = {"ate": cam, "ate_on_5fps_frames_only": cam5, "scale_vs_reference": scale, "views_all_shots": rec["geometry_sets"].get(cfg),
                              "da3_forwards": len(da3), "da3_gpu_s": round(sum(r["s"] for r in da3), 2), "da3_max_views_per_forward": max([r["n"]["views"] for r in da3] or [0]),
                              "da3_peak_gib_per_gpu": peak(da3), "tsdf_s_all_shots": round(sum(r["s"] for r in tsdf), 3), "tsdf_peak_gib": peak(tsdf),
                              "main_shot_tsdf": {k: g.get(k) for k in ("views", "tsdf_points", "tsdf_triangles", "mesh_volume_path")},
                              "held_out_depth": (rec.get("held_out_depth") or {}).get(cfg)}
        out["geometry"] = geo_rows_
        out["chunks"] = rec.get("chunks")
        out["g30_single"] = rec.get("g30_single")
        # objects
        obj = {}
        for cfg, lst in (rec.get("object_lists") or {}).items():
            row = {k: v for k, v in rec["objects"].get(cfg, {}).items() if k != "lift"}
            row["lift_stats"] = rec["objects"].get(cfg, {}).get("lift")
            if "G5" in align:
                row["harness_objects_3d"] = ev.objects3d_row({"cameras": {"shots": rec["cameras"]["G5"]}, "objects": {"objects": lst}}, ref, align["G5"])
                row["recall_by_delivered_size"] = object_recalls(lst, ref, align["G5"])
            row["sam3_s_2gpu_MxN"] = sam_s(cfg)
            row["production_layout_gib_MxN"] = (rec.get("production_layout_gb") or {}).get(cfg)
            obj[cfg] = row
        out["objects"] = obj
        out["outlines"] = rec.get("outlines", {}).get("rows")
        out["outline_eval_frames"] = len(rec.get("outlines", {}).get("eval_frames") or [])
        # people
        ppl = {}
        base_rules = (rec.get("people") or {}).get("b6", {}).get("rules") or []
        for cfg, pr in (rec.get("people") or {}).items():
            row = {k: pr[k] for k in ("detections", "s", "keyframes", "speed_mps_estimated")}
            row["tracks"] = len(pr["tracks"])
            if "G5" in align:
                row["harness_people"] = ev.people_row({"cameras": {"shots": rec["cameras"]["G5"]}, "people": {"tracks": pr["tracks"], "rules": pr["rules"]}},
                                                      ref, align["G5"], fps)
                row["identity"] = id_switches(pr["tracks"], ref, align["G5"], fps)
                sh = align["G5"][3]
                row["rules_vs_5fps"] = ev.rule_agreement(base_rules, pr["rules"], min(sh["keyframes"]) / fps, max(sh["keyframes"]) / fps) if cfg != "b6" else "baseline"
            row["sam3_person_s_2gpu_MxN"] = sam_s(cfg, vocab=False)
            ppl[cfg] = row
        out["people"] = ppl
        out["render_compare"] = rec.get("render_compare")
        result["sites"][site] = out
    result["spend_usd_estimate"] = round(result["spend_usd_estimate"], 3)
    result["summary"] = summarize(result)
    extra = run_dir / "results-notes.json"  # hand-written context (provenance, spend of earlier runs, conclusions), merged as is
    if extra.exists():
        result.update(json.loads(extra.read_text()))
    (run_dir / "results.json").write_text(json.dumps(result, indent=1, default=plain))
    print(json.dumps({s: {"objects": {c: {k: v.get(k) for k in ("objects", "small", "thin_long", "new_vs_current", "sam3_s_2gpu_MxN")} | {
        "recall_0.5": (v.get("harness_objects_3d") or {}).get("reference_recall_0.5m")} for c, v in o["objects"].items()},
        "geometry": {c: {"ate_m": (v.get("ate") or {}).get("ate_m"), "hole": (v.get("held_out_depth") or {}).get("hole_fraction"),
                         "absrel": (v.get("held_out_depth") or {}).get("absrel_median")} for c, v in o["geometry"].items()},
        "errors": list((o.get("errors") or {}).keys())} for s, o in result["sites"].items()}, indent=1, default=plain))
    return result


# ---------- summary tables (every number from results.json's site rows) ----------

OBJ_ORDER = ["b6x3", "b6", "b3", "b2", "b1", "a1-motion", "a1-novel", "a2-motion", "a2-novel"]
GEO_ORDER = ["G5", "G10", "Gmotion", "G30i", "G30c", "G30single"]
PEOPLE_ORDER = ["b6", "b3", "b2", "p-adapt"]
LABELS = {"b6x3": "current: every 3rd 5 fps keyframe (~1.7 fps; 1.4 at 25 fps)", "b6": "every 5 fps keyframe (4.2 at 25 fps)",
          "b3": "sharpest of 3 (~10 fps; 8.3)", "b2": "sharpest of 2 (~15 fps; 12.5)", "b1": "every frame (30 / 25 fps), held-out frames out",
          "a1-motion": "current + extras by camera motion to the 5 fps frame count", "a1-novel": "current + extras by new content to the 5 fps count",
          "a2-motion": "5 fps + extras by camera motion to the 10 fps count", "a2-novel": "5 fps + extras by new content to the 10 fps count",
          "p-adapt": "5 fps, and 15 fps frames only between keyframes where SAM 3 saw a person",
          "G5": "DA3 any-view on 5 fps keyframes, one forward per shot (production)", "G10": "~10 fps, one forward per shot",
          "Gmotion": "5 fps + extras by camera motion, 10 fps view budget, one forward", "G30i": "every frame: interleaved chunks (all 5 fps frames + every m-th other, <= 300 views) Sim3 onto G5",
          "G30c": "every frame: consecutive 150-view chunks chained by Sim3 on shared frames' depth", "G30single": "every frame of the reference shot in ONE forward"}


def summarize(result):
    sites = result["sites"]
    objects = {cfg: {"meaning": LABELS[cfg], **{s: {
        "fps_equivalent": r["sets"][cfg]["fps_equivalent"], "segmented_frames": r["sets"][cfg]["frames"],
        "recall_0.5m": _dig(r, ("objects", cfg, "harness_objects_3d", "reference_recall_0.5m")),
        "recall_0.3m": _dig(r, ("objects", cfg, "harness_objects_3d", "reference_recall_0.3m")),
        "recall_0.5m_small_delivered": _dig(r, ("objects", cfg, "recall_by_delivered_size", "small_lt_0.3m", "recall_0.5m")),
        "recall_0.5m_thin_delivered": _dig(r, ("objects", cfg, "recall_by_delivered_size", "thin_long", "recall_0.5m")),
        "word_matched_recall_0.5m": _dig(r, ("objects", cfg, "recall_by_delivered_size", "word_matched_named_0.5m", "recall")),
        "objects": _dig(r, ("objects", cfg, "objects")), "small": _dig(r, ("objects", cfg, "small")), "thin_long": _dig(r, ("objects", cfg, "thin_long")),
        "views_mean": _dig(r, ("objects", cfg, "views_mean")), "new_vs_current": _dig(r, ("objects", cfg, "new_vs_current")),
        "new_with_3plus_views": _dig(r, ("objects", cfg, "new_with_3plus_views")),
        "view_centroid_spread_median_m": _dig(r, ("objects", cfg, "view_centroid_spread_m", "median")),
        "outline_iou_area_weighted_projected": _dig(r, ("outlines", cfg, "projected", "iou_area_weighted")),
        "outline_iou_area_weighted_held": _dig(r, ("outlines", cfg, "held", "iou_area_weighted")),
        "outline_object_pixels_iou_projected": _dig(r, ("outlines", cfg, "projected", "object_pixels_iou")),
        "sam3_s_2gpu_MxN": _dig(r, ("objects", cfg, "sam3_s_2gpu_MxN")), "lift_s": _dig(r, ("objects", cfg, "lift_s")),
        "a_core_layout_gib_MxN": _dig(r, ("objects", cfg, "production_layout_gib_MxN"))} for s, r in sites.items()}} for cfg in OBJ_ORDER}
    for cfg, row in objects.items():
        rec = [row[s]["recall_0.5m"] for s in sites if row[s]["recall_0.5m"] is not None]
        row["mean_recall_0.5m_3_videos"] = round(float(np.mean(rec)), 3) if len(rec) == len(sites) else None
        row["recall_gain_per_extra_sam3_s_vs_current"] = {s: (round((row[s]["recall_0.5m"] - objects["b6x3"][s]["recall_0.5m"]) /
                                                                    (row[s]["sam3_s_2gpu_MxN"] - objects["b6x3"][s]["sam3_s_2gpu_MxN"]), 5)
                                                              if cfg != "b6x3" and row[s]["recall_0.5m"] is not None and row[s]["sam3_s_2gpu_MxN"] != objects["b6x3"][s]["sam3_s_2gpu_MxN"] else None)
                                                          for s in sites}
    geometry = {cfg: {"meaning": LABELS[cfg], **{s: {
        "views": _dig(r, ("geometry", cfg, "views_all_shots")), "ate_m": _dig(r, ("geometry", cfg, "ate", "ate_m")),
        "ate_on_5fps_frames_m": _dig(r, ("geometry", cfg, "ate_on_5fps_frames_only", "ate_m")),
        "ate_share_of_path": _dig(r, ("geometry", cfg, "ate", "ate_share_of_path")), "path_m": _dig(r, ("geometry", cfg, "ate", "path_m")),
        "rotation_error_median_deg": _dig(r, ("geometry", cfg, "ate", "rotation_error_deg", "median")),
        "scale_ours_over_reference": _dig(r, ("geometry", cfg, "scale_vs_reference", "ours_over_reference")),
        "held_out_coverage": _dig(r, ("geometry", cfg, "held_out_depth", "coverage")), "held_out_hole_fraction": _dig(r, ("geometry", cfg, "held_out_depth", "hole_fraction")),
        "held_out_absrel_median": _dig(r, ("geometry", cfg, "held_out_depth", "absrel_median")),
        "held_out_within_10pct": _dig(r, ("geometry", cfg, "held_out_depth", "within_10pct")),
        "tsdf_points_reference_shot": _dig(r, ("geometry", cfg, "main_shot_tsdf", "tsdf_points")),
        "da3_gpu_s": _dig(r, ("geometry", cfg, "da3_gpu_s")), "da3_forwards": _dig(r, ("geometry", cfg, "da3_forwards")),
        "da3_max_views_per_forward": _dig(r, ("geometry", cfg, "da3_max_views_per_forward")),
        "da3_peak_gib_per_gpu_device_wide": _dig(r, ("geometry", cfg, "da3_peak_gib_per_gpu")),
        "tsdf_s": _dig(r, ("geometry", cfg, "tsdf_s_all_shots"))} for s, r in sites.items()}} for cfg in GEO_ORDER}
    people = {cfg: {"meaning": LABELS.get(cfg, cfg), **{s: {
        "keyframes": _dig(r, ("people", cfg, "keyframes")),
        "path_difference_median_m": _dig(r, ("people", cfg, "harness_people", "path_difference_floor_m", "median")),
        "path_difference_p90_m": _dig(r, ("people", cfg, "harness_people", "path_difference_floor_m", "p90")),
        "points_matched": _dig(r, ("people", cfg, "harness_people", "matched")), "points_extra": _dig(r, ("people", cfg, "harness_people", "extra")),
        "id_switches": _dig(r, ("people", cfg, "identity", "id_switches_inside_our_tracks")),
        "extra_tracks_per_delivered_person": _dig(r, ("people", cfg, "identity", "extra_tracks_per_delivered_person")),
        "delivered_people_matched": _dig(r, ("people", cfg, "identity", "delivered_people_matched")),
        "R3_speed_agree_vs_5fps_before_gate": _dig(r, ("people", cfg, "rules_vs_5fps", "R3_speed_before_gate", "agree_share")),
        "pass_fail_flips_vs_5fps": (sum(v["pass_fail_flips"] for v in r["people"][cfg]["rules_vs_5fps"].values())
                                    if isinstance(_dig(r, ("people", cfg, "rules_vs_5fps")), dict) else None),
        "R3_speed_agree_vs_E3_ref10": _dig(r, ("people", cfg, "harness_people", "rules_vs_e3_ref10", "R3_speed_before_gate", "agree_share")),
        "peopleloop_s": _dig(r, ("people", cfg, "s")), "sam3_person_s_2gpu_MxN": _dig(r, ("people", cfg, "sam3_person_s_2gpu_MxN"))}
        for s, r in sites.items()}} for cfg in PEOPLE_ORDER}
    sam3 = {s: {"gpu_s_per_frame": r["sam3"]["gpu_s_per_frame"], "words": r["words"], "phases": r["sam3"]["phases"],
                "peak_gib_per_gpu_device_wide": r["sam3"]["peak_gib_per_gpu_during_sam3"], "masks": r["sam3"]["masks"]} for s, r in sites.items()}
    return {"objects": objects, "geometry": geometry, "people": people, "sam3": sam3}


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    elif sys.argv[1:2] == ["--evaluate"]:
        evaluate(sys.argv[2])
    else:
        print(__doc__)
