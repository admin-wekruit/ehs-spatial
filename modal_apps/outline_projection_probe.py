"""E6b (fast-path follow-up): full masks only on keyframes, in-between outlines projected from 3D (ME340 walk shot).

Keyframes (every 2 / 5 / 10 frames of today's 10 fps mask grid = 5 / 2 / 1 fps) keep today's SAM 2 masks ('segmented').
Their masks are lifted to 3D objects with the round-1 GPU lift (m3_exp_geometry.Context.lift: 5 cm voxels, overlap
merge across frames). Every in-between frame gets 'projected' outlines by rendering keyframe points into it with the
camera poses. Variants:
  prev_r1  round 1 exactly: the previous keyframe's label map only, people kept (a reproduction check)
  prev     the previous keyframe only, people's masks left out
  pair     the two keyframes around the frame, the nearer first; the other fills what the nearer cannot see
  fused    every keyframe's points at once (half resolution, one z-buffer): the fused 3D objects
  pair_vis / fused_vis  the same points, but visibility only: among the points within 5% of a pixel's nearest surface,
           the keyframe closest in time gives the label (no z-fighting between keyframes' slightly different depths)
  *_snap   + depth-edge snapping: a projected label survives only where its depth agrees with the frame's own depth
           (|dz| <= tau z; pair at tau 0.04 / 0.08 / 0.16 = pair_snap04 / pair_snap / pair_snap16, fused at 0.08); the
           rest refills from 3x3 neighbours on the same depth surface
  dec_box  each projected keyframe mask's box (pair projection) as a prompt to the SAM 2 mask decoder alone, on image
           embeddings computed once per frame and cached on the GPU
  dec_pick box + its most interior pixel, 3 candidate masks, keep the one closest to the projection
  dec_pt   that pixel alone (AMG's own prompt kind), 3 candidates, keep the one closest to the projection
Geometry chains: ref = DROID poses + DA3 posed depth (round 1's cached inputs); da3 = DA3-GIANT any-view on all 224
grid frames (the fast path's own geometry; its output is cached in runs/m3-fu-e6b-outlines-da3 after the first call).
Scores, on the in-between frames, DROID raster, against today's per-frame masks (me340-masks-194):
  M1  best IoU of each of today's masks (>= 300 px) with one projected region (round 1's metric); regions are keyframe
      masks ('kf', round 1's granularity) or lifted 3D objects ('lift'); 'static' leaves out masks >= 50% on people
      (me340-dynamic-masks-188)
  M2  object outlines: each object-map entity (me340-entity-names-200) seen on an in-between frame vs the union of the
      projected 3D objects that hold its keyframe masks ('lift'), or of its own keyframe masks ('oracle' grouping)
Snapping uses the frame's depth with its edges (the object map's edge rule removes them). Speed: the same A100 times
today's AMG on a few frames; speedup = 224 x AMG / (keyframe AMG + lift + render [+ snap]
[+ encoder + decoder]). Depth and poses are not charged: the geometry layer computes them anyway (see the README).

  modal run modal_apps/outline_projection_probe.py --out RUNS/m3-fu-e6b-outlines-NNN [--chains ref,da3] [--smoke]
  python modal_apps/outline_projection_probe.py --self-check | --verdicts RUN_DIR
"""
import gzip
import io
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import m3_exp_geometry as geo  # noqa: E402  DA3 image + pins, align_sim3, edge_filter, the lift's constants
import segment_fast_probe as seg  # noqa: E402  WALK grid, DROID raster, SAM 2 image, agreement, decode

RUNS = seg.PHASE2 / "runs"
INPUTS = RUNS / "m3-exp-e2-e6-segment-inputs"  # round 1: today's masks as label maps, DROID poses + DA3 posed depth
DA3_CACHE = RUNS / "m3-fu-e6b-outlines-da3"
N, H, W = len(seg.WALK), 480, 640  # 224 grid frames on the DROID raster
SLOT = 1 << 12  # gid = keyframe slot * SLOT + today's label (labels <= 82); gids stay < 2**20
VIS_TOL = .05  # pair_vis / fused_vis: points within 5% of the nearest surface count as visible
TAUS, SNAP_ITERS, HOLE_ITERS, INTERIOR_ITERS = {"pair_snap04": .04, "pair_snap": .08, "pair_snap16": .16, "fused_snap": .08}, 12, 3, 8
STEPS = (2, 4, 5, 6, 8, 10)  # grid frames between keyframes: 5, 2.5, 2, 1.7, 1.25, 1 fps
DA3_RES = 504  # E1's pick: DA3-GIANT camera head at 504
VARIANTS = ("prev_r1", "prev", "pair", "fused", "pair_vis", "fused_vis", *TAUS, "dec_box", "dec_pick", "dec_pt")
SAVED = ("pair", "pair_snap16", "fused_snap", "dec_pt")  # label maps, outlines and the contact sheet, at --save-step
USD_PER_S = seg.USD_PER_S
RESERVED = (8, 32)  # cpu cores, GiB per container below

app = modal.App("panoptes-e6b-outline-probe")
PROBE_IMAGE = seg.S2_IMAGE.add_local_python_source("segment_fast_probe", "m3_exp_geometry")
DA3_IMAGE = geo.image.add_local_python_source("segment_fast_probe", "m3_exp_geometry", "sam2_everything", "sam3_app", "moge3_app")
GPU = dict(gpu="A100-80GB", retries=0, scaledown_window=2, cpu=RESERVED[0], memory=RESERVED[1] * 1024)


# ---------- pure helpers (numpy; also run by the self-check) ----------

SX, SY = 1.1 / 1.5, 512 / 480 / 1.5  # source (1280x720) px -> DROID raster px: 640x480 raster = source x 160..1120 at 2/3, then to_droid


def source_to_droid(x, y):
    return (x + .5 - 160) * SX - 32.5, (y + .5) * SY - 16.5


def droid_to_source(u, v):
    return (u + 32.5) / SX + 160 - .5, (v + 16.5) / SY - .5


def droid_k_from_da3(k, w, h):
    """DA3 intrinsics (3x3) at its w x h processing size of the 1280x720 frame -> [fx, fy, cx, cy] on the DROID raster."""
    cx, cy = source_to_droid((k[0, 2] + .5) * 1280 / w - .5, (k[1, 2] + .5) * 720 / h - .5)
    return [k[0, 0] * 1280 / w * SX, k[1, 1] * 720 / h * SY, cx, cy]


def components(n, a, b):
    """Connected components of n nodes with edges a-b (union-find): component index per node."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in zip(a, b):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    return np.unique([find(i) for i in range(n)], return_inverse=True)[1]


def frame_scores(uniq, table, moving, obs, members):
    """One frame. uniq: predicted ids (0 = none); table[t, c]: pixels with today's label t and prediction uniq[c].

    Returns M1 rows (best IoU, area, moving) for today's masks >= MIN_AREA and M2 rows (IoU, area) per entity
    observation (entity, today's label); an entity with no projected id scores None."""
    area_t, area_p = table.sum(1), table.sum(0)
    iou = table / np.maximum(area_t[:, None] + area_p[None] - table, 1)
    iou[:, uniq == 0] = 0
    m1 = [(float(iou[t].max()), int(area_t[t]), bool(moving[t])) for t in range(1, len(area_t)) if area_t[t] >= seg.MIN_AREA]
    m2 = []
    for entity, label in obs:
        if area_t[label] < seg.MIN_AREA:
            continue
        ids = members.get(entity)
        if not ids:
            m2.append((None, int(area_t[label])))
            continue
        cols = np.isin(uniq, list(ids)) & (uniq != 0)
        inter = table[label, cols].sum()
        m2.append((float(inter / max(area_t[label] + area_p[cols].sum() - inter, 1)), int(area_t[label])))
    return m1, m2


def summary(rows):
    """rows: [(IoU, area, distance in source frames since the previous keyframe)]."""
    if not rows:
        return {"n": 0}
    iou, area, dist = (np.array([r[i] for r in rows], float) for i in range(3))
    return {"n": len(iou), "mean": round(float(iou.mean()), 4), "area_weighted": round(float((iou * area).sum() / area.sum()), 4),
            "share_ge_0.8": round(float((iou >= .8).mean()), 4), "share_ge_0.5": round(float((iou >= .5).mean()), 4),
            "by_distance": {int(d): round(float(iou[dist == d].mean()), 4) for d in np.unique(dist)}}


def self_check():
    k = np.array([[300., 0, 250], [0, 290, 140], [0, 0, 1]])  # a DA3-like K at 504x280
    point = np.array([.3, -.2, 2.])
    x, y = (k @ point)[:2] / point[2]
    u, v = source_to_droid((x + .5) * 1280 / 504 - .5, (y + .5) * 720 / 280 - .5)
    fx, fy, cx, cy = droid_k_from_da3(k, 504, 280)
    assert abs(point[0] / point[2] * fx + cx - u) < 1e-6 and abs(point[1] / point[2] * fy + cy - v) < 1e-6, "DA3 K -> DROID K"
    assert np.allclose(source_to_droid(*droid_to_source(123.4, 56.7)), (123.4, 56.7)), "raster maps invert"
    raw = [377.51247406005854, 377.51247406005854, 319.5, 239.5]  # clip K on the 640x480 raster; round 1's droid_k
    ref = seg.droid_k(raw)
    fx_s, cx_s, cy_s = raw[0] * 1.5, (raw[2] + .5) * 1.5 + 160 - .5, (raw[3] + .5) * 1.5 - .5
    cx, cy = source_to_droid(cx_s, cy_s)
    assert abs(fx_s * SX - ref[0]) < 1e-6 and abs(cx - ref[2]) < .06 and abs(cy - ref[3]) < .06, "matches round 1's K (to the half-pixel convention)"
    assert components(6, [0, 2, 1], [1, 3, 3]).tolist() == [0, 0, 0, 0, 1, 2], "union-find"
    today = np.zeros((40, 40), np.int64)
    today[:20, :20], today[20:, 20:] = 1, 2
    pred = np.zeros_like(today)
    pred[:20, :10], pred[:20, 10:20], pred[30:, 20:] = 7, 8, 9
    uniq, inv = np.unique(pred, return_inverse=True)
    table = np.bincount(today.ravel() * len(uniq) + inv.ravel(), minlength=3 * len(uniq)).reshape(3, -1)
    m1, m2 = frame_scores(uniq, table, [False] * 3, [("a", 1), ("b", 2), ("c", 2)], {"a": {7, 8}, "b": {9}, "c": set()})
    best, _ = seg.agreement(today, pred, min_area=1)
    assert [r[0] for r in m1] == best.tolist() == [.5, .5], "M1 = round 1's agreement"
    assert [r[0] for r in m2] == [1., .5, None], "M2: union of the entity's objects; no object -> None"
    s = summary([(1., 100, 3), (.5, 300, 6)])
    assert s["mean"] == .75 and s["area_weighted"] == .625 and s["by_distance"] == {3: 1., 6: .5}
    print("outline_projection_probe self-check passed: raster maps, K conversion, union-find, M1/M2 scoring; no GPU invoked")


# ---------- GPU helpers (torch, inside the containers) ----------

EMPTY = 2 ** 63 - 1


def backproject(depth, k, c2w, step=1):
    """(H,W) depth (metres, 0 = none) -> world points of the pixels with depth, and their flat index on the full raster."""
    import torch
    v, u = torch.meshgrid(torch.arange(0, H, step, device="cuda", dtype=torch.float32),
                          torch.arange(0, W, step, device="cuda", dtype=torch.float32), indexing="ij")
    z = depth[::step, ::step]
    ok = z > 0
    fx, fy, cx, cy = k
    cam = torch.stack([(u[ok] - cx) / fx * z[ok], (v[ok] - cy) / fy * z[ok], z[ok]], -1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3], v[ok].long() * W + u[ok].long()


def splat(world, ids, k, w2c, wide3=False):
    """Z-buffer splat of world points into one view: per pixel an int64 key (z in 0.1 mm << 20 | id), EMPTY where none.

    Round 1's footprint: the rounded pixel, then the 2x2 floor block where that left gaps (the target is closer, so
    magnified). wide3: a 3x3 block around the rounded pixel, for half-resolution points."""
    import torch
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    front = cam[:, 2] > 1e-3
    cam, ids = cam[front], ids[front]
    fx, fy, cx, cy = k
    x, y = cam[:, 0] / cam[:, 2] * fx + cx, cam[:, 1] / cam[:, 2] * fy + cy
    key = (cam[:, 2] * 1e4).long().clamp(0, 2 ** 40) * 2 ** 20 + ids

    def scatter(out, xi, yi):
        inside = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        return out.scatter_reduce_(0, (yi * W + xi)[inside], key[inside], reduce="amin")
    rx, ry = torch.round(x).long(), torch.round(y).long()
    near = scatter(torch.full((H * W,), EMPTY, dtype=torch.int64, device="cuda"), rx, ry)
    wide = torch.full_like(near, EMPTY)
    base, offsets = ((rx, ry), (-1, 0, 1)) if wide3 else ((torch.floor(x).long(), torch.floor(y).long()), (0, 1))
    for dx in offsets:
        for dy in offsets:
            scatter(wide, base[0] + dx, base[1] + dy)
    return torch.where(near != EMPTY, near, wide)


def splat_vis(world, ids, frame, n, k, w2c, wide3):
    """Visibility from every point, label from the nearest keyframe in time: among the points that land on a pixel within
    VIS_TOL (relative) of its nearest surface, the one whose keyframe is closest to frame n gives the label. Same key
    layout as splat(). frame: the keyframe (grid index) of each point."""
    import torch
    cam = world @ w2c[:3, :3].T + w2c[:3, 3]
    front = cam[:, 2] > 1e-3
    cam, ids, dt = cam[front], ids[front], (frame[front] - n).abs()
    fx, fy, cx, cy = k
    x, y = cam[:, 0] / cam[:, 2] * fx + cx, cam[:, 1] / cam[:, 2] * fy + cy
    rx, ry = torch.round(x).long(), torch.round(y).long()
    if wide3:
        cells = [(rx + dx, ry + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    else:
        fx0, fy0 = torch.floor(x).long(), torch.floor(y).long()
        cells = [(rx, ry)] + [(fx0 + dx, fy0 + dy) for dx in (0, 1) for dy in (0, 1)]
    cells = [(xi, yi, (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)) for xi, yi in cells]
    zmin = torch.full((H * W,), torch.inf, device="cuda")
    for xi, yi, inside in cells:
        zmin.scatter_reduce_(0, (yi * W + xi)[inside], cam[:, 2][inside], reduce="amin")
    best = torch.full((H * W,), EMPTY, dtype=torch.int64, device="cuda")
    key = dt * 2 ** 20 + ids
    for xi, yi, inside in cells:
        index = (yi * W + xi).clamp(0, H * W - 1)
        ok = inside & (cam[:, 2] <= zmin[index] * (1 + VIS_TOL))
        best.scatter_reduce_(0, index[ok], key[ok], reduce="amin")
    return torch.where(best == EMPTY, EMPTY, (zmin * 1e4).nan_to_num(0, 0, 0).long().clamp(0, 2 ** 40) * 2 ** 20 + best % 2 ** 20)


def finish(key):
    """Packed keys -> (gid map, -1 = nothing; rendered z, 0 = unknown). Holes up to 3 px take a neighbour's label (round 1)."""
    import torch
    lab = torch.where(key == EMPTY, -1, key % 2 ** 20).float().view(1, 1, H, W)
    z = torch.where(key == EMPTY, 0, key // 2 ** 20).float().view(H, W) / 1e4
    for _ in range(HOLE_ITERS):
        lab = torch.where(lab < 0, torch.nn.functional.max_pool2d(lab, 3, 1, 1), lab)
    return lab[0, 0].long(), z


def snap(lab, z_render, z_frame, tau):
    """Depth-edge snapping: where the frame has depth, a projected label stays only if its rendered depth agrees within
    tau (relative); everything else (disagreement, 3 px hole fill, nothing projected) refills from 3x3 neighbours whose
    frame depth is within tau of the pixel's own, nearest depth first, up to SNAP_ITERS px. So outlines end on depth
    edges. A pixel no such neighbour reaches (in-between depth right on an edge) keeps its projected label.
    Returns the labels and the pixels whose projected label was rejected."""
    import torch
    unfold = torch.nn.functional.unfold
    bad = (z_frame > 0) & ((lab < 0) | (z_render <= 0) | ((z_render - z_frame).abs() > tau * z_frame))
    projected, lab = lab, torch.where(bad, -2, lab)  # -2 unknown; -1 known empty
    zn = unfold(z_frame[None, None], 3, padding=1).view(9, H, W)
    cost = (zn - z_frame).abs() / z_frame.clamp(min=1e-6)
    cost = torch.where((zn > 0) & (cost <= tau), cost, torch.inf)
    for _ in range(SNAP_ITERS):
        unknown = lab == -2
        if not bool(unknown.any()):
            break
        ln = unfold(lab[None, None].float(), 3, padding=1).view(9, H, W)
        c = torch.where(ln > -2, cost, torch.inf)
        best, which = c.min(0)
        lab = torch.where(unknown & torch.isfinite(best), ln.gather(0, which[None])[0].long(), lab)
    return torch.where(lab == -2, projected, lab), bad & (z_render > 0)


def lift(labels, depth, k, c2w, dyn):
    """The round-1 GPU lift (m3_exp_geometry.Context.lift) on keyframe label maps, half resolution: masks mostly on people
    are dropped, the rest -> 5 cm voxels -> pairwise voxel overlap -> connected components = 3D objects.

    labels/depth/dyn: (n,H,W) on the GPU (depth metres); k (n,4) torch; c2w (n,4,4) torch (metres).
    Returns lift id per gid (numpy, 0 = a mask on people: not rendered; masks with too little depth to lift keep an id
    of their own), and stats."""
    import torch
    n = len(labels)
    lab, z, dy = labels[:, ::2, ::2].long(), depth[:, ::2, ::2], dyn[:, ::2, ::2]
    gid = torch.arange(n, device="cuda")[:, None, None] * SLOT + lab
    has = lab > 0
    area = torch.bincount(gid[has], minlength=n * SLOT)
    moving = torch.bincount(gid[has & dy], minlength=n * SLOT) >= geo.MOVING * area.clamp(min=1)
    points = has & (z > 0) & ~dy
    keep = (torch.bincount(gid[points], minlength=n * SLOT) >= geo.MIN_PIXELS) & ~moving
    f, vy, vx = torch.nonzero(points & keep[gid], as_tuple=True)
    g, zz = gid[f, vy, vx], z[f, vy, vx]
    cam = torch.stack([(vx * 2. - k[f, 2]) / k[f, 0] * zz, (vy * 2. - k[f, 3]) / k[f, 1] * zz, zz], 1)
    world = (c2w[f, :3, :3] @ cam[:, :, None])[:, :, 0] + c2w[f, :3, 3]
    ijk = torch.floor(world / geo.LIFT_VOXEL).long()
    B, OFF = 1 << 21, 1 << 20
    vox, vid = torch.unique(((ijk[:, 0] + OFF) * B + (ijk[:, 1] + OFF)) * B + (ijk[:, 2] + OFF), return_inverse=True)
    masks, mid = torch.unique(g, return_inverse=True)
    pair = torch.unique(mid * len(vox) + vid)
    pm, pv = pair // len(vox), pair % len(vox)
    size = torch.bincount(pm, minlength=len(masks)).float()
    A = torch.sparse_coo_tensor(torch.stack([pm, pv]), torch.ones_like(pm, dtype=torch.float32), (len(masks), len(vox))).coalesce()
    inter = torch.sparse.mm(A, A.t()).coalesce()
    (a, b), c = inter.indices(), inter.values()
    frame = masks // SLOT
    small, large = torch.minimum(size[a], size[b]), torch.maximum(size[a], size[b])
    edge = (a < b) & (frame[a] != frame[b]) & (c >= geo.MATCH_MIN * small) & (c >= geo.MATCH_MAX * large)
    comp = components(len(masks), a[edge].cpu().numpy(), b[edge].cpu().numpy())
    table = np.zeros(n * SLOT, np.int64)
    table[masks.cpu().numpy()] = comp + 1
    singles = torch.nonzero((area > 0) & ~moving & ~keep).squeeze(1).cpu().numpy()
    table[singles] = comp.max(initial=-1) + 2 + np.arange(len(singles))
    frames_of = np.bincount(np.unique(comp * n + masks.cpu().numpy() // SLOT) // n, minlength=comp.max(initial=-1) + 1)
    return table, {"masks_in": int((area > 0).sum()), "masks_on_people": int(moving.sum()), "masks_lifted": len(masks),
                   "masks_unlifted_kept": len(singles), "voxels": len(vox), "edges": int(edge.sum()),
                   "objects": int(comp.max(initial=-1) + 1), "objects_seen_on_3plus_keyframes": int((frames_of >= geo.CONFIRMED).sum())}


def prompts(ids):
    """Projected object map (H,W) -> for each object >= MIN_AREA px: its id, box [x0, y0, x1, y1] and most interior pixel."""
    import torch
    pool = torch.nn.functional.max_pool2d
    uniq, inv = torch.unique(ids, return_inverse=True)
    inv = inv.view(-1)
    area = torch.bincount(inv, minlength=len(uniq))
    xs = torch.arange(W, device="cuda").repeat(H)
    ys = torch.arange(H, device="cuda").repeat_interleave(W)

    def reduce(values, how, fill):
        return torch.full((len(uniq),), fill, device="cuda", dtype=torch.long).scatter_reduce(0, inv, values, how)
    box = torch.stack([reduce(xs, "amin", W), reduce(ys, "amin", H), reduce(xs, "amax", -1) + 1, reduce(ys, "amax", -1) + 1], 1)
    f = ids.float()[None, None]
    inside = (pool(f, 3, 1, 1) == f) & (-pool(-f, 3, 1, 1) == f)
    depth = inside.float()
    for _ in range(INTERIOR_ITERS - 1):
        inside = inside & (-pool(-inside.float(), 3, 1, 1) > 0)
        depth = depth + inside.float()
    best = reduce(depth.view(-1).long() * (H * W) + torch.arange(H * W, device="cuda"), "amax", 0) % (H * W)
    keep = (uniq != 0) & (area >= seg.MIN_AREA)
    return uniq[keep], box[keep].float(), torch.stack([best % W, best // W], 1)[keep].float()


# ---------- Modal functions ----------

@app.function(image=DA3_IMAGE, volumes={"/cache": geo.volume}, timeout=900, max_containers=1, **GPU)
def da3_views(mp4_full):
    """DA3-GIANT any-view (camera head, 504) on all 224 grid frames of the walk shot; 112 (every other) timed as well."""
    import torch
    from depth_anything_3.api import DepthAnything3
    enter = time.time()
    t = time.perf_counter()
    frames = seg.decode_walk(mp4_full, set(seg.WALK))
    rgb = [np.ascontiguousarray(frames[i][..., ::-1]) for i in seg.WALK]
    report = {"gpu": torch.cuda.get_device_name(), "decode_s": round(time.perf_counter() - t, 2)}
    t = time.perf_counter()
    name = geo.MODELS["giant"]
    model = DepthAnything3.from_pretrained(name, revision=geo.REVISIONS[name]).to("cuda").eval()
    torch.cuda.synchronize()
    report["model_load_s"] = round(time.perf_counter() - t, 2)
    forward = {}
    inner = model._run_model_forward

    def timed(*a, **kw):  # m3_exp_geometry's wrapper: the pure forward, GPU-synchronised
        torch.cuda.synchronize()
        s = time.perf_counter()
        r = inner(*a, **kw)
        torch.cuda.synchronize()
        forward["s"] = time.perf_counter() - s
        return r
    model._run_model_forward = timed

    def infer(views):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        s = time.perf_counter()
        with torch.inference_mode():
            out = model.inference(views, process_res=DA3_RES, use_ray_pose=False)
        torch.cuda.synchronize()
        return out, {"views": len(views), "call_s": round(time.perf_counter() - s, 3), "forward_s": round(forward.get("s", -1), 3),
                     "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 1)}
    infer(rgb[:8])  # warm-up
    _, report["views_112"] = infer(rgb[::2])
    out, report["views_224"] = infer(rgb)
    report["function_wall_s"] = round(time.time() - enter, 2)
    report["enter_at"] = enter
    return json.dumps(report), seg.npz_bytes(depth=np.asarray(out.depth, np.float16), extrinsics=np.asarray(out.extrinsics, np.float64),
                                             intrinsics=np.asarray(out.intrinsics, np.float64), frames=np.array(seg.WALK))


@app.function(image=PROBE_IMAGE, volumes={"/cache": seg.s2.volume}, timeout=2400, max_containers=2, **GPU)
def probe(chain, mp4_raw, labels_npz, dyn_npz, geometry_npz, droid_npz, entity_obs, steps, save_step, smoke=False, snap_npz=None):
    import cv2
    import torch
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    enter = time.time()
    sync = torch.cuda.synchronize
    report = {"chain": chain, "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "snap_tau": TAUS}
    n_frames = 36 if smoke else N

    # today's masks and people on the DROID raster; this chain's depth (metres), poses and intrinsics
    t = time.perf_counter()
    labels_raw = np.load(io.BytesIO(labels_npz))["labels"][:n_frames]
    today = torch.from_numpy(np.stack([seg.to_droid(x) for x in labels_raw]).astype(np.int64)).cuda()
    dyn_raw = np.unpackbits(np.load(io.BytesIO(dyn_npz))["packed"], axis=-1)[:n_frames].astype(np.uint8)
    dyn = torch.from_numpy(np.stack([seg.to_droid(x) for x in dyn_raw]) > 0).cuda()
    droid = np.load(io.BytesIO(droid_npz))
    mpn = float(droid["metres_per_native"])
    g = np.load(io.BytesIO(geometry_npz))
    if chain == "ref":
        depth = torch.from_numpy(g["depth"][:n_frames].astype(np.float32)).cuda() * mpn
        c2w = g["c2w"][:n_frames].copy()
        c2w[:, :3, 3] *= mpn
        ks = np.tile(np.asarray(g["k"], np.float64), (n_frames, 1))
        z_snap = torch.from_numpy(np.load(io.BytesIO(snap_npz))["depth"][:n_frames].astype(np.float32)).cuda() * mpn  # edges kept
    else:  # DA3 any-view: its own poses and depth; metres from the trajectory Sim3 to DROID x the reference scale (borrowed)
        w2c = np.tile(np.eye(4), (N, 1, 1))
        w2c[:, :3] = g["extrinsics"]
        own = np.linalg.inv(w2c)
        s, _, _ = geo.align_sim3(own, droid["c2w"])
        report["da3_to_native_scale"], report["metres_per_da3_unit"] = float(s), float(s * mpn)
        h, w = g["depth"].shape[1:]
        u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
        x, y = droid_to_source(u, v)  # nearest DA3 pixel of each DROID raster pixel
        xi = np.clip(np.floor((x + .5) * w / 1280), 0, w - 1).astype(np.int64)
        yi = np.clip(np.floor((y + .5) * h / 720), 0, h - 1).astype(np.int64)
        d = torch.from_numpy(g["depth"][:n_frames].astype(np.float32)).cuda()
        depth = d[:, torch.from_numpy(yi).cuda(), torch.from_numpy(xi).cuda()].contiguous()
        z_snap = depth * (s * mpn)  # snapping needs the depth at the edges too
        geo.edge_filter(depth)  # flying pixels at depth jumps -> 0, as the fast path's TSDF does
        depth = depth * (s * mpn)
        c2w = own[:n_frames].copy()
        c2w[:, :3, 3] *= s * mpn
        ks = np.array([droid_k_from_da3(k, w, h) for k in g["intrinsics"][:n_frames]])
    c2w_t = torch.from_numpy(c2w).float().cuda()
    w2c_t = torch.linalg.inv(torch.from_numpy(c2w)).float().cuda()
    k_t = torch.from_numpy(ks).float().cuda()
    kk = [tuple(float(a) for a in row) for row in ks]
    frames = seg.decode_walk(mp4_raw, set(seg.WALK[:n_frames]))
    raw_rgb = [frames[i][..., ::-1].copy() for i in seg.WALK[:n_frames]]
    droid_rgb = [cv2.resize(x, (704, 512), interpolation=cv2.INTER_LINEAR)[16:-16, 32:-32].copy() for x in raw_rgb]
    report["load_s"] = round(time.perf_counter() - t, 2)

    # today's cost on this GPU: AMG with today's settings, raw frames
    t = time.perf_counter()
    model = seg.s2.pinned_sam2()
    report["sam2_load_s"] = round(time.perf_counter() - t, 2)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)  # noqa: E731
    generator = SAM2AutomaticMaskGenerator(model, **seg.s2.SETTINGS)
    amg = []
    for i in (0, 1) + tuple(range(10, n_frames, max(1, n_frames // 6)))[:6]:
        sync()
        t = time.perf_counter()
        with torch.inference_mode(), amp():
            generator.generate(raw_rgb[i])
        sync()
        amg.append(time.perf_counter() - t)
    amg_s = float(np.mean(amg[2:]))
    report["amg_s_per_frame"], report["amg_timed_frames"] = round(amg_s, 4), len(amg) - 2
    del generator

    # SAM 2 image embeddings: computed once per frame, kept on the GPU for every prompt set
    predictor = SAM2ImagePredictor(model)
    with torch.inference_mode(), amp():
        predictor.set_image_batch(droid_rgb[:2])  # warm-up
    sync()
    t = time.perf_counter()
    embeds, highres = [], []
    for s0 in range(0, n_frames, 16):
        with torch.inference_mode(), amp():
            predictor.set_image_batch(droid_rgb[s0:s0 + 16])
        embeds.append(predictor._features["image_embed"])
        highres.append(predictor._features["high_res_feats"])
    sync()
    enc_s = (time.perf_counter() - t) / n_frames
    predictor._features = {"image_embed": torch.cat(embeds), "high_res_feats": [torch.cat([h[i] for h in highres]) for i in range(len(highres[0]))]}
    predictor._orig_hw, predictor._is_image_set, predictor._is_batch = [(H, W)] * n_frames, True, True
    del embeds, highres
    per_frame_bytes = sum(x.numel() * x.element_size() for x in [predictor._features["image_embed"], *predictor._features["high_res_feats"]]) / n_frames
    report["encoder"] = {"s_per_frame": round(enc_s, 4), "batch": 16, "cached_mb_per_frame": round(per_frame_bytes / 2 ** 20, 2),
                         "dtype": str(predictor._features["image_embed"].dtype)}

    # scoring inputs: people share of each of today's masks; entity observations per frame
    moving = []
    for n in range(n_frames):
        area = torch.bincount(today[n].view(-1), minlength=128)
        on = torch.bincount(today[n][dyn[n]], minlength=128)
        moving.append((on >= geo.MOVING * area.clamp(min=1)).cpu().numpy())
    obs_of = {}
    for entity, n, label in entity_obs:
        if n < n_frames:
            obs_of.setdefault(n, []).append((entity, label))

    def score(maps, members, rows1, rows2, dist):
        for n, ids in maps.items():
            uniq, inv = torch.unique(ids, return_inverse=True)
            table = torch.bincount(today[n].view(-1) * len(uniq) + inv.view(-1), minlength=128 * len(uniq)).view(128, -1).cpu().numpy()
            m1, m2 = frame_scores(uniq.cpu().numpy(), table, moving[n], obs_of.get(n, []) if members is not None else [], members or {})
            if rows1 is not None:
                rows1.extend((iou, area, dist[n], mov) for iou, area, mov in m1)
            rows2.extend((iou, area, dist[n]) for iou, area in m2)

    maps_out, outlines_out, sheet = {}, {}, None
    report["steps"] = {}
    for step in steps:
        keys = list(range(0, n_frames, step))
        between = [n for n in range(n_frames) if n % step]
        dist = {n: 3 * (n % step) for n in between}
        row = {"keyframes": len(keys), "in_between": len(between), "seconds": {}}
        sync()
        t = time.perf_counter()
        table_np, row["lift"] = lift(today[keys], depth[keys], k_t[keys], c2w_t[keys], dyn[keys])
        sync()
        row["seconds"]["lift"] = time.perf_counter() - t
        table = torch.from_numpy(table_np).cuda()

        t = time.perf_counter()  # keyframe points: round 1 fills depth-rule gaps with the farther neighbour's depth
        full, full_r1, half = {}, {}, []
        for slot, key in enumerate(keys):
            z = depth[key]
            for _ in range(2):
                z = torch.where(z > 0, z, torch.nn.functional.max_pool2d(z[None, None], 5, 1, 2)[0, 0])
            world, pix = backproject(z, kk[key], c2w_t[key])
            gid = slot * SLOT + today[key].view(-1)[pix]
            full_r1[key] = (world, gid)
            still = (table[gid] > 0) | (gid % SLOT == 0)  # people's masks are not rendered
            full[key] = (world[still], gid[still])
            world, pix = backproject(z, kk[key], c2w_t[key], step=2)
            gid = slot * SLOT + today[key].view(-1)[pix]
            still = (table[gid] > 0) | (gid % SLOT == 0)
            half.append((world[still], gid[still], torch.full_like(gid[still], key)))
        fused_world, fused_gid, fused_frame = (torch.cat([h[i] for h in half]) for i in range(3))
        del half
        sync()
        row["seconds"]["keyframe_points"] = time.perf_counter() - t
        row["fused_points"] = int(len(fused_gid))

        def render(name, fn):
            sync()
            t = time.perf_counter()
            out = {n: finish(fn(n)) for n in between}
            sync()
            row["seconds"][name] = time.perf_counter() - t
            return out

        def order(n):
            before, after = n - n % step, n - n % step + step
            return [before] if after >= n_frames else sorted([before, after], key=lambda x: (abs(x - n), x))

        def pair(n):
            first, *rest = order(n)
            key = splat(*full[first], kk[n], w2c_t[n])
            for other in rest:
                key = torch.where(key != EMPTY, key, splat(*full[other], kk[n], w2c_t[n]))
            return key
        rendered = {"prev_r1": render("prev_r1", lambda n: splat(*full_r1[n - n % step], kk[n], w2c_t[n])),
                    "prev": render("prev", lambda n: splat(*full[n - n % step], kk[n], w2c_t[n])),
                    "pair": render("pair", pair),
                    "fused": render("fused", lambda n: splat(fused_world, fused_gid, kk[n], w2c_t[n], wide3=True))}

        def pair_vis(n):
            keys_n = order(n)
            world = torch.cat([full[x][0] for x in keys_n])
            gid = torch.cat([full[x][1] for x in keys_n])
            frame = torch.cat([torch.full_like(full[x][1], x) for x in keys_n])
            return splat_vis(world, gid, frame, n, kk[n], w2c_t[n], wide3=False)
        rendered["pair_vis"] = render("pair_vis", pair_vis)
        rendered["fused_vis"] = render("fused_vis", lambda n: splat_vis(fused_world, fused_gid, fused_frame, n, kk[n], w2c_t[n], wide3=True))
        row["snap_rejected_share_of_projected_px"] = {}
        for name, tau in TAUS.items():
            sync()
            t = time.perf_counter()
            rendered[name] = {n: snap(lab, z, z_snap[n], tau) for n, (lab, z) in rendered[name.split("_")[0]].items()}
            sync()
            row["seconds"][f"{name}_only"] = time.perf_counter() - t
            projected = sum(int((z > 0).sum()) for _, z in rendered[name.split("_")[0]].values())
            row["snap_rejected_share_of_projected_px"][name] = round(sum(int(bad.sum()) for _, bad in rendered[name].values()) / max(projected, 1), 4)
        # every variant as a gid map: which keyframe mask each pixel came from (0 = none, or a keyframe pixel with no mask)
        gmaps = {name: {n: torch.where((lab > 0) & (lab % SLOT != 0), lab, 0) for n, (lab, _) in maps.items()} for name, maps in rendered.items()}
        del rendered

        # SAM 2 decoder alone on the cached embeddings, one prompt per keyframe mask of the pair projection: its box (one
        # mask); box + most interior pixel, or that pixel alone, with 3 candidate masks, keeping the one closest to the projection
        for name, use_box, use_point in (("dec_box", True, False), ("dec_pick", True, True), ("dec_pt", False, True)):
            pick = use_point
            sync()
            t = time.perf_counter()
            out = {}
            for n in between:
                uid, boxes, points = prompts(gmaps["pair"][n])
                if not len(uid):
                    out[n] = torch.zeros(H, W, dtype=torch.long, device="cuda")
                    continue
                with torch.inference_mode(), amp():
                    mask_input, coords, point_labels, box = predictor._prep_prompts(
                        points[:, None] if use_point else None, torch.ones(len(uid), 1, device="cuda") if use_point else None,
                        boxes if use_box else None, None, True, img_idx=n)
                    masks, _, _ = predictor._predict(coords, point_labels, box, mask_input, multimask_output=pick, img_idx=n)
                if pick:
                    projected = (gmaps["pair"][n][None] == uid[:, None, None])[:, None]
                    fit = (masks & projected).sum((2, 3)) / (masks | projected).sum((2, 3)).clamp(min=1)
                    masks = masks[torch.arange(len(uid), device="cuda"), fit.argmax(1)]
                else:
                    masks = masks[:, 0]
                area = masks.sum((1, 2)).float()
                cost = torch.where(masks, area[:, None, None], torch.inf)  # a smaller mask wins where masks overlap (today's AMG rule)
                best, which = cost.min(0)
                out[n] = torch.where(torch.isfinite(best), uid[which], 0)
            sync()
            row["seconds"][f"{name}_prompts_and_decoder"] = time.perf_counter() - t
            gmaps[name] = out

        # scores: keyframe-mask ids (gid) and 3D-object ids (lift)
        members_lift, members_oracle = {}, {}
        slot_of = {key: slot for slot, key in enumerate(keys)}
        for entity, n, label in entity_obs:
            if n in slot_of:
                gid = slot_of[n] * SLOT + label
                members_oracle.setdefault(entity, set()).add(gid)
                if table_np[gid] > 0:
                    members_lift.setdefault(entity, set()).add(int(table_np[gid]))
        holders = {}
        for entity, ids in members_lift.items():
            for i in ids:
                holders.setdefault(i, set()).add(entity)
        entity_gids = set().union(*members_oracle.values()) if members_oracle else set()
        lifted = np.nonzero(table_np)[0]
        per_kf = np.unique(table_np[lifted] * len(keys) + lifted // SLOT, return_counts=True)[1]
        row["lift"].update(entities_on_keyframes=len(members_oracle), entities_lifted=len(members_lift),
                           masks_not_in_entities_sharing_an_entity_object=int(sum(1 for gid in lifted if gid not in entity_gids and int(table_np[gid]) in holders)),
                           object_keyframe_pairs_with_2plus_masks=int((per_kf > 1).sum()),
                           entities_sharing_an_object=sum(any(len(holders[i]) > 1 for i in ids) for ids in members_lift.values()),
                           objects_per_entity_mean=round(float(np.mean([len(ids) for ids in members_lift.values()] or [0])), 3))
        row["variants"] = {}
        for name in VARIANTS:
            result = {}
            for view, maps, members in (("kf", gmaps[name], members_oracle), ("lift", {n: table[m] for n, m in gmaps[name].items()}, members_lift)):
                m1, m2 = [], []
                score(maps, members, m1, m2, dist)
                result[f"m1_{view}_all"] = summary([r[:3] for r in m1])
                result[f"m1_{view}_static"] = summary([r[:3] for r in m1 if not r[3]])
                projectable = [r for r in m2 if r[0] is not None]
                result["m2_oracle" if view == "kf" else "m2_lift"] = {
                    **summary(projectable), "observations": len(m2), "projectable_share": round(len(projectable) / max(len(m2), 1), 4),
                    "mean_unprojectable_as_0": round(sum(r[0] for r in projectable) / max(len(m2), 1), 4)}
            row["variants"][name] = result
        row["seconds"] = {k: round(v, 4) for k, v in row["seconds"].items()}
        report["steps"][str(step)] = row

        if step == save_step:  # 'segmented' keyframes (today's masks as 3D objects) + each saved variant's in-between frames
            key_maps = {n: table[slot_of[n] * SLOT + today[n]] * (today[n] > 0) for n in keys}
            for name in SAVED:
                stack = np.stack([(key_maps[n] if n in key_maps else table[gmaps[name][n]]).cpu().numpy() for n in range(n_frames)])
                maps_out[name] = stack.astype(np.uint32)
                outlines_out[name] = outlines(stack, set(keys), name)
            sheet = contact_sheet(droid_rgb, today.cpu().numpy(), {v: maps_out[v] for v in ("pair", "pair_snap16", "dec_pt")}, between, step)
        del gmaps
        torch.cuda.empty_cache()
    report["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    report["function_wall_s"] = round(time.time() - enter, 2)
    report["enter_at"] = enter
    return json.dumps(report), seg.npz_bytes(**maps_out, frames=np.array(seg.WALK[:n_frames])), gzip.compress(json.dumps(outlines_out).encode()), sheet


def outlines(stack, keys, variant):
    """Per frame, per 3D object: outer polygons in source-video pixels (1280x720), labelled 'segmented' (keyframe: today's
    SAM 2 mask) or 'projected' (in-between: rendered from 3D). The decoder variant's in-between masks are SAM 2 output
    from projected prompts: 'segmented', prompt 'projected'."""
    import cv2
    frames = []
    for n, ids in enumerate(stack):
        source = "segmented" if n in keys else "projected"
        entry = {"frame": seg.WALK[n], "source": source, "objects": []}
        if n not in keys and variant.startswith("dec"):
            entry.update(source="segmented", prompt="projected")
        for i in np.unique(ids):
            if i == 0:
                continue
            contours, _ = cv2.findContours((ids == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygons = []
            for c in contours:
                if cv2.contourArea(c) < 20:
                    continue
                c = cv2.approxPolyDP(c, 1.0, True)[:, 0].astype(np.float64)
                x, y = droid_to_source(c[:, 0], c[:, 1])
                polygons.append(np.round(np.stack([x, y], 1), 1).tolist())
            if polygons:
                entry["objects"].append({"id": int(i), "polygons": polygons})
        frames.append(entry)
    return {"variant": variant, "raster": "source video pixels 1280x720", "frames": frames}


def contact_sheet(rgb, today, maps, between, step):
    """Four in-between frames farthest from their keyframes: today's outlines green, the variant's magenta, one row each."""
    import cv2
    gap = {n: min(n % step, step - n % step) for n in between}
    far = [n for n in between if gap[n] == max(gap.values())]
    picks = far[::max(1, len(far) // 4)][:4]

    def edges(ids):
        ids = ids.astype(np.int64)
        e = np.zeros(ids.shape, bool)
        e[:-1] |= ids[:-1] != ids[1:]
        e[:, :-1] |= ids[:, :-1] != ids[:, 1:]
        return cv2.dilate(e.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    rows = []
    for name, stack in maps.items():
        tiles = []
        for n in picks:
            tile = rgb[n][..., ::-1].copy()
            tile[edges(today[n])] = (0, 200, 0)
            tile[edges(stack[n])] = (255, 0, 255)
            cv2.putText(tile, f"{name} frame {seg.WALK[n]} (+{3 * (n % step)})", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, 1))
    sheet = cv2.resize(np.concatenate(rows, 0), None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".png", sheet)[1].tobytes()


# ---------- local side ----------

def ref_raw_depth():
    """DA3 posed depth of the walk grid (native units, DROID raster) before the object map's edge rule: snapping needs edges."""
    path = RUNS / "m3-fu-e6b-outlines-inputs/ref-raw-depth.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=False)
        scale = json.loads((seg.DEPTH / "fuse-metrics.json").read_text())["scale_native_per_mono_metre_median"]
        stack = []
        for i in seg.WALK:
            mono = np.load(seg.DEPTH / "mono" / f"{i:05d}.npz")
            stack.append(np.where(mono["mask"], mono["depth"] * scale, 0).astype(np.float16))
        path.write_bytes(seg.npz_bytes(depth=np.stack(stack)))
    return path.read_bytes()


def inputs():
    """People masks (packed), entity observations on the walk grid, DROID poses + the reference metric scale."""
    import cv2
    dyn = np.stack([cv2.imread(str(RUNS / f"me340-dynamic-masks-188/masks/{i:05d}-0.png"), cv2.IMREAD_GRAYSCALE) > 0 for i in seg.WALK])
    omap = json.loads((seg.NAMES / "object-map.json").read_text())
    grid = {f: n for n, f in enumerate(seg.WALK)}
    obs = []
    for e, entity in enumerate(omap["entities"]):
        for o in entity["observations"]:
            kind, f, i = o.split(":")
            if kind == "object" and int(f) in grid:
                obs.append((e, grid[int(f)], int(i) + 1))
    geometry = np.load(INPUTS / "geometry.npz")
    mpn = json.loads((RUNS / "da3-posed-me340-223-shotc/metric-scale.json").read_text())["metres_per_native_unit"]
    return {"dyn": seg.npz_bytes(packed=np.packbits(dyn, axis=-1)), "entity_obs": obs,
            "droid": seg.npz_bytes(c2w=geometry["c2w"], metres_per_native=np.array(mpn))}


@app.local_entrypoint()
def main(out: str, chains: str = "ref,da3", steps: str = ",".join(map(str, STEPS)), save_step: int = 5, smoke: bool = False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run dir
    t = time.time()
    shared = inputs()
    mp4_raw, labels_npz = (seg.CLIP / "source-rgb.mp4").read_bytes(), (INPUTS / "labels.npz").read_bytes()
    record = {"local_inputs_s": round(time.time() - t, 1), "calls": {}}
    geometry = {"ref": (INPUTS / "geometry.npz").read_bytes()}
    if "da3" in chains:
        cache = DA3_CACHE / "da3-views.npz"
        if not cache.exists():
            DA3_CACHE.mkdir(parents=True, exist_ok=False)
            t = time.time()
            report, blob = da3_views.remote((seg.CLIP / "source-full.mp4").read_bytes())
            report = json.loads(report)
            report.update(client_call_wall_s=round(time.time() - t, 1), cold_start_s_approx=round(report["enter_at"] - t, 1))
            (DA3_CACHE / "da3-views.json").write_text(json.dumps(report, indent=1))
            record["da3_views_called"] = True
            cache.write_bytes(blob)
            print("da3", json.dumps(report))
        geometry["da3"] = cache.read_bytes()
    snap = {"ref": ref_raw_depth()} if "ref" in chains else {}
    calls = {}
    for chain in chains.split(","):
        calls[chain] = (time.time(), probe.spawn(chain, mp4_raw, labels_npz, shared["dyn"], geometry[chain], shared["droid"], shared["entity_obs"],
                                                 [int(s) for s in steps.split(",")], save_step, smoke, snap.get(chain)))
    for chain, (started, call) in calls.items():
        try:
            report, maps, lines, sheet = call.get()
        except Exception as error:  # keep the other chain's results
            (out / f"{chain}-error.txt").write_text(repr(error)[:4000])
            print(chain, "failed:", repr(error)[:400])
            continue
        report = json.loads(report)
        report.update(client_call_wall_s=round(time.time() - started, 1), cold_start_s_approx=round(report["enter_at"] - started, 1))
        (out / f"{chain}.json").write_text(json.dumps(report, indent=1))
        (out / f"{chain}-maps-step{save_step}.npz").write_bytes(maps)
        (out / f"{chain}-outlines-step{save_step}.json.gz").write_bytes(lines)
        if sheet:
            (out / f"{chain}-sheet-step{save_step}.png").write_bytes(sheet)
        record["calls"][chain] = {"client_call_wall_s": report["client_call_wall_s"], "function_wall_s": report["function_wall_s"]}
        print(chain, "done in", report["client_call_wall_s"], "s")
    (out / "record.json").write_text(json.dumps(record, indent=1))
    (out / "verdicts.json").write_text(json.dumps(verdicts(out), indent=1))


def verdicts(out):
    """Speed and IoU per chain, keyframe rate and variant; pass = IoU >= 0.8 at >= 5x (M = measured, E = seconds x list price)."""
    out = Path(out)
    result = {"usd_E": {}, "chains": {}}
    per_s = USD_PER_S["A100-80GB"] + RESERVED[0] * USD_PER_S["cpu_core"] + RESERVED[1] * USD_PER_S["mem_gib"]
    da3, record = DA3_CACHE / "da3-views.json", out / "record.json"
    if da3.exists() and record.exists() and json.loads(record.read_text()).get("da3_views_called"):  # charged to the run that called it
        result["usd_E"]["da3_views"] = round((json.loads(da3.read_text())["client_call_wall_s"] + 2) * per_s, 3)
    for path in sorted(out.glob("*.json")):
        if path.stem not in ("ref", "da3"):
            continue
        report = json.loads(path.read_text())
        result["usd_E"][path.stem] = round((report["client_call_wall_s"] + 2) * per_s, 3)
        amg, enc = report["amg_s_per_frame"], report["encoder"]["s_per_frame"]
        today_s = amg * N
        rows = {}
        for step, row in report["steps"].items():
            sec = row["seconds"]
            base = row["keyframes"] * amg + sec["lift"] + sec["keyframe_points"]
            cost = {**{name: sec[name] for name in ("prev_r1", "prev", "pair", "fused", "pair_vis", "fused_vis") if name in sec},
                    **{name: sec[name.split("_")[0]] + sec[f"{name}_only"] for name in TAUS},
                    "dec_box": sec["pair"] + row["in_between"] * enc + sec["dec_box_prompts_and_decoder"],
                    "dec_pick": sec["pair"] + row["in_between"] * enc + sec["dec_pick_prompts_and_decoder"],
                    "dec_pt": sec["pair"] + row["in_between"] * enc + sec["dec_pt_prompts_and_decoder"]}
            rows[step] = {}
            for name, scores in row["variants"].items():
                total = base + cost[name]
                entry = {"total_s_M": round(total, 2), "speedup_M": round(today_s / total, 2)}
                for metric in ("m1_kf_all", "m1_kf_static", "m1_lift_static", "m2_oracle", "m2_lift"):
                    entry[f"{metric}_mean"] = scores[metric].get("mean")
                    entry[f"{metric}_area_weighted"] = scores[metric].get("area_weighted")
                entry["m2_lift_share_ge_0.8"] = scores["m2_lift"].get("share_ge_0.8")
                entry["m2_lift_projectable"] = scores["m2_lift"].get("projectable_share")
                entry["pass_at_5x"] = {f"{m}_{w}": bool(entry["speedup_M"] >= 5 and (entry[f"{m}_{w}"] or 0) >= .8)
                                       for m in ("m1_kf_static", "m1_lift_static", "m2_oracle", "m2_lift") for w in ("mean", "area_weighted")}
                rows[step][name] = entry
        result["chains"][path.stem] = {"today_amg_s_M": round(today_s, 1), "amg_s_per_frame_M": amg, "encoder_s_per_frame_M": enc,
                                       "gpu": report["gpu"], "cold_start_s_approx_M": report["cold_start_s_approx"], "steps": rows}
    result["usd_E"]["total"] = round(sum(result["usd_E"].values()), 3)
    return result


if __name__ == "__main__":
    if sys.argv[1:2] == ["--verdicts"]:
        print(json.dumps(verdicts(sys.argv[2]), indent=1))
    else:
        assert sys.argv[1:] == ["--self-check"], "usage: python modal_apps/outline_projection_probe.py --self-check | --verdicts RUN_DIR"
        self_check()
