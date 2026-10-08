"""Coverage of a Scene (C1 `Scene.coverage`) by space-carving the frozen photo geometry. README-coverage.md has method and limits.

Input: one directory per frame, the measurement pipeline's frozen geometry
(research/module-swap-2026-10-07/data/checks/bbab-geom/<cell>-mvs-fill-padded/geometry/frames/frame_000k):
  pts3d.npy            H x W x 3 surface point behind every pixel, in the producer's world frame (native units, see `native_to_m`)
  valid_mask.npy       H x W bool: pixels that have a surface point
  camera_to_world.npy  4 x 4; the camera centre is column 3
  conf.npy (optional)  H x W; 1 on measured pixels, lower on model-filled ones. Pixels below `min_conf` cast no ray.
  intrinsics.npy       not needed: pts3d already is the end of every pixel's ray

Every ray is marched from the camera centre to its surface point in steps of cell_m / 2. The plan cell under a sample is marked
observed when the sample lies in the slab [h_min, h_max] above the floor plane: free space along the ray, the surface at its end.
A cell no ray touched stays unknown; it is never 'empty'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ehs_spatial.verdict.contracts import Coverage, Scene

H_MIN_M, H_MAX_M = -0.10, 2.0     # the slab that counts: -0.10 keeps noisy floor hits, 2.0 = the reach height the rules care about


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def plan_basis(ground_normal) -> list[list[float]]:
    """[e1, e2] spanning the floor plane, the trial's relations.plan_basis convention, so L2's Grid can share the frame."""
    n = unit(ground_normal)
    e1 = unit(np.cross(n, [1, 0, 0]) if abs(n[0]) < 0.9 else np.cross(n, [0, 1, 0]))
    return [e1.tolist(), np.cross(n, e1).tolist()]


def floor_level(scene: Scene) -> float:
    """n . p of the floor plane (metres), from the objects: a box base sits at n . center - H / 2 (axes[2] is the floor normal),
    bottom_m above the floor. Median over the objects."""
    n = unit(scene.ground_normal)
    return float(np.median([n @ np.asarray(o.center_m) - o.size_m[2] / 2 - o.bottom_m for o in scene.objects]))


def plan_frame(scene: Scene, cell_m: float = 0.10, margin_m: float = 1.0) -> tuple[list[list[float]], list[float], list[int]]:
    """(basis, origin_xy, shape) of the plan grid: the boxes' corner extent plus a margin (= the trial's relations.plan_grid)."""
    basis = plan_basis(scene.ground_normal)
    E = np.asarray(basis)
    xy = []
    for o in scene.objects:
        c, a, h = np.asarray(o.center_m), np.asarray(o.axes, float), np.asarray(o.size_m) / 2
        xy += [(c + a[0] * sx * h[0] + a[1] * sy * h[1] + a[2] * sz * h[2]) @ E.T for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    xy = np.asarray(xy)
    lo, hi = xy.min(0) - margin_m, xy.max(0) + margin_m
    return basis, lo.tolist(), [int(np.ceil((hi[0] - lo[0]) / cell_m)), int(np.ceil((hi[1] - lo[1]) / cell_m))]


def _rays(frame_dir: Path, native_to_m: float, stride: int, min_conf: float) -> tuple[np.ndarray, np.ndarray]:
    """(camera centre, N x 3 ray ends) in metres for the kept pixels of one frame, subsampled by `stride` on both image axes."""
    d = Path(frame_dir)
    keep = np.load(d / "valid_mask.npy")[::stride, ::stride].astype(bool)
    if (d / "conf.npy").exists():
        keep &= np.load(d / "conf.npy")[::stride, ::stride] >= min_conf
    ends = np.asarray(np.load(d / "pts3d.npy", mmap_mode="r")[::stride, ::stride], dtype=np.float64)[keep]
    ends = ends[np.isfinite(ends).all(1)]
    cam = np.load(d / "camera_to_world.npy")[:3, 3].astype(np.float64)
    return cam * native_to_m, ends * native_to_m


def estimate_floor_level(frame_dirs, ground_normal, native_to_m: float = 1.0, stride: int = 8, min_conf: float = 0.5, bin_m: float = 0.05) -> float:
    """Floor offset along the normal from the frames alone: the most populated height bin, refined by the median within one bin of it.
    Prefer floor_level(scene) when a Scene exists; this is the fallback for frames without one."""
    n = unit(ground_normal)
    h = np.concatenate([_rays(d, native_to_m, stride, min_conf)[1] @ n for d in frame_dirs])
    edges = np.arange(np.floor(h.min() / bin_m) * bin_m, h.max() + 2 * bin_m, bin_m)
    counts, _ = np.histogram(h, edges)
    centre = edges[counts.argmax()] + bin_m / 2
    return float(np.median(h[np.abs(h - centre) <= bin_m]))


def coverage_from_frames(frame_dirs, ground_normal, basis, origin_xy, cell_m, shape, *, floor_level: float | None = None,
                         native_to_m: float = 1.0, stride: int = 4, min_conf: float = 0.5, h_min: float = H_MIN_M, h_max: float = H_MAX_M,
                         max_range_m: float = 15.0, chunk: int = 1024) -> Coverage:
    """Space-carve the frames onto the plan grid (basis / origin_xy / cell_m / shape as the Coverage contract defines them).

    floor_level  n . p of the floor plane in metres (floor_level(scene)); None = estimate_floor_level from the frames
    native_to_m  producer units -> metres (the measurement layer's scale.nativeToMeters for the frozen MVS frames)
    stride       pixel subsampling; 4 on a 518 px frame is one ray per ~6 cm at 6 m, finer than a 10 cm cell
    max_range_m  rays are cut there (memory, not physics); chunk = rays marched at once (chunk x max_range_m / step x 3 doubles)
    """
    frame_dirs = [Path(d) for d in frame_dirs]
    n, E, lo = unit(ground_normal), np.asarray(basis, float), np.asarray(origin_xy, float)
    nx, ny = int(shape[0]), int(shape[1])
    if floor_level is None:
        floor_level = estimate_floor_level(frame_dirs, n, native_to_m, stride, min_conf)
    step = cell_m / 2
    t_all = step * np.arange(1, int(np.ceil(max_range_m / step)) + 1)         # sample distances along a ray
    observed = np.zeros((nx, ny), bool)
    for d in frame_dirs:
        cam, ends = _rays(d, native_to_m, stride, min_conf)
        for i in range(0, len(ends), chunk):
            e = ends[i:i + chunk]
            length = np.linalg.norm(e - cam, axis=1)
            e, length = e[length > 0], length[length > 0]
            if not len(e):
                continue
            u = (e - cam) / length[:, None]
            k = min(len(t_all), int(np.ceil(length.max() / step)) + 1)
            t = np.minimum(t_all[None, :k], length[:, None])                  # clamped: past the surface the end point repeats
            p = (cam + u[:, None, :] * t[:, :, None]).reshape(-1, 3)
            h = p @ n - floor_level
            ij = np.floor((p[(h >= h_min) & (h <= h_max)] @ E.T - lo) / cell_m).astype(int)
            ok = (ij[:, 0] >= 0) & (ij[:, 0] < nx) & (ij[:, 1] >= 0) & (ij[:, 1] < ny)
            observed[ij[ok, 0], ij[ok, 1]] = True
    return Coverage(cell_m=float(cell_m), origin_xy=[float(x) for x in lo], basis=[[float(x) for x in e] for e in E], shape=[nx, ny],
                    observed=np.argwhere(observed).tolist(),
                    method=f"space-carve frames={len(frame_dirs)} stride={stride} step_m={step:g} slab_m=[{h_min:g},{h_max:g}] "
                           f"min_conf={min_conf:g} native_to_m={native_to_m:g} floor_level_m={floor_level:.4f}")


def attach_coverage(scene: Scene, coverage: Coverage | None) -> Scene:
    return scene.model_copy(update={"coverage": coverage})
