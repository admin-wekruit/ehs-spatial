"""  python scripts/lingbot_icp_refine.py MAP_DIR NEW_DIR FUSED_RUN

Refine a LingBot dense map's Sim3 by trimmed ICP against the report's fused mesh surface (a similarity: scale, rotation,
translation). Static points only: the map already leaves moving-entity pixels out and the fused mesh was fused without them.
Writes a new map folder; the source map folder is not touched."""
import json, shutil, sys
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "modal_apps")]
from lingbot_dense_map import read_points_glb, write_points_glb, umeyama

src, out, fused = (Path(p) for p in sys.argv[1:4])
metres = json.loads((fused / "metric-scale.json").read_text())["metres_per_native_unit"]
mesh = trimesh.load(fused / "mono-anchored-mesh.ply", process=False)
target = trimesh.sample.sample_surface(mesh, 3_000_000, seed=0)[0]
tree = cKDTree(target)
xyz, rgb = read_points_glb(src / "dense-points.glb")
attrs = dict(np.load(src / "point-attributes.npz"))
rng = np.random.default_rng(0)
confident = np.flatnonzero(~attrs["fill"])
sub = xyz[rng.choice(confident, min(400_000, len(confident)), replace=False)].astype(np.float64)
TRIM = .7  # each iteration fits the closest 70% of the correspondences
s, R, t, log = 1., np.eye(3), np.zeros(3), []
for reach_m, iterations in ((1., 15), (.5, 15), (.25, 20)):
    for _ in range(iterations):
        cur = s * sub @ R.T + t
        d, idx = tree.query(cur, distance_upper_bound=reach_m / metres, workers=-1)
        ok = np.isfinite(d)
        keep = ok & (d <= np.quantile(d[ok], TRIM))
        ds, dR, dt = umeyama(cur[keep], target[idx[keep]])
        s, R, t = ds * s, dR @ R, ds * dR @ t + dt
        log.append({"reach_m": reach_m, "used": int(keep.sum()), "median_cm": round(float(np.median(d[keep])) * metres * 100, 2)})
        if abs(ds - 1) < 1e-6 and np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))) < 1e-4 and np.linalg.norm(dt) * metres < 1e-4:
            break
angle = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
out.mkdir(parents=True, exist_ok=False)
moved = (s * xyz.astype(np.float64) @ R.T + t).astype(np.float32)
write_points_glb(out / "dense-points.glb", moved, rgb)
attrs["spacing"] = (attrs["spacing"].astype(np.float32) * s).astype(np.float16)
np.savez_compressed(out / "point-attributes.npz", **attrs)
for name in ("plan.json", "lingbot-run.json", "diagnose.json"):
    if (src / name).exists():
        shutil.copy(src / name, out / name)
remote = json.loads((src / "remote.json").read_text())
remote["cell_native_final"] *= s
(out / "remote.json").write_text(json.dumps(remote))
info = json.loads((src / "points.json").read_text())
info.pop("metrics", None); info.pop("comparison_images", None)
info.update(cell_native=info["cell_native"] * s, file_bytes=(out / "dense-points.glb").stat().st_size)
import hashlib
info["file_sha256"] = hashlib.sha256((out / "dense-points.glb").read_bytes()).hexdigest()
info["icp_refinement"] = {"of": str(src), "against": str(fused / "mono-anchored-mesh.ply"),
    "rule": f"trimmed similarity ICP (closest {TRIM:.0%} of correspondences, reach 1.0 -> 0.5 -> 0.25 m), confident points only, 3M area-uniform samples of the fused mesh; applied after the camera-centre Sim3",
    "scale": s, "rotation_deg": angle, "rotation": R.tolist(), "translation_native": t.tolist(), "translation_m": float(np.linalg.norm(t) * metres),
    "iterations": len(log), "first": log[0], "last": log[-1]}
info["note"] = info["note"] + " Placed by the camera-centre Sim3, then refined by trimmed ICP against the fused mesh (icp_refinement)."
(out / "points.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info["icp_refinement"] | {"rotation": None}, indent=1))
