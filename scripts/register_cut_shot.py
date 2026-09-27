"""Place a second shot of an edited clip (another camera) into the walk's map.

DROID gave the cut-away frames cameras, but wrong ones (another lens, another place), so the report leaves them out of
all 3D. Here a few of its frames go through Depth Anything 3 together with walk frames that see the same things; DA3
predicts all their cameras in one frame of its own. The predicted walk cameras are aligned to the known walk cameras
(similarity: rotation, translation, scale), which carries the cut-away cameras into the map. The walk cameras' residuals
after that alignment are the test: if they do not agree, the registration is refused.

    python scripts/register_cut_shot.py --droid-run R --depth-run D --clip C --shot 14:226 --output OUT [--invoke]
    python scripts/register_cut_shot.py --registration OUT --scene DYNAMIC_DIR --analysis ANALYSIS.json --mesh M --output OUT2 [--invoke]
    python scripts/register_cut_shot.py --self-check
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_apps"))
MAX_CENTRE_RESIDUAL = .1  # walk camera centres after alignment, as a share of their spread
MAX_ROTATION_RESIDUAL_DEG = 3.


def similarity(src, dst):
    """Least-squares s, R, t with dst ~ s R src + t (Umeyama)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(b.T @ a / len(src))
    d = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    s = np.trace(np.diag(sig) @ d) / (a ** 2).sum(1).mean()
    return s, r, mu_d - s * r @ mu_s


def align(predicted_c2w, known_c2w):
    """Carry predicted cameras into the known frame; residuals of the known ones say whether to trust it."""
    s, r, t = similarity(predicted_c2w[:, :3, 3], known_c2w[:, :3, 3])
    moved = lambda c2w: np.block([[r @ c2w[:3, :3], (s * r @ c2w[:3, 3] + t)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    placed = np.stack([moved(c) for c in predicted_c2w])
    centre = np.linalg.norm(placed[:, :3, 3] - known_c2w[:, :3, 3], axis=1)
    spread = np.linalg.norm(known_c2w[:, :3, 3] - known_c2w[:, :3, 3].mean(0), axis=1).mean()
    turn = [np.degrees(np.arccos(np.clip((np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2, -1, 1))) for a, b in zip(placed, known_c2w)]
    return {"scale": float(s), "rotation": r, "translation": t, "moved": moved, "centreResidualShare": float(np.median(centre) / spread),
            "rotationResidualDeg": float(np.median(turn)), "spreadNative": float(spread)}


def pick_walk(rows, frames, cut, count, gap=9):
    """Walk views that share the most SIFT matches with the cut-away frame, at least `gap` frames apart."""
    sift, bf = cv2.SIFT_create(4000), cv2.BFMatcher()
    gray = lambda i: cv2.imread(str(frames[i]), cv2.IMREAD_GRAYSCALE)
    _, dc = sift.detectAndCompute(gray(cut), None)
    scored = []
    for r in rows:
        _, dw = sift.detectAndCompute(gray(r["source_index"]), None)
        scored.append((sum(m.distance < .75 * n.distance for m, n in bf.knnMatch(dc, dw, k=2)), r["source_index"]))
    taken = []
    for n, i in sorted(scored, reverse=True):
        if len(taken) < count and all(abs(i - j) >= gap for _, j in taken):
            taken.append((n, i))
    return taken


def render(mesh_path, k, c2w, size=(640, 480)):
    """Colour and depth of the fused room seen from one camera (open3d ray casting)."""
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    faces, colours = np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = scene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(np.linalg.inv(c2w)), size[0], size[1])
    hit = scene.cast_rays(rays)
    ids = hit["primitive_ids"].numpy().astype(np.int64)
    inside = ids != scene.INVALID_ID
    image = np.zeros((size[1], size[0], 3))
    image[inside] = colours[faces[ids[inside]]].mean(1)
    depth = np.where(inside, hit["t_hit"].numpy() * (rays.numpy()[..., 3:] @ c2w[:3, 2]), 0)  # along the optical axis
    return (image * 255).astype(np.uint8), depth.astype(np.float32)


def run(args):
    import mono_room
    args.output.mkdir(parents=True, exist_ok=True)
    a, b = map(int, args.shot.split(":"))
    frames = sorted((args.clip / "rgb").glob("*.png"))
    rows = [r for r in mono_room.load(args.droid_run, None, args.depth_run) if not a <= r["source_index"] < b and r["scale"]]
    cuts = args.cut_frames or [a + (b - a) // 6, (a + b) // 2, b - 1 - (b - a) // 6]
    walk = pick_walk(rows, frames, cuts[len(cuts) // 2], args.walk_count)
    order = cuts + [i for _, i in walk]
    journal = args.output / "da3-unposed.npz"
    if not journal.exists():
        if not args.invoke:
            return print(json.dumps({"planned": order, "walkMatches": walk, "note": "no call without --invoke"}))
        import modal
        from da3_unposed import app, infer_unposed
        pngs = [cv2.imencode(".png", cv2.imread(str(frames[i])))[1].tobytes() for i in order]
        with modal.enable_output(), app.run():
            journal.write_bytes(infer_unposed.remote(pngs, args.model))
    out = np.load(journal)
    predicted = np.stack([np.linalg.inv(np.vstack([w, [0, 0, 0, 1]])) for w in out["w2c"]])
    known = {r["source_index"]: r["c2w"] for r in rows}
    n = len(cuts)
    fit = align(predicted[n:], np.stack([known[i] for i in order[n:]]))
    cut_c2w = [fit["moved"](c) for c in predicted[:n]]
    accepted = fit["centreResidualShare"] <= MAX_CENTRE_RESIDUAL and fit["rotationResidualDeg"] <= MAX_ROTATION_RESIDUAL_DEG
    k = np.median(out["K"][:n], 0)
    result = {"shot": [a, b], "cutFrames": cuts, "walkFrames": order[n:], "walkMatches": walk, "model": args.model,
              "scaleDa3ToNative": fit["scale"], "centreResidualShare": fit["centreResidualShare"], "rotationResidualDeg": fit["rotationResidualDeg"],
              "gate": {"maxCentreResidualShare": MAX_CENTRE_RESIDUAL, "maxRotationResidualDeg": MAX_ROTATION_RESIDUAL_DEG}, "accepted": bool(accepted),
              "K": k.tolist(), "fovXDeg": float(np.degrees(2 * np.arctan(320 / k[0, 0]))),
              "cutC2w": {str(i): c.tolist() for i, c in zip(cuts, cut_c2w)},
              "cutCentreSpreadNative": float(np.linalg.norm(np.stack([c[:3, 3] for c in cut_c2w]) - np.mean([c[:3, 3] for c in cut_c2w], 0), axis=1).max()),
              "coordinateFrame": "droid_final_native_world"}
    if args.mesh:
        tiles = []
        for i, c2w in zip(cuts, cut_c2w):
            colour, depth = render(args.mesh, k, c2w)
            frame = cv2.imread(str(frames[i]))
            ratio = out["depth"][cuts.index(i)] * fit["scale"] / np.where(depth > 0, depth, np.nan)
            result.setdefault("depthRatioMedian", []).append(float(np.nanmedian(ratio)))
            tiles.append(np.hstack([frame, cv2.addWeighted(frame, .5, colour[..., ::-1], .5, 0), colour[..., ::-1]]))
        cv2.imwrite(str(args.output / "overlay.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    (args.output / "registration.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("accepted", "fovXDeg", "scaleDa3ToNative", "centreResidualShare", "rotationResidualDeg", "cutCentreSpreadNative")}
                     | {"depthRatioMedian": result.get("depthRatioMedian")}))


def dynamic(args):
    """The moving entities of the registered shot as timed surfaces: a copy of the dynamic scene whose cut-away frames get
    the registered camera and a surface from DA3 depth tied to the map on that view's static pixels. With --merge
    OLD=NEW, a track split at the cut becomes the walk's track (the same text prompt found one instance in each shot)."""
    import shutil
    import subprocess
    import mono_room
    from build_video_object_models import observed_surface
    from build_video_surfaces import export_surface
    mono_room.use_clip(args.droid_run)
    registration = json.loads((args.registration / "registration.json").read_text())
    assert registration["accepted"], "the registration was refused: no surfaces from that shot"
    a, b = registration["shot"]
    scene = json.loads((args.scene / "scene.json").read_text())
    analysis = json.loads(args.analysis.read_text())
    merge = dict(item.split("=") for item in args.merge)
    wanted = [f["sourceFrame"] for f in scene["frames"] if a <= f["sourceFrame"] < b]
    frames = sorted((args.clip / "rgb").glob("*.png"))
    journal = args.output / "da3-unposed.npz"
    walk_frames = registration["walkFrames"]
    order = wanted + walk_frames
    args.output.mkdir(parents=True, exist_ok=True)
    if not journal.exists():
        if not args.invoke:
            return print(json.dumps({"planned": len(order), "cutFrames": len(wanted), "note": "no call without --invoke"}))
        import modal
        from da3_unposed import app, infer_unposed
        pngs = [cv2.imencode(".png", cv2.imread(str(frames[i])))[1].tobytes() for i in order]
        with modal.enable_output(), app.run():
            journal.write_bytes(infer_unposed.remote(pngs, args.model))
    out = np.load(journal)
    predicted = np.stack([np.linalg.inv(np.vstack([w, [0, 0, 0, 1]])) for w in out["w2c"]])
    rows = {r["source_index"]: r for r in mono_room.load(args.droid_run, None, args.depth_run) if r["source_index"] in walk_frames}
    n = len(wanted)
    fit = align(predicted[n:], np.stack([rows[i]["c2w"] for i in walk_frames]))
    accepted = fit["centreResidualShare"] <= MAX_CENTRE_RESIDUAL and fit["rotationResidualDeg"] <= MAX_ROTATION_RESIDUAL_DEG
    assert accepted, fit
    cams = [fit["moved"](c) for c in predicted[:n]]
    centres = np.stack([c[:3, 3] for c in cams])
    k = np.median(out["K"][:n], 0)
    # one static camera: every frame uses the median placement (per-frame predictions differ only by their noise)
    rotation = cv2.Rodrigues(np.median([cv2.Rodrigues(c[:3, :3])[0].ravel() for c in cams], 0))[0]
    c2w = np.block([[rotation, np.median(centres, 0)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]])
    _, map_depth = render(args.mesh, k, c2w)
    target = args.output / "scene"
    if target.exists():
        shutil.rmtree(target)
    subprocess.run(["cp", "-c", "-R", args.scene, target], check=True)  # clones: the walk's surfaces stay byte-identical
    masks = {(f["sourceFrame"], o["entityId"]): o["maskUrl"] for f in analysis["frames"] for o in f["objects"] if o.get("maskUrl")}
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    ties, surfaces = [], 0
    by_frame = {f["sourceFrame"]: f for f in scene["frames"]}
    for i, index in enumerate(wanted):
        frame = by_frame[index]
        bgr, _ = mono_room.prepare_image(cv2.imread(str(mono_room.DATASET / manifest["frames"][index]["relative_path"])), mono_room.CALIBRATION, 2)
        depth = out["depth"][i] * fit["scale"]
        moving = mono_room.moving_mask(args.analysis.parent / "masks", index)
        static = ~moving & (map_depth > 0) & (depth > 0)
        tie = float(np.median(map_depth[static] / depth[static])) if static.sum() >= 5000 else None
        if tie is None:
            frame["objects"] = []
            continue
        ties.append(tie)
        depth = np.where(mono_room.unreliable(depth * tie, None, None, .03), 0, depth * tie)
        objects = []
        for o in frame["objects"]:
            url = masks.get((index, o["entityId"]))
            if not url:
                continue
            mask = mono_room.prepare_image(cv2.imread(str(args.analysis.parent / url), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
            if not (mask & (depth > 0)).any():
                continue
            reach = float(np.median(depth[mask & (depth > 0)]))
            # a mask's edge pixels can take the background's depth (the wall 20 m behind): one body spans far less than +-30% of its distance
            vertices, faces, colours, _, _ = observed_surface(bgr[..., ::-1], depth, mask, k, c2w, max_edge_m=.04 * reach, depth_range=(.7 * reach, 1.3 * reach))
            if not len(faces):
                continue
            path = target / "dynamic" / f"registered-{index:05d}-{len(objects)}.glb"
            export_surface(path, vertices, faces, colours)
            objects.append({**{key: o[key] for key in ("representation", "world_motion") if key in o}, "entityId": merge.get(o["entityId"], o["entityId"]),
                            "keypoints3d": [], "bones": [], "centroid": np.median(vertices, 0).tolist(),
                            "surface": {"meshUrl": f"dynamic/{path.name}", "sourceFrame": index, "representation": "visible_monocular_surface",
                                        "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(), "triangles": len(faces)}})
            surfaces += 1
        frame.update(objects=objects, c2w=c2w.tolist(), poseSource="registered_cut_shot: DA3 unposed with walk anchors, aligned to the walk cameras (register_cut_shot.py)")
    scene["registeredShots"] = [{"frames": [a, b], "surfaces": surfaces, "K": k.tolist(), "cameraToWorld": c2w.tolist(), "merged": merge,
                                 "centreResidualShare": fit["centreResidualShare"], "rotationResidualDeg": fit["rotationResidualDeg"],
                                 "depthTieMedian": float(np.median(ties)) if ties else None, "depthTieSpread": float(np.ptp(ties)) if ties else None}]
    (target / "scene.json").write_text(json.dumps(scene, allow_nan=False, separators=(",", ":")))
    # the analysis the import reads for outlines and names: split tracks merged, and a text-prompted track named by its prompt
    for f in analysis["frames"]:
        for o in f["objects"]:
            o["entityId"] = merge.get(o["entityId"], o["entityId"])
            if args.person_tracks and o["entityId"] in args.person_tracks and not o.get("sourceLabel"):
                o.update(sourceLabel="person", sourceLabelBasis="SAM 3.1 text prompt 'person' (tracks.json of the same run)")
    (args.output / "analysis").mkdir(exist_ok=True)
    link = args.output / "analysis" / "masks"
    if not link.exists():
        link.symlink_to((args.analysis.parent / "masks").resolve())
    (args.output / "analysis" / "analysis.json").write_text(json.dumps(analysis, separators=(",", ":")))
    print(json.dumps(scene["registeredShots"][0] | {"cutFramesWithSurfaces": sum(bool(by_frame[i]["objects"]) for i in wanted), "cutFrames": n}))


def self_check():
    rng = np.random.default_rng(0)
    rot = lambda v: cv2.Rodrigues(np.asarray(v, float))[0]
    known = np.stack([np.block([[rot(rng.normal(0, .3, 3)), rng.normal(0, 1, (3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]) for _ in range(8)])
    r, s, t = rot([.2, -.4, .1]), 2.5, np.array([1., -2, .5])
    predicted = np.stack([np.block([[r.T @ c[:3, :3], (r.T @ (c[:3, 3] - t) / s)[:, None]], [np.zeros((1, 3)), np.ones((1, 1))]]) for c in known])
    fit = align(predicted, known)
    assert abs(fit["scale"] - s) < 1e-9 and fit["centreResidualShare"] < 1e-9 and fit["rotationResidualDeg"] < 1e-5, fit
    assert np.allclose(fit["moved"](predicted[3]), known[3]), "a predicted camera lands on its known pose"
    noisy = predicted.copy()
    noisy[:, :3, 3] += rng.normal(0, .5, (8, 3))
    assert align(noisy, known)["centreResidualShare"] > MAX_CENTRE_RESIDUAL, "cameras that do not agree are refused"
    print("register_cut_shot self-check passed: similarity recovered exactly, a predicted camera lands on its pose, disagreeing cameras refused")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    for name in ("droid-run", "depth-run", "clip", "output", "mesh", "registration", "scene", "analysis"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--merge", nargs="*", default=[], metavar="OLD=NEW", help="dynamic stage: a track split at the cut joins the walk's track")
    parser.add_argument("--person-tracks", nargs="*", default=[], help="dynamic stage: text-prompted 'person' tracks whose analysis lost the label")
    parser.add_argument("--shot", default="14:226", help="START:END frames of the other shot")
    parser.add_argument("--cut-frames", nargs="*", type=int, default=[])
    parser.add_argument("--walk-count", type=int, default=14)
    parser.add_argument("--model", default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--invoke", action="store_true", help="make the one A100 call (else plan only, or reuse da3-unposed.npz)")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    dynamic(args) if args.registration else run(args)


if __name__ == "__main__":
    main()
