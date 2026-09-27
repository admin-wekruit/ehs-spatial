"""The eye review of box models as a per-pixel free-space test, so no box reaches a report on a person's say-so.

A box (complete_video_objects.py --generator box) passes the fit gate by explaining the points the video saw, yet it can
still claim space the video saw empty, swallow the thing next to it, or show a face nothing was seen on. Each box is
raycast into its source view and its held-out views (the odd fit views, which never shaped it) on the depth raster. On
its silhouette, eroded ERODE px (the rim is decided by a pixel of pose or mask error), wherever the view has reliable
depth d, d is compared with the box's entry depth zf and exit depth zb:

  overhang   outside the object's mask and d > zf (1 + SLACK): the camera saw past the box's surface, so it is free space;
  foreign    in another instance's mask (not the object's) and zf (1 - SLACK) <= d < zb (1 - SLACK): another thing inside;
  occluded   outside the object's mask and d < zf (1 - SLACK): something in front, the view cannot judge the box there;

each as a share of the view's judged pixels, and the median over the views with >= MIN_JUDGED of them (one drifted view
does not decide; what only other views see still counts). sourceFaceBacking is the area share of the box side turned
most toward the source camera that lies within 2 voxels of an observed point (observed_points, the points the model's
alpha is marked from). A box fails when a median crosses THRESHOLDS or the face is less backed; a box with no judged view
fails (no pass without coverage), and so does one hidden past OCCLUDED in its median view.

THRESHOLDS come from --study: fitted on two scenes and tested on the third for all three rotations, the eye review of
the delivered merges as truth, then fitted once on all three. Held out it rejects 6 of the 10 eye-dropped boxes and keeps
27 of 32 (M, m0-box-test-study: ME340 rejects 1 of 4, as 021, 145 and 175 pass; 021 is rejected only in-sample): a box
that swallows an unsegmented neighbour lying at the object's own depth passes (ME340 145, 175). It does not replace the
eye review: merge_object_models.py records it and takes no box without a review.

  python scripts/box_free_space.py --box BOX_RUN --output NEW_DIR
  python scripts/box_free_space.py --study TEST_DIR=DELIVERED_MERGED_DIR ... [--output NEW_DIR]
  python scripts/box_free_space.py --self-check
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import complete_video_objects as cvo

ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
SLACK = .05       # relative depth; the posed depth's own error is 2-4 % (source-view gate medians)
ERODE = 2         # raster px not judged at the silhouette's rim
MIN_JUDGED = 50   # judged pixels for a view to count
OCCLUDED = .5     # coverage, not fitted: past this the median view does not see the box
BUDGET = 1 - 30 / 32  # --study: share of the eye-kept boxes the fit may reject (the plan keeps >= 30 of 32)
FITTED = ("overhang", "foreign", "sourceFaceBacking")
THRESHOLDS = {"overhang": .1994, "foreign": .1209, "sourceFaceBacking": .6746}  # --study, all three scenes (m0-box-test-study)


def box_depths(scene, k, c2w, size=(640, 480)):
    """Per-pixel z-depth where each camera ray enters and leaves the (convex, closed) box; inf where it misses."""
    import open3d as o3d
    v, u = np.indices(size[::-1])
    direction = np.stack([(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], np.ones(u.shape)], -1).reshape(-1, 3) @ c2w[:3, :3].T
    origin = np.broadcast_to(c2w[:3, 3], direction.shape)
    cast = lambda o, d: scene.cast_rays(o3d.core.Tensor(np.hstack([o, d]).astype(np.float32)))["t_hit"].numpy().astype(np.float64)
    near = cast(origin, direction)  # rays have camera z = 1, so the ray parameter is z-depth
    far = np.full(near.shape, np.inf)
    hit = np.isfinite(near)
    step = 1e-4 * np.maximum(near[hit], 1)
    far[hit] = near[hit] + step + cast(origin[hit] + direction[hit] * (near[hit] + step)[:, None], direction[hit])
    far[hit & ~np.isfinite(far)] = near[hit & ~np.isfinite(far)]  # grazing an edge: no inside
    return near.reshape(size[::-1]), far.reshape(size[::-1])


def classify(near, far, depth, own, others):
    """Judged-pixel counts of one view (see the module doc)."""
    silhouette = cv2.erode(np.isfinite(near).astype(np.uint8), np.ones((2 * ERODE + 1,) * 2, np.uint8)) > 0
    judged = silhouette & (depth > 0)
    front = judged & (depth < near * (1 - SLACK))
    return {"judged": int(judged.sum()), "overhang": int((judged & ~own & (depth > near * (1 + SLACK))).sum()),
            "foreign": int((judged & others & ~own & ~front & (depth < far * (1 - SLACK))).sum()), "occluded": int((front & ~own).sum())}


def facing_face(vertices, faces, eye):
    """Centres and areas of the triangles of the box side turned most toward `eye` (a box has six normals)."""
    import trimesh
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    normals = mesh.face_normals * (1. if mesh.volume >= 0 else -1.)
    sides, side = np.unique(np.round(normals, 2), axis=0, return_inverse=True)
    best = side.ravel() == int(np.argmax(sides @ (eye - vertices.mean(0))))
    return mesh.triangles_center[best], mesh.area_faces[best]


def face_backing(vertices, faces, eye, observed, tolerance):
    """Area share of the box side facing the source camera with an observed point within `tolerance`."""
    from scipy.spatial import cKDTree
    centres, area = facing_face(vertices, faces, eye)
    backed = cKDTree(observed).query(centres, distance_upper_bound=tolerance)[0] < np.inf
    return float(area[backed].sum() / area.sum())


def instance_masks(masks, frame, clip):
    """{observation id: raster mask} of every instance segmented in the frame (every label folder)."""
    return {f"{p.parent.parent.name.rsplit('-', 1)[0]}:{frame}:{p.stem.split('-')[1]}": clip.raster_mask(p)
            for p in sorted(masks.glob(f"*-*/frame-{frame:05d}/instance-*-mask.png"))}


def load_scene(box_run):
    """The box run's inputs as complete_video_objects read them: clip rasters, posed views (its skipped frames removed), entities."""
    import mono_room
    manifest = json.loads((box_run / "manifest.json").read_text())
    inputs = {k: Path(v) for k, v in manifest["inputs"].items() if k != "clip"}
    clip = cvo.Clip(inputs["droid_run"], ART / "data/clips" / json.loads((inputs["droid_run"] / "run.json").read_text())["clip"])  # the manifest kept the object's repr
    cvo.VOXEL = manifest["acceptance"]["max_fit_median_native"]  # the gate's residual limit is the run's voxel
    accepted = [o["entityId"] for o in manifest["objects"] if o.get("accepted")]
    skip = set(json.loads((box_run / "models" / accepted[0] / "validation.json").read_text())["skipFrames"]) if accepted else set()
    rows = {r["source_index"]: r for r in mono_room.load(inputs["droid_run"], None, inputs["depth_run"]) if r["source_index"] not in skip}
    for row in rows.values():
        for key in ("conf", "droid", "retained"):
            row.pop(key, None)
    entities = {e["entityId"]: e for e in json.loads((inputs["object_map"] / "object-map.json").read_text())["entities"]}
    return manifest, clip, rows, entities, argparse.Namespace(masks=inputs["masks"], dynamic_masks=inputs["dynamic_masks"])


def measure(box_run):
    """{entity: per-view pixel counts, face backing, mesh hash} for every accepted box of the run. Measured, CPU only."""
    import trimesh
    manifest, clip, rows, entities, args = load_scene(box_run)
    out = {}
    for o in manifest["objects"]:
        if not o.get("accepted"):
            continue
        eid, folder = o["entityId"], box_run / "models" / o["entityId"]
        v = json.loads((folder / "validation.json").read_text())
        mesh = trimesh.load(folder / "model.glb", force="mesh", process=False)
        vertices, faces = np.asarray(mesh.vertices, float), np.asarray(mesh.faces)
        names = v["fitViews"]
        source = names.index(v["observation"])
        scene, counts = cvo.ray_scene(vertices, faces), {}
        for name in [v["observation"]] + [n for i, n in enumerate(names) if i % 2 and i != source]:  # assess()'s held-out split
            frame = int(name.rsplit(":", 2)[1])
            instances = instance_masks(args.masks, frame, clip) if frame in rows else {}
            own = instances.pop(name, None)
            if own is None:
                continue
            others = np.any(list(instances.values()), 0) if instances else np.zeros(own.shape, bool)
            counts[name] = classify(*box_depths(scene, clip.k_raster, rows[frame]["c2w"]), cvo.reliable(rows[frame], clip, args.dynamic_masks)[0], own, others)
        view = {"frame": v["sourceFrame"], "observation": v["observation"], "mask": v["viewMask"]}
        observed = cvo.observed_points(entities[eid], view, rows, clip, args)[0]
        out[eid] = {"label": o["label"], "mesh_sha256": v["mesh_sha256"], "source": v["observation"], "views": counts,
                    "sourceFaceBacking": face_backing(vertices, faces, rows[v["sourceFrame"]]["c2w"][:3, 3], observed, 2 * cvo.VOXEL)}
        print(json.dumps({"entity": eid, "label": o["label"], **summary(out[eid])}), flush=True)
    return out


def summary(m):
    """Median share over the judged views, the face backing, and how many views judged."""
    judged = [c for c in m["views"].values() if c["judged"] >= MIN_JUDGED]
    median = {k: float(np.median([c[k] / c["judged"] for c in judged])) if judged else None for k in ("overhang", "foreign", "occluded")}
    return {**median, "sourceFaceBacking": m["sourceFaceBacking"], "judgedViews": len(judged)}


def verdict(m, thresholds=None):
    """Reasons the box fails the free-space test; empty when it passes."""
    t, s = thresholds or THRESHOLDS, summary(m)
    if not s["judgedViews"]:
        return [f"no view with {MIN_JUDGED} judged pixels: nothing to test the box against"]
    reasons = [f"hidden: {s['occluded']:.0%} of the median view's judged pixels occluded > {OCCLUDED:.0%}"] if s["occluded"] > OCCLUDED else []
    reasons += [f"{k} {s[k]:.0%} of the median view's judged pixels > {t[k]:.0%}" for k in ("overhang", "foreign") if s[k] > t[k]]
    if s["sourceFaceBacking"] < t["sourceFaceBacking"]:
        reasons.append(f"only {s['sourceFaceBacking']:.0%} of the face toward the source camera is backed by observed points < {t['sourceFaceBacking']:.0%}")
    return reasons


def fit(boxes, bad, budget=BUDGET):
    """THRESHOLDS for the FITTED measures: fewest bad boxes passed with at most `budget` of the good ones rejected, then
    fewest good ones rejected, then the widest smallest gap (share of the measure's range) to any box."""
    s = [summary(m) for m in boxes]
    bad = np.asarray(bad, bool)
    base = np.array([bool(verdict(m, {"overhang": np.inf, "foreign": np.inf, "sourceFaceBacking": -np.inf})) for m in boxes])  # the unfitted rules
    worse = {"overhang": 1, "foreign": 1, "sourceFaceBacking": -1}  # sign that makes larger worse
    grids = []
    for k in FITTED:
        x = np.array([worse[k] * (r[k] or 0.) for r in s])
        v = np.unique(x)
        cut = np.r_[(v[1:] + v[:-1]) / 2, np.inf]  # inf: the measure rejects nothing
        grids.append([(c, x > c, min(np.abs(x - c).min() / (np.ptp(x) or 1), 1.)) for c in cut])
    best = None
    for combo in itertools.product(*grids):
        rejected = base | np.any([r for _, r, _ in combo], 0)
        passed, lost = int((bad & ~rejected).sum()), int((~bad & rejected).sum())
        key = (lost > budget * (~bad).sum(), passed, lost, -min(g for _, _, g in combo))
        if best is None or key < best[0]:
            best = key, {k: worse[k] * c for k, (c, _, _) in zip(FITTED, combo)}
    return best[1]


def study(pairs, output=None):
    """Thresholds fitted on two scenes and tested on the third, all rotations, then fitted on all; per-rotation confusion."""
    scenes = {}
    for test, delivered in pairs:
        boxes = json.loads((test / "box-test.json").read_text())["boxes"]
        merge = json.loads((delivered / "merge.json").read_text())
        truth = {e: False for e, g in merge["choice"].items() if g == "box"}
        truth.update({e: True for e, why in merge.get("boxesDropped", {}).items() if why.startswith("excluded at review")})
        scenes[test.name] = [(e, boxes[e], truth[e]) for e in sorted(truth)]
    confusion = lambda rows, t: {"badRejected": [e for e, m, b in rows if b and verdict(m, t)], "badPassed": [e for e, m, b in rows if b and not verdict(m, t)],
                                 "goodKept": len([e for e, m, b in rows if not b and not verdict(m, t)]), "goodRejected": [e for e, m, b in rows if not b and verdict(m, t)]}
    result = {"rotations": {}, "rule": {"slack": SLACK, "erodePx": ERODE, "minJudged": MIN_JUDGED, "occluded": OCCLUDED, "budget": BUDGET}}
    for test in scenes:
        train = [r for name, rows in scenes.items() if name != test for r in rows]
        t = fit([m for _, m, _ in train], [b for _, _, b in train])
        result["rotations"][test] = {"trainedOn": [n for n in scenes if n != test], "thresholds": t, "train": confusion(train, t), "test": confusion(scenes[test], t)}
    every = [r for rows in scenes.values() for r in rows]
    t = fit([m for _, m, _ in every], [b for _, _, b in every])
    result["allScenes"] = {"thresholds": t, "inSample": confusion(every, t)}
    for name, r in result["rotations"].items():
        c = r["test"]
        print(f"test {name}: bad rejected {len(c['badRejected'])}/{len(c['badRejected']) + len(c['badPassed'])} (passed {c['badPassed']}), "
              f"good kept {c['goodKept']}/{c['goodKept'] + len(c['goodRejected'])} (rejected {c['goodRejected']}), thresholds {json.dumps(r['thresholds'])}")
    print("all scenes:", json.dumps(result["allScenes"]))
    if output:
        output.mkdir(parents=True, exist_ok=False)
        (output / "study.json").write_text(json.dumps(result, indent=1))
    return result


def self_check():
    """A 1 x 1 x 2 box 5 in front of a camera: its entry/exit depths; each pixel class; the face backing; the verdicts
    (pass, each failure, no coverage); the fit finding a separating threshold and respecting its budget."""
    import trimesh
    unit = trimesh.creation.box(bounds=[[-.5, -.5, 5], [.5, .5, 7]])
    vertices, faces = trimesh.remesh.subdivide_to_size(unit.vertices, unit.faces, .05)  # cut like box_mesh's boxes
    k, c2w = np.array([[100., 0, 32], [0, 100, 24], [0, 0, 1]]), np.eye(4)
    near, far = box_depths(cvo.ray_scene(vertices, faces), k, c2w, (64, 48))
    assert np.isclose(near[24, 32], 5, atol=1e-3) and np.isclose(far[24, 32], 7, atol=1e-3) and np.isinf(near[0, 0]), (near[24, 32], far[24, 32])
    own = np.isfinite(near)
    own[:, 32:] = False  # the object is the box's left half
    c = classify(near, far, np.where(own, 5.4, 9.), own, ~own)  # right half: another thing far behind shows through the box (free
    # space, not inside); left half: the object's own surface behind the box's face (a bag's curve) is not overhang
    assert c["judged"] > 200 and .4 < c["overhang"] / c["judged"] < .6 and c["foreign"] == c["occluded"] == 0, c
    c = classify(near, far, np.full(near.shape, 5.02), own, ~own)  # another instance at the box's face on the right half: inside it
    assert .4 < c["foreign"] / c["judged"] < .6 and c["overhang"] == c["occluded"] == 0, c
    c = classify(near, far, np.where(own, 5.02, 2.), own, ~own)  # something well in front of the right half
    assert .4 < c["occluded"] / c["judged"] < .6 and c["foreign"] == c["overhang"] == 0, c
    front = unit.sample(20000, seed=0)
    front = front[front[:, 2] < 5.001]
    assert face_backing(vertices, faces, np.zeros(3), front, .05) > .99
    assert .4 < face_backing(vertices, faces, np.zeros(3), front[front[:, 0] < 0], .05) < .6, "half the front face seen"
    view = lambda o, f=0, occ=0: {"judged": 100, "overhang": o, "foreign": f, "occluded": occ}
    good = {"views": {"a": view(0), "b": view(5), "c": view(90)}, "sourceFaceBacking": .9}  # the median ignores one drifted view
    assert verdict(good) == [], verdict(good)
    assert verdict({**good, "views": {"a": view(40), "b": view(50)}})[0].startswith("overhang 45%")
    assert verdict({**good, "views": {"a": view(0, 30)}})[0].startswith("foreign 30%")
    assert verdict({**good, "sourceFaceBacking": .3})[0].startswith("only 30% of the face")
    assert verdict({**good, "views": {"a": view(0, 0, 60)}})[0].startswith("hidden")
    assert verdict({**good, "views": {"a": {**view(0), "judged": 10}}})[0].startswith("no view"), "no pass without coverage"
    boxes = [{**good, "views": {"a": view(o)}} for o in (0, 2, 4, 6, 8, 30, 35, 50)]
    t = fit(boxes, [False] * 5 + [True] * 3)
    assert .08 < t["overhang"] < .30 and t["foreign"] == np.inf and t["sourceFaceBacking"] == -np.inf, t
    t = fit(boxes, [False] * 4 + [True, False, True, True], budget=0)  # the bad one at 8 % cannot go without rejecting the good at 30 %
    assert .30 < t["overhang"] < .35, t
    print("box_free_space self-check: passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--box", type=Path, help="a complete_video_objects.py --generator box run")
    p.add_argument("--study", nargs="*", metavar="TEST_DIR=DELIVERED_DIR", help="box-test outputs, each with the delivered merge whose eye review is the truth")
    p.add_argument("--output", type=Path, help="new folder")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        return self_check()
    if args.study:
        return study([tuple(map(Path, pair.split("=", 1))) for pair in args.study], args.output)
    boxes = measure(args.box)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "box-test.json").write_text(json.dumps({"box": str(args.box), "rule": {"slack": SLACK, "erodePx": ERODE, "minJudged": MIN_JUDGED,
                                                                                          "measured": "per view pixel counts on the depth raster; face backing"},
                                                           "boxes": boxes}, indent=1))


if __name__ == "__main__":
    main()
