"""The M2 decision rules (docs/phase2 M2 runner design, section 5): every hand decision of the delivered reports as a rule over
cached run outputs, run as its own keyed CPU stage.

  RULES[name](**{role: Path}) -> {"value": ..., "evidence": ..., "rule": "name@version"}
  python -m report_runner.decide NAME --out DIR role=PATH ...      -> DIR/NAME.json
  python -m report_runner.decide --self-check

A role names what the rule reads (a producing stage's output: a file, or the run folder holding it); the roles of a rule are
the parameters of its function, and an unknown role is refused. Only `value` is meant for downstream stages; `evidence` says
why. Spans are end-exclusive [a, b) source frames. A rule that cannot decide raises; it never guesses (wrong is worse than
blank). dense_gate and static_filter also write files into --out (the refined map; the filtered object map).
"""
import argparse
import inspect
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
ROOT = SCRIPTS.parent
for extra in (SCRIPTS, ROOT / "modal_apps"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

MIN_SHOT_FRAMES, MIN_SHOT_KEYFRAMES = 60, 8
LENS_FRAMES, LENS_KEEP_DEG, MIN_PLANE_INLIERS = 15, 3., .90
VOXEL_M = .04
OVERLAY_PAIRS, OVERLAY_MOVE_PX, OVERLAY_ROW_PX, OVERLAY_SHARE, OVERLAY_PER_PAIR, OVERLAY_PAD_PX = 150, 4., 8, .10, 5, 12
CAPTION_LAYERS = ("texture", "fill", "lingbot", "splat", "objects")
LINGBOT_FRAMES, LINGBOT_SHARE_4PCT = 450, .90
DENSE_WITHIN_25CM, DENSE_FLOOR_M, FLOOR_BAND_M, FLOOR_CELL_M = .45, .05, .3, .25
TRACK_CAP, TRACK_WINDOW, TRACK_STEP = 400, 300, 260
SPLAT_DB = .2
FLOOR_MASK_FRAMES = 12
PERSON_SHARE = .30
PERSON = ("man", "woman", "person", "people", "worker", "human")  # the person words of complete_video_objects.EXCLUDED
UNNAMED = "unnamed surface"
# bump a rule's version when its code changes (tests/versions.json 'rule:NAME' guards it; stages.rule_code is what counts)
# floor_frames 2: the operator list applies only to the clip it names (source video sha256 and window)
# inferred_floor 2: only a dense-validated floor on a plane the lens gate passed
VERSIONS = {name: 1 for name in ("shots", "other_shot", "lens", "voxel", "overlay", "lingbot_stride", "lingbot_conf", "dense_gate",
                                 "track_windows", "splat_pick", "sam2_frames", "generator_plan", "static_filter")} | {"floor_frames": 2, "inferred_floor": 2}


def _file(path, name):
    """A role may name the file itself or the run folder that holds it."""
    path = Path(path)
    return path / name if path.is_dir() else path


def _json(path, name):
    return json.loads(_file(path, name).read_text())


def _value(path, name):
    """The value of an earlier decision (DIR/NAME.json or the file)."""
    return _json(path, f"{name}.json")["value"]


def _result(name, value, evidence):
    return {"value": value, "evidence": evidence, "rule": f"{name}@{VERSIONS[name]}"}


def _keyframes(census):
    return np.load(_file(census, "prediction.npz"))["keyframe_source_indices"].astype(int)


# ---------------------------------------------------------------- D2, D3: which shot is mapped, what happens to the others
def pick_shots(spans, keyframes):
    """spans [[a, b)], keyframe indices -> (primary index, keyframes per span). Most census keyframes, ties to more frames."""
    counts = [int(((keyframes >= a) & (keyframes < b)).sum()) for a, b in spans]
    return max(range(len(spans)), key=lambda i: (counts[i], spans[i][1] - spans[i][0])), counts


def shots(segments, census):
    """D2. segments: detect_shot_cuts segments.json (inclusive [a, b]); census: a whole-clip DROID run (only keyframes used)."""
    spans = [[int(a), int(b) + 1] for a, b in _json(segments, "segments.json")["segments"]]
    best, counts = pick_shots(spans, _keyframes(census))
    a, b = spans[best]
    mapped = b - a >= MIN_SHOT_FRAMES and counts[best] >= MIN_SHOT_KEYFRAMES
    return _result("shots", {"primary": spans[best], "others": [s for i, s in enumerate(spans) if i != best], "mapped": mapped},
                   {"spans": spans, "censusKeyframes": counts, "minFrames": MIN_SHOT_FRAMES, "minKeyframes": MIN_SHOT_KEYFRAMES})


def other_shot(registration):
    """D3. register_cut_shot's own gate, recomputed from the fit (the record's `accepted` is evidence, not the decision)."""
    from register_cut_shot import MAX_CENTRE_RESIDUAL, MAX_ROTATION_RESIDUAL_DEG
    fit = _json(registration, "registration.json")
    accepted = fit["centreResidualShare"] <= MAX_CENTRE_RESIDUAL and fit["rotationResidualDeg"] <= MAX_ROTATION_RESIDUAL_DEG
    return _result("other_shot", {"shot": [int(v) for v in fit["shot"]], "accepted": bool(accepted),
                                  "centreResidualShare": fit["centreResidualShare"], "rotationResidualDeg": fit["rotationResidualDeg"]},
                   {"gate": {"maxCentreResidualShare": MAX_CENTRE_RESIDUAL, "maxRotationResidualDeg": MAX_ROTATION_RESIDUAL_DEG},
                    "recordedAccepted": fit.get("accepted")})


# ---------------------------------------------------------------- D4, D5: lens and voxel
def lens_frames(shot):
    """The MoGe-3 frames of a shot [a, b): 15 evenly spaced, first and last frame included."""
    a, b = shot
    return [int(round(v)) for v in np.linspace(a, b - 1, LENS_FRAMES)]


def fov_deg(fx, half_width=320.):
    return math.degrees(2 * math.atan(half_width / fx))


def fx_of(fov, half_width=320.):
    return half_width / math.tan(math.radians(fov) / 2)


def _moge_fovs(record):
    """Both MoGe-3 records on disk: {frames: [int], per_frame_deg} (fov-check.json) or {frames: [{frame, fov_x_deg}]} (shot-fov.json)."""
    frames = record["frames"]
    if frames and isinstance(frames[0], dict):
        return [int(f["frame"]) for f in frames], [float(f["fov_x_deg"]) for f in frames]
    return [int(f) for f in frames], [float(v) for v in record["per_frame_deg"]]


def lens(shots, clip, moge=None, metric=None):
    """D4. Keep the clip's K when the shot's MoGe-3 median FOV is within 3 deg of it, else derive fx = 320 / tan(fov / 2).
    Then the floor-plane gate: plane inliers < 0.90 makes the scale intrinsics_uncertain (no measurements; only the import
    reads it). moge: the MoGe-3 record on lens_frames(primary); metric: metric-scale.json of the mapped shot. Either may
    still be pending, not both."""
    assert moge or metric, "lens needs the MoGe-3 record, the metric scale, or both"
    primary = _value(shots, "shots")["primary"]
    frames = lens_frames(primary)
    clip_fov = fov_deg(_json(clip, "clip.json")["K"][0])
    value = {"frames": frames, "fov_deg": None, "clip_fov_deg": clip_fov, "keep": None, "fx": None, "plane_inlier_fraction": None, "scale_status": None}
    evidence = {"keepWithinDeg": LENS_KEEP_DEG, "minPlaneInliers": MIN_PLANE_INLIERS}
    if moge:
        seen, fovs = _moge_fovs(_json(moge, "fov.json"))
        if seen != frames:
            raise ValueError(f"MoGe-3 ran on {seen}, the rule asks for {frames}: refuse rather than mix frame sets")
        fov = float(np.median(fovs))
        keep = abs(fov - clip_fov) <= LENS_KEEP_DEG
        value.update(fov_deg=fov, keep=bool(keep), fx=None if keep else fx_of(fov))
        evidence.update(perFrameDeg=fovs, spreadDeg=float(np.ptp(fovs)))
    if metric:
        scale = _json(metric, "metric-scale.json")
        inliers = float(scale["plane_inlier_fraction"])
        value.update(plane_inlier_fraction=inliers, scale_status=scale["scale_status"] if inliers >= MIN_PLANE_INLIERS else "intrinsics_uncertain")
        evidence.update(recordedScaleStatus=scale["scale_status"])
    return _result("lens", value, evidence)


def voxel(metric):
    """D5. 4 cm in native units, for fuse --voxel-length-native and complete_video_objects --voxel-native."""
    metres = _json(metric, "metric-scale.json")["metres_per_native_unit"]
    return _result("voxel", round(VOXEL_M / metres, 6), {"metresPerNativeUnit": metres, "metres": VOXEL_M})


# ---------------------------------------------------------------- D6: burned-in overlay band
def overlay_rows(images):
    """Raster rows of ORB matches that stay still although the pair's scene homography moves them >= 4 px (captions, logos).
    images: consecutive-frame grey pairs. Returns (rows, pairs used)."""
    import cv2
    orb, matcher = cv2.ORB_create(1000), cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    rows, used = [], 0
    for a, b in images:
        (ka, da), (kb, db) = orb.detectAndCompute(a, None), orb.detectAndCompute(b, None)
        if da is None or db is None:
            continue
        matches = matcher.match(da, db)
        if len(matches) < 20:
            continue
        pa = np.float32([ka[m.queryIdx].pt for m in matches])
        pb = np.float32([kb[m.trainIdx].pt for m in matches])
        moving = np.linalg.norm(pb - pa, axis=1) > 1.5
        if moving.sum() < 12:
            continue
        H, _ = cv2.findHomography(pa[moving], pb[moving], cv2.RANSAC, 3.)
        if H is None:
            continue
        predicted = cv2.perspectiveTransform(pa.reshape(-1, 1, 2), H).reshape(-1, 2)
        rows += pa[~moving & (np.linalg.norm(predicted - pa, axis=1) >= OVERLAY_MOVE_PX), 1].tolist()
        used += 1
    return rows, used


def overlay_bands(rows, pairs, crop_xywh, raster_h=480):
    """Rows -> bands in video pixels [y0, y1): 8-px raster rows holding >= 10% of the still matches, only when there are >= 5
    per pair; x1.5 (crop height / raster), padded 12 px, snapped outward to the 8-px grid."""
    rows = np.asarray(rows, float)
    if not pairs or len(rows) < OVERLAY_PER_PAIR * pairs:
        return []
    hist = np.histogram(rows, bins=raster_h // OVERLAY_ROW_PX, range=(0, raster_h))[0]
    hot = np.flatnonzero(hist >= OVERLAY_SHARE * len(rows))
    x, y, w, h = crop_xywh
    bands, scale = [], h / raster_h
    for run in np.split(hot, np.flatnonzero(np.diff(hot) > 1) + 1) if len(hot) else []:
        v0, v1 = y + run[0] * OVERLAY_ROW_PX * scale - OVERLAY_PAD_PX, y + (run[-1] + 1) * OVERLAY_ROW_PX * scale + OVERLAY_PAD_PX
        bands.append([int(max(math.floor(v0 / 8) * 8, 0)), int(math.ceil(v1 / 8) * 8)])
    return bands


def overlay_flags(bands, crop_xywh):
    """The flags each caption-aware stage takes (their defaults are ME340's band, so every value is explicit)."""
    x, _, w, _ = crop_xywh
    if len(bands) > 1:  # two caption blocks: no rule places both, so those layers stay blank
        return {"bands": bands, "texture_rows": None, "fill_rows": None, "lingbot_rows": None, "splat_captions": None,
                "objects_no_captions": None, "refuse": list(CAPTION_LAYERS)}
    if not bands:
        return {"bands": [], "texture_rows": None, "fill_rows": None, "lingbot_rows": "0:0", "splat_captions": [0, 0, 0, 0],
                "objects_no_captions": True, "refuse": []}
    (y0, y1), = bands
    return {"bands": bands, "texture_rows": f"{y0}:{y1}", "fill_rows": f"{y0}:{y1}", "lingbot_rows": f"{y0 - 2}:{y1 + 2}",
            "splat_captions": [y0, y1, int(x), int(x + w)], "objects_no_captions": False, "refuse": []}


def overlay(clip):
    """D6 over the clip's raster frames: 150 consecutive pairs spread over the whole clip."""
    import cv2
    clip = Path(clip)
    names = [line.split()[1] for line in (clip / "rgb.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    crop = json.loads((clip / "clip.json").read_text())["source"]["crop_xywh"]
    read = lambda i: cv2.imread(str(clip / names[i]), cv2.IMREAD_GRAYSCALE)
    rows, pairs = overlay_rows((read(i), read(i + 1)) for i in np.linspace(0, len(names) - 2, OVERLAY_PAIRS).astype(int))
    bands = overlay_bands(rows, pairs, crop)
    return _result("overlay", overlay_flags(bands, crop), {"pairs": pairs, "stillMovedMatches": len(rows), "cropXywh": crop,
                   "histogram8px": np.histogram(rows, bins=60, range=(0, 480))[0].tolist() if rows else []})


# ---------------------------------------------------------------- D7, D8, D9: LingBot and the dense map
def lingbot_stride(shots):
    """D7. At most 450 LingBot frames per shot."""
    a, b = _value(shots, "shots")["primary"]
    return _result("lingbot_stride", math.ceil((b - a) / LINGBOT_FRAMES), {"frames": b - a, "maxFrames": LINGBOT_FRAMES})


def conf_from_deciles(deciles):
    """The upper edge of the last decile in the leading run whose share within 4% is < 0.90 (none: keep everything)."""
    last = None
    for row in deciles:
        if row["share_within_4pct"] >= LINGBOT_SHARE_4PCT:
            break
        last = row
    return round(last["conf_to"] if last else deciles[0]["conf_from"], 2)


def lingbot_conf(diagnose):
    """D7. The confidence threshold from lingbot_dense_map diagnose's by_conf_decile."""
    deciles = _json(diagnose, "diagnose.json")["by_conf_decile"]
    return _result("lingbot_conf", conf_from_deciles(deciles), {"shareWithin4pct": [round(d["share_within_4pct"], 4) for d in deciles],
                                                                 "confTo": [d["conf_to"] for d in deciles], "rule": LINGBOT_SHARE_4PCT})


def plane_frame(scale):
    up, origin = np.array(scale["up_native"], float), np.array(scale["plane_point_native"], float)
    a = np.cross(up, [1., 0, 0]); a /= np.linalg.norm(a)
    return up, origin, a, np.cross(up, a)


def floor_offset_m(points, mesh_vertices, scale):
    """Median over observed floor cells of the median signed height of the dense points in the cell (metres, + is up).
    Observed floor: fused mesh vertices within the plane tolerance of metric-scale.json; points: those within 0.3 m of the
    plane; cells: 0.25 m squares on the plane. None when no cell has both."""
    m = scale["metres_per_native_unit"]
    up, origin, a, b = plane_frame(scale)
    floor = mesh_vertices[np.abs((mesh_vertices - origin) @ up) <= scale["plane_tolerance_native"]]
    key = lambda p: np.floor(np.c_[(p - origin) @ a, (p - origin) @ b] / (FLOOR_CELL_M / m)).astype(np.int64) @ np.array([1, 1 << 32], np.int64)
    height = (points - origin) @ up
    near = np.abs(height) <= FLOOR_BAND_M / m
    cells, heights = key(points[near]), height[near]
    inside = np.isin(cells, key(floor))
    cells, heights = cells[inside], heights[inside]
    if not len(cells):
        return None, 0
    order = np.argsort(cells, kind="stable")
    cells, heights = cells[order], heights[order]
    starts = np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]])
    per_cell = [np.median(part) for part in np.split(heights, starts[1:])]
    return float(np.median(per_cell) * m), len(per_cell)


def dense_metrics(folder, fused):
    """The display gate's two numbers for one dense map against the report's fused mesh."""
    import open3d as o3d
    import trimesh
    from lingbot_dense_map import read_points_glb
    scale = json.loads((fused / "metric-scale.json").read_text())
    m = scale["metres_per_native_unit"]
    mesh = trimesh.load(fused / "mono-anchored-mesh.ply", process=False)
    xyz = read_points_glb(folder / "dense-points.glb")[0].astype(np.float64)
    fill = np.load(folder / "point-attributes.npz")["fill"]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, np.float32)), o3d.core.Tensor(np.asarray(mesh.faces, np.uint32)))
    dist = scene.compute_distance(o3d.core.Tensor(xyz.astype(np.float32))).numpy()  # as lingbot_dense_map.evaluate (a)
    offset, cells = floor_offset_m(xyz[~fill], np.asarray(mesh.vertices, np.float64), scale)
    return {"within_25cm_all_points": round(float((dist <= .25 / m).mean()), 4), "floor_offset_m": offset, "floor_cells": cells, "points": len(xyz)}


def dense_passes(metrics):
    return metrics["within_25cm_all_points"] >= DENSE_WITHIN_25CM and metrics["floor_offset_m"] is not None and abs(metrics["floor_offset_m"]) <= DENSE_FLOOR_M


def dense_gate(map, fused, out):
    """D8. The un-refined map if >= 45% of its points are within 25 cm of the fused mesh and its floor is within 5 cm;
    otherwise lingbot_icp_refine.py (per-iteration caps) into OUT/icp and the same gate; otherwise no dense points.
    value.dir is relative to OUT (None: the map input itself)."""
    map, fused, out = Path(map), Path(fused), Path(out)
    metrics, steps, use, folder = {"raw": dense_metrics(map, fused)}, None, None, None
    if dense_passes(metrics["raw"]):
        use = "raw"
    else:
        run = subprocess.run([sys.executable, str(SCRIPTS / "lingbot_icp_refine.py"), str(map), str(out / "icp"), str(fused)], capture_output=True, text=True)
        if run.returncode not in (0, 2):
            raise RuntimeError(f"lingbot_icp_refine.py failed ({run.returncode}): {run.stderr[-2000:]}")
        logged = json.loads((out / "icp" / "steps.json").read_text())
        steps = {"iterations": len(logged), "max_step_deg": max(s["step_deg"] for s in logged), "max_step_move_m": max(s["step_move_m"] for s in logged),
                 "refused": run.returncode == 2}
        if run.returncode == 0:
            metrics["icp"] = dense_metrics(out / "icp", fused)
            use, folder = ("icp", "icp") if dense_passes(metrics["icp"]) else (None, None)
    return _result("dense_gate", {"use": use, "dir": folder, "metrics": metrics, "steps": steps},
                   {"map": str(map), "fused": str(fused), "gate": {"minWithin25cm": DENSE_WITHIN_25CM, "maxFloorOffsetM": DENSE_FLOOR_M,
                    "floorBandM": FLOOR_BAND_M, "floorCellM": FLOOR_CELL_M}})


def inferred_floor(floor, gate=None):
    """D9. Import the inferred floor only if its own tests kept it: a model tested against a dense map (dense.validation), on a
    floor plane the lens gate passed (gate: the lens_gate decision, lens.json). The convex outline (no dense map) was never
    tested, and an intrinsics_uncertain plane is no verified floor: both withheld (a wrong floor is worse than none)."""
    record = _json(floor, "inferred-floor.json")
    status = _value(gate, "lens")["scale_status"] if gate else None
    why = (record.get("reason") or "withheld by its own tests" if record["kind"] == "inferred_floor_withheld" else
           "no dense-map validation: the convex outline was never tested against the video" if not (record.get("dense") or {}).get("validation") else
           "the floor plane failed the lens gate (intrinsics_uncertain): no verified plane to extend" if status == "intrinsics_uncertain" else None)
    return _result("inferred_floor", why is None, {"kind": record["kind"], "lensScaleStatus": status, **({"withheld": why} if why else {})})


# ---------------------------------------------------------------- D10, D11, D12: windows, splat cleanup, SAM 2.1 frames
def windows(a, b):
    """Greedy: while more than 400 frames remain, a 300-frame window and 260 on; the last window takes the rest."""
    out, s = [], a
    while b - s > TRACK_CAP:
        out.append([s, s + TRACK_WINDOW])
        s += TRACK_STEP
    return out + [[s, b]]


def track_windows(shots):
    """D10. SAM 3.1 windows of the mapped shot (shot) and of each other shot (others), never across a cut."""
    decided = _value(shots, "shots")
    primary = windows(*decided["primary"])
    return _result("track_windows", {"shot": primary, "stitch": len(primary) > 1, "others": [windows(*s) for s in decided["others"]]},
                   {"cap": TRACK_CAP, "window": TRACK_WINDOW, "step": TRACK_STEP})


def pick_cleanup(results):
    """Fewest Gaussians among the cleanup rules whose held-out PSNR is at most 0.2 dB below 'none'."""
    floor = results["none"]["held_out"]["psnr"] - SPLAT_DB
    return min((r for r in results if results[r]["held_out"]["psnr"] >= floor), key=lambda r: (results[r]["kept"], r))


def splat_pick(clean):
    """D11 over splat_train.py --clean's clean/clean.json."""
    path = Path(clean)
    record = json.loads((path / "clean" / "clean.json" if (path / "clean").is_dir() else _file(path, "clean.json")).read_text())
    results = record["results"]
    return _result("splat_pick", pick_cleanup(results), {"dropDb": {r: round(results["none"]["held_out"]["psnr"] - v["held_out"]["psnr"], 4) for r, v in results.items()},
                                                          "kept": {r: v["kept"] for r, v in results.items()}, "maxDropDb": SPLAT_DB})


def sam2_frames(census, segments):
    """D12. Census keyframes and every 3rd frame of the whole clip (frames outside the shot cost L4 seconds, never views)."""
    total = int(_json(segments, "segments.json")["segments"][-1][1]) + 1
    keyframes = _keyframes(census)
    return _result("sam2_frames", sorted(set(keyframes.tolist()) | set(range(0, total, 3))), {"frames": total, "censusKeyframes": len(keyframes)})


# ---------------------------------------------------------------- D17, D18, D20: generators, floor masks, people
def _manifest(run):
    return _json(run, "manifest.json") if run else None


def plan_generators(sam3d, recgen=None, review=None):
    """Manifests and review record (dicts) -> the generator plan. Review exclusions (recorded by any generator run, or in the
    review record's entitiesExcluded) apply to every generator, the box included."""
    runs = [m for m in (sam3d, recgen) if m]
    accepted = {o["entityId"]: f"accepted by {m['generator'].split(' (')[0]}" for m in runs for o in m["objects"] if o.get("accepted") is True}
    excluded = {}  # an entity another generator accepted was excluded to save a call (Walmart RecGen), not at review
    for manifest in runs:
        excluded |= {r["entityId"]: r["reason"] for r in manifest.get("rejected", []) if r["reason"].startswith("excluded at review") and r["entityId"] not in accepted}
    excluded |= (review or {}).get("entitiesExcluded", {})
    rejects = [o["entityId"] for o in sam3d["objects"] if o.get("accepted") is False and o["entityId"] not in excluded]
    box = {k: v for k, v in sorted((accepted | excluded).items())}
    return {"recgen_entities": rejects, "box_excludes": box, "excluded_all": dict(sorted(excluded.items()))}


def generator_plan(sam3d, recgen=None, review=None):
    """D17. RecGen gets the SAM 3D rejects; the box excludes every learned-accepted and every review-excluded entity."""
    value = plan_generators(_manifest(sam3d), _manifest(recgen), json.loads(Path(review).read_text()) if review else None)
    return _result("generator_plan", value, {"sam3d": str(sam3d), "recgen": str(recgen) if recgen else None, "review": str(review) if review else None})


def spaced_floor_frames(shot):
    a, b = shot
    grid = [f for f in range(a, b) if f % 3 == 0]  # the DA3 view grid, so every floor mask has a depth view
    return sorted({grid[i] for i in np.round(np.linspace(0, len(grid) - 1, FLOOR_MASK_FRAMES)).astype(int)})


def floor_frames(shots, operator=None, clip=None):
    """D18. The operator's list (O1: REVIEW/<site>.floor-frames.json) when it names this clip (source video sha256 and window in
    clip.json), else 12 evenly spaced frames on the every-3rd grid of the mapped shot."""
    evidence = {"source": "rule", "count": FLOOR_MASK_FRAMES}
    if operator:
        given, source = json.loads(Path(operator).read_text()), _json(clip, "clip.json")["source"]
        this = {k: source[k] for k in ("video_sha256", "start_s", "end_s")}
        if given["clip"] == this:
            return _result("floor_frames", [int(f) for f in given["floorFrames"]], {"source": "operator", "clip": this})
        evidence["operatorListFor"] = given["clip"]  # another clip's hand list is never used
    return _result("floor_frames", spaced_floor_frames(_value(shots, "shots")["primary"]), evidence)


def classify_people(shares, labels):
    """shares {entity: [person-mask share per view]}, labels {entity: label} -> (moved, cleared)."""
    moved = sorted(e for e, s in shares.items() if s and sum(x >= PERSON_SHARE for x in s) > len(s) / 2)
    cleared = sorted(e for e in shares if e not in moved and labels[e] in PERSON)
    return moved, cleared


def static_filter(object_map, masks, dynamic_masks, droid, out=None):
    """D20. An entity whose views overlap the person masks by >= 30% in more than half of them moves to the dynamic layer; a
    static entity with a person label loses the label. With out: OUT/object-map.json is the filtered map (surfaces/ and the
    sheets are the input's; stage them beside it)."""
    import functools
    import cv2
    import mono_room
    mono_room.use_clip(Path(droid))
    masks, dynamic_masks = Path(masks), Path(dynamic_masks)
    document = _json(object_map, "object-map.json")
    person = functools.lru_cache(maxsize=64)(lambda index: mono_room.moving_mask(dynamic_masks, index))
    shares = {}
    for entity in document["entities"]:
        row = []
        for observation in entity["observations"]:
            label, index, instance = observation.split(":")
            moving = person(int(index))
            if not moving.any():
                row.append(0.)
                continue
            path = next(masks.glob(f"{label}-*/frame-{int(index):05d}")) / f"instance-{instance}-mask.png"
            mask = mono_room.prepare_image(cv2.imread(str(path), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
            row.append(float((mask & moving).sum() / max(mask.sum(), 1)))
        shares[entity["entityId"]] = row
    moved, cleared = classify_people(shares, {e["entityId"]: e["label"] for e in document["entities"]})
    if out:
        kept = []
        for entity in document["entities"]:
            if entity["entityId"] in cleared:
                entity.update(clearedLabel=entity["label"], label=UNNAMED, labelSource="person label cleared: a static entity (decision static_filter)")
            if entity["entityId"] not in moved:
                kept.append(entity)
        document["movedToDynamic"] = [{"entityId": e, "personShareMedian": float(np.median(shares[e]))} for e in moved]
        document["entities"] = kept
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / "object-map.json").write_text(json.dumps(document, indent=1, allow_nan=False))
    return _result("static_filter", {"moved": moved, "cleared": cleared},
                   {"personShare": PERSON_SHARE, "medianShare": {e: round(float(np.median(s)), 4) for e, s in shares.items() if s and (e in moved or e in cleared or np.median(s) > 0)}})


RULES = {"shots": shots, "other_shot": other_shot, "lens": lens, "voxel": voxel, "overlay": overlay, "lingbot_stride": lingbot_stride,
         "lingbot_conf": lingbot_conf, "dense_gate": dense_gate, "inferred_floor": inferred_floor, "track_windows": track_windows,
         "splat_pick": splat_pick, "sam2_frames": sam2_frames, "generator_plan": generator_plan, "floor_frames": floor_frames,
         "static_filter": static_filter}


def roles(name):
    """(required, optional) roles of a rule; `out` is the stage directory, not a role."""
    params = [p for p in inspect.signature(RULES[name]).parameters.values() if p.name != "out"]
    return [p.name for p in params if p.default is p.empty], [p.name for p in params if p.default is not p.empty]


def decide(name, out, paths):
    """Run one rule and write OUT/NAME.json. Unknown or missing roles are refused before anything runs."""
    required, optional = roles(name)
    unknown, missing = set(paths) - set(required) - set(optional), set(required) - set(paths)
    if unknown or missing:
        raise SystemExit(f"{name}: unknown roles {sorted(unknown)}, missing roles {sorted(missing)}; takes {required} + optional {optional}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = {k: Path(v) for k, v in paths.items()} | ({"out": out} if "out" in inspect.signature(RULES[name]).parameters else {})
    result = RULES[name](**kwargs)
    (out / f"{name}.json").write_text(json.dumps(result, indent=1, allow_nan=False))
    return result


def self_check():
    kf = np.array([0, 14, 226, 300, 400, 500, 600, 700, 800, 850, 890])
    best, counts = pick_shots([[0, 14], [14, 226], [226, 899]], kf)
    assert best == 2 and counts == [1, 1, 9]
    assert pick_shots([[0, 100], [100, 300]], np.array([5, 150]))[0] == 1, "a tie goes to the longer shot"
    assert lens_frames([383, 750])[:3] == [383, 409, 435] and lens_frames([383, 750])[-1] == 749
    assert lens_frames([0, 420]) == [int(round(v)) for v in np.linspace(0, 419, 15)]
    assert abs(fov_deg(627.1843719482421) - 54.06) < .01 and abs(fx_of(fov_deg(610.5529)) - 610.5529) < 1e-9
    assert windows(0, 899) == [[0, 300], [260, 560], [520, 899]] and windows(0, 420) == [[0, 300], [260, 420]] and windows(383, 750) == [[383, 750]]
    assert windows(0, 400) == [[0, 400]] and windows(0, 401) == [[0, 300], [260, 401]]
    deciles = [{"conf_from": 1., "conf_to": 1.1, "share_within_4pct": .7}, {"conf_from": 1.1, "conf_to": 1.737, "share_within_4pct": .85},
               {"conf_from": 1.737, "conf_to": 2., "share_within_4pct": .92}, {"conf_from": 2., "conf_to": 3., "share_within_4pct": .8}]
    assert conf_from_deciles(deciles) == 1.74, "only the leading run counts, a later dip does not"
    assert conf_from_deciles(deciles[2:]) == 1.74 and conf_from_deciles([{"conf_from": 1., "conf_to": 2., "share_within_4pct": .95}]) == 1.
    held = lambda psnr, kept: {"kept": kept, "held_out": {"psnr": psnr}}
    assert pick_cleanup({"none": held(30, 100), "a": held(29.85, 60), "b": held(29.7, 40)}) == "a"
    crop = [160, 0, 960, 720]
    rows = [445.] * 700 + [450.] * 300 + list(np.linspace(0, 400, 100))
    assert overlay_bands(rows, 150, crop) == [[648, 696]], overlay_bands(rows, 150, crop)
    assert overlay_bands(rows[:500], 150, crop) == [], "fewer than 5 still-moved matches per pair: no band"
    assert overlay_bands([100.] * 500 + [445.] * 500, 150, crop) == [[128, 168], [648, 688]]
    assert overlay_flags([[648, 696]], crop) | {} == {"bands": [[648, 696]], "texture_rows": "648:696", "fill_rows": "648:696", "lingbot_rows": "646:698",
                                                     "splat_captions": [648, 696, 160, 1120], "objects_no_captions": False, "refuse": []}
    assert overlay_flags([], crop)["lingbot_rows"] == "0:0" and overlay_flags([[1, 2], [3, 4]], crop)["refuse"] == list(CAPTION_LAYERS)
    sam = {"generator": "SAM 3D Objects (licence)", "objects": [{"entityId": "a", "accepted": True}, {"entityId": "b", "accepted": False}, {"entityId": "c", "accepted": False}],
           "rejected": [{"entityId": "x", "reason": "excluded at review: on a moving cart"}, {"entityId": "y", "reason": "no usable view"}]}
    plan = plan_generators(sam, {"generator": "RecGen", "objects": [{"entityId": "b", "accepted": True}], "rejected": [{"entityId": "a", "reason": "excluded at review: accepted by SAM 3D"}]},
                           {"entitiesExcluded": {"c": "a reflection"}})
    assert plan["recgen_entities"] == ["b"] and list(plan["box_excludes"]) == ["a", "b", "c", "x"] and list(plan["excluded_all"]) == ["c", "x"]
    assert spaced_floor_frames([0, 420]) == sorted({list(range(0, 420, 3))[i] for i in np.round(np.linspace(0, 139, 12)).astype(int)})
    assert all(f % 3 == 0 and 383 <= f < 750 for f in spaced_floor_frames([383, 750])) and len(spaced_floor_frames([383, 750])) == 12
    moved, cleared = classify_people({"w": [.9, .8, .1], "m": [0., .01], "d": [.5, .1]}, {"w": "man", "m": "man", "d": "desk"})
    assert moved == ["w"] and cleared == ["m"], (moved, cleared)
    up = np.array([0., -1, 0])
    scale = {"metres_per_native_unit": 2., "up_native": up.tolist(), "plane_point_native": [0., 0, 0], "plane_tolerance_native": .01}
    grid = np.stack(np.meshgrid(np.linspace(-2, 2, 41), np.linspace(-2, 2, 41)), -1).reshape(-1, 2)
    floor = np.c_[grid[:, 0], np.zeros(len(grid)), grid[:, 1]]
    offset, cells = floor_offset_m(floor + [0, .04, 0], floor, scale)  # native +0.04 along +y is 0.08 m DOWN
    assert abs(offset + .08) < 1e-9 and cells > 50, offset
    for name in RULES:
        required, _ = roles(name)
        assert required, name
    print("decide self-check passed: shots, lens frames and fx, windows, deciles, splat pick, overlay bands and flags, generator plan, floor frames, people, floor offset")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", nargs="?", choices=sorted(RULES))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("roles", nargs="*", metavar="ROLE=PATH")
    args = parser.parse_intermixed_args(argv)
    if args.self_check:
        return self_check()
    if not args.name or not args.out:
        parser.error("NAME and --out are required")
    paths = dict(item.split("=", 1) for item in args.roles)
    print(json.dumps(decide(args.name, args.out, paths)["value"]))


if __name__ == "__main__":
    main()
