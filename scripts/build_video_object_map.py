"""One entity per physical object from per-keyframe masks.

--method clique  : the platform's own cross-view associator, run to a fixed point (tried first; under-merges video).
--method overlap : a persistent 3D accumulator after ConceptGraphs (slam/mapping.py, the README recipe): each entity keeps
                   world points; a new mask joins the same-label entity that most of its points fall next to. The text
                   label SAM3 was prompted with replaces their CLIP term; one entity takes at most one mask per frame.


Masks (discover_video_keyframes output, one folder per prompt) are lifted with the posed depth of the same
view and grouped by ehs_spatial.platform.spatial.associate_observations. That function demands a full clique
for a new group but lets a confirmed group gain any uniquely supported view, so it is run to a fixed point:
groups found so far are passed back as confirmed_groups. No thresholds are changed and no labels are merged.

  python scripts/build_video_object_map.py --droid-run RUN --depth-run FUSED_RUN --masks MASK_ROOT --output NEW_DIR
  python scripts/build_video_object_map.py --self-check
"""
import argparse
import functools
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "modal_apps")]
from ehs_spatial.platform.spatial import FrameGeometry, MaskObservation, associate_observations  # noqa: E402

STEP = 2  # association grid: 320x240 of the 640x480 depth raster


def distinct(masks, overlap=.5):
    """Keep the larger of two same-label masks in one frame that mostly cover each other; duplicates would compete."""
    kept = []
    for mask in sorted(masks, key=lambda item: -item[1].sum()):
        if all((mask[1] & other[1]).sum() / min(mask[1].sum(), other[1].sum()) < overlap for other in kept):
            kept.append(mask)
    return kept


def fixed_point(observations, frames):
    groups, rounds = [], 0
    while True:
        result = associate_observations(observations, frames, confirmed_groups=[g for g in groups if len(g) > 1])
        rounds += 1
        if result["groups"] == groups:
            return result, rounds
        groups = result["groups"]


MATCH, MERGE, CONFIRMED = .5, .7, 3  # ConceptGraphs: overlap needed beside a CLIP term of .6-.9 under sim_threshold 1.2; merge_overlap_thresh; obj_min_detections
PART = .25  # a class-agnostic group is (part of) a prompted entity only if it covers at least this share of it
AGNOSTIC, SURFACES_PER_AGNOSTIC = "object", 4  # label of segment-everything masks; how many of their views keep a full-resolution surface
STUFF = ("floor", "wall", "ceiling")  # extents without instances: no two views need to overlap, so one entity per label (ConceptGraphs' background classes)
NEAR = .03  # "next to" = the platform associator's relative_depth_tolerance, times the mask's own median range


def near_fraction(points, cloud, radius):
    from scipy.spatial import cKDTree
    if (points.min(0) - radius > cloud.max(0)).any() or (points.max(0) + radius < cloud.min(0)).any():
        return 0.  # boxes apart, nothing can be near: spares the tree for the many far pairs of a class-agnostic map
    return float((cKDTree(cloud).query(points, distance_upper_bound=radius)[0] < np.inf).mean())


def pack(cells):
    """Integer cell triples as one comparable value each."""
    return np.ascontiguousarray(cells, np.int64).view([("", np.int64)] * 3).ravel()


def consensus(point_sets, cell):
    """Per view, the points at least one other view of the same entity also puts there (same or adjacent cell).

    A mask that spills past the object lands on a different piece of background in every view, so spill does not
    survive the vote, while the object itself does. Returns the keep-masks and the agreed cells.
    """
    offsets = np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), -1).reshape(-1, 3)
    cells = [np.floor(points / cell).astype(np.int64) for points in point_sets]
    reach = [np.unique((np.unique(c, axis=0)[:, None] + offsets).reshape(-1, 3), axis=0) for c in cells]  # what each view vouches for
    voted, counts = np.unique(np.concatenate(reach), axis=0, return_counts=True)
    agreed = voted[counts >= 2]
    return [np.isin(pack(c), pack(agreed)) for c in cells], agreed


def thin(points, cell):
    """One point per cell once a cloud is large: 'is something near' is answered by a cell as well as by its thousand points."""
    return points if len(points) < 20000 else points[np.unique(np.floor(points / cell).astype(np.int64), axis=0, return_index=True)[1]]


def accumulate(observations, frames):
    """observations: MaskObservation in frame order. Returns groups of observation ids."""
    entities = []  # {"points", "ids", "frames", "radius"}
    for image_id in sorted({o.image_id for o in observations}, key=int):
        frame, taken = frames[image_id], set()
        current = [o for o in observations if o.image_id == image_id]
        clouds = {o.id: frame.points[o.mask] for o in current}
        radius = {o.id: NEAR * float(np.median(np.linalg.norm(clouds[o.id] - frame.camera_to_world[:3, 3], axis=1))) for o in current}
        lows, highs = (np.array([e[key] for e in entities]).reshape(-1, 3) for key in ("low", "high"))  # boxes first: thousands of entities, a handful nearby
        reach = {o.id: np.flatnonzero(((clouds[o.id].min(0) - radius[o.id] <= highs) & (clouds[o.id].max(0) + radius[o.id] >= lows)).all(1)) for o in current}
        scores = sorted(((near_fraction(clouds[o.id], entities[n]["points"], radius[o.id]), o.id, int(n)) for o in current for n in reach[o.id]
                         if image_id not in entities[n]["frames"]), reverse=True)
        for score, oid, n in scores:  # best pairs first; one mask per entity per frame and one entity per mask
            if score < MATCH or oid in taken or image_id in entities[n]["frames"]:
                continue
            entities[n]["points"] = thin(np.concatenate([entities[n]["points"], clouds[oid]]), entities[n]["radius"] / 2)
            entities[n].update(low=np.minimum(entities[n]["low"], clouds[oid].min(0)), high=np.maximum(entities[n]["high"], clouds[oid].max(0)))
            entities[n]["ids"].append(oid); entities[n]["frames"].add(image_id); taken.add(oid)
        entities += [{"points": clouds[o.id], "ids": [o.id], "frames": {image_id}, "radius": radius[o.id], "low": clouds[o.id].min(0), "high": clouds[o.id].max(0)}
                     for o in current if o.id not in taken]
    merged = True
    while merged:  # the same object entered twice from views that did not overlap at first; passes until nothing merges
        merged, a = False, 0
        while a < len(entities):
            lows, highs = (np.array([e[key] for e in entities]) for key in ("low", "high"))
            reach = max(e["radius"] for e in entities)
            for b in sorted((int(b) for b in np.flatnonzero(((entities[a]["low"] - reach <= highs) & (entities[a]["high"] + reach >= lows)).all(1)) if b > a), reverse=True):
                drop, keep = sorted((a, b), key=lambda n: len(entities[n]["points"]))
                small, large = entities[drop], entities[keep]
                if near_fraction(small["points"], large["points"], small["radius"]) > MERGE:
                    large["points"] = thin(np.concatenate([large["points"], small["points"]]), large["radius"] / 2)
                    large.update(low=np.minimum(large["low"], small["low"]), high=np.maximum(large["high"], small["high"]))
                    large["ids"] += small["ids"]; large["frames"] |= small["frames"]
                    entities[a] = large
                    del entities[b]; merged = True  # by position, from the back so earlier positions stay valid
            a += 1
    return sorted(sorted(e["ids"]) for e in entities)


def absorb(grouped, lookup, frames):
    """Class-agnostic groups that are a prompted object join it; the rest stay entities of their own.

    Segment-everything also finds the monitor the prompt found. Names are attributes of one object, not two objects:
    a class-agnostic group whose points mostly lie on a prompted entity adds its views to that entity (one mask per frame).
    """
    def cloud(ids):
        return np.concatenate([frames[lookup[o].image_id].points[lookup[o].mask] for o in ids])
    named = [(label, ids, thin(cloud(ids), .01)) for label, ids in grouped if label != AGNOSTIC and label not in STUFF]
    kept, joined = [g for g in grouped if g[0] != AGNOSTIC], 0
    for label, ids in grouped:
        if label != AGNOSTIC:
            continue
        points = cloud(ids)
        first = lookup[ids[0]]
        radius = NEAR * float(np.median(np.linalg.norm(frames[first.image_id].points[first.mask] - frames[first.image_id].camera_to_world[:3, 3], axis=1)))
        score, target = max(((near_fraction(points, other, radius), n) for n, (_, _, other) in enumerate(named)), default=(0, None))
        # a sheet of paper lies within reach of the desk too: the same object also covers a fair part of the entity, something resting on it does not
        if score >= MATCH and near_fraction(named[target][2], points, radius) >= PART:
            have = {lookup[o].image_id for o in named[target][1]}
            named[target][1].extend(o for o in ids if lookup[o].image_id not in have)
            joined += 1
        else:
            kept.append((label, ids))
    return kept, joined


def build(args):
    import mono_room
    mono_room.use_clip(args.droid_run)
    rows = {r["source_index"]: r for r in mono_room.load(args.droid_run, args.support, args.depth_run)}
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    frames, mask_paths, by_label, dropped = {}, {}, {}, {"no_depth_view": 0, "duplicate_in_frame": 0, "too_small": 0, "on_moving_entity": 0}
    for row in rows.values():
        row["conf"] = None  # not used here; hundreds of views have to fit in memory

    @functools.lru_cache(maxsize=6)
    def view(index):
        """Rectified image, reliable depth and K of one view, rebuilt on demand instead of held for every view."""
        bgr, k = mono_room.prepare_image(cv2.imread(str(mono_room.DATASET / manifest["frames"][index]["relative_path"])), mono_room.CALIBRATION, 2)
        depth = np.where(mono_room.unreliable(rows[index]["mono"], None, None, .03), 0, rows[index]["scale"] * rows[index]["mono"]).astype(np.float32)
        return bgr, depth, np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]])

    def full_mask(name):
        raster = mono_room.prepare_image(cv2.imread(str(mask_paths[name]), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
        return raster & (view(int(name.split(":")[1]))[1] > 0)

    for folder in sorted(args.masks.glob("*/frame-*")):
        index, label = int(folder.name.split("-")[1]), folder.parent.name.rsplit("-", 1)[0]
        row = rows.get(index)
        if row is None:
            dropped["no_depth_view"] += len(list(folder.glob("instance-*-mask.png")))
            continue
        if str(index) not in frames:
            K = view(index)[2]
            depth = view(index)[1][::STEP, ::STEP].astype(np.float64)
            v, u = np.indices((480, 640))[:, ::STEP, ::STEP]
            local = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
            frames[str(index)] = FrameGeometry(str(index), "droid_final_native_world", manifest["frames"][index]["sha256"],
                                               (local @ row["c2w"][:3, :3].T + row["c2w"][:3, 3]).astype(np.float32), depth > 0,
                                               K / [[STEP], [STEP], [1]], row["c2w"])
        found, moving = [], mono_room.moving_mask(args.dynamic_masks, index)  # people walking through are not part of the static map
        for path in sorted(folder.glob("instance-*-mask.png")):
            mask = mono_room.prepare_image(cv2.imread(str(path), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
            if (mask & moving).sum() >= .5 * max(mask.sum(), 1):
                dropped["on_moving_entity"] += 1
                continue
            name = f"{label}:{index}:{path.stem.split('-')[1]}"
            mask_paths[name] = path
            mask = mask[::STEP, ::STEP] & frames[str(index)].valid
            if mask.sum() < 32:  # AssociationConfig.min_support: such a mask can never be compared
                dropped["too_small"] += 1
            else:
                found.append((name, mask))
        kept = distinct(found)
        dropped["duplicate_in_frame"] += len(found) - len(kept)
        by_label.setdefault(label, []).extend(MaskObservation(name, str(index), mask) for name, mask in kept)
    entities, report, plan = [], {}, None
    if args.floor:  # plan axes on the fitted floor: footprints and heights are measured against it, still in native units
        floor = json.loads(args.floor.read_text())
        up, origin = np.array(floor["up_native"]), np.array(floor["plane_point_native"])
        axis_a = np.cross(up, [1., 0, 0]); axis_a /= np.linalg.norm(axis_a)
        plan = {"origin": origin, "up": up, "a": axis_a, "b": np.cross(up, axis_a)}
    lookup, grouped = {o.id: o for observations in by_label.values() for o in observations}, []
    for label, observations in sorted(by_label.items()):
        if label in STUFF:
            groups = [[o.id for o in sorted(observations, key=lambda o: int(o.image_id))]]
            report[label] = {"observations": len(observations), "groups": 1, "rule": "one entity per background label"}
        elif args.method == "clique":
            result, rounds = fixed_point(observations, frames)
            groups = result["groups"]
            report[label] = {"observations": len(observations), "groups": len(groups), "rounds": rounds,
                             "ambiguous_links": len(result["ambiguous"]), "config": result["config"]}
        else:
            groups = accumulate(sorted(observations, key=lambda o: int(o.image_id)), frames)
            report[label] = {"observations": len(observations), "groups": len(groups), "confirmed": sum(len(g) >= CONFIRMED for g in groups),
                             "config": {"match_overlap": MATCH, "merge_overlap": MERGE, "near_relative": NEAR, "confirmed_min_observations": CONFIRMED}}
        grouped += [(label, group) for group in groups]
    grouped, joined = absorb(grouped, lookup, frames)
    report["class_agnostic_groups_joined_to_prompted_entities"] = joined
    for label, group in grouped:
        group.sort(key=lambda o: (int(lookup[o].image_id), o))
        sets = [frames[lookup[o].image_id].points[lookup[o].mask].astype(np.float64)[::4 if label in STUFF else 1] for o in group]
        points, agreed, cell, share = np.concatenate(sets), None, None, None
        if len(group) >= CONFIRMED:  # a confirmed entity is measured on what its views agree on, not on every mask pixel
            cameras = np.stack([frames[lookup[o].image_id].camera_to_world[:3, 3] for o in group])
            # "next to" scales with viewing range: to the entity's middle, or for a background extent each mask's own depth
            cell = NEAR * float(np.median([np.median(np.linalg.norm(p - c, axis=1)) for p, c in zip(sets, cameras)]) if label in STUFF else
                                np.median(np.linalg.norm(cameras - np.median(points, 0), axis=1)))
            if label not in STUFF:  # background patches need not overlap, so there is nothing to vote on
                keep, cells = consensus(sets, cell)
                share = float(np.concatenate(keep).mean())
                if share >= .3:  # views of opposite sides share too little to vote: keep everything and say so
                    points, agreed = points[np.concatenate(keep)], cells
        entity = {"entityId": f"{label}-{len(entities):03d}", "label": label, "observations": group,
                  "sourceFrames": sorted({int(lookup[o].image_id) for o in group}), "supportPoints": len(points),
                  "multiViewAgreedShare": share, "measuredOn": "points at least two views agree on" if agreed is not None else "all mask points",
                  "centroidNative": np.median(points, 0).tolist(),
                  "boundsNative": [np.percentile(points, 2, 0).tolist(), np.percentile(points, 98, 0).tolist()]}
        if plan:
            from shapely.geometry import MultiPoint
            relative = points - plan["origin"]
            hull = MultiPoint(np.stack([relative @ plan["a"], relative @ plan["b"]], 1)[::max(1, len(points) // 20000)]).convex_hull
            cameras = np.stack([frames[lookup[o].image_id].camera_to_world[:3, 3] for o in group])
            entity.update(footprintPlanNative=[list(xy) for xy in hull.exterior.coords[:-1]], heightNative=float(np.percentile(relative @ plan["up"], 98)),
                          baseNative=float(np.percentile(relative @ plan["up"], 2)),
                          rangeNative=float(np.median(np.linalg.norm(cameras - np.median(points, 0), axis=1))))
        entity["observationBoxes"] = {}
        for o in group:
            ys, xs = np.where(lookup[o].mask)
            entity["observationBoxes"][o] = [int(xs.min() * STEP), int(ys.min() * STEP), int(xs.max() * STEP + STEP), int(ys.max() * STEP + STEP)]
        if label == AGNOSTIC and len(group) < CONFIRMED:  # thousands of one- or two-view fragments: listed, never shown, so no surface or sheet is built
            entity["surfaces"] = []
            entities.append(entity)
            continue
        from build_video_object_models import observed_surface
        entity["surfaces"] = []  # one visible side per observation, as the photo pipeline keeps one observed surface per photo; nothing behind it is made up
        # every prompted view keeps its surface; of the many segment-everything views only the largest few (disk)
        agnostic = sorted((o for o in group if o.startswith(AGNOSTIC + ":")), key=lambda o: -int(lookup[o].mask.sum()))[:SURFACES_PER_AGNOSTIC]
        for o in [o for o in group if not o.startswith(AGNOSTIC + ":") or o in agnostic]:
            frame, index = frames[lookup[o].image_id], int(lookup[o].image_id)
            bgr, depth, K = view(index)
            mask = full_mask(o)
            # association runs on the STEP grid; the surface a person looks at keeps every source pixel
            vertices, faces, colors, _, _ = observed_surface(bgr[..., ::-1], depth, mask, K, frame.camera_to_world,
                                                             max_edge_m=.04 * float(np.median(depth[mask])), depth_range=(0, np.inf))
            if len(faces) and agreed is not None:  # the shown surface is cut to the agreed part too
                inside = np.isin(pack(np.floor(vertices / cell).astype(np.int64)), pack(agreed))
                faces = faces[inside[faces].all(1)]
                used, inverse = np.unique(faces, return_inverse=True)
                vertices, colors, faces = vertices[used], colors[used], inverse.reshape(-1, 3)
            if len(faces):
                (args.output / "surfaces").mkdir(exist_ok=True)
                name = f"surfaces/{entity['entityId']}--{o.replace(':', '-')}.npz"
                np.savez_compressed(args.output / name, vertices=vertices, faces=faces, colors=colors)
                entity["surfaces"].append({"file": name, "observation": o, "triangles": len(faces)})
        if cell is not None:  # where the entity is, as cells: lets a consumer cut the entity out of the fused multi-view mesh
            (args.output / "surfaces").mkdir(exist_ok=True)
            np.savez_compressed(args.output / "surfaces" / f"{entity['entityId']}--cells.npz", cells=np.unique(np.floor(points / cell).astype(np.int64), axis=0), cell=cell)
            entity["cells"] = {"file": f"surfaces/{entity['entityId']}--cells.npz", "cellNative": cell}
        entities.append(entity)
        tiles = []
        for o in group[:12]:  # contact sheet: the evidence a person needs to spot a wrong merge
            index, mask = int(lookup[o].image_id), cv2.resize(lookup[o].mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
            tile = view(index)[0].copy()
            tile[mask == 0] = tile[mask == 0] // 2
            cv2.drawContours(tile, cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 255, 255), 2)
            cv2.putText(tile, f"{entity['entityId']} f{index}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
            tiles.append(cv2.resize(tile, (320, 240)))
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
        cv2.imwrite(str(args.output / f"{entity['entityId']}.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    summary = {"schema": "phase2-video-object-map-v1", "depth_run": str(args.depth_run), "masks": str(args.masks), "views": len(frames),
               "coordinate_frame": "droid_final_native_world",
               "associator": ("ehs_spatial.platform.spatial.associate_observations, fixed point over confirmed_groups" if args.method == "clique" else
                              "3D point-overlap accumulator after ConceptGraphs; label gate; one mask per entity per frame; same-label merge"),
               "dropped_masks": dropped, "per_label": report, "entities": entities,
               "plan": None if plan is None else {"floor": str(args.floor), "origin_native": plan["origin"].tolist(), "up": plan["up"].tolist(),
                                                  "axis_a": plan["a"].tolist(), "axis_b": plan["b"].tolist(),
                                                  "hull": "convex hull of the entity's lifted mask points projected on the floor: visible sides only, never completed"},
               "limitations": ["Labels are never merged: one object prompted as both desk and cabinet yields two entities.",
                               "An entity seen in fewer than 3 views stays a candidate; it is listed, not confirmed.",
                               "Identity is geometric only; no appearance feature is used."]}
    (args.output / "object-map.json").write_text(json.dumps(summary, indent=1, allow_nan=False))
    multi = [e for e in entities if len(e["sourceFrames"]) >= 3]
    print(json.dumps({"views": len(frames), "dropped": dropped, "per_label": report, "entities": len(entities), "entities_seen_in_3plus_views": len(multi)}, indent=1))


def self_check():
    """Two boxes seen from three cameras: the fixed point must join each box across all views and never join the two."""
    K = np.array([[200., 0, 160], [0, 200., 120], [0, 0, 1]])
    v, u = np.indices((240, 320))
    frames, observations = {}, []
    for index, shift in enumerate([-.4, 0., .4]):
        c2w = np.eye(4); c2w[0, 3] = shift
        depth = np.full((240, 320), 4.)
        masks = {}
        for name, centre in (("left", -.5), ("right", .5)):
            x = (u - K[0, 2]) / K[0, 0] * 2. + shift  # world x on the z=2 plane
            inside = (np.abs(x - centre) < .25) & (np.abs(v - 120) < 50)
            depth[inside], masks[name] = 2., inside
        local = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
        frames[str(index)] = FrameGeometry(str(index), "w", "0" * 64, local + c2w[:3, 3], depth > 0, K, c2w)
        observations += [MaskObservation(f"{name}:{index}", str(index), mask) for name, mask in masks.items()]
    result, rounds = fixed_point(observations, frames)
    assert result["groups"] == [[f"left:{i}" for i in range(3)], [f"right:{i}" for i in range(3)]], result["groups"]
    big, small = np.zeros((4, 4), bool), np.zeros((4, 4), bool)
    big[:3], small[:2] = True, True
    assert [name for name, _ in distinct([("small", small), ("big", big)])] == ["big"]
    ordered = sorted(observations, key=lambda o: int(o.image_id))
    assert accumulate(ordered, frames) == result["groups"], "accumulator must agree on the clean case"
    half = MaskObservation("left:9", "0", frames["0"].valid & observations[0].mask & (u < u[observations[0].mask].mean()))  # a partial view of the left box
    frames["9"] = frames["0"]; half = MaskObservation("left:9", "9", half.mask)
    groups = accumulate(ordered + [half], frames)
    assert ["left:0", "left:1", "left:2", "left:9"] in groups and len(groups) == 2, groups
    rng = np.random.default_rng(0)
    thing = rng.uniform(0, .3, (400, 3))  # three views of one object, each with its own piece of background in the mask
    spill = [rng.uniform(0, .3, (150, 3)) + shift for shift in ([2, 0, 0], [0, 2, 0], [0, 0, 2])]
    keep, agreed = consensus([np.vstack([thing, extra]) for extra in spill], .05)
    assert all(k[:400].all() and not k[400:].any() for k in keep), "views must keep the shared object and drop their own spill"
    grouped, joined = absorb([("left", [f"left:{i}" for i in range(3)]), (AGNOSTIC, ["right:0"])], {o.id: o for o in observations}, frames)
    assert joined == 0 and len(grouped) == 2, "a class-agnostic group elsewhere stays its own entity"
    grouped, joined = absorb([("left", ["left:0", "left:1"]), (AGNOSTIC, ["left:2"])], {o.id: o for o in observations}, frames)
    assert joined == 1 and grouped == [("left", ["left:0", "left:1", "left:2"])], "a class-agnostic view of a prompted object joins it"
    box = observations[0].mask
    halves = [MaskObservation("left:a", "20", box & (u < u[box].mean())), MaskObservation("left:b", "21", box & (u >= u[box].mean())),
              MaskObservation("left:c", "22", box)]  # two disjoint halves enter as two entities; the full view must leave one
    frames.update({"20": frames["0"], "21": frames["0"], "22": frames["0"]})
    assert accumulate(halves, frames) == [["left:a", "left:b", "left:c"]], accumulate(halves, frames)
    print(f"object map check passed: halves merge once the whole is seen; accumulator joins a partial view to the right entity; two boxes stay two entities across three views ({rounds} rounds); in-frame duplicate dropped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--depth-run", type=Path, help="mono_room output holding mono/*.npz")
    parser.add_argument("--support", type=Path)
    parser.add_argument("--masks", type=Path, help="folder of PROMPT-PART/frame-XXXXX/instance-N-mask.png")
    parser.add_argument("--floor", type=Path, help="metric-scale.json of mono_room metric: adds floor-plan footprints and heights (native units)")
    parser.add_argument("--dynamic-masks", type=Path, help="directory of SOURCEINDEX-*.png masks of moving entities; masks lying on them are left out of the static map")
    parser.add_argument("--method", choices=["clique", "overlap"], default="overlap")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else build(a)
