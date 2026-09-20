"""One entity per physical object from per-keyframe masks, using the platform's own cross-view associator.

Masks (discover_video_keyframes output, one folder per prompt) are lifted with the posed depth of the same
view and grouped by ehs_spatial.platform.spatial.associate_observations. That function demands a full clique
for a new group but lets a confirmed group gain any uniquely supported view, so it is run to a fixed point:
groups found so far are passed back as confirmed_groups. No thresholds are changed and no labels are merged.

  python scripts/build_video_object_map.py --droid-run RUN --depth-run FUSED_RUN --masks MASK_ROOT --output NEW_DIR
  python scripts/build_video_object_map.py --self-check
"""
import argparse
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


def build(args):
    import mono_room
    mono_room.use_clip(args.droid_run)
    rows = {r["source_index"]: r for r in mono_room.load(args.droid_run, args.support, args.depth_run)}
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    frames, images, by_label, dropped = {}, {}, {}, {"no_depth_view": 0, "duplicate_in_frame": 0, "too_small": 0}
    for folder in sorted(args.masks.glob("*/frame-*")):
        index, label = int(folder.name.split("-")[1]), folder.parent.name.rsplit("-", 1)[0]
        row = rows.get(index)
        if row is None:
            dropped["no_depth_view"] += len(list(folder.glob("instance-*-mask.png")))
            continue
        if index not in frames:
            bgr, k = mono_room.prepare_image(cv2.imread(str(mono_room.DATASET / manifest["frames"][index]["relative_path"])), mono_room.CALIBRATION, 2)
            depth = np.where(mono_room.unreliable(row["mono"], None, None, .03), 0, row["scale"] * row["mono"])[::STEP, ::STEP]
            v, u = np.indices((480, 640))[:, ::STEP, ::STEP]
            local = np.stack([(u - k[2]) / k[0] * depth, (v - k[3]) / k[1] * depth, depth], -1)
            frames[str(index)] = FrameGeometry(str(index), "droid_final_native_world", manifest["frames"][index]["sha256"],
                                               local @ row["c2w"][:3, :3].T + row["c2w"][:3, 3], depth > 0,
                                               np.array([[k[0] / STEP, 0, k[2] / STEP], [0, k[1] / STEP, k[3] / STEP], [0, 0, 1.]]), row["c2w"])
            images[index] = bgr
        found = []
        for path in sorted(folder.glob("instance-*-mask.png")):
            mask = mono_room.prepare_image(cv2.imread(str(path), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
            mask = mask[::STEP, ::STEP] & frames[str(index)].valid
            if mask.sum() < 32:  # AssociationConfig.min_support: such a mask can never be compared
                dropped["too_small"] += 1
            else:
                found.append((f"{label}:{index}:{path.stem.split('-')[1]}", mask))
        kept = distinct(found)
        dropped["duplicate_in_frame"] += len(found) - len(kept)
        by_label.setdefault(label, []).extend(MaskObservation(name, str(index), mask) for name, mask in kept)
    entities, report = [], {}
    for label, observations in sorted(by_label.items()):
        result, rounds = fixed_point(observations, frames)
        lookup = {o.id: o for o in observations}
        report[label] = {"observations": len(observations), "groups": len(result["groups"]), "rounds": rounds,
                         "ambiguous_links": len(result["ambiguous"]), "config": result["config"]}
        for group in result["groups"]:
            points = np.concatenate([frames[lookup[o].image_id].points[lookup[o].mask] for o in group])
            entity = {"entityId": f"{label}-{len(entities):03d}", "label": label, "observations": group,
                      "sourceFrames": sorted({int(lookup[o].image_id) for o in group}), "supportPoints": len(points),
                      "centroidNative": np.median(points, 0).tolist(),
                      "boundsNative": [np.percentile(points, 2, 0).tolist(), np.percentile(points, 98, 0).tolist()]}
            entities.append(entity)
            tiles = []
            for o in group[:12]:  # contact sheet: the evidence a person needs to spot a wrong merge
                index, mask = int(lookup[o].image_id), cv2.resize(lookup[o].mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
                tile = images[index].copy()
                tile[mask == 0] = tile[mask == 0] // 4
                cv2.putText(tile, f"{entity['entityId']} f{index}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
                tiles.append(cv2.resize(tile, (320, 240)))
            tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
            cv2.imwrite(str(args.output / f"{entity['entityId']}.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    summary = {"schema": "phase2-video-object-map-v1", "depth_run": str(args.depth_run), "masks": str(args.masks), "views": len(frames),
               "coordinate_frame": "droid_final_native_world", "associator": "ehs_spatial.platform.spatial.associate_observations, fixed point over confirmed_groups",
               "dropped_masks": dropped, "per_label": report, "entities": entities,
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
    print(f"object map check passed: two boxes stay two entities across three views ({rounds} rounds); in-frame duplicate dropped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--depth-run", type=Path, help="mono_room output holding mono/*.npz")
    parser.add_argument("--support", type=Path)
    parser.add_argument("--masks", type=Path, help="folder of PROMPT-PART/frame-XXXXX/instance-N-mask.png")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else build(a)
