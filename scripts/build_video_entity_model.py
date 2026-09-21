"""Generated object models for video object-map entities, through the pinned photo RecGen transport.

The view is the one that holds the entity's whole 3D extent unoccluded; the mask is SAM 2.1's answer to the box of the
entity's points in that view (one GPU call for all entities), so a part-level segment or a wrong name cannot shrink or
misplace it.

Same request, provider, adapter and source-view check as build_lingbot_object_model.py; only the evidence differs:
the entity's largest mask, the rectified keyframe, its DROID camera and the posed depth of that view. One bounded,
journaled provider call per entity; an entity with a saved validation is never requested again.
Unseen sides are the generator's estimate. RecGen weights are non-commercial.

  python scripts/build_video_entity_model.py --droid-run RUN --depth-run FUSED --object-map DIR --masks ROOT \
      --output NEW_DIR --function-id fu-... --entities monitor-036 [--invoke]
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "modal_apps"), str(ROOT / "scripts")]
from ehs_spatial.platform.recgen import RecGenRequest, adapt_output, source_grid_crop  # noqa: E402
from build_lingbot_object_model import evaluate, save  # noqa: E402
from reconstruct_room_rgb import digest  # noqa: E402


def framed_view(entity, rows, object_map):
    """The view that holds the whole entity: all eight corners of its 3D extent inside the frame, most of it unoccluded, as large as possible.

    Judged on the entity's 3D extent, not on a mask: segment-everything masks are often parts, and the view with the
    largest mask is usually a close-up that cuts the object off. Returns (source index, pixels of the entity's points there).
    """
    low, high = np.array(entity["boundsNative"])
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    data = np.load(object_map / entity["cells"]["file"])
    points = (data["cells"] + .5) * float(data["cell"])
    best = (0, None, None)
    for index, row in rows.items():
        k, c2w = row["k"] * 2, row["c2w"]
        project = lambda p: ((p - c2w[:3, 3]) @ c2w[:3, :3])
        box = project(corners)
        if box[:, 2].min() <= .05:
            continue
        u, v = box[:, 0] / box[:, 2] * k[0] + k[2], box[:, 1] / box[:, 2] * k[1] + k[3]
        if u.min() < 26 or v.min() < 19 or u.max() > 614 or v.max() > 461:  # 4% of 640x480 clear of every border
            continue
        local = project(points)
        x, y = np.clip((local[:, 0] / local[:, 2] * k[0] + k[2]).astype(int), 0, 639), np.clip((local[:, 1] / local[:, 2] * k[1] + k[3]).astype(int), 0, 479)
        depth = row["scale"] * row["mono"][y, x]
        seen = (depth > 0) & (np.abs(depth / local[:, 2] - 1) < .06)
        score = seen.mean() * (u.max() - u.min()) * (v.max() - v.min())
        if seen.mean() >= .6 and score > best[0]:
            best = (score, index, np.column_stack([x, y])[seen])
    return best[1], best[2]


def evidence(args, entity, rows, manifest, folder):
    """The request built from the entity's framed view and its box-prompted whole mask (anchor.json, written by main's first pass)."""
    import mono_room
    anchor = json.loads((folder / "anchor.json").read_text())
    index, mask_path = anchor["sourceFrame"], Path(anchor["mask"])
    observation = f"{entity['label']}:{index}:prompted"
    row, record = rows[int(index)], manifest["frames"][int(index)]
    bgr, k = mono_room.prepare_image(cv2.imread(str(mono_room.DATASET / record["relative_path"])), mono_room.CALIBRATION, 2)
    mask = mono_room.prepare_image(cv2.imread(str(mask_path), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0  # SAM leaves pinholes on textured surfaces; a generator reads them as holes in the object
    if mono_room.METRIC_CAMERAS:  # the device refocuses per frame; the raster only knows the clip's median K
        k = row["k"] * 2
    depth = np.where(mono_room.unreliable(row["mono"], None, None, .03), 0, row["scale"] * row["mono"]).astype(np.float32)
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]])
    crop = source_grid_crop(np.ascontiguousarray(bgr[..., ::-1]), mask, depth, K, np.eye(3))  # image, mask and depth share one raster
    payload = {"entityId": entity["entityId"], "anchorObservationId": observation, "seed": 42, "views": [{
        "observationId": observation, "observationRevision": 1, "imageId": f"{record['sha256']}:{index}", "imageSha256": record["sha256"],
        "maskSha256": digest(mask_path), "geometrySolutionSha256": digest(args.depth_run / "mono" / f"{int(index):05d}.npz"),
        "coordinateFrameId": "arkit_device_world" if mono_room.METRIC_CAMERAS else "droid_final_native_world", "cameraToWorld": row["c2w"], **crop}]}
    return payload, observation, (depth, np.full(depth.shape, 2., np.float32), depth > 0, mask, K, row["c2w"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("droid-run", "depth-run", "object-map", "masks", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--entities", nargs="+", required=True)
    parser.add_argument("--function-id", required=True)
    parser.add_argument("--invoke", action="store_true", help="submit the provider calls; without it only the inputs are prepared")
    args = parser.parse_args()
    import mono_room
    import trimesh
    mono_room.use_clip(args.droid_run)
    rows = {r["source_index"]: r for r in mono_room.load(args.droid_run, args.support, args.depth_run)}
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    entities = {e["entityId"]: e for e in json.loads((args.object_map / "object-map.json").read_text())["entities"]}
    # First pass, geometry only: the frame that holds each entity whole, and the box of its points there. One SAM2 call then gives
    # the whole-object mask for every box. No name is involved: a VLM's name for a class-agnostic segment is too often wrong to segment by.
    wanted, skipped = {}, {}
    for name in args.entities:
        folder = args.output / name
        folder.mkdir(parents=True, exist_ok=True)
        if (folder / "validation.json").exists() or (folder / "anchor.json").exists():
            continue
        index, pixels = framed_view(entities[name], rows, args.object_map)
        if index is None:
            skipped[name] = "no view holds the whole entity unoccluded"
            continue
        image = cv2.imread(str(mono_room.DATASET / manifest["frames"][index]["relative_path"]))
        scale = image.shape[1] / 640
        low, high = pixels.min(0), pixels.max(0)
        pad = .1 * (high - low) + 4
        wanted[name] = (image, ((np.concatenate([low - pad, high + pad])) * scale).tolist(), index, pixels)
    if wanted:
        import sam2_everything
        for name, (mask, score) in sam2_everything.box_masks({n: (w[0], w[1]) for n, w in wanted.items()}).items():
            image, box, index, pixels = wanted[name]
            folder, small = args.output / name, cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST) > 0
            share = float(small[pixels[:, 1], pixels[:, 0]].mean())
            ys, xs = np.nonzero(mask)
            cut = not len(xs) or xs.min() < 4 or ys.min() < 4 or xs.max() > mask.shape[1] - 5 or ys.max() > mask.shape[0] - 5
            if share < .5 or cut:
                skipped[name] = f"the box-prompted mask in frame {index} " + ("runs into the image border" if cut else f"holds only {share:.0%} of the entity's points")
                continue
            cv2.imwrite(str(folder / "whole-mask.png"), mask.astype(np.uint8) * 255)
            cv2.imwrite(str(folder / f"frame-{index}.png"), image)
            save(folder / "anchor.json", {"entity": name, "sourceFrame": int(index), "mask": str((folder / "whole-mask.png").resolve()), "image": str((folder / f"frame-{index}.png").resolve()),
                                          "box_source_pixels": box, "sam_score": score, "share_of_entity_points_inside_mask": share, "prompt": "box of the entity's projected 3D points (SAM 2.1)"})
    for name, reason in skipped.items():
        save(args.output / name / "skipped.json", {"entity": name, "label": entities[name]["label"], "reason": reason})
        print(json.dumps({"entity": name, "status": "skipped", "reason": reason}))
    for name in args.entities:
        folder = args.output / name
        if (folder / "validation.json").exists():
            print(json.dumps({"entity": name, "status": "already validated, not requested again"}))
            continue
        if not (folder / "anchor.json").exists():
            continue
        payload, observation, geometry = evidence(args, entities[name], rows, manifest, folder)
        request = RecGenRequest.from_payload(payload)
        (folder / "input.npz").write_bytes(request.to_npz())
        save(folder / "input-manifest.json", {"entity": name, "observation": observation, "input_sha256": hashlib.sha256(request.to_npz()).hexdigest(),
             "crop_shape": list(payload["views"][0]["depth"].shape), "object_map": str(args.object_map), "depth_run": str(args.depth_run)})
        if not args.invoke:
            print(json.dumps({"entity": name, "status": "prepared", "observation": observation}))
            continue
        from ehs_spatial.platform.recgen_transport import invoke
        os.environ["PANOPTES_RECGEN_JOURNAL"] = str(folder / "journal")
        from ehs_spatial.platform.reconstruction import ProviderResponseError
        try:
            response = invoke(payload, {"modalApp": "lucida-private-assets", "modalFunction": "generate_object",
                                        "modalFunctionId": args.function_id, "modalVolume": "panoptes-lucida-weights"})
        except ProviderResponseError as error:  # journaled as outcome unknown and never resubmitted; the other entities still get their one call
            save(folder / "failure.json", {"entity": name, "observation": observation, "error": str(error), "crop_shape": list(payload["views"][0]["depth"].shape)})
            print(json.dumps({"entity": name, "status": "provider_outcome_unknown_not_resubmitted", "crop_shape": list(payload["views"][0]["depth"].shape)}))
            continue
        save(folder / "provider-record.json", {k: v for k, v in response.items() if k not in ("vertices", "faces", "colors", "officialPosedVertices")})
        if response.get("providerError"):
            print(json.dumps({"entity": name, "status": "provider_error", "error": response["providerError"]}))
            continue
        result = adapt_output(request, response)
        mesh = trimesh.Trimesh(result["vertices"], result["faces"], vertex_colors=result["colors"], process=False)
        mesh.apply_transform(result["proposedObjectToNative"])
        mesh.export(folder / "model.glb")
        exported = trimesh.load(folder / "model.glb", force="mesh", process=False)  # judge the serialized deliverable, not the adapter arrays
        report = evaluate(exported.vertices, exported.faces, *geometry)
        save(folder / "validation.json", {**report, "entity": name, "observation": observation, "mesh_sha256": digest(folder / "model.glb"),
             "provenance": result["provenance"], "object_to_native": result["proposedObjectToNative"].tolist(), "licence": "RecGen weights: non-commercial research"})
        print(json.dumps({"entity": name, **{k: report[k] for k in ("supported_pixels", "silhouette_iou", "relative_depth_median", "relative_depth_p95", "accepted_source_consistency")}}))


if __name__ == "__main__":
    main()
