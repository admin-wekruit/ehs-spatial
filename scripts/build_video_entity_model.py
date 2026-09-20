"""Generated object models for video object-map entities, through the pinned photo RecGen transport.

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


def evidence(args, entity, rows, manifest):
    import mono_room
    observation = max(entity["observationBoxes"], key=lambda o: np.prod(np.subtract(entity["observationBoxes"][o][2:], entity["observationBoxes"][o][:2])))
    label, index, instance = observation.split(":")
    row, record = rows[int(index)], manifest["frames"][int(index)]
    mask_path = next(args.masks.glob(f"{label}-*/frame-{int(index):05d}/instance-{instance}-mask.png"))
    bgr, k = mono_room.prepare_image(cv2.imread(str(mono_room.DATASET / record["relative_path"])), mono_room.CALIBRATION, 2)
    mask = mono_room.prepare_image(cv2.imread(str(mask_path), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
    depth = np.where(mono_room.unreliable(row["mono"], None, None, .03), 0, row["scale"] * row["mono"]).astype(np.float32)
    K = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]])
    crop = source_grid_crop(np.ascontiguousarray(bgr[..., ::-1]), mask, depth, K, np.eye(3))  # image, mask and depth share one raster
    payload = {"entityId": entity["entityId"], "anchorObservationId": observation, "seed": 42, "views": [{
        "observationId": observation, "observationRevision": 1, "imageId": f"{record['sha256']}:{index}", "imageSha256": record["sha256"],
        "maskSha256": digest(mask_path), "geometrySolutionSha256": digest(args.depth_run / "mono" / f"{int(index):05d}.npz"),
        "coordinateFrameId": "droid_final_native_world", "cameraToWorld": row["c2w"], **crop}]}
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
    for name in args.entities:
        folder = args.output / name
        if (folder / "validation.json").exists():
            print(json.dumps({"entity": name, "status": "already validated, not requested again"}))
            continue
        payload, observation, geometry = evidence(args, entities[name], rows, manifest)
        request = RecGenRequest.from_payload(payload)
        folder.mkdir(parents=True, exist_ok=True)
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
