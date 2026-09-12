"""Read every recorded 2D instance, independently of metric-geometry gates.

No provider calls, label-based identity merging, or inferred physical measurements.
"""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .providers.map_anything import input_mask_to_canonical, _input_mask_mapping
from .providers.sam3 import decode_coco_rle
from .measurements import measure_observed_points


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _mask_sha(mask: np.ndarray) -> str:
    return hashlib.sha256(str(mask.shape).encode() + np.packbits(mask).tobytes()).hexdigest()


def _local(run: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        # Persisted repo-relative evidence names the run, not its mount's parents.
        if path.parts[:2] == ("runs", run.name):
            path = Path(*path.parts[2:])
        path = run / path
    resolved = path.resolve()
    if resolved.is_relative_to(run) and resolved.is_file():
        return resolved
    raise ValueError("Evidence file is missing or outside this run")


def load_candidate_mask(run: Path, candidate: dict) -> np.ndarray:
    """Decode the bound original source, rejecting changed files or dimensions."""
    run = Path(run).resolve()
    ref = candidate["mask"]["ref"]
    path = _local(run, ref["path"])
    if _sha(path) != ref["sha256"]:
        raise ValueError("Source mask file changed")
    if ref["encoding"] == "png":
        pixels = np.asarray(Image.open(path))
        if pixels.ndim != 2 or not np.isin(pixels, [0, 1, 255]).all():
            raise ValueError("Mask image is not a two-dimensional binary image")
        mask = pixels.astype(bool)
    else:
        value = json.loads(path.read_text())
        for key in ref["pointer"]:
            value = value[key]
        if isinstance(value, dict):
            value = json.dumps(value)
        height, width = ref["shape_hw"]
        mask = decode_coco_rle(value, height=height, width=width).astype(bool)
    if mask.shape != tuple(ref["shape_hw"]):
        raise ValueError("Mask resolution disagrees with its recorded source grid")
    if candidate["mask"].get("sha256") and _mask_sha(mask) != candidate["mask"]["sha256"]:
        raise ValueError("Decoded mask content changed")
    return mask


def build_object_evidence(run: Path) -> dict:
    """Pure read-only product registry. Generation readiness is checked on export."""
    run = Path(run).resolve()
    hashes, documents, frames, geometry = {}, {}, [], {}

    def document(relative, default):
        path = run / relative
        if not path.resolve().is_relative_to(run):
            raise ValueError("Document is outside this run")
        if not path.is_file():
            return default
        if relative not in documents:
            documents[relative] = json.loads(path.read_text())
            hashes[relative] = _sha(path)
        return documents[relative]

    for index, path in enumerate(sorted((run / "input").glob("image_*")), 1):
        frame_id = f"frame_{index:04d}"
        with Image.open(path) as image:
            width, height = image.size
        relative = path.relative_to(run).as_posix()
        hashes[relative] = _sha(path)
        frame = {"frame_id": frame_id, "image_path": relative, "width": width, "height": height,
                 "source_sha256": hashes[relative], "canonical_image_path": None,
                 "canonical_width": None, "canonical_height": None, "input_to_canonical_pixel_centres": None}
        directory = run / "geometry/frames" / frame_id
        canonical = directory / "canonical.png"
        if canonical.is_file():
            cw, ch = Image.open(canonical).size
            rel = canonical.relative_to(run).as_posix()
            hashes[rel] = _sha(canonical)
            frame.update(canonical_image_path=rel, canonical_width=cw, canonical_height=ch, canonical_sha256=hashes[rel])
            provider = run / "geometry/provider" / f"{frame_id}.json"
            if provider.exists():
                try:
                    stat = provider.stat()
                    source_shape, canonical_shape, rect, transform = _input_mask_mapping(provider.resolve(), stat.st_mtime_ns, stat.st_size)
                    if source_shape != (height, width) or canonical_shape != (ch, cw):
                        raise ValueError("Provider input/canonical dimensions disagree with the saved images")
                    frame["input_to_canonical_pixel_centres"] = transform["input_to_canonical_pixel_centres"]
                    frame["input_mask_transform"] = transform
                    frame["content_rect_xyxy"] = list(rect)
                    hashes[provider.relative_to(run).as_posix()] = _sha(provider)
                except ValueError as error:
                    frame["mapping_error"] = str(error).replace(str(run), "<run>")
        frames.append(frame)
        try:
            arrays = {}
            for name in ["pts3d", "valid_mask", "conf", "intrinsics", "camera_to_world"]:
                file = directory / f"{name}.npy"
                arrays[name] = np.load(file, allow_pickle=False)
                hashes[file.relative_to(run).as_posix()] = _sha(file)
            points, valid, conf, K, C = (arrays[n] for n in ["pts3d", "valid_mask", "conf", "intrinsics", "camera_to_world"])
            shape = (frame["canonical_height"], frame["canonical_width"])
            if (points.shape != (*shape, 3) or valid.shape != shape or conf.shape != shape or
                    K.shape != (3, 3) or C.shape != (4, 4) or not np.isfinite(K).all() or not np.isfinite(C).all() or
                    not np.allclose(C[3], [0, 0, 0, 1]) or not np.allclose(C[:3, :3].T @ C[:3, :3], np.eye(3), atol=1e-4)):
                raise ValueError("Geometry grid or native camera matrix is invalid")
            inverse = np.linalg.inv(C)
            depth = points @ inverse[2, :3] + inverse[2, 3]
            keep = valid.astype(bool) & np.isfinite(conf) & (conf >= .1) & np.isfinite(points).all(-1)
            keep &= (np.linalg.norm(points, axis=-1) > 1e-6) & np.isfinite(depth) & (depth > 0)
            rect = frame.get("content_rect_xyxy")
            if rect is not None:
                content = np.zeros(shape, bool)
                x0, y0, x1, y1 = rect
                content[y0:y1, x0:x1] = True
                keep &= content
            geometry[frame_id] = (shape, keep, depth, points, None)
        except (OSError, ValueError, KeyError) as error:
            geometry[frame_id] = (None, None, None, None, str(error).replace(str(run), "<run>"))
    frame_map = {f["frame_id"]: f for f in frames}
    first_frame = frames[0]["frame_id"] if frames else None
    scene_record = document("scene.json", {})
    calibration = None
    for relative in ["calibration.json", "capture.json", "manifest.json"]:
        path = run / relative
        if not path.is_file():
            continue
        try:
            record = json.loads(path.read_text())
            value = record.get("calibration", record.get("capture", record))
            if value.get("camera_height_provided") is True:
                document(relative, {})
                calibration = {"provided": True, "camera_height_m": value.get("camera_height_m"),
                               "source_sha256": hashes[relative]}
                break
        except (ValueError, AttributeError):
            continue
    inventory = document("inventory/inventory.json", {})
    measured = inventory.get("objects", [])
    unresolved = document("inventory/unresolved.json", [])
    aliases, candidates = {}, {}

    def add(kind, relative, pointer, frame_id, label, resolution, *, encoding="rle", metadata=None, invs=()):
        metadata = metadata or {}
        source = {"type": kind, "path": relative, "pointer": pointer, "label": label, **metadata}
        ref = None
        error = None
        mask = None
        support_points = np.empty((0, 3))
        canonical_pixels = 0
        try:
            path = _local(run, relative)
            relative = path.relative_to(run).as_posix()
            source["path"] = relative
            source["sha256"] = hashes.setdefault(relative, _sha(path))
            frame = frame_map[frame_id]
            grid_shape = [frame["height"], frame["width"]] if resolution == "original" else [frame["canonical_height"], frame["canonical_width"]]
            ref = {"path": relative, "sha256": source["sha256"], "pointer": pointer,
                   "encoding": encoding, "shape_hw": grid_shape}
            if metadata.get("source_image_sha256") and metadata["source_image_sha256"] != frame["source_sha256"]:
                raise ValueError("Instance is bound to a different source photograph")
            if metadata.get("mask_sha256") and metadata["mask_sha256"] != source["sha256"]:
                raise ValueError("Instance source-mask hash changed")
            mask = load_candidate_mask(run, {"mask": {"ref": ref}})
            if not mask.any():
                raise ValueError("Source mask is empty")
        except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
            error = ((metadata.get("reason") or "No instance mask recorded for this unresolved claim")
                     if kind == "unresolved" else (str(exc) or type(exc).__name__))
            error = error.replace(str(run), "<run>")
        if mask is not None and not error:
            signature = _mask_sha(mask)
            identity = f"{frame_id}:{frame_map[frame_id]['source_sha256']}:{resolution}:{signature}"
            y, x = np.nonzero(mask)
            mask_record = {"status": "available", "resolution": resolution, "shape_hw": list(mask.shape),
                           "bbox": [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1],
                           "sha256": signature, "ref": ref}
            shape, keep, depth, native_points, reason = geometry[frame_id]
            count = 0
            if reason is None:
                try:
                    canonical_mask = input_mask_to_canonical(mask, run, frame_id, shape)
                    canonical_pixels = int(canonical_mask.sum())
                    support_points = native_points[canonical_mask & keep]
                    count = int((canonical_mask & keep).sum())
                    if count < 8:
                        reason = "Fewer than 8 independent valid canonical depth pixels"
                    elif resolution == "original" and frame_map[frame_id]["input_to_canonical_pixel_centres"] is None:
                        reason = "Original-image crop has no verified input-to-canonical affine"
                    elif float(depth[keep].max()) > 30:
                        reason = "Native depth exceeds the generator unit guard"
                except (OSError, ValueError) as exc:
                    reason = str(exc).replace(str(run), "<run>")
            state = "supported" if reason is None else "insufficient"
            generation = {"status": "pending_validation" if reason is None else "blocked",
                          "input_mode": "original_rgb_crop" if resolution == "original" else "canonical",
                          "reason": "Exact payload and erosion preflight required" if reason is None else reason}
        else:
            identity = json.dumps([frame_id, source["path"], pointer], sort_keys=True)
            mask_record = {"status": "unavailable", "resolution": resolution, "shape_hw": None,
                           "bbox": metadata.get("box"), "sha256": None, "ref": ref, "reason": error}
            state, count, reason = "unavailable", 0, error
            generation = {"status": "blocked", "input_mode": None, "reason": error}
        candidate_id = "object_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        if candidate_id not in candidates:
            candidates[candidate_id] = {"id": candidate_id, "kind": "instance", "frame_id": frame_id,
                "label": label, "labels": [], "source_refs": [], "inventory_indices": [], "mask": mask_record,
                "geometry": {"status": state, "valid_points": count, "reason": reason}, "generation": generation,
                "identity": {"scope": "same_frame_exact_mask", "cross_view_verified": False},
                "measurement_rejections": []}
            measurement = measure_observed_points(support_points, scene_record, mask_pixels=canonical_pixels,
                calibration=calibration, source={"frame_id":frame_id,"candidate_id":candidate_id,
                    "scene_sha256":hashes.get("scene.json"),
                    "image_sha256":frame_map.get(frame_id,{}).get("source_sha256"),
                    "mask_sha256":mask_record.get("sha256"),
                    "pointmap_sha256":hashes.get(f"geometry/frames/{frame_id}/pts3d.npy")})
            if error:
                measurement["reason"] = error
            candidates[candidate_id]["measurements"] = measurement
        candidate = candidates[candidate_id]
        if label not in candidate["labels"]:
            candidate["labels"].append(label)
        if source not in candidate["source_refs"]:
            candidate["source_refs"].append(source)
        candidate["inventory_indices"] = sorted(set(candidate["inventory_indices"]) | set(invs))
        if candidate["inventory_indices"]:
            candidate["geometry"]["status"] = "measured"
        aliases[(frame_id, relative, json.dumps(pointer))] = candidate_id
        return candidate_id

    # Observations already carry canonical masks. No enlargement is called native evidence.
    for i, observation in enumerate(document("observations.json", [])):
        frame_id = observation["frame_id"]
        if observation.get("mask_path"):
            path, pointer, encoding = observation["mask_path"], [], "png"
        else:
            path, pointer, encoding = "observations.json", [i, "mask_reference"], "rle"
        add("observation", path, pointer, frame_id, observation["label"], "canonical", encoding=encoding,
            metadata={k: observation[k] for k in ["observation_id", "instance_id", "score", "source_prompt"] if k in observation})

    def sam_path(frame, label):
        return f"inventory/sam/{frame}__{re.sub(r'[^a-z0-9]+', '_', label).strip('_')}.json"

    bindings = {}
    for index, entry in enumerate(measured):
        if entry.get("refine_slug"):
            continue
        for member in entry.get("merged_instances") or [entry.get("instance")]:
            if member is not None:
                bindings.setdefault((entry["frame"], sam_path(entry["frame"], entry["label"]), int(member)), []).append(entry.get("inv", index))
    labels = list(dict.fromkeys(inventory.get("phrases", []) + [o["label"] for o in measured] + [u["phrase"] for u in unresolved]))
    labels_by_slug = {re.sub(r"[^a-z0-9]+", "_", label).strip("_"): label for label in labels}
    for path in sorted((run / "inventory/sam").glob("*.json")):
        if "__" not in path.stem:
            continue
        frame_id, slug = path.stem.split("__", 1)
        relative = path.relative_to(run).as_posix()
        data = document(relative, {})
        rles = data.get("rle") or []
        indices = [None] if isinstance(rles, str) else list(range(len(rles)))
        for index in indices:
            instance = 0 if index is None else index
            add("text-sam", relative, ["rle"] if index is None else ["rle", index], frame_id,
                labels_by_slug.get(slug, slug.replace("_", " ")), "canonical",
                metadata={"instance": instance, "score": (data.get("scores") or [])[instance] if instance < len(data.get("scores") or []) else None},
                invs=bindings.get((frame_id, relative, instance), []))
    for (frame_id, relative, instance), invs in bindings.items():
        if (frame_id, relative, json.dumps(["rle", instance])) not in aliases and (frame_id, relative, json.dumps(["rle"])) not in aliases:
            label = next(e["label"] for i, e in enumerate(measured) if e.get("inv", i) == invs[0])
            add("inventory", relative, ["rle", instance], frame_id, label, "canonical", invs=invs)

    detections = document("detection/detections.json", {})
    for index, item in enumerate(detections.get("detections", [])):
        frame_id = item.get("frame_id", item.get("frame", detections.get("frame_id", first_frame)))
        add("detection", "detection/detections.json", ["detections", index, "rle"], frame_id, item["label"], "original",
            metadata={k: item[k] for k in ["item_id", "box", "sam_score", "category", "zh", "source_image_sha256"] if k in item})

    refinement_bindings = {}
    for index, entry in enumerate(measured):
        if entry.get("refine_slug"):
            refinement_bindings.setdefault((entry["frame"], entry["refine_slug"]), []).append(entry.get("inv", index))
    refinement_rows = list(document("refinements.json", []))
    known_slugs = {r.get("refine_slug") or r["label"].replace(" ", "_") + "_" + "_".join(map(str, r["box"])) for r in refinement_rows}
    for entry in measured:
        if entry.get("refine_slug") and entry["refine_slug"] not in known_slugs:
            refinement_rows.append({"label": entry["label"], "refine_slug": entry["refine_slug"], "frame_id": entry["frame"]})
    for item in refinement_rows:
        slug = item.get("refine_slug") or item["label"].replace(" ", "_") + "_" + "_".join(map(str, item["box"]))
        frame_id = item.get("frame_id", item.get("frame", first_frame))
        relative = item.get("mask_path", f"refinements/{slug}.json")
        try:
            response = document(_local(run, relative).relative_to(run).as_posix(), {})
        except ValueError:
            response = {}
        rles = response.get("rle") or []
        index = int(np.argmax(response.get("scores") or [1] * max(1, len(rles))))
        pointer = ["rle"] if isinstance(rles, str) else ["rle", index]
        candidate_id = add("refinement", relative, pointer, frame_id, item["label"], "original",
            metadata={k: item[k] for k in ["box", "source", "sam_score", "evidence_id", "source_image_sha256", "mask_sha256", "geometry_status", "geometry_reason"] if k in item} | {"refine_slug": slug},
            invs=refinement_bindings.get((frame_id, slug), []))
        aliases[(frame_id, slug, "refinement")] = candidate_id

    for index, item in enumerate(unresolved):
        frame_id = item.get("frame_id", item.get("frame"))
        candidate_id = aliases.get((frame_id, item.get("refine_slug"), "refinement"))
        if candidate_id is None and item.get("instance") is not None:
            relative = item.get("mask_path", sam_path(frame_id, item["phrase"]))
            candidate_id = aliases.get((frame_id, relative, json.dumps(["rle", item["instance"]])))
            if candidate_id is None and item["instance"] == 0:
                candidate_id = aliases.get((frame_id, relative, json.dumps(["rle"])))
        rejection = {k: item[k] for k in ["stage", "reason", "source", "box"] if k in item}
        if candidate_id:
            candidates[candidate_id]["measurement_rejections"].append(rejection)
        else:
            candidate_id = add("unresolved", "inventory/unresolved.json", [index, "unavailable_mask"], frame_id,
                               item["phrase"], "unknown", metadata=rejection)
            candidates[candidate_id]["kind"] = "unresolved_claim"
    result = {"version": 1, "run_id": run.name, "frames": frames,
              "candidates": sorted(candidates.values(), key=lambda c: (c["frame_id"] or "", c["id"])),
              "source_sha256": dict(sorted(hashes.items())),
              "measurement_reference": {"scene": {k:scene_record.get(k) for k in
                  ["floor_plane","scale_source","scale_factor","scale_confidence","warnings"]},
                  "scene_sha256": hashes.get("scene.json"), "calibration": calibration},
              "notes": ["Identity merges exact masks only within one source frame and pixel grid.",
                        "Measured status refers to linked saved inventory; no policy verdict is inferred from a generated asset.",
                        "Pending validation is not generation readiness; the exporter must validate the exact model payload."]}
    assert len({c["id"] for c in result["candidates"]}) == len(result["candidates"])
    assert all(_sha(run / path) == value for path, value in hashes.items()), "Source changed during evidence collection"
    json.dumps(result, allow_nan=False)
    return result


def write_object_evidence(run: Path) -> Path:
    """Atomically persist a freshly built registry after the product chain changes."""
    run = Path(run)
    value = build_object_evidence(run)
    path = run / "object-evidence.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=run, prefix=".object-evidence-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path
