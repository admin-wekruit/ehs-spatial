"""HOV-inspired observation embeddings and native-point cross-view association.

CPU preparation/analysis; --encode runs one existing route_jev Encoder on the
externally selected GPU. Existing object identity/classes are evaluation proxies.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
ENCODERS = {"pe-core-l": "timm/PE-Core-L-14-336", "siglip2-so400m": "google/siglip2-so400m-patch16-384"}
VARIANTS = ("single_view", "multiview", "spatial_semantic")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def config_checked(config):
    for group in ("classes", "queries"):
        rows = config[group]
        if not isinstance(rows, list) or (group == "classes" and not rows):
            raise ValueError(f"{group} must be a list; classes cannot be empty")
        if any(not isinstance(r.get(k), str) or not r[k].strip() for r in rows for k in ("id", "label", "phrase")):
            raise ValueError(f"Invalid {group} candidate")
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate {group} IDs")
    for key in ("overlapNative", "overlapThreshold", "semanticThreshold"):
        if not np.isfinite(config[key]) or not 0 < config[key] <= 1:
            raise ValueError(f"Invalid {key}")
    if not isinstance(config["referenceClasses"], dict):
        raise ValueError("referenceClasses must be an evaluation mapping")
    classes = {r["id"] for r in config["classes"]}
    if any(v not in classes for v in config["referenceClasses"].values()):
        raise ValueError("Unknown reference class")
    return config


def normalize(values):
    values = np.asarray(values, np.float32)
    lengths = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.isfinite(values).all() or np.any(lengths <= 1e-12):
        raise ValueError("Embeddings must be finite and nonzero")
    return values / lengths


def polygon_mask(polygons, shape):
    import cv2
    if not polygons:
        raise ValueError("Observation has no polygons")
    contours = []
    for poly in polygons:
        xy = np.asarray(poly, float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3 or not np.isfinite(xy).all():
            raise ValueError("Invalid canonical polygon")
        if np.any(xy < 0) or np.any(xy[:, 0] > shape[1]) or np.any(xy[:, 1] > shape[0]):
            raise ValueError("Polygon exceeds canonical image grid")
        contours.append(np.rint(xy).astype(np.int32))
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, contours, 1)
    if not mask.any():
        raise ValueError("Observation polygon has no pixels")
    return mask.astype(bool)


def observed_points(pointmap, valid, mask, transform, cap=1024):
    """Only finite, declared-valid pixels support geometry; empty is unavailable."""
    pointmap, valid, mask, transform = map(np.asarray, (pointmap, valid, mask, transform))
    if pointmap.shape != (*mask.shape, 3) or valid.shape != mask.shape or mask.ndim != 2:
        raise ValueError("Pointmap, validity and canonical mask shapes disagree")
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("Invalid scene transform")
    if not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5):
        raise ValueError("Scene transform must preserve native distances")
    if not np.isin(valid, [0, 1]).all() or not np.isin(mask, [0, 1]).all():
        raise ValueError("Validity and mask must be binary")
    supported = mask.astype(bool) & valid.astype(bool)
    bad = supported & ~np.isfinite(pointmap).all(axis=-1)
    if bad.any():
        raise ValueError("Declared-valid observation points contain nonfinite values")
    points = pointmap[supported]
    count = len(points)
    if count > cap:
        # ponytail: deterministic raster-order sampling caps this four-photo experiment;
        # larger captures should replace it with an explicit spatial sampling protocol.
        points = points[np.linspace(0, count - 1, cap, dtype=int)]
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32), count


def prepare(root, out, config):
    import cv2
    from scripts.route_jev import tight_crop
    from scripts.workcell_photo_oneshot import _array, _frame

    started = time.perf_counter()
    catalog, report = read_json(root / "objects.json"), read_json(root / "scene-report.json")
    objects = catalog["objects"]
    if not objects or len({o["id"] for o in objects}) != len(objects):
        raise ValueError("Object IDs must be nonempty and unique")
    if set(config["referenceClasses"]) - {o["id"] for o in objects}:
        raise ValueError("Reference mapping names an absent entity")
    photos = sorted({int(obs["photo"]) for obj in objects for obs in obj["observations"]})
    out.mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(exist_ok=True)
    transform = np.asarray(report["sceneTransformNative"], float)
    document = report["revision"].get("document", {})
    scene_observations = {obs["id"]: obs for obs in document.get("observations", [])}
    scene_entities = {entity["id"]: entity for entity in document.get("entities", [])}
    frames, photo_rows, sources = {}, [], []
    for name in ("objects.json", "scene-report.json"):
        sources.append({"file": name, "sha256": sha256(root / name)})
    for photo in photos:
        image_name, frame_name = f"photo-{photo}.png", f"frame_{photo:04d}.json.gz"
        bgr = cv2.imread(str(root / image_name))
        frame = _frame(root, photo)
        points, valid = _array(frame["pts3d"]), _array(frame["non_ambiguous_mask"])
        if bgr is None or bgr.shape != points.shape or tuple(frame["image"]["shape"]) != bgr.shape:
            raise ValueError(f"Photo {photo}: canonical image/pointmap grid mismatch")
        frames[photo] = (bgr, points, valid)
        path = f"photo-{photo}.png"
        if not cv2.imwrite(str(out / path), bgr):
            raise OSError(f"Could not write {path}")
        photo_rows.append({"photo": photo, "imagePath": path, "width": bgr.shape[1], "height": bgr.shape[0]})
        sources.extend({"file": name, "sha256": sha256(root / name)} for name in (image_name, frame_name))
    observations, points_by_id = [], {}
    for obj in objects:
        for source_index, obs in enumerate(obj["observations"]):
            photo = int(obs["photo"])
            bgr, points, valid = frames[photo]
            mask = polygon_mask(obs["polygons"], bgr.shape[:2])
            support, supported_count = observed_points(points, valid, mask, transform)
            observation_id = f"obs-{len(observations):04d}"
            crop, crop_mask = tight_crop(bgr, obs["polygons"])
            crop_path, mask_path = f"crops/{observation_id}.png", f"crops/{observation_id}-mask.png"
            if not cv2.imwrite(str(out / crop_path), crop) or not cv2.imwrite(str(out / mask_path), crop_mask):
                raise OSError(f"Could not write {observation_id}")
            points_by_id[observation_id] = support
            scene_refs = [scene_observations[ref] for ref in scene_entities.get(obj["id"], {}).get("observationRefs", [])
                          if ref in scene_observations and scene_observations[ref]["imageId"] == f"photo-{photo}"]
            exact_refs = [ref for ref in scene_refs if ref.get("originalPixelPolygons") == obs["polygons"]]
            scene_refs = exact_refs or scene_refs
            scene_id = scene_refs[0]["id"] if len(scene_refs) == 1 else None
            observations.append({"observationId": observation_id, "entityId": obj["id"], "sourceObservationIndex": source_index,
                                 "sceneObservationId": scene_id,
                                 "photo": photo, "cropPath": crop_path, "maskPath": mask_path,
                                 "maskPixels": int(mask.sum()), "supportedPixels": supported_count,
                                 "sampledPoints": len(support), "geometryStatus": "available" if len(support) else "unavailable"})
    np.savez_compressed(out / "points.npz", **points_by_id)
    manifest = {"schemaVersion": 1, "sourceRevisionId": report["revision"]["id"], "sources": sources,
                "sceneTransformNative": transform.tolist(), "pointCoordinateSystem": "scene_native_unscaled",
                "pointSource": "MapAnything pts3d under canonical polygons and non_ambiguous_mask; no mesh points",
                "pointCapPerObservation": 1024, "photos": photo_rows, "observations": observations,
                "prepareSeconds": round(time.perf_counter() - started, 4)}
    write_json(out / "manifest.json", manifest)
    return {"observations": len(observations), "objects": len(objects), "photos": len(photos),
            "unavailableGeometry": sum(not r["sampledPoints"] for r in observations), "prepareSeconds": manifest["prepareSeconds"]}


def encode(out, config, key):
    # The Encoder's HF imports must see the existing X13 cache from the outset.
    os.environ["HF_HUB_CACHE"] = "/v/x13/hf"
    import cv2
    from fast_report.visual_encoder import Encoder, TEMPLATES
    import torch

    started = time.perf_counter()
    manifest = read_json(out / "manifest.json")
    observations = manifest["observations"]
    rgb = np.stack([cv2.imread(str(out / r["cropPath"]))[:, :, ::-1] for r in observations])
    masks = np.stack([cv2.imread(str(out / r["maskPath"]), 0) > 127 for r in observations])
    full = [cv2.imread(str(out / r["imagePath"]))[:, :, ::-1] for r in manifest["photos"]]
    timing = {"encoder": key, "modelId": ENCODERS[key], "device": torch.cuda.get_device_name(),
              "observationCount": len(observations), "photoCount": len(full), "textTemplates": list(TEMPLATES)}
    begin = time.perf_counter()
    encoder = Encoder(key)
    torch.cuda.synchronize()
    timing["loadSeconds"] = round(time.perf_counter() - begin, 4)
    arrays = {}
    for name, action in (("plain", lambda: encoder.image(rgb)), ("masked", lambda: encoder.image(rgb, masks)),
                         ("global", lambda: np.concatenate([encoder.image(im[None]) for im in full])),
                         ("classText", lambda: encoder.text([r["phrase"] for r in config["classes"]]))):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        arrays[name] = normalize(action())
        torch.cuda.synchronize()
        timing[name + "Seconds"] = round(time.perf_counter() - begin, 4)
    begin = time.perf_counter()
    arrays["queryText"] = normalize(encoder.text([r["phrase"] for r in config["queries"]])) if config["queries"] else np.empty((0, arrays["classText"].shape[1]), np.float32)
    torch.cuda.synchronize()
    timing["queryTextSeconds"] = round(time.perf_counter() - begin, 4)
    timing["peakAllocatedGiB"] = round(torch.cuda.max_memory_allocated() / 2**30, 4)
    timing["totalSeconds"] = round(time.perf_counter() - started, 4)
    np.savez_compressed(out / f"embeddings-{key}.npz", **arrays,
                        observationIds=np.array([r["observationId"] for r in observations]),
                        photoIds=np.array([r["photo"] for r in manifest["photos"]]),
                        classPhrases=np.array([r["phrase"] for r in config["classes"]]),
                        queryPhrases=np.array([r["phrase"] for r in config["queries"]]),
                        manifestSha256=np.array(sha256(out / "manifest.json")))
    write_json(out / f"timing-{key}.json", timing)
    return timing


def fuse(plain, masked, global_features, photo_indices):
    """HOV-inspired local/global fusion, before any object group pooling."""
    plain, masked, global_features = map(normalize, (plain, masked, global_features))
    if plain.shape != masked.shape or len(photo_indices) != len(plain):
        raise ValueError("Fusion observation shape mismatch")
    photo_indices = np.asarray(photo_indices)
    if np.any(photo_indices < 0) or np.any(photo_indices >= len(global_features)):
        raise ValueError("Invalid fusion photo index")
    local = normalize(.75 * masked + .25 * plain)
    weights = np.zeros(len(plain), np.float32)
    for photo in np.unique(photo_indices):
        indices = np.flatnonzero(photo_indices == photo)
        scores = local[indices] @ global_features[photo]
        exp = np.exp(scores - scores.max())
        weights[indices] = exp / exp.sum()
    fused = normalize(weights[:, None] * global_features[photo_indices] + (1 - weights[:, None]) * local)
    return local, fused, weights


def point_overlaps(points, photos, distance):
    from scipy.spatial import cKDTree
    if not np.isfinite(distance) or distance <= 0 or len(points) != len(photos):
        raise ValueError("Invalid overlap inputs")
    for p in points:
        if np.asarray(p).ndim != 2 or np.asarray(p).shape[1] != 3 or not np.isfinite(p).all():
            raise ValueError("Association points must be finite N by 3 arrays")
    trees = [cKDTree(p) if len(p) else None for p in points]
    overlap = np.full((len(points), len(points)), np.nan, np.float32)
    # ponytail: all cross-photo pairs fit 97 observations; spatial prefiltering
    # should replace this quadratic pair scan for substantially larger captures.
    for i, a in enumerate(points):
        for j in range(i + 1, len(points)):
            if photos[i] == photos[j] or trees[i] is None or trees[j] is None:
                continue
            ab = np.mean(trees[j].query(a, k=1)[0] <= distance)
            ba = np.mean(trees[i].query(points[j], k=1)[0] <= distance)
            overlap[i, j] = overlap[j, i] = max(ab, ba)
    return overlap


def associate(features, photos, overlap, overlap_threshold, semantic_threshold):
    """No entity IDs or labels enter the greedy association algorithm."""
    features = normalize(features)
    photos = np.asarray(photos)
    if overlap.shape != (len(features), len(features)) or len(photos) != len(features):
        raise ValueError("Association shape mismatch")
    cosine = np.clip(features @ features.T, -1, 1)
    gated = np.triu((photos[:, None] != photos[None, :]) & (overlap >= overlap_threshold) & (cosine >= semantic_threshold), 1)
    edges = [(int(i), int(j)) for i, j in zip(*np.where(gated))]
    edges.sort(key=lambda p: (-float(overlap[p] * cosine[p]), p[0], p[1]))
    parent, members = list(range(len(features))), {i: [i] for i in range(len(features))}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    diagnostics = []
    for i, j in edges:
        a, b = find(i), find(j)
        state = "already_connected"
        if a != b:
            if set(photos[members[a]]) & set(photos[members[b]]):
                state = "rejected_duplicate_photo"
            else:
                parent[b] = a
                members[a].extend(members.pop(b))
                state = "merged"
        diagnostics.append({"left": i, "right": j, "overlap": float(overlap[i, j]), "cosine": float(cosine[i, j]),
                            "score": float(overlap[i, j] * cosine[i, j]), "decision": state})
    return sorted([sorted(group) for group in members.values()], key=lambda group: group[0]), diagnostics


def nearest_matches(features, photos, entity_ids, overlap=None, threshold=None):
    """One nearest cross-photo observation per source, including catalog singletons."""
    cosine, photos = normalize(features) @ normalize(features).T, np.asarray(photos)
    cross = photos[:, None] != photos[None, :]
    eligible = int(cross.any(axis=1).sum())
    allowed = cross if overlap is None else cross & (overlap >= threshold)
    matches = []
    for i in range(len(features)):
        candidates = np.flatnonzero(allowed[i])
        if len(candidates):
            j = int(candidates[np.argmax(cosine[i, candidates])])
            matches.append({"left": i, "right": j, "cosine": float(cosine[i, j]),
                            "referenceSameEntity": entity_ids[i] == entity_ids[j]})
    correct = sum(r["referenceSameEntity"] for r in matches)
    return {"eligible": eligible, "matched": len(matches), "referenceCorrect": correct,
            "precision": correct / len(matches) if matches else None, "coverage": len(matches) / eligible if eligible else None}, matches


def ranked(feature, text_features, candidates, count=3):
    scores = np.clip(normalize(text_features) @ normalize(feature), -1, 1)
    return [{**candidates[int(i)], "score": float(scores[i])} for i in np.argsort(-scores, kind="stable")[:count]]


def analyze(root, out, config):
    started = time.perf_counter()
    manifest = read_json(out / "manifest.json")
    for source in manifest["sources"]:
        if sha256(root / source["file"]) != source["sha256"]:
            raise ValueError(f"Source changed since preparation: {source['file']}")
    catalog = read_json(root / "objects.json")["objects"]
    observations = manifest["observations"]
    ids = [o["observationId"] for o in observations]
    entities, photos = [o["entityId"] for o in observations], [o["photo"] for o in observations]
    with np.load(out / "points.npz", allow_pickle=False) as loaded:
        points = [loaded[key] for key in ids]
    overlap = point_overlaps(points, photos, config["overlapNative"])
    objects = [{"entityId": obj["id"], "label": obj["label"],
                "sourceRefs": [{k: o[k] for k in ("photo", "observationId", "sceneObservationId", "cropPath")} for o in observations if o["entityId"] == obj["id"]],
                "results": [], "policyContext": {"applicability": "unknown", "machineResult": None, "candidateTopics": [], "facts": [],
                    "missingEvidence": ["applicability_confirmation", "verified_metric_scale", "hazard_relationship", "policy_source_and_version"]}}
               for obj in catalog]
    queries = [{**q, "results": []} for q in config["queries"]]
    summaries, associations, timings = [], [], {}
    for key in ENCODERS:
        with np.load(out / f"embeddings-{key}.npz", allow_pickle=False) as loaded:
            emb = dict(loaded)
        expected = {"observationIds": ids, "photoIds": [r["photo"] for r in manifest["photos"]],
                    "classPhrases": [r["phrase"] for r in config["classes"]], "queryPhrases": [r["phrase"] for r in config["queries"]]}
        if str(emb["manifestSha256"]) != sha256(out / "manifest.json") or any(emb[k].tolist() != v for k, v in expected.items()):
            raise ValueError(f"Stale or misordered {key} embeddings")
        photo_index = {photo: i for i, photo in enumerate(emb["photoIds"].tolist())}
        local, fused, weights = fuse(emb["plain"], emb["masked"], emb["global"], [photo_index[p] for p in photos])
        clusters, edges = associate(fused, photos, overlap, config["overlapThreshold"], config["semanticThreshold"])
        groups = {i: cluster for cluster in clusters for i in cluster}
        cross_view, nearest = {}, {}
        for variant, feature in zip(VARIANTS, (emb["plain"], fused, fused)):
            cross_view[variant], nearest[variant] = nearest_matches(feature, photos, entities,
                overlap if variant == "spatial_semantic" else None, config["overlapThreshold"])
            for row in nearest[variant]:
                row["left"], row["right"] = ids[row["left"]], ids[row["right"]]
        correct, total, available = dict.fromkeys(VARIANTS, 0), dict.fromkeys(VARIANTS, 0), dict.fromkeys(VARIANTS, 0)
        pooled = []
        for obj in objects:
            indices = [i for i, entity in enumerate(entities) if entity == obj["entityId"]]
            supported = [i for i in indices if observations[i]["supportedPixels"] > 0]
            anchor = max(supported, key=lambda i: (observations[i]["supportedPixels"], observations[i]["maskPixels"], -i)) if supported else None
            multiview = normalize(fused[indices].mean(axis=0)) if indices else None
            pooled.append(multiview)
            for variant in VARIANTS:
                selection = indices if variant == "multiview" else ([anchor] if variant == "single_view" else groups[anchor]) if anchor is not None else []
                features = emb["plain"] if variant == "single_view" else fused
                top = ranked(normalize(features[selection].mean(axis=0)), emb["classText"], config["classes"]) if selection else []
                obj["results"].append({"encoder": key, "variant": variant, "top3": top, "viewCount": len({photos[i] for i in selection}),
                    "supportObservationIds": [ids[i] for i in selection], "anchorObservationId": ids[anchor] if anchor is not None else None,
                    "status": "available" if selection else "unavailable_no_valid_point_support"})
                available[variant] += bool(top)
                reference = config["referenceClasses"].get(obj["entityId"])
                if reference is not None:
                    total[variant] += 1
                    correct[variant] += bool(top and top[0]["id"] == reference)
        for variant in VARIANTS:
            summaries.append({"encoder": key, "variant": variant, "objectCount": len(objects), "availableObjectCount": available[variant],
                              "referenceAgreement": {"correct": correct[variant], "total": total[variant]}, "crossView": cross_view[variant]})
        for query_index, query in enumerate(queries):
            scores = [(i, float(np.clip(feature @ emb["queryText"][query_index], -1, 1))) for i, feature in enumerate(pooled) if feature is not None]
            for i, score in sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:5]:
                query["results"].append({"encoder": key, "entityId": objects[i]["entityId"], "label": objects[i]["label"], "score": score})
        for edge in edges:
            edge["left"], edge["right"] = ids[edge["left"]], ids[edge["right"]]
        associations.append({"encoder": key, "clusters": [[ids[i] for i in cluster] for cluster in clusters], "edges": edges,
                             "nearestMatches": nearest, "localGlobalWeights": {ids[i]: float(w) for i, w in enumerate(weights)},
                             "unavailableGeometryObservationIds": [ids[i] for i, p in enumerate(points) if not len(p)],
                             "crossPhotoPairs": int(np.triu(np.array(photos)[:, None] != np.array(photos)[None, :], 1).sum()),
                             "availableGeometryPairs": int(np.isfinite(overlap[np.triu_indices(len(ids), 1)]).sum()),
                             "geometryGatedPairs": int(np.triu(overlap >= config["overlapThreshold"], 1).sum())})
        timings[key] = read_json(out / f"timing-{key}.json")
    timings.update(prepareSeconds=manifest["prepareSeconds"], analyzeSeconds=round(time.perf_counter() - started, 4))
    result = {"schemaVersion": 1, "status": "model_interpretation_not_ground_truth", "sourceRevisionId": manifest["sourceRevisionId"],
        "method": {"name": "HOV-inspired local/global fusion and observed-point association", "encoders": ENCODERS,
                   "backboneDisclosure": "Uses PE-Core-L and SigLIP 2 so400m via route_jev, different backbones from HOV; this is not a HOV reproduction.",
                   "local": "normalize(0.75 * masked + 0.25 * plain)",
                   "global": "One full-image feature per photo; per-photo softmax of cosine(local, global) over observation masks",
                   "fused": "normalize(w * global + (1 - w) * local)",
                   "association": "Cross-photo edges gated by native-point overlap and fused cosine; descending overlap * cosine, at most one observation per photo per cluster",
                   "overlap": "max of the two directed nearest-neighbor fractions within overlapNative; at most 1024 observed points per mask",
                   "anchor": "Largest supported-pixel count; mask area and source order break ties; same anchor for single_view and spatial_semantic",
                   "variants": {"single_view": "Plain feature of anchor", "multiview": "Mean of fused features in existing catalog group",
                                "spatial_semantic": "Mean of fused features in label-blind cluster containing anchor"},
                   "crossViewAblation": {"single_view": "Plain nearest cross-photo observation", "multiview": "Fused nearest cross-photo observation",
                                         "spatial_semantic": "Fused nearest among overlap-gated cross-photo observations; no cosine gate for this nearest-neighbor ablation"},
                   "crossViewDenominator": "Every observation with a cross-photo candidate, including catalog singletons; one directed match per source",
                   "referenceCaveat": "Existing entity IDs and referenceClasses are evaluation proxies, not ground truth; precision means catalog agreement",
                   "scoreMeaning": "Cosine similarity, not probability or calibrated confidence",
                   "limitations": ["Native scale and camera/depth errors are unverified", "Changing robot poses may fail geometry gates",
                                   "Clusters do not combine generated meshes or hazard envelopes", "Candidate classes and queries are a closed text list"]},
        "protocol": {"frozenAt": config.get("frozenAt"), "referenceStatus": config.get("referenceStatus", "proxy catalog labels, not ground truth"),
                     "notes": config.get("notes", []), "overlapNative": config["overlapNative"], "overlapThreshold": config["overlapThreshold"], "semanticThreshold": config["semanticThreshold"],
                     "sources": manifest["sources"], "pointSource": manifest["pointSource"], "sceneTransformNative": manifest["sceneTransformNative"],
                     "classes": config["classes"], "queries": config["queries"], "referenceClasses": config["referenceClasses"],
                     "referenceUse": "Evaluation only; no labels or entity identities in neural inputs or spatial/semantic association"},
        "timing": timings, "summary": summaries, "objects": objects, "queries": queries, "associations": associations}
    write_json(out / "semantic-experiment.json", result)
    return {"objects": len(objects), "observations": len(observations), "summary": summaries, "analyzeSeconds": timings["analyzeSeconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare", action="store_true")
    stage.add_argument("--encode", choices=ENCODERS)
    stage.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    config = config_checked(read_json(args.config))
    result = prepare(args.root, args.out, config) if args.prepare else encode(args.out, config, args.encode) if args.encode else analyze(args.root, args.out, config)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
