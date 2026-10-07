"""Texture a frozen BOR1 surface from observed photos, without filling holes.

  .venv/bin/python scripts/research/texture_workcell.py --self-check
  .venv/bin/python scripts/research/texture_workcell.py --surface surface.ply --selection selection.json --output textured-workcell.glb
"""

import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import trimesh

from check_multiview_consistency import digest, frame_arrays, load_frame, project


ROOT = Path(__file__).resolve().parents[2]
CONTENT = [63, 0, 455, 518]
FRAME_IDS = tuple(f"frame_{i:04d}" for i in range(1, 5))
RELATIVE_DEPTH = 0.04
TILE = (768, 1024)


def visible_samples(points, frame, content_rect):
    """Nearest observed z must agree; invalid/background/occluded pixels reject."""
    uv, z = project(points, frame)
    left, top, right, bottom = content_rect
    # ponytail: require pixel centers inside the photo; this also leaves a
    # half-pixel margin after atlas resizing, without adding a texture gutter.
    inside = (np.isfinite(uv).all(-1) & np.isfinite(z) & (z > 0)
              & (uv[:, 0] >= left) & (uv[:, 0] <= right - 1)
              & (uv[:, 1] >= top) & (uv[:, 1] <= bottom - 1))
    good = np.zeros(len(points), bool)
    candidates = np.flatnonzero(inside)
    xy = np.floor(uv[candidates] + 0.5).astype(np.int64)
    observed = frame["depth"][xy[:, 1], xy[:, 0]]
    trusted = frame["texture_mask"][xy[:, 1], xy[:, 0]]
    good[candidates] = trusted & (np.abs(z[candidates] - observed) <= RELATIVE_DEPTH * observed)
    return uv, z, good


def atlas_uv(canonical_uv, frame_id):
    """Original-photo pixel centers -> bottom-left UV expected by trimesh."""
    index = FRAME_IDS.index(frame_id)
    photo_uv = (np.asarray(canonical_uv) - [63, 0] + 0.5) / [392, 518]
    return np.stack(((index % 2 + photo_uv[..., 0]) / 2,
                     1 - (index // 2 + photo_uv[..., 1]) / 2), axis=-1)


def photo_texture(uv, image):
    material = trimesh.visual.material.PBRMaterial(
        baseColorFactor=[255, 255, 255, 255], baseColorTexture=image,
        metallicFactor=0, roughnessFactor=1, doubleSided=True)
    return trimesh.visual.texture.TextureVisuals(uv=uv, material=material)


def glb_base_colors(data):
    json_size = int.from_bytes(data[12:16], "little")
    document = json.loads(data[20:20 + json_size])
    return [m.get("pbrMetallicRoughness", {}).get("baseColorFactor", [1, 1, 1, 1])
            for m in document["materials"]]


def texture(surface, selection_path, output):
    if output.suffix.lower() != ".glb":
        raise ValueError("Output must be a new .glb file")
    metrics_path = output.parent / "texture-metrics.json"
    if output.exists() or metrics_path.exists():
        raise ValueError("Output GLB and texture-metrics.json must not already exist")
    selection = json.loads(selection_path.read_text())
    if (selection["content_rect"] != CONTENT or selection["minimum_confidence"] != 0.1
            or selection["metric_scale_known"] is not False):
        raise ValueError("Expected frozen BOR1 content rectangle, confidence=0.1 and uncalibrated scale")
    include = selection.get("include_frames", list(FRAME_IDS))
    if not isinstance(include, list) or not include or len(set(include)) != len(include) or not set(include) <= set(FRAME_IDS):
        raise ValueError("include_frames must contain unique frame_0001..frame_0004 identifiers")
    include = [name for name in FRAME_IDS if name in include]
    source_geometry = Path(selection["source_geometry"])
    mesh = trimesh.load(surface, force="mesh", process=False)
    if (not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all()):
        raise ValueError("Expected a finite triangular surface")
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    triangles = vertices[faces]
    centers = triangles.mean(1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normal_lengths = np.linalg.norm(normals, axis=1)
    nondegenerate = normal_lengths > 1e-12
    normals /= np.maximum(normal_lengths[:, None], 1e-12)
    winner = np.full(len(faces), -1, dtype=np.int8)
    best = np.full(len(faces), -1.0)
    atlas = Image.new("RGB", (TILE[0] * 2, TILE[1] * 2))
    frames, frame_metrics = {}, []
    started = time.perf_counter()
    for frame_id in include:
        index = FRAME_IDS.index(frame_id)
        directory = source_geometry / "frames" / frame_id
        frame = load_frame(directory)
        if frame["depth"].shape != (518, 518):
            raise ValueError(f"{frame_id}: expected the frozen 518x518 point grid")
        confidence_path = directory / "conf.npy"
        confidence = np.load(confidence_path, allow_pickle=False)
        if confidence.shape != frame["depth"].shape:
            raise ValueError(f"{frame_id}: confidence shape does not match point grid")
        frame["texture_mask"] = frame["usable"] & np.isfinite(confidence) & (confidence >= 0.1)
        for rectangle in selection["exclude_rects"][frame_id]:
            if (len(rectangle) != 4 or any(int(x) != x for x in rectangle)
                    or not (0 <= rectangle[0] < rectangle[2] <= 518 and 0 <= rectangle[1] < rectangle[3] <= 518)):
                raise ValueError(f"{frame_id}: invalid exclusion rectangle")
            left, top, right, bottom = rectangle
            frame["texture_mask"][top:bottom, left:right] = False
        photo_path = ROOT / "runs" / "user-bor1-02" / "input" / f"image_{index+1:02d}.jpg"
        with Image.open(photo_path) as photo:
            if photo.size != (3024, 4032) or photo.getexif().get(274, 1) != 1:
                raise ValueError(f"Unexpected source photo size/orientation: {photo_path}")
            atlas.paste(photo.convert("RGB").resize(TILE, Image.Resampling.LANCZOS),
                        (index % 2 * TILE[0], index // 2 * TILE[1]))
        _, _, center_ok = visible_samples(centers, frame, CONTENT)
        candidates = np.flatnonzero(center_ok & nondegenerate)
        _, _, corner_ok = visible_samples(triangles[candidates].reshape(-1, 3), frame, CONTENT)
        supported = candidates[corner_ok.reshape(-1, 3).all(1)]
        rays = frame["c2w"][:3, 3] - centers[supported]
        distances = np.linalg.norm(rays, axis=1)
        cosine = np.abs(np.sum(normals[supported] * rays, axis=1)) / np.maximum(distances, 1e-12)
        quality = cosine / np.maximum(distances * distances, 1e-12)
        chosen = quality > best[supported]
        winner[supported[chosen]], best[supported[chosen]] = index, quality[chosen]
        frames[frame_id] = frame
        frame_metrics.append({"frame_id": frame_id, "source_image": str(photo_path),
                              "source_image_sha256": digest(photo_path), "geometry_sha256": frame["sha256"],
                              "confidence_sha256": digest(confidence_path),
                              "centroid_supported_faces": int(center_ok.sum()),
                              "whole_triangle_supported_faces": int(len(supported))})
        print(f"{frame_id}: {len(supported)}/{len(faces)} faces supported by observed depth", flush=True)
    retained = np.flatnonzero(winner >= 0)
    if not len(retained):
        raise ValueError("No surface triangles have a supporting real photograph")
    kept_vertices = triangles[retained].reshape(-1, 3)
    uv = np.empty((len(retained), 3, 2), dtype=float)
    for item in frame_metrics:
        frame_id = item["frame_id"]
        selected = winner[retained] == FRAME_IDS.index(frame_id)
        canonical_uv, _ = project(triangles[retained[selected]].reshape(-1, 3), frames[frame_id])
        uv[selected] = atlas_uv(canonical_uv, frame_id).reshape(-1, 3, 2)
        item["assigned_faces"] = int(selected.sum())
    if not np.isfinite(uv).all() or not ((uv > 0) & (uv < 1)).all():
        raise ValueError("Texture coordinates leave the observed photograph atlas")
    # Face-local vertices keep UV seams exact; geometry is neither moved nor remeshed.
    textured = trimesh.Trimesh(vertices=kept_vertices, faces=np.arange(len(kept_vertices)).reshape(-1, 3),
                               visual=photo_texture(uv.reshape(-1, 2), atlas), process=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    exported = textured.export(file_type="glb")
    with output.open("xb") as stream:
        stream.write(exported)
    reread = trimesh.load(output, force="mesh", process=False)
    texture_image = getattr(reread.visual.material, "baseColorTexture", None)
    if texture_image is None:
        texture_image = getattr(reread.visual.material, "image", None)
    readback = {"face_count": len(reread.faces), "vertex_count": len(reread.vertices),
                "visual_kind": reread.visual.kind, "texture_size": list(texture_image.size) if texture_image is not None else None,
                "base_color_factors": glb_base_colors(exported),
                "vertices_match": bool(np.allclose(reread.vertices, textured.vertices, atol=1e-6, rtol=0)),
                "faces_match": bool(np.array_equal(reread.faces, textured.faces)),
                "uv_match": bool(np.allclose(reread.visual.uv, textured.visual.uv, atol=1e-6, rtol=0))}
    valid = (readback["face_count"] == len(retained) and readback["visual_kind"] == "texture"
             and readback["texture_size"] == [1536, 2048]
             and all(factor == [1, 1, 1, 1] for factor in readback["base_color_factors"])
             and all(readback[key] for key in ("vertices_match", "faces_match", "uv_match")))
    metrics = {"status": "complete" if valid else "readback_failed", "source_surface": str(surface.resolve()),
               "source_surface_sha256": digest(surface), "selection_path": str(selection_path.resolve()),
               "selection_sha256": digest(selection_path), "selection": selection,
               "source_geometry": str(source_geometry), "include_frames": include,
               "input_faces": len(faces), "retained_faces": len(retained),
               "face_retention_fraction": len(retained) / len(faces), "removed_unsupported_faces": len(faces)-len(retained),
               "support_rule": "Centroid and three vertices must hit original valid/conf>=0.1 photo content outside excluded rectangles, and agree with nearest native camera-z depth within 4% of observed z.",
               "assignment_rule": "Maximum abs(face normal dot unit camera direction) / camera distance squared; ties use lower frame index.",
               "canonical_to_photo_uv": "u=(x-63+0.5)/392; v=(y+0.5)/518; atlas fixed 2x2 in original frame order; trimesh UV uses bottom-left origin.",
               "texture_rule": "Only original JPG photos resized to 768x1024; no generated texture, blending, hole fill or world coordinate change.",
               "metric_scale_known": False, "atlas_size": [1536, 2048], "frames": frame_metrics,
               "readback": readback, "output": {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": digest(output)},
               "elapsed_seconds": time.perf_counter()-started, "trimesh_version": trimesh.__version__}
    with metrics_path.open("x") as stream:
        stream.write(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    if not valid:
        raise ValueError(f"Textured GLB failed readback checks: {readback}")
    print(json.dumps({"output": str(output), "retained_faces": len(retained), "fraction": metrics["face_retention_fraction"]}), flush=True)


def self_check():
    K = np.array([[100., 0, 259.], [0, 100., 259.], [0, 0, 1.]])
    yy, xx = np.indices((518, 518))
    points = np.stack(((xx-259)/50, (yy-259)/50, np.full_like(xx, 2.)), axis=-1)
    frame = frame_arrays("check", points, np.ones((518, 518), bool), np.eye(4), K)
    frame["texture_mask"] = frame["usable"].copy()
    samples = np.array([[0., 0, 2.], [0., 0, 2.1], [0., 0, 1.8], [-4.2, 0, 2.], [0, 0, -2.]])
    uv, _, good = visible_samples(samples, frame, CONTENT)
    assert np.allclose(uv[0], [259, 259]) and good.tolist() == [True, False, False, False, False]
    frame["texture_mask"][259, 259] = False
    assert not visible_samples(samples[:1], frame, CONTENT)[2][0]
    assert np.allclose(atlas_uv(np.array([[63., 0.]]), "frame_0001"), [[0.5/392/2, 1-0.5/518/2]])
    assert np.allclose(atlas_uv(np.array([[454., 517.]]), "frame_0004"), [[(1+391.5/392)/2, 1-(1+517.5/518)/2]])
    image = Image.new("RGB", (1536, 2048), "red")
    image.paste(Image.new("RGB", TILE, "blue"), (768, 1024))
    mesh = trimesh.Trimesh(vertices=[[0, 0, 2], [1, 0, 2], [0, 1, 2]], faces=[[0, 1, 2]],
                           visual=photo_texture([[.1, .9], [.2, .9], [.1, .8]], image), process=False)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "check.glb"
        mesh.export(path)
        assert glb_base_colors(path.read_bytes()) == [[1, 1, 1, 1]]
        read = trimesh.load(path, force="mesh", process=False)
        assert len(read.faces) == 1 and np.allclose(read.visual.uv, mesh.visual.uv)
        assert read.visual.material.baseColorTexture.getpixel((1000, 1500))[:3] == (0, 0, 255)
    print("self-check passed: pinhole projection, UV orientation, occlusion/invalid rejection, white PBR base color and textured GLB roundtrip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.surface is None or args.selection is None or args.output is None:
        parser.error("--surface, --selection and --output are required")
    else:
        texture(args.surface, args.selection, args.output)
