"""Side-by-side video of a clip split into its static and dynamic layers, seen from the clip's own camera at every sampled view.

Four panels per frame: the source photo, static + dynamic, static only, dynamic only. The static layer is the fused
mesh (moving pixels left out before fusion); the dynamic layer is that view's moving-object surfaces written by
mono_room.py dynamic. Ray casting on the CPU, no model call. Encoded with the platform's H.264 writer (avc1) so a
browser plays it without ffmpeg.

  python scripts/render_static_dynamic_video.py --droid-run RUN --fused FUSED_RUN --output FILE.mp4 [--width 1920]
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_apps"))


def caster(vertices, faces):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(np.asarray(vertices, np.float32)), o3d.core.Tensor(np.asarray(faces, np.uint32)))
    return scene


def cast(scene, rays, faces, colors):
    """Depth along the ray and colour per pixel (inf / 0 where nothing is hit)."""
    hit = scene.cast_rays(rays)
    t, ids = hit["t_hit"].numpy(), hit["primitive_ids"].numpy()
    ok = np.isfinite(t)
    ids = np.where(ok, ids, 0)
    uv = hit["primitive_uvs"].numpy()
    w = np.stack([1 - uv[..., 0] - uv[..., 1], uv[..., 0], uv[..., 1]], -1)
    rgb = (colors[faces[ids]] * w[..., None]).sum(-2)
    return np.where(ok, t, np.inf), np.where(ok[..., None], rgb, 0)


def label(tile, text, dark=False):
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 30), (40, 40, 40) if dark else (255, 255, 255), -1)
    cv2.putText(tile, text, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255) if dark else (0, 0, 0), 2)
    return tile


def run(args):
    import open3d as o3d
    import trimesh
    import mono_room as M
    M.use_clip(args.droid_run)
    manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
    scene = json.loads((args.fused / "scene.json").read_text())
    mesh = o3d.io.read_triangle_mesh(str(args.fused / "mono-anchored-mesh.ply"))
    sv, sf, sc = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)[:, ::-1] * 255
    static = caster(sv, sf)
    views = [f for f in scene["frames"] if (args.fused / "mono" / f"{f['sourceFrame']:05d}.npz").exists()]
    fps = len(views) / (views[-1]["endTimeSec"] - views[0]["timeSec"])
    height = round(args.width / 4 * 3 / 4) // 2 * 2
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"avc1"), fps, (args.width, height + 0))
    assert writer.isOpened(), "no H.264 writer on this platform"
    counts = []
    for frame in views:
        index = frame["sourceFrame"]
        photo, k = M.prepare_image(cv2.imread(str(M.DATASET / manifest["frames"][index]["relative_path"])), M.CALIBRATION, 2)
        K = o3d.core.Tensor(np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]]))
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(K, o3d.core.Tensor(np.linalg.inv(np.array(frame["c2w"]))), 640, 480)
        t_static, c_static = cast(static, rays, sf, sc)
        parts = [trimesh.load(args.fused / o["surface"]["meshUrl"], force="mesh", process=False) for o in frame["objects"] if o.get("surface")]
        if parts:
            dv = np.vstack([p.vertices for p in parts]); offs = np.cumsum([0] + [len(p.vertices) for p in parts])
            df = np.vstack([p.faces + o for p, o in zip(parts, offs)]); dc = np.vstack([np.asarray(p.visual.vertex_colors)[:, 2::-1] for p in parts]).astype(float)
            t_dynamic, c_dynamic = cast(caster(dv, df), rays, df, dc)
        else:
            t_dynamic, c_dynamic = np.full((480, 640), np.inf), np.zeros((480, 640, 3))
        paint = lambda t, c, back: np.where(np.isfinite(t)[..., None], c, back).clip(0, 255).astype(np.uint8)
        both = paint(np.minimum(t_static, t_dynamic), np.where((t_dynamic < t_static)[..., None], c_dynamic, c_static), 245)
        tiles = [label(photo.copy(), f"source frame {index}  {frame['timeSec']:.1f} s"), label(both, "static + dynamic"),
                 label(paint(t_static, c_static, 245), "static only"), label(paint(t_dynamic, c_dynamic, 40), f"dynamic only ({len(parts)})", dark=True)]
        writer.write(cv2.resize(np.hstack(tiles), (args.width, height), interpolation=cv2.INTER_AREA))
        counts.append(len(parts))
    writer.release()
    report = {"video": str(args.output), "frames": len(views), "fps": fps, "size": [args.width, height], "views_with_dynamic_objects": sum(c > 0 for c in counts),
              "panels": ["source photo", "static + dynamic", "static only", "dynamic only"], "fused": str(args.fused), "newModelCalls": 0}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("droid-run", "fused", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--width", type=int, default=1920)
    run(parser.parse_args())
