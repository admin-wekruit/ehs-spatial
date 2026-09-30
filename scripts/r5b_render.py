"""r5b (models): draw a display model over the video frame it was seen in, for looking at its outline by eye (contact sheets).

  raster(tris, c2w, K, wh)       world triangles -> the frame's pixels they cover (nearest first: a z-buffer over the triangles)
  overlay(img, cov, rgb, alpha)  the covered pixels tinted, the silhouette's edge in a bright line
  crop_box(mask)                 the square crop around a mask (1.6 x its box)

Numpy + OpenCV only (the Mac); a few thousand triangles a tile.
"""
import numpy as np


def k_full(K, grid_wh, full_wh):
    """K on the DA3 grid (504 x 280) -> K on the source frame (1280 x 720): the same pixel-centre mapping as sam3d.fast_clip."""
    sx, sy = full_wh[0] / grid_wh[0], full_wh[1] / grid_wh[1]
    S = np.array([[sx, 0, (sx - 1) / 2], [0, sy, (sy - 1) / 2], [0, 0, 1.]])
    return S @ np.asarray(K, float)


def raster(tris, c2w, K, wh, colors=None):
    """tris (n, 3, 3) world; c2w (4, 4); K at wh -> (covered (h, w) bool, rgb (h, w, 3) uint8 of the nearest triangle, or None).
    Triangles with a vertex behind the camera are dropped; painter's order far to near (each triangle's mean depth)."""
    import cv2
    w, h = wh
    T = np.asarray(tris, float).reshape(-1, 3, 3)
    X = (T - c2w[:3, 3]) @ c2w[:3, :3]
    z = X[..., 2]
    ok = (z > .05).all(1)
    X, z = X[ok], z[ok]
    cols = None if colors is None else np.asarray(colors)[ok]
    uv = np.stack([K[0, 0] * X[..., 0] / z + K[0, 2], K[1, 1] * X[..., 1] / z + K[1, 2]], -1)
    keep = np.isfinite(uv).all((1, 2)) & (np.abs(uv).max((1, 2)) < 1e5)
    uv, z = uv[keep], z[keep]
    cols = None if cols is None else cols[keep]
    cov = np.zeros((h, w), np.uint8)
    rgb = np.zeros((h, w, 3), np.uint8) if cols is not None else None
    for i in np.argsort(-z.mean(1)):
        pts = np.round(uv[i] * 8).astype(np.int32)
        cv2.fillConvexPoly(cov, pts, 1, cv2.LINE_8, 3)
        if rgb is not None:
            cv2.fillConvexPoly(rgb, pts, [int(c) for c in cols[i][:3]], cv2.LINE_8, 3)
    return cov > 0, rgb


def overlay(img, cov, rgb=None, tint=(80, 200, 255), alpha=.45, edge=(0, 255, 255)):
    """BGR img, covered pixels tinted (or painted with the model's own colours), the silhouette edge drawn."""
    import cv2
    out = img.copy()
    if cov.any():
        paint = np.zeros_like(img)
        paint[:] = tint
        if rgb is not None:
            paint = rgb[..., ::-1]
        out[cov] = (out[cov] * (1 - alpha) + paint[cov] * alpha).astype(np.uint8)
        cs, _ = cv2.findContours(cov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, edge, 2, cv2.LINE_AA)
    return out


def outline(img, mask, colour=(255, 255, 255)):
    import cv2
    out = img.copy()
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.drawContours(out, cs, -1, colour, 2, cv2.LINE_AA)
    return out


def crop_box(mask, scale=1.6, min_side=64):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min(), min_side) * scale / 2
    return int(cx - half), int(cy - half), int(cx + half), int(cy + half)


def cut(img, box, side=240):
    """The crop (grey outside the frame), resized to side x side."""
    import cv2
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    out = np.full((y1 - y0, x1 - x0, 3), 96, np.uint8)
    a0, b0, a1, b1 = max(0, y0), max(0, x0), min(h, y1), min(w, x1)
    if a1 > a0 and b1 > b0:
        out[a0 - y0:a1 - y0, b0 - x0:b1 - x0] = img[a0:a1, b0:b1]
    return cv2.resize(out, (side, side), interpolation=cv2.INTER_AREA)


def label(img, text, y=None, scale=.4):
    import cv2
    y = img.shape[0] - 6 if y is None else y
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def glb_nodes(raw):
    """A GLB (the tier-0 shot GLB, or a generated model's one-mesh GLB) -> {node name (or 'mesh<i>'): (V (n, 3) with the node's
    translation, F (m, 3), RGBA (n, 4) uint8 or None)}. Only what our writers emit: float32 positions, uint8 colours, uint32/uint16 indices."""
    import json
    import struct
    n = struct.unpack("<I", raw[12:16])[0]
    doc, binary = json.loads(raw[20:20 + n]), raw[20 + n + 8:]
    dt = {5121: np.uint8, 5125: np.uint32, 5126: np.float32, 5123: np.uint16}
    width = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}

    def acc(i):
        a = doc["accessors"][i]
        v = doc["bufferViews"][a["bufferView"]]
        off = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(binary, dt[a["componentType"]], a["count"] * width[a["type"]], off).reshape(a["count"], -1)
    out = {}
    for i, node in enumerate(doc.get("nodes", [])):
        if node.get("mesh") is None:
            continue
        prim = doc["meshes"][node["mesh"]]["primitives"][0]
        V = acc(prim["attributes"]["POSITION"]).astype(float) + np.asarray(node.get("translation", [0, 0, 0]), float)
        C = acc(prim["attributes"]["COLOR_0"]) if "COLOR_0" in prim["attributes"] else None
        out[node.get("name") or f"mesh{i}"] = (V, acc(prim["indices"]).reshape(-1, 3).astype(np.int64), C)
    return out


def self_check():
    K = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1]])
    c2w = np.eye(4)
    tri = np.array([[[-.5, -.5, 2.], [.5, -.5, 2.], [0, .5, 2.]]])
    cov, rgb = raster(tri, c2w, K, (640, 480), colors=[[255, 0, 0]])
    assert cov[240, 320] and not cov[10, 10] and tuple(rgb[240, 320]) == (255, 0, 0)
    assert np.allclose(k_full(np.eye(3), (504, 280), (1008, 560))[0, 0], 2.)
    from fast_report import surface
    V = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.]])
    glb, _ = surface.shot_glb([("a", V + 5, np.array([[0, 1, 2]]), np.full((3, 3), 9, np.uint8))])
    got = glb_nodes(glb)
    assert np.allclose(got["a"][0], V + 5) and got["a"][1].tolist() == [[0, 1, 2]] and got["a"][2][0, 0] == 9
    print("r5b_render self-check ok")


if __name__ == "__main__":
    import sys
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
