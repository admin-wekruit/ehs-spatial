"""The report viewer's Gaussian splat file: splats.splat (32 bytes per Gaussian) and splats.json.

Record, little-endian: float32 x, y, z (world, the report's frame and native units); float32 sx, sy, sz (linear scales);
uint8 r, g, b, a (base colour as sRGB 0..255, a = opacity * 255); uint8 quaternion w, x, y, z, normalised, round(q*128+128).
Records are sorted by importance (opacity x volume, descending), so every prefix of the file is a coarser scene. This byte
layout is the contract with web/src/viewer/splat-layer.ts; modal_apps/splat_train.py prunes, packs and writes through here.

  python scripts/splat_to_web.py --inspect splats.splat
  python scripts/splat_to_web.py --self-check
"""
import argparse
import json
from pathlib import Path

import numpy as np

RECORD = np.dtype([("position", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rotation", "u1", 4)])


def keep(positions, scales, opacity, max_count, min_opacity=.5 / 255, centre=np.zeros(3), max_distance=np.inf):
    """Indices of the Gaussians worth exporting, most important first: finite, visible in 8 bits (alpha byte > 0), within
    max_distance of centre, at most max_count.

    Huge Gaussians are kept: on ME340 they are the far walls and ceiling, and dropping them cost up to 3 dB held-out PSNR. The
    distance cut is for strays far beyond the scene, which would stretch a viewer's depth-sort range.
    """
    ok = np.isfinite(positions).all(1) & np.isfinite(scales).all(1) & (opacity >= min_opacity)
    ok &= np.linalg.norm(positions - centre, axis=1) <= max_distance
    index = np.nonzero(ok)[0]
    return index[np.argsort(-(opacity[index] * scales[index].prod(1)), kind="stable")[:max_count]]


def pack(positions, scales, rgb, opacity, quats):
    """Records in the given order; scales linear, rgb and opacity in 0..1, quats w, x, y, z (any norm)."""
    out = np.zeros(len(positions), RECORD)
    out["position"], out["scale"] = positions, scales
    out["rgba"] = np.clip(np.round(np.column_stack([rgb, opacity]) * 255), 0, 255)
    q = quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    out["rotation"] = np.clip(np.round(q * 128 + 128), 0, 255)
    return out


def unpack(data):
    """splat32 bytes -> float32 positions, scales, rgb (0..1), opacity (0..1) and unit quats (w, x, y, z), as a viewer reads them."""
    r = np.frombuffer(data, RECORD)
    q = (r["rotation"].astype(np.float32) - 128) / 128
    return {"positions": r["position"].copy(), "scales": r["scale"].copy(), "rgb": r["rgba"][:, :3] / np.float32(255),
            "opacity": r["rgba"][:, 3] / np.float32(255), "quats": q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)}


def write(output, records, name="splats", **meta):
    """NAME.splat and NAME.json (format, count and bounds from the records; frame, metrics, frames, versions from meta)."""
    (output / f"{name}.splat").write_bytes(records.tobytes())
    p = records["position"]
    info = {**meta, "format": "splat32", "count": len(records), "bytesPerGaussian": RECORD.itemsize,
            "bounds": {"min": p.min(0).tolist(), "max": p.max(0).tolist()}}
    (output / f"{name}.json").write_text(json.dumps(info, indent=1))
    return info


def inspect(path):
    data = Path(path).read_bytes()
    assert len(data) % RECORD.itemsize == 0, "not a whole number of 32-byte records"
    s = unpack(data)
    importance = s["opacity"] * s["scales"].prod(1)
    print(json.dumps({"count": len(s["positions"]), "bounds": [s["positions"].min(0).round(3).tolist(), s["positions"].max(0).round(3).tolist()],
                      "opacity_median": float(np.median(s["opacity"])), "scale_median": float(np.median(s["scales"])),
                      "sorted_by_importance": bool(np.all(np.diff(importance[::max(1, len(importance) // 10000)]) <= 1e-6))}))


def self_check():
    rng = np.random.default_rng(0)
    n = 1000
    positions, scales = rng.normal(size=(n, 3)).astype(np.float32), np.exp(rng.normal(-3, 1, (n, 3))).astype(np.float32)
    rgb, opacity, quats = rng.random((n, 3)), rng.random(n), rng.normal(size=(n, 4))
    opacity[:3], positions[3] = [0, .4 / 255, .6 / 255], np.nan  # invisible in 8 bits, just visible, broken
    index = keep(positions, scales, opacity, max_count=900)
    assert len(index) == 900 and not {0, 1, 3} & set(index.tolist()), "invisible and non-finite Gaussians dropped, count capped"
    assert 2 in keep(positions, scales, opacity, max_count=n), "an alpha byte of 1 is kept"
    positions[4] = 50
    assert 4 not in keep(positions, scales, opacity, max_count=n, max_distance=20) and 5 in keep(positions, scales, opacity, max_count=n, max_distance=20), "far strays dropped"
    records = pack(positions[index], scales[index], rgb[index], opacity[index], quats[index])
    assert RECORD.itemsize == 32 and len(records.tobytes()) == 32 * 900, "32 bytes per Gaussian"
    back = unpack(records.tobytes())
    assert np.array_equal(back["positions"], positions[index]) and np.array_equal(back["scales"], scales[index]), "float32 fields exact"
    assert np.abs(back["rgb"] - rgb[index]).max() <= .5 / 255 + 1e-6 and np.abs(back["opacity"] - opacity[index]).max() <= .5 / 255 + 1e-6, "8-bit colour"
    unit = quats[index] / np.linalg.norm(quats[index], axis=1, keepdims=True)
    assert np.abs(np.abs((back["quats"] * unit).sum(1)) - 1).max() < 1e-3, "8-bit quaternion keeps the rotation (w, x, y, z order)"
    importance = back["opacity"] * back["scales"].prod(1)
    assert importance[0] >= importance[-1] and np.all(np.diff(opacity[index] * scales[index].prod(1)) <= 0), "most important first"
    raw = np.frombuffer(records.tobytes()[:32], np.uint8)
    assert np.frombuffer(raw[:12].tobytes(), "<f4").tolist() == positions[index[0]].tolist() and raw[28] == records["rotation"][0, 0], "byte layout: xyz, scales, rgba, wxyz"
    print("splat_to_web check passed: 32-byte records round-trip, pruning and importance order hold")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--inspect", type=Path, help="print count, bounds and ordering of a splats.splat")
    a = parser.parse_args()
    self_check() if a.self_check else inspect(a.inspect)
