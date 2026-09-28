"""Package the ME340 digital twin from its verify rounds (M4 twin spec, docs/phase2/TWIN-SPEC.md 7.4-7.5 and 10).

The last round's verify run is what ships: twin.glb (metres, +Y up, origin on the floor), twin.usda and its binary twin.usd
(Isaac Sim), textures/, twin.json, verify.json. Added here, nothing re-measured:

  sheets/before/, sheets/after/        the first and the last round's contact sheets (same walk views), and
  sheets/scene-NN-before-after.jpg     the two stacked;
  metrics.json                         per round and per object: held-out IoU, depth agreement, free space, control
                                       resolution, status; objects by representation and status; shell; explained share; spend;
  view/                                index.html (three.js, orbit controls) with twin.glb, twin-all.glb (every stand-in, each
                                       labelled with its status) and the observed layers in the twin's frame for comparison:
                                       the LingBot dense points (voxel-thinned) and the observed textured mesh.

Every twin node stays an inferred stand-in (extras.panoptes.layer twin_inferred, notForMeasurement); the observed layers are
copied only to be looked at next to it.

  python scripts/package_twin.py --rounds VERIFY_1 [VERIFY_2 ...] --output R/runs/m4-twin-me340
  python scripts/package_twin.py --self-check
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs")
LEDGER = RUNS / "m4-twin-spend.jsonl"
OBSERVED = {"points": RUNS / "me340-lingbot-map-222/dense-points.glb", "mesh": RUNS / "me340-filled-225/textured-scene.glb"}
POINT_VOXEL_M = .015
SHIPPED = ("twin.glb", "twin-all.glb", "twin.usda", "twin.usd", "twin.json", "verify.json", "fixes.json", "critique.json")


def round_metrics(folder):
    """One verify round: per object numbers (with category, representation, size from objects.json) and summaries."""
    verify, twin = json.loads((folder / "verify.json").read_text()), json.loads((folder / "twin.json").read_text())
    built = {o["twinId"]: o for o in json.loads(Path(twin["inputs"]["objects.json"]["path"]).read_text())["objects"]}
    rows = []
    for tid, r in verify["objects"].items():
        o, s, c = built.get(tid, {}), r.get("summary") or {}, r.get("control") or {}
        rows.append({"twinId": tid, "priority": o.get("priority"), "category": o.get("category"), "representation": o.get("representation"),
                     "kind": (o.get("representation") or "?").split(":")[0], "status": r["status"], "sizeM": (o.get("box") or {}).get("sizeM"),
                     "dimSource": (o.get("box") or {}).get("dimSource"), **{k: s.get(k) for k in (
                         "judgedViews", "iou", "depthRelMedian", "depthRelP95", "overhang", "foreign", "occluded")},
                     "controlResolvedAtM": c.get("resolvedAtM"), "controlFailedShare": c.get("failedShare"), "reasons": r["reasons"]})
    rows.sort(key=lambda r: (r["priority"] or 9, r["twinId"]))
    median = lambda key, which: (lambda xs: round(float(np.median(xs)), 4) if xs else None)([r[key] for r in which if r[key] is not None])
    passed = [r for r in rows if r["status"] == "pass"]
    selected = [r for r in rows if (r["priority"] or 9) <= 2]
    return {"verifyRun": str(folder), "objectsRun": str(Path(twin["inputs"]["objects.json"]["path"]).parent),
            "shellRun": str(Path(twin["inputs"]["shell.json"]["path"]).parent) if "shell.json" in twin["inputs"] else None,
            "counts": {**verify["counts"], "byStatus": dict(Counter(r["status"] for r in rows)),
                       "byRepresentation": {k: dict(Counter(r["status"] for r in rows if r["kind"] == k)) for k in sorted({r["kind"] for r in rows})},
                       "byPriority": {p: dict(Counter(r["status"] for r in rows if r["priority"] == p)) for p in sorted({r["priority"] or 9 for r in rows})},
                       "p1p2PassShare": round(len([r for r in selected if r["status"] == "pass"]) / max(len(selected), 1), 3),
                       "skipped": len(twin.get("skipped", [])), "absentShell": len(twin.get("absent", []))},
            "medians": {"all": {k: median(k, rows) for k in ("iou", "depthRelMedian", "depthRelP95")},
                        "passed": {k: median(k, passed) for k in ("iou", "depthRelMedian", "depthRelP95")}},
            "shell": {n: {k: e.get(k) for k in ("kind", "status", "depthRelMedian", "limit", "judgedViews", "reasons")} for n, e in verify["shell"].items()},
            "explainedShare": twin["explainedShare"], "consistency": {k: verify["consistency"][k] for k in ("floorPenetration", "wallPenetration", "manhattanYawDeg")}
            | {"overlapPairs": len(verify["consistency"]["overlap"])},
            "usdCheck": verify.get("usdCheck"), "rule": verify["rule"], "fixes": fix_counts(folder), "objects": rows}


def fix_counts(folder):
    if not (folder / "fixes.json").exists():
        return None
    fixes = json.loads((folder / "fixes.json").read_text())
    return {"fixes": len(fixes["fixes"]), "byAction": dict(Counter(f["action"] for f in fixes["fixes"])),
            "byKind": dict(Counter(f["kind"] for f in fixes["fixes"])), "ignored": len(fixes["ignored"])}


def spend(ledger):
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()] if ledger.exists() else []
    by = lambda key: {k: round(sum(r["usd"] for r in rows if r[key] == k), 4) for k in sorted({r[key] for r in rows})}
    return {"ledger": str(ledger), "totalUsd": round(sum(r["usd"] for r in rows), 4), "byBuilder": by("builder"), "byKind": by("kind"),
            "paidCalls": len(rows), "capUsd": 10.}


def points_to_twin(vertices, colours, transform, voxel):
    """Native points -> the twin frame, one point per `voxel` metres (the first in each cell)."""
    metric = vertices @ transform[:3, :3].T + transform[:3, 3]
    _, keep = np.unique(np.floor(metric / voxel).astype(np.int64), axis=0, return_index=True)
    keep.sort()
    return metric[keep], colours[keep]


def observed_layers(view, transform):
    """The observed dense points (thinned) and textured mesh, moved into the twin's frame for the viewer."""
    import trimesh
    cloud = next(g for g in trimesh.load(OBSERVED["points"], force="scene").dump() if isinstance(g, trimesh.PointCloud))
    colours = np.asarray(cloud.colors) if len(cloud.colors) else np.full((len(cloud.vertices), 4), 200, np.uint8)
    vertices, colours = points_to_twin(np.asarray(cloud.vertices), colours, transform, POINT_VOXEL_M)
    trimesh.Scene(trimesh.PointCloud(vertices, colours)).export(view / "observed-points.glb")
    mesh = trimesh.load(OBSERVED["mesh"], force="scene")
    mesh.apply_transform(transform)
    mesh.export(view / "observed-mesh.glb")
    return {"points": {"source": str(OBSERVED["points"]), "count": int(len(vertices)), "voxelM": POINT_VOXEL_M},
            "mesh": {"source": str(OBSERVED["mesh"])}}


def package(rounds, output):
    output.mkdir(parents=True, exist_ok=False)
    first, last = rounds[0], rounds[-1]
    for name in SHIPPED:
        if (last / name).exists():
            shutil.copy2(last / name, output / name)
    shutil.copytree(last / "textures", output / "textures")
    for tag, folder in (("before", first), ("after", last)):
        shutil.copytree(folder / "sheets", output / "sheets" / tag)
    for before in sorted((first / "sheets").glob("scene-*.jpg")):
        after = last / "sheets" / before.name
        if after.exists():
            a, b = cv2.imread(str(before)), cv2.imread(str(after))
            width = max(a.shape[1], b.shape[1])
            pad = lambda x: np.pad(x, ((0, 0), (0, width - x.shape[1]), (0, 0)), constant_values=255)
            cv2.imwrite(str(output / "sheets" / before.name.replace(".jpg", "-before-after.jpg")),
                        np.vstack([pad(a), np.full((8, width, 3), 255, np.uint8), pad(b)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    twin = json.loads((last / "twin.json").read_text())
    view = output / "view"
    view.mkdir()
    shutil.copy2(ROOT / "scripts/twin_viewer.html", view / "index.html")
    for name in ("twin.glb", "twin-all.glb"):
        shutil.copy2(last / name, view / name)
    observed = observed_layers(view, np.asarray(twin["transform"]["nativeToTwin"]))
    metrics = {"schema": "m4-twin-metrics-v1", "label": twin["label"], "layer": "twin_inferred", "notForMeasurement": True,
               "scale": {k: twin["transform"][k] for k in ("metresPerNativeUnit", "scaleStatus", "assumption", "metresPerNativeUnitRange")},
               "rounds": [round_metrics(r) for r in rounds], "spend": spend(LEDGER), "observedLayersInViewer": observed,
               "view": "view/index.html: serve the folder over http (python3 -m http.server) and open /view/"}
    metrics["before"], metrics["after"] = metrics["rounds"][0]["counts"], metrics["rounds"][-1]["counts"]
    (output / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    print(json.dumps({"output": str(output), "before": metrics["before"], "after": metrics["after"], "spendUsd": metrics["spend"]["totalUsd"],
                      "observedPoints": observed["points"]["count"]}))


def self_check():
    """The point thinning lands in the twin frame (floor at y = 0, metres) and keeps one point per voxel."""
    s = 2.
    transform = np.eye(4)
    transform[:3, :3] = s * np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.]])  # native up = -y
    native = np.array([[0, 0, 0], [0, 0, -.001], [1, -.5, 0], [1, -.5, -.002]])
    vertices, colours = points_to_twin(native, np.arange(4), transform, .015)
    assert len(vertices) == 2 and list(colours) == [0, 2], (vertices, colours)
    assert np.allclose(vertices[1], [2, 1, 0]), vertices  # 0.5 native units above the floor = 1 m up
    print("package_twin self-check: passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=Path, nargs="+", help="verify runs in order: the first is 'before', the last ships")
    p.add_argument("--output", type=Path)
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        return self_check()
    if not args.rounds or not args.output:
        p.error("--rounds and --output are required")
    package(args.rounds, args.output)


if __name__ == "__main__":
    sys.exit(main())
