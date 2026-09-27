"""One set of complete object models per scene out of the per-generator runs of complete_video_objects.py.

The delivered sets (22 / 67 / 40) were merged by hand-run scratch scripts; this is that rule as code, so a replay of the
same runs gives the same set and a changed profile (no RecGen, say) gives its own number without anyone choosing:

  1. a learned model that passed the gate beats a box, a box beats nothing;
  2. where both learned generators passed, the lower held-out fit residual (fitResidualNative) wins; a tie goes to the
     generator listed first (RecGen);
  3. a box is only for an object no learned model passed. It is dropped as the same object as, or a part of, a model
     already chosen (learned models first, then earlier boxes, in entity order) when their 3D bounding boxes overlap by
     IoU >= DUP_IOU or >= DUP_NEAR of the box's vertices lie within 2 voxels of that model's surface;
  4. every other box must pass box_free_space.py's per-pixel free-space test (--box-test, measured on that box run for
     the same mesh; a box it did not measure fails); an eye review can still exclude more (--review: a merge.json whose
     boxesDropped reasons start 'excluded at review').

Model folders are hard-linked, nothing is regenerated. --against compares with a delivered merge (entity sets, the
choice per entity, the dropped boxes) and exits 1 on any difference; a box the delivered merge excluded at review and
this one drops by the test is the same decision (listed as reviewByTest).

  python scripts/merge_object_models.py [--recgen R] --sam3d S --box B --box-test BOX_TEST_JSON [--review MERGE_JSON] [--output NEW_DIR] [--against DELIVERED_DIR]
  python scripts/merge_object_models.py --self-check
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complete_video_objects import box_iou, closest  # the same IoU and surface distance the delivered merge used
import box_free_space

DUP_IOU, DUP_NEAR = .25, .5
REVIEWED, TESTED = "excluded at review", "failed the free-space test"


def accepted(run):
    """{entity: (held-out fit residual, model folder, label)} for every model the run's gate accepted; the gate's voxel;
    and every label the run lists (its skipped and rejected objects included)."""
    manifest = json.loads((run / "manifest.json").read_text())
    out = {}
    for o in manifest["objects"]:
        folder = run / "models" / o["entityId"]
        if o.get("accepted") and (folder / "model.glb").exists():
            v = json.loads((folder / "validation.json").read_text())
            assert v["accepted_source_consistency"] and not v.get("rejectionReasons"), (run, o["entityId"])
            out[o["entityId"]] = (v["fitResidualNative"], folder, o["label"])
    labels = {o["entityId"]: o["label"] for o in manifest["objects"] + manifest.get("rejected", [])}
    return out, manifest["acceptance"]["max_fit_median_native"], labels  # the gate's residual limit is the depth run's voxel


def load(folder):
    import trimesh
    m = trimesh.load(folder / "model.glb", force="mesh", process=False)
    return np.asarray(m.vertices, float), np.asarray(m.faces)


def duplicate_of(box, have, voxel):
    """The first chosen model this box is the same object as (or a part of), with the numbers, else None."""
    bv, bf = box
    for other, (ov, of) in have.items():
        iou = box_iou((bv.min(0), bv.max(0)), (ov.min(0), ov.max(0)))
        reach = ((bv >= ov.min(0) - 2 * voxel) & (bv <= ov.max(0) + 2 * voxel)).all(1)  # no vertex outside can be that near
        near = float((closest(ov, of, bv[reach])[1] <= 2 * voxel).sum()) / len(bv) if reach.any() else 0.
        if iou >= DUP_IOU or near >= DUP_NEAR:
            return f"same object as or part of {other}: 3D box IoU {iou:.2f}, {near:.0%} of the box within {2 * voxel} native of it"
    return None


def merge(learned, boxes, excluded, voxel, mesh=load):
    """learned: [(generator, {entity: (residual, folder, label)})] in tie-break order; boxes: the same for the box run.
    Returns (choice {entity: generator}, folders, dropped {entity: reason})."""
    choice, folders = {}, {}
    for e in sorted(set().union(*(runs for _, runs in learned))):
        name, runs = min(((n, r) for n, r in learned if e in r), key=lambda nr: nr[1][e][0])  # min() keeps the first on a tie
        choice[e], folders[e] = name, runs[e][1]
    have = {e: mesh(folders[e]) for e in choice} if boxes else {}
    dropped = {}
    for e, (_, folder, _) in sorted(boxes.items()):
        if e in choice:
            continue
        box = mesh(folder)
        why = duplicate_of(box, have, voxel) or excluded.get(e)  # a repeat of a chosen model is that; the rest must pass the test
        if why:
            dropped[e] = why
            continue
        choice[e], folders[e], have[e] = "box", folder, box
    return choice, folders, dropped


def untested(boxes, tested):
    """{box: reason} for every box the free-space test fails, or did not measure on this mesh."""
    out = {}
    for e, (_, folder, _) in boxes.items():
        m = tested["boxes"].get(e)
        same = m and m["mesh_sha256"] == json.loads((folder / "validation.json").read_text())["mesh_sha256"]
        reasons = box_free_space.verdict(m) if same else ["not measured on this mesh"]
        if reasons:
            out[e] = TESTED + ": " + "; ".join(reasons)
    return out


def against(delivered, choice, dropped, labels):
    """Differences from a delivered merge (empty when the replay reproduces it), and the boxes whose review exclusion
    the test made."""
    theirs = json.loads((delivered / "merge.json").read_text())
    models = {p.name for p in (delivered / "models").iterdir() if p.is_dir()}
    assert models == set(theirs["choice"]), ("delivered models/ and merge.json disagree", sorted(models ^ set(theirs["choice"])))
    tell = lambda e, g: f"{labels.get(e) or theirs.get('labels', {}).get(e, '?')} ({g})"
    diff = {"missing": {e: tell(e, theirs["choice"][e]) for e in sorted(models - set(choice))},
            "extra": {e: tell(e, choice[e]) for e in sorted(set(choice) - models)},
            "changed": {e: f"{theirs['choice'][e]} -> {choice[e]}" for e in sorted(models & set(choice)) if theirs["choice"][e] != choice[e]},
            "droppedDiffers": {}}
    by_test = []
    for e in sorted(set(dropped) | set(theirs.get("boxesDropped", {}))):
        pair = [theirs.get("boxesDropped", {}).get(e), dropped.get(e)]
        if pair[0] and pair[1] and pair[0].startswith(REVIEWED) and pair[1].startswith(TESTED):
            by_test.append(e)
        elif pair[0] != pair[1]:
            diff["droppedDiffers"][e] = pair
    return {k: v for k, v in diff.items() if v}, by_test


def run(args):
    runs = {name: accepted(path) for name, path in (("recgen", args.recgen), ("sam3d", args.sam3d), ("box", args.box)) if path}
    learned = [(name, r[0]) for name, r in runs.items() if name != "box"]
    boxes, voxel, _ = runs["box"]
    labels = {e: label for r in runs.values() for e, label in r[2].items()}
    review = json.loads(args.review.read_text()).get("boxesDropped", {}) if args.review else {}
    tested = json.loads(args.box_test.read_text())
    assert Path(tested["box"]).resolve() == args.box.resolve(), ("the free-space test measured another box run", tested["box"])
    excluded = {**untested(boxes, tested), **{e: why for e, why in review.items() if why.startswith(REVIEWED)}}
    choice, folders, dropped = merge(learned, boxes, excluded, voxel)
    residual = {e: {name: (r[e][0] if e in r else None) for name, r in learned + [("box", boxes)]} for e in choice}
    result = {"rule": " ".join(__doc__.split("\n\n")[2].split()), "inputs": {name: str(p) for name, p in
                                                                    (("recgen", args.recgen), ("sam3d", args.sam3d), ("box", args.box),
                                                                     ("boxTest", args.box_test), ("review", args.review)) if p},
              "boxTestThresholds": box_free_space.THRESHOLDS,
              "voxelNative": voxel, "counts": {"models": len(choice), **{g: sum(v == g for v in choice.values()) for g in ("recgen", "sam3d", "box")},
                                               "boxesDropped": len(dropped)},
              "choice": choice, "labels": {e: labels[e] for e in choice}, "fitResidualNative": residual, "boxesDropped": dropped}
    if args.against:
        differences, by_test = against(args.against, choice, dropped, labels)
        result["against"] = {"delivered": str(args.against), "differences": differences, "reviewByTest": by_test}
    if args.output:
        (args.output / "models").mkdir(parents=True, exist_ok=False)
        for e, folder in folders.items():
            (args.output / "models" / e).mkdir()
            for f in folder.iterdir():
                if f.is_file():
                    os.link(f, args.output / "models" / e / f.name)
        (args.output / "merge.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    print(json.dumps({"counts": result["counts"], **({"against": result["against"]} if args.against else {})}, ensure_ascii=False))
    return 1 if args.against and result["against"]["differences"] else 0


def self_check():
    """Tiny runs on disk: residual choice and its tie, box only where no learned model passed, the two duplicate tests,
    a box that rests on a model (kept), a box duplicating an earlier box, the free-space test gating boxes (a failed box,
    a box measured on another mesh, a test of another run), the optional review, and the RecGen-free profile."""
    import hashlib
    import tempfile
    import trimesh

    def cube(lo, hi):
        """A box with the same number of vertices on every face, as the fitted boxes have."""
        m = trimesh.creation.box(extents=np.subtract(hi, lo)).subdivide().subdivide()
        m.apply_translation(np.add(lo, hi) / 2)
        return m

    def make(root, name, models):
        run = root / name
        objects = []
        for e, (mesh, residual) in models.items():
            (run / "models" / e).mkdir(parents=True)
            mesh.export(run / "models" / e / "model.glb")
            (run / "models" / e / "validation.json").write_text(json.dumps({"accepted_source_consistency": True, "rejectionReasons": [], "fitResidualNative": residual,
                                                                            "mesh_sha256": hashlib.sha256((run / "models" / e / "model.glb").read_bytes()).hexdigest()}))
            objects.append({"entityId": e, "label": f"thing {e[-1]}", "accepted": True})
        (run / "models" / "object-rej").mkdir(parents=True)  # a rejected attempt leaves a folder but is never taken
        (run / "models" / "object-rej" / "model.glb").write_bytes(b"")
        objects.append({"entityId": "object-rej", "label": "rejected", "accepted": False})
        (run / "manifest.json").write_text(json.dumps({"objects": objects, "acceptance": {"max_fit_median_native": .01}}))
        return run

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        table = cube([0, 0, 0], [1, .5, 1])
        recgen = make(root, "recgen", {"object-a": (table, .004), "object-b": (cube([3, 0, 0], [4, 1, 1]), .003),
                                       "object-c": (cube([6, 0, 0], [7, 1, 1]), .002)})
        sam3d = make(root, "sam3d", {"object-a": (table, .003), "object-b": (cube([3, 0, 0], [4, 1, 1]), .003),
                                     "object-d": (cube([9, 0, 0], [10, 1, 1]), .005)})
        boxes = make(root, "box", {
            "object-a": (cube([0, 0, 0], [1, .5, 1]), .001),          # has a learned model: never a box
            "object-e": (cube([.1, .1, .1], [.9, .45, .9]), .002),    # inside the table: IoU 0.45
            "object-f": (cube([6.2, .995, .2], [6.8, 1.015, .8]), .002),  # a thin slab on object-c's top face: all of it near c
            "object-g": (cube([.2, .5, .2], [.8, .9, .8]), .002),     # resting on the table: touches, but only its bottom face
            "object-h": (cube([20, 0, 0], [21, 1, 1]), .002),         # alone: kept
            "object-i": (cube([20.05, 0, 0], [21, 1, 1]), .002),      # the same thing again: duplicate of the earlier box h (fails the test too)
            "object-j": (cube([30, 0, 0], [31, 1, 1]), .002),         # alone, passes the test, but excluded at review
            "object-k": (cube([40, 0, 0], [41, 1, 1]), .002),         # alone, fails the free-space test (and excluded at review)
            "object-m": (cube([50, 0, 0], [51, 1, 1]), .002)})        # alone, but the test measured an earlier mesh
        clean = {"views": {"object:1:0": {"judged": 100, "overhang": 0, "foreign": 0, "occluded": 0}}, "sourceFaceBacking": 1.}
        sha = lambda e: json.loads((boxes / "models" / e / "validation.json").read_text())["mesh_sha256"]
        measured = {e: {**clean, "mesh_sha256": sha(e)} for e in ("object-e", "object-f", "object-g", "object-h", "object-i", "object-j")}
        measured["object-k"] = {**clean, "mesh_sha256": sha("object-k"), "views": {"object:1:0": {**clean["views"]["object:1:0"], "overhang": 60}}}
        measured["object-m"] = {**clean, "mesh_sha256": "0" * 64}
        measured["object-i"] = measured["object-k"] | {"mesh_sha256": sha("object-i")}  # a duplicate that fails the test stays a duplicate
        (root / "box-test.json").write_text(json.dumps({"box": str(boxes), "boxes": measured}))
        review = root / "review.json"
        review.write_text(json.dumps({"boxesDropped": {"object-j": "excluded at review: sticks out", "object-k": "excluded at review: sticks out",
                                                       "object-x": "same object as or part of y"}}))
        args = argparse.Namespace(recgen=recgen, sam3d=sam3d, box=boxes, box_test=root / "box-test.json", review=review, output=root / "merged", against=None)
        assert run(args) == 0
        got = json.loads((root / "merged" / "merge.json").read_text())
        assert got["choice"] == {"object-a": "sam3d", "object-b": "recgen", "object-c": "recgen", "object-d": "sam3d",
                                 "object-g": "box", "object-h": "box"}, got["choice"]
        d = got["boxesDropped"]
        assert set(d) == {"object-e", "object-f", "object-i", "object-j", "object-k", "object-m"}, d
        assert "of object-a: 3D box IoU 0.4" in d["object-e"] and "of object-c" in d["object-f"] and "of object-h" in d["object-i"], d
        assert d["object-j"].startswith(REVIEWED) and d["object-k"].startswith(REVIEWED) and "IoU 0.00" in d["object-f"], d
        assert d["object-m"] == TESTED + ": not measured on this mesh", d
        # without the review the test alone drops k (the review's decision, by the test) but not j, which the review alone dropped
        alone = argparse.Namespace(**{**vars(args), "review": None, "output": root / "alone", "against": root / "merged"})
        assert run(alone) == 1
        replay = json.loads((root / "alone" / "merge.json").read_text())["against"]
        assert replay["differences"] == {"extra": {"object-j": "thing j (box)"}, "droppedDiffers": {"object-j": ["excluded at review: sticks out", None]}}, replay
        assert replay["reviewByTest"] == ["object-k"], replay
        assert json.loads((root / "alone" / "merge.json").read_text())["boxesDropped"]["object-k"].startswith(TESTED + ": overhang 60%")
        try:
            run(argparse.Namespace(**{**vars(args), "box": sam3d, "output": None}))
            raise RuntimeError("a free-space test of another box run was used")
        except AssertionError:
            pass
        assert sorted(p.name for p in (root / "merged" / "models").iterdir()) == sorted(got["choice"])
        assert os.path.samefile(root / "merged" / "models" / "object-a" / "model.glb", sam3d / "models" / "object-a" / "model.glb")
        # replaying against itself is clean; the RecGen-free profile loses c (RecGen only), b falls to SAM 3D, and the slab f
        # on c is no longer anyone's duplicate
        assert run(argparse.Namespace(**{**vars(args), "output": None, "against": root / "merged"})) == 0
        free = argparse.Namespace(**{**vars(args), "recgen": None, "output": root / "free", "against": root / "merged"})
        assert run(free) == 1
        diff = json.loads((root / "free" / "merge.json").read_text())["against"]["differences"]
        assert diff["missing"] == {"object-c": "thing c (recgen)"} and diff["extra"] == {"object-f": "thing f (box)"}, diff
        assert diff["changed"] == {"object-b": "recgen -> sam3d"}, diff
    print("merge_object_models self-check: passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recgen", type=Path, help="complete_video_objects.py run with --generator recgen (omit for the commercial profile)")
    p.add_argument("--sam3d", type=Path, help="run with --generator sam3d")
    p.add_argument("--box", type=Path, help="run with --generator box")
    p.add_argument("--box-test", type=Path, help="box_free_space.py's box-test.json of the --box run: only a box that passes it is taken")
    p.add_argument("--review", type=Path, help="optional: a merge.json whose eye-review exclusions of boxes also apply")
    p.add_argument("--output", type=Path, help="new folder: merge.json and hard-linked models/")
    p.add_argument("--against", type=Path, help="a delivered merged folder to compare with; exit 1 on any difference")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        return self_check()
    assert args.sam3d and args.box and args.box_test, "--sam3d, --box and --box-test are required"
    sys.exit(run(args))


if __name__ == "__main__":
    main()
