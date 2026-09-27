"""One set of complete object models per scene out of the per-generator runs of complete_video_objects.py.

The delivered sets (22 / 67 / 40) were merged by hand-run scratch scripts; this is that rule as code, so a replay of the
same runs gives the same set and a changed profile (no RecGen, say) gives its own number without anyone choosing:

  1. a learned model that passed the gate beats a box, a box beats nothing;
  2. where both learned generators passed, the lower held-out fit residual (fitResidualNative) wins; a tie goes to the
     generator listed first (RecGen);
  3. a box is only for an object no learned model passed, and boxes are off unless an eye review of them is given
     (--review): the per-pixel free-space test that would replace the review is below its acceptance (plan section 5:
     boxes stay off until it passes). With a review, a box is dropped as the same object as, or a part of, a model
     already chosen (learned models first, then earlier boxes, in entity order) when their 3D bounding boxes overlap by
     IoU >= DUP_IOU or >= DUP_NEAR of the box's vertices lie within 2 voxels of that model's surface; else it is taken
     only when the review approves that entity under the sha256 of these model.glb bytes (boxesApproved), and dropped
     when the review excluded it (boxesDropped reasons starting 'excluded at review'), does not approve it ('not
     reviewed') or approved other bytes;
  4. box_free_space.py's test (--box-test, measured on that box run for the same mesh bytes) is recorded for every box
     under boxTest (empty = passes); it admits and drops nothing.

Model folders are hard-linked, nothing is regenerated. --against compares with a delivered merge (entity sets, the
choice per entity, the dropped boxes) and exits 1 on any difference. The eye reviews behind the delivered merges 303 are
docs/phase2/box-review-303/SCENE.json: with them the replay of runs 231...303 gives 22 / 67 / 40, entity for entity;
without a review 20 / 46 / 31 (M).

  python scripts/merge_object_models.py [--recgen R] --sam3d S --box B --box-test BOX_TEST_JSON [--review REVIEW_JSON] [--output NEW_DIR] [--against DELIVERED_DIR]
  python scripts/merge_object_models.py --self-check
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from complete_video_objects import box_iou, closest  # the same IoU and surface distance the delivered merge used
import box_free_space

DUP_IOU, DUP_NEAR = .25, .5
REVIEWED = "excluded at review"
NOT_REVIEWED = "not reviewed: the eye review does not approve this box"
OTHER_BYTES = "not reviewed on these mesh bytes: the eye review approved a model.glb with another sha256"
BOXES_OFF = ("boxes off: no eye review given, and the free-space test is below its acceptance (held out it rejects 6 of "
             "the 10 eye-dropped boxes and keeps 27 of 32; the plan asks 10 and >= 30, m0-box-test-study)")


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


def sha256(folder):
    return hashlib.sha256((folder / "model.glb").read_bytes()).hexdigest()


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


def merge(learned, boxes, approved, excluded, voxel, mesh=load):
    """learned: [(generator, {entity: (residual, folder, label)})] in tie-break order; boxes: the same for the box run;
    approved: {entity: sha256 of the model.glb the eye review kept}; excluded: {entity: the review's reason}.
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
        why = duplicate_of(box, have, voxel) or excluded.get(e) or (  # a repeat of a chosen model is that; the rest the eye decides
            NOT_REVIEWED if e not in approved else OTHER_BYTES if approved[e] != sha256(folder) else None)
        if why:
            dropped[e] = why
            continue
        choice[e], folders[e], have[e] = "box", folder, box
    return choice, folders, dropped


def free_space(boxes, tested):
    """{box: the free-space test's reasons to reject it ([] passes)}, for every box; a box it did not measure on these
    mesh bytes, or measured under other test constants, is not tested."""
    rule = {k: tested.get("rule", {}).get(k) for k in ("slack", "erodePx", "minJudged")}
    if rule != {"slack": box_free_space.SLACK, "erodePx": box_free_space.ERODE, "minJudged": box_free_space.MIN_JUDGED}:
        return {e: [f"measured under other test constants {rule}"] for e in sorted(boxes)}
    out = {}
    for e, (_, folder, _) in sorted(boxes.items()):
        m = tested["boxes"].get(e)
        same = m and m["mesh_sha256"] == sha256(folder)
        out[e] = box_free_space.verdict(m) if same else ["not measured on these mesh bytes"]
    return out


def against(delivered, choice, dropped, labels):
    """Differences from a delivered merge; empty when the replay reproduces it."""
    theirs = json.loads((delivered / "merge.json").read_text())
    models = {p.name for p in (delivered / "models").iterdir() if p.is_dir()}
    assert models == set(theirs["choice"]), ("delivered models/ and merge.json disagree", sorted(models ^ set(theirs["choice"])))
    tell = lambda e, g: f"{labels.get(e) or theirs.get('labels', {}).get(e, '?')} ({g})"
    diff = {"missing": {e: tell(e, theirs["choice"][e]) for e in sorted(models - set(choice))},
            "extra": {e: tell(e, choice[e]) for e in sorted(set(choice) - models)},
            "changed": {e: f"{theirs['choice'][e]} -> {choice[e]}" for e in sorted(models & set(choice)) if theirs["choice"][e] != choice[e]},
            "droppedDiffers": {e: [theirs.get("boxesDropped", {}).get(e), dropped.get(e)]
                               for e in sorted(set(dropped) | set(theirs.get("boxesDropped", {})))
                               if theirs.get("boxesDropped", {}).get(e) != dropped.get(e)}}
    return {k: v for k, v in diff.items() if v}


def run(args):
    runs = {name: accepted(path) for name, path in (("recgen", args.recgen), ("sam3d", args.sam3d), ("box", args.box)) if path}
    learned = [(name, r[0]) for name, r in runs.items() if name != "box"]
    boxes, voxel, _ = runs["box"]
    labels = {e: label for r in runs.values() for e, label in r[2].items()}
    review = json.loads(args.review.read_text()) if args.review else {}
    tested = json.loads(args.box_test.read_text())
    assert Path(tested["box"]).resolve() == args.box.resolve(), ("the free-space test measured another box run", tested["box"])
    excluded = {e: why for e, why in review.get("boxesDropped", {}).items() if why.startswith(REVIEWED)}
    choice, folders, dropped = merge(learned, boxes if args.review else {}, review.get("boxesApproved", {}), excluded, voxel)
    residual = {e: {name: (r[e][0] if e in r else None) for name, r in learned + [("box", boxes)]} for e in choice}
    result = {"rule": " ".join(__doc__.split("\n\n")[2].split()), "inputs": {name: str(p) for name, p in
                                                                    (("recgen", args.recgen), ("sam3d", args.sam3d), ("box", args.box),
                                                                     ("boxTest", args.box_test), ("review", args.review)) if p},
              "boxesOff": None if args.review else BOXES_OFF,
              "voxelNative": voxel, "counts": {"models": len(choice), **{g: sum(v == g for v in choice.values()) for g in ("recgen", "sam3d", "box")},
                                               "boxesDropped": len(dropped)},
              "choice": choice, "labels": {e: labels[e] for e in choice}, "fitResidualNative": residual, "boxesDropped": dropped,
              "boxTest": {"admitsBoxes": False, "thresholds": box_free_space.THRESHOLDS, "reasons": free_space(boxes, tested)}}
    if args.against:
        result["against"] = {"delivered": str(args.against), "differences": against(args.against, choice, dropped, labels)}
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
    a box that rests on a model (kept), a box duplicating an earlier box, the review excluding boxes, a box the review
    does not approve or approved on other mesh bytes (dropped), no box at all without a review or under a merge.json
    written without one, the free-space test recorded but deciding nothing (a failed box, a box whose mesh bytes it did
    not measure, a test under each other constant, a test of another run), and the RecGen-free profile."""
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
            "object-m": (cube([50, 0, 0], [51, 1, 1]), .002)})        # alone, kept by the review; the test measured other bytes
        clean = {"views": {"object:1:0": {"judged": 100, "overhang": 0, "foreign": 0, "occluded": 0}}, "sourceFaceBacking": 1.}
        sha = lambda e: json.loads((boxes / "models" / e / "validation.json").read_text())["mesh_sha256"]
        measured = {e: {**clean, "mesh_sha256": sha(e)} for e in ("object-a", "object-e", "object-f", "object-g", "object-h", "object-i", "object-j")}
        measured["object-k"] = {**clean, "mesh_sha256": sha("object-k"), "views": {"object:1:0": {**clean["views"]["object:1:0"], "overhang": 60}}}
        stale = boxes / "models" / "object-m" / "validation.json"  # its validation.json names bytes that are not its model.glb
        stale.write_text(json.dumps({**json.loads(stale.read_text()), "mesh_sha256": "1" * 64}))
        measured["object-m"] = {**clean, "mesh_sha256": "1" * 64}
        measured["object-i"] = measured["object-k"] | {"mesh_sha256": sha("object-i")}  # a duplicate that fails the test stays a duplicate
        rule = {"slack": box_free_space.SLACK, "erodePx": box_free_space.ERODE, "minJudged": box_free_space.MIN_JUDGED}
        (root / "box-test.json").write_text(json.dumps({"box": str(boxes), "rule": rule, "boxes": measured}))
        review = root / "review.json"
        # the eye kept g, h and m; approving i and j too admits neither (a duplicate stays one, an exclusion wins)
        approved = {e: sha256(boxes / "models" / e) for e in ("object-g", "object-h", "object-i", "object-j", "object-m")}
        excluded = {"object-j": "excluded at review: sticks out", "object-k": "excluded at review: sticks out", "object-x": "same object as or part of y"}
        review.write_text(json.dumps({"boxesApproved": approved, "boxesDropped": excluded}))
        args = argparse.Namespace(recgen=recgen, sam3d=sam3d, box=boxes, box_test=root / "box-test.json", review=review, output=root / "merged", against=None)
        assert run(args) == 0
        got = json.loads((root / "merged" / "merge.json").read_text())
        assert got["choice"] == {"object-a": "sam3d", "object-b": "recgen", "object-c": "recgen", "object-d": "sam3d",
                                 "object-g": "box", "object-h": "box", "object-m": "box"}, got["choice"]
        d = got["boxesDropped"]
        assert set(d) == {"object-e", "object-f", "object-i", "object-j", "object-k"} and got["boxesOff"] is None, d
        assert "of object-a: 3D box IoU 0.4" in d["object-e"] and "of object-c" in d["object-f"] and "of object-h" in d["object-i"], d
        assert d["object-j"].startswith(REVIEWED) and d["object-k"].startswith(REVIEWED) and "IoU 0.00" in d["object-f"], d
        # the test is recorded for every box and decides nothing: m, which it could not measure, is kept by the review
        tested = got["boxTest"]["reasons"]
        assert {e for e, why in tested.items() if why} == {"object-i", "object-k", "object-m"} and tested["object-k"][0].startswith("overhang 60%"), tested
        assert tested["object-m"] == ["not measured on these mesh bytes"] and not got["boxTest"]["admitsBoxes"], tested
        for key, other in (("slack", .1), ("erodePx", 3), ("minJudged", 10)):
            other_rule = {"box": str(boxes), "rule": {**rule, key: other}, "boxes": measured}
            assert all(why[0].startswith("measured under other test constants") for why in free_space(accepted(boxes)[0], other_rule).values()), key
        # the review admits per entity and per mesh bytes: a box it leaves out (g) is not reviewed, and a box approved on
        # other bytes (m, under the sha256 its stale validation.json names) is not reviewed on these; both drop
        for name, approve, lost in (("omits-g", {e: s for e, s in approved.items() if e != "object-g"}, {"object-g": NOT_REVIEWED}),
                                    ("other-bytes", {**approved, "object-m": "1" * 64}, {"object-m": OTHER_BYTES})):
            (root / f"{name}.json").write_text(json.dumps({"boxesApproved": approve, "boxesDropped": excluded}))
            assert run(argparse.Namespace(**{**vars(args), "review": root / f"{name}.json", "output": root / name})) == 0
            partial = json.loads((root / name / "merge.json").read_text())
            assert partial["choice"] == {e: g for e, g in got["choice"].items() if e not in lost} and partial["boxesDropped"] == {**d, **lost}, (name, partial)
        # without a review no box is chosen at all, whatever the test says; the test is still recorded
        alone = argparse.Namespace(**{**vars(args), "review": None, "output": root / "alone", "against": root / "merged"})
        assert run(alone) == 1
        off = json.loads((root / "alone" / "merge.json").read_text())
        assert off["choice"] == {e: g for e, g in got["choice"].items() if g != "box"} and off["boxesOff"] == BOXES_OFF and not off["boxesDropped"], off
        assert off["boxTest"] == got["boxTest"] and set(off["against"]["differences"]["missing"]) == {"object-g", "object-h", "object-m"}, off["against"]
        # a merge.json written without a review, given back as the review, approves nothing: every box that is no duplicate drops
        assert run(argparse.Namespace(**{**vars(args), "review": root / "alone" / "merge.json", "output": root / "alone-as-review"})) == 0
        echo = json.loads((root / "alone-as-review" / "merge.json").read_text())
        assert echo["choice"] == off["choice"] and {e for e, why in echo["boxesDropped"].items() if why == NOT_REVIEWED} == {
            "object-g", "object-h", "object-i", "object-j", "object-k", "object-m"}, echo["boxesDropped"]
        try:
            run(argparse.Namespace(**{**vars(args), "box": sam3d, "output": None}))
            raise RuntimeError("a free-space test of another box run was used")
        except AssertionError:
            pass
        assert sorted(p.name for p in (root / "merged" / "models").iterdir()) == sorted(got["choice"])
        assert os.path.samefile(root / "merged" / "models" / "object-a" / "model.glb", sam3d / "models" / "object-a" / "model.glb")
        # replaying against itself is clean; the RecGen-free profile loses c (RecGen only), b falls to SAM 3D, and the slab f
        # on c is no longer anyone's duplicate, yet no eye ever approved it: it stays off, not reviewed
        assert run(argparse.Namespace(**{**vars(args), "output": None, "against": root / "merged"})) == 0
        free = argparse.Namespace(**{**vars(args), "recgen": None, "output": root / "free", "against": root / "merged"})
        assert run(free) == 1
        diff = json.loads((root / "free" / "merge.json").read_text())["against"]["differences"]
        assert diff["missing"] == {"object-c": "thing c (recgen)"} and "extra" not in diff and diff["droppedDiffers"]["object-f"][1] == NOT_REVIEWED, diff
        assert diff["changed"] == {"object-b": "recgen -> sam3d"}, diff
    print("merge_object_models self-check: passed")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recgen", type=Path, help="complete_video_objects.py run with --generator recgen (omit for the commercial profile)")
    p.add_argument("--sam3d", type=Path, help="run with --generator sam3d")
    p.add_argument("--box", type=Path, help="run with --generator box")
    p.add_argument("--box-test", type=Path, help="box_free_space.py's box-test.json of the --box run: recorded for every box, decides nothing")
    p.add_argument("--review", type=Path, help="an eye review of the boxes: boxesApproved {entity: sha256 of the model.glb it kept}, boxesDropped "
                                                "{entity: 'excluded at review: ...'} (docs/phase2/box-review-303/); without it no box is taken")
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
