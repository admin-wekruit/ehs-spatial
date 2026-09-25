"""Give the same moving object one identity across the tracking windows we cut the video into.

SAM 3 keeps identity inside one session, but a long clip does not fit in one, so we track it in overlapping
windows and each window numbers its objects from scratch. The overlap is the answer: both windows segment the
*same frames* there, so this is set matching, not re-identification — no motion model, no appearance model, no
camera-motion compensation. Mean mask IoU over the co-visible frames, then Hungarian, which is what SAM 3 does
internally between detections and masklets (assoc_iou_thresh 0.5) and what DEVA, ViP-DeepLab and Cellpose do
between clips. Pairs caught in a many-to-many high-IoU cluster are refused rather than guessed, copying SAM 3's
own guard against two objects crossing inside the overlap.

What counts as an object is the same as in motion_tracks_to_analysis.py: a motion track the run's own motion check
kept, or any track of the run's text prompt (a standing person is still a person). Inside one window a motion track
or a second text track that is mostly the same pixels as another track in most shared frames is that track, so one
person is one object before windows are compared. Window position and stride come from each run's tracks.json, never from the command line.

Known limit: a mask that merges two objects (a person pushing a cart) can join the person-only track of the next
window, because their overlap is still high.

  python scripts/stitch_track_windows.py --tracks RUN [RUN ...] --output NEW_DIR   (runs in video order)
  python scripts/stitch_track_windows.py --self-check
"""
import argparse
import json
from pathlib import Path

import numpy as np

ACCEPT, MIN_SHARED, AMBIGUOUS, SAME = .5, 5, .5, .5  # join, shared frames needed, runner-up veto, same-object-inside-a-window


class Window:
    """One tracker run: {source frame: {"stage/id": packed mask key}} for the objects it contributes, with a way to read them."""

    def __init__(self, npz, first, stride, kept_motion, shapes):
        self.npz, self.shapes, self.frames = npz, shapes, {}
        for key in npz.files:
            if key == "report":
                continue
            stage, local, ident = key.split("/")
            if stage not in ("motion", "text") or stage not in shapes:
                continue
            if stage == "motion" and kept_motion is not None and int(ident) not in kept_motion:
                continue
            self.frames.setdefault(first + int(local) * stride, {})[f"{stage}/{ident}"] = key

    def mask(self, key):
        h, w = self.shapes[key.split("/")[0]]
        return np.unpackbits(self.npz[key])[:h * w].reshape(h, w).astype(bool)

    def objects(self):
        return sorted({i for f in self.frames.values() for i in f})


def load(run_dir):
    """A Window for one sam3_motion_tracks.py run directory."""
    state = json.loads((run_dir / "tracks.json").read_text())
    npz = np.load(run_dir / "tracks.npz")
    report = json.loads(str(npz["report"]))
    shapes = {name: tuple(stage["shape"]) for name, stage in report.get("stages", {}).items() if "shape" in stage}
    kept = {o["object"] for o in state.get("objects", {}).get("motion", []) if o.get("kept")}
    return Window(npz, state["frames"][0], state.get("stride", 1), kept, shapes)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else None  # both empty: no evidence either way


def fold_duplicates(window):
    """Inside one window, a track that is mostly another track's pixels in most shared frames is that track.

    Two sources of duplicates, both seen on the Lightning windows: the motion session and the text session each
    track the same walker, and the text session switches one person's id mid-window (text/1 to frame 255, text/5
    from 256, motion/2 across both). The survivor is the named (text) track, then the longer one. Folding repeats
    until nothing changes, because a fold can make two tracks overlap that never shared a frame before.
    """
    alias = {}
    while True:
        found = _fold_once(window)
        if not found:
            break
        alias.update(found)
    for lesser in alias:  # a keeper folded in a later pass: point at the final survivor
        while alias[lesser] in alias:
            alias[lesser] = alias[alias[lesser]]
    return alias


def _fold_once(window):
    length = lambda i: sum(i in objects for objects in window.frames.values())
    order = sorted(window.objects(), key=lambda i: (i.startswith("text/"), length(i)), reverse=True)
    alias = {}
    for n, lesser in enumerate(order):
        for keeper in order[:n]:
            if keeper in alias:
                continue
            shared = same = 0
            for objects in window.frames.values():
                if lesser in objects and keeper in objects:
                    a, b = window.mask(objects[lesser]), window.mask(objects[keeper])
                    if a.any() and b.any():
                        shared += 1
                        same += (a & b).sum() >= SAME * min(a.sum(), b.sum())
            if shared >= MIN_SHARED and same >= SAME * shared:
                alias[lesser] = keeper
                break
    for objects in window.frames.values():  # the survivor's own mask wins where both exist
        for lesser, keeper in alias.items():
            if lesser in objects:
                key = objects.pop(lesser)
                objects.setdefault(keeper, key)
    return alias


def pair_scores(left, right):
    """Mean IoU per (left id, right id) over shared frames where at least one of the two masks is non-empty."""
    shared = sorted(set(left.frames) & set(right.frames))
    ids_l, ids_r = sorted({i for f in shared for i in left.frames[f]}), sorted({i for f in shared for i in right.frames[f]})
    total, seen = np.zeros((len(ids_l), len(ids_r))), np.zeros((len(ids_l), len(ids_r)))
    for frame in shared:
        here_l = {i: left.mask(k) for i, k in left.frames[frame].items()}
        here_r = {i: right.mask(k) for i, k in right.frames[frame].items()}
        for m, i in enumerate(ids_l):
            for n, j in enumerate(ids_r):
                if i in here_l and j in here_r:
                    value = iou(here_l[i], here_r[j])
                    if value is not None:
                        total[m, n] += value
                        seen[m, n] += 1
    mean = np.where(seen > 0, total / np.maximum(seen, 1), 0.)
    return ids_l, ids_r, mean, seen


def match(mean, seen):
    """Hungarian on 1 - mean IoU; refuse a pair that is not clearly the best on both sides."""
    if mean.size == 0:
        return [], []
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(1 - mean)
    joined, refused = [], []
    for m, n in zip(rows, cols):
        score, frames = float(mean[m, n]), int(seen[m, n])
        others = max([mean[m, k] for k in range(mean.shape[1]) if k != n] +
                     [mean[k, n] for k in range(mean.shape[0]) if k != m] + [0.])
        if frames < MIN_SHARED:
            refused.append((m, n, score, frames, "too few shared frames"))
        elif score < ACCEPT:
            refused.append((m, n, score, frames, "below accept threshold"))
        elif others > AMBIGUOUS * score:
            refused.append((m, n, score, frames, f"ambiguous, runner-up {others:.2f}"))
        else:
            joined.append((m, n, score, frames))
    return joined, refused


def stitch(windows, names):
    """{(window, id): identity}, joins, refusals and the in-window folds, for windows in video order."""
    folds = [fold_duplicates(w) for w in windows]
    label = {(n, i): f"{names[n]}-{i.replace('/', '-')}" for n, w in enumerate(windows) for i in w.objects()}
    joins, refusals = [], []
    for n in range(len(windows) - 1):
        ids_l, ids_r, mean, seen = pair_scores(windows[n], windows[n + 1])
        joined, refused = match(mean, seen)
        for m, k, score, frames in joined:
            label[(n + 1, ids_r[k])] = label[(n, ids_l[m])]
            joins.append({"left": f"w{n}:{ids_l[m]}", "right": f"w{n + 1}:{ids_r[k]}", "mean_iou": round(score, 3),
                          "shared_frames": frames, "identity": label[(n, ids_l[m])]})
        refusals += [{"left": f"w{n}:{ids_l[m]}", "right": f"w{n + 1}:{ids_r[k]}", "mean_iou": round(score, 3),
                      "shared_frames": frames, "reason": why} for m, k, score, frames, why in refused]
    return label, joins, refusals, folds


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    windows = [load(t) for t in args.tracks]
    label, joins, refusals, folds = stitch(windows, [t.name for t in args.tracks])
    report = {"method": "kept motion + all text tracks; motion folded into text inside a window; mean mask IoU over the overlap, Hungarian, ambiguous pairs refused",
              "accept": ACCEPT, "min_shared_frames": MIN_SHARED,
              "windows": [{"run": str(t), "objects": w.objects(), "frames": [min(w.frames), max(w.frames)] if w.frames else None,
                           "folded_into_text": folds[n]} for n, (t, w) in enumerate(zip(args.tracks, windows))],
              "joins": joins, "refused": refusals,
              "identities": {f"w{n}:{i}": name for (n, i), name in sorted(label.items())},
              "identity_count": len(set(label.values()))}
    (args.output / "stitched.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({"joins": len(joins), "refused": len(refusals), "folded": sum(len(f) for f in folds),
                      "identities_before": len(label), "identities_after": report["identity_count"]}, ensure_ascii=False))


def self_check():
    shape = (60, 80)

    def box(x, y, w=20, h=30):
        m = np.zeros(shape, bool); m[y:y + h, x:x + w] = True
        return m

    class Z:  # stand-in for a tracks.npz
        def __init__(self, frames): self.d = {f"{s}/{f}/{i}": np.packbits(m) for (s, f, i), m in frames.items()}
        files = property(lambda self: ["report"] + list(self.d))
        def __getitem__(self, k): return self.d[k]

    def window(masks, first, stride=1, kept=None):
        return Window(Z(masks), first, stride, kept, {"motion": shape, "text": shape})

    # a walker and a stander, renumbered between two windows that share frames 10-19
    left = window({**{("motion", f, 1): box(10 + f, 10) for f in range(20)}, **{("text", f, 1): box(50, 10) for f in range(20)}}, 0)
    right = window({**{("motion", f, 7): box(20 + f, 10) for f in range(20)}, **{("text", f, 4): box(50, 10) for f in range(20)}}, 10)
    label, joins, _, _ = stitch([left, right], ["a", "b"])
    assert label[(1, "motion/7")] == label[(0, "motion/1")] and label[(1, "text/4")] == label[(0, "text/1")] and len(joins) == 2, joins

    # the same person tracked by motion AND text in both windows: folded first, so the join is not vetoed as ambiguous
    person = lambda f: box(10 + f, 10)
    left = window({**{("motion", f, 1): person(f) for f in range(20)}, **{("text", f, 2): person(f) for f in range(20)}}, 0)
    right = window({**{("motion", f, 3): person(f + 10) for f in range(20)}, **{("text", f, 5): person(f + 10) for f in range(20)}}, 10)
    label, joins, refused, folds = stitch([left, right], ["a", "b"])
    assert folds == [{"motion/1": "text/2"}, {"motion/3": "text/5"}] and len(joins) == 1 and not refused, (folds, joins, refused)

    # frames where both masks are empty are no evidence; a rejected motion track takes no part; stride maps to source frames
    blank = np.zeros(shape, bool)
    left = window({**{("motion", f, 1): (blank if f % 2 else box(10, 10)) for f in range(20)},
                   **{("motion", f, 9): box(40, 10) for f in range(20)}}, 0, kept={1})
    right = window({**{("motion", f, 2): (blank if f % 2 else box(10, 10)) for f in range(20)}}, 10)  # both blank on odd frames
    label, joins, refused, _ = stitch([left, right], ["a", "b"])
    assert "motion/9" not in left.objects() and len(joins) == 1 and joins[0]["mean_iou"] == 1.0, joins
    assert sorted(window({("motion", f, 1): box(10, 10) for f in range(4)}, 100, stride=3).frames) == [100, 103, 106, 109]

    # two text ids on one person in the same window are one object, so the join is not vetoed by the copy
    left = window({**{("text", f, 1): person(f) for f in range(20)}, **{("text", f, 5): person(f) for f in range(8, 20)}}, 0)
    right = window({("text", f, 1): person(f + 10) for f in range(20)}, 10)
    label, joins, refused, folds = stitch([left, right], ["a", "b"])
    assert folds[0] == {"text/5": "text/1"} and len(joins) == 1 and not refused, (folds, joins, refused)

    # the text session switches one person's id mid-window while the motion track spans both: all three are one object
    left = window({**{("text", f, 1): person(f) for f in range(12)}, **{("text", f, 5): person(f) for f in range(12, 20)},
                   **{("motion", f, 2): person(f) for f in range(20)}}, 0)
    right = window({("text", f, 1): person(f + 10) for f in range(20)}, 10)
    label, joins, refused, folds = stitch([left, right], ["a", "b"])
    assert folds[0] == {"motion/2": "text/1", "text/5": "text/1"} and len(joins) == 1 and not refused, (folds, joins, refused)

    # a track that only turns into another inside the overlap is not a duplicate, and the join it makes ambiguous is refused
    left = window({**{("motion", f, 1): box(10, 10) for f in range(20)},
                   **{("motion", f, 2): (box(50, 10) if f < 14 else box(10, 10)) for f in range(20)}}, 0)
    right = window({("motion", f, 3): box(10, 10) for f in range(20)}, 10)
    label, joins, refused, folds = stitch([left, right], ["a", "b"])
    assert not folds[0] and not joins and "ambiguous" in refused[0]["reason"], (folds, joins, refused)

    # a window with nothing kept gives an empty result instead of a crash
    assert stitch([window({}, 0), window({}, 10)], ["a", "b"])[1] == []
    print("stitch check passed: renumbered objects rejoin; motion and text tracks of one person fold first, so the join is "
          "not vetoed; empty frames are no evidence; rejected tracks take no part; stride maps to source frames; "
          "duplicate text ids fold, also across a mid-window id switch; a track that only merges inside the overlap is refused; empty windows give an empty result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--tracks", type=Path, nargs="+", help="sam3_motion_tracks.py run directories, in video order")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
