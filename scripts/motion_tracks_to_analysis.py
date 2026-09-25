"""Motion tracks -> the per-frame analysis mono_room.py dynamic lifts into the dynamic layer (one surface per object per view).

Dynamic = moved OR is a kind of thing that moves. A motion track counts only if the run's track-level motion check kept it;
a track of the run's text prompt (e.g. "person", "forklift") counts whether or not it moved, because a person standing
still while the camera pans shows no motion at all (fr1/room) and is still the thing a safety rule is about.

Fusion only needs the union of what moved (assemble_dynamic_masks.py); the dynamic layer needs each object on its own.
Each retained track of each sam3_motion_tracks.py window becomes one entity; where windows overlap, each frame is taken
from the window whose centre is nearer, so the same person is not drawn twice. Identity is the tracker session's
(a person in two windows gets two ids) unless --stitched gives stitch_track_windows.py's identities, which join a
person across windows and fold a window's duplicate tracks; nothing here decides what moves.

  python scripts/motion_tracks_to_analysis.py --droid-run RUN --tracks RUN_A RUN_B ... [--stitched DIR] --output NEW_DIR
  python scripts/motion_tracks_to_analysis.py --self-check
"""
import argparse
import io
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_apps"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from assemble_dynamic_masks import to_source  # noqa: E402


def owner_of(windows, frame):
    """Index of the window that holds this frame and whose centre is nearest; None if no window holds it."""
    holding = [n for n, (first, last) in enumerate(windows) if first <= frame < last]
    return min(holding, key=lambda n: abs(frame - (windows[n][0] + windows[n][1]) / 2)) if holding else None


def redundant(mask, others, overlap=.5):
    """A motion track mask that is mostly another (text-prompted) track's mask in the same frame: the same thing, keep the named one."""
    return any((mask & o).sum() >= overlap * min(mask.sum(), o.sum()) for o in others)


def identity_of(stitched):
    """(window, "stage/id") -> stitched identity, following each window's folds; None without a stitch."""
    if stitched is None:
        return lambda n, track: None
    report = json.loads((stitched / "stitched.json").read_text())
    folds = [w.get("folded_into_text", {}) for w in report["windows"]]
    return lambda n, track: report["identities"].get(f"w{n}:{folds[n].get(track, track)}")


def run(args):
    import cv2
    import mono_room as M
    M.use_clip(args.droid_run)
    runs = [(t, json.loads((t / "tracks.json").read_text())) for t in args.tracks]
    windows = [tuple(state["frames"]) for _, state in runs]
    identity = identity_of(args.stitched)
    (args.output / "masks").mkdir(parents=True, exist_ok=False)
    frames, kept_ids, pixels = {}, [], {}
    for n, (path, state) in enumerate(runs):
        kept = {("motion", o["object"]) for o in state.get("objects", {}).get("motion", []) if o["kept"]}  # the track-level motion check of that run, as recorded
        found = np.load(io.BytesIO((path / "tracks.npz").read_bytes()))
        stages = json.loads(str(found["report"]))["stages"]
        first = windows[n][0]
        for key in found.files:
            stage = key.split("/")[0]
            if stage not in ("motion", "text") or "shape" not in stages.get(stage, {}):
                continue
            _, local, ident = key.split("/")
            frame, ident = first + int(local) * state.get("stride", 1), int(ident)  # npz keys are tracker indices; with a stride only every Nth source frame has a mask
            if (stage == "motion" and ("motion", ident) not in kept) or owner_of(windows, frame) != n:
                continue
            h, w = stages[stage]["shape"]
            mask = np.unpackbits(found[key])[:h * w].reshape(h, w).astype(bool)
            if mask.sum() < 200:
                continue
            entity = identity(n, f"{stage}/{ident}") or f"{stage}-{path.name.rsplit('-', 1)[-1]}-{ident}"
            if any(o["entityId"] == entity for o in frames.get(frame, [])):
                continue  # a folded duplicate of a track already drawn on this frame
            pixels.setdefault(frame, []).append((stage, mask))
            name = f"{frame:05d}-{entity}.png"
            cv2.imwrite(str(args.output / "masks" / name), to_source(mask, M.CALIBRATION, M.RASTER).astype(np.uint8) * 255)
            frames.setdefault(frame, []).append({"entityId": entity, "maskUrl": f"masks/{name}", "label": "moving object" if stage == "motion" else state["text"],
                                                 "sourceLabel": None if stage == "motion" else state["text"],
                                                 "source": "motion-seeded SAM 3.1 track" if stage == "motion" else f"SAM 3.1 text prompt '{state['text']}'"})
            if entity not in kept_ids:
                kept_ids.append(entity)
    for frame, objects in frames.items():  # a mover also found by the text prompt appears once, under its name
        named = [m for stage, m in pixels[frame] if stage == "text"]
        frames[frame] = [o for o, (stage, m) in zip(objects, pixels[frame]) if stage == "text" or not redundant(m, named)]
    kept_ids = [e for e in kept_ids if any(o["entityId"] == e for objects in frames.values() for o in objects)]
    analysis = {"method": "motion_masks.py cue -> sam3_motion_tracks.py (SAM 3.1 instance tracks, no text prompt), retained tracks only",
                "tracks": [str(t) for t in args.tracks], "entities": kept_ids,
                "frames": [{"sourceFrame": f, "objects": objects} for f, objects in sorted(frames.items())]}
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(json.dumps({"entities": kept_ids, "frames_with_objects": len(frames), "object_masks": sum(map(len, frames.values()))}))


def self_check():
    windows = [(0, 480), (380, 859)]
    assert owner_of(windows, 100) == 0 and owner_of(windows, 800) == 1 and owner_of(windows, 900) is None
    assert owner_of(windows, 420) == 0 and owner_of(windows, 440) == 1, "the overlap is split where the centres are equally far"
    a, b, c = (np.zeros((10, 10), bool) for _ in range(3))
    a[2:8, 2:8], b[3:8, 3:8], c[0:2, 0:2] = True, True, True
    assert redundant(a, [b]) and not redundant(c, [b]), "a mover inside a named track is dropped; a separate one is kept"
    import tempfile
    with tempfile.TemporaryDirectory() as folder:
        (Path(folder) / "stitched.json").write_text(json.dumps({"windows": [{"folded_into_text": {"text/5": "text/1"}}, {"folded_into_text": {}}],
                                                               "identities": {"w0:text/1": "p", "w1:text/3": "p"}}))
        identity = identity_of(Path(folder))
        assert identity(0, "text/5") == identity(1, "text/3") == "p" and identity(1, "text/9") is None, "stitched ids follow folds and joins"
    print("motion track analysis check passed: each frame comes from one window, split between window centres; named tracks win")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--tracks", type=Path, nargs="+")
    parser.add_argument("--stitched", type=Path, help="stitch_track_windows.py output of the same --tracks, in the same order")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
