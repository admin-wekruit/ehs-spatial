"""Motion tracks -> the per-frame analysis mono_room.py dynamic lifts into the dynamic layer (one surface per object per view).

Fusion only needs the union of what moved (assemble_dynamic_masks.py); the dynamic layer needs each object on its own.
Each retained track of each sam3_motion_tracks.py window becomes one entity; where windows overlap, each frame is taken
from the window whose centre is nearer, so the same person is not drawn twice. Identity is the tracker session's
(a person in two windows gets two ids); nothing here decides what moves.

  python scripts/motion_tracks_to_analysis.py --droid-run RUN --tracks RUN_A RUN_B ... --output NEW_DIR
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


def run(args):
    import cv2
    import mono_room as M
    M.use_clip(args.droid_run)
    runs = [(t, json.loads((t / "tracks.json").read_text())) for t in args.tracks]
    windows = [tuple(state["frames"]) for _, state in runs]
    (args.output / "masks").mkdir(parents=True, exist_ok=False)
    frames, kept_ids = {}, []
    for n, (path, state) in enumerate(runs):
        kept = {o["object"] for o in state["objects"]["motion"] if o["kept"]}  # the track-level motion check of that run, as recorded
        found = np.load(io.BytesIO((path / "tracks.npz").read_bytes()))
        h, w = json.loads(str(found["report"]))["stages"]["motion"]["shape"]
        first = windows[n][0]
        for key in found.files:
            if not key.startswith("motion/"):
                continue
            _, local, ident = key.split("/")
            frame, ident = first + int(local), int(ident)
            if ident not in kept or owner_of(windows, frame) != n:
                continue
            mask = np.unpackbits(found[key])[:h * w].reshape(h, w).astype(bool)
            if mask.sum() < 200:
                continue
            entity = f"motion-{path.name.rsplit('-', 1)[-1]}-{ident}"
            name = f"{frame:05d}-{entity}.png"
            cv2.imwrite(str(args.output / "masks" / name), to_source(mask, M.CALIBRATION, M.RASTER).astype(np.uint8) * 255)
            frames.setdefault(frame, []).append({"entityId": entity, "maskUrl": f"masks/{name}", "label": "moving object", "source": "motion-seeded SAM 3.1 track"})
            if entity not in kept_ids:
                kept_ids.append(entity)
    analysis = {"method": "motion_masks.py cue -> sam3_motion_tracks.py (SAM 3.1 instance tracks, no text prompt), retained tracks only",
                "tracks": [str(t) for t in args.tracks], "entities": kept_ids,
                "frames": [{"sourceFrame": f, "objects": objects} for f, objects in sorted(frames.items())]}
    (args.output / "analysis.json").write_text(json.dumps(analysis, indent=1))
    print(json.dumps({"entities": kept_ids, "frames_with_objects": len(frames), "object_masks": sum(map(len, frames.values()))}))


def self_check():
    windows = [(0, 480), (380, 859)]
    assert owner_of(windows, 100) == 0 and owner_of(windows, 800) == 1 and owner_of(windows, 900) is None
    assert owner_of(windows, 420) == 0 and owner_of(windows, 440) == 1, "the overlap is split where the centres are equally far"
    print("motion track analysis check passed: each frame comes from one window, split between window centres")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--tracks", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
