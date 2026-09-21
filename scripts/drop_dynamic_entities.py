"""Take the entities that are a moving person out of a finished video object map. No model call.

A static map must not hold the person who walks through the video. build_video_object_map.py --dynamic-masks keeps such
masks out from the start; this applies the same rule to a map built without it: an entity goes if at least half of its
views that have moving-entity masks lie at least half inside them. Geometry decides, never the entity's name.

  python scripts/drop_dynamic_entities.py --object-map DIR --masks ROOT --dynamic-masks DIR --droid-run RUN --output NEW_DIR
"""
import argparse
import json
import os
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "modal_apps")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("object-map", "masks", "dynamic-masks", "droid-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    import mono_room
    mono_room.use_clip(args.droid_run)
    document = json.loads((args.object_map / "object-map.json").read_text())
    covered = {int(p.name.split("-")[0]) for p in args.dynamic_masks.glob("*.png")}
    gone = []
    for entity in document["entities"]:
        votes = []
        for observation in entity["observations"]:
            label, index, instance = observation.split(":")
            if int(index) in covered:
                path = next(args.masks.glob(f"{label}-*/frame-{int(index):05d}/instance-{instance}-mask.png"))
                mask = mono_room.prepare_image(cv2.imread(str(path), cv2.IMREAD_COLOR), mono_room.CALIBRATION, 2)[0][..., 0] > 0
                votes.append((mask & mono_room.moving_mask(args.dynamic_masks, int(index))).sum() >= .5 * max(mask.sum(), 1))
        if votes and sum(votes) >= .5 * len(votes):
            gone.append({"entityId": entity["entityId"], "label": entity["label"], "views": len(entity["observations"]), "views_on_moving_entity": int(sum(votes)), "views_checked": len(votes)})
    args.output.mkdir(parents=True, exist_ok=False)
    names = {g["entityId"] for g in gone}
    document["entities"] = [e for e in document["entities"] if e["entityId"] not in names]
    document["dynamic_entities_removed"] = gone
    for item in args.object_map.iterdir():  # geometry and sheets stay where they were built
        if item.name != "object-map.json":
            os.symlink(os.path.relpath(item.resolve(), args.output.resolve()), args.output / item.name)
    (args.output / "object-map.json").write_text(json.dumps(document, indent=1, allow_nan=False))
    print(json.dumps({"removed": gone, "entities_left": len(document["entities"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
