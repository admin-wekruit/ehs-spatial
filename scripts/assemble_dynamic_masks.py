"""Motion tracks -> the --dynamic-masks directory fusion and the object map already read (SOURCEINDEX-N.png, source pixels).

Tracks come from sam3_motion_tracks.py windows on the clip's 640x480 raster; fusion rectifies mask files from source
pixels itself (mono_room.moving_mask), so each union mask is put back on the source raster here. A TUM raster is a
resize and a border crop (16 px sides, 8 px top/bottom at scale 1); the cropped border is lost, which is the same
border fusion never integrates. Nothing here decides what moves: it only changes the file layout.

  python scripts/assemble_dynamic_masks.py --droid-run RUN --tracks RUN_A RUN_B ... --output NEW_DIR [--reference MASK_DIR]
  python scripts/assemble_dynamic_masks.py --self-check
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_apps"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def to_source(mask, calibration, raster):
    """Inverse of mono_room.prepare_image for a mask (nearest); source is 640x480 for both rasters here."""
    import cv2
    if raster != "tum":
        return cv2.resize(mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST) > 0
    assert not np.any(calibration["source_distortion"]), "undistorted clips only; a distorted one needs the inverse map"
    padded = np.zeros((512, 704), np.uint8)
    padded[16:-16, 32:-32] = mask.astype(np.uint8)
    return cv2.resize(padded, (640, 480), interpolation=cv2.INTER_NEAREST) > 0


def run(args):
    import cv2
    import mono_room as M
    M.use_clip(args.droid_run)
    union = {}
    for tracks in args.tracks:
        first = json.loads((tracks / "tracks.json").read_text())["frames"][0]
        for path in sorted((tracks / "motion").glob("*.png")):
            index, mask = int(path.stem), cv2.imread(str(path), 0) > 0
            union[index] = union[index] | mask if index in union else mask
    (args.output / "masks").mkdir(parents=True, exist_ok=False)
    for index, mask in union.items():
        cv2.imwrite(str(args.output / "masks" / f"{index:05d}-0.png"), to_source(mask, M.CALIBRATION, M.RASTER).astype(np.uint8) * 255)
    report = {"tracks": [str(t) for t in args.tracks], "frames_with_mask": len(union), "mean_share_of_frame": float(np.mean([m.mean() for m in union.values()])),
              "layout": "SOURCEINDEX-0.png in source pixels, as mono_room --dynamic-masks and build_video_object_map --dynamic-masks read"}
    if args.reference:  # measured after the round trip through the source raster, exactly as fusion will see both
        pairs = [(M.moving_mask(args.output / "masks", i), M.moving_mask(args.reference, i)) for i in range(0, max(union) + 1)]
        pairs = [(a, b) for a, b in pairs if b.any()]
        report["against_reference"] = {"frames": len(pairs), "median_iou": float(np.median([(a & b).sum() / max((a | b).sum(), 1) for a, b in pairs])),
                                       "pixel_recall": sum((a & b).sum() for a, b in pairs) / max(sum(b.sum() for _, b in pairs), 1),
                                       "pixel_precision": sum((a & b).sum() for a, b in pairs) / max(sum(a.sum() for a, _ in pairs), 1),
                                       "frames_reference_empty_but_moving": sum(1 for i in union if i <= max(union) and not M.moving_mask(args.reference, i).any() and union[i].any())}
    (args.output / "dynamic-masks.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def self_check():
    import cv2
    from droid_room import prepare_image
    calibration = {"source_K_fx_fy_cx_cy": [535.4, 539.2, 320.1, 247.6], "source_distortion": [0., 0, 0, 0, 0]}
    mask = np.zeros((480, 640), bool)
    mask[100:300, 200:400] = True
    back = prepare_image(np.repeat(to_source(mask, calibration, "tum")[..., None].astype(np.uint8) * 255, 3, -1), calibration, 2)[0][..., 0] > 0
    assert (back & mask).sum() / (back | mask).sum() > .98, "a mask survives the trip to source pixels and back"
    print("dynamic mask assembly check passed: raster round trip keeps the mask")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--droid-run", type=Path)
    parser.add_argument("--tracks", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path, help="existing person masks to measure against (never an input)")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
