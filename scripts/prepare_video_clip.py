"""An ordinary video file -> a clip the reconstruction pipeline reads: frames, timestamps, camera intrinsics.

Every model in the pipeline works frame by frame, so a video has to become the same thing the research datasets ship
as: a folder of 640x480 frames, an index of timestamps (TUM's rgb.txt) and one set of intrinsics. This does that for
any video: cut [start, end), centre-crop to 4:3 (the raster DROID and the depth runs are built on), resize.

Intrinsics are the honest weak point. A phone or tour video carries no calibration, so the focal length is either
stated (--fov-deg) or estimated by MoGe-3 on a few frames of the cropped clip (median horizontal field of view). Either
way the clip records where the number came from and how far the per-frame estimates spread; nothing downstream may
treat it as a calibration. Lens distortion is assumed zero, which is wrong for wide lenses and is written down as such.

Output directory (default research-notes/phase2/data/clips/NAME) holds rgb/*.png, rgb.txt, clip.json and source-rgb.mp4
(the same frames as a CFR H.264 file, which keyframe discovery, fusion --video and the report's video panel read, as
they read the TUM clips' source-rgb.mp4); droid_room registers every clip.json it finds there, so a new video needs no
code change.

  python scripts/prepare_video_clip.py --video V --start 3572 --end 3611 --name lightning-3572 [--fov-deg 70]
  python scripts/prepare_video_clip.py --self-check
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
W, H = 640, 480  # the raster droid_room.prepare_image and mono_room's tum raster require
FOV_FRAMES = 5   # frames MoGe-3 looks at when the focal length is estimated


def crop_box(width, height):
    """Largest centred 4:3 box inside the source frame, as (x, y, w, h) in source pixels."""
    if width * 3 >= height * 4:
        w = int(round(height * 4 / 3))
        return (width - w) // 2, 0, w, height
    h = int(round(width * 3 / 4))
    return 0, (height - h) // 2, width, h


def to_raster(frame, box):
    import cv2
    x, y, w, h = box
    return cv2.resize(frame[y:y + h, x:x + w], (W, H), interpolation=cv2.INTER_AREA)


def intrinsics(fov_x_deg):
    """fx, fy, cx, cy on the 640x480 raster for a horizontal field of view; square pixels, centred principal point."""
    fx = (W / 2) / np.tan(np.radians(fov_x_deg) / 2)
    return [float(fx), float(fx), (W - 1) / 2, (H - 1) / 2]


def extract(video, start, end, out):
    """Write rgb/<t>.png and rgb.txt for [start, end) seconds; t is seconds from the start of the video file."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert fps and np.isfinite(fps) and count > 0, "the video has no readable frame rate or frame count"
    first, last = int(round(start * fps)), min(int(round(end * fps)), count)
    assert last > first, "empty interval"
    box = crop_box(width, height)
    (out / "rgb").mkdir(parents=True, exist_ok=False)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    lines, index = ["# color images", "# made by scripts/prepare_video_clip.py from an ordinary video", "# timestamp filename"], first
    while index < last:
        ok, frame = cap.read()
        if not ok:
            break
        stamp = f"{index / fps:.6f}"
        cv2.imwrite(str(out / "rgb" / f"{stamp}.png"), to_raster(frame, box))
        lines.append(f"{stamp} rgb/{stamp}.png")
        index += 1
    cap.release()
    (out / "rgb.txt").write_text("\n".join(lines) + "\n")
    return {"fps": fps, "source_wh": [width, height], "crop_xywh": list(box), "first_frame": first,
            "frames": index - first, "requested_frames": last - first}


def playback(out, fps):
    """Encode the clip's frames, in rgb.txt order, as source-rgb.mp4: MP4 frame i is data row i of rgb.txt."""
    import cv2
    rows = [line.split()[1] for line in (out / "rgb.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
    video = out / "source-rgb.mp4"
    assert not video.exists(), f"{video} exists; clips are never overwritten"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"avc1"), fps, (W, H))
    assert writer.isOpened(), "H.264 encoder unavailable"
    for row in rows:
        writer.write(cv2.imread(str(out / row)))
    writer.release()
    cap, decoded = cv2.VideoCapture(str(video)), 0
    while cap.read()[0]:
        decoded += 1
    cap.release()
    assert decoded == len(rows), f"encoded {len(rows)} frames, decoded {decoded}"
    return {"path": video.name, "codec": "H264 avc1", "fps": fps, "frames": decoded,
            "frame_mapping": "MP4 frame i is data row i of rgb.txt (zero-based, comments excluded)"}


def estimate_fov(out):
    """Median horizontal FoV MoGe-3 predicts on a few frames spread over the clip; one Modal L4 call per frame."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_apps"))
    import moge3_app
    frames = sorted((out / "rgb").glob("*.png"))
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, FOV_FRAMES)]
    with moge3_app.app.run():
        model = moge3_app.MoGe3()
        per_frame = [model.infer.remote(p.read_bytes())["fov_x_deg"] for p in picks]
    return {"source": "moge3_median_fov", "per_frame_deg": [round(v, 2) for v in per_frame],
            "frames": [p.name for p in picks], "fov_x_deg": float(np.median(per_frame)),
            "spread_deg": float(np.ptp(per_frame))}


def run(args):
    out = args.output or ART / "data/clips" / args.name
    assert not out.exists(), f"{out} exists; clips are never overwritten"
    facts = extract(args.video, args.start, args.end, out)
    fov = ({"source": "stated", "fov_x_deg": args.fov_deg} if args.fov_deg else estimate_fov(out))
    digest = hashlib.sha256()
    with open(args.video, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    clip = {"name": args.name, "dataset": str(out), "K": intrinsics(fov["fov_x_deg"]), "D": [0.] * 5, "raster": "tum",
            "playback": playback(out, facts["fps"]),
            "source_wh": [W, H], "calibrated": False,
            "source": {"video": str(args.video), "video_sha256": digest.hexdigest(), "start_s": args.start, "end_s": args.end, **facts},
            "intrinsics": fov,
            "limitations": ["intrinsics are not a calibration: " + ("stated by the operator" if args.fov_deg else "estimated by MoGe-3 from a few frames"),
                            "lens distortion assumed zero",
                            "centre-cropped to 4:3: the left and right edges of the source frame are not used",
                            "no ground truth: camera and depth accuracy on this clip cannot be measured"]}
    (out / "clip.json").write_text(json.dumps(clip, indent=1))
    print(json.dumps({"clip": args.name, "frames": facts["frames"], "fps": round(facts["fps"], 3), "crop": facts["crop_xywh"],
                      "fov_x_deg": round(fov["fov_x_deg"], 2), "fov_spread_deg": fov.get("spread_deg"), "K": [round(v, 1) for v in clip["K"]]}))


def self_check():
    assert crop_box(1280, 720) == (160, 0, 960, 720), crop_box(1280, 720)   # 16:9 loses the sides
    assert crop_box(3840, 2160) == (480, 0, 2880, 2160)
    assert crop_box(640, 480) == (0, 0, 640, 480)                             # already 4:3
    assert crop_box(1080, 1920) == (0, 555, 1080, 810)                        # portrait loses top and bottom
    fx = intrinsics(90.)[0]
    assert abs(fx - 320.) < 1e-9, fx                                          # 90 deg across 640 px -> fx = 320
    assert abs(np.degrees(2 * np.arctan(W / 2 / intrinsics(62.)[0])) - 62.) < 1e-9
    import tempfile
    import cv2
    with tempfile.TemporaryDirectory() as folder:                             # 2 s of a 10 fps 1280x720 video, cut 0.5-1.5 s
        path = Path(folder) / "v.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1280, 720))
        for n in range(20):
            frame = np.zeros((720, 1280, 3), np.uint8); frame[:, 160 + n * 10:170 + n * 10] = 255
            writer.write(frame)
        writer.release()
        facts = extract(path, .5, 1.5, Path(folder) / "clip")
        rows = [l for l in (Path(folder) / "clip/rgb.txt").read_text().splitlines() if not l.startswith("#")]
        assert facts["frames"] == 10 and len(rows) == 10 and rows[0].startswith("0.500000 rgb/0.500000.png"), (facts, rows[:2])
        image = cv2.imread(str(Path(folder) / "clip" / rows[0].split()[1]))
        assert image.shape == (H, W, 3)
        assert playback(Path(folder) / "clip", facts["fps"])["frames"] == 10          # one MP4 frame per clip frame
    print("video clip check passed: 16:9, 4K, 4:3 and portrait crop to 4:3; focal length matches the stated field of view; "
          "frames and timestamps cover exactly the requested seconds; the playback MP4 has one frame per clip frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fov-deg", type=float, help="stated horizontal field of view of the CROPPED 4:3 frame; omit to estimate with MoGe-3")
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
