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
  python scripts/prepare_video_clip.py --full-video CLIP_DIR   # the same frames uncropped, for the report's video panel
  python scripts/prepare_video_clip.py --fov-of CLIP_DIR --frames A:B --output NEW_DIR   # MoGe-3 on 15 frames of one shot
  python scripts/prepare_video_clip.py --derive CLIP_DIR --frames 0:B --fov-deg F --name N --output NEW_DIR
  python scripts/prepare_video_clip.py --self-check

--derive: the parent's frames 0..B-1 (hard links, so frame i stays frame i) with K from a shot's own field of view; the
playback and uncropped MP4s are written by the same functions as any clip (samsclub-337-a2 was made this way by hand).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
W, H = 640, 480  # the raster droid_room.prepare_image and mono_room's tum raster require
FOV_FRAMES = 5   # frames MoGe-3 looks at when the focal length is estimated
FOV_SHOT_FRAMES = 15  # --fov-of: frames spread over one shot (round(linspace(A, B-1, 15)), as samsclub-337-a2/shot-fov.py)


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


def moge_fov(picks):
    """MoGe-3's horizontal FoV of each given frame file; one Modal L4 container, calls in sequence."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modal_apps"))
    import moge3_app
    with moge3_app.app.run():
        model = moge3_app.MoGe3()
        return [model.infer.remote(p.read_bytes())["fov_x_deg"] for p in picks]


def estimate_fov(out):
    """Median horizontal FoV MoGe-3 predicts on a few frames spread over the clip; one Modal L4 call per frame."""
    frames = sorted((out / "rgb").glob("*.png"))
    picks = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, FOV_FRAMES)]
    per_frame = moge_fov(picks)
    return {"source": "moge3_median_fov", "per_frame_deg": [round(v, 2) for v in per_frame],
            "frames": [p.name for p in picks], "fov_x_deg": float(np.median(per_frame)),
            "spread_deg": float(np.ptp(per_frame))}


def data_rows(clip_dir):
    return [line for line in (clip_dir / "rgb.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]


def shot_picks(span, count=FOV_SHOT_FRAMES):
    """Clip frames MoGe-3 looks at on the shot [a, b): round(linspace(a, b - 1, count))."""
    a, b = span
    return [int(round(i)) for i in np.linspace(a, b - 1, count)]


def fov_of(clip_dir, span, out):
    """fov.json: MoGe-3's FoV on 15 frames of one shot of the clip (the schema of samsclub-337-a2/shot-fov.json)."""
    out.mkdir(parents=True, exist_ok=True)
    rows = data_rows(clip_dir)
    picks = shot_picks(span)
    per_frame = moge_fov([clip_dir / rows[i].split()[1] for i in picks])
    result = {"model": "Ruicheng/moge-3-vitl (moge3_app.MoGe3, L4)", "clip": str(clip_dir), "shot": list(span),
              "frames": [{"frame": i, "file": rows[i].split()[1], "fov_x_deg": f} for i, f in zip(picks, per_frame)],
              "median_fov_x_deg": float(np.median(per_frame)), "mean": float(np.mean(per_frame)), "spread_deg": float(np.ptp(per_frame)),
              "p25_p75": np.percentile(per_frame, [25, 75]).tolist()}
    (out / "fov.json").write_text(json.dumps(result, indent=1))
    return result


def derive(parent, span, fov_deg, name, out):
    """A clip of the parent's frames 0..B-1 (hard links: frame i stays frame i, so every run on the parent lines up) with K
    from one shot's own field of view; source-rgb.mp4 and source-full.mp4 come from playback() and full_video()."""
    a, b = span
    assert a == 0, "a derived clip starts at the parent's frame 0, so clip frame i is parent frame i"
    assert not out.exists(), f"{out} exists; clips are never overwritten"
    rows = data_rows(parent)
    assert 0 < b <= len(rows), f"--frames 0:{b} outside the parent's {len(rows)} frames"
    rows = rows[:b]
    (out / "rgb").mkdir(parents=True)
    for row in rows:
        os.link(parent / row.split()[1], out / row.split()[1])
    header = [line for line in (parent / "rgb.txt").read_text().splitlines() if line.startswith("#")]
    header.insert(2, f"# derived from {parent.name}: its frames 0-{b - 1}, K from a {fov_deg} deg field of view")
    (out / "rgb.txt").write_text("\n".join(header + rows) + "\n")
    clip = json.loads((parent / "clip.json").read_text())
    fps = clip["source"]["fps"]
    clip["name"], clip["dataset"], clip["K"] = name or out.name, str(out), intrinsics(fov_deg)
    clip["playback"] = playback(out, fps)
    clip["source"].update(end_s=round(clip["source"]["start_s"] + b / fps, 6), frames=b, requested_frames=b)
    clip["intrinsics"] = {"source": "stated_for_this_shot", "fov_x_deg": fov_deg, "replaces": {"clip": parent.name, **clip["intrinsics"]}}
    clip["shot_of"] = {"clip": parent.name, "frames": [0, b - 1], "why": f"frames 0-{b - 1} of {parent.name} (frame i here is frame i there) with K of this shot"}
    clip["limitations"][0] = "intrinsics are not a calibration: the field of view estimated on this shot (--fov-deg)"
    (out / "clip.json").write_text(json.dumps(clip, indent=1))
    return {"clip": clip["name"], "frames": b, "K": clip["K"], "full": full_video(out)}


def full_video(clip_dir, max_width=1280):
    """source-full.mp4 + source-full.json: the clip's frames from the source video without the 4:3 crop.

    The report shows this instead of the cropped clip; source-full.json says where the cropped raster sits in it, so the
    outlines drawn on the 640x480 raster land on the same pixels. MP4 frame i is clip frame i, as in source-rgb.mp4.
    """
    import cv2
    clip = json.loads((clip_dir / "clip.json").read_text())
    source = clip["source"]
    video, meta = clip_dir / "source-full.mp4", clip_dir / "source-full.json"
    assert not video.exists() and not meta.exists(), f"{video} exists; clips are never overwritten"
    width, height = source["source_wh"]
    scale = min(1., max_width / width)
    size = (int(round(width * scale)) // 2 * 2, int(round(height * scale)) // 2 * 2)
    cap = cv2.VideoCapture(source["video"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, source["first_frame"])
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"avc1"), source["fps"], size)
    assert writer.isOpened(), "H.264 encoder unavailable"
    written = 0
    while written < source["frames"]:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if scale < 1 else frame)
        written += 1
    cap.release(), writer.release()
    assert written == source["frames"], f"read {written} of {source['frames']} clip frames"
    x, y, w, h = source["crop_xywh"]
    record = {"video": video.name, "width": size[0], "height": size[1], "fps": source["fps"], "frames": written,
              "raster_wh": [W, H], "raster_in_video_xywh": [x * size[0] / width, y * size[1] / height, w * size[0] / width, h * size[1] / height],
              "frame_mapping": "MP4 frame i is clip frame i (the same source frames as source-rgb.mp4, not cropped)"}
    meta.write_text(json.dumps(record, indent=1))
    return record


def run(args):
    out = args.output or ART / "data/clips" / args.name
    assert not out.exists(), f"{out} exists; clips are never overwritten"
    facts = extract(args.video, args.start, args.end, out)
    fov = ({"source": "stated", "fov_x_deg": args.fov_deg} if args.fov_deg else estimate_fov(out))
    digest = hashlib.sha256()
    with open(args.video, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            digest.update(block)
    name = args.name or out.name  # droid_room registers clips by this name: the directory's own name is unique
    clip = {"name": name, "dataset": str(out), "K": intrinsics(fov["fov_x_deg"]), "D": [0.] * 5, "raster": "tum",
            "playback": playback(out, facts["fps"]),
            "source_wh": [W, H], "calibrated": False,
            "source": {"video": str(args.video), "video_sha256": digest.hexdigest(), "start_s": args.start, "end_s": args.end, **facts},
            "intrinsics": fov,
            "limitations": ["intrinsics are not a calibration: " + ("stated by the operator" if args.fov_deg else "estimated by MoGe-3 from a few frames"),
                            "lens distortion assumed zero",
                            "centre-cropped to 4:3: the left and right edges of the source frame are not used",
                            "no ground truth: camera and depth accuracy on this clip cannot be measured"]}
    (out / "clip.json").write_text(json.dumps(clip, indent=1))
    print(json.dumps({"clip": name, "frames": facts["frames"], "fps": round(facts["fps"], 3), "crop": facts["crop_xywh"],
                      "fov_x_deg": round(fov["fov_x_deg"], 2), "fov_spread_deg": fov.get("spread_deg"), "K": [round(v, 1) for v in clip["K"]]}))


def self_check():
    assert crop_box(1280, 720) == (160, 0, 960, 720), crop_box(1280, 720)   # 16:9 loses the sides
    assert crop_box(3840, 2160) == (480, 0, 2880, 2160)
    assert crop_box(640, 480) == (0, 0, 640, 480)                             # already 4:3
    assert crop_box(1080, 1920) == (0, 555, 1080, 810)                        # portrait loses top and bottom
    fx = intrinsics(90.)[0]
    assert abs(fx - 320.) < 1e-9, fx                                          # 90 deg across 640 px -> fx = 320
    assert abs(np.degrees(2 * np.arctan(W / 2 / intrinsics(62.)[0])) - 62.) < 1e-9
    assert shot_picks((0, 420))[:3] == [0, 30, 60] and shot_picks((0, 420))[-1] == 419, "samsclub-337-a2/shot-fov.py's 15 frames"
    assert shot_picks((383, 750)) == [383, 409, 435, 461, 488, 514, 540, 566, 592, 618, 644, 671, 697, 723, 749], "Walmart fov-check.json's frames"
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
        (Path(folder) / "clip/clip.json").write_text(json.dumps({"source": {"video": str(path), **facts}}))
        full = full_video(Path(folder) / "clip")
        cap = cv2.VideoCapture(str(Path(folder) / "clip/source-full.mp4")); ok, first = cap.read(); cap.release()
        assert full["frames"] == 10 and first.shape == (720, 1280, 3) and full["raster_in_video_xywh"] == [160., 0., 960., 720.], full
        assert first[:, 160 + 5 * 10 + 3].mean() > 200, "full frame 0 is source frame 5 (0.5 s), uncropped" 
    print("video clip check passed: 16:9, 4K, 4:3 and portrait crop to 4:3; focal length matches the stated field of view; "
          "frames and timestamps cover exactly the requested seconds; the playback MP4 has one frame per clip frame; "
          "the uncropped video holds the same frames and places the raster at the crop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fov-deg", type=float, help="stated horizontal field of view of the CROPPED 4:3 frame; omit to estimate with MoGe-3")
    parser.add_argument("--full-video", type=Path, help="an existing clip directory: write its frames uncropped as source-full.mp4")
    parser.add_argument("--fov-of", type=Path, help="an existing clip directory: MoGe-3 FoV on 15 frames of the shot --frames A:B, written to --output/fov.json")
    parser.add_argument("--derive", type=Path, help="parent clip directory: a new clip of its frames --frames 0:B with K from --fov-deg")
    parser.add_argument("--frames", type=lambda s: tuple(int(v) for v in s.split(":")), metavar="A:B", help="with --fov-of / --derive: clip frames A..B-1")
    a = parser.parse_args()
    if a.self_check:
        self_check()
    elif a.full_video:
        print(json.dumps(full_video(a.full_video)))
    elif a.fov_of:
        print(json.dumps({k: v for k, v in fov_of(a.fov_of, a.frames, a.output).items() if k != "frames"}))
    elif a.derive:
        print(json.dumps(derive(a.derive, a.frames, a.fov_deg, a.name, a.output)))
    else:
        run(a)
