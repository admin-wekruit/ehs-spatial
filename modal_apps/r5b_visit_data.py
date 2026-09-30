"""r5b (visits): ground-truth revisits on a Modal Volume (never the Mac). CPU only.

TUM RGB-D fr1 sequences share one office and one motion-capture frame: each sequence is a visit of the same site with
metric camera truth (and Kinect depth for the change oracle). ARKitScenes sequences of one venue (visit_id) are static
revisits. Everything lands on the Volume `panoptes-r5b-visit` under /data; the MP4 windows (the pipeline's input: 16:9
1280x720 centre crop, scripts/accuracy_gt.py's contract) come back small.

    modal run modal_apps/r5b_visit_data.py::arkit_meta --ids 47333932,42445448
    modal run modal_apps/r5b_visit_data.py::tum --names desk,desk2,360,plant,teddy --out LOCAL_DIR
"""
import json
import subprocess
from pathlib import Path

import modal

app = modal.App("panoptes-r5b-visit-data")
VOL = modal.Volume.from_name("panoptes-r5b-visit", create_if_missing=True)
REPO = Path(__file__).resolve().parents[1]
image = (modal.Image.debian_slim(python_version="3.11").apt_install("ffmpeg", "libgl1", "libglib2.0-0")
         .pip_install("numpy", "scipy", "opencv-python-headless", "pandas", "requests")
         .add_local_dir(REPO / "scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
         .add_local_dir(REPO / "fast_report", "/repo/fast_report", ignore=["**/__pycache__/**"]))
TUM_URL = "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_{}.tgz"
ARKIT = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1"
MAX_FRAMES = 900  # the pipeline's 30 s window at 30 fps


def encode(frames_bgr, path, fps=30.):
    """H.264 (libx264, CRF 18) from BGR frames through ffmpeg's stdin: OpenCV's wheel has no H.264 encoder on Linux."""
    h, w = frames_bgr[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames_bgr:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0, "ffmpeg failed"


@app.function(image=image, volumes={"/data": VOL}, timeout=1800, retries=0, cpu=4, memory=8192)
def tum_one(name: str, window: list | None = None):
    """Download fr1/<name> to /data/tum/<name>, write window-<a>-<b>.mp4 (source frames [a, b)) and gt.json (per frame:
    stamp, c2w (mocap, metres), K_src, depth path) like accuracy_gt.prepare; floor.json = the GT floor of that sequence.
    -> {mp4 bytes, gt summary}."""
    import sys
    import cv2
    import requests
    sys.path[:0] = ["/repo", "/repo/scripts"]
    import accuracy_gt as ag
    root = Path("/data/tum")
    seq_dir = root / f"rgbd_dataset_freiburg1_{name}"
    if not (seq_dir / "groundtruth.txt").exists():
        root.mkdir(parents=True, exist_ok=True)
        tgz = root / f"{name}.tgz"
        with requests.get(TUM_URL.format(name), stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tgz, "wb") as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        subprocess.run(["tar", "-xzf", str(tgz), "-C", str(root)], check=True)
        tgz.unlink()
        VOL.commit()
    rows = ag.tum_rows({"frames": seq_dir})
    a, b = window or [0, min(len(rows), MAX_FRAMES)]
    out = root / name
    out.mkdir(exist_ok=True)
    first = cv2.imread(str(rows[0][1]))
    sh, sw = first.shape[:2]
    y0, sc = ag.crop_params(sw, sh)
    mp4 = out / f"window-{a}-{b}.mp4"
    encode([ag.to_video(cv2.imread(str(p)), y0, sc) for _, p, _, _, _ in rows[a:b]], mp4)
    frames = [{"stamp": s, "image": str(p), "c2w": None if c is None else c.round(6).tolist(), "K_src": k.round(4).tolist(),
               "depth": str(dp) if dp is not None else None} for s, p, c, k, dp in rows]
    g = {"name": name, "kind": "tum", "source_wh": [sw, sh], "crop_y0": y0, "scale": sc, "rot180": False,
         "depth": {"wh": [640, 480], "K": list(ag.TUM_DEPTH_K), "unit_m": 1 / ag.TUM_DEPTH_SCALE, "rot180": False},
         "windows": [{"mp4": mp4.name, "source_frames": [a, b], "hold": 1, "cuts": "rule"}], "frames": frames}
    (out / "gt.json").write_text(json.dumps(g))
    try:
        floor = ag.gt_floor(out)
    except Exception as error:  # noqa: BLE001  a sequence that never sees the floor (the room's floor is the site's)
        floor = {"error": repr(error)[:300]}
    VOL.commit()
    posed = [f for f in frames[a:b] if f["c2w"]]
    return {"name": name, "mp4_name": mp4.name, "mp4": mp4.read_bytes(), "frames": len(rows), "window": [a, b], "posed_in_window": len(posed),
            "floor": floor, "gt_small": {k: v for k, v in g.items() if k != "frames"},
            "poses": [{"i": i, "stamp": f["stamp"], "c2w": f["c2w"]} for i, f in enumerate(frames)]}


@app.function(image=image, volumes={"/data": VOL}, timeout=3600, retries=0, cpu=4, memory=16384)
def arkit_raw(video_id: str, split: str = "Validation", per_window: int = 128):
    """A raw ARKitScenes capture as the round-4 GT runs had 47333932 (scripts/prepare_arkit_clip.py --raw, then
    accuracy_gt's windows: every capture frame held for 6 video frames at 30 fps, cuts 'none'). -> {mp4s, gt summary}."""
    import argparse
    import sys
    import cv2
    import requests
    sys.path[:0] = ["/repo", "/repo/scripts"]
    import accuracy_gt as ag
    import prepare_arkit_clip as pac
    root = Path("/data/arkit") / video_id
    dl = root / "download"
    dl.mkdir(parents=True, exist_ok=True)
    for asset in ("wide.zip", "wide_intrinsics.zip", "lowres_wide.traj", "lowres_depth.zip", "confidence.zip"):
        if not (dl / asset).exists():
            with requests.get(f"{ARKIT}/raw/{split}/{video_id}/{asset}", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dl / (asset + ".part"), "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
            (dl / (asset + ".part")).rename(dl / asset)
    VOL.commit()
    data, run = root / "clip", root / "cameras"
    if not (run / "prediction.npz").exists():
        pac.write_preview = lambda *a, **k: None  # its preview MP4 needs an H.264 encoder OpenCV's Linux wheel lacks; not used here
        pac.main(argparse.Namespace(raw=dl, data=data, run=run, video_id=video_id, scene=None, scenes_root=None))
        VOL.commit()
    seq = {"frames": data, "poses": run / "prediction.npz"}
    rows = [(s, p, c, k, data / "evaluation_only/lowres_depth_confident" / f"{s:.3f}.png") for s, p, c, k in ag.arkit_rows(seq)]
    first = cv2.imread(str(rows[0][1]))
    sh, sw = first.shape[:2]
    y0, sc = ag.crop_params(sw, sh)
    out = root / "gt"
    out.mkdir(exist_ok=True)
    windows, mp4s = [], {}
    for a in range(0, len(rows), per_window):
        b = min(len(rows), a + per_window)
        if b - a < 30:
            break
        path = out / f"window-{a}-{b}.mp4"
        encode([v for _, p, _, _, _ in rows[a:b] for v in [ag.to_video(cv2.imread(str(p)), y0, sc)] * ag.HOLD], path)
        windows.append({"mp4": path.name, "source_frames": [a, b], "hold": ag.HOLD, "cuts": "none"})
        mp4s[path.name] = path.read_bytes()
    frames = [{"stamp": s, "image": str(p), "c2w": c.round(6).tolist(), "K_src": k.round(4).tolist(), "depth": str(dp) if dp.exists() else None}
              for s, p, c, k, dp in rows]
    g = {"name": f"arkit{video_id}", "kind": "arkit", "source_wh": [sw, sh], "crop_y0": y0, "scale": sc, "rot180": False,
         "depth": {"wh": [256, 192], "K": "K_src x depth_w / source_w", "unit_m": .001, "rot180": False}, "windows": windows, "frames": frames}
    (out / "gt.json").write_text(json.dumps(g))
    try:
        floor = ag.gt_floor(out)
    except Exception as error:  # noqa: BLE001
        floor = {"error": repr(error)[:300]}
    VOL.commit()
    return {"name": f"arkit{video_id}", "mp4s": mp4s, "frames": len(rows), "windows": windows, "floor": floor,
            "gt_small": {k: v for k, v in g.items() if k != "frames"}, "poses": [{"i": i, "stamp": f["stamp"], "c2w": f["c2w"]} for i, f in enumerate(frames)]}


@app.local_entrypoint()
def arkit(video_id: str, out: str):
    r = arkit_raw.remote(video_id)
    d = Path(out) / r["name"]
    d.mkdir(parents=True, exist_ok=True)
    for name, b in r.pop("mp4s").items():
        (d / name).write_bytes(b)
    (d / "gt-poses.json").write_text(json.dumps(r))
    print(r["name"], r["frames"], json.dumps(r["windows"]), json.dumps(r["floor"])[:300], flush=True)


@app.function(image=image, timeout=600, retries=0)
def arkit_meta(ids: str):
    """ARKitScenes metadata rows (raw, 3dod, upsampling) of the venues (visit_id) holding these video ids."""
    import io
    import pandas as pd
    import requests
    out = {}
    for sub in ("raw", "threedod", "upsampling"):
        try:
            r = requests.get(f"{ARKIT}/{sub}/metadata.csv", timeout=60)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
        except Exception as error:  # noqa: BLE001
            out[sub] = {"error": repr(error)[:300]}
            continue
        want = [int(x) for x in ids.split(",")]
        vcol = "visit_id" if "visit_id" in df.columns else None
        rows = df[df["video_id"].isin(want)]
        visits = sorted(set(rows[vcol].tolist())) if vcol else []
        sib = df[df[vcol].isin(visits)] if vcol else rows
        out[sub] = {"columns": list(df.columns), "n": len(df), "rows": rows.to_dict("records"), "same_venue": sib.to_dict("records")}
    return out


@app.local_entrypoint()
def tum(names: str, out: str, windows: str = ""):
    """names: comma list of fr1 sequences; windows: optional 'name:a-b,...' source-frame windows."""
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    win = {w.split(":")[0]: [int(x) for x in w.split(":")[1].split("-")] for w in windows.split(",") if w}
    for r in tum_one.starmap([(n, win.get(n)) for n in names.split(",")]):
        d = o / r["name"]
        d.mkdir(exist_ok=True)
        (d / r["mp4_name"]).write_bytes(r.pop("mp4"))
        (d / "gt-poses.json").write_text(json.dumps({k: r[k] for k in ("name", "frames", "window", "posed_in_window", "floor", "gt_small", "poses")}))
        print(r["name"], r["frames"], r["window"], r["posed_in_window"], json.dumps(r["floor"])[:200], flush=True)


@app.local_entrypoint()
def meta(ids: str = "47333932,42445448"):
    print(json.dumps(arkit_meta.remote(ids), indent=1, default=str)[:20000])
