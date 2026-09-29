"""The judge on a finished run's mirror, off the GPU: the final cards, cameras, people, outlines and the room's TSDF points
(fetched from the layers Volume when the mirror left them there) -> judge.context -> judge.evaluate. Used to develop the
rules and the hazard judge on round-1 runs, and to build agent-audit contact sheets.

  python scripts/judge_offline.py --run RUNS/mvp-integrate-me340-005 --report mvp-me340-e84efffd-1790666683 --out DIR
"""
import argparse
import glob
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]
MODAL = "/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal"
CACHE = Path("/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad/blobs")
WARM = {"me340": ("mvp-integrate-me340-005", "mvp-me340-e84efffd-1790666683"),
        "samsclub": ("mvp-integrate-samsclub-004", "mvp-samsclub-a2-d5e0c855-1790666685"),
        "walmart": ("mvp-integrate-walmart-004", "mvp-walmart-c0761a2a-1790666653")}
RUNS = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs")


def blob(run, sha):
    """A mirror blob, else the cached copy, else fetched from the layers Volume (sha256-checked)."""
    import hashlib
    for p in (run / "mirror/blobs/sha256" / sha, CACHE / sha):
        if p.exists():
            return p
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run([MODAL, "volume", "get", "panoptes-fb-layers", f"blobs/sha256/{sha}", str(CACHE / sha)], check=True, capture_output=True)
    assert hashlib.sha256((CACHE / sha).read_bytes()).hexdigest() == sha
    return CACHE / sha


def glb_points(raw):
    """points_glb's layout -> float32 xyz (n, 3)."""
    jlen = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + jlen])
    n = gltf["accessors"][0]["count"]
    off = 20 + jlen + 8
    return np.frombuffer(raw, "<f4", count=3 * n, offset=off).reshape(-1, 3).astype(float)


class Frames:
    """ctx['frames'][source frame] decoded on demand from the report's MP4 (BGR, as the core's list)."""

    def __init__(self, mp4):
        import threading
        import cv2
        self.cap, self.cache, self.lock = cv2.VideoCapture(str(mp4)), {}, threading.Lock()  # one decoder: not thread-safe

    def __getitem__(self, k):
        import cv2
        with self.lock:
            if k not in self.cache:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
                ok, img = self.cap.read()
                if not ok:
                    raise KeyError(k)
                self.cache[k] = img
            return self.cache[k]


def load(site=None, run=None, report=None):
    """-> dict(cards, ctx, judgements (the run's last), card_shots, outline frames, report, run)."""
    from fast_report import judge
    if site:
        run, report = WARM[site]
    run = RUNS / run if not Path(run).is_absolute() else Path(run)
    pdir = run / "mirror/reports" / report / "patches"
    last = {}
    for f in sorted(glob.glob(str(pdir / "*.json"))):
        p = json.loads(Path(f).read_text())
        key = p["layer"] + (":" + p["data"].get("kind", "") if p["layer"] == "room" else "")
        last[key] = p
    cam = last["cameras"]["data"]
    people = last["people"]["data"]
    outl = last["outlines"]
    frames_out = json.loads(blob(run, outl["blobs"]["analysis"]["sha256"]).read_text())["frames"] if outl["data"].get("analysis") == "blob" else outl["data"]["frames"]
    cardp = last["object_cards"]
    cards = json.loads(blob(run, cardp["blobs"]["cards"]["sha256"]).read_text()) if cardp["data"]["cards"] == "blob" else cardp["data"]["cards"]
    room = {}
    for k, b in last["room:full"]["blobs"].items():
        if k.startswith("points-"):
            room[int(k.split("-")[1])] = glb_points(blob(run, b["sha256"]).read_bytes())
    video = last["video"]
    ctx = judge.context(cam["shots"], frames_out, people, Frames(blob(run, video["blobs"]["video"]["sha256"])), room, cam["fps"],
                        tuple(cam["shots"][0].get("source_wh", (1280, 720))), version_of={"object_cards": cardp["data"]["version"]})
    by = {x["index"]: x for x in cardp["data"]["shots"]}
    for x in ctx["shots"]:  # judge_hook's merge of the cards' own pose, floor and plumb readings
        a_ = by.get(x["index"]) or {}
        x.update(u_pose_m=a_.get("u_pose_m", x["u_pose_m"]), u_floor_m=a_.get("u_floor_m") or x["u_floor_m"], angles_usable=a_.get("angles_usable"),
                     plumb_u_deg=a_.get("plumb_u_deg"), plumb_deg=a_.get("plumb_deg"))
    return {"cards": cards, "ctx": ctx, "judgements": last.get("judgements", {}).get("data"), "card_shots": cardp["data"]["shots"],
            "outlines": frames_out, "report": report, "run": run, "cameras": cam}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--site", choices=sorted(WARM))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    from collections import Counter
    from fast_report import judge
    for site in [a.site] if a.site else sorted(WARM):
        d = load(site)
        rows = judge.evaluate(d["cards"], d["ctx"])
        by = {}
        for r in rows:
            by.setdefault(r["check"], Counter())[r["verdict"]] += 1
        print(site, len(d["cards"]), {k: dict(v) for k, v in sorted(by.items())})
        if a.out:
            a.out.mkdir(parents=True, exist_ok=True)
            (a.out / f"rules-{site}.json").write_text(json.dumps(rows, indent=1, default=str))
