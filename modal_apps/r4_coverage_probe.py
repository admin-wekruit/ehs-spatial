"""r4/coverage probe: which box source puts a mask under the clicks rounds 2 and 3 left on 'unknown region'? One A100-80GB,
the fast report's image and volumes. For every labelled dev click (seeds 2 and 61: agent labels on rounds 2/3's sheets) the
click's pick keyframe goes through SAM 3's vision (person / floor too), every box source (fast_report.coverage.Detector), and
the SAM 3 tracker's box decode on the shared backbone features (coverage.Boxer); the shared path is checked against the
tracker's own pixel path on the first frames. Per click and source: every mask under the click (score, word, area, share on
SAM 3 floor / people, share on the pixels the round's own pick map already held); time per frame per source; GPU peaks;
contact sheets of the smallest mask under each missed / background click.

    PYTHONPATH=. .venv/bin/modal run modal_apps/r4_coverage_probe.py --out RUNS/r4-coverage-probe-001
"""
import io
import json
import sys
import time
from pathlib import Path

import modal
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent if (HERE.parent / "fast_report").is_dir() else Path("/repo")  # the container mounts the repo at /repo
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "modal_apps"), str(HERE)]
from fast_report_app import VOLUMES, image  # noqa: E402  the fast report's own image and volumes

# ultralytics (AGPL-3.0: YOLOE / YOLO11 / YOLO26) only in the probe's image, as the last layer, no deps (its opencv-python would shadow
# the headless one; torch stays E9's); the pipeline's chosen source (OWLv2, Apache-2.0) needs none of it
image = image.pip_install("polars==1.44.2", "psutil", "pyyaml", "requests", "matplotlib", "nvidia-ml-py", "cloudpickle", "filelock") \
    .run_commands("pip install --no-deps ultralytics==8.4.165 ultralytics-thop==2.2.1") if modal.is_local() else image
app = modal.App("panoptes-r4-coverage-probe")
SRCS = ("yoloe", "yoloepf", "yolo11", "yolo26", "owlv2")
PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
DEV = {"me340": ("mvp2-results/clicks/me340/{}-me340-random-s2.json", "mvp3-results/clicks/me340/{}-me340-random-s61.json"),
       "samsclub-a2": ("mvp2-results/clicks/samsclub-a2/{}-samsclub-a2-random-s2.json", "mvp3-results/clicks/samsclub-a2/{}-samsclub-a2-random-s61.json"),
       "walmart": ("mvp2-results/clicks/walmart/{}-walmart-random-s2.json", "mvp3-results/clicks/walmart/{}-walmart-random-s61.json")}
CLIPS = {"me340": "me340-165", "samsclub-a2": "samsclub-337", "walmart": "walmart-190"}


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=48 * 1024, timeout=2400, retries=0, volumes=VOLUMES)
def probe(videos: dict, clicks: dict, claimed: dict):
    import os
    import cv2
    import torch
    import torch.nn.functional as F
    import sam3_app
    from fast_report import coverage, segment
    dev = torch.device("cuda:0")
    rec, t0 = {"boot": {}}, time.perf_counter()
    lap = lambda k: rec["boot"].__setitem__(k, round(time.perf_counter() - t0, 2))  # noqa: E731
    os.makedirs(coverage.WDIR, exist_ok=True)
    from ultralytics.utils.downloads import attempt_download_asset
    import ultralytics
    for f in coverage.WEIGHTS.values():  # public release assets (ultralytics/assets), once, to the models volume
        p = Path(coverage.WDIR) / f
        if not p.exists():
            attempt_download_asset(str(p))
    rec["ultralytics"] = ultralytics.__version__
    lap("weights_s")
    from transformers import Sam3Model, Sam3Processor, Sam3TrackerProcessor
    proc = Sam3Processor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
    sam = segment.Sam3(Sam3Model.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub",
                                                 torch_dtype=torch.bfloat16).eval().to(dev), proc, dev)
    mem = {"sam3_detector": torch.cuda.memory_allocated(dev)}
    boxer = coverage.Boxer(dev, sam3_app.MODEL_ID, sam3_app.REVISION, "/v/sam3/huggingface/hub", keep_backbone=True)
    tproc = Sam3TrackerProcessor.from_pretrained(sam3_app.MODEL_ID, revision=sam3_app.REVISION, cache_dir="/v/sam3/huggingface/hub")
    mem["sam3_tracker_with_backbone"] = torch.cuda.memory_allocated(dev) - mem["sam3_detector"]
    lap("sam3_s")
    dets = {}
    for s in SRCS:
        a = torch.cuda.memory_allocated(dev)
        dets[s] = coverage.Detector(s, dev)
        mem[s] = torch.cuda.memory_allocated(dev) - a
        lap(f"{s}_s")
    VOLUMES["/v/models"].commit()
    rec["model_bytes"] = mem
    out_rows, times, check, sheets_in = [], {s: [] for s in SRCS} | {"sam3_vision": [], "boxes_embed": []}, [], []
    decode_t = {s: [] for s in SRCS}
    peaks, all_imgs = {}, {}
    for site, cl in clicks.items():
        Path("/tmp/v.mp4").write_bytes(videos[site])
        want = sorted({c["pick_frame"] for c in cl})
        cap, imgs, i = cv2.VideoCapture("/tmp/v.mp4"), {}, 0
        while len(imgs) < len(want):
            ok, img = cap.read()
            if not ok:
                break
            if i in want:
                imgs[i] = img
            i += 1
        H, W = next(iter(imgs.values())).shape[:2]
        all_imgs.update({(site, f): im for f, im in imgs.items()})
        by_frame = {}
        for c in cl:
            by_frame.setdefault(c["pick_frame"], []).append(c)
        fr = sorted(imgs)
        for b0 in range(0, len(fr), 8):
            fb = fr[b0:b0 + 8]
            x = torch.from_numpy(np.stack([imgs[f] for f in fb])).to(dev)
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                vision = sam.vision(x)
            torch.cuda.synchronize()
            times["sam3_vision"].append((time.perf_counter() - t) / len(fb))
            with torch.inference_mode():
                pf = sam.detect(vision, len(fb), ("person", "floor"), segment.PERSON_SCORE, top=segment.PERSON_TOP)
            pm = torch.zeros((len(fb), *segment.DA3_HW), dtype=torch.bool, device=dev)
            fm = torch.zeros_like(pm)
            for j in range(len(fb)):
                pm[j] = pf["mask"][(pf["frame"] == j) & (pf["word"] == 0)].any(0) if ((pf["frame"] == j) & (pf["word"] == 0)).any() else pm[j]
                fm[j] = pf["mask"][(pf["frame"] == j) & (pf["word"] == 1)].any(0) if ((pf["frame"] == j) & (pf["word"] == 1)).any() else fm[j]
            pm_map = F.interpolate(pm[:, None].float(), size=(H // 2, W // 2), mode="nearest")[:, 0] > 0
            fm_map = F.interpolate(fm[:, None].float(), size=(H // 2, W // 2), mode="nearest")[:, 0] > 0
            torch.cuda.synchronize()
            t = time.perf_counter()
            emb = boxer.embed(vision.last_hidden_state)
            torch.cuda.synchronize()
            times["boxes_embed"].append((time.perf_counter() - t) / len(fb))
            found = {}
            for s in SRCS:
                torch.cuda.reset_peak_memory_stats(dev)
                base = torch.cuda.memory_allocated(dev)
                torch.cuda.synchronize()
                t = time.perf_counter()
                boxes = dets[s].detect([imgs[f] for f in fb], .05)
                torch.cuda.synchronize()
                times[s].append((time.perf_counter() - t) / len(fb))
                peaks[s] = max(peaks.get(s, 0), torch.cuda.max_memory_allocated(dev) - base)
                for j, f in enumerate(fb):
                    xyxy, sc, words = boxes[j]
                    keep = [i for i in range(len(sc)) if coverage.keep_box(xyxy[i], (H, W), words[i], agnostic=s == "owlv2")]
                    t = time.perf_counter()
                    lg, iou = boxer.decode(emb, j, coverage.to_1008(xyxy[keep], (H, W)))
                    mk = F.interpolate(lg[None], size=(H // 2, W // 2), mode="bilinear", align_corners=False)[0] > 0 if len(keep) else \
                        torch.zeros((0, H // 2, W // 2), dtype=torch.bool, device=dev)
                    torch.cuda.synchronize()
                    decode_t[s].append((time.perf_counter() - t, len(keep)))
                    cl_mask = torch.from_numpy(np.unpackbits(np.frombuffer(claimed[site][str(f)], np.uint8)).reshape(H // 2, W // 2).astype(bool)).to(dev) \
                        if str(f) in claimed[site] else torch.zeros((H // 2, W // 2), dtype=torch.bool, device=dev)
                    area = mk.flatten(1).sum(1).float()
                    on_c = (mk & cl_mask).flatten(1).sum(1).float()
                    on_f = (mk & fm_map[j]).flatten(1).sum(1).float()
                    on_p = (mk & pm_map[j]).flatten(1).sum(1).float()
                    for c in by_frame[f]:
                        yy, xx = min(int(c["y"] / 2), H // 2 - 1), min(int(c["x"] / 2), W // 2 - 1)
                        under = torch.nonzero(mk[:, yy, xx]).squeeze(1).tolist()
                        inbox = [i for i in keep if xyxy[i][0] <= c["x"] <= xyxy[i][2] and xyxy[i][1] <= c["y"] <= xyxy[i][3]]
                        found.setdefault((c["seed"], c["k"]), {})[s] = {
                            "boxes_over_click": len(inbox),
                            "masks": [[round(float(sc[keep[u]]), 3), words[keep[u]], int(area[u]), round(float(on_c[u] / area[u]), 3),
                                       round(float(on_f[u] / area[u]), 3), round(float(on_p[u] / area[u]), 3), round(float(iou[u]), 3)] for u in under]}
                        if under and c["label"] in ("miss", "background", "background-hit", "wrong"):
                            u = min(under, key=lambda q: int(area[q]))
                            sheets_in.append((s, site, c, f, mk[u].cpu().numpy(), float(sc[keep[u]]), words[keep[u]]))
                    if s == "yoloe" and len(check) < 12 and len(keep):  # the shared path against the tracker's own pixels
                        rgb = imgs[f][..., ::-1].copy()
                        bx = [[float(v) for v in xyxy[i]] for i in keep[:16]]
                        inp = tproc(images=rgb, input_boxes=[bx], return_tensors="pt")
                        with torch.inference_mode():
                            o = boxer.model(pixel_values=inp["pixel_values"].to(dev, torch.bfloat16), input_boxes=inp["input_boxes"].to(dev), multimask_output=False)
                        ref = F.interpolate(o.pred_masks[0, :, 0].float()[None], size=(H // 2, W // 2), mode="bilinear", align_corners=False)[0] > 0
                        mine = mk[:len(bx)]
                        inter = (ref & mine).flatten(1).sum(1).float()
                        uni = (ref | mine).flatten(1).sum(1).float().clamp(min=1)
                        check += (inter / uni).tolist()
            for c in [c for f in fb for c in by_frame[f]]:
                yy, xx = min(int(c["y"] / 2), H // 2 - 1), min(int(c["x"] / 2), W // 2 - 1)
                j = fb.index(c["pick_frame"])
                out_rows.append({**{k: c[k] for k in ("site", "seed", "k", "label", "note", "frame", "pick_frame", "x", "y")},
                                 "on_floor": bool(fm_map[j, yy, xx]), "on_person": bool(pm_map[j, yy, xx]), "src": found.get((c["seed"], c["k"]), {})})
    # sheets: per source, the smallest mask under each missed / background click
    sheet_blobs = {s: [t[1:] for t in sheets_in if t[0] == s] for s in SRCS}
    rec.update(rows=out_rows, times_per_frame_s={k: [round(float(np.median(v)), 4), round(float(np.percentile(v, 90)), 4), len(v)] for k, v in times.items() if v},
               decode={s: {"per_box_s": round(sum(a for a, _ in v) / max(1, sum(n for _, n in v)), 5), "boxes_per_frame": round(float(np.mean([n for _, n in v])), 1),
                           "per_frame_s": round(float(np.median([a for a, _ in v])), 4)} for s, v in decode_t.items()},
               peak_bytes_during_detect=peaks, shared_vs_pixel_iou={"n": len(check), "median": float(np.median(check)) if check else None,
                                                                    "p10": float(np.percentile(check, 10)) if check else None},
               gpu=torch.cuda.get_device_name(0), boot_total_s=round(time.perf_counter() - t0, 1))
    # tiles rendered here (the frames are here): a JPEG per source
    jpgs = {}
    for s, items in sheet_blobs.items():
        tiles = []
        for site, c, f, m, sc, w in items:
            img = all_imgs[(site, f)]
            tiles.append(tile(img, m, c, f"{s} {w} {sc:.2f} [{c['label']}] {c['note'] or ''}"))
        for p0 in range(0, len(tiles), 12):
            part = tiles[p0:p0 + 12]
            th, tw = part[0].shape[:2]
            rows = -(-len(part) // 2)
            sheet = np.full((rows * (th + 6), 2 * (tw + 6), 3), 200, np.uint8)
            for i, t in enumerate(part):
                r, q = divmod(i, 2)
                sheet[r * (th + 6):r * (th + 6) + th, q * (tw + 6):q * (tw + 6) + tw] = t
            jpgs[f"sheet-{s}-{p0 // 12}.jpg"] = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
    return rec, jpgs


def tile(img, m, c, caption, tw=400, th=225, zoom=110):
    import cv2
    H, W = img.shape[:2]
    vis = img.copy()
    mm = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    vis[mm] = (.55 * vis[mm] + .45 * np.array([255, 255, 0])).astype(np.uint8)
    cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cs, -1, (0, 255, 255), 2)
    x, y = c["x"], c["y"]
    full = cv2.resize(vis, (tw, th), interpolation=cv2.INTER_AREA)
    cv2.circle(full, (int(x * tw / W), int(y * th / H)), 8, (255, 0, 255), 2)
    a, b = int(np.clip(x - zoom, 0, W - 2 * zoom)), int(np.clip(y - zoom, 0, H - 2 * zoom))
    z = cv2.resize(vis[b:b + 2 * zoom, a:a + 2 * zoom], (th, th), interpolation=cv2.INTER_CUBIC)
    zx, zy = int((x - a) * th / (2 * zoom)), int((y - b) * th / (2 * zoom))
    cv2.drawMarker(z, (zx, zy), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
    out = np.full((th + 28, tw + th, 3), 255, np.uint8)
    out[:th, :tw], out[:th, tw:] = full, z
    cv2.putText(out, f"{c['site'][:4]} s{c['seed']} #{c['k']} " + caption[:60], (4, th + 20), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


@app.local_entrypoint()
def main(out: str):
    import click_audit as ca
    outp = Path(out)
    outp.mkdir(parents=True, exist_ok=False)
    videos, clicks, claimed = {}, {}, {}
    for site, files in DEV.items():
        videos[site] = (PHASE2 / "data/clips" / CLIPS[site] / "source-full.mp4").read_bytes()
        cl, cm = [], {}
        for f in files:
            meta = json.loads((PHASE2 / "runs" / f.format("clicks")).read_text())
            labels = json.loads((PHASE2 / "runs" / f.format("labels")).read_text())["labels"]
            pick, _, _, _, _ = ca.load_run(meta["mirror"], meta["report"], meta.get("pick_version"))
            at = {int(fr["frame"]): i for i, fr in enumerate(pick.frames)}
            for c in meta["clicks"]:
                lab = labels[str(c["k"])]
                cl.append({"site": site, "seed": meta["seed"], "k": c["k"], "label": lab["label"], "note": lab.get("note"), "frame": c["frame"],
                           "pick_frame": c["pick_frame"], "x": c["x"], "y": c["y"], "entity": c.get("entity")})
                i = at.get(int(c["pick_frame"]))
                if i is not None and str(c["pick_frame"]) not in cm:  # the pixels that round's pick map held (objects and people)
                    cm[str(c["pick_frame"])] = np.packbits(pick.map(i) != 0).tobytes()
        clicks[site], claimed[site] = cl, cm
    t = time.time()
    rec, jpgs = probe.remote(videos, clicks, claimed)
    rec["wall_s"] = round(time.time() - t, 1)
    (outp / "probe.json").write_text(json.dumps(rec, indent=1))
    (outp / "sheets").mkdir()
    for k, v in jpgs.items():
        (outp / "sheets" / k).write_bytes(v)
    print(json.dumps({k: rec[k] for k in ("times_per_frame_s", "decode", "shared_vs_pixel_iou", "model_bytes", "boot", "wall_s")}, indent=1))
