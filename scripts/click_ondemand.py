"""mvp3 D4: the two ways to make clicks on real things open a card, measured on the held-out click audit (click_audit.py
sample, a fresh seed), in one ephemeral app:
  (b) on demand: every audit click that opens no entity asks the report container (FastReport.click, fast_report.ondemand);
      the round trip is timed here, where the viewer's local server calls it;
  (a) precomputed (X10's box route, measured as an upper bound): OWLv2-L objectness, top 100 boxes, -> SAM 3 tracker box
      masks on the very keyframe each click resolves to (the report would run it on the object keyframes only, every 3rd,
      and keep a mask only once the lift confirms it on two), X10's size limits, people out, a mask mostly on an existing
      entity joins it; per-keyframe GPU time, x the object keyframes, is a lower bound on its added analysis time.
Contact sheets for the agent's labels, then the score. Nothing here touches the report pipeline.

    modal run scripts/click_ondemand.py::measure --audit RUNS/mvp3-click-audit --out RUNS/mvp3-click-ondemand-001
    python scripts/click_ondemand.py sheets RUNS/mvp3-click-ondemand-001 --audit RUNS/mvp3-click-audit
    python scripts/click_ondemand.py score RUNS/mvp3-click-ondemand-001 --audit RUNS/mvp3-click-audit
    python scripts/click_ondemand.py --self-check
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts"), str(REPO / "modal_apps"), "/repo", "/repo/scripts", "/repo/modal_apps"]
import modal  # noqa: E402

from modal_apps.fast_report_app import VOLUMES, FastReport, app, image  # noqa: E402

SITES = {"me340": "mvp2-integrate-me340-007", "samsclub-a2": "mvp2-integrate-samsclub-a2-007", "walmart": "mvp2-integrate-walmart-007"}
OWL_MODEL, OWL_TOP = "google/owlv2-large-patch14-ensemble", 100  # X10's objectness route: top 100 boxes a keyframe
MIN_SHARE, MAX_SHARE = .0005, .2  # X10's size limits (GEO_MIN_PX / GEO_MAX_SHARE at its grid): specks and background out
JOIN, PERSON_OUT = .5, .5  # a mask this much on one existing entity joins it; this much on people: out
LABELS_B = ("right", "coarse", "wrong", "background-right", "fixture-named", "background-hit")  # on-demand card, by eye (see score())
LABELS_A = ("right", "coarse", "wrong", "none", "background-hit")


@app.function(image=image, gpu="A100-80GB:2", cpu=16, memory=65536, volumes=VOLUMES, timeout=1200, retries=0)
def owl_probe(requests: dict):
    """{report: [[pick index, x, y], ...]} -> per click the smallest filtered OWLv2-box mask under it (RLE at the pick grid), its
    box and objectness; per keyframe the GPU seconds (OWLv2, the tracker's image + box decode), frames split over the 2 GPUs."""
    import threading
    import cv2
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from fast_report import ondemand
    import sam3_app
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    t = time.perf_counter()
    owl = {d: Owlv2ForObjectDetection.from_pretrained(OWL_MODEL, cache_dir="/v/models/hf", torch_dtype=torch.float16).to(d).eval() for d in devs}
    proc = Owlv2Processor.from_pretrained(OWL_MODEL, cache_dir="/v/models/hf")
    point = {d: ondemand.Point(d, sam3_app.MODEL_ID, sam3_app.REVISION, "/v/sam3/huggingface/hub") for d in devs}
    boot_s = round(time.perf_counter() - t, 1)
    ip = proc.image_processor

    def boxes(d, rgb):
        """X10's Owl.detect objectness path: pad to a square, the model's side, top OWL_TOP boxes by objectness (xyxy px)."""
        import torch.nn.functional as F
        h, w = rgb.shape[:2]
        side, m = max(h, w), owl[d]
        x = torch.full((1, 3, side, side), .5, device=d)
        x[0, :, :h, :w] = torch.from_numpy(rgb).to(d).permute(2, 0, 1).float() / 255
        s = m.config.vision_config.image_size
        x = F.interpolate(x, size=(s, s), mode="bilinear", antialias=True, align_corners=False)
        x = ((x - torch.tensor(ip.image_mean, device=d).view(1, 3, 1, 1)) / torch.tensor(ip.image_std, device=d).view(1, 3, 1, 1)).half()
        with torch.inference_mode():
            emb = m.image_embedder(pixel_values=x)[0]
            feats = emb.reshape(1, -1, emb.shape[-1])
            bx = m.box_predictor(feats, emb)[0].float()
            obj = m.objectness_predictor(feats)[0].float().sigmoid()
            order = obj.argsort(descending=True)[:OWL_TOP]
            b = bx[order]
            xyxy = torch.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1) * side
            xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, w)
            xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, h)
        return xyxy.cpu().numpy(), obj[order].cpu().numpy()

    out, timing, lock = {}, [], threading.Lock()
    jobs = [(r, i, [c for c in cl if c[0] == i]) for r, cl in requests.items() for i in sorted({c[0] for c in cl})]

    def worker(d, part):
        with torch.cuda.device(d):
            for report, i, clicks in part:
                st = ondemand.state("/v/layers", report)
                f = st["pick"]["data"]["frames"][i]
                rgb = np.ascontiguousarray(ondemand.frame_bgr(st, f["frame"])[..., ::-1])
                ids = ondemand.chunk_of(st, i, "pick")
                ents = st["pick"]["data"]["entities"]
                torch.cuda.synchronize(d)
                t0 = time.perf_counter()
                bx, obj = boxes(d, rgb)
                torch.cuda.synchronize(d)
                t1 = time.perf_counter()
                masks, iou = point[d].masks(rgb, boxes=bx, size=(f["h"], f["w"]))
                t2 = time.perf_counter()
                masks = masks[:, 0]
                area = masks.reshape(len(masks), -1).sum(1)
                person = np.isin(ids, [k for k, e in enumerate(ents) if e and e.startswith("person")])
                keep = (area >= MIN_SHARE * ids.size) & (area <= MAX_SHARE * ids.size) & ((masks & person).reshape(len(masks), -1).sum(1) < PERSON_OUT * np.maximum(area, 1))
                with lock:
                    timing.append({"report": report, "i": i, "owl_s": round(t1 - t0, 4), "tracker_s": round(t2 - t1, 4), "boxes": int(len(bx)), "kept": int(keep.sum())})
                    for _, x, y in clicks:
                        c, r = min(int(x * f["w"] / 1280), f["w"] - 1), min(int(y * f["h"] / 720), f["h"] - 1)
                        under = [j for j in np.flatnonzero(keep) if masks[j, r, c]]
                        rec = {"candidates": len(under)}
                        if under:
                            j = min(under, key=lambda j: area[j])  # the pick map's paint rule: the smaller wins
                            on = ids[masks[j]]
                            codes, n = np.unique(on[on > 0], return_counts=True) if (on > 0).any() else (np.zeros(0, int), np.zeros(0, int))
                            joined = ents[codes[int(np.argmax(n))]] if len(n) and n.max() >= JOIN * area[j] else None
                            rec.update(mask=ondemand.rle(masks[j]), area=int(area[j]), box=np.round(bx[j], 1).tolist(), objectness=round(float(obj[j]), 3),
                                       iou=round(float(iou[j, 0]), 3), joins=joined)
                        out[f"{report}|{i}|{x}|{y}"] = rec
    th = [threading.Thread(target=worker, args=(d, jobs[k::2])) for k, d in enumerate(devs)]
    [x.start() for x in th]
    [x.join() for x in th]
    return {"clicks": out, "timing": timing, "boot_s": boot_s, "gpu_peak_gib": [round(torch.cuda.max_memory_reserved(d) / 2 ** 30, 2) for d in devs]}


def audit_clicks(audit, site, seed=29):
    import fast_report_eval as ev
    from click_audit import load_run
    path = Path(audit) / site / f"clicks-{site}-random-s{seed}.json"
    meta = json.loads(path.read_text())
    pick, _, _, fps, _ = load_run(meta["mirror"], meta["report"])
    for c in meta["clicks"]:
        c["i"] = int(pick.frame_at(c["frame"] / fps))
    return meta, pick, fps, ev


@app.local_entrypoint()
def measure(audit: str, out: str, sites: str = "me340,samsclub-a2,walmart", owl_from: str = "", runs: str = "29:outline",
            hold_s: int = 0, mirror: str = "", port: int = 8793):
    """runs: 'seed:style+style,...' (e.g. '29:dim,37:outline+dim'): the audit's clicks of that seed that open no entity, on demand,
    once per naming style. owl_from: an earlier box probe ('skip': none). hold_s: then serve `mirror` with on-demand clicks
    (the viewer check) for that long, the container kept up by the viewer's heartbeat."""
    outd = Path(out)
    outd.mkdir(parents=True, exist_ok=False)
    fr = FastReport()
    plan = [(int(r.split(":")[0]), r.split(":")[1].split("+")) for r in runs.split(",")]
    metas = {(seed, site): audit_clicks(audit, site, seed)[0] for seed, _ in plan for site in sites.split(",")}
    if owl_from != "skip":  # its own 2-GPU container first, then the report container's (never 4 GPUs at once)
        reqs = {}
        for (seed, site), meta in metas.items():
            reqs.setdefault(meta["report"], []).extend([c["i"], c["x"], c["y"]] for c in meta["clicks"] if c["entity"] is None and c["i"] >= 0)
        owl = json.loads(Path(owl_from).read_text()) if owl_from else owl_probe.remote(reqs)
        (outd / "owl.json").write_text(json.dumps(owl, indent=1, default=str))
    t = time.time()
    boot = fr.boot_info.remote()
    boot["client_submit_to_ready_s"] = round(time.time() - t, 1)
    (outd / "boot.json").write_text(json.dumps(boot, indent=1, default=str))
    rows = []
    for seed, styles in plan:
        for site in sites.split(","):
            meta = metas[(seed, site)]
            for c in meta["clicks"]:
                if c["entity"] is not None or c["i"] < 0:
                    continue
                for style in styles:
                    t0 = time.perf_counter()
                    try:
                        card, err = fr.click.remote(meta["report"], c["i"], c["x"], c["y"], style), None
                    except Exception as e:  # noqa: BLE001  recorded, never retried
                        card, err = None, repr(e)[:400]
                    rows.append({"seed": seed, "style": style, "site": site, "k": c["k"], "frame": c["frame"], "x": c["x"], "y": c["y"], "i": c["i"],
                                 "round_trip_s": round(time.perf_counter() - t0, 3), "card": card, "error": err})
                    print(seed, style, site, c["k"], rows[-1]["round_trip_s"], (card or {}).get("status"), ((card or {}).get("identity") or {}).get("name"),
                          err or "", flush=True)
            (outd / "ondemand.json").write_text(json.dumps({"boot": boot, "rows": rows}, indent=1, default=str))
    print(json.dumps({"clicks": len(rows), "errors": sum(r["error"] is not None for r in rows),
                      "round_trip_p50_s": float(np.median([r["round_trip_s"] for r in rows]))}), flush=True)
    if hold_s:
        from fast_report import layers as fl
        fl.serve(mirror, port, click=lambda r, i, x, y: fr.click.remote(r, i, x, y), alive=lambda: fr.alive.remote())
        (outd / "HOLDING").write_text(str(time.time()))
        print(f"holding {hold_s} s: {mirror} on :{port} with on-demand clicks", flush=True)
        time.sleep(hold_s)
        (outd / "HOLDING").unlink()


@app.local_entrypoint()
def view(mirror: str, hold_s: int = 600, port: int = 8793):
    """The viewer on a mirrored report with on-demand clicks live: this app's report container behind the local server's click
    route, held hold_s (web: npx vite --host 127.0.0.1; open http://127.0.0.1:5173/app.html#/live/<report on the Volume>)."""
    from fast_report import layers as fl
    fr = FastReport()
    boot = fr.boot_info.remote()
    fl.serve(mirror, port, click=lambda r, i, x, y: fr.click.remote(r, i, x, y), alive=lambda: fr.alive.remote())
    print(f"report container ready (cold start {boot.get('ready_s')} s); serving {mirror} on :{port} with on-demand clicks for {hold_s} s", flush=True)
    time.sleep(hold_s)


# ---------------------------------------------------------------- local: sheets and score

def decode(r):
    return np.repeat(np.arange(len(r["runs"])) % 2, r["runs"]).reshape(r["h"], r["w"]).astype(bool)


def tile(img, mask, c, caption):
    """click_audit.tile's layout with a given mask (the pick grid) tinted."""
    import cv2
    import click_audit as ca
    H, W = img.shape[:2]
    vis = img.copy()
    if mask is not None:
        m = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        vis[m > 0] = (.55 * vis[m > 0] + .45 * np.array([0, 200, 255])).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (0, 128, 255), 2)
    x, y = c["x"], c["y"]
    full = cv2.resize(vis, (ca.TILE_W, ca.TILE_H), interpolation=cv2.INTER_AREA)
    s = ca.TILE_W / W
    cv2.circle(full, (int(x * s), int(y * s)), 9, (0, 0, 0), 4)
    cv2.circle(full, (int(x * s), int(y * s)), 9, (255, 0, 255), 2)
    Z = ca.ZOOM
    a, b = int(np.clip(x - Z, 0, W - 2 * Z)), int(np.clip(y - Z, 0, H - 2 * Z))
    zoom = cv2.resize(vis[b:b + 2 * Z, a:a + 2 * Z], (ca.TILE_H, ca.TILE_H), interpolation=cv2.INTER_CUBIC)
    zx, zy = int((x - a) * ca.TILE_H / (2 * Z)), int((y - b) * ca.TILE_H / (2 * Z))
    cv2.circle(zoom, (zx, zy), 6, (0, 0, 0), 3)
    cv2.circle(zoom, (zx, zy), 6, (255, 0, 255), 1)
    o = np.full((ca.TILE_H + 34, ca.TILE_W + ca.TILE_H, 3), 255, np.uint8)
    o[:ca.TILE_H, :ca.TILE_W], o[:ca.TILE_H, ca.TILE_W:] = full, zoom
    cv2.putText(o, caption[:72], (6, ca.TILE_H + 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 2, cv2.LINE_AA)
    return o


def sheets(run, audit, per=10, seed=29, style="outline"):
    import cv2
    import click_audit as ca
    od = [r for r in json.loads((Path(run) / "ondemand.json").read_text())["rows"] if r.get("seed", 29) == seed and r.get("style", "outline") == style]
    owl = json.loads((Path(run) / "owl.json").read_text())["clicks"] if (Path(run) / "owl.json").exists() else None
    for site in SITES:
        meta, pick, fps, ev = audit_clicks(audit, site, seed)
        rows = [r for r in od if r["site"] == site]
        if not rows:
            continue
        video = Path(meta["mirror"]) / "blobs/sha256" / ev.patch_versions(meta["mirror"], meta["report"], "video")[0]["blobs"]["video"]["sha256"]
        cards = ca.load_run(meta["mirror"], meta["report"])[2]
        imgs = ca.frames_of(video, [r["frame"] for r in rows])
        for kind in ("b", "a") if owl is not None else ("b",):
            tiles = []
            for r in rows:
                cd = r["card"] or {}
                if kind == "b":
                    m = decode(cd["mask"]) if cd.get("mask") else None
                    idn = cd.get("identity") or {}
                    cap = f"#{r['k']:02d} f{r['frame']} {cd.get('status') or r['error']}: " + (
                        f"{ca.name_of(cards, cd['entity'])} ({cd['entity']})" if cd.get("status") == "entity" else
                        f"floor ({(cd.get('identity') or {}).get('namer', {}).get('name')})" if cd.get("surface") == "floor" else
                        f"{idn.get('name')} [{(idn.get('namer') or {}).get('status')}]")
                else:
                    a = owl.get(f"{meta['report']}|{r['i']}|{r['x']}|{r['y']}") or {}
                    m = decode(a["mask"]) if a.get("mask") else None
                    cap = f"#{r['k']:02d} f{r['frame']} OWL box mask" + (f" joins {ca.name_of(cards, a['joins'])}" if a.get("joins") else "") + ("" if m is not None else ": none")
                tiles.append(tile(imgs[r["frame"]], m, r, cap))
            for s0 in range(0, len(tiles), per):
                part = tiles[s0:s0 + per]
                th, tw = part[0].shape[:2]
                rws = -(-len(part) // 2)
                sheet = np.full((rws * (th + 8), 2 * (tw + 8), 3), 200, np.uint8)
                for q, tl in enumerate(part):
                    rr, cc = divmod(q, 2)
                    sheet[rr * (th + 8):rr * (th + 8) + th, cc * (tw + 8):cc * (tw + 8) + tw] = tl
                p = Path(run) / site / (f"sheet-{kind}-{s0 // per}.jpg" if (seed, style) == (29, "outline") else f"sheet-{kind}-s{seed}-{style}-{s0 // per}.jpg")
                p.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(p), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
                print(p)


def score(run, audit):
    """Per video, over the audit's clicks (agent labels: labels-<site>-random-s29.json for the baseline, <site>/labels-b.json and
    labels-a.json here): real-object clicks that open the right thing (baseline, + on demand, + boxes), background clicks
    that stay unknown, on-demand names, time to card. A 'coarse' label counts as right (a group or the whole for a part)."""
    od = json.loads((Path(run) / "ondemand.json").read_text())["rows"]
    out = {}
    for site in SITES:
        meta = json.loads((Path(audit) / site / f"clicks-{site}-random-s29.json").read_text())
        base = json.loads((Path(audit) / site / f"labels-{site}-random-s29.json").read_text())["labels"]
        lb = json.loads((Path(run) / site / "labels-b.json").read_text())["labels"] if (Path(run) / site / "labels-b.json").exists() else {}
        la = json.loads((Path(run) / site / "labels-a.json").read_text())["labels"] if (Path(run) / site / "labels-a.json").exists() else {}
        rows = {r["k"]: r for r in od if r["site"] == site}
        real = [c for c in meta["clicks"] if base[str(c["k"])]["label"] in ("correct", "wrong", "miss")]
        bg = [c for c in meta["clicks"] if base[str(c["k"])]["label"] in ("background", "background-hit")]
        ok = lambda lab: lab in ("right", "coarse")  # noqa: E731

        def opens(c, extra):
            g = base[str(c["k"])]["label"]
            if c["entity"] is not None:
                return g == "correct"
            return ok((extra.get(str(c["k"])) or {}).get("label"))
        names = [lb[str(c["k"])].get("name") for c in real if c["entity"] is None and ok((lb.get(str(c["k"])) or {}).get("label"))]
        rt = [r["round_trip_s"] for r in rows.values() if r["card"]]
        out[site] = {"clicks": len(meta["clicks"]), "real_objects": len(real), "background": len(bg),
                     "real_open_right_baseline": sum(base[str(c["k"])]["label"] == "correct" for c in real),
                     "real_open_right_with_ondemand": sum(opens(c, lb) for c in real) if lb else None,
                     "real_open_right_with_boxes_upper_bound": sum(opens(c, la) for c in real) if la else None,
                     "background_unknown_baseline": sum(base[str(c["k"])]["label"] == "background" for c in bg),
                     "background_unknown_with_ondemand": sum(c["entity"] is None and (lb.get(str(c["k"])) or {}).get("label") == "background-right" for c in bg) if lb else None,
                     "background_unknown_with_boxes": sum(c["entity"] is None and (la.get(str(c["k"])) or {}).get("label") in ("none", "background-right") for c in bg) if la else None,
                     "ondemand_names_right_close_wrong": [names.count(x) for x in ("right", "close", "wrong")],
                     "ondemand_round_trip_s_p50_p95_max": [round(float(np.percentile(rt, q)), 2) for q in (50, 95, 100)] if rt else None,
                     "ondemand_container_s_p50": round(float(np.median([r["card"]["timing"]["container_s"] for r in rows.values() if r["card"]])), 2) if rt else None}
    return out


def self_check():
    from fast_report import ondemand
    m = np.zeros((36, 64), bool)
    m[3:9, 10:30] = True
    assert (decode(ondemand.rle(m)) == m).all() and (decode(ondemand.rle(~m)) == ~m).all()
    print("click_ondemand self-check ok: mask RLE round trip (both starting values)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", nargs="?", choices=("sheets", "score"))
    p.add_argument("run", nargs="?")
    p.add_argument("--audit")
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--style", default="outline")
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif a.mode == "sheets":
        sheets(a.run, a.audit, seed=a.seed, style=a.style)
    elif a.mode == "score":
        print(json.dumps(score(a.run, a.audit), indent=1))
