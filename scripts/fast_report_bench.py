"""Three videos through one FastReport container (FAST-BUILD-SPEC.md sections 12 D and 13): one app.run(), one boot; per
video one first call and two warm calls; the last call of the last video runs the background layers (background_s).

Each call streams its patches into one local mirror (layers.mirror); with --serve a poller plays the viewer against
layers.serve (500 ms, like live-report.ts) and records when each patch is first served. After each call the quality table
runs (scripts/fast_report_eval.py; its GPU rows once per video, on the first warm call). summary.json: analysis time to
each layer (written, from the MP4 bytes in the container), sent / received / served, boot, upload, per-stage per-GPU
peaks and >90% flags, quality, acceptance against section 13, and the Modal list-price estimate.

    python scripts/fast_report_bench.py --out RUNS/fb-bench-NNN [--sites me340,samsclub-a2,walmart] [--calls 3]
        [--background-s 1800] [--serve] [--no-gpu-eval]
    python scripts/fast_report_bench.py --billing RUNS/fb-bench-NNN     # reconcile with `modal billing report` (hours settle late)
    python scripts/fast_report_bench.py --self-check                    # the loop and the summary on a fake container
"""
import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "scripts")]
import fast_report_eval as ev  # noqa: E402
from fast_report.instrument import usd_per_s  # noqa: E402

MODAL = "/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal"
PORT = 8793
TARGETS = {"cameras": 20, "room": 20, "people": 20, "objects": 30, "outlines": 30, "events": 30, "models": 90, "splat": 180}  # section 13, s
RUN_SCHEMA = "panoptes-fast-run-v1"


class Poller(threading.Thread):
    """The viewer's loop: GET /fast/reports/<id>/patches?after=<seq> every 0.5 s; first time each seq is served (unix)."""

    def __init__(self, report, port=PORT):
        super().__init__(daemon=True)
        self.report, self.port, self.seen, self.halt = report, port, {}, threading.Event()

    def run(self):
        after = -1
        while not self.halt.is_set():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/fast/reports/{self.report}/patches?after={after}", timeout=5) as r:
                    for p in json.loads(r.read()):
                        self.seen.setdefault(p["seq"], time.time())
                        after = max(after, p["seq"])
            except OSError:
                pass  # not up yet, or nothing to serve
            self.halt.wait(.5)


def call(fr, mirror, root, mp4, site, report, options, poller=None):
    """One analysis: stream, mirror, keep the last run.json; local receive times per event (unix)."""
    started, received, run_json = time.time(), [], None
    if poller:
        poller.start()
    for event in fr.run.remote_gen(mp4, site, report, options):
        if isinstance(event, dict) and event.get("schema") == RUN_SCHEMA:
            run_json = event
            continue
        mirror(event, root)
        received.append((time.time(), event.get("seq") if isinstance(event, dict) else None, event.get("layer") if isinstance(event, dict) else None))
    finished = time.time()
    if poller:
        time.sleep(1.5)  # one more poll after the last patch
        poller.halt.set()
    assert run_json is not None, f"{report}: the stream ended without run.json"
    return {"client_call_unix": started, "client_done_unix": finished, "received": received,
            "served": dict(poller.seen) if poller else {}, "run": run_json}


def layer_times(rec):
    """Per layer: first and last version, sent/written (container clock), received/served (local clock - t0_unix: two clocks)."""
    t0, rows = rec["run"]["t0_unix"], rec["run"].get("layers", [])
    got = {seq: t for t, seq, _ in rec["received"] if seq is not None}
    out = {}
    for r in sorted(rows, key=lambda r: r["seq"]):
        o = out.setdefault(r["layer"], {"versions": 0})
        o["versions"] += 1
        x = {"seq": r["seq"], "sent_s": r.get("sent_s"), "written_s": r.get("written_s"),
             "received_s": round(got[r["seq"]] - t0, 3) if r["seq"] in got else None,
             "served_s": round(rec["served"][r["seq"]] - t0, 3) if r["seq"] in rec["served"] else None}
        o.setdefault("first", x)
        o["last"] = x
    return out


def acceptance(layers):
    """Section 13 on one warm call: first written_s of each layer against its target (the first SAM 3D model is the first
    'models' version)."""
    out = {}
    for layer, target in TARGETS.items():
        w = (layers.get(layer) or {}).get("first", {}).get("written_s")
        out[layer] = {"written_s": w, "target_s": target, "pass": None if w is None else w <= target}
    return out


def summarize(out, records, boot, meta):
    rows = []
    for rec in records:
        run, layers = rec["run"], layer_times(rec)
        rows.append({"site": rec["site"], "call": rec["call"], "report": run.get("report"), "first_call_after_boot": rec["call"] == 0,
                     "options": {k: v for k, v in rec["options"].items() if k != "eval_holdout"},
                     "upload_and_dispatch_s": round(run["t0_unix"] - rec["client_call_unix"], 3),
                     "analysis_elapsed_s": run.get("elapsed_s"), "layers": layers,
                     "acceptance": acceptance(layers) if rec["call"] else "first call after boot: reported, not judged",
                     "gpu_peak": run.get("gpu_peak"), "flags": run.get("flags"), "hardware": run.get("hardware"),
                     "stages": {s["stage"]: {k: s.get(k) for k in ("where", "start_s", "end_s", "s", "peak_gb", "over_90")} for s in run.get("stages", [])},
                     "quality": rec.get("quality"), "usd_estimate": run.get("usd_estimate")})
    warm = [r for r in rows if not r["first_call_after_boot"] and isinstance(r["acceptance"], dict)]
    table = {}
    for r in warm:
        for layer, a in r["acceptance"].items():
            table.setdefault(layer, []).append(a["written_s"])
    return {"schema": "panoptes-fast-bench-v1", **meta, "boot": boot, "calls": rows,
            "acceptance_warm": {k: {"written_s": v, "target_s": TARGETS[k], "all_pass": all(x is not None and x <= TARGETS[k] for x in v)} for k, v in table.items()},
            "flags": [f"{r['site']}#{r['call']}: {f}" for r in rows for f in r["flags"] or []],
            "quality_verdicts": {f"{r['site']}#{r['call']}": (r["quality"] or {}).get("verdict") for r in rows},
            "clock_note": "written_s/sent_s: container clock from t0 (MP4 bytes in the container); received_s/served_s and upload: local "
                          "clock minus the container's t0_unix (two clocks); cold start only in boot"}


def bench(a):
    import modal
    from fast_report import layers as fl  # C: mirror, serve
    from modal_apps.fast_report import FastReport, app  # A: the resident class
    out = a.out
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run folder
    mirror_root = out / "mirror"
    mirror_root.mkdir()
    if a.serve:
        threading.Thread(target=fl.serve, args=(mirror_root, PORT), daemon=True).start()
    sites = a.sites.split(",")
    records, meta = [], {"sites": sites, "calls_per_site": a.calls, "background_s": a.background_s, "started_unix": time.time(),
                         "container": {"gpu": "A100-80GB:2", "cpu": 32, "memory_gib": 160}, "usd_per_s_list": usd_per_s()}
    with modal.enable_output(), app.run():
        meta["app_id"] = app.app_id
        fr = FastReport()
        submitted = time.time()
        boot = fr.boot_info.remote()  # waits for the container: cold start, recorded, never counted as analysis
        boot = {**boot, "client_submitted_unix": submitted, "client_ready_unix": time.time(), "submit_to_ready_s_two_clocks": round(time.time() - submitted, 1)}
        (out / "boot.json").write_text(json.dumps(boot, indent=1))
        for site in sites:
            mp4 = (ev.PHASE2 / "data/clips" / ev.CLIPS[site] / "source-full.mp4").read_bytes()  # prepare_video_clip.full_video's cut of [S, E)
            sha = hashlib.sha256(mp4).hexdigest()
            for i in range(a.calls):
                last = site == sites[-1] and i == a.calls - 1
                report = f"fb-{site}-{sha[:8]}-{int(time.time())}"
                options = {"vocab": a.vocab, "splat_preview_s": 120, "background_s": a.background_s if last else 0,
                           "eval_holdout": ev.holdout_frames(site)}
                rec = call(fr, fl.mirror, mirror_root, mp4, site, report, options, Poller(report) if a.serve else None)
                rec.update(site=site, call=i, options=options)
                try:
                    q = ev.evaluate(ev.load_layers(mirror_root, report), site, gpu=a.gpu_eval and i == 1)
                    (out / f"eval-{report}.json").write_text(json.dumps(q, indent=1))
                    rec["quality"] = {"verdict": q["verdict"], "rows": q["rows"], "not_scored": q["not_scored"]}
                except Exception as error:  # a quality failure must not lose the timing
                    rec["quality"] = {"error": repr(error)[:600]}
                (out / f"call-{report}.json").write_text(json.dumps(rec, indent=1, default=str))
                records.append(rec)
                print(site, i, json.dumps({k: v["first"]["written_s"] for k, v in layer_times(rec).items()}), flush=True)
    meta["finished_unix"] = time.time()
    meta["usd_estimate_upper"] = round((meta["finished_unix"] - submitted) * usd_per_s(), 2)  # the container's whole life, list prices
    summary = summarize(out, records, boot, meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"acceptance_warm": summary["acceptance_warm"], "flags": summary["flags"], "quality": summary["quality_verdicts"],
                      "usd_estimate_upper": meta["usd_estimate_upper"]}, indent=1))


def billing(out, prefix="panoptes-fb"):
    """`modal billing report` rows since the run started, for this run's app id (or every app named `prefix`*)."""
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    start = time.strftime("%Y-%m-%d", time.gmtime(summary.get("started_unix", time.time())))
    raw = subprocess.run([MODAL, "billing", "report", "--start", start, "-r", "h", "--json"], capture_output=True, text=True).stdout
    rows = [r for r in json.loads(raw[raw.find("["):]) if r["object_id"] == summary.get("app_id") or r["description"].startswith(prefix)]
    by_app = {}
    for r in rows:
        by_app[r["description"]] = round(by_app.get(r["description"], 0) + float(r["cost"]), 4)
    result = {"since_utc": start, "by_app_usd": by_app, "total_usd": round(sum(by_app.values()), 4), "note": "only settled hours are listed"}
    if summary:
        summary["billing"] = result
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(result, indent=1))


def self_check():
    """The loop and the summary against a fake container: events in order, run.json last, served times, acceptance."""
    import tempfile

    class Fake:
        class run:  # noqa: N801  mimics the Modal method's .remote_gen
            @staticmethod
            def remote_gen(mp4, site, report, options):
                t0 = time.time() - .01
                for seq, layer in enumerate(["video", "cameras", "objects", "models"]):
                    yield {"kind": "patch", "seq": seq, "layer": layer}
                yield {"schema": RUN_SCHEMA, "report": report, "t0_unix": t0, "elapsed_s": 40.,
                       "layers": [{"layer": "video", "seq": 0, "sent_s": 1., "written_s": 2.}, {"layer": "cameras", "seq": 1, "sent_s": 17., "written_s": 18.5},
                                  {"layer": "objects", "seq": 2, "sent_s": 29., "written_s": 31.}, {"layer": "models", "seq": 3, "sent_s": 60., "written_s": 61.}],
                       "gpu_peak": [], "flags": ["gpu0 73.0 GiB > 90% at 22.0 s (sam3.vocab.wave2@gpu0)"], "stages": [{"stage": "decode", "where": "cpu", "start_s": 0, "end_s": 1.7, "s": 1.7}]}
    mirrored = []
    with tempfile.TemporaryDirectory() as tmp:
        rec = call(Fake(), lambda e, root: mirrored.append(e["seq"]), Path(tmp), b"", "me340", "r1", {})
    assert mirrored == [0, 1, 2, 3] and rec["run"]["report"] == "r1", "every patch mirrored in order, run.json kept apart"
    rec.update(site="me340", call=1, options={"eval_holdout": [1]}, quality=None)
    rec["served"] = {1: rec["run"]["t0_unix"] + 19.}
    layers = layer_times(rec)
    assert layers["cameras"]["first"]["served_s"] == 19. and layers["cameras"]["first"]["received_s"] is not None
    acc = acceptance(layers)
    assert acc["cameras"]["pass"] and acc["objects"]["pass"] is False and acc["models"]["pass"] and acc["splat"]["pass"] is None
    s = summarize(Path("."), [rec], {"ready_s": 100}, {"sites": ["me340"]})
    assert s["acceptance_warm"]["objects"] == {"written_s": [31.], "target_s": 30, "all_pass": False}
    assert s["flags"] == ["me340#1: gpu0 73.0 GiB > 90% at 22.0 s (sam3.vocab.wave2@gpu0)"] and "eval_holdout" not in s["calls"][0]["options"]
    assert json.loads(json.dumps(s, default=str))
    print("fast_report_bench self-check passed: stream -> mirror, run.json kept, layer times, section 13 acceptance, summary")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path)
    p.add_argument("--sites", default="me340,samsclub-a2,walmart")
    p.add_argument("--calls", type=int, default=3)
    p.add_argument("--background-s", type=int, default=1800)
    p.add_argument("--vocab", default="qwen", choices=("qwen", "gemini"))
    p.add_argument("--serve", action="store_true")
    p.add_argument("--no-gpu-eval", dest="gpu_eval", action="store_false")
    p.add_argument("--billing", type=Path)
    p.add_argument("--self-check", action="store_true")
    a = p.parse_args()
    if a.self_check:
        self_check()
    elif a.billing:
        billing(a.billing)
    else:
        assert a.out, "--out RUNS/fb-bench-NNN"
        bench(a)
