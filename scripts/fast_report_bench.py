"""Three videos through one FastReport container (FAST-BUILD-SPEC.md sections 12 D and 13; CLICK-MVP-SPEC 8): one
app.run(), one boot; per video the calls of --plan (default: a first call, two warm calls, one shifted-window call +5 s,
cut from the clip's source video with fast_report_app.cut); the last call of the last video runs the background layers.
After the container is released, fast_report_eval.mvp scores the MVP rows (clicks against SAM 3 references, identity,
physical info vs the delivered report, repeatability warm vs warm and warm vs shifted, latency vs spec 7's targets) into
OUT/mvp, and summary.md holds one table for the three videos. Only the first call of the first video is a first call
after boot (L5); the other videos' call 0 runs on a warm container and is labelled so.

Each call streams its patches into one local mirror (layers.mirror); with --serve a poller plays the viewer against
layers.serve (500 ms, like live-report.ts) and records when each patch is first served. After each call the quality table's
CPU rows run (seconds: the container's 60 s scale-down window must not close between calls); its GPU rows run once per video
on the first warm call, after the container is released. summary.json: analysis time to
each layer (written, from the MP4 bytes in the container), sent / received / served, boot, upload, per-stage per-GPU
peaks and >90% flags, quality, acceptance against section 13, and the Modal list-price estimate.

    python scripts/fast_report_bench.py --out RUNS/mvp-bench-NNN [--sites me340,samsclub-a2,walmart]
        [--plan first,warm,warm,shifted] [--shift-s 5] [--background-s 0] [--serve] [--no-gpu-eval] [--click-latency C.json]
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
import urllib.error
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
                    for p in json.loads(r.read())["patches"]:
                        self.seen.setdefault(p["seq"], time.time())
                        after = max(after, p["seq"])
            except OSError:
                pass  # not up yet, or nothing to serve
            self.halt.wait(.5)


class NamerRelay:
    """mvp2/identity: the run's naming requests (fast_report.vlm.namer_requests, sent on the event stream) into the deployed
    report container's GeminiAdapter (scripts/name_video_entities.py's mechanism: modal_apps.sam3_video_fal.container_command;
    no key leaves that container), every request at once; each answer back on the modal.Queue partition named by the report.
    Provider outputs (no images) are kept under OUT/namer/<report>/."""

    def __init__(self, queue, out, workers=160):  # mvp2/integrate: 64 queued copies behind each other on Sam's Club (57 requests + copies)
        from concurrent.futures import ThreadPoolExecutor
        from review_video_object_semantics import REMOTE
        self.queue, self.out, self.pool, self.lock, self.container = queue, out, ThreadPoolExecutor(workers), threading.Lock(), None
        self.ready = threading.Event()
        self.ready.set()
        self.program = REMOTE.replace("'video.object_semantics'", "'video.entity_naming'").replace("max_output_tokens=2048", "max_output_tokens=8192")

    def resolve(self):
        """Wake the report container (its /health) and find it: sam3_video_fal.execute's steps, once per call."""
        import modal
        from modal_apps.sam3_video_fal import APP_NAME
        with self.lock:
            self.container = None
            url = modal.Function.from_name(APP_NAME, "web").get_web_url()
            try:
                urllib.request.urlopen(url.rstrip("/") + "/health", timeout=50).close()
            except urllib.error.HTTPError as error:
                if error.code != 404:
                    raise
            apps = [x for x in json.loads(subprocess.check_output([MODAL, "app", "list", "--json"], timeout=30))
                    if x["description"] == APP_NAME and x["state"] == "deployed"]
            live = json.loads(subprocess.check_output([MODAL, "container", "list", "--app-id", apps[0]["app_id"], "--json"], timeout=30))
            self.container = max(live, key=lambda x: str(x.get("start_time") or x.get("started_at") or ""))["container_id"]  # the newest
            return self.container

    def warm(self):
        """At each call's start (the report container may have scaled down between calls): wake and find it in the background,
        then keep it awake (its /health every 10 s) until stop(): Sam's Club's warm call found it gone 25 s after the wake
        ('Task has already finished', every request)."""
        self.ready, self.awake = threading.Event(), threading.Event()
        threading.Thread(target=lambda: (self._try(self.resolve), self.ready.set()), daemon=True).start()

        def ping(awake=self.awake):
            import modal
            from modal_apps.sam3_video_fal import APP_NAME
            url = self._try(lambda: modal.Function.from_name(APP_NAME, "web").get_web_url())
            while url and not awake.wait(10):
                self._try(lambda: urllib.request.urlopen(url.rstrip("/") + "/health", timeout=20).close())
        threading.Thread(target=ping, daemon=True).start()

    def stop(self):
        getattr(self, "awake", threading.Event()).set()

    @staticmethod
    def _try(fn):
        try:
            return fn()
        except Exception:  # noqa: BLE001  a failed wake is retried by the first request
            return None

    def submit(self, event):
        self.pool.submit(self.one, event)

    def run_program(self, payload):
        import base64
        import gzip
        from modal._utils.async_utils import synchronizer
        from modal_apps.sam3_video_fal import container_command
        chunks, meta = [], {}

        def on_line(line):
            e = json.loads(line)
            if e["phase"] == "provider_output_meta":
                meta.update(e["data"])
            elif e["phase"] == "provider_output_chunk":
                chunks.append(e["data"]["chunk"])
        getattr(self, "ready", threading.Event()).wait(90)
        try:
            code, _ = synchronizer.create_blocking(container_command)(self.container or self.resolve(), payload, on_line, self.program)
        except Exception as e:  # noqa: BLE001  the container scaled down under us: find (wake) it again, once
            if "already finished" not in str(e) and "ConflictError" not in repr(e):
                raise
            chunks.clear()
            meta.clear()
            code, _ = synchronizer.create_blocking(container_command)(self.resolve(), payload, on_line, self.program)
        raw = gzip.decompress(base64.b64decode("".join(chunks), validate=True))
        if code or hashlib.sha256(raw).hexdigest() != meta.get("sha256"):
            raise RuntimeError(f"provider exit {code}, output hash {'ok' if raw else 'missing'}")
        return json.loads(raw)

    def one(self, event):
        t, provider, error = time.time(), None, None
        payload = {"input": event["blocks"], "response_format": {"type": "text", "mime_type": "application/json", "schema": vlm_schema()}}
        try:
            provider = self.run_program(payload)
        except Exception as e:  # noqa: BLE001  one failed request: its objects go to the Qwen decider in the container
            error = repr(e)[:300]
        rec = {"report": event["report"], "request": event["request"], "attempt": event.get("attempt", 1), "n": event.get("n"), "s": round(time.time() - t, 2),
               "status": (provider or {}).get("status"), "usage": (provider or {}).get("usage"), "error": error}
        try:
            self.queue.put({**rec, "provider": provider}, partition=event["report"])
        except Exception as e:  # noqa: BLE001  a late copy after the call (its queue closed): kept on disk only
            rec["put_error"] = repr(e)[:200]
        d = self.out / "namer" / event["report"]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"request-{event['request']:02d}-a{event.get('attempt', 1)}.json").write_text(json.dumps({**rec, "provider": provider}, indent=1))


def vlm_schema():
    from fast_report import vlm
    return vlm.NAMER_SCHEMA


def call(fr, mirror, root, mp4, site, report, options, poller=None, relay=None):
    """One analysis: stream, mirror, keep the last run.json; local receive times per event (unix); naming requests to the relay."""
    started, received, run_json = time.time(), [], None
    if poller:
        poller.start()
    if relay:
        relay.warm()
    for event in fr.run.remote_gen(mp4, site, report, options):  # fast_report.layers events, then {"type": "run", "run": run.json}
        if event.get("type") == "namer_request" and relay:
            relay.submit(event)
        if event.get("type") == "run" and relay:
            relay.stop()
        if event.get("type") in ("patch", "written", "run"):
            mirror(event, root)
        if event.get("type") == "run":
            run_json = event["run"]
        elif event.get("type") == "patch":
            received.append((time.time(), event["patch"]["seq"], event["patch"]["layer"]))
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
        rows.append({"site": rec["site"], "call": rec["call"], "kind": rec.get("kind", "warm"), "report": run.get("report"),
                     "window_s": rec.get("window_s"), "first_call_after_boot": bool((run.get("boot") or {}).get("first_call_after_boot", rec["call"] == 0)),
                     "mvp_latency": rec.get("mvp_latency"),
                     "options": {k: v for k, v in rec["options"].items() if k not in ("eval_holdout", "hazard_queues")},
                     "upload_and_dispatch_s": round(run["t0_unix"] - rec["client_call_unix"], 3),
                     "analysis_elapsed_s": run.get("elapsed_s"), "layers": layers,
                     "acceptance": acceptance(layers) if rec.get("kind", "warm") == "warm" else f"{rec.get('kind')} call: reported, not judged",
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


def quality(rec, mirror_root, out, gpu):
    report = rec["run"]["report"]
    try:
        q = ev.evaluate(ev.load_layers(mirror_root, report), rec["site"], gpu=gpu)
        (out / f"eval-{report}.json").write_text(json.dumps(q, indent=1))
        rec["quality"] = {"verdict": q["verdict"], "rows": q["rows"], "not_scored": q["not_scored"]}
    except Exception as error:  # a quality failure must not lose the timing
        rec["quality"] = {"error": repr(error)[:600]}
    (out / f"call-{report}.json").write_text(json.dumps(rec, indent=1, default=str))


def window(site, kind, shift_s, out):
    """(mp4 bytes, [start, end] s, frame offset vs the base window) for a call: the base window is the clip's source-full.mp4
    (prepare_video_clip.full_video's cut of [S, E)); 'shifted' cuts [S + shift, E + shift) from the clip's source video."""
    clip = json.loads((ev.PHASE2 / "data/clips" / ev.CLIPS[site] / "clip.json").read_text())["source"]
    a, b = clip["start_s"], clip["end_s"]
    if kind != "shifted":
        return (ev.PHASE2 / "data/clips" / ev.CLIPS[site] / "source-full.mp4").read_bytes(), [a, b], 0
    from modal_apps.fast_report_app import cut
    path = out / f"input-{site}-{a + shift_s:g}-{b + shift_s:g}.mp4"
    if not path.exists():
        cut(clip["video"], a + shift_s, b + shift_s, path)
    return path.read_bytes(), [a + shift_s, b + shift_s], int(round((a + shift_s) * clip["fps"])) - int(round(a * clip["fps"]))


def mvp_latency(rec, mirror_root, site, click_latency=None):
    """Spec 7's targets on one call (fast_report_eval.latency_row), fb/integrate's warm call of the same video beside."""
    fb_dir, fb_report = ev.FB_RUNS[site]
    fb_run = json.loads((ev.PHASE2 / "runs" / fb_dir / "reports" / fb_report / "run.json").read_text())
    patches = [json.loads(p.read_text()) for p in (mirror_root / "reports" / rec["run"]["report"] / "patches").glob("*.json")]
    return ev.latency_row(rec["run"], patches, fb_run, click_latency)


def bench(a):
    import modal
    from fast_report import layers as fl  # C: mirror, serve
    from modal_apps.fast_report_app import FastReport, app  # A: the resident class
    out = a.out
    out.mkdir(parents=True, exist_ok=False)  # never reuse a run folder
    mirror_root = out / "mirror"
    mirror_root.mkdir()
    holder = {}  # mvp3: the viewer's on-demand clicks go to this bench's report container once it exists
    if a.serve:
        threading.Thread(target=fl.serve, args=(mirror_root, PORT), kwargs={"click": lambda r, i, x, y: holder["fr"].click.remote(r, i, x, y),
                                                                          "alive": lambda: holder["fr"].alive.remote()}, daemon=True).start()
    sites, plan = a.sites.split(","), a.plan.split(",")
    click_latency = json.loads(a.click_latency.read_text()) if a.click_latency else None
    records, meta = [], {"sites": sites, "plan": plan, "shift_s": a.shift_s, "background_s": a.background_s, "started_unix": time.time(),
                         "container": {"gpu": "A100-80GB:2", "cpu": 32, "memory_gib": 160}, "usd_per_s_list": usd_per_s()}
    import contextlib
    hazard_ctx = contextlib.ExitStack()
    queues = None
    if a.hazard == "gemini":  # round 2: the hazard judge's Gemini questions through two ephemeral queues and this CLI's relay
        from fast_report import hazard
        queues = (hazard_ctx.enter_context(modal.Queue.ephemeral()), hazard_ctx.enter_context(modal.Queue.ephemeral()))
        stop = threading.Event()
        hazard_ctx.callback(stop.set)
        threading.Thread(target=hazard.relay, args=(*queues, stop, hazard_ctx.enter_context(open(out / "hazard-relay-events.jsonl", "a"))),
                         daemon=True).start()
        meta["hazard"] = {"decider": "gemini via the report-workspace container (scripts/name_video_entities.py's exec mechanism)",
                          "relay": "two modal.Queue.ephemeral(), this CLI"}
    with hazard_ctx, modal.enable_output(), app.run(), modal.Queue.ephemeral() as namer_q:
        meta["app_id"] = app.app_id
        relay = NamerRelay(namer_q, out) if a.namer == "gemini" else None
        fr = holder["fr"] = FastReport()
        submitted = time.time()
        boot = fr.boot_info.remote()  # waits for the container: cold start, recorded, never counted as analysis
        boot = {**boot, "client_submitted_unix": submitted, "client_ready_unix": time.time(), "submit_to_ready_s_two_clocks": round(time.time() - submitted, 1)}
        (out / "boot.json").write_text(json.dumps(boot, indent=1))
        for site in sites:
            for i, kind in enumerate(plan):
                mp4, span, offset = window(site, kind, a.shift_s, out)
                sha = hashlib.sha256(mp4).hexdigest()
                last = site == sites[-1] and i == len(plan) - 1
                report = f"mvp-{site}-{sha[:8]}-{int(time.time())}"
                fl.put_blob(mirror_root, mp4)  # the client's own MP4 is never sent back
                options = {"vocab": a.vocab, "discover": a.discover, "client_has": [sha], "background_s": a.background_s if last else 0, "window_s": span,
                           "coverage": a.coverage and kind != "warm-off", "eval_holdout": [f - offset for f in ev.holdout_frames(site) if f - offset >= 0], **({"namer": namer_q} if relay else {}),
                           "coverage_debug": json.loads(a.coverage_debug.read_text()).get(site, []) if a.coverage_debug else [],
                           **({} if a.judge == "on" else {"judge": False}), **({} if a.display == "on" else {"display": False}),
                           "judge_vlm": a.hazard != "off", "identity_vlm": a.identity_vlm, "naming": a.naming}  # r4: the VLMs only when asked
                options.update({k: False for k in ("judge", "identity", "display") if k in a.off} | ({"dump": True} if a.dump else {}))
                if queues is not None:
                    hazard.workspace_container()  # awake before the call (the report service is up in production): off the analysis clock
                    options["hazard_queues"] = queues
                rec = call(fr, lambda e, r: fl.mirror(e, r, int(a.mirror_max_mb * 1e6) if a.mirror_max_mb else None), mirror_root, mp4, site, report, options,
                           Poller(report) if a.serve else None, relay)
                rec.update(site=site, call=i, kind=kind, window_s=span, frame_offset=offset, options={k: v for k, v in options.items() if k not in ("namer", "hazard_queues")} | ({"namer": "gemini relay"} if relay else {})
                           | ({"hazard": "gemini relay"} if queues is not None else {}))
                if kind != "shifted":  # the delivered report's frames are the base window's
                    quality(rec, mirror_root, out, gpu=False)
                rec["mvp_latency"] = mvp_latency(rec, mirror_root, site, click_latency)
                records.append(rec)
                (out / f"call-{report}.json").write_text(json.dumps(rec, indent=1, default=str))
                print(site, i, kind, json.dumps({k: v["first"]["written_s"] for k, v in layer_times(rec).items()}), flush=True)
    meta["finished_unix"] = time.time()
    meta["usd_estimate_upper"] = round((meta["finished_unix"] - submitted) * usd_per_s(), 2)  # the container's whole life, list prices
    summary = summarize(out, records, boot, meta)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    runs, repeats = {}, {}
    for site in sites:  # the MVP rows: the first warm call, against the second warm call and the shifted window
        mine = [r for r in records if r["site"] == site]
        warm = [r for r in mine if r["kind"] == "warm"] or [r for r in mine if r["kind"] != "shifted"]
        if not warm:
            continue
        runs[site] = (mirror_root, warm[0]["run"]["report"])
        repeats[site] = [(mirror_root, r["run"]["report"], r["frame_offset"], f"{r['kind']} call {r['call']}") for r in mine
                         if r is not warm[0] and r["kind"] in ("warm", "shifted")]
    if runs:
        try:
            ev.mvp(out / "mvp", runs, repeats, gpu=a.gpu_eval, labels_dir=out / "mvp")
        except Exception as error:  # a scoring failure must not lose the timing
            summary["mvp_error"] = repr(error)[:600]
    summary["mvp_latency_table"] = latency_table(records)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    (out / "summary.md").write_text(summary_md(summary, out))
    print((out / "summary.md").read_text())


def latency_table(records):
    """{layer: {'<site> <kind> <call>': 'written s (target) ok'}}: first call after boot, warm and shifted calls side by side (L5)."""
    table = {}
    for rec in records:
        col = f"{rec['site']} {rec.get('kind', '')} {rec['call']}"
        for layer, v in (rec.get("mvp_latency") or {}).get("layers", {}).items():
            ok = "" if v["ok"] is None else " ✓" if v["ok"] else " ✗"
            table.setdefault(layer, {})[col] = f"{v['written_s']} ({v.get('target_s') or v.get('fb_written_s') or v.get('target')}){ok}"
    return table


def summary_md(summary, out):
    """The bench's one page: the MVP table (fast_report_eval.summary_md), then every call's clock against spec 7, boot, spend."""
    md = (out / "mvp" / "summary.md").read_text() if (out / "mvp" / "summary.md").exists() else f"# Click MVP bench\n\nMVP rows not scored: {summary.get('mvp_error')}\n"
    t = summary.get("mvp_latency_table") or {}
    cols = sorted({c for v in t.values() for c in v}, key=lambda c: (c.split(" ")[0], int(c.split(" ")[-1])))
    md += "\n## Every call against spec 7 (s from the MP4 in the container; target in brackets)\n\n| layer | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
    for layer, v in t.items():
        md += f"| {layer} | " + " | ".join(v.get(c, "—") for c in cols) + " |\n"
    md += f"\nCold start (not analysis time): {summary.get('boot', {}).get('ready_s')} s in the container; submit to ready {summary.get('boot', {}).get('submit_to_ready_s_two_clocks')} s.\n"
    md += f"Flags: {summary.get('flags') or 'none'}. Spend: list-price upper bound ${summary.get('usd_estimate_upper')}; `--billing` reconciles with Modal's report.\n"
    return md


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
                    yield {"type": "patch", "report": report, "patch": {"seq": seq, "layer": layer}, "blobs": {}}
                yield {"type": "run", "report": report, "run": {"schema": RUN_SCHEMA, "report": report, "t0_unix": t0, "elapsed_s": 40.,
                       "layers": [{"layer": "video", "seq": 0, "sent_s": 1., "written_s": 2.}, {"layer": "cameras", "seq": 1, "sent_s": 17., "written_s": 18.5},
                                  {"layer": "objects", "seq": 2, "sent_s": 29., "written_s": 31.}, {"layer": "models", "seq": 3, "sent_s": 60., "written_s": 61.}],
                       "gpu_peak": [], "flags": ["gpu0 73.0 GiB > 90% at 22.0 s (sam3.vocab.wave2@gpu0)"], "stages": [{"stage": "decode", "where": "cpu", "start_s": 0, "end_s": 1.7, "s": 1.7}]}}
    mirrored = []
    with tempfile.TemporaryDirectory() as tmp:
        rec = call(Fake(), lambda e, root: mirrored.append(e["patch"]["seq"] if e["type"] == "patch" else e["type"]), Path(tmp), b"", "me340", "r1", {})
    assert mirrored == [0, 1, 2, 3, "run"] and rec["run"]["report"] == "r1", "every patch mirrored in order, run.json last"
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
    # the MVP pieces: the base window is the clip's own cut (offset 0); the per-call latency table and the page
    mp4, span, offset = window("me340", "warm", 5., Path("."))
    assert offset == 0 and span == [165., 195.] and mp4[:12].find(b"ftyp") >= 0
    rec["mvp_latency"] = {"layers": {"pick v1": {"written_s": 33.5, "target_s": 34., "ok": True}, "cameras": {"written_s": 20., "fb_written_s": 18.1, "ok": False}}}
    rec.update(kind="shifted", call=3)
    t = latency_table([rec])
    assert t == {"pick v1": {"me340 shifted 3": "33.5 (34.0) ✓"}, "cameras": {"me340 shifted 3": "20.0 (18.1) ✗"}}, t
    with tempfile.TemporaryDirectory() as tmp:
        md = summary_md({"mvp_latency_table": t, "boot": {"ready_s": 131.}, "mvp_error": "x"}, Path(tmp))
    assert "| pick v1 | 33.5 (34.0) ✓ |" in md and "131.0 s" in md
    print("fast_report_bench self-check passed: stream -> mirror, run.json kept, layer times, section 13 acceptance, summary, MVP window/latency table/page")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path)
    p.add_argument("--sites", default="me340,samsclub-a2,walmart")
    p.add_argument("--plan", default="first,warm,warm,shifted", help="calls per video: first | warm | shifted | warm-off (r4: warm, --coverage off)")
    p.add_argument("--shift-s", type=float, default=5.)
    p.add_argument("--click-latency", type=Path, help="C's headless click check result {p50_ms, p95_ms, n}")
    p.add_argument("--background-s", type=int, default=0)
    p.add_argument("--mirror-max-mb", type=float, default=0., help="larger blobs stay on the Modal Volume (a nearly full disk)")
    p.add_argument("--vocab", default="qwen", choices=("qwen", "gemini"))
    p.add_argument("--hazard", default="off", choices=("off", "qwen", "gemini"),
                   help="the hazard judge's VLM (r4: off by default, the rules alone; gemini: relayed by this CLI)")
    p.add_argument("--namer", default="none", choices=("gemini", "none"), help="mvp2/identity: object names from Gemini through the relay, or none (r4 default)")
    p.add_argument("--identity-vlm", action="store_true", help="r4: the Qwen decider names what the namer did not (off: the SAM 3 word stays)")
    p.add_argument("--judge", default="off", choices=("on", "off"), help="r4: off = no judgements at all (no rules, no hazard VLM questions); r4 integrate default off: judgement paused")
    p.add_argument("--display", default="on", choices=("on", "off"), help="r4: off = no SAM 3D models and no splat (the facts only)")
    p.add_argument("--coverage", action=argparse.BooleanOptionalAction, default=True, help="r4/coverage (on by default in r4/integrate): detector boxes -> SAM 3 tracker masks in densify ('warm-off' calls leave it off)")
    p.add_argument("--coverage-debug", type=Path, help="r4 dev: {site: [{id, frame, x, y}]} points whose box masks' fates the run records")
    p.add_argument("--discover", action="store_true", help="X10's catch-all and label words in SAM 3's wave 1 (vlm.DISCOVER)")
    p.add_argument("--naming", default="cascade", choices=("cascade", "decider"), help="r4/naming: the cascade (the VLM last), or round 3's Qwen decider for every object")
    p.add_argument("--serve", action="store_true")
    p.add_argument("--off", default="", help="r4: comma list of judge,identity,display to switch off (judgement paused; no VLM identity)")
    p.add_argument("--dump", action="store_true", help="r4/instances: the instance layer's inputs to the layers Volume (reports/<id>/r4-instances-dump.pkl.gz)")
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
