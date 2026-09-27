"""The live map worker (scripts/live_map.py) in a Modal CPU container behind a websocket. The Mac plays the phone: it replays
an ARKitScenes raw capture at 1x into the worker and reads the patches back, so the latencies include the real network both
ways and every number is taken on the Mac's clock (t_capture is stamped there too).

    modal run modal_apps/live_map_worker.py --raw DIR --output NEW_DIR [--rate 15] [--offline-mesh MESH --offline-cameras RUN]
    python modal_apps/live_map_worker.py --self-check

Ephemeral app only, never deployed: the endpoint lives as long as `modal run` does. The websocket wants a key this run makes
in memory; the container gets it as an in-memory secret and the Mac sends it as a header, never in a URL or a file.
"""
import io
import json
import os
import secrets
import sys
import threading
import time
from pathlib import Path

import modal

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
CPU, MEMORY_MB, TIMEOUT_S = 4, 8192, 1200
USD_PER_CORE_H, USD_PER_GIB_H = .0473, .008  # `modal billing rates`, 2026-09-27

app = modal.App("panoptes-live-map")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("libgl1", "libgomp1")
         .pip_install("numpy==2.5.1", "scipy==1.18.0", "opencv-python-headless==5.0.0.93", "open3d==0.19.0", "fastapi[standard]==0.139.0")
         .add_local_file(SCRIPTS / "live_map.py", "/root/live_map.py"))
KEY = secrets.token_urlsafe(32) if modal.is_local() else ""


# no retries argument: Modal refuses one on a web function, which never retries (retries 0)
@app.function(image=image, cpu=CPU, memory=MEMORY_MB, timeout=TIMEOUT_S, max_containers=1, scaledown_window=10,
              secrets=[modal.Secret.from_dict({"LIVE_MAP_KEY": KEY})])
@modal.asgi_app()
def worker():
    """GET /ready answers with the container clock (cold start, clock offset); /stream is one phone's session."""
    import asyncio
    from tempfile import TemporaryDirectory

    from fastapi import FastAPI, WebSocket
    import live_map
    import open3d  # noqa: F401  at container start, not in the stream: it takes 8.5 s and 1.25 GB here (M), and cost 43 frames in the first run

    api = FastAPI()

    @api.get("/ready")
    def ready():
        return {"time": time.time(), "cpus": os.cpu_count()}

    @api.websocket("/stream")
    async def stream(ws: WebSocket):
        if not secrets.compare_digest(ws.headers.get("authorization", ""), "Bearer " + os.environ["LIVE_MAP_KEY"]):
            await ws.close(code=1008)
            return
        await ws.accept()
        loop, inbox = asyncio.get_running_loop(), live_map.Inbox()

        def sink(data):  # the patch writer thread hands each patch to the socket
            asyncio.run_coroutine_threadsafe(ws.send_bytes(data), loop).result()

        with TemporaryDirectory() as output:
            work = loop.run_in_executor(None, live_map.serve, inbox, Path(output), live_map.MAX_BLOCKS, sink)
            while True:
                message = await ws.receive()
                data = (message.get("bytes") or b"") if message["type"] == "websocket.receive" else b""
                inbox.send_bytes(data)
                if not data:  # an empty message (or a hang-up) ends the stream
                    break
            stats = await work
        await ws.send_text(json.dumps(stats))
        await ws.close()

    return api


def clock(url, pings=5):
    """(container clock - Mac clock, round trip) from the fastest of a few /ready calls."""
    import urllib.request
    best = None
    for _ in range(pings):
        sent = time.time()
        there = json.load(urllib.request.urlopen(url + "/ready", timeout=60))["time"]
        back = time.time()
        if best is None or back - sent < best[1]:
            best = (there - (sent + back) / 2, back - sent)
    return best


def summarise(received, sent, done, offset, rtt, cold_start_s, wall_s):
    """Mac-side numbers: capture -> patch received, first extractable 3D, throughput, drops, bytes, cost."""
    import numpy as np
    latency = [x for r in received for x in r["latency"]]
    surface = [r["t"] for r in received if r["surface"]]
    span = sent["capture_s"]
    worker_side = [s["map_latency_s_p50_p95_max"] for s in done["submaps"]]
    usd = wall_s / 3600 * (CPU * USD_PER_CORE_H + MEMORY_MB / 1024 * USD_PER_GIB_H)
    return {
        "machine": f"worker: Modal CPU container ({CPU} cores, {MEMORY_MB // 1024} GiB) behind a websocket; phone: this Mac over the internet",
        "rate_cap_hz": sent["rate_cap_hz"], "frames_sent": sent["frames_sent"], "sent_fps": sent["sent_fps"],
        "frames_integrated": done["frames_integrated"], "integrated_fps": done["frames_integrated"] / span,
        "frames_dropped_latest_wins": done["coverage_gap_frames"].get("dropped", 0), "coverage_gap_frames": done["coverage_gap_frames"],
        "coverage_gap_longest_s": done["coverage_gap_longest_s"],
        "e2e_map_latency_s_p50_p95_max": [float(np.percentile(latency, 50)), float(np.percentile(latency, 95)), float(max(latency))] if latency else None,
        "e2e_map_latency_meaning": "frame capture on the Mac -> the Mac receives the first patch in which the frame's sampled surface is extractable",
        "frames_extractable": len(latency), "frames_never_extractable_30s": sum(s["frames_never_extractable"] for s in done["submaps"]),
        "e2e_time_to_first_extractable_3d_s": surface[0] - sent["stream_start"] if surface else None,
        "e2e_time_to_first_patch_s": received[0]["t"] - sent["stream_start"] if received else None,
        "worker_side_map_latency_s_p50_p95_max_clock_corrected_E": [[v - offset for v in w] if w else None for w in worker_side],
        "clock_offset_s_E": offset, "http_round_trip_s": rtt,
        "uplink_mbit_s": sent["uplink_mbit_s"], "uplink_depth_mbit_s": sent["uplink_depth_mbit_s"],
        "downlink_mbit_s": sum(r["bytes"] for r in received) * 8 / span / 1e6, "patches_received": len(received),
        "producer_send_lateness_s_p50_p95_max": sent["send_done_after_capture_s_p50_p95_max"],
        "decode_integrate_ms_p50_p95_max": done["decode_integrate_ms_p50_p95_max"], "worker_peak_rss_mb": done["worker_peak_rss_mb"],
        "submaps": [{k: s[k] for k in ("epoch", "patches", "patch_bytes_total", "patch_files_kept", "store_bytes", "peak_blocks", "patch_write_ms_p50_p95")}
                    for s in done["submaps"]],
        "cold_start_s": cold_start_s, "wall_s": wall_s,
        "usd_E": usd, "usd_meaning": f"wall_s x ({CPU} cores x ${USD_PER_CORE_H}/h + {MEMORY_MB // 1024} GiB x ${USD_PER_GIB_H}/h): an upper bound on the container's own seconds, image build not included"}


@app.local_entrypoint()
def main(raw: str, output: str, rate: float = 15., offline_mesh: str = "", offline_cameras: str = ""):
    import numpy as np
    from websockets.sync.client import connect
    import live_map
    t0, out = time.time(), Path(output)
    out.mkdir(parents=True)
    url = worker.get_web_url()
    while True:  # the first request boots the container
        try:
            offset, rtt = clock(url)
            break
        except (OSError, ValueError):  # not up yet, or a proxy page instead of our JSON
            assert time.time() - t0 < 600, "the worker did not come up in 10 minutes"
            time.sleep(2)
    cold_start_s = time.time() - t0
    received, maps, done = [], {}, {}

    def receive(ws):
        for message in ws:
            t = time.time()
            if isinstance(message, str):
                done.update(json.loads(message))
                continue
            with np.load(io.BytesIO(message)) as patch:
                blocks, epoch = live_map.as_blocks(patch), int(patch["epoch"])
                maps.setdefault(epoch, ({}, float(patch["voxel_size"])))[0].update(blocks)
                received.append({"t": t, "seq": int(patch["seq"]), "epoch": epoch, "bytes": len(message), "blocks": len(blocks),
                                 "latency": (t - patch["extractable_t_capture"]).tolist(),
                                 "surface": not any(r["surface"] for r in received) and bool(len(live_map.surface_points(blocks, float(patch["voxel_size"]))))})

    with connect(url.replace("https://", "wss://") + "/stream", additional_headers={"Authorization": f"Bearer {KEY}"},
                 max_size=None, compression=None, open_timeout=120) as ws:
        reader = threading.Thread(target=receive, args=(ws,))
        reader.start()
        sent = live_map.produce(Path(raw), ws.send, rate)
        reader.join(timeout=300)
    wall_s = time.time() - t0
    metrics = summarise(received, sent, done, offset, rtt, cold_start_s, wall_s)
    for name, value in (("producer", sent), ("worker", done), ("received", received)):
        (out / f"{name}.json").write_text(json.dumps(value, indent=1))
    if offline_mesh and maps:  # the consumer's map, rebuilt on the Mac from the patches alone
        blocks, voxel = max(maps.values(), key=lambda m: len(m[0]))
        metrics["against_offline"] = live_map.evaluate(live_map.surface_points(blocks, voxel), out, Path(offline_mesh),
                                                       Path(offline_cameras) if offline_cameras else None)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "against_offline"}, indent=1))


def self_check():
    received = [{"t": 11., "bytes": 1000, "latency": [], "surface": False}, {"t": 12., "bytes": 3000, "latency": [1.5, 2.], "surface": True}]
    sent = {"rate_cap_hz": 15, "frames_sent": 30, "sent_fps": 15., "capture_s": 2., "stream_start": 10., "uplink_mbit_s": 4.,
            "uplink_depth_mbit_s": 3.5, "send_done_after_capture_s_p50_p95_max": [0, 0, 0]}
    done = {"frames_integrated": 28, "coverage_gap_frames": {"dropped": 2}, "coverage_gap_longest_s": .1, "decode_integrate_ms_p50_p95_max": [1, 2, 3],
            "worker_peak_rss_mb": 300, "submaps": [{"epoch": 0, "patches": 2, "patch_bytes_total": 4000, "patch_files_kept": 2, "store_bytes": 1,
                                                    "peak_blocks": 5, "patch_write_ms_p50_p95": [1, 2], "map_latency_s_p50_p95_max": [5., 6., 7.],
                                                    "frames_never_extractable": 0}]}
    m = summarise(received, sent, done, offset=4., rtt=.05, cold_start_s=3., wall_s=3600.)
    assert m["e2e_map_latency_s_p50_p95_max"][0] == 1.75 and m["e2e_time_to_first_extractable_3d_s"] == 2. and m["e2e_time_to_first_patch_s"] == 1.
    assert m["worker_side_map_latency_s_p50_p95_max_clock_corrected_E"] == [[1., 2., 3.]]  # the container clock runs 4 s ahead
    assert m["frames_dropped_latest_wins"] == 2 and m["downlink_mbit_s"] == 4000 * 8 / 2 / 1e6 and m["integrated_fps"] == 14.
    assert abs(m["usd_E"] - (4 * .0473 + 8 * .008)) < 1e-9
    print("live map worker check passed: Mac-side latency, first 3D, clock correction, throughput and cost")


if __name__ == "__main__" and "--self-check" in sys.argv:
    self_check()
