"""Stand-ins for D's instrument (Clock, Vram) and C's layers (Writer, mirror) with the spec's interfaces (FAST-BUILD-SPEC
sections 7-8), so fb/a-core runs end to end on its own. Deleted at fb/build, where the real modules replace them.
ponytail: minimal on purpose; the real ones own run.json's full schema, the served_s clock and the HTTP endpoint."""
import hashlib
import json
import queue
import threading
import time
from contextlib import contextmanager
from pathlib import Path


def where(gpu):
    return "cpu" if gpu is None else f"gpu{getattr(gpu, 'index', gpu)}"


class Clock:
    """Seconds from t0 = the MP4 bytes in the container (the first line of run())."""

    def __init__(self):
        self.t0, self.t0_unix, self.rows, self.marks, self.lock = time.perf_counter(), time.time(), [], {}, threading.Lock()

    def now(self):
        return round(time.perf_counter() - self.t0, 3)

    def mark(self, name):
        self.marks.setdefault(name, self.now())

    def add(self, stage, start, gpu=None, n=None):
        with self.lock:
            self.rows.append({"stage": stage, "where": where(gpu), "start_s": start, "end_s": self.now(),
                              "s": round(self.now() - start, 3), **({"n": n} if n else {})})

    @contextmanager
    def stage(self, stage, gpu=None, n=None, sync=True):
        """sync: wait for that GPU's current stream at the end, so the row is GPU time, not launch time."""
        start = self.now()
        yield
        if gpu is not None and sync:
            import torch
            torch.cuda.current_stream(gpu).synchronize()
        self.add(stage, start, gpu, n)

    def external(self, stage, gpu=None, start_unix=0., end_unix=0., n=None):
        with self.lock:
            a, b = round(start_unix - self.t0_unix, 3), round(end_unix - self.t0_unix, 3)
            self.rows.append({"stage": stage, "where": where(gpu), "start_s": a, "end_s": b, "s": round(b - a, 3),
                              **({"n": n} if n else {})})

    def report(self, vram):
        stages = []
        for r in sorted(self.rows, key=lambda r: r["start_s"]):
            peak = vram.peak(r["start_s"], r["end_s"])
            stages.append({**r, "peak_gb": peak, "over_90": [p > .9 * t for p, t in zip(peak, vram.total)]})
        gpu_peak, flags = [], []
        for i, total in enumerate(vram.total):
            at, gb = max(((t, g[i]) for t, g in vram.samples), key=lambda x: x[1], default=(0., 0.))
            active = [r["stage"] for r in self.rows if r["start_s"] <= at <= r["end_s"]]
            gpu_peak.append({"gpu": i, "total_gb": round(total, 1), "peak_gb": round(gb, 2), "at_s": round(at, 2), "stages_active": active})
            if gb > .9 * total:
                flags.append(f"gpu{i} {gb:.1f} GB > 90% at {at:.1f} s ({', '.join(active)})")
        return {"clock": "seconds from the MP4 bytes in the container", "stages": stages, "marks": self.marks, "gpu_peak": gpu_peak, "flags": flags}


class Vram(threading.Thread):
    """Whole-device memory in use (every process: main, vLLM, Open3D), sampled every 50 ms: NVML through torch."""

    def __init__(self, gpus, clock):
        super().__init__(daemon=True)
        import torch
        self.gpus, self.clock, self.samples, self.halt = gpus, clock, [], threading.Event()
        self.total = [torch.cuda.mem_get_info(g)[1] / 1e9 for g in gpus]

    def run(self):
        import torch
        while not self.halt.is_set():
            self.samples.append((self.clock.now(), [(t - f) / 1e9 for f, t in (torch.cuda.mem_get_info(g) for g in self.gpus)]))
            time.sleep(.05)

    def stop(self):
        self.halt.set()
        self.join(1)

    def peak(self, a, b):
        inside = [g for t, g in self.samples if a - .05 <= t <= b + .05] or [[0.] * len(self.gpus)]
        return [round(max(g[i] for g in inside), 2) for i in range(len(self.gpus))]


def plain(o):
    return o.item() if hasattr(o, "item") else o.tolist() if hasattr(o, "tolist") else str(o)


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Writer:
    """put() never blocks the pipeline. One thread: per round, every waiting patch: blobs (content-addressed, never
    rewritten), patch JSON, the event to run() (sent_s), then ONE volume commit and a 'written' event (written_s)."""

    def __init__(self, volume, root, report_id, clock, client_has=()):
        self.volume, self.root, self.report, self.clock, self.client_has = volume, Path(root), report_id, clock, set(client_has)
        self.q, self.out, self.seq, self.versions, self.layers = queue.Queue(), queue.Queue(), 0, {}, []
        (self.root / "reports" / report_id / "patches").mkdir(parents=True, exist_ok=True)
        (self.root / "blobs/sha256").mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def put(self, layer, data, blobs=None, status=None, labels=()):
        self.seq += 1
        self.versions[layer] = self.versions.get(layer, 0) + 1
        data = json.loads(json.dumps(data, default=plain))  # a snapshot: the pipeline may change its objects later
        self.q.put((self.seq, layer, self.versions[layer], data, blobs or {}, status, list(labels), self.clock.now()))
        return self.seq

    def close(self):
        self.q.put(None)

    def events(self):
        while True:
            e = self.out.get()
            if e is None:
                return
            yield e

    def _loop(self):
        done = False
        while not done:
            batch = [self.q.get()]
            while True:
                try:
                    batch.append(self.q.get_nowait())
                except queue.Empty:
                    break
            if None in batch:
                done, batch = True, [b for b in batch if b is not None]
            if not batch:
                break
            start, written = self.clock.now(), []
            for seq, layer, version, data, blobs, status, labels, put_s in batch:
                refs, payload = {}, {}
                for role, (raw, meta) in blobs.items():
                    h = sha(raw)
                    path = self.root / "blobs/sha256" / h
                    if not path.exists():
                        path.write_bytes(raw)
                    refs[role] = {"sha256": h, "bytes": len(raw), **meta}
                    if h not in self.client_has:
                        payload[h] = raw
                patch = {"schema": "panoptes-fast-patch-v1", "report": self.report, "seq": seq, "layer": layer, "version": version,
                         "status": status, "labels": labels, "put_s": put_s, "sent_s": self.clock.now(), "data": data, "blobs": refs}
                (self.root / "reports" / self.report / "patches" / f"{seq:06d}-{layer}.json").write_text(json.dumps(patch))
                self.out.put({"type": "patch", "patch": patch, "blobs": payload})
                written.append(patch)
            self.volume.commit()
            now = self.clock.now()
            for p in written:
                row = {"layer": p["layer"], "seq": p["seq"], "version": p["version"], "put_s": p["put_s"], "sent_s": p["sent_s"],
                       "write_start_s": start, "written_s": now, "bytes": sum(b["bytes"] for b in p["blobs"].values()) + len(json.dumps(p["data"]))}
                self.layers.append(row)
                self.out.put({"type": "written", **row})
        self.out.put(None)


def mirror(event, root):
    """CLI side: a patch event's blobs checked against their sha256, then the patch and blobs written under root/<report>/."""
    patch = event["patch"]
    folder = Path(root) / patch["report"]
    (folder / "blobs").mkdir(parents=True, exist_ok=True)
    (folder / "patches").mkdir(exist_ok=True)
    for h, raw in event["blobs"].items():
        assert sha(raw) == h, f"blob {h} arrived damaged"
        (folder / "blobs" / h).write_bytes(raw)
    (folder / "patches" / f"{patch['seq']:06d}-{patch['layer']}.json").write_text(json.dumps(patch, indent=1))
