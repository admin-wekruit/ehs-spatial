"""Timing and GPU memory for the fast report (FAST-BUILD-SPEC.md section 8): Clock, Vram, run.json.

    clock = Clock()                                   # first line of run(): t0 = the MP4 bytes are in the container
    with clock.stage("da3.shot1", gpu=0, n={"views": 113}, sync=True): ...   # sync: GPU time, not launch time
    clock.external("sam3d.generate", gpu=0, start_unix=a, end_unix=b, n={})  # child processes report unix times
    clock.layer("cameras", seq=2, version=1, sent_s=17.1)                     # the writer; later: written_s=18.6
    vram = Vram([0, 1]).start()                       # at boot, one sampler for every run of the container
    run_json = clock.report(vram, report=..., site=..., video=..., boot=..., processes={"main": torch_peaks()})

A stage's peak is the whole-device peak inside its window: overlapping stages share one peak, and gpu_peak lists the
stages active at the moment of the peak. Over 90 % of a card is flagged, never hidden. Memory is in GiB (2^30 bytes) under
the spec's *_gb keys: an A100-80GB is 80 GiB, so the flag sits at 72 (E9's tables used 1e9-byte GB: 77.5 GB = 72.2 GiB).
Measured on 2 x A100-80GB with MPS (runs/fb-d-harness-vram-001): nvml and torch agree to 0.01 GiB and both see other
processes' (MPS clients') memory; nvml is the default because it needs no CUDA context in this process.

    python -m fast_report.instrument --self-check     # no GPU
"""
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from bisect import bisect_left, bisect_right
from contextlib import contextmanager

FLAG_SHARE = .9
STAGES = {"decode", "cuts", "sam3.person", "vlm.vocab", "sam3.vocab.wave1", "sam3.vocab.wave2", "da3.shot", "scale.shot",
          "tsdf.shot", "people", "lift", "outlines", "vlm.events", "sam3d.prepare", "sam3d.generate", "sam3d.assess",
          "sam3d.decimate", "splat.preview", "splat.full",
          # the core's finer stages (fb/a-core), fixed as well
          "vlm.vocab.frames", "vlm.events.frames", "gather.person_floor", "pack.room.shot", "people.shot", "dedupe",
          "outlines.segmented", "outlines.projected", "outlines.polygons", "cascade.embed", "cascade.decide", "vlm.name.crops",
          "vlm.name", "frames.shared", "sam3d.inputs", "splat.setup", "splat.score", "da3.restore",
          # click MVP (mvp/a-cards): pick layer, cards, densify, identity
          "objects.boxes", "pick.maps", "pick.encode", "cards.inputs", "cards.v1", "cards.v3", "densify.sam3", "densify.lift",
          "outlines.polygons.v2", "vlm.identity", "vlm.identity.ehs", "vlm.identity.other", "vlm.identity.all_densify", "identity.gemini", "identity.gemini.sheets",
          # the judgement engine (mvp/b-judge)
          "judge.rules", "judge.som", "judge.vlm", "judge.evidence", "judge.gemini_send", "judge.gemini", "judge.qwen",
          # r4/coverage: box sources and SAM 3 tracker masks in densify
          "coverage.detect", "coverage.masks",
          # r4/naming: the cascade
          "naming.signals", "naming.signals.densify", "naming.settle.first", "naming.settle.densify", "naming.vlm.first", "naming.vlm.densify",
          "surfaces", "recgen.select", "recgen.generate", "recgen.gate"}  # r5 (models): the observed surfaces, the internal profile; plus write.<layer>
PRICE = {"A100-80GB": .000694, "cpu_core": .0000131, "gib": .00000222}  # Modal list prices, $/s


def usd_per_s(gpus=2, cpu=32, gib=160):
    return gpus * PRICE["A100-80GB"] + cpu * PRICE["cpu_core"] + gib * PRICE["gib"]


def known_stage(name):
    base = name.split("@")[0]
    return base in STAGES or re.sub(r"\d+$", "", base) in STAGES or (name.startswith("write.") and len(name) > 6)


def torch_peaks(reset=False):
    """This process's torch max_memory_reserved per device, GiB (the per-process split of a whole-device peak)."""
    import torch
    if not torch.cuda.is_initialized():
        return []
    out = [round(torch.cuda.max_memory_reserved(d) / 2 ** 30, 2) for d in range(torch.cuda.device_count())]
    if reset:
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
    return out


def parse_smi(line):
    """'0, 12345, 81920' (index, memory.used MiB, memory.total MiB) -> (0, used GiB, total GiB); None for anything else."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    return int(parts[0]), int(parts[1]) / 1024, int(parts[2]) / 1024


class Vram(threading.Thread):
    """Whole-device memory in use (every process: main, vLLM, SAM 3D, splat, Open3D), sampled every `period` s.

    source 'nvml': one `nvidia-smi -lms` stream, device-wide under MPS too and no CUDA context in this process;
    'torch': torch.cuda.mem_get_info per device; 'auto': nvml when nvidia-smi exists, else torch.
    ponytail: samples kept in memory, 20/s x 3 numbers ~ 1 MB per hour; add a ring buffer if containers live for days."""

    def __init__(self, gpus=(0, 1), period=.05, source="auto"):
        super().__init__(daemon=True)
        self.gpus, self.period = list(gpus), period
        self.source = ("nvml" if shutil.which("nvidia-smi") else "torch") if source == "auto" else source
        self.t, self.gb, self.total_gb = [], [], [0.] * len(self.gpus)
        self.halt, self.proc, self.error = threading.Event(), None, None

    def start(self):
        super().start()
        return self

    def stop(self):
        self.halt.set()
        if self.proc:
            self.proc.terminate()

    def add(self, t, gb):
        self.t.append(t)
        self.gb.append(tuple(gb))

    def run(self):
        try:
            self.nvml() if self.source == "nvml" else self.torch()
        except Exception as error:  # a dead sampler must show in run.json, not stop the pipeline
            self.error = repr(error)[:500]

    def torch(self):
        import torch
        while not self.halt.is_set():
            now = []
            for i, d in enumerate(self.gpus):
                free, total = torch.cuda.mem_get_info(d)
                now.append((total - free) / 2 ** 30)
                self.total_gb[i] = total / 2 ** 30
            self.add(time.perf_counter(), now)
            time.sleep(self.period)

    def nvml(self):
        self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits",
                                      f"-lms={max(1, int(self.period * 1000))}"], stdout=subprocess.PIPE, text=True)
        now = [0.] * len(self.gpus)
        for line in self.proc.stdout:
            got = parse_smi(line)
            if self.halt.is_set():
                break
            if got is None or got[0] not in self.gpus:
                continue
            i = self.gpus.index(got[0])
            now[i], self.total_gb[i] = got[1], got[2]
            if i == len(self.gpus) - 1:  # one sample per tick, when the last device of the tick is read
                self.add(time.perf_counter(), now)

    def window(self, a, b):
        """Per-device peak (GiB) over samples in [a, b] (perf_counter), widened by one period so short stages get a sample."""
        n = len(self.t)
        lo, hi = bisect_left(self.t, a - self.period, 0, n), bisect_right(self.t, b + self.period, 0, n)
        if lo >= hi:
            return [None] * len(self.gpus)
        return [round(max(g[i] for g in self.gb[lo:hi]), 2) for i in range(len(self.gpus))]

    def now(self):
        return [round(x, 2) for x in self.gb[-1]] if self.gb else None


class Clock:
    """Stages of one analysis, in seconds from t0 = the MP4 bytes in the container (perf_counter; unix for child processes)."""

    def __init__(self):
        self.t0, self.t0_unix = time.perf_counter(), time.time()
        self.rows, self.layers, self.flags, self.marks, self.lock = [], {}, [], {}, threading.Lock()

    def now(self):
        return round(time.perf_counter() - self.t0, 3)

    def unix_to_s(self, unix):
        return round(unix - self.t0_unix, 3)

    def mark(self, name):
        """A named moment (first time only): 'cameras_put', 'sam3_done', ..."""
        with self.lock:
            self.marks.setdefault(name, self.now())

    def add(self, name, gpu, start, end, n=None, error=None):
        where = "cpu" if gpu is None else f"gpu{getattr(gpu, 'index', gpu)}"  # gpu: an int or a torch.device
        row = {"stage": name, "where": where, "start_s": round(start, 3), "end_s": round(end, 3), "s": round(end - start, 3), "n": n or {}}
        if error:
            row["error"] = error
        with self.lock:
            self.rows.append(row)
            if not known_stage(name):
                self.flags.append(f"unknown stage name {name!r} (section 8's fixed set)")

    @contextmanager
    def stage(self, name, gpu=None, n=None, sync=True):
        start, error = self.now(), None
        try:
            yield
        except BaseException as e:
            error = repr(e)[:300]
            raise
        finally:
            if sync and gpu is not None and error is None:
                import torch
                torch.cuda.current_stream(gpu).synchronize()
            self.add(name, gpu, start, self.now(), n, error)

    def external(self, name, gpu=None, start_unix=0., end_unix=0., n=None):
        self.add(name, gpu, self.unix_to_s(start_unix), self.unix_to_s(end_unix), n)

    def layer(self, layer, seq, **fields):
        """Upsert one patch's row: sent_s first, written_s after the commit; bytes, version, served_s_unix as known."""
        with self.lock:
            self.layers.setdefault(seq, {"layer": layer, "seq": seq}).update(fields)

    def report(self, vram=None, price_per_s=None, **meta):
        """run.json so far (the timing layer is this, rewritten at every commit)."""
        end = self.now()
        with self.lock:
            rows, flags = [dict(r) for r in self.rows], list(self.flags)
            layers = [dict(v) for _, v in sorted(self.layers.items())]
        gpu_peak = []
        if vram is not None:
            total = [round(t, 1) for t in vram.total_gb]
            for r in rows:
                r["peak_gb"] = vram.window(self.t0 + r["start_s"], self.t0 + r["end_s"])
                r["over_90"] = [p is not None and t > 0 and p > FLAG_SHARE * t for p, t in zip(r["peak_gb"], total)]
            n = len(vram.t)
            lo = bisect_left(vram.t, self.t0, 0, n)
            for i, gpu in enumerate(vram.gpus):
                if lo >= n:
                    break
                k = max(range(lo, n), key=lambda j: vram.gb[j][i])
                at = round(vram.t[k] - self.t0, 3)
                active = [r["stage"] for r in rows if r["start_s"] - vram.period <= at <= r["end_s"] + vram.period]
                peak = round(vram.gb[k][i], 2)
                gpu_peak.append({"gpu": gpu, "total_gb": total[i], "peak_gb": peak, "at_s": at, "stages_active": active})
                if total[i] and peak > FLAG_SHARE * total[i]:
                    flags.append(f"gpu{gpu} {peak:.1f} GiB > 90% at {at:.1f} s ({', '.join(active) or 'no stage open'})")
            if vram.error:
                flags.append(f"vram sampler stopped: {vram.error}")
            meta.setdefault("vram_source", vram.source)
            meta.setdefault("memory_unit", "GiB (2^30 bytes) in every *_gb field")
        return {"schema": "panoptes-fast-run-v1", **meta, "clock": "seconds from the MP4 bytes in the container (t0_unix)",
                "t0_unix": self.t0_unix, "elapsed_s": end, "stages": sorted(rows, key=lambda r: (r["start_s"], r["stage"])), "marks": dict(self.marks),
                "gpu_peak": gpu_peak, "flags": flags, "layers": layers,
                "usd_estimate": round(end * price_per_s, 4) if price_per_s else None}


def self_check():
    assert parse_smi("1, 40960, 81920\n") == (1, 40., 80.) and parse_smi("[N/A]") is None
    assert known_stage("sam3.person@gpu1") and known_stage("da3.shot12") and known_stage("write.cameras")
    assert not known_stage("da3") and not known_stage("write.")
    clock = Clock()
    vram = Vram([0, 1], source="torch")  # never started: samples written by hand below
    vram.total_gb = [80., 80.]
    t = clock.t0
    for k, (a, b) in enumerate([(10, 20), (30, 70), (75, 40), (12, 20)]):  # at 0.0, 0.05, 0.10, 0.15 s
        vram.add(t + .05 * k, (a, b))
    clock.add("da3.shot1", 0, 0.0, .06)
    clock.add("sam3.person@gpu1", 1, .09, .11)
    clock.add("mystery", None, .2, .3)
    clock.external("sam3d.generate", 0, clock.t0_unix + .1, clock.t0_unix + .12, {"calls": 1})
    clock.layer("cameras", 2, version=1, sent_s=.08)
    clock.layer("cameras", 2, written_s=.12, bytes=10)
    with clock.stage("cuts"):
        pass
    try:
        with clock.stage("lift", gpu=0):
            raise KeyError("x")
    except KeyError:
        pass
    r = clock.report(vram, price_per_s=usd_per_s(), report="r", site="me340")
    rows = {x["stage"]: x for x in r["stages"]}
    assert rows["da3.shot1"]["peak_gb"] == [75, 70], "widened by one period: samples 0..0.10 s"
    assert rows["sam3.person@gpu1"]["peak_gb"] == [75, 70] and rows["sam3.person@gpu1"]["over_90"] == [True, False]
    assert rows["sam3d.generate"]["start_s"] == .1 and rows["sam3d.generate"]["n"] == {"calls": 1}
    assert "KeyError" in rows["lift"]["error"], "a stage that raised is still recorded"
    assert r["gpu_peak"][0]["peak_gb"] == 75 and r["gpu_peak"][0]["at_s"] == .1
    assert set(r["gpu_peak"][0]["stages_active"]) >= {"sam3.person@gpu1", "sam3d.generate", "da3.shot1"}
    assert r["gpu_peak"][1]["peak_gb"] == 70 and not any(f.startswith("gpu1") for f in r["flags"])
    assert any(f.startswith("gpu0 75.0 GiB > 90%") for f in r["flags"]), r["flags"]
    assert any("mystery" in f for f in r["flags"]), "names outside section 8's set are flagged"
    assert r["layers"] == [{"layer": "cameras", "seq": 2, "version": 1, "sent_s": .08, "written_s": .12, "bytes": 10}]
    assert r["usd_estimate"] is not None and json.loads(json.dumps(r)) == r, "run.json is plain JSON"
    assert Vram([0], source="torch").window(0, 1) == [None]
    print("instrument self-check passed: stage windows, shared peaks, 90% flags, layer upsert, child-process times")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--self-check"], __doc__
    self_check()
