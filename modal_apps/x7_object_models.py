"""X7: per-unique-object models from the best views of a video (fast_report/x7.py), on fb/a-core's merged objects.

One container, 2 x A100-80GB, MPS; every model resident from boot (cold start and loads reported apart, never in the
analysis time): RecGen (non-commercial, demo only) x RECGEN_GPUS, SAM 3D s1cfg12 x SAM3D_GPUS, CPU_PROCS select/gate/fit
processes. The core's hand-off comes from the fixture fast_report_app.py::x7_fixtures dumped (the same tensors the core
holds in memory at objects.v2); t0 = that hand-off is in this container's memory. Analysis time = t0 -> each model's gate
decided and its display GLB written (/v/x7, committed once per video at the end: commit time reported).

  M=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal
  $M run modal_apps/x7_object_models.py --out RUNS/fx-x7-object-models-NNN/models [--clips a,b] [--max-objects 100]
  python modal_apps/x7_object_models.py --evaluate RUNS/fx-x7-object-models-NNN   # local: delivered overlap, results.json
"""
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import modal
import numpy as np

WT = Path(__file__).resolve().parents[1]
LOCAL = modal.is_local()
sys.path[:0] = [str(WT), str(WT / "modal_apps"), str(WT / "scripts")] if LOCAL else ["/repo", "/repo/scripts", "/repo/modal_apps"]
PHASE2 = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
CPU, MEM_GIB, CPU_PROCS = 32, 96, 20
RECGEN_GPUS, SAM3D_GPUS = (0, 0, 1, 1), (0, 1)
PRICE = {"A100-80GB": .000694, "cpu": .0000131, "gib": .00000222}
USD_PER_S = 2 * PRICE["A100-80GB"] + CPU * PRICE["cpu"] + MEM_GIB * PRICE["gib"]
METHODS = ("recgen", "sam3d_b", "sam3d_c", "param")

app = modal.App("panoptes-x7-object-models")
out_volume = modal.Volume.from_name("panoptes-x7", create_if_missing=True)
VOLS = {"/weights": modal.Volume.from_name("panoptes-sam3d-weights"), "/cache": modal.Volume.from_name("panoptes-lucida-weights"),
        "/v/layers": modal.Volume.from_name("panoptes-fb-layers"), "/v/x7": out_volume}
if LOCAL:
    from fast_report import sam3d as fb_sam3d, x7 as fx7
    base = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")  # bench_b's base: its layers are cached
            .pip_install("numpy==2.2.6", "opencv-python-headless==4.11.0.86", "scipy==1.15.3", "pydantic==2.11.7", "trimesh==4.6.10"))
    image = (fx7.with_recgen(fb_sam3d.with_envs(base))
             .add_local_dir(WT / "fast_report", "/repo/fast_report", ignore=["**/__pycache__/**"])
             .add_local_dir(WT / "scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
             .add_local_dir(WT / "modal_apps", "/repo/modal_apps", ignore=["**/__pycache__/**"])
             .add_local_dir(WT / "ehs_spatial", "/repo/ehs_spatial", ignore=["**/__pycache__/**"]))
else:
    image = modal.Image.debian_slim()


def stats(values):
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), 2), "p10": round(float(np.percentile(v, 10)), 2),
            "p90": round(float(np.percentile(v, 90)), 2), "max": round(float(v.max()), 2), "sum": round(float(v.sum()), 1)}


@app.cls(image=image, gpu="A100-80GB:2", cpu=CPU, memory=MEM_GIB * 1024, volumes=VOLS, timeout=3000, retries=0, max_containers=1,
         scaledown_window=20)
class ObjectModels:
    @modal.enter()
    def boot(self):
        import complete_video_objects as cvo
        from fast_report import sam3d, x7
        from fast_report.instrument import Vram
        self.entered = time.time()
        env = {"CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe", "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log"}
        for d in env.values():
            Path(d).mkdir(parents=True, exist_ok=True)
        os.environ.update(env)
        mps = subprocess.run(["nvidia-cuda-mps-control", "-d"], capture_output=True, text=True).returncode == 0  # before any CUDA process
        self.vram = Vram([0, 1]).start()
        sam_env = [sam3d.worker_env(sam3d.SAM3D_PY, CUDA_VISIBLE_DEVICES=g, HF_HOME=f"{sam3d.WEIGHTS}/huggingface", HF_HUB_OFFLINE=1,
                                    LIDRA_SKIP_INIT="true", TORCH_HOME=sam3d.TORCH_HUB, CUDA_HOME="/usr/local/cuda",
                                    SAM3D_MODEL_REVISION=cvo.SAM3D["modelRevision"]) for g in SAM3D_GPUS]
        rg_env = []
        for g in RECGEN_GPUS:
            e = sam3d.worker_env(x7.RECGEN_PY, CUDA_VISIBLE_DEVICES=g, ATTN_BACKEND="xformers", SPCONV_ALGO="native", HF_HUB_OFFLINE=1)
            e["PYTHONPATH"] += os.pathsep + x7.RECGEN_DIR
            rg_env.append(e)
        cpu_env = sam3d.worker_env(sam3d.GATE_PY, OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1)
        t = time.time()
        self.sam = sam3d.Pool([sam3d.SAM3D_PY, "-c", "from fast_report.sam3d import sam3d_worker; sam3d_worker()"], len(SAM3D_GPUS), sam_env, "sam3d")
        self.recgen = sam3d.Pool([x7.RECGEN_PY, "-c", "from fast_report.x7 import recgen_worker; recgen_worker()"], len(RECGEN_GPUS), rg_env, "recgen")
        self.cpu = sam3d.Pool([sam3d.GATE_PY, "-c", "from fast_report.x7 import cpu_worker; cpu_worker()"], CPU_PROCS, cpu_env, "cpu")
        rec = {"entered_unix": self.entered, "mps": mps, "gpus": subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.splitlines()}
        for name, pool in (("cpu", self.cpu), ("sam3d", self.sam), ("recgen", self.recgen)):
            try:
                rec[name] = pool.ready(900)
            except Exception as error:  # noqa: BLE001  a dead pool is reported, the others still run
                rec[name] = {"error": str(error)[-3000:]}
            rec[f"{name}_ready_s"] = round(time.time() - t, 1)
        rec["ready_s"] = round(time.time() - self.entered, 1)
        rec["vram_after_boot_gib"] = self.vram.now()
        rec["note"] = "cold start: model loads and warm-up calls, never part of the analysis time"
        self.boot_record = rec

    @modal.exit()
    def stop(self):
        for p in ("cpu", "sam", "recgen"):
            if hasattr(self, p):
                getattr(self, p).close()

    @modal.method()
    def boot_info(self):
        return self.boot_record

    @modal.method()
    def run(self, name: str, fixture: str, out: str, options: dict):
        import complete_video_objects as cvo
        from fast_report import sam3d, x7
        from fast_report.instrument import Clock
        # ---------- the core's hand-off into memory (not analysis time: in a report it is already there)
        t_load = time.time()
        folder = Path("/v/layers/x7") / fixture / name
        meta = json.loads((folder / "fixture.json").read_text())
        z = np.load(folder / "fixture.npz")
        shots = [{"index": s["index"], "keys": z[f"s{s['index']}_keys"].tolist(), "depth_m": z[f"s{s['index']}_depth"], "c2w_m": z[f"s{s['index']}_c2w"],
                  "K": z[f"s{s['index']}_K"], "person": z[f"s{s['index']}_person"]} for s in meta["shots"]]
        plane = {s["index"]: s["plane"] for s in meta["shots"]}
        objs = []
        for i, o in enumerate(meta["objects"]):
            fr, lg = z[f"o{i}_frames"], z[f"o{i}_logits"]
            objs.append({**o, "masks_lr": {int(f): lg[j] for j, f in enumerate(fr)}})
        frames = sam3d.frames_buffer(meta["n_frames"], name=f"x7-{name}-frames.npy")
        for f, img in zip(z["frame_ids"], z["frames"]):
            frames[int(f)] = img
        frames.flush()
        fixture_load_s = round(time.time() - t_load, 2)
        # ---------- t0
        clock = Clock()
        with clock.stage("x7.rank_stage"):
            ranked = sam3d.rank(objs)[:options.get("max_objects", 100)]
            spec, keys = sam3d.stage(ranked, shots, str(frames.filename))
        rec = {o["id"]: {"rank": r, "id": o["id"], "shot": int(o["shot"]), "word": o["word"], "label": (o.get("cascade") or {}).get("label") or o.get("label"),
                         "methods": {}} for r, o in enumerate(ranked)}
        sel, tiles, lowres, events, pending = {}, {}, {}, queue.Queue(), [0]
        fps = meta["fps"]

        def submit(pool, msg, prio, kind, r, method=None):
            pending[0] += 1
            pool.submit(msg, prio).add_done_callback(lambda f: events.put((kind, r, method, f)))
        for r, o in enumerate(ranked):
            submit(self.cpu, {"op": "select", "src": spec, "shot": int(o["shot"]), "key": keys[o["id"]], "obj": {"centroid_m": o["centroid_m"]}},
                   (1, r), "selected", r)
        glb_root = Path("/v/x7") / out / name / "models"
        while pending[0]:
            kind, r, method, fut = events.get()
            pending[0] -= 1
            o = ranked[r]
            R = rec[o["id"]]
            try:
                res = fut.result()
            except Exception as error:  # noqa: BLE001  one object's failure is recorded, the rest go on
                R.setdefault("errors", []).append(f"{kind} {method}: {str(error)[-1500:]}")
                if method:
                    R["methods"].setdefault(method, {})["error"] = str(error)[-300:]
                continue
            if kind == "selected":
                clock.external("x7.select", None, res["start_unix"], res["end_unix"], n={"object": o["id"]})
                lowres[o["id"]] = (res["frames"], res["lowres"])
                R["selection"] = {k: res.get(k) for k in ("eligible", "reason", "views_with_depth", "good_views", "gen", "held", "min_pair_deg",
                                                          "recgen_views", "sam3d_b_rejected", "select_s", "select_parts_s")}
                R["observed_s"] = [round(min(res["frames"]) / fps, 2), round(max(res["frames"]) / fps, 2)] if res["frames"] else None
                if not res["eligible"]:
                    continue
                tiles.setdefault(o["id"], {})["crop"] = res["held_tile"]
                gen, held = [g["frame"] for g in res["gen"]], res["held"]["frame"]
                sel[r] = {"src": spec, "shot": int(o["shot"]), "key": keys[o["id"]], "gen": gen, "held": held, "crop": res["held_crop"]}
                now = clock.now()
                if res.get("recgen_job"):
                    R["methods"]["recgen"] = {"attempted": True, "views": len(res["recgen_job"]["views"]), "dispatched_s": now}
                    sel[r]["recgen_anchor"] = res["recgen_views"][0]
                    submit(self.recgen, res["recgen_job"], (r,), "generated", r, "recgen")
                else:
                    R["methods"]["recgen"] = {"attempted": True, "input_rejected": "no generation view's crop keeps mask pixels with depth"}
                if res["sam3d_b_rejected"]:
                    R["methods"]["sam3d_b"] = {"attempted": True, "prepare_rejected": res["sam3d_b_rejected"]}
                j = res.get("sam3d_job")
                if j:
                    R["methods"][j["method"]] = {"attempted": True, "view": j["frame"], "dispatched_s": now}
                    sel[r][j["method"]] = j["frame"]
                    submit(self.sam, {k: j[k] for k in ("rgb", "mask", "pointmap", "seed")}, (r,), "generated", r, j["method"])
                elif res["sam3d_b_rejected"]:
                    R["methods"]["sam3d_c"] = {"attempted": True, "prepare_rejected": "no other view passes the prepare rule"}
                pk = x7.param_kind([o["word"], R["label"]])
                if pk and plane.get(int(o["shot"])):
                    R["methods"]["param"] = {"attempted": True, "kind": pk, "dispatched_s": now}
                    submit(self.cpu, {"op": "fit", **sel[r], "kind": pk, "up": plane[int(o["shot"])]["normal"],
                                      "floor_point": plane[int(o["shot"])]["point_m"], "colour": res["colour"]}, (0, r), "fitted", r, "param")
            elif kind == "generated":
                M = R["methods"][method]
                gpu = int(res["gpu"]) if method == "recgen" else SAM3D_GPUS[res["worker"]]
                clock.external("recgen.generate" if method == "recgen" else "sam3d.generate", gpu, res["start_unix"], res["end_unix"], n={"object": o["id"]})
                M.update(generate_s=round(res["seconds"], 2), gpu=gpu, max_reserved_gb=res.get("max_reserved_gb"),
                         started_s=clock.unix_to_s(res["start_unix"]), generated_s=clock.unix_to_s(res["end_unix"]), faces=int(len(res["faces"])))
                mesh = {k: res[k] for k in ("vertices", "faces", "colors")}
                if method != "recgen":
                    mesh["objectToCamera"] = res["objectToCamera"]
                submit(self.cpu, {"op": "gate", **sel[r], "kind": "recgen" if method == "recgen" else "sam3d",
                                  "source_frame": sel[r]["recgen_anchor"] if method == "recgen" else sel[r][method], "mesh": mesh,
                                  "glb_path": str(glb_root / f"{o['id']}-{method}.glb")}, (0, r), "gated", r, method)
            else:
                clock.external("x7.gate" if kind == "gated" else "x7.fit", None, res["start_unix"], res["end_unix"], n={"object": o["id"], "method": method})
                M = R["methods"][method]
                if "error" in res:
                    M["error"] = res["error"]
                    continue
                M.update({k: v for k, v in res.items() if k not in ("tile", "start_unix", "end_unix", "pid", "worker")})
                M.update(accepted=bool(res["gate"]["accepted_source_consistency"]), decided_s=clock.unix_to_s(res["end_unix"]))
                if M["accepted"]:
                    tiles.setdefault(o["id"], {})[method] = res["tile"]
        t = time.time()
        out_volume.commit()
        commit_s = round(time.time() - t, 2)
        rep = clock.report(self.vram, price_per_s=USD_PER_S, report=f"x7-{name}", site=name)
        by_stage = {}
        for row in rep["stages"]:
            s = by_stage.setdefault(row["stage"], {"n": 0, "seconds": [], "peak_gib": [0., 0.]})
            s["n"] += 1
            s["seconds"].append(row["s"])
            s["peak_gib"] = [max(a, b or 0.) for a, b in zip(s["peak_gib"], row.get("peak_gb") or [0., 0.])]
        stage_mem = {k: {"n": v["n"], "s": stats(v["seconds"]), "peak_gib_per_gpu": [round(x, 2) for x in v["peak_gib"]],
                         "over_90": [x > .9 * 80 for x in v["peak_gib"]]} for k, v in by_stage.items()}
        summary = {m: self.summarize(rec, m) for m in METHODS}
        summary["sam3d_b_or_c"] = self.summarize(rec, "sam3d_b", also="sam3d_c")
        lr = io.BytesIO()
        np.savez_compressed(lr, **{f"{oid}__frames": np.asarray(f, np.int32) for oid, (f, _) in lowres.items()},
                            **{f"{oid}__masks": m for oid, (_, m) in lowres.items()})
        return {"name": name, "fixture": {"folder": str(folder), "report": meta["report"], "load_s_not_analysis": fixture_load_s,
                                          "objects_in_core": len(objs), "ranked_attempt_list": len(ranked)},
                "t0_unix": clock.t0_unix, "elapsed_s": rep["elapsed_s"], "commit_s": commit_s, "summary": summary, "objects": list(rec.values()),
                "stages": stage_mem, "gpu_peak": rep["gpu_peak"], "flags": [f for f in rep["flags"] if "unknown stage" not in f],
                "usd_estimate_analysis": rep["usd_estimate"], "tiles": tiles, "lowres_npz": lr.getvalue(),
                "worker_boot": {k: self.boot_record.get(k) for k in ("sam3d", "recgen")}}

    @staticmethod
    def summarize(rec, method, also=None):
        rows = [R["methods"][m] for R in rec.values() for m in (method, also) if m and m in R["methods"]
                and not (also and m == method and "prepare_rejected" in R["methods"][m] and also in R["methods"])]
        attempted = [m for m in rows if m.get("attempted")]
        generated = [m for m in attempted if "generate_s" in m or "fit_s" in m]
        accepted = [m for m in attempted if m.get("accepted")]
        decided = sorted(m["decided_s"] for m in accepted)
        return {"attempted": len(attempted), "prepare_or_input_rejected": sum("prepare_rejected" in m or "input_rejected" in m for m in attempted),
                "generated": len(generated), "errors": sum("error" in m for m in attempted), "gated": sum("gate" in m for m in attempted),
                "accepted_held_out": len(accepted), "acceptance_rate": round(len(accepted) / max(len(attempted), 1), 3),
                "acceptance_rate_of_generated": round(len(accepted) / max(len(generated), 1), 3),
                "first_accepted_s": decided[0] if decided else None, "last_accepted_s": decided[-1] if decided else None,
                "last_decided_s": max((m["decided_s"] for m in attempted if "decided_s" in m), default=None),
                "inference_s": stats(m.get("generate_s") for m in attempted), "fit_s": stats(m.get("fit_s") for m in attempted),
                "gate_s": stats(m.get("gate_s") for m in attempted), "queue_wait_s": stats(m["started_s"] - m["dispatched_s"] for m in attempted
                                                                                         if "started_s" in m and "dispatched_s" in m),
                "max_reserved_gib_per_process": max((m.get("max_reserved_gb") or 0 for m in attempted), default=None)}


def contact_sheet(result, path, per_row=2):
    """Per object with an accepted model: held-out crop | RecGen | SAM 3D (b or c) | parametric, each from the held-out camera."""
    import cv2
    t, cap = 128, 16
    rows = []
    blank = np.full((t, t, 3), 235, np.uint8)
    for R in result["objects"]:
        tl = result["tiles"].get(R["id"], {})
        if not any(m in tl for m in METHODS):
            continue
        cells = [cv2.imdecode(np.frombuffer(tl["crop"], np.uint8), cv2.IMREAD_COLOR)]
        for m in ("recgen", "sam3d", "param"):
            k = next((x for x in METHODS if x.startswith(m) and x in tl), None)
            cells.append(cv2.imdecode(np.frombuffer(tl[k], np.uint8), cv2.IMREAD_COLOR) if k else blank)
        strip = np.hstack(cells)
        head = np.full((cap, strip.shape[1], 3), 255, np.uint8)
        ious = " ".join(f"{m.split('_')[-1]}:{R['methods'][m]['gate']['silhouette_iou']:.2f}" for m in METHODS if m in tl)
        cv2.putText(head, f"{R['id']} {str(R['label'] or R['word'])[:18]} {ious}", (3, 12), cv2.FONT_HERSHEY_SIMPLEX, .36, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.vstack([head, strip]))
    if not rows:
        return 0
    rows += [np.full_like(rows[0], 255)] * (-len(rows) % per_row)
    grid = np.vstack([np.hstack([np.pad(r, ((0, 0), (0, 6), (0, 0)), constant_values=255) for r in rows[i:i + per_row]]) for i in range(0, len(rows), per_row)])
    top = np.full((24, grid.shape[1], 3), 255, np.uint8)
    cv2.putText(top, f"{result['name']}: held-out crop | RecGen | SAM 3D | parametric (rendered from the held-out camera; blank = not accepted)",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), np.vstack([top, grid]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return len(rows)


@app.local_entrypoint()
def main(out: str, clips: str = "me340-165,samsclub-337,walmart-190", fixture: str = "fx-x7-001", max_objects: int = 100):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    om = ObjectModels()
    called = time.time()
    boot = om.boot_info.remote()
    boot["client_called_unix"] = called
    (out / "boot.json").write_text(json.dumps(boot, indent=1, default=str))
    print("boot:", json.dumps({k: v for k, v in boot.items() if k.endswith("_s") or k == "mps"}), flush=True)
    for name in clips.split(","):
        started = time.time()
        res = om.run.remote(name, fixture, f"{out.parent.name}/{out.name}", {"max_objects": max_objects})
        res["client_wall_s"] = round(time.time() - started, 1)
        (out / f"{name}-lowres.npz").write_bytes(res.pop("lowres_npz"))
        n = contact_sheet(res, out / f"{name}-contact.jpg")
        tiles = res.pop("tiles")
        res["contact_sheet"] = {"file": f"{name}-contact.jpg", "objects": n, "tiles": sum(len(v) - 1 for v in tiles.values())}
        (out / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
        print(name, json.dumps(res["summary"], default=str)[:3000], "elapsed", res["elapsed_s"], flush=True)
    (out / "done.json").write_text(json.dumps({"client_called_unix": called, "client_done_unix": time.time()}, indent=1))


# ---------------------------------------------------------------- local evaluation: overlap with the delivered report
DELIVERED = {"me340-165": ("me340-entity-names-200", "me340-masks-194", "me340-object-models-303-merged", "c40fbd08"),
             "samsclub-337": ("samsclub-a2-entity-names-288", "samsclub-masks-162", "samsclub-a2-object-models-303-merged", "913daf2a"),
             "walmart-190": ("walmart-entity-names-261", "walmart-masks-182", "walmart-object-models-303-merged", "32cec650")}


def overlap(name, lowres_path, accepted):
    """Our objects vs the delivered report's models: mask IoU >= 0.5 on a shared frame (ours on its object keyframes, theirs
    on the delivered mask frames <= 1 frame away), both at 160x120 on the 4:3 crop (fb/a-core evaluate's rule, downscaled).
    accepted: {method: set of our object ids}. -> counts per method and the matched pairs."""
    import cv2
    names, masks, merged, pub = DELIVERED[name]
    runs = PHASE2 / "runs"
    omap = {e["entityId"]: e for e in json.loads((runs / names / "object-map.json").read_text())["entities"]}
    models = json.loads((runs / merged / "merge.json").read_text())["choice"]
    root = runs / masks / "object-a"
    have = {int(p.name.split("-")[1]) for p in root.iterdir() if p.name.startswith("frame-")}
    theirs = {}  # frame -> [(entity, mask 160x120)]
    for ent in models:
        for ob in omap[ent]["observations"]:
            _, f, inst = ob.split(":")
            path = root / f"frame-{int(f):05d}" / f"instance-{inst}-mask.png"
            if path.exists():
                m = cv2.resize((cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8), (160, 120), interpolation=cv2.INTER_AREA) > 0
                theirs.setdefault(int(f), []).append((ent, m))
    z = np.load(lowres_path)
    best = {}  # our object -> (iou, entity)
    for key in z.files:
        if not key.endswith("__frames"):
            continue
        oid = key[:-8]
        ours = np.unpackbits(z[oid + "__masks"], axis=1)[:, :160 * 120].reshape(-1, 120, 160).astype(bool)
        for f, m in zip(z[key], ours):
            g = min(have, key=lambda x: abs(x - f))
            if abs(g - f) > 1 or not m.any():
                continue
            for ent, t in theirs.get(g, []):
                iou = (m & t).sum() / max((m | t).sum(), 1)
                if iou > best.get(oid, (0, None))[0]:
                    best[oid] = (float(iou), ent)
    matched = {oid: ent for oid, (iou, ent) in best.items() if iou >= .5}
    out = {"publication": pub, "delivered_models": len(models), "our_objects_matching_a_delivered_model": len(matched),
           "delivered_models_matched_by_any_attempted_object": len(set(matched.values()))}
    for method, ids in accepted.items():
        hit = {matched[i] for i in ids if i in matched}
        out[method] = {"accepted": len(ids), "accepted_matching_delivered": sum(i in matched for i in ids), "delivered_models_covered": len(hit),
                       "delivered_coverage": round(len(hit) / max(len(models), 1), 3),
                       "delivered_by_generator": {k: sum(models[e] == k for e in hit) for k in ("recgen", "sam3d", "box")}}
    out["pairs"] = {oid: {"entity": ent, "iou": round(best[oid][0], 3), "delivered_generator": models[ent]} for oid, ent in matched.items()}
    return out


def self_check_overlap():
    """packbits round trip of the lowres masks, as the container writes them."""
    m = np.zeros((120, 160), bool)
    m[10:50, 20:90] = True
    back = np.unpackbits(np.packbits(m)[None], axis=1)[:, :160 * 120].reshape(-1, 120, 160).astype(bool)[0]
    assert np.array_equal(back, m)
    print("x7_object_models self-check passed: lowres mask packing")


def object_type(words):
    """Parametric kind, else complete_video_objects' EHS group, else 'other'."""
    import complete_video_objects as cvo
    from fast_report import x7
    kind = x7.param_kind(words)
    if kind:
        return kind
    return next((g for g, terms in cvo.RELEVANT.items() if any(w and cvo.matches(w, terms) for w in words)), "other")


def container_usd(entered, ended, scaledown, per_s):
    return round((ended - entered + scaledown) * per_s, 3)


def aggregate(run_dir, models="models"):
    """Every reported number -> run_dir/results.json (the core's run, the model container's boot and runs, overlap, spend)."""
    run_dir = Path(run_dir)
    core = json.loads((run_dir / "core/summary.json").read_text())
    core_rate = 2 * PRICE["A100-80GB"] + 32 * PRICE["cpu"] + 160 * PRICE["gib"]  # fast_report_app's container
    core_runs = {}
    ends = []
    for row in core["runs"]:
        rj = json.loads((run_dir / "core" / row["report"] / "run.json").read_text())
        ends.append(rj["t0_unix"] + rj["analysis_wall_s"] + (rj.get("x7_fixture") or {}).get("dump_s", 0))
        core_runs[row["clip"]] = {"report": row["report"], "objects": row["objects"], "layers_written_s": row["layers"],
                                  "core_outputs_ready_s": row["layers"]["objects.v2"], "gpu_peak_gib": row["gpu_peak_gb"], "flags": row["flags"],
                                  "fixture": row["fixture"], "analysis_usd_estimate": row["usd_estimate"]}
    spend = {"core_container_usd": container_usd(core["boot"]["entered_unix"], max(ends), 60, core_rate)}
    videos, facts, by_type = {}, [], {}
    for mdir in sorted(run_dir.glob("models*")) + sorted(run_dir.glob("smoke*")):
        boot = json.loads((mdir / "boot.json").read_text())
        done = json.loads((mdir / "done.json").read_text()) if (mdir / "done.json").exists() else None
        if done:
            spend[f"{mdir.name}_container_usd"] = container_usd(boot["entered_unix"], done["client_done_unix"], 20, USD_PER_S)
    mdir = run_dir / models
    boot = json.loads((mdir / "boot.json").read_text())
    for res_path in sorted(mdir.glob("*.json")):
        if res_path.name in ("boot.json", "done.json"):
            continue
        res = json.loads(res_path.read_text())
        name = res["name"]
        for R in res["objects"]:
            t = object_type([R["word"], R["label"]])
            for m, M in R["methods"].items():
                row = by_type.setdefault(t, {}).setdefault(m, {"attempted": 0, "accepted": 0})
                row["attempted"] += bool(M.get("attempted"))
                row["accepted"] += bool(M.get("accepted"))
            P = R["methods"].get("param")
            if P and P.get("accepted"):
                sel = R["selection"]
                facts.append({"video": name, "object": R["id"], "label": R["label"] or R["word"], "kind": P["kind"], "facts": P["facts"],
                              "observed_s": R["observed_s"], "evidence_frames": {"fit": [g["frame"] for g in sel["gen"]], "held_out": sel["held"]["frame"]},
                              "held_out_iou": P["gate"]["silhouette_iou"], "held_out_depth_median": P["gate"]["relative_depth_median"],
                              "residual_median_m": P["residual_median_m"], "status": "estimated (scale from floor plane + assumed 1.6 m camera height)"})
        sel_rows = [R["selection"] for R in res["objects"] if "selection" in R]
        videos[name] = {"core": core_runs.get(name), "fixture_load_s_not_analysis": res["fixture"]["load_s_not_analysis"],
                        "objects_in_core": res["fixture"]["objects_in_core"], "ranked_attempt_list": res["fixture"]["ranked_attempt_list"],
                        "selection": {"eligible": sum(bool(s.get("eligible")) for s in sel_rows), "not_eligible": sum(not s.get("eligible") for s in sel_rows),
                                      "generation_views": {str(k): sum(len(s.get("gen") or []) == k for s in sel_rows if s.get("eligible")) for k in range(1, 5)},
                                      "select_s": stats(s.get("select_s") for s in sel_rows)},
                        "analysis_elapsed_s": res["elapsed_s"], "volume_commit_s": res["commit_s"], "methods": res["summary"],
                        "delivered_overlap": {k: v for k, v in (res.get("delivered_overlap") or {}).items() if k != "pairs"},
                        "per_gpu_peak_gib_by_stage": res["stages"], "gpu_peak": res["gpu_peak"], "flags_over_90": res["flags"],
                        "analysis_usd_estimate": res["usd_estimate_analysis"], "contact_sheet": res["contact_sheet"]}
    for t in by_type.values():
        for row in t.values():
            row["rate"] = round(row["accepted"] / max(row["attempted"], 1), 3)
    results = {"experiment": "X7 per-unique-object models from the best views (fx/x7-object-models)", "run": run_dir.name,
               "rules": {"clock": "t0 = the core's hand-off in the model container's memory; each model: its held-out gate decided and GLB written",
                         "held_out_gate": "build_lingbot_object_model.evaluate on a view never used to generate or place: IoU >= 0.65, rel. depth median <= 0.04, p95 <= 0.10, >= 300 px (DA3 grid 504x280)",
                         "views": f"good views >= {15} deg apart, up to 4 generate + 1 held out", "scale": "estimated", "models": "display only, never measurements",
                         "recgen_licence": "non-commercial research (demo only)"},
               "core_reused": {"branch": "fb/a-core (+ masks_lr hand-off, dump after the clock)", "runs": core_runs, "boot": core["boot"]},
               "model_container_boot": {k: boot.get(k) for k in ("ready_s", "cpu_ready_s", "sam3d_ready_s", "recgen_ready_s", "vram_after_boot_gib", "mps")} |
                                       {"sam3d_workers": boot.get("sam3d"), "recgen_workers": boot.get("recgen")},
               "videos": videos, "acceptance_by_type": by_type, "facts": facts, "spend_usd": {**spend, "total": round(sum(spend.values()), 3)}}
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    return results


if __name__ == "__main__":
    if sys.argv[1:2] == ["--self-check"]:
        self_check_overlap()
    else:
        assert sys.argv[1:2] == ["--evaluate"], __doc__
        run_dir = Path(sys.argv[2])
        models = sys.argv[3] if len(sys.argv) > 3 else "models"
        for res_path in sorted((run_dir / models).glob("*.json")):
            if res_path.name in ("boot.json", "done.json"):
                continue
            res = json.loads(res_path.read_text())
            accepted = {m: {R["id"] for R in res["objects"] if R["methods"].get(m, {}).get("accepted")} for m in METHODS}
            accepted["sam3d_b_or_c"] = accepted["sam3d_b"] | accepted["sam3d_c"]
            accepted["any_generator"] = accepted["recgen"] | accepted["sam3d_b_or_c"]
            accepted["any"] = accepted["any_generator"] | accepted["param"]
            res["delivered_overlap"] = overlap(res["name"], run_dir / models / f"{res['name']}-lowres.npz", accepted)
            res_path.write_text(json.dumps(res, indent=1, default=str))
            print(res["name"], json.dumps({k: v for k, v in res["delivered_overlap"].items() if k != "pairs"}))
        aggregate(run_dir, models)
