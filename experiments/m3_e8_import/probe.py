"""M3 experiment E8: where the import's time goes (a), and a cloud write path for report layers (b). README.md beside this file.

  modal run experiments/m3_e8_import/probe.py::stage     # once: upload c40fbd08's ME340 inputs to a Modal Volume
  modal run experiments/m3_e8_import/probe.py::profile   # (a) import_video_scene.py on a CPU container, throwaway Postgres + blob root
  modal run experiments/m3_e8_import/probe.py::layer     # (b) write report layers as blobs + a patch, serve them over HTTP

Nothing here touches the Mac's Postgres (54329), .platform/ or any published report; results go to runs/m3-exp-e8-import-*.
"""
import collections
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import modal

ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")  # every run records this path; the container links it to its copy
REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
DATA = modal.Volume.from_name("panoptes-m3-e8-import", create_if_missing=True)
LAYERS = modal.Volume.from_name("panoptes-m3-e8-layers", create_if_missing=True)
CPU, MEMORY_MIB = 4.0, 32768
CPU_USD, GIB_USD = 0.0000131, 0.00000222  # Modal list price per physical core-second and per GiB-second

# c40fbd08's import argv (tests/fixtures/delivered-303/me340.json, node IMPORT) without --republish: a new project in a throwaway DB
RUNS = {"droid": "runs/droid-me340-165-171", "depth": "runs/da3-posed-me340-223-shotc", "map": "runs/me340-entity-names-200",
        "masks": "runs/me340-masks-194", "clip": "data/clips/me340-165", "dynamic": "runs/me340-cutaway-dynamic-305"}
ARGV = ["--droid-run", ART / RUNS["droid"], "--depth-run", ART / RUNS["depth"], "--object-map", ART / RUNS["map"], "--masks", ART / RUNS["masks"],
        "--video", ART / RUNS["clip"] / "source-rgb.mp4", "--full-video", ART / RUNS["clip"] / "source-full.json",
        "--analysis", ART / "runs/me340-frame-selection-201/analysis.json", "--shell-glb", ART / "runs/me340-filled-225/textured-scene.glb",
        "--dynamic-scene", ART / RUNS["dynamic"] / "scene/scene.json", "--dynamic-analysis", ART / RUNS["dynamic"] / "analysis/analysis.json",
        "--video-events", ART / "runs/me340-events-197/events.json", "--splats", ART / "runs/me340-splat-232/splats.splat",
        "--dense-points", ART / "runs/me340-lingbot-map-222", "--inferred-floor", ART / "runs/me340-inferred-floor-244",
        "--models", ART / "runs/me340-object-models-303-merged/models", "--exclude-frames", "14:226",
        "--title", "E8 throwaway import of ME340 (c40fbd08 inputs)", "--request-suffix", "m3-e8", "--output-dir", "/tmp/throwaway/imports"]

app = modal.App("panoptes-m3-e8-import")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("postgresql", "libgl1", "libglib2.0-0", "libgomp1", "libegl1", "libx11-6")
         .pip_install("numpy==2.5.1", "opencv-python-headless==5.0.0.93", "trimesh==5.1.0", "scipy==1.18.0", "shapely==2.1.2",
                      "pillow==12.3.0", "open3d==0.19.0", "pydantic==2.13.4", "psycopg[binary]==3.3.5")  # the platform venv's pins
         .add_local_dir(REPO / "ehs_spatial", "/repo/ehs_spatial", ignore=["**/__pycache__/**"])
         .add_local_dir(REPO / "scripts", "/repo/scripts", ignore=["**/__pycache__/**"])
         .add_local_dir(REPO / "modal_apps", "/repo/modal_apps", ignore=["**/__pycache__/**"]))


def needed():
    """ART-relative files the import reads for ARGV: whole small runs, and only the walk views' masks and frames it keeps
    (every prompted view and the 8 largest class-agnostic views per entity, as import_video_scene.build_document keeps them)."""
    files = {f"{RUNS['droid']}/{n}" for n in ("run.json", "input-manifest.json", "prediction.npz")}
    files |= {f"{RUNS['depth']}/{n}" for n in ("metric-scale.json", "predicted-scene.glb")}
    files |= {f"{RUNS['clip']}/{n}" for n in ("clip.json", "source-rgb.mp4", "source-full.json", "source-full.mp4")}
    files |= {"runs/me340-frame-selection-201/analysis.json", "runs/me340-filled-225/textured-scene.glb", "runs/me340-filled-225/fill.json",
              f"{RUNS['dynamic']}/scene/scene.json", "runs/me340-lingbot-map-222/dense-points.glb", "runs/me340-lingbot-map-222/points.json",
              *(f"runs/me340-splat-232/{n}" for n in ("splats.splat", "splats.json", "refined-cameras.json"))}
    for folder in (RUNS["map"], "runs/me340-events-197", "runs/me340-inferred-floor-244", "runs/me340-object-models-303-merged/models",
                   f"{RUNS['dynamic']}/scene/dynamic", f"{RUNS['dynamic']}/analysis"):
        for root, _, names in os.walk(ART / folder, followlinks=True):
            files |= {str((Path(root) / n).relative_to(ART)) for n in names}
    walk = lambda o: not 14 <= int(o.split(":")[1]) < 226
    frames = set()
    for e in json.loads((ART / RUNS["map"] / "object-map.json").read_text())["entities"]:
        seen, box = [o for o in e["observations"] if walk(o)], e["observationBoxes"]
        area = lambda o: (box[o][2] - box[o][0]) * (box[o][3] - box[o][1])
        for o in [o for o in seen if not o.startswith("object:")] + sorted((o for o in seen if o.startswith("object:")), key=area, reverse=True)[:8]:
            label, index, instance = o.split(":")
            files |= {str(p.relative_to(ART)) for p in (ART / RUNS["masks"]).glob(f"{label}-*/frame-{int(index):05d}/instance-{instance}-mask.png")}
            frames.add(int(index))
    manifest = json.loads((ART / RUNS["droid"] / "input-manifest.json").read_text())
    return files | {f"{RUNS['clip']}/{manifest['frames'][i]['relative_path']}" for i in frames}


@app.local_entrypoint()
def stage():
    files = sorted(needed())
    size = sum((ART / f).stat().st_size for f in files)
    start = time.time()
    with DATA.batch_upload(force=True) as batch:
        for f in files:
            batch.put_file(ART / f, "/" + f)
    print(json.dumps({"files": len(files), "bytes": size, "upload_s": round(time.time() - start, 1)}))


def throwaway_postgres():
    """A Postgres that lives and dies with this container; returns its DSN."""
    import glob
    import shutil
    import subprocess
    bin_dir = sorted(glob.glob("/usr/lib/postgresql/*/bin"))[-1]
    os.makedirs("/tmp/pg", exist_ok=True)
    shutil.chown("/tmp/pg", "postgres")
    as_pg = lambda *a: subprocess.run(["runuser", "-u", "postgres", "--", *a], check=True, capture_output=True, text=True)
    as_pg(f"{bin_dir}/initdb", "-D", "/tmp/pg/data", "-A", "trust", "-U", "postgres")
    as_pg(f"{bin_dir}/pg_ctl", "-D", "/tmp/pg/data", "-l", "/tmp/pg/log", "-o", "-c listen_addresses=127.0.0.1 -p 5432 -k /tmp/pg", "-w", "start")
    as_pg(f"{bin_dir}/createdb", "-h", "127.0.0.1", "-p", "5432", "throwaway")
    return "postgresql://postgres@127.0.0.1:5432/throwaway", bin_dir.split("/")[-2]


# import_video_scene.py line where each piece of work starts (checked against the file in the container before use)
BUILD = [(234, "setup: object map, cuts, scale, DROID poses"), (288, "cameras + source images (read, rectify, PNG)"),
         (312, "room shell + point clouds"), (340, "textured shell load"), (341, "setup: floors, facts"),
         (346, "entities: observation masks (read, rectify, PNG)"), (357, "entities: measurements"),
         (363, "entities: observed surfaces + plan outlines"), (383, "entities: generated models (light_model + outline)"),
         (411, "entities: inferred floor model"), (421, "entities: fused cut / best view + outline"), (447, "entities: append"),
         (451, "movers: timed surfaces (light_model)"), (473, "movers: skeletons"), (499, "movers: motion summary"),
         (511, "video replay: outlines + videos"), (537, "splats"), (548, "video events"), (562, "provenance + metric check")]
RUN = [(580, "run: mono_room import + use_clip"), (586, "run: services + migrate"), (589, "run: create project"),
       (603, "run: build_document"), (604, "run: validate_document"), (607, "run: create + claim job"),
       (613, "run: finish_job (revision insert)"), (615, "run: commit_edits (schema v2 migration)"),
       (618, "run: create_publication"), (620, "run: write record")]


def section(table, line):
    return [name for start, name in table if start <= line][-1]


class Sampler(threading.Thread):
    """Wall-clock stack sampler of one thread (stdlib; the sandbox has no ptrace for py-spy), every 20 ms.

    The time since the previous sample goes to the previous sample's stack: a C call that holds the GIL (open3d) blocks the
    sampler until it returns, so charging the gap to the stack seen after it would credit the wrong code (run 1 did that).
    """

    def __init__(self, thread_id, every=.02):
        super().__init__(daemon=True)
        self.thread_id, self.every, self.halt = thread_id, every, threading.Event()
        self.sections, self.leaves, self.repo, self.total = collections.Counter(), collections.Counter(), collections.Counter(), 0

    def keys(self):
        frame = sys._current_frames().get(self.thread_id)
        stack = []
        while frame is not None:
            stack.append(frame)
            frame = frame.f_back
        if not stack:
            return None
        here = "outside run()"
        for f in reversed(stack):  # outermost first: the innermost import_video_scene frame decides
            if f.f_code.co_filename.endswith("import_video_scene.py") and f.f_code.co_name in ("run", "build_document"):
                here = section(BUILD if f.f_code.co_name == "build_document" else RUN, f.f_lineno)
        if any(f.f_code.co_name == "alternative_outline" for f in stack):
            here = "probe overhead: alternative outline (not import work)"
        leaf = stack[0].f_code
        mine = next((f.f_code for f in stack if f.f_code.co_filename.startswith("/repo/")), None)
        return here, f"{Path(leaf.co_filename).name}:{leaf.co_name}", mine and f"{Path(mine.co_filename).name}:{mine.co_name}"

    def run(self):
        previous, last = None, time.perf_counter()
        while True:
            stop = self.halt.wait(self.every)
            now, current = time.perf_counter(), self.keys()
            if previous:
                here, leaf, mine = previous
                self.total += 1
                self.sections[here] += now - last
                self.leaves[(here, leaf)] += now - last
                if mine:
                    self.repo[mine] += now - last
            previous, last = current, now
            if stop:
                return


def asset_label(m):
    """An asset kind fine enough to say where bytes and seconds go."""
    if m.get("kind") != "geometry":
        return m.get("kind")
    record = str(m.get("sourceRecordId", ""))
    if "decimation" in m:
        return "geometry: mover surface"
    if record.endswith(":skeleton"):
        return "geometry: mover skeleton"
    if record.endswith(":fused-surface"):
        return "geometry: fused cut"
    if "generator" in m:
        return "geometry: generated model" if record != "inferred-floor" else "geometry: inferred floor"
    if m.get("format") == "panoptes-mesh-v1" and not record.startswith("fused-room"):
        return "geometry: observed surface"
    return f"geometry: {record}"


def alternative_outline(outline, a, glb, exact):
    """Probe only (not import work): the same exact triangle union on the light display mesh light_model just made for this model,
    its seconds and its outline's IoU with the full-resolution outline the import keeps."""
    import io
    import shapely
    import trimesh
    shape = lambda r: shapely.unary_union([shapely.Polygon(p["exterior"], p["holes"]) for p in r["polygons"]])
    mesh = trimesh.load(io.BytesIO(glb), file_type="glb", force="mesh", process=False)
    start = time.perf_counter()
    lighter = outline(a[0], a[1], mesh, *a[3:])
    spent = time.perf_counter() - start
    try:
        full, light = shape(exact), shape(lighter)
        iou = full.intersection(light).area / full.union(light).area
    except Exception as error:  # an outline that shapely cannot rebuild is reported, not fatal
        iou = repr(error)
    return {"light_triangles": len(mesh.faces), "light_outline_s": spent, "light_outline_iou": iou}


def copy_parallel(source, target, workers=32):
    """Volume reads are latency-bound per file (shutil.copytree ran at ~3 files/s, M): copy many files at once."""
    from concurrent.futures import ThreadPoolExecutor
    import shutil
    files = [p for p in Path(source).rglob("*") if p.is_file()]

    def one(path):
        (Path(target) / path.relative_to(source)).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, Path(target) / path.relative_to(source))
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))
    return len(files)


@app.function(image=image, cpu=CPU, memory=MEMORY_MIB, volumes={"/data": DATA}, timeout=3600, retries=0, max_containers=1)
def profile_import():
    import subprocess
    import textwrap
    began = time.time()
    t = time.perf_counter()
    copied = copy_parallel("/data", "/root/art")
    copy_s = time.perf_counter() - t
    ART.parent.mkdir(parents=True, exist_ok=True)
    ART.symlink_to("/root/art")
    t = time.perf_counter()
    dsn, pg_version = throwaway_postgres()
    postgres_s = time.perf_counter() - t
    os.environ.update(PANOPTES_DATABASE_URL=dsn, PANOPTES_BLOB_ROOT="/tmp/throwaway/blobs")
    sys.path[:0] = ["/repo", "/repo/scripts", "/repo/modal_apps"]
    lines = Path("/repo/scripts/import_video_scene.py").read_text().splitlines()
    assert lines[233].startswith("def build_document") and lines[579].startswith("def run("), "BUILD/RUN line table is stale"

    sampler = Sampler(threading.get_ident())
    sampler.start()
    t0 = time.perf_counter()
    import import_video_scene as ivs
    from ehs_spatial.platform import postgres, storage
    import ehs_spatial.platform.reconstruction as reconstruction
    import attach_entities_to_replay
    import shapely
    import_s = time.perf_counter() - t0

    timers, kinds, last_put, kept = collections.defaultdict(lambda: [0, 0.]), collections.defaultdict(lambda: [0, 0, 0.]), [0., 0], {}
    calls, light = [], {}  # one row per _plan_projection / light_model call; the last light_model output

    def timed(owner, name, key=None, after=None):
        original = getattr(owner, name)

        def wrapper(*a, **k):
            start, result = time.perf_counter(), None
            try:
                result = original(*a, **k)
                return result
            finally:
                spent = time.perf_counter() - start
                timers[key or name][0] += 1
                timers[key or name][1] += spent
                if after:
                    after(spent, a, k, result)
        wrapper.original = original
        setattr(owner, name, wrapper)

    def where():
        frame = sys._getframe(2)
        while frame is not None and frame.f_code.co_name != "build_document":
            frame = frame.f_back
        return section(BUILD, frame.f_lineno) if frame else "outside build_document"

    def on_put(spent, a, k, result):
        last_put[:] = [spent, len(a[1])]

    def on_register(spent, a, k, result):
        row = kinds[asset_label(a[2].get("metadata", {}))]
        row[0] += 1
        row[1] += last_put[1]
        row[2] += last_put[0] + spent

    def on_light(spent, a, k, result):
        light["glb"] = result[0]
        calls.append({"fn": "light_model", "where": where(), "s": spent, "triangles_in": result[1].get("from_triangles", result[1]["triangles"]),
                      "triangles_out": result[1]["triangles"]})

    def on_outline(spent, a, k, result):
        row = {"fn": "_plan_projection", "where": where(), "s": spent, "triangles": len(a[2].faces)}
        if row["where"].startswith("entities: generated models") and result and light.get("glb"):
            light["probing"] = True
            try:
                row.update(alternative_outline(reconstruction._plan_projection.original, a, light["glb"], result))
            except Exception as error:  # the probe must never stop the import it measures
                row["light_outline_error"] = repr(error)
            light["probing"] = False
        calls.append(row)

    def on_union(spent, a, k, result):
        if result is None and not light.get("probing"):  # union_all never returns None: it raised (the GEOS error _plan_projection retries on a 1e-6 grid)
            row = timers["shapely.union_all raised GEOSException"]
            row[0] += 1
            row[1] += spent

    for name in ("png", "rectify", "fused_part", "atlas", "dense_points", "inferred_floor_points", "build_document"):
        timed(ivs, name)
    timed(ivs, "validate_document", "validate_document (import)")
    timed(postgres, "validate_document", "validate_document (repository)")
    timed(reconstruction, "_plan_projection", after=on_outline)
    timed(attach_entities_to_replay, "light_model", after=on_light)
    timed(shapely, "union_all", "shapely.union_all", on_union)
    for name in ("migrate", "create_project", "create_job", "claim_job", "commit_edits", "create_publication", "_insert_revision"):
        timed(postgres.PostgresRepository, name, f"repository.{name}")
    timed(postgres.PostgresRepository, "register_asset", "repository.register_asset", on_register)
    timed(postgres.PostgresRepository, "finish_job", "repository.finish_job", lambda s, a, k, r: kept.setdefault("document", k.get("document")))
    timed(storage.LocalBlobStore, "put", "blobs.put (write + fsync + verify)", on_put)
    timed(storage.LocalBlobStore, "get", "blobs.get (read + sha256 verify)")

    captured = {}
    namespace = {**vars(ivs), "run": lambda args: captured.setdefault("args", args), "__name__": "__main__"}
    sys.argv = ["import_video_scene.py", *map(str, ARGV)]
    exec(textwrap.dedent(Path(ivs.__file__).read_text().split('if __name__ == "__main__":', 1)[1]), namespace)  # the script's own parser
    t1 = time.perf_counter()
    ivs.run(captured["args"])
    run_s = time.perf_counter() - t1
    sampler.halt.set()
    sampler.join()

    document = kept["document"]
    blob_bytes = sum(p.stat().st_size for p in Path("/tmp/throwaway/blobs").rglob("*") if p.is_file())
    sql = lambda q: subprocess.run(["psql", dsn, "-Atc", q], check=True, capture_output=True, text=True).stdout.strip()
    return {"container": {"cpu": CPU, "memory_mib": MEMORY_MIB, "nproc": os.cpu_count(), "cpu_model": next((l.split(":", 1)[1].strip() for l in
                          open("/proc/cpuinfo") if l.startswith("model name")), None), "postgres": pg_version, "files_copied": copied},
            "seconds": {"copy_volume_to_disk": copy_s, "start_postgres": postgres_s, "python_imports": import_s, "run": run_s,
                        "probe_overhead_in_run": sampler.sections.get("probe overhead: alternative outline (not import work)", 0.),
                        "function_total": time.time() - began},
            "timers": {k: {"calls": v[0], "s": v[1]} for k, v in sorted(timers.items(), key=lambda kv: -kv[1][1])},
            "assets_by_kind": {k: {"count": v[0], "bytes": v[1], "put_plus_register_s": v[2]} for k, v in sorted(kinds.items(), key=lambda kv: -kv[1][2])},
            "sampled": {"samples": sampler.total, "sections_s": dict(sampler.sections.most_common()),
                        "section_leaves_s": [[s, leaf, v] for (s, leaf), v in sampler.leaves.most_common(60)],
                        "repo_functions_s": dict(sampler.repo.most_common(40))},
            "calls": calls,
            "document": {"json_bytes": len(json.dumps(document, separators=(",", ":"))), "assets": len(document["assets"]),
                         "entities": len(document["entities"]), "observations": len(document["observations"]), "cameras": len(document["cameras"])},
            "stores": {"blob_bytes": blob_bytes, "blob_files": sum(1 for p in Path("/tmp/throwaway/blobs").rglob("*") if p.is_file()),
                       "db_bytes": int(sql("SELECT pg_database_size('throwaway')")), "db_assets": int(sql("SELECT count(*) FROM assets")),
                       "db_revisions": int(sql("SELECT count(*) FROM scene_revisions")), "db_publications": int(sql("SELECT count(*) FROM publications"))}}


def next_dir(stem):
    n = 1 + max([int(p.name.rsplit("-", 1)[1]) for p in (ART / "runs").glob(f"{stem}-*") if p.name.rsplit("-", 1)[1].isdigit()] or [0])
    folder = ART / "runs" / f"{stem}-{n}"
    folder.mkdir()
    return folder


def usd(seconds, cpu, memory_mib):
    return seconds * (cpu * CPU_USD + memory_mib / 1024 * GIB_USD)


@app.local_entrypoint()
def profile():
    start = time.time()
    result = profile_import.remote()
    wall = time.time() - start
    result["seconds"]["local_wall_incl_cold_start"] = wall
    result["usd_list_price_upper_bound"] = usd(wall, CPU, MEMORY_MIB)
    folder = next_dir("m3-exp-e8-import-profile")
    (folder / "profile.json").write_text(json.dumps(result, indent=1))
    print(folder)
    print(json.dumps({k: result[k] for k in ("seconds", "document", "stores", "usd_list_price_upper_bound")}, indent=1))


# ---------- (b) cloud write path: content-addressed blobs + one small patch per layer, on a Volume; an HTTP reader serves them

def room_layer():
    """The room surface as the report stores it (panoptes-mesh-v1 of the fused mesh) and the walk's camera path."""
    import numpy as np
    import trimesh
    t = time.perf_counter()
    shell = trimesh.load(f"/data/{RUNS['depth']}/predicted-scene.glb", force="mesh", process=False)
    prediction = np.load(f"/data/{RUNS['droid']}/prediction.npz")
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    rows = np.hstack([shell.vertices, shell.vertex_normals, np.asarray(shell.visual.vertex_colors)[:, :3] / 255.]).astype("<f4")
    mesh = rows.tobytes() + np.asarray(shell.faces, "<u4").tobytes()
    walk = [(i, c) for i, c in enumerate(prediction["poses_c2w"]) if not 14 <= i < 226]  # every walk frame, as the import places its cameras
    path = json.dumps({"coordinateFrameId": "droid_final_native_world", "cameras": [{"sourceFrame": i, "cameraToWorld": np.round(c, 6).tolist()}
                                                                                    for i, c in walk]}, separators=(",", ":")).encode()
    return [("room-surface", "application/octet-stream", mesh, {"format": "panoptes-mesh-v1", "vertexCount": len(rows)}),
            ("camera-path", "application/json", path, {"cameras": len(walk)})], load_s, time.perf_counter() - t


def file_layer(role, relative, media):
    t = time.perf_counter()
    data = Path(f"/data/{relative}").read_bytes()
    return [(role, media, data, {})], time.perf_counter() - t, 0.


def put_blob(data):
    """Content addressed and immutable: the key is the bytes' sha256; an existing key is never rewritten."""
    sha = hashlib.sha256(data).hexdigest()
    path = Path(f"/layers/blobs/sha256/{sha}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        (tmp := path.with_suffix(".part")).write_bytes(data)
        tmp.rename(path)
    return sha


@app.function(image=image, cpu=2.0, memory=8192, volumes={"/data": DATA, "/layers": LAYERS}, timeout=600, retries=0, max_containers=1)
def write_layers(report):
    layers = [("room", room_layer), ("room-photo", lambda: file_layer("room-photo-textured", "runs/me340-filled-225/textured-scene.glb", "model/gltf-binary")),
              ("splats", lambda: file_layer("gaussian-splats", "runs/me340-splat-232/splats.splat", "application/octet-stream"))]
    out = []
    for seq, (name, make) in enumerate(layers):
        blobs, load_s, encode_s = make()  # the layer as it would sit in memory after its stage; load/encode are the stage's, not the write's
        t = time.perf_counter()
        assets = [{"role": role, "sha256": put_blob(data), "sizeBytes": len(data), "mediaType": media, **extra} for role, media, data, extra in blobs]
        blobs_s = time.perf_counter() - t
        patch = json.dumps({"schema": "panoptes-layer-patch-v0", "report": report, "seq": seq, "layer": name, "assets": assets}, indent=None).encode()
        folder = Path(f"/layers/reports/{report}/patches")
        folder.mkdir(parents=True, exist_ok=True)
        (tmp := folder / f".{seq:04d}-{name}.json.part").write_bytes(patch)
        tmp.rename(folder / f"{seq:04d}-{name}.json")
        t2 = time.perf_counter()
        LAYERS.commit()
        commit_s = time.perf_counter() - t2
        out.append({"layer": name, "load_s": load_s, "encode_s": encode_s, "blobs_write_s": blobs_s, "commit_s": commit_s,
                    "write_total_s": time.perf_counter() - t, "blob_bytes": sum(a["sizeBytes"] for a in assets), "patch_bytes": len(patch),
                    "committed_at": time.time()})
    return out


@app.function(image=image, cpu=1.0, memory=2048, volumes={"/layers": LAYERS}, timeout=600, retries=0, max_containers=1)
def serve_layers(report, expected):
    """A reader that is already up: sees each committed patch on the Volume and serves the patch and its blobs over HTTP on loopback
    (no public URL); a client in this container fetches them the way a viewer would and checks every sha256."""
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import urllib.request

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory="/layers", **k)

        def do_GET(self):
            if self.path.endswith("/patches"):  # the patch log: what a viewer polls (or is pushed) to learn of a new layer
                body = json.dumps(sorted(p.name for p in Path("/layers" + self.path).glob("*.json"))).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                super().do_GET()

        def end_headers(self):
            if "/blobs/sha256/" in self.path:
                self.send_header("Cache-Control", "public, max-age=31536000, immutable")
            super().end_headers()

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 8799), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    get = lambda path: urllib.request.urlopen(f"http://127.0.0.1:8799{path}").read()
    seen, out, deadline = set(), [], time.time() + 480
    while len(seen) < expected and time.time() < deadline:
        LAYERS.reload()
        for name in json.loads(get(f"/reports/{report}/patches")):
            if name in seen:
                continue
            visible_at = time.time()
            t = time.perf_counter()
            patch = json.loads(get(f"/reports/{report}/patches/{name}"))
            fetched = 0
            for asset in patch["assets"]:
                data = get(f"/blobs/sha256/{asset['sha256']}")
                assert hashlib.sha256(data).hexdigest() == asset["sha256"] and len(data) == asset["sizeBytes"], name
                fetched += len(data)
            out.append({"layer": patch["layer"], "visible_at": visible_at, "http_fetch_s": time.perf_counter() - t, "http_bytes": fetched})
            seen.add(name)
        time.sleep(.2)
    server.shutdown()
    return out


@app.local_entrypoint()
def layer():
    report = f"e8-{int(time.time())}"
    start = time.time()
    reader = serve_layers.spawn(report, 3)
    time.sleep(20)  # let the reader come up first: it stands for a server that is already running
    t = time.time()
    written = write_layers.remote(report)
    writer_wall = time.time() - t
    served = {s["layer"]: s for s in reader.get()}
    wall = time.time() - start
    rows = [{**w, **served.get(w["layer"], {}), "visible_after_commit_s": served[w["layer"]]["visible_at"] - w["committed_at"]
             if w["layer"] in served else None} for w in written]
    for r in rows:
        r["write_to_served_s"] = r["write_total_s"] + (r["visible_after_commit_s"] or 0) + r.get("http_fetch_s", 0)
    result = {"report": report, "layers": rows, "writer_wall_incl_cold_start_s": writer_wall, "local_wall_s": wall,
              "usd_list_price_upper_bound": usd(writer_wall, 2.0, 8192) + usd(wall, 1.0, 2048)}
    folder = next_dir("m3-exp-e8-import-layer")
    (folder / "layer.json").write_text(json.dumps(result, indent=1))
    print(folder)
    print(json.dumps(result, indent=1))
