"""One-time adopt of the three delivered reports into the runner's store (STREAMING-PLAN M2; runner design §7, §8).

  cd scripts && python -m report_runner.adopt ../tests/fixtures/delivered-303/walmart.json --video MP4 [--replay]
  python -m report_runner.adopt --fingerprint PUBLICATION_ID     # read-only SELECT, prints fingerprint + counts
  python -m report_runner.adopt --self-check

Per site: hash the named --video (the file only, never its folder) against clip.json; append the published report's
imports.jsonl row (read-only SELECT: fingerprint, title, record path) so the import keeps its title and republishes its
record; resolve the delivered graph (profile delivered, decisions served from the fixture as the graph asks). Every graph
stage names its delivered run in the fixture (node.graph: outputs by role, the recorded commands); the runner's resolved
argv is compared with the recorded one flag by flag (paths as real paths, defaults filled, key-free flags and the flags a
mode never reads dropped; D/I flags must also pass their evidence) and its staged inputs with the run's folder. No
difference: the delivered key is recorded. Differences at a node the register (section 9) lists: recorded too, delivered
scope only, with the differences in its lock and in the report. Any other difference: refused, reported exactly, not
recorded. Then the decision rules run on the adopted inputs and are diffed against the fixture, and research keys are
recorded where the M2 argv is the delivered one and the code is verified (recorded-rev, content, or --replay).

Never opens .platform/imports/* (the records hold project capabilities): a record is found by a stat of its name only.
"""
import argparse
import dataclasses
from dataclasses import asdict, dataclass, field
import inspect
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ART = Path(os.environ.get("PANOPTES_ART", "/Users/adam/Desktop/panoptes-public/research-notes/phase2"))  # = droid_room.ART
DATABASE_URL = "postgresql://panoptes@127.0.0.1:54329/panoptes_video"
SITES = ("me340", "samsclub-a2", "walmart")
REDACT = re.compile(r"capabilit|token|secret", re.I)
FIXTURE_KEYS = {"site", "video_sha256", "start", "end", "publication", "nodes", "decisions", "deviations"}
NODE_KEYS = {"id", "stage", "dirs", "outputs", "commands", "basis", "evidence", "patch", "staging"}
DEVIATION_KEYS = {"id", "class", "nodes", "delivered", "m2", "reason"}
REGISTER = {"X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "D1", "D2", "D3", "D4", "D5", "D6", "O1",
            "X14", "X15"}  # X14, X15: found when the runner resolved the delivered graph (M2 integration); not in design section 9
U_ALLOWED = {("R37", "--title"), ("R37", "--republish"), ("R37", "--request-suffix"), ("R24", "--marks")}  # R24 only on me340


def say(*parts):
    """Print, minus any line that names a capability, token or secret."""
    for line in " ".join(map(str, parts)).splitlines():
        if not REDACT.search(line):
            print(line)


def forbidden(path):
    return "/.platform/imports/" in str(Path(path)).replace(os.sep, "/") + "/"


def read_json(path):
    assert not forbidden(path), f"refusing to open an import record: {path}"
    return json.loads(Path(path).read_text())


def sha256_file(path):
    assert not forbidden(path), f"refusing to open an import record: {path}"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_path(path):
    """(size, mtime_ns, sha256) of a file, or of a folder as sha256 over sorted (relpath, file sha256), links followed."""
    path = Path(path)
    if path.is_file():
        st = path.stat()
        return st.st_size, st.st_mtime_ns, sha256_file(path)
    assert path.is_dir(), f"missing output: {path}"
    rows, size, mtime = [], 0, 0
    for root, dirs, files in os.walk(path, followlinks=True):
        dirs.sort()
        for name in sorted(files):
            p = Path(root) / name
            st = p.stat()
            size, mtime = size + st.st_size, max(mtime, st.st_mtime_ns)
            rows.append([str(p.relative_to(path)), sha256_file(p)])
    return size, mtime, hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# ------------------------------------------------------------------------------------------------ fingerprint (§8 P5)

def fingerprint_body(document, title=None):
    """What must survive an offline rebuild of the import: asset bytes, entity identity/labels/state/models, camera and
    observation counts, annotation kinds, leftOutOfReport, the scale record, the title. Not `limitations` (X13).
    Asset ids are random per import, so every asset reference is replaced by the asset's sha256; model sha is taken
    from the generated_mesh representations, so a schema-1 document (build_document) and its schema-2 migration agree."""
    sha = {a["id"]: a["sha256"] for a in document["assets"]}
    frame = (document.get("coordinateFrames") or [{}])[0]
    scale = json.loads(json.dumps(frame.get("scale")))
    for ref in (scale or {}).get("sourceRefs") or []:
        if "assetId" in ref:
            ref["assetId"] = sha.get(ref["assetId"], ref["assetId"])
    provenance = [a for a in document["annotations"] if a["kind"] == "import_provenance"]
    return {"assets": sorted(sha.values()),
            "entities": sorted([e["id"], e.get("label"), e["associationState"],
                                sorted(sha[r["assetId"]] for r in e.get("representations", []) if r["kind"] == "generated_mesh")]
                               for e in document["entities"]),
            "cameras": len(document["cameras"]), "observations": len(document["observations"]),
            "annotations": sorted(a["kind"] for a in document["annotations"]),
            "leftOutOfReport": provenance[0]["leftOutOfReport"] if provenance else None,
            "scale": scale, "title": title}


def fingerprint(document: dict, title=None) -> str:
    return hashlib.sha256(canonical(fingerprint_body(document, title)).encode()).hexdigest()


def counts(document):
    """The numbers the plan quotes: entities / confirmed / learned-or-box models (the inferred floor plane is not one)."""
    assets = {a["id"]: a for a in document["assets"]}
    models = 0
    for e in document["entities"]:
        for r in e.get("representations", []):
            if r["kind"] == "generated_mesh" and not str(assets[r["assetId"]].get("metadata", {}).get("generator", "")).startswith("parametric plane"):
                models += 1
    return {"entities": len(document["entities"]), "confirmed": sum(e["associationState"] == "confirmed" for e in document["entities"]),
            "models": models, "cameras": len(document["cameras"]), "observations": len(document["observations"]), "assets": len(document["assets"])}


# --------------------------------------------------------------------------------------------- database (read-only)

class Database:
    """SELECT-only access to the platform database; every transaction is read-only on the server side too."""

    def __init__(self, url=None):
        self.url = url or os.environ.get("PANOPTES_DATABASE_URL") or DATABASE_URL

    def rows(self, sql, params=()):
        assert re.match(r"\s*select\b", sql, re.I) and ";" not in sql.strip().rstrip(";"), "SELECT only"
        import psycopg
        with psycopg.connect(self.url, options="-c default_transaction_read_only=on") as conn:
            conn.read_only = True
            with conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall()

    def publication(self, publication_id):
        (row,) = self.rows("SELECT p.id::text, p.project_id::text, p.title, r.document, p.created_at FROM publications p "
                           "JOIN scene_revisions r ON r.id = p.scene_revision_id AND r.project_id = p.project_id WHERE p.id::text = %s", (publication_id,))
        return {"publicationId": row[0], "projectId": row[1], "title": row[2], "document": row[3], "createdAt": row[4].isoformat()}

    def project_publications(self, project_id):
        return [r[0] for r in self.rows("SELECT id::text FROM publications WHERE project_id::text = %s ORDER BY created_at DESC", (project_id,))]


def import_record(art, publication_id):
    """The import record's path when it exists: a stat of the name, never an open (the record holds a capability)."""
    path = Path(art) / ".platform/imports" / f"video-import-{publication_id}.json"
    return str(path) if path.exists() else None


def import_row(site, key, publication_id, art, db):
    published = db.publication(publication_id)
    later = [p for p in db.project_publications(published["projectId"]) if import_record(art, p)]
    return {"site": site, "key": key, "publicationId": publication_id, "projectId": published["projectId"],
            "importRecordPath": import_record(art, publication_id), "republishNext": import_record(art, later[0]) if later else None,
            "title": published["title"], "fingerprint": fingerprint(published["document"], published["title"]),
            "counts": counts(published["document"]), "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


# ------------------------------------------------------------------------------------------------ fixtures (§7.1)

def load_fixture(path):
    fx = json.loads(Path(path).read_text())
    problems = validate_fixture(fx)
    assert not problems, f"{path}: " + "; ".join(problems)
    return fx


def validate_fixture(fx):
    """Schema and graph checks: keys, unique ids, inputs point backwards (topological order), patches carry a register
    id, every deviation's nodes exist, the basis only uses M/D/I/U and names flags that appear in the commands."""
    p = [f"missing {k}" for k in FIXTURE_KEYS - set(fx)]
    if p:
        return p
    p += [] if fx["site"] in SITES else [f"unknown site {fx['site']}"]
    p += [] if re.fullmatch(r"[0-9a-f]{64}", fx["video_sha256"]) else ["video_sha256 is not a sha256"]
    seen, deviation_ids = set(), {d.get("id") for d in fx["deviations"]}
    for n in fx["nodes"]:
        p += [f"{n.get('id')}: missing {k}" for k in NODE_KEYS - set(n)]
        if NODE_KEYS - set(n):
            continue
        p += [f"duplicate node {n['id']}"] if n["id"] in seen else []
        p += [f"{n['id']}: input {i} is not an earlier node" for i in n.get("inputs", []) if i not in seen]
        seen.add(n["id"])
        p += [f"{n['id']}: patch {n['patch']} not in the register" for _ in [0] if n["patch"] is not None and (n["patch"] not in REGISTER or n["patch"] not in deviation_ids)]
        p += [f"{n['id']}: basis {v} for {k}" for k, v in n["basis"].items() if v not in "MDIU" or len(v) != 1]
        p += [f"{n['id']}: D/I flag {k} without evidence" for k, v in n["basis"].items() if v in "DI" and not n["patch"] and not n["evidence"]]
        p += [f"{n['id']}: U flag {k} not allowed" for k, v in n["basis"].items() if v == "U" and not u_allowed(fx["site"], n["stage"], k)]
        p += [f"{n['id']}: outputs must be ART-relative" for rel in n["outputs"].values() if rel.startswith("/") or rel.startswith("$")]
    ids, occupied = {n.get("id"): n for n in fx["nodes"]}, {}
    for n in fx["nodes"]:  # node.graph: the graph stages this delivered run stands for, their outputs and recorded commands
        for stage, entry in n.get("graph", {}).items():
            p += [f"{n['id']}: stage {stage} is also {occupied[stage]}'s"] if stage in occupied else []
            occupied[stage] = n["id"]
            p += [f"{n['id']}: {stage} outputs must be ART-relative" for rel in entry.get("outputs", {}).values() if rel is not None and rel.startswith(("/", "$"))]
            p += [f"{n['id']}: {stage} names command {c}, which is not recorded" for c in entry.get("commands", [])
                  if not (c[0] in ids and 0 <= c[1] < len(ids[c[0]].get("commands", [])))]
    for d in fx["deviations"]:
        p += [f"deviation {d.get('id')}: missing {k}" for k in DEVIATION_KEYS - set(d)]
        p += [f"deviation {d.get('id')}: not in the register" for _ in [0] if d.get("id") not in REGISTER]
        p += [f"deviation {d.get('id')}: node {i} not in this fixture" for i in d.get("nodes", []) if i not in seen]
        p += [f"deviation {d.get('id')}: class {d.get('class')}" for _ in [0] if d.get("class") not in ("X", "D", "O")]
    return p


def u_allowed(site, stage, flag):
    return (stage, flag) in U_ALLOWED and (stage != "R24" or site == "me340")


def expand(word, art, repo=REPO, video=None):
    return str(word).replace("$ART", str(art)).replace("$REPO", str(repo)).replace("$VIDEO", str(video or "$VIDEO"))


# ---------------------------------------------------------------------------------------------- evidence checks

def _get(data, dotted):
    for part in dotted.split(".") if dotted else []:
        data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def _close(a, b, tol):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(a - b) <= tol
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def _views(run):
    return sorted(int(p.stem) for p in (Path(run) / "mono").glob("*.npz"))


def _outside(spans):
    return lambda f: not any(a <= f < b for a, b in spans)


def check_view_set(art, c):
    """mono/*.npz == (DROID keyframes ∪ range(0, frames, every)) minus the excluded spans (mono_room.infer)."""
    import numpy as np
    prediction = np.load(Path(art) / c["droid"] / "prediction.npz")
    keep = _outside(c.get("exclude", []))
    expected = sorted({int(i) for i in prediction["keyframe_source_indices"]} | set(range(0, len(prediction["poses_c2w"]), c["every"])))
    expected = [i for i in expected if keep(i)]
    got = _views(Path(art) / c["run"])
    return got == expected, f"{len(got)} views on disk, {len(expected)} by the rule"


def check_view_subset(art, c):
    keep = _outside(c["exclude"])
    got, of = _views(Path(art) / c["run"]), _views(Path(art) / c["of"])
    return got == [i for i in of if keep(i)], f"{len(got)} views = {c['of']} outside {c['exclude']}"


FUSE_FIELDS = {"--voxel-length-native": ("voxel_native", float), "--support-relative": ("fusion_support_relative_tolerance", float),
               "--edge-jump": ("depth_edge_relative_jump_removed", float)}


def check_fuse_metrics(art, c):
    """fuse-metrics.json fields map 1:1 to the fuse flags (mono_room.fuse)."""
    m = read_json(Path(art) / c["run"] / "fuse-metrics.json")
    bad = [f for f, (k, t) in FUSE_FIELDS.items() if f in c["flags"] and not _close(m[k], t(c["flags"][f]), 1e-12)]
    if "--support-all-views" in c["flags"]:
        bad += [] if m["fusion_support_views"].startswith("every keyframe") else ["--support-all-views"]
    if "--carve" in c["flags"]:
        bad += [] if m.get("free_space_carving") else ["--carve"]
    if "--floor-plane" in c["flags"]:
        bad += [] if m.get("floor_plane_support") else ["--floor-plane"]
    if "--dynamic-masks" in c["flags"]:
        bad += [] if m.get("dynamic_masks") == expand(c["flags"]["--dynamic-masks"], art) else ["--dynamic-masks"]
    if "views" in c:
        bad += [] if m["views"] == c["views"] else ["views"]
    return not bad, f"fuse-metrics disagrees on {bad}" if bad else "fuse-metrics fields match the flags"


def check_json_equal(art, c):
    a, b = (_get(read_json(Path(art) / f), k) for f, k in (c["a"], c["b"]))
    return _close(a, b, c.get("tol", 1e-9)), f"{c['a'][1]}={a} vs {c['b'][1]}={b}"


def check_json_value(art, c):
    got = _get(read_json(Path(art) / c["file"]), c["path"])
    return _close(got, c["value"], c.get("tol", 1e-9)), f"{c['file']}:{c['path']}={got}, expected {c['value']}"


def check_carry_height(art, c):
    """Median camera height above the metric floor plane over the kept cameras (mono_room.metric)."""
    import numpy as np
    scale = read_json(Path(art) / c["scale"])
    cameras = np.load(Path(art) / c["droid"] / "prediction.npz")["poses_c2w"][:, :3, 3].astype(np.float64)
    keep = _outside(c.get("exclude", []))
    heights = (cameras[[i for i in range(len(cameras)) if keep(i)]] - scale["plane_point_native"]) @ np.asarray(scale["up_native"])
    got = float(np.median(heights))
    return abs(got - c["value"]) <= c.get("tol", 5e-6) and abs(got - scale["camera_height_native_median"]) <= 1e-9, f"carry-height median {got:.6f} (recorded {scale['camera_height_native_median']:.6f})"


def check_same_bytes(art, c):
    shas = {f: sha256_file(Path(art) / f) for f in c["files"]}
    return len(set(shas.values())) == 1, f"{len(set(shas.values()))} distinct contents among {len(shas)} files"


def check_frame_list(art, c):
    """SAM 2.1 frames == census keyframes ∪ range(0, frames, every) (rule D12)."""
    import numpy as np
    prediction = np.load(Path(art) / c["census"] / "prediction.npz")
    expected = sorted({int(i) for i in prediction["keyframe_source_indices"]} | set(range(0, c["frames"], c["every"])))
    got = read_json(Path(art) / c["segment"])["frames"]
    return got == expected, f"{len(got)} frames recorded, {len(expected)} by the rule"


def check_published_assets(art, c, db=None):
    """Every named file's bytes are in the published document (read-only SELECT) as an asset or as the validated mesh
    an asset was made from (models are published as decimated display copies naming their source's sha256)."""
    def shas(value, key=""):
        if isinstance(value, dict):
            return {s for k, v in value.items() for s in shas(v, k)}
        return {value} if isinstance(value, str) and key.lower().endswith("sha256") else set()
    published = set().union(*(shas(a) for a in (db or Database()).publication(c["publication"])["document"]["assets"]))
    files = [Path(art) / f for f in c.get("files", [])] + [p for g in c.get("globs", []) for p in sorted(Path(art).glob(g))]
    missing = [str(f.relative_to(art)) for f in files if sha256_file(f) not in published]
    return bool(files) and not missing, f"{len(files) - len(missing)}/{len(files)} files are published assets" + (f"; not: {missing[:3]}" if missing else "")


def check_absent(art, c):
    text = (Path(art) / c["file"]).read_text()
    return c["text"] not in text, f"{c['text']!r} {'absent from' if c['text'] not in text else 'found in'} {c['file']}"


def check_replay(art, c):
    ok, detail, _ = run_replay(c["replay"], art)
    return ok, detail


CHECKS = {"view_set": check_view_set, "view_subset": check_view_subset, "fuse_metrics": check_fuse_metrics, "json_equal": check_json_equal,
          "json_value": check_json_value, "carry_height": check_carry_height, "same_bytes": check_same_bytes, "frame_list": check_frame_list,
          "published_assets": check_published_assets, "absent": check_absent, "replay": check_replay}


def run_check(art, check):
    try:
        return CHECKS[check["kind"]](art, check)
    except (OSError, KeyError, ValueError, AssertionError) as error:
        return False, f"{check['kind']}: {type(error).__name__}: {error}"


# ----------------------------------------------------------------------------------------------------- replays (P4)

def run_replay(spec, art, python=sys.executable, out=None, root=None):
    """A CPU re-run at HEAD into runs/report-runner/replays/<name>-<time>/ ($OUT), compared with the delivered outputs.
    spec: {name, argv:[script, ...] with $ART/$REPO/$OUT, seed:{"out-rel": "ART-rel"} copied in first (journals),
           compare:{"exit0": bool, "sha": {"out-rel": "ART-rel"}, "json": {"out-rel": ["ART-rel", [dotted fields], tol]}}}"""
    import shutil
    folder = Path(out or Path(root or Path(art) / "runs/report-runner") / "replays" / f"{spec['name']}-{time.strftime('%Y%m%d-%H%M%S')}")
    folder.mkdir(parents=True, exist_ok=False)
    out = folder / "out"  # $OUT: most tools refuse an output folder that exists, so it is made here only when the replay needs it
    if spec.get("seed") or any(str(w).startswith("$OUT/") for w in spec["argv"]):
        out.mkdir()
    for rel, src in spec.get("seed", {}).items():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(art) / src, out / rel)
    argv = [python] + [expand(w, art).replace("$OUT", str(out)) for w in spec["argv"]]
    argv[1] = str(REPO / argv[1]) if not Path(argv[1]).is_absolute() else argv[1]
    done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True)
    (folder / "replay.log").write_text("\n".join(l for l in (done.stdout + done.stderr).splitlines() if not REDACT.search(l)))
    compare, bad = spec.get("compare", {}), []
    if compare.get("exit0", True) and done.returncode:
        bad.append(f"exit {done.returncode}")
    for rel, delivered in compare.get("sha", {}).items():
        if not (out / rel).exists() or sha256_file(out / rel) != sha256_file(Path(art) / delivered):
            bad.append(f"{rel} differs from {delivered}")
    for rel, (delivered, fields, tol) in compare.get("json", {}).items():
        if not (out / rel).exists():
            bad.append(f"{rel} missing")
            continue
        mine, theirs = read_json(out / rel), read_json(Path(art) / delivered)
        bad += [f"{rel}:{f}" for f in fields if not _close(_get(mine, f), _get(theirs, f), tol)]
    return not bad, "; ".join(bad) or f"{spec['name']}: equal to the delivered outputs", out


# ------------------------------------------------------------------------------------------- argv and staging checks

TOKEN = re.compile(r"(?<![\w.])@([A-Za-z][\w-]*)(?::([\w.-]+))?")  # spec.py's token grammar


def flag_map(norm):
    """B's normalised argv as {flag: value}; accepts a dict, a list of (flag, value) pairs or a flat word list."""
    if isinstance(norm, dict):
        return dict(norm.get("flags", norm))
    if norm and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in norm):
        return {k: v for k, v in norm}
    out, key = {"": []}, ""
    for word in norm or []:
        if isinstance(word, str) and word.startswith("--"):
            key = word
            out.setdefault(key, [])
        else:
            out.setdefault(key, []).append(word)
    return out


def _shown(value, limit=400):
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + f"... ({len(text)} chars)"


def compare_argv(site, node, mine, theirs, art):
    """[(flag, basis, status)] where status is 'equal', 'unrecorded', 'evidence-failed: …' or 'different: runner … vs
    delivered …'. mine/theirs normalised (paths as real paths)."""
    a, b = flag_map(mine), flag_map(theirs)
    if isinstance(mine, dict) and isinstance(theirs, dict):
        a[""], b[""] = [mine.get("script"), *mine.get("args", [])], [theirs.get("script"), *theirs.get("args", [])]
    out = []
    for flag in sorted(set(a) | set(b)):
        basis = node["basis"].get(flag, "M")
        if basis == "U" and u_allowed(site, node["stage"], flag):
            out.append((flag, "U", "unrecorded"))
        elif a.get(flag) != b.get(flag):
            out.append((flag or "<script and positionals>", basis, f"different: runner {_shown(a.get(flag))} vs delivered {_shown(b.get(flag))}"))
        elif basis in "DI":
            failed = [d for ok, d in (run_check(art, c) for c in node["evidence"]) if not ok]
            out.append((flag, basis, "evidence-failed: " + "; ".join(failed) if failed else "equal"))
    return out


def check_staging(node, art):
    """Links resolve to the owner's files and copies are byte-equal (hard links count as copies)."""
    art, bad = Path(art), []
    for link, target in node.get("links", {}).items():
        if not (art / link).exists() or os.path.realpath(art / link) != os.path.realpath(art / target):
            bad.append(f"link {link} does not resolve to {target}")
    for copy, source in node.get("copies", {}).items():
        a, b = art / copy, art / source
        if a.is_dir():
            if digest_path(a)[2] != digest_path(b)[2]:
                bad.append(f"{copy} differs from {source}")
        elif not a.exists() or sha256_file(a) != sha256_file(b):
            bad.append(f"copy {copy} differs from {source}")
    return bad


def verify_code(node, art, repo=REPO):
    """How the node's code is known current, or None. recorded-rev: the recorded script sha256 equals the file at HEAD,
    or the recorded revision is the pin written in that script; content: the named check passes. Replays are run
    separately (--replay), since they take minutes."""
    v = node.get("verify")
    if not v:
        return None
    if v["kind"] == "recorded-rev":
        try:
            recorded = _get(read_json(Path(art) / v["file"]), v["field"])
        except (OSError, KeyError, ValueError):
            return None
        script = Path(repo) / v["script"]
        if v["field"].endswith("sha256"):
            return "recorded-rev" if script.exists() and sha256_file(script) == recorded else None
        return "recorded-rev" if script.exists() and str(recorded) in script.read_text() else None
    if v["kind"] == "content":
        return "content" if all(ok for ok, _ in (run_check(art, c) for c in v.get("checks", []))) else None
    return None


# -------------------------------------------------------------------------------------------------------- adopt

class MissingDecision(LookupError):
    """The graph asks for a decision the fixture has no delivered value for."""


@dataclass
class AdoptReport:
    site: str
    delivered_keys: dict = field(default_factory=dict)   # graph stage (or side node id) -> key
    research_keys: dict = field(default_factory=dict)    # graph stage -> [key, verification]
    deviations_seen: list = field(default_factory=list)  # register ids whose difference showed up
    unlisted_diffs: list = field(default_factory=list)   # anything else: the proof fails
    import_row: dict = None
    counts: dict = field(default_factory=dict)           # template / patch / staging / side / U
    decisions: dict = field(default_factory=dict)        # decision -> "equal" | "listed X.." | "differs" | "not run: ..."
    listed: dict = field(default_factory=dict)           # graph stage -> [node, register ids, [differences]]: adopted, delivered-only
    refused: dict = field(default_factory=dict)          # graph stage -> [node, [differences]]: not adopted


def occupants(fx):
    """{graph stage: node}: the delivered run standing for each stage of the delivered graph (fixture node.graph)."""
    out = {}
    for node in fx["nodes"]:
        for stage in node.get("graph", {}):
            out[stage] = node
    return out


def stage_outputs(node, stage, art):
    """{role: absolute path | None} of the delivered run for this graph stage (None: recorded as absent)."""
    return {role: None if rel is None else Path(art) / rel for role, rel in node["graph"][stage]["outputs"].items()}


def recorded_commands(fx, node, stage):
    """[(node that ran it, recorded argv)] of this stage: its basis and evidence are that node's."""
    by_id = {n["id"]: n for n in fx["nodes"]}
    return [(by_id[i], by_id[i]["commands"][k]) for i, k in node["graph"][stage]["commands"]]


def unassigned_commands(fx, node):
    """Recorded commands of this node that no graph stage stands for and no note explains: the runner would not run them."""
    used = {(i, k) for n in fx["nodes"] for entry in n.get("graph", {}).values() for i, k in entry["commands"]}
    ignored = {int(k) for entry in node.get("graph", {}).values() for k in entry.get("ignore", {})}
    return [c for k, c in enumerate(node["commands"]) if (node["id"], k) not in used and k not in ignored]


def split_command(cmd):
    """(script, argv) of a runner command ('python SCRIPT ...' or 'python -m MODULE ...') or of a recorded one ('SCRIPT ...')."""
    words = list(cmd)
    if Path(str(words[0])).name.startswith("python"):
        words = words[1:]
    if words and words[0] == "-m":
        return words[1], words[2:]
    return str(words[0]).replace("$REPO/", "").replace(str(REPO) + "/", ""), words[1:]


def substitute(word, own_dir, own_out, adopted, art, video=None):
    """A runner word with its tokens replaced by the adopted runs' paths (@new/out and @clip: the stage's own 'out')."""
    def one(m):
        name, role = m.groups()
        if not role and name in ("new", "clip", "key"):
            return {"new": str(own_dir), "clip": str(own_out), "key": "KEY"}[name]
        found = adopted[name]
        return str(found["outputs"][role] if role else found["dir"])
    return TOKEN.sub(one, expand(str(word).replace("@new/out", str(own_out)), art, video=video))


def argv_differences(fx, node, stage, spec, adopted, normalize, art, video):
    """The resolved runner argv of `stage` against the delivered run's recorded argv, flag by flag (paths compared as real
    paths, defaults filled, key-free flags and note text dropped): [(flag, basis, status)] that are not equal."""
    own = stage_outputs(node, stage, art)
    own_dir = Path(art) / node["dirs"][0] if node["dirs"] else Path(art)  # the import has no run folder of its own
    own_out = own.get("out") or own_dir
    real = lambda path: ("path", os.path.realpath(str(path)))
    mine = [c for c in spec.commands if c and c[0] != "ln"]  # 'ln -s @clip @new/out' places the clip; it runs no tool
    theirs = recorded_commands(fx, node, stage)
    rows = [("<commands>", "M", f"different: runner runs {len(mine)} tool commands, delivered {len(theirs)}")] if len(mine) != len(theirs) else []
    for m, (owner, t) in zip(mine, theirs):
        try:
            script, argv = split_command([substitute(w, own_dir, own_out, adopted, art, video) for w in m])
            ours = normalize(script, argv, real)
        except KeyError as error:
            rows.append(("<inputs>", "M", f"different: a producer is not adopted ({error})"))
            continue
        tscript, targv = split_command([expand(w, art, video=video) for w in t])
        try:
            recorded = normalize(tscript, targv, real)
        except (KeyError, ValueError) as error:  # a hand-written script, or a flag the runner does not know
            rows.append(("<script and positionals>", "M", f"different: runner {script} vs delivered {tscript} ({type(error).__name__}: {error})"))
            continue
        rows += [r for r in compare_argv(fx["site"], owner, ours, recorded, art) if r[2] != "equal"]
    return rows


def staging_differences(spec, node, stage, adopted, art):
    """The graph stages a stage's inputs (stage_from): each link must resolve to, and each copy hold, what the delivered run's
    folder has at that place (same real path, or the same bytes). A copy the tool rewrites (one of its outputs) is not compared."""
    own = stage_outputs(node, stage, art)
    own_dir = Path(art) / node["dirs"][0]
    rows, produced = [], set(spec.outputs.values())
    for kind, refs in spec.stage_from.items():
        for rel, ref in sorted(refs.items()):
            if kind == "copy" and rel in produced:
                continue
            place = (own.get("out") or own_dir) / rel[4:] if rel.startswith("out/") else own_dir / rel
            try:
                source = Path(substitute(ref, own_dir, own.get("out") or own_dir, adopted, art))
            except KeyError as error:
                rows.append((f"<staged {rel}>", "M", f"different: its source {ref} is not adopted ({error})"))
                continue
            if not place.exists():
                rows.append((f"<staged {rel}>", "M", f"different: the delivered run has no {rel} ({kind} of {ref})"))
            elif os.path.realpath(place) != os.path.realpath(source) and digest_path(place)[2] != digest_path(source)[2]:
                rows.append((f"<staged {rel}>", "M", f"different: runner {kind}s {ref} = {os.path.relpath(source, art)}, delivered "
                                                      f"{os.path.relpath(os.path.realpath(place), art)} (other bytes)"))
    return rows


def side_spec(site, node, art):
    """A delivered-only StageSpec for a run no M2 stage stands for (an intermediate merge, the old-K camera...): its
    argv is the recorded one, its name unique, nothing consumes it. Kept for provenance, never looked up."""
    from report_runner.spec import StageSpec
    outputs = {role: os.path.relpath(Path(art) / rel, Path(art) / node["dirs"][0]) for role, rel in node["outputs"].items()}
    return StageSpec(name=f"{node['stage']}~{node['id']}", site=site, version=0, deps=(), commands=tuple(tuple(expand(w, art) for w in c) for c in node["commands"]),
                     inputs={}, consumes={}, leaves={}, outputs=outputs, stage_from={}, compute="adopted", gpu=None, timeout_s=0,
                     worst_usd=0.0, est_usd=0.0, est_s=0.0, budget_flags={}, models=tuple(tuple(m) for m in node.get("models", [])), rules={}, env={})


def resolve(graph, ctx, on_pending):
    """graph(ctx) until it stops raising Pending; on_pending(p) must fill ctx.decisions[p.decision] or raise."""
    from report_runner.spec import Pending
    while True:
        try:
            return graph(ctx)
        except Pending as p:
            if p.decision in ctx.decisions:
                raise RuntimeError(f"graph asked again for decision {p.decision!r}") from p
            on_pending(p)


def compare_decision(got, want, how):
    """Rule value against the delivered value: whole value (floats within 1e-6), named fields, dict keys, or a band inside."""
    if not how:
        return _close(got, want, 1e-6)
    if "fields" in how:
        return all(_close((got or {}).get(f), (want or {}).get(f), 1e-6) for f in how["fields"])
    if "keys" in how:
        return sorted((got or {}).get(how["keys"], {})) == sorted((want or {}).get(how["keys"], {}))
    if "band_within" in how:
        inner, outer = how["band_within"]["inner"], how["band_within"]["outer"]
        bands = (got or {}).get("bands") or []
        return len(bands) == 1 and outer[0] <= bands[0][0] <= inner[0] and inner[1] <= bands[0][1] <= outer[1]
    raise ValueError(f"unknown comparison {how}")


def decision_payload(value, source):
    return {"value": value, "evidence": {"adoptedFrom": source}, "rule": "adopted@1"}


def rule_of(decision):
    """The rule behind a graph decision: other_shot-14-226 and generator_plan-box are other_shot and generator_plan."""
    return decision.split("-")[0]


def run_rules(fx, art, root, rules, only=None):
    """Every rule with inputs in the fixture (keyed by graph decision), fed the adopted (delivered) values of earlier
    decisions. Returns {decision: (result | None, error | None)}. Writes only under ROOT/adopted/rule-inputs/<site>/
    (never over the recorded decision files, whose size and mtime are their cache signature) and a fresh replays/ folder."""
    folder = root / "adopted/rule-inputs" / fx["site"]
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in fx["decisions"].items():
        (folder / f"{name}.json").write_text(json.dumps(decision_payload(value, fx["site"])))
        (folder / f"{name}.operator.json").write_text(json.dumps({"frames": value} if isinstance(value, list) else value))
    out = {}
    for name, inputs in fx.get("decision_inputs", {}).items():
        rule = rules.get(rule_of(name))
        if rule is None or (only is not None and name not in only and rule_of(name) not in only):
            continue
        paths = {}
        for role, ref in inputs.items():
            if ref.startswith("@decision:"):
                paths[role] = folder / f"{ref.split(':', 1)[1]}.json"
            elif ref.startswith("@value:"):
                paths[role] = folder / f"{ref.split(':', 1)[1]}.operator.json"
            else:
                paths[role] = Path(expand(ref, art)) if ref.startswith("$") else Path(art) / ref
        if "out" in inspect.signature(rule).parameters:
            paths["out"] = root / "replays" / f"decision-{fx['site']}-{name}-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            result = rule(**paths)
            if "out" in paths:  # the rule wrote files there (dense_gate: icp/): its decision file goes beside them
                paths["out"].mkdir(parents=True, exist_ok=True)
                write_if_changed(paths["out"] / f"{rule_of(name)}.json", json.dumps(result, allow_nan=False))
                result = {**result, "folder": str(paths["out"])}
            out[name] = (result, None)
        except Exception as error:  # a rule that cannot run on cached data (e.g. ME340 lens needs a MoGe call) is reported, not fatal
            out[name] = (None, f"{type(error).__name__}: {error}")
    return out


def write_if_changed(path, text):
    """Keep a recorded file's mtime (its cache signature) when a re-run of adopt writes the same bytes."""
    path = Path(path)
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def adopt(fixture: Path, video: Path, store, *, art=None, graph=None, normalize=None, rules=None, db=None, replay=False, only_rules=None) -> AdoptReport:
    """store: Part A's Store; graph/normalize: Part B's report_runner.stages; rules: Part C's decide.RULES. Unit tests pass
    fakes for all of them. A stage is adopted (delivered scope) when its resolved argv equals the delivered run's, or when
    every difference is at a node the register lists; any other difference refuses it (reported exactly, not recorded).
    The proof fails when unlisted_diffs is not empty."""
    from report_runner.spec import Ctx
    fx = load_fixture(fixture)
    art = Path(art or getattr(store, "art", None) or DEFAULT_ART)
    root = Path(getattr(store, "state", None) or art / "runs/report-runner")
    if graph is None or normalize is None:
        from report_runner import stages  # Part B
        graph, normalize = graph or stages.graph, normalize or stages.normalize
    if rules is None:
        try:
            from report_runner.decide import RULES as rules  # Part C
        except ImportError:
            rules = {}
    site = fx["site"]
    report = AdoptReport(site=site, counts={"template": 0, "patch": 0, "staging": 0, "side": 0, "U": 0})
    listed = {}
    for d in fx["deviations"]:
        for i in d["nodes"]:
            listed.setdefault(i, []).append(d["id"])
    for n in fx["nodes"]:
        if n["patch"] and n["patch"] not in listed.get(n["id"], []):
            listed.setdefault(n["id"], []).append(n["patch"])

    # 1. the video: its bytes only, never its folder
    if sha256_file(video) != fx["video_sha256"]:
        report.unlisted_diffs.append(f"--video sha256 is not clip.json source.video_sha256 {fx['video_sha256'][:12]}")
        return report
    # 6a. the published report first: the import stage keeps its title and republishes its record (imports.jsonl)
    row = None
    if db is not False:
        row = import_row(site, None, fx["publication"], art, db or Database())
        if row["fingerprint"] != fx.get("expect", {}).get("fingerprint"):
            report.unlisted_diffs.append(f"published fingerprint {row['fingerprint'][:12]} is not the fixture's {fx.get('expect', {}).get('fingerprint', '')[:12]}")
        root.mkdir(parents=True, exist_ok=True)
        with open(root / "imports.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    occupant, adopted, specs_by_stage = occupants(fx), {}, {}

    def refuse(stage, node, rows):
        report.refused[stage] = [node["id"] if node else None, [f"{f} ({b}): {s}" for f, b, s in rows]]
        report.unlisted_diffs += [f"{node['id'] if node else stage}: {stage}: {f} ({b}): {s}" for f, b, s in rows]

    def adopt_stage(spec, decision=None, is_import=False):
        """Steps 2-3 for one graph stage: compare, check, hash, record the delivered key (under the graph's key)."""
        if spec.name in report.delivered_keys or spec.name in report.refused:
            return
        node = occupant.get(spec.name)
        if node is None and decision is None:
            return refuse(spec.name, None, [("<stage>", "M", "different: no delivered run stands for this stage")])
        rows = []
        if decision is not None:  # a decision stage: the fixture's value, plus the refined map it names (dense gate)
            folder = root / "adopted/decisions" / site / spec.name
            folder.mkdir(parents=True, exist_ok=True)
            write_if_changed(folder / spec.outputs["decision"], json.dumps(decision_payload(decision, f"{site} fixture")))
            outputs = {"decision": spec.outputs["decision"]}
            for role, target in (node["graph"][spec.name]["outputs"] if node else {}).items():
                link = folder / role
                if not link.is_symlink():
                    link.symlink_to(os.path.relpath(Path(art) / target, folder))
                outputs[role] = role
            where, lock = folder, {"stage": spec.name, "value": "adopted from the fixture", "occupant": node["id"] if node else None}
        elif is_import:  # the graph's last stage; its outputs are the published document (P5), its record the imports.jsonl row
            folder = root / "adopted/imports"
            folder.mkdir(parents=True, exist_ok=True)
            write_if_changed(folder / f"{site}.json", json.dumps(dict(row or {}, key=None), indent=1, ensure_ascii=False))
            rows = argv_differences(fx, node, spec.name, spec, adopted, normalize, art, video)
            outputs, where = {role: f"{site}.json" for role in spec.outputs}, folder
            lock = {"stage": spec.name, "publicationId": fx["publication"]}
        else:
            own = stage_outputs(node, spec.name, art)
            where = Path(art) / node["dirs"][0]
            missing = [r for r in spec.outputs if r not in own] + [r for r, p in own.items() if p is not None and not p.exists()]
            if missing:
                return refuse(spec.name, node, [("<outputs>", "M", f"different: graph roles {sorted(missing)} are not in the delivered run")])
            outputs = {r: None if p is None else os.path.relpath(p, where) for r, p in own.items() if r in spec.outputs}
            if node["staging"]:
                report.counts["staging"] += 1
                rows += [("<staging>", "M", f"different: {p}") for p in check_staging(node, art)]
            rows += argv_differences(fx, node, spec.name, spec, adopted, normalize, art, video)
            rows += staging_differences(spec, node, spec.name, adopted, art)
            rows += [("<commands>", "M", f"different: delivered also ran {' '.join(map(str, c))[:300]}") for c in unassigned_commands(fx, node)
                     if spec.name == next(s for s in node["graph"])]
            lock = {"stage": spec.name, "version": spec.version, "argv": [[expand(w, art) for w in c] for _, c in recorded_commands(fx, node, spec.name)],
                    "argvSource": "recorded", "models": node.get("models", []), "git": {"commit": None, "dirty": False}, "patch": node["patch"]}
        if node and not node["staging"] and decision is None and not is_import:
            report.counts["patch" if node["patch"] else "template"] += 1
        report.counts["U"] += sum(r[2] == "unrecorded" for r in rows)
        rows = [r for r in rows if r[2] != "unrecorded"]
        if rows and not (node and node["id"] in listed):
            return refuse(spec.name, node, rows)
        if rows:
            report.listed[spec.name] = [node["id"], listed[node["id"]], [f"{f} ({b}): {s}" for f, b, s in rows]]
            report.deviations_seen += listed[node["id"]]
        lock.update(adoptedFrom=node["dirs"] if node else None, differences=report.listed.get(spec.name, [None, [], []])[2],
                    note="adopted: a recorded run, not re-executed" + (f"; listed {listed[node['id']]}: the runner's argv differs as recorded" if rows else ""))
        try:
            key = store.record(dataclasses.replace(spec, outputs=outputs), where, "delivered-only", ["delivered"], lock,
                               adopted_from=node["id"] if node else f"decision:{spec.name}")
        except Exception as error:  # a producer was refused: this stage's key cannot be computed
            report.refused[spec.name] = [node["id"] if node else None, [f"not recorded: {type(error).__name__}: {error}"]]
            report.unlisted_diffs.append(f"{spec.name}: not recorded ({type(error).__name__}: {error})")
            return
        report.delivered_keys[spec.name] = key
        specs_by_stage[spec.name] = spec
        adopted[spec.name] = {"dir": where, "outputs": {r: where / rel for r, rel in outputs.items() if rel is not None}}

    def on_pending(p):
        for spec in p.specs:
            if "decision" not in spec.outputs:  # a decision stage is adopted when the graph asks for its value
                adopt_stage(spec)
        if p.decision not in fx["decisions"]:
            raise MissingDecision(f"the graph needs decision {p.decision!r}; the fixture has no delivered value for it")
        adopt_stage(next(s for s in p.specs if s.name == p.decision), decision=fx["decisions"][p.decision])
        ctx.decisions[p.decision] = decision_payload(fx["decisions"][p.decision], f"{site} fixture")

    # 2-3. the delivered graph, decisions served from the fixture as the graph asks for them
    ctx = Ctx(site, Path(video), fx["start"], fx["end"], "delivered", REPO / "docs/phase2/box-review-303", art, store)
    try:
        final = resolve(graph, ctx, on_pending)
        for spec in final[:-1]:
            adopt_stage(spec)
        adopt_stage(final[-1], is_import=True)
    except MissingDecision as error:
        report.unlisted_diffs.append(str(error))
        final = []
    for node in fx["nodes"]:  # side runs: provenance only, never looked up
        if node.get("side") and not node.get("graph"):
            report.counts["side"] += 1
            spec = side_spec(site, node, art)
            report.delivered_keys[node["id"]] = store.record(spec, Path(art) / node["dirs"][0], "delivered-only", ["delivered"],
                                                             {"stage": spec.name, "argvSource": "recorded", "patch": node["patch"], "side": True},
                                                             adopted_from=node["id"])
        elif node.get("graph") and not set(node["graph"]) & {s.name for s in final}:
            report.unlisted_diffs.append(f"{node['id']}: its stages {sorted(node['graph'])} are not in the delivered graph")
    if row is not None and final and final[-1].name in report.delivered_keys:
        row = dict(row, key=report.delivered_keys[final[-1].name])
        with open(root / "imports.jsonl", "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    report.import_row = row

    # 5. decisions: every rule whose inputs are adopted, against the delivered value
    computed = run_rules(fx, art, root, rules, only_rules)
    for name, (result, error) in computed.items():
        ids = [d["id"] for d in fx["deviations"] if name in d.get("decisions", [])]
        if error:
            report.decisions[name] = f"not run: {error}"
        elif compare_decision(result["value"], fx["decisions"].get(name), fx.get("decision_compare", {}).get(name)):
            report.decisions[name] = "equal"
        elif ids:
            report.decisions[name] = "listed " + " ".join(ids)
            report.deviations_seen += ids
        else:
            report.decisions[name] = "differs"
            report.unlisted_diffs.append(f"decision {name}: rule {result['value']!r} vs delivered {fx['decisions'].get(name)!r}")

    # 4. research keys: stages whose M2 argv is the delivered one (no difference at all) and whose code is known current,
    # and the research decisions the rules computed, as far as the research graph resolves on them
    research_ctx = Ctx(site, Path(video), fx["start"], fx["end"], "research", REPO / "docs/phase2/box-review-303", art, store)

    def research_stage(spec):
        node, delivered = occupant.get(spec.name), specs_by_stage.get(spec.name)
        if (spec.name in report.research_keys or node is None or delivered is None or spec.name in report.listed
                or node["patch"] or tuple(spec.commands) != tuple(delivered.commands)):
            return
        verified = verify_code(node, art)
        if not verified and replay and (node.get("verify") or {}).get("kind") == "replayed":
            verified = "replayed" if run_replay(node["verify"]["replay"], art, root=root)[0] else None
        if verified:
            try:
                key = store.record(dataclasses.replace(spec, outputs={r: os.path.relpath(p, adopted[spec.name]["dir"]) for r, p in adopted[spec.name]["outputs"].items()}),
                                   adopted[spec.name]["dir"], verified, ["delivered", "research"], {"stage": spec.name, "verification": verified},
                                   adopted_from=node["id"])
            except Exception:  # a producer has no research key
                return
            report.research_keys[spec.name] = [key, verified]

    def research_pending(p):
        for spec in p.specs:
            if "decision" not in spec.outputs:
                research_stage(spec)
        result, error = computed.get(p.decision, (None, "no rule output"))
        result = {"evidence": {}, "rule": f"{rule_of(p.decision)}@unversioned", **(result or {})}
        spec = next(s for s in p.specs if s.name == p.decision)
        wants = {w.split("=", 1)[0] for c in spec.commands for w in c[1:] if "=" in w and not w.startswith("-")}
        missing = wants - set(fx.get("decision_inputs", {}).get(p.decision, {}))
        if not error and missing:  # e.g. ME340 lens: no MoGe-3 run exists, the rule saw only the metric scale
            error = f"the rule ran without {sorted(missing)}"
        if error:
            raise MissingDecision(f"research decision {p.decision}: {error}")
        folder = Path(result.pop("folder", None) or root / "adopted/research-decisions" / site / p.decision)
        folder.mkdir(parents=True, exist_ok=True)
        write_if_changed(folder / spec.outputs["decision"], json.dumps(result, allow_nan=False))
        try:
            key = store.record(dataclasses.replace(spec, outputs={"decision": spec.outputs["decision"]}), folder, "replayed", ["research"],
                               {"stage": p.decision, "rule": result.get("rule")}, adopted_from=f"rule:{p.decision}")
            report.research_keys[p.decision] = [key, "replayed"]
        except Exception:
            pass
        research_ctx.decisions[p.decision] = result
    try:
        for spec in resolve(graph, research_ctx, research_pending):
            research_stage(spec)
    except MissingDecision as error:
        report.decisions["research-graph"] = f"stopped: {error}"
    except Exception as error:  # the delivered keys are recorded; a research graph that cannot resolve only has fewer keys
        report.decisions["research-graph"] = f"stopped: {type(error).__name__}: {error}"
    report.deviations_seen = sorted(set(report.deviations_seen))
    (root / "adopted").mkdir(parents=True, exist_ok=True)
    (root / "adopted" / f"{site}.json").write_text(json.dumps(asdict(report), indent=1, default=str, ensure_ascii=False))
    return report


# ---------------------------------------------------------------------------------------------------- self-check

def self_check():
    import tempfile
    doc = {"assets": [{"id": "a1", "sha256": "s1"}, {"id": "a2", "sha256": "s2", "metadata": {"generator": "SAM 3D"}},
                      {"id": "a3", "sha256": "s3", "metadata": {"generator": "parametric plane extension, no model call"}}],
           "entities": [{"id": "e1", "label": "bin", "associationState": "confirmed", "representations": [{"id": "r", "kind": "generated_mesh", "assetId": "a2"}]},
                        {"id": "e2", "label": "floor", "associationState": "confirmed", "representations": [{"id": "f", "kind": "generated_mesh", "assetId": "a3"}]},
                        {"id": "e3", "label": "x", "associationState": "association_pending", "representations": []}],
           "cameras": [{}], "observations": [{}, {}], "coordinateFrames": [{"scale": {"status": "model_estimated", "sourceRefs": [{"assetId": "a1"}]}}],
           "annotations": [{"kind": "import_provenance", "leftOutOfReport": {"cut_away_frames": ["0:1"]}, "limitations": ["old text"]}]}
    base = fingerprint(doc, "T")
    renamed = json.loads(json.dumps(doc).replace('"a1"', '"z1"').replace('"a2"', '"z2"').replace('"a3"', '"z3"'))
    assert fingerprint(renamed, "T") == base, "random asset ids do not change the fingerprint"
    fixed = json.loads(json.dumps(doc))
    fixed["annotations"][0]["limitations"] = ["new text"]
    assert fingerprint(fixed, "T") == base, "limitations are excluded (X13)"
    for change in (lambda d: d["assets"].__setitem__(1, {"id": "a2", "sha256": "other"}), lambda d: d["entities"][0].__setitem__("label", "box"),
                   lambda d: d["observations"].pop(), lambda d: d["annotations"][0]["leftOutOfReport"].__setitem__("cut_away_frames", []),
                   lambda d: d["coordinateFrames"][0]["scale"].__setitem__("status", "operator_anchored")):
        changed = json.loads(json.dumps(doc))
        change(changed)
        assert fingerprint(changed, "T") != base
    assert fingerprint(doc, "other title") != base
    assert counts(doc) == {"entities": 3, "confirmed": 2, "models": 1, "cameras": 1, "observations": 2, "assets": 3}
    assert forbidden("/x/.platform/imports/video-import-1.json") and not forbidden("/x/.platform/blobs/y")
    try:
        read_json("/x/.platform/imports/video-import-1.json")
        raise SystemExit("an import record was opened")
    except AssertionError:
        pass
    assert flag_map(["--a", "1", "2", "--b"]) == {"": [], "--a": ["1", "2"], "--b": []} and flag_map({"--a": 1}) == {"--a": 1}
    assert u_allowed("me340", "R24", "--marks") and not u_allowed("walmart", "R24", "--marks") and not u_allowed("me340", "R07", "--frames")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "d").mkdir()
        (tmp / "d/f").write_bytes(b"x")
        (tmp / "c").write_bytes(b"x")
        (tmp / "l").symlink_to("d")
        assert check_staging({"links": {"l": "d"}, "copies": {"c": "d/f"}}, tmp) == []
        (tmp / "c").write_bytes(b"y")
        assert check_staging({"links": {"l": "c"}, "copies": {"c": "d/f"}}, tmp) == ["link l does not resolve to c", "copy c differs from d/f"]
        assert digest_path(tmp / "l")[2] == digest_path(tmp / "d")[2], "a folder digest follows links"
    say("adopt self-check passed: fingerprint ignores asset ids and limitations but sees every other field; counts; "
        "import records never opened; staging links/copies; U flags only where allowed")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", nargs="?", type=Path)
    parser.add_argument("--video", type=Path, help="the source MP4 named in the fixture's clip.json (the file is hashed; its folder is never listed)")
    parser.add_argument("--replay", action="store_true", help="run the CPU replays (cuts, ICP, merge, register, floor, outlines) to verify research keys")
    parser.add_argument("--fingerprint", metavar="PUBLICATION_ID", help="print the published document's fingerprint and counts (read-only SELECT)")
    parser.add_argument("--self-check", action="store_true")
    a = parser.parse_args()
    if a.self_check:
        return self_check()
    if a.fingerprint:
        p = Database().publication(a.fingerprint)
        return say(json.dumps({"publicationId": p["publicationId"], "title": p["title"], "fingerprint": fingerprint(p["document"], p["title"]), **counts(p["document"])}, ensure_ascii=False))
    from report_runner.store import Store, art_root  # Part A
    report = adopt(a.fixture, a.video, Store(art_root()), replay=a.replay)
    say(json.dumps(asdict(report), indent=1, default=str, ensure_ascii=False))
    sys.exit(1 if report.unlisted_diffs else 0)


if __name__ == "__main__":
    main()
