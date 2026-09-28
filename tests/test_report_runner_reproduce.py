"""Part D of the M2 report runner: the delivered fixtures, adopt, the published fingerprints, publish, and the proof of
reproduction P1-P7 (runner design §8).

Unit tests run anywhere (a toy fixture, a fake store and graph). The rest needs this Mac's research data ($ART) and skips
without it; P1-P7 also need Parts A-C merged (report_runner.store/stages/decide/profiles) and skip until then. P2 and P7
read the one-time adopt ($ART/runs/report-runner/adopted/<site>.json, `python -m report_runner.adopt FIXTURE --video MP4`).
P4 and P5 are long CPU runs (ICP ~5 min per site; P5 builds each document offline, 24-60 min per site): set
PANOPTES_LONG_TESTS=1 to run them. Nothing here calls a GPU or Modal, opens .platform/imports/*, or writes under the
existing runs/ or data/ (adopt state and replays go to a temporary folder).
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps"), str(REPO)]
from report_runner import adopt as A  # noqa: E402
from report_runner import publish as P  # noqa: E402
from report_runner.spec import Ctx, Pending, StageSpec  # noqa: E402

FIXTURES = REPO / "tests/fixtures/delivered-303"
ART = A.DEFAULT_ART
HAVE_ART = (ART / "runs").is_dir()
REGISTER_73 = {"me340": 11, "samsclub-a2": 13, "walmart": 1}  # §7.3 patches per site (222 counts its infer and build)
needs_art = pytest.mark.skipif(not HAVE_ART, reason=f"no research data at {ART}")
long_run = pytest.mark.skipif(os.environ.get("PANOPTES_LONG_TESTS") != "1", reason="long CPU run: set PANOPTES_LONG_TESTS=1")


def merged():
    try:
        for part in ("store", "stages", "decide", "profiles"):
            importlib.import_module(f"report_runner.{part}")
        return True
    except ImportError:
        return False


needs_merge = pytest.mark.skipif(not (HAVE_ART and merged()), reason="needs $ART and Parts A-C (report_runner.store/stages/decide/profiles)")


def fixture(site):
    return json.loads((FIXTURES / f"{site}.json").read_text())


# ------------------------------------------------------------------------------------------ the three delivered fixtures

@pytest.mark.parametrize("site", A.SITES)
def test_fixture_schema(site):
    fx = fixture(site)
    assert A.validate_fixture(fx) == [] and fx["site"] == site
    assert set(fx["patches_73"]) <= {n["id"] for n in fx["nodes"] if n["patch"]}, "every §7.3 patch is a patch node"
    assert len(fx["patches_73"]) == REGISTER_73[site]
    assert all(n["patch"] in {d["id"] for d in fx["deviations"]} for n in fx["nodes"] if n["patch"]), "every patch carries a listed deviation"
    # every run that is not a side run stands for graph stages (node.graph); validate_fixture checks one occupant per stage
    assert [n["id"] for n in fx["nodes"] if not n.get("side") and not n.get("graph")] == []
    assert all(not n.get("graph") for n in fx["nodes"] if n.get("side"))
    assert [n for n in fx["nodes"] if n["stage"] == "R37"] and fx["expect"]["fingerprint"]
    assert set(fx["decision_inputs"]) <= set(fx["decisions"])
    for d in fx["deviations"]:
        assert set(d["decisions"]) <= set(fx["decisions"]), d["id"]


def test_register_is_covered():
    """The union of the three fixtures lists every §9 deviation; the sites are named as box-review-303."""
    seen = {d["id"] for s in A.SITES for d in fixture(s)["deviations"]}
    assert seen == A.REGISTER
    assert {p.stem for p in (REPO / "docs/phase2/box-review-303").glob("*.json")} >= set(A.SITES)
    shared = [{d["id"] for d in fixture(s)["deviations"]} for s in A.SITES]
    assert {"X13", "D6", "O1"} <= set.intersection(*shared)


@needs_art
@pytest.mark.parametrize("site", A.SITES)
def test_fixture_matches_disk(site):
    """Every output exists; every D/I evidence check and every staging check passes on the cached runs (CPU, read-only)."""
    fx, bad = fixture(site), []
    for n in fx["nodes"]:
        bad += [f"{n['id']}: missing {rel}" for rel in n["outputs"].values() if not (ART / rel).exists()]
        for c in n["evidence"]:
            if c["kind"] != "replay":
                ok, detail = A.run_check(ART, c)
                bad += [] if ok else [f"{n['id']}: {detail}"]
        if n["staging"]:
            bad += [f"{n['id']}: {p}" for p in A.check_staging(n, ART)]
    assert bad == []
    clip = json.loads((ART / f"data/clips/{fx['clip']}/clip.json").read_text())
    assert clip["source"]["video_sha256"] == fx["video_sha256"]


@needs_art
def test_published_fingerprints():
    """fingerprint() of c40fbd08 / 913daf2a / 32cec650 (read-only SELECT) is the one recorded in the fixtures, with
    ME340 124 entities / 123 confirmed / 22 models, Sam's Club 67 models, Walmart 40."""
    try:
        db = A.Database()
        rows = {s: db.publication(fixture(s)["publication"]) for s in A.SITES}
    except Exception as error:  # no local platform database on this machine
        pytest.skip(f"platform database not reachable: {type(error).__name__}")
    for site, p in rows.items():
        expect = fixture(site)["expect"]
        assert A.fingerprint(p["document"], p["title"]) == expect["fingerprint"], site
        assert A.counts(p["document"]) == {k: expect[k] for k in ("entities", "confirmed", "models", "cameras", "observations", "assets")}
    assert [A.counts(rows[s]["document"])["models"] for s in A.SITES] == [22, 67, 40]
    assert A.counts(rows["me340"]["document"])["entities"] == 124 and A.counts(rows["me340"]["document"])["confirmed"] == 123


# ------------------------------------------------------------------------------------------------------------ publish

def test_publish_dry_run_runs_nothing(capsys):
    calls = []
    argvs = P.publish("0000-pub", dry_run=True, catalog="/cat", http="/http", run=lambda *a, **k: calls.append(a))
    printed = capsys.readouterr().out.strip().splitlines()
    assert calls == [] and len(argvs) == 4 and len(printed) == 4
    assert "export_platform_publication.py" in printed[0] and "--api http://127.0.0.1:8792" in printed[0] and "--output /cat/0000-pub" in printed[0]
    assert "check_publication_site.py --catalog /cat --source-api" in printed[1] and "prepare_publication_site.py --catalog /cat --output /http" in printed[2]
    assert printed[3].startswith("PANOPTES_PUBLICATION_CATALOG=/cat PANOPTES_PUBLICATION_HTTP=/http ") and "deploy" in printed[3]
    P.self_check()


# ------------------------------------------------------------------------------------------------ adopt, toy fixture

SCRIPT = "scripts/report_runner/spec.py"  # any repo file: a recorded script sha256 that equals HEAD's


class FakeStore:
    """Part A's Store surface adopt uses: key(spec) raises until producers are recorded; record(...) -> key."""

    def __init__(self, art):
        self.art, self.state, self.entries, self.resolved = Path(art), Path(art) / "state", [], {}

    def key(self, spec):
        digests = []
        for role, (producer, _) in sorted(spec.inputs.items()):
            if (spec.site, producer) not in self.resolved:
                raise LookupError(f"unresolved {producer}")
            digests.append([role, self.resolved[(spec.site, producer)]["outputDigest"]])
        return hashlib.sha256(json.dumps([spec.name, spec.version, [list(c) for c in spec.commands], digests, spec.rules]).encode()).hexdigest()

    def record(self, spec, dir, verification, scope, lock, adopted_from=None):
        outputs = {r: [rel, A.digest_path(Path(dir) / rel)[2]] for r, rel in spec.outputs.items()}
        entry = {"key": self.key(spec), "stage": spec.name, "dir": str(dir), "outputs": outputs, "verification": verification, "scope": scope,
                 "outputDigest": hashlib.sha256(json.dumps(sorted(outputs.items())).encode()).hexdigest(), "adoptedFrom": adopted_from}
        (self.state / "adopted/locks").mkdir(parents=True, exist_ok=True)
        (self.state / "adopted/locks" / f"{entry['key']}.json").write_text(json.dumps(lock))
        self.entries.append(entry)
        self.resolved[(spec.site, spec.name)] = entry
        return entry["key"]


def S(name, commands, inputs=None, outputs=None, rules=None):
    return StageSpec(name=name, site="toy", version=1, deps=(), commands=tuple(tuple(c) for c in commands), inputs=inputs or {}, consumes={}, leaves={},
                     outputs=outputs or {}, stage_from={}, compute="cpu", gpu=None, timeout_s=60, worst_usd=0., est_usd=0., est_s=1., budget_flags={},
                     models=(), rules=rules or {}, env={})


def toy_graph(ctx):
    """R01 clip -> R02 cuts -> decision 'shots' -> R07 camera (--frames from the decision) -> R10 mask root, R12 tracks -> R37."""
    specs = [S("R01", [["scripts/prepare_video_clip.py", "--video", str(ctx.video), "--start", "1", "--end", "2", "--name", "toy", "--output", "@new"]],
               outputs={"clip": "clip.json"}),
             S("R02", [["scripts/detect_shot_cuts.py", "--clip", "@R01", "--output", "@new/segments.json"]], {"clip": ("R01", ())}, {"segments": "segments.json"}),
             S("shots", [["python", "-m", "report_runner.decide", "shots", "--out", "@new", "segments=@R02:segments"]], {"segments": ("R02", ("segments",))},
               {"decision": "shots.json"})]
    if "shots" not in ctx.decisions:
        raise Pending("shots", specs)
    a, b = ctx.decisions["shots"]["value"]["primary"]
    return specs + [
        S("R07", [["modal_apps/droid_room.py", "execute", "--clip", "@R01", "--frames", f"{a}:{b}", "--output", "@new"]], {"clip": ("R01", ())},
          {"prediction": "prediction.npz"}, {"shots": ctx.decisions["shots"]["rule"]}),
        S("R10", [], {"camera": ("R07", ())}, {"floor_a": "floor-a"}),
        S("R12", [["modal_apps/sam3_motion_tracks.py", "--droid-run", "@R07", "--output", "@new"]], {"camera": ("R07", ())}, {"tracks": "tracks.json"}),
        S("R37", [["scripts/import_video_scene.py", "--droid-run", "@R07", "--title", "Toy"]], {"camera": ("R07", ())}, {"record": "import.json"})]


def toy_normalize(script, argv, owner_of):
    """Enough of Part B's normalize for the toy: --output dropped, @STAGE[:ROLE] and adopted paths both become (stage, role)."""
    out, skip = [script], False
    for word in argv:
        if skip:
            skip = False
            continue
        if word == "--output":
            skip = True
            continue
        if word.startswith("@"):
            stage, _, role = word[1:].partition(":")
            out.append([stage, role])
        else:
            owned = owner_of(word) if os.path.isabs(word) or word.startswith("$") else None
            out.append(list(owned) if owned else word)
    return out


def shots_rule(segments):
    cuts = json.loads(Path(segments).read_text())["segments"]
    return {"value": {"primary": [cuts[-1][0], cuts[-1][1] + 1], "others": [], "mapped": True}, "evidence": {}, "rule": "shots@1"}


class FakeDB:
    def __init__(self, document):
        self.document = document

    def publication(self, publication_id):
        return {"publicationId": publication_id, "projectId": "proj", "title": "Toy", "document": self.document, "createdAt": "t"}

    def project_publications(self, project_id):
        return ["pub-new", "pub-toy"]


DOC = {"assets": [{"id": "x", "sha256": "a" * 64, "metadata": {}}], "entities": [], "cameras": [], "observations": [], "coordinateFrames": [{"scale": {}}],
       "annotations": [{"kind": "import_provenance", "leftOutOfReport": {}}]}


def make_toy(tmp):
    """A toy $ART, its video and its fixture: a template clip/cuts/camera, a staging mask root, a patch occupant, a side run."""
    art = tmp / "art"
    files = {"data/clips/toy/clip.json": json.dumps({"source": {"frames": 10}}), "runs/toy-cuts/segments.json": json.dumps({"segments": [[0, 9]]}),
             "runs/toy-cam/prediction.npz": "npz", "runs/toy-cam/run.json": json.dumps({"shot_frames": [0, 10], "script_sha256": A.sha256_file(REPO / SCRIPT)}),
             "runs/toy-tracks/tracks.json": "{}", "runs/toy-old/prediction.npz": "old"}
    for rel, text in files.items():
        (art / rel).parent.mkdir(parents=True, exist_ok=True)
        (art / rel).write_text(text)
    (art / "runs/toy-masks").mkdir(parents=True)
    (art / "runs/toy-masks/floor-a").symlink_to("../toy-cam")
    (art / ".platform/imports").mkdir(parents=True)
    record = art / ".platform/imports/video-import-pub-toy.json"
    record.write_text('{"capability": "never read"}')
    record.chmod(0)  # adopt finds it by name; reading it would fail the test
    video = tmp / "video.mp4"
    video.write_bytes(b"toy video")
    nodes = [
        {"id": "T01", "stage": "R01", "name": "clip", "dirs": ["data/clips/toy"], "outputs": {"clip": "data/clips/toy/clip.json"},
         "commands": [["scripts/prepare_video_clip.py", "--video", "$VIDEO", "--start", "1", "--end", "2", "--name", "toy", "--output", "$ART/data/clips/toy"]],
         "basis": {}, "evidence": [], "patch": None, "staging": False, "inputs": [],
         "verify": {"kind": "content", "checks": [{"kind": "json_value", "file": "data/clips/toy/clip.json", "path": "source.frames", "value": 10}]},
         "graph": {"R01": {"outputs": {"clip": "data/clips/toy/clip.json"}, "commands": [["T01", 0]]}}},
        {"id": "T02", "stage": "R02", "name": "cuts", "dirs": ["runs/toy-cuts"], "outputs": {"segments": "runs/toy-cuts/segments.json"},
         "commands": [["scripts/detect_shot_cuts.py", "--clip", "$ART/data/clips/toy", "--output", "$ART/runs/toy-cuts/segments.json"]],
         "basis": {}, "evidence": [], "patch": None, "staging": False, "inputs": ["T01"],
         "graph": {"R02": {"outputs": {"segments": "runs/toy-cuts/segments.json"}, "commands": [["T02", 0]]}}},
        {"id": "T07", "stage": "R07", "name": "camera", "dirs": ["runs/toy-cam"], "outputs": {"prediction": "runs/toy-cam/prediction.npz"},
         "commands": [["modal_apps/droid_room.py", "execute", "--clip", "$ART/data/clips/toy", "--frames", "0:10", "--output", "$ART/runs/toy-cam"]],
         "basis": {"--frames": "D"}, "evidence": [{"kind": "json_value", "file": "runs/toy-cam/run.json", "path": "shot_frames", "value": [0, 10]}],
         "patch": None, "staging": False, "inputs": ["T01", "T02"], "verify": {"kind": "recorded-rev", "file": "runs/toy-cam/run.json", "field": "script_sha256", "script": SCRIPT},
         "graph": {"R07": {"outputs": {"prediction": "runs/toy-cam/prediction.npz"}, "commands": [["T07", 0]]}}},
        {"id": "T10", "stage": "R10", "name": "mask root", "dirs": ["runs/toy-masks"], "outputs": {"floor_a": "runs/toy-masks/floor-a"}, "commands": [],
         "basis": {}, "evidence": [], "patch": None, "staging": True, "inputs": ["T07"], "links": {"runs/toy-masks/floor-a": "runs/toy-cam"},
         "graph": {"R10": {"outputs": {"floor_a": "runs/toy-masks/floor-a"}, "commands": []}}},
        {"id": "T11", "stage": "R07", "name": "old camera", "dirs": ["runs/toy-old"], "outputs": {"prediction": "runs/toy-old/prediction.npz"},
         "commands": [["modal_apps/droid_room.py", "execute", "--clip", "old"]], "basis": {}, "evidence": [], "patch": "X1", "staging": False, "inputs": ["T01"], "side": True},
        {"id": "T12", "stage": "R12", "name": "tracks", "dirs": ["runs/toy-tracks"], "outputs": {"tracks": "runs/toy-tracks/tracks.json"},
         "commands": [["modal_apps/sam3_motion_tracks.py", "--droid-run", "$ART/runs/toy-old", "--output", "$ART/runs/toy-tracks"]],
         "basis": {}, "evidence": [], "patch": "X1", "staging": False, "inputs": ["T11"],
         "graph": {"R12": {"outputs": {"tracks": "runs/toy-tracks/tracks.json"}, "commands": [["T12", 0]]}}},
        {"id": "IMPORT", "stage": "R37", "name": "import", "dirs": [], "outputs": {}, "commands": [["scripts/import_video_scene.py", "--droid-run", "$ART/runs/toy-cam", "--title", "Toy"]],
         "basis": {"--title": "M"}, "evidence": [], "patch": None, "staging": False, "inputs": ["T07"],
         "graph": {"R37": {"outputs": {}, "commands": [["IMPORT", 0]]}}}]
    fx = {"site": "me340", "clip": "toy", "video_sha256": A.sha256_file(video), "start": 1, "end": 2, "publication": "pub-toy",
          "expect": {"fingerprint": A.fingerprint(DOC, "Toy")}, "nodes": nodes,
          "decisions": {"shots": {"primary": [0, 10], "others": [], "mapped": True}}, "decision_inputs": {"shots": {"segments": "runs/toy-cuts/segments.json"}},
          "deviations": [{"id": "X1", "class": "X", "nodes": ["T11", "T12"], "delivered": "old", "m2": "new", "reason": "toy", "decisions": {},
                          "flags": {"T12": ["--droid-run"]}}]}
    return art, video, fx


def run_toy(tmp, change=None, rules=None):
    art, video, fx = make_toy(tmp)
    if change:
        change(fx, art)
    path = tmp / "fx.json"
    path.write_text(json.dumps(fx))
    store = FakeStore(art)
    report = A.adopt(path, video, store, graph=toy_graph, normalize=toy_normalize, rules={"shots": shots_rule} if rules is None else rules, db=FakeDB(DOC))
    return report, store, art


def test_adopt_toy_happy_path(tmp_path):
    report, store, art = run_toy(tmp_path)
    assert report.unlisted_diffs == []
    assert set(report.delivered_keys) == {"R01", "R02", "shots", "R07", "R10", "T11", "R12", "R37"}, "keys by graph stage; a side run by its node"
    assert set(report.research_keys) == {"R01", "R07", "shots"} and report.research_keys["R07"][1] == "recorded-rev" and report.research_keys["R01"][1] == "content"
    assert report.deviations_seen == ["X1"] and report.decisions == {"shots": "equal"}
    assert report.counts == {"template": 3, "patch": 1, "staging": 1, "side": 1, "U": 0}
    assert report.listed["R12"][:2] == ["T12", ["X1"]] and report.refused == {}, "the patch's argv differs as listed; nothing refused"
    by = {}
    for e in store.entries:  # the delivered entry first; a research entry for the same node follows it
        by.setdefault(e["adoptedFrom"], e)
    assert [e["scope"] for e in store.entries if e["adoptedFrom"] == "T07"] == [["delivered"], ["delivered", "research"]]
    assert by["T12"]["stage"] == "R12" and by["T11"]["stage"] == "R07~T11", "patch occupants keep the graph's stage; side runs a name of their own"
    assert by["T07"]["verification"] == "delivered-only" and all(e["scope"] == ["delivered"] for e in store.entries if e["verification"] == "delivered-only")
    assert json.loads((store.state / "adopted/decisions/me340/shots/shots.json").read_text())["value"] == {"primary": [0, 10], "others": [], "mapped": True}
    row = json.loads((store.state / "imports.jsonl").read_text().splitlines()[-1])
    assert row["fingerprint"] == A.fingerprint(DOC, "Toy") and row["importRecordPath"].endswith("video-import-pub-toy.json")
    assert row["republishNext"].endswith("video-import-pub-toy.json"), "the newest publication with a record on disk (pub-new has none)"
    assert len(list((store.state / "adopted/locks").glob("*.json"))) == len({e["key"] for e in store.entries})
    assert json.loads((store.state / "adopted/me340.json").read_text())["site"] == "me340"
    for e in store.entries:  # nothing adopt writes later (rule inputs, the import row) touches a recorded output
        assert all(A.digest_path(Path(e["dir"]) / rel)[2] == sha for rel, sha in e["outputs"].values()), e["stage"]


def test_adopt_refuses_another_video(tmp_path):
    def other(fx, art):
        fx["video_sha256"] = "0" * 64
    report, store, _ = run_toy(tmp_path, other)
    assert report.unlisted_diffs and "video" in report.unlisted_diffs[0] and store.entries == []


def test_adopt_argv_differences(tmp_path):
    def m_flag(fx, art):  # the recorded --frames is not what the graph resolves: unlisted
        fx["nodes"][2]["commands"][0][5] = "0:9"
    report, _, _ = run_toy(tmp_path / "a", m_flag)
    assert any(d.startswith("T07: R07: --frames (D): different: runner [\"0:10\"] vs delivered [\"0:9\"]") for d in report.unlisted_diffs), report.unlisted_diffs
    assert "R07" in report.refused and "R07" not in report.delivered_keys, "an unlisted difference is refused, not recorded"

    def listed(fx, art):  # the same difference, named by a register entry: seen, not unlisted
        m_flag(fx, art)
        fx["deviations"][0]["nodes"].append("T07")
        fx["deviations"][0]["flags"]["T07"] = ["--frames"]
    report, _, _ = run_toy(tmp_path / "b", listed)
    assert report.unlisted_diffs == [] and report.deviations_seen == ["X1"] and report.listed["R07"][1] == ["X1"]

    def elsewhere(fx, art):  # the entry lists the node but names another flag: the --frames difference is still refused
        listed(fx, art)
        fx["deviations"][0]["flags"]["T07"] = ["--clip"]
    report, _, _ = run_toy(tmp_path / "d", elsewhere)
    assert "R07" in report.refused and any(d.startswith("T07: R07: --frames (D): different") for d in report.unlisted_diffs), report.unlisted_diffs

    def evidence(fx, art):  # equal argv, but the D flag's evidence check fails
        (art / "runs/toy-cam/run.json").write_text(json.dumps({"shot_frames": [0, 9]}))
    report, _, _ = run_toy(tmp_path / "c", evidence)
    assert any("T07: R07: --frames (D): evidence-failed" in d for d in report.unlisted_diffs)


def test_adopt_staging_and_decisions(tmp_path):
    def relink(fx, art):
        (art / "runs/toy-masks/floor-a").unlink()
        (art / "runs/toy-masks/floor-a").symlink_to("../toy-cuts")
    report, _, _ = run_toy(tmp_path / "a", relink)
    assert any("link runs/toy-masks/floor-a does not resolve" in d for d in report.unlisted_diffs)

    off = {"shots": lambda segments: {"value": {"primary": [1, 10], "others": [[0, 1]], "mapped": True}}}
    report, _, _ = run_toy(tmp_path / "b", rules=off)
    assert report.decisions == {"shots": "differs: others, primary"} and any(d.startswith("decision shots") for d in report.unlisted_diffs)

    def adopted_with(tmp, decisions, compare=None):
        art, video, fx = make_toy(tmp)
        fx["deviations"][0]["decisions"] = decisions
        fx["decision_compare"] = compare or {}
        (tmp / "fx.json").write_text(json.dumps(fx))
        return A.adopt(tmp / "fx.json", video, FakeStore(art), graph=toy_graph, normalize=toy_normalize, rules=off, db=FakeDB(DOC))
    report = adopted_with(tmp_path / "c", {"shots": ["primary", "others"]})
    assert report.decisions == {"shots": "listed X1: others, primary"} and report.unlisted_diffs == []
    report = adopted_with(tmp_path / "d", {"shots": ["primary"]})  # an entry excuses only the fields it names
    assert report.decisions == {"shots": "differs: others"} and len(report.unlisted_diffs) == 1
    report = adopted_with(tmp_path / "e", {"shots": ["primary"]}, {"shots": {"fields": ["primary", "mapped"]}})  # 'others' is evidence here
    assert report.decisions == {"shots": "listed X1: primary"} and report.unlisted_diffs == []
    assert A.decision_differences({"bands": [[648, 696]]}, {"bands": [[648, 704]]}) == ["bands"], "a band inside a tolerance is still a difference"
    assert A.decision_differences({"ids": {"a": "accepted by SAM 3D"}}, {"ids": {"a": "modelled in 242"}}, {"ids": ["ids"]}) == [], "ids, not notes"


def test_a_research_decision_names_exactly_what_its_rule_read(tmp_path):
    """The blocker of the M2 review: a research decision is recorded only when its key's roles are the fixture's
    decision_inputs, both ways, each the same file or the same earlier value; otherwise it is refused and the research graph
    stops there. Omitting an input the rule read, naming one it did not read, or another file: refused."""
    report, _, _ = run_toy(tmp_path / "ok")
    assert report.research_keys["shots"][1] == "replayed" and "research-graph" not in report.decisions
    both = {"shots": lambda segments=None, operator=None: shots_rule(segments) if segments else
            {"value": {"primary": [0, 10], "others": [], "mapped": True}, "evidence": {}, "rule": "shots@1"}}

    def read_more(fx, art):  # the rule also read the operator list; the research key does not name it
        fx["decision_inputs"]["shots"]["operator"] = "runs/toy-cuts/segments.json"

    def read_less(fx, art):  # the research key names segments; the rule read nothing
        fx["decision_inputs"]["shots"] = {}

    def other_file(fx, art):  # the same bytes in another run: not the file the key names
        (art / "runs/toy-cuts2").mkdir()
        (art / "runs/toy-cuts2/segments.json").write_bytes((art / "runs/toy-cuts/segments.json").read_bytes())
        fx["decision_inputs"]["shots"]["segments"] = "runs/toy-cuts2/segments.json"
    for change, why in ((read_more, "the rule read ['operator'], which the research key does not name"),
                        (read_less, "the research key names ['segments'], which the rule did not read"),
                        (other_file, "segments: the rule read runs/toy-cuts2/segments.json, the research key names @R02:segments")):
        report, store, _ = run_toy(tmp_path / change.__name__, change, rules=both)
        assert report.decisions["shots"] == "equal" and "shots" not in report.research_keys, change.__name__
        assert why in report.decisions["research-graph"] and not [e for e in store.entries if e["stage"] == "shots" and "research" in e["scope"]]


def test_a_register_node_gets_no_research_key(tmp_path):
    """P7: a node the register lists is a research miss even when its argv is the delivered one (D8: the clip's K). Its
    consumers still get research keys: they hold its bytes, so a research rebuild with equal bytes is an early cutoff."""
    def unverified(fx, art):
        fx["deviations"].append({"id": "D8", "class": "D", "nodes": ["T01"], "delivered": "K", "m2": "pinned", "reason": "unverified",
                                 "decisions": {}, "flags": {}})
    report, _, _ = run_toy(tmp_path, unverified)
    assert report.unlisted_diffs == [] and "R01" in report.delivered_keys and "R01" not in report.research_keys and "R07" in report.research_keys


def test_adopt_fingerprint_mismatch_is_reported(tmp_path):
    def stale(fx, art):
        fx["expect"]["fingerprint"] = "f" * 64
    report, _, _ = run_toy(tmp_path, stale)
    assert any("published fingerprint" in d for d in report.unlisted_diffs)


def test_validate_catches_bad_fixtures(tmp_path):
    _, _, fx = make_toy(tmp_path)
    fx["nodes"][2]["basis"]["--clip"] = "U"
    fx["nodes"][5]["patch"] = "X99"
    fx["nodes"][1]["inputs"] = ["T07"]
    problems = A.validate_fixture(fx)
    assert any("U flag --clip not allowed" in p for p in problems) and any("X99" in p for p in problems) and any("T02: input T07" in p for p in problems)


def test_import_records_are_never_opened(tmp_path):
    folder = tmp_path / ".platform/imports"
    folder.mkdir(parents=True)
    (folder / "video-import-abc.json").write_text("x")
    (folder / "video-import-abc.json").chmod(0)
    assert A.import_record(tmp_path, "abc").endswith("video-import-abc.json") and A.import_record(tmp_path, "zzz") is None
    with pytest.raises(AssertionError):
        A.sha256_file(folder / "video-import-abc.json")


def test_self_checks():
    A.self_check()


# ----------------------------------------------------------------------------------- P1-P7 (after Parts A-C are merged)

def real_store(tmp):
    """Part A's Store on the real $ART, with its state (index, locks, decisions, replays) in a temporary folder."""
    from report_runner.store import Store
    store = Store(ART)
    store.state, store.index = tmp / "report-runner", tmp / "report-runner/keys.jsonl"
    return store


def video_of(fx):
    path = Path(fx["video_path_recorded"])  # the recorded source MP4: hashed, its folder never listed
    if not path.is_file():
        pytest.skip(f"source video for {fx['site']} is not at its recorded path")
    return path


@needs_merge
@pytest.mark.parametrize("site", A.SITES)
def test_p1_argv_and_p3_decisions(site, tmp_path):
    """P1: every non-patch, non-staging node's normalised runner argv equals the fixture's, nothing unlisted.
    P3: each rule on cached evidence equals the delivered value or is in the register (dense_gate skipped: P4 runs ICP)."""
    fx = fixture(site)
    rules = [n for n in fx["decision_inputs"] if n != "dense_gate"]
    report = A.adopt(FIXTURES / f"{site}.json", video_of(fx), real_store(tmp_path), only_rules=rules)
    print(site, report.counts, report.decisions)
    assert report.unlisted_diffs == []
    assert not [n for n, v in report.decisions.items() if v.startswith("differs")]


def served(store, site, fx, profile):
    """A's main() loop, dry: decisions served from the store (a hit) or the plan stops there."""
    from report_runner import stages
    from report_runner.store import print_plan
    ctx = Ctx(site, video_of(fx), fx["start"], fx["end"], profile, REPO / "docs/phase2/box-review-303", ART, store)
    while True:
        try:
            specs = stages.graph(ctx)
            return store.plan(specs), ctx
        except Pending as p:
            rows = store.plan(p.specs)
            print_plan(rows)
            spec = next(s for s in p.specs if s.name == p.decision)
            hit = store.lookup(spec)
            if hit is None:
                return rows, ctx
            ctx.decisions[p.decision] = json.loads((hit.outputs.get("decision") or next(iter(hit.outputs.values()))).read_text())


def adopted(site):
    if not (ART / "runs/report-runner/adopted" / f"{site}.json").exists():
        pytest.skip(f"run the one-time adopt first: python -m report_runner.adopt tests/fixtures/delivered-303/{site}.json --video MP4")


@needs_merge
@pytest.mark.parametrize("site", A.SITES)
def test_p2_delivered_profile_is_all_hits(site):
    """P2: --profile delivered --review box-review-303 --dry-run: every stage a hit, 0 calls, $0.00; --verify: every sha256."""
    adopted(site)
    from report_runner.store import Store
    fx = fixture(site)
    for verify in (False, True):
        rows, _ = served(Store(ART, scope="delivered", verify=verify), site, fx, "delivered")
        assert [r["stage"] for r in rows if r["status"] != "hit"] == []
        assert sum(r["usd"] for r in rows) == 0


@needs_merge
@long_run
@pytest.mark.parametrize("site", A.SITES)
def test_p4_replays(site, tmp_path):
    """P4: merge --against exits 0, ICP byte-identical, cuts equal, register-from-journal equals 304, inferred-floor kind."""
    fx = fixture(site)
    failed = []
    for n in fx["nodes"]:
        v = n.get("verify") or {}
        if v.get("kind") == "replayed":
            ok, detail, _ = A.run_replay(v["replay"], ART, root=tmp_path)
            failed += [] if ok else [f"{n['id']}: {detail}"]
    assert failed == []


def offline_document(fx):
    """P5: import_video_scene.build_document with the fixture's import argv and a stub asset store (no DB, no blobs)."""
    import argparse
    import import_video_scene
    import mono_room
    argv = next(n for n in fx["nodes"] if n["stage"] == "R37")["commands"][0][1:]
    names = ("droid-run depth-run object-map masks policy models video exclude-frames dense-points splats full-video analysis inferred-floor shell-glb "
             "comparison-video republish dynamic-scene video-events skeleton-scene dynamic-analysis title output-dir request-suffix").split()
    args = argparse.Namespace(**{n.replace("-", "_"): None for n in names})
    i = 0
    while i < len(argv):
        flag, values = argv[i], []
        i += 1
        while i < len(argv) and not argv[i].startswith("--"):
            values.append(A.expand(argv[i], ART))
            i += 1
        name = flag[2:].replace("-", "_")
        setattr(args, name, values if name == "exclude_frames" else values[0] if name in ("title", "request_suffix") else Path(values[0]))

    def put_asset(data, media_type, metadata):
        sha = hashlib.sha256(data).hexdigest()
        return {"id": "stub-" + sha, "sha256": sha, "sizeBytes": len(data), "mediaType": media_type, "storageKey": "sha256/" + sha, "metadata": metadata}
    mono_room.use_clip(args.droid_run)
    clip = {"K": mono_room.SOURCE_K, "D": mono_room.SOURCE_D, "dataset": mono_room.DATASET}
    return import_video_scene.build_document(args, put_asset, clip, clip["dataset"])[0], args.title


@needs_merge
@long_run
@pytest.mark.parametrize("site", A.SITES)
def test_p5_offline_import_fingerprint(site):
    """P5: the offline document's fingerprint equals the published one; only `limitations` may differ (X13), which the
    fingerprint leaves out."""
    fx = fixture(site)
    document, title = offline_document(fx)
    assert A.counts(document)["models"] == fx["expect"]["models"]
    assert A.fingerprint(document, title) == fx["expect"]["fingerprint"]


@needs_merge
def test_p6_negatives(tmp_path, monkeypatch):
    """P6: a byte-changed prediction.npz is refused; a changed approved sha drops that box; an unknown flag is rejected;
    the dry run completes with subprocess and socket patched to raise."""
    import shutil
    import socket
    from report_runner.store import Store
    art = tmp_path / "art"
    run = art / "runs/droid-copy"
    run.mkdir(parents=True)
    shutil.copyfile(ART / "runs/droid-walmart-190-shot383-250/prediction.npz", run / "prediction.npz")
    store = Store(art)
    spec = StageSpec(name="camera", site="negative", version=1, deps=(), commands=(("true",),), inputs={}, consumes={}, leaves={}, outputs={"prediction": "prediction.npz"},
                     stage_from={}, compute="cpu", gpu=None, timeout_s=1, worst_usd=0., est_usd=0., est_s=0., budget_flags={}, models=(), rules={}, env={})
    store.record(spec, run, "delivered-only", ["delivered"], {}, adopted_from="P6")
    st = (run / "prediction.npz").stat()
    data = bytearray((run / "prediction.npz").read_bytes())
    data[-1] ^= 1
    (run / "prediction.npz").write_bytes(bytes(data))
    os.utime(run / "prediction.npz", ns=(st.st_atime_ns, st.st_mtime_ns))  # same size and mtime: only --verify sees it
    assert Store(art, verify=True).lookup(spec) is None
    (run / "prediction.npz").touch()
    assert Store(art).lookup(spec) is None

    review = json.loads((REPO / "docs/phase2/box-review-303/walmart.json").read_text())
    entity = sorted(review["boxesApproved"])[0]
    review["boxesApproved"][entity] = "0" * 64
    (tmp_path / "walmart.json").write_text(json.dumps(review))
    merge = [sys.executable, str(REPO / "scripts/merge_object_models.py"), "--recgen", str(ART / "runs/walmart-object-models-264"),
             "--sam3d", str(ART / "runs/walmart-object-models-267-sam3d"), "--box", str(ART / "runs/walmart-object-models-300-box"),
             "--box-test", str(ART / "runs/m0-box-test-walmart/box-test.json"), "--review", str(tmp_path / "walmart.json"), "--output", str(tmp_path / "merged")]
    assert subprocess.run(merge, capture_output=True).returncode == 0
    assert entity not in json.loads((tmp_path / "merged/merge.json").read_text())["choice"]

    unknown = subprocess.run([sys.executable, str(REPO / "scripts/run_video_report.py"), "--video", "x.mp4", "--start", "0", "--end", "1", "--site", "x", "--bogus"],
                             capture_output=True, text=True)
    assert unknown.returncode == 2 and "--bogus" in unknown.stderr

    def refuse(*a, **k):
        raise AssertionError("a dry run must not start a process or open a socket")
    fx = fixture("walmart")
    video = video_of(fx)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    import argparse
    from report_runner import store as runner
    args = argparse.Namespace(video=str(video), start=fx["start"], end=fx["end"], site=fx["site"], profile="delivered", review=str(REPO / "docs/phase2/box-review-303"),
                              dry_run=True, publish=False, verify=False)
    assert runner.main(args) == 0


def closure(specs, seeds):
    """Stage names whose key changes when `seeds` change: the seeds and everything downstream of them."""
    out, changed = set(seeds), True
    while changed:
        changed = False
        for s in specs:
            if s.name not in out and any(p in out for p, _ in s.inputs.values()):
                out.add(s.name)
                changed = True
    return out


@needs_merge
@pytest.mark.parametrize("site", A.SITES)
def test_p7_research_misses_are_the_register(site):
    """P7: --profile research --dry-run misses exactly the closure of the site's deviations over the graph."""
    pytest.skip("not provable in the M2 CPU workflow: adopt's research graph stops at lens on all three sites (Walmart and Sam's "
                "Club: the adopted lens rule read the metric scale, which the research lens key does not name; ME340: no MoGe-3 run), "
                "the research source is unverified (D8: MoGe-3 revision), and the rest needs a GPU call (U5); adopt records "
                "research keys only up to there (cuts, census, shots)")


def delivered_ctx(site, fx, store, review=REPO / "docs/phase2/box-review-303", end=None, profile="delivered", extra=None):
    """The delivered graph fed the adopted decision values directly (no decision stage is looked up), so a plan covers every
    stage even when a changed input makes a decision stage miss."""
    values = {**fx["decisions"], **(extra or {})}
    ctx = Ctx(site, video_of(fx), fx["start"], fx["end"] if end is None else end, profile, review, ART, store)
    while True:
        try:
            return stages_graph(ctx)
        except Pending as p:
            ctx.decisions[p.decision] = {"value": values[p.decision], "evidence": {}, "rule": "adopted@1"}


def stages_graph(ctx):
    from report_runner import stages
    return stages.graph(ctx)


@needs_merge
def test_p8_one_changed_input_misses_exactly_its_dependents(tmp_path):
    """Change one input of the adopted Walmart graph: the review file (one approved box sha) or the window's end. The stages
    that are not hits are exactly the changed input's consumers and everything downstream of them; the misses are the
    consumers whose producers are all hits, and the plan prices them."""
    adopted("walmart")
    from report_runner.store import Store
    fx = fixture("walmart")
    store = Store(ART, scope="delivered")
    specs = delivered_ctx("walmart", fx, store)
    rows = {r["stage"]: r for r in store.plan(specs)}
    assert {n for n, r in rows.items() if r["status"] != "hit"} == set(), "the unchanged graph is all hits"

    review = tmp_path / "review"
    review.mkdir()
    for f in (REPO / "docs/phase2/box-review-303").glob("*.json"):
        (review / f.name).write_bytes(f.read_bytes())
    specs = delivered_ctx("walmart", fx, Store(ART, scope="delivered"), review=review)  # control: the same bytes in another folder
    assert {r["stage"] for r in Store(ART, scope="delivered").plan(specs) if r["status"] != "hit"} == set(), "a review enters keys by its bytes, never its path"
    doc = json.loads((review / "walmart.json").read_text())
    doc["boxesApproved"][sorted(doc["boxesApproved"])[0]] = "0" * 64
    (review / "walmart.json").write_text(json.dumps(doc))
    specs = delivered_ctx("walmart", fx, Store(ART, scope="delivered"), review=review)
    readers = {s.name for s in specs if "review" in s.leaves}
    assert readers == {"generator_plan", "generator_plan-box", "merge"}
    rows = {r["stage"]: r for r in Store(ART, scope="delivered").plan(specs)}
    missed = {n for n, r in rows.items() if r["status"] != "hit"}
    assert missed == closure(specs, readers) == readers | {"import"}
    assert {n for n, r in rows.items() if r["status"] == "miss"} == readers and rows["import"]["status"] == "unresolved"
    assert sum(r["usd"] for r in rows.values() if r["status"] != "hit") == 0 and rows["merge"]["s"] > 0, "CPU stages: $0, seconds estimated"

    specs = delivered_ctx("walmart", fx, Store(ART, scope="delivered"), end=fx["end"] - 1)
    rows = {r["stage"]: r for r in Store(ART, scope="delivered").plan(specs)}
    assert {n for n, r in rows.items() if r["status"] == "miss"} == {"source"}, "a new window changes the clip, and so every key after it"
    assert {n for n, r in rows.items() if r["status"] != "hit"} == closure(specs, {"source"}) == {s.name for s in specs}
    assert rows["source"]["usd"] > 0


@needs_merge
def test_p9_commercial_refuses_the_non_commercial_stages():
    """--profile commercial: every stage whose model licence is not verified for commercial use (or not pinned) is refused
    with its reason, before any lookup; the others are misses (no commercial entry exists)."""
    from report_runner import profiles
    from report_runner.store import Store
    fx = fixture("walmart")
    extra = {"static_filter": {"moved": [], "cleared": []}, "lens_gate": fx["decisions"]["lens"], "other_shot-0-383": {"accepted": False},
             "shots": {k: v for k, v in fx["decisions"]["shots"].items() if k != "registered"}}  # M2: every other shot is registered (D3)
    store = Store(ART, scope="commercial", refuse=lambda spec: profiles.refuse("commercial", spec))
    specs = delivered_ctx("walmart", fx, store, profile="commercial", extra=extra)
    rows = {r["stage"]: r for r in store.plan(specs)}
    refused = {n: r["why"] for n, r in rows.items() if r["status"] == "refused"}
    assert {"source", "census", "moge", "camera", "register-0-383"} <= set(refused)
    assert "lingbot" not in rows and "recgen" not in rows and "names" not in rows, "commercial: no LingBot, no RecGen, names blank until Qwen3-VL passes"
    for name, why in refused.items():
        spec = next(s for s in specs if s.name == name)
        assert spec.models and ("commercial use" in why or "pinned" in why), (name, why)
    assert not [n for n, r in rows.items() if r["status"] == "hit"], "no stage is served to commercial from research or delivered entries"
