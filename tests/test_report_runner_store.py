"""Part A of the M2 report runner: the stage store, the budgeted executor and the CLI, on toy graphs in tmp dirs
(no $ART, no GPU, no Modal). Each toy stage is `python -c CODE @new ARGS...`."""
import ast
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import types

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from report_runner import store as st  # noqa: E402
from report_runner.spec import Hit, Pending, StageSpec  # noqa: E402
import run_video_report  # noqa: E402

PY = sys.executable
COPY = "import sys,shutil; shutil.copy(sys.argv[2], sys.argv[1] + '/out.txt')"
UPPER = "import sys; open(sys.argv[1] + '/out.txt', 'w').write(open(sys.argv[2]).read().upper())"
COUNT = "import sys; open(sys.argv[1] + '/out.txt', 'w').write(str(len(open(sys.argv[2]).read())))"
ARGV = "import sys,json; open(sys.argv[1] + '/out.txt', 'w').write(json.dumps(sys.argv[2:]))"
TOUCH = "import sys; open(sys.argv[1] + '/out.txt', 'w').write('x')"


def spec(name, code, *args, **kw):
    fields = dict(name=name, site="toy", version=1, deps=(), commands=((PY, "-c", code, "@new", *args),), inputs={}, consumes={},
                  leaves={}, outputs={"out": "out.txt"}, stage_from={}, compute="cpu", gpu=None, timeout_s=60, worst_usd=0.,
                  est_usd=0., est_s=1., budget_flags={}, models=(), rules={}, env={})
    fields.update(kw)
    return StageSpec(**fields)


def toy(tmp, version=1):
    src = tmp / "src.txt"
    if not src.exists():
        src.write_text("hello")
    return [spec("A", COPY, str(src), leaves={"src": src}, version=version),
            spec("B", UPPER, "@A:out", inputs={"a": ("A", ("out",))}),
            spec("C", COUNT, "@B:out", inputs={"b": ("B", ("out",))})]


@pytest.fixture
def art(tmp_path):
    return tmp_path / "art"


def entries(art):
    return [json.loads(line) for line in (art / "runs/report-runner/keys.jsonl").read_text().splitlines()]


def test_runs_then_serves_hits_with_stable_keys(tmp_path, art):
    first = st.Store(art).execute(toy(tmp_path), st.Ledger())
    assert all(isinstance(h, Hit) and h.verification == "ran" for h in first.values())
    assert (first["C"].dir / "out.txt").read_text() == "5"
    lock = json.loads((first["C"].dir / "lock.json").read_text())
    assert len(lock["git"]["commit"]) == 40 and lock["wall_s"] >= 0 and first["C"].dir.name == f"toy-C-{lock['key'][:10]}"
    again = st.Store(art)
    second = again.execute(toy(tmp_path), st.Ledger())
    assert {n: h.dir for n, h in second.items()} == {n: h.dir for n, h in first.items()}
    assert [e["key"] for e in entries(art)] == [again.key(s) for s in toy(tmp_path)]  # 3 lines: hits append nothing
    assert all(Path(h.dir).name.startswith(f"toy-{n}-") for n, h in first.items())


def test_key_ignores_output_budgets_run_ids_and_note_text(art):
    s = st.Store(art)

    def key(*args, **kw):
        return s.key(spec("G", ARGV, *args, **kw))
    base = key("--output", "/x/1", "--max-usd", "4", "--entities", "object-1=a pallet", "--run-id", "r1")
    assert base == key("--output", "/y/2", "--max-usd", "9", "--invoke", "--entities", "object-1=a blue pallet", "--run-id", "r2",
                       "--reuse-build-from", "/z", worst_usd=5., est_usd=2.)
    assert base != key("--output", "/x/1", "--entities", "object-2=a pallet")  # the ids stay
    assert base != key("--output", "/x/1", "--max-usd", "4", "--entities", "object-1=a pallet", version=2)
    assert base != key("--output", "/x/1", "--entities", "object-1=a pallet", models=(("depth", "da3", "unpinned", None),))


def test_a_leaf_enters_the_key_by_its_bytes_never_its_path(tmp_path, art):
    """A review file (or any leaf) reaches a key as its bytes, whether a tool takes it as '--review PATH' or a decision as
    'review=PATH': a byte-identical copy elsewhere keys the same, one changed byte does not."""
    s = st.Store(art)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    review, copy = tmp_path / "a/walmart.json", tmp_path / "b/walmart.json"
    review.write_text('{"boxesApproved": {}}')
    copy.write_text('{"boxesApproved": {}}')

    def keys(path):
        leaves = {"review": path}
        return (s.key(spec("plan", ARGV, "sam3d=x", f"review={path}", leaves=leaves)), s.key(spec("merge", ARGV, "--review", str(path), leaves=leaves)))
    assert keys(review) == keys(copy)
    copy.write_text('{"boxesApproved": {"object-1": "0"}}')
    assert all(a != b for a, b in zip(keys(review), keys(copy)))


def test_early_cutoff_keeps_downstream_hits(tmp_path, art):
    first = st.Store(art).execute(toy(tmp_path), st.Ledger())
    rebuilt = st.Store(art).execute(toy(tmp_path, version=2), st.Ledger())  # A reruns, same bytes
    assert rebuilt["A"].dir != first["A"].dir and rebuilt["A"].verification == "ran"
    assert rebuilt["B"].dir == first["B"].dir and rebuilt["C"].dir == first["C"].dir


def test_edited_output_invalidates_and_verify_catches_same_size_same_mtime(tmp_path, art):
    s = st.Store(art)
    first = s.execute(toy(tmp_path), st.Ledger())
    (first["B"].dir / "out.txt").write_text("HELLO!")
    assert s.lookup(toy(tmp_path)[1]) is None
    rerun = s.execute(toy(tmp_path), st.Ledger())
    assert rerun["B"].dir != first["B"].dir and rerun["C"].dir == first["C"].dir  # B rebuilt HELLO: C cut off
    out = first["C"].dir / "out.txt"
    before = os.stat(out)
    out.write_text("6")
    os.utime(out, ns=(before.st_atime_ns, before.st_mtime_ns))
    c = toy(tmp_path)[2]
    assert st.Store(art).execute(toy(tmp_path), st.Ledger())["C"].dir == first["C"].dir  # size and mtime still match
    plain = st.Store(art)
    plain.execute(toy(tmp_path)[:2], st.Ledger())
    assert plain.lookup(c) is not None and plain.lookup(c, verify=True) is None
    assert st.Store(art, verify=True).execute(toy(tmp_path), st.Ledger())["C"].dir != first["C"].dir  # --verify rebuilds it


def test_budget_refuses_over_cap_and_caps_generators(tmp_path, art):
    marker = tmp_path / "paid-ran"
    paid = spec("P", f"open({str(marker)!r}, 'w')", paid=True, worst_usd=5.)
    graph = [paid, spec("Q", UPPER, "@P:out", inputs={"p": ("P", ("out",))}), spec("I", TOUCH)]
    result = st.Store(art).execute(graph, st.Ledger(1.))
    assert result["P"] == "refused" and result["Q"] == "blocked" and isinstance(result["I"], Hit) and not marker.exists()

    ledger = st.Ledger(3.)
    gen = spec("G", ARGV, paid=True, worst_usd=5., budget_flags={"--max-usd": "usd"})
    splat = spec("S", ARGV, "--max-minutes", "5", paid=True, worst_usd=10., budget_flags={"--max-minutes": ("minutes", .001)})
    result = st.Store(art).execute([gen], ledger)
    assert json.loads((result["G"].dir / "out.txt").read_text()) == ["--max-usd", "3.0"] and ledger.spent == 3.
    result = st.Store(art).execute([splat], st.Ledger(20.))  # 10 USD at .001/s is 166 min: the 5 in the argv wins
    assert json.loads((result["S"].dir / "out.txt").read_text()) == ["--max-minutes", "5.0"]


def test_no_budget_stops_before_the_first_paid_miss(tmp_path, art):
    marker = tmp_path / "paid-ran"
    graph = [spec("P", f"open({str(marker)!r}, 'w')", paid=True, worst_usd=.01), spec("Z", TOUCH),
             spec("Y", UPPER, "@Z:out", inputs={"z": ("Z", ("out",))})]
    result = st.Store(art).execute(graph, st.Ledger(None))
    assert result["P"] == "refused" and result["Y"] == "blocked" and not marker.exists()
    rows = st.Store(art).plan(graph, st.Ledger(None))
    assert {r["stage"]: r["status"] for r in rows}["P"] == "refused"


def test_dirty_or_untracked_dependency_is_refused(tmp_path, art):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "dep.py").write_text("x = 1\n")
    git = ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "dep.py"], check=True)
    subprocess.run([*git, "commit", "-qm", "dep"], check=True)
    s = st.Store(art, repo=repo)
    assert isinstance(s.execute([spec("D", TOUCH, deps=("dep.py",))], st.Ledger())["D"], Hit)
    (repo / "dep.py").write_text("x = 2\n")
    assert s.execute([spec("D", TOUCH, deps=("dep.py",), version=2)], st.Ledger())["D"] == "refused"
    (repo / "new.py").write_text("")
    assert s.execute([spec("E", TOUCH, deps=("new.py",))], st.Ledger())["E"] == "refused"


def test_staging_links_are_relative_copies_are_real_and_dangling_links_fail(tmp_path, art):
    make = "import os,sys; d=sys.argv[1]; open(d+'/out.txt','w').write('a'); os.mkdir(d+'/tree'); os.symlink('../out.txt', d+'/tree/x')"
    edit = "import sys; d=sys.argv[1]; open(d+'/out.txt','w').write(open(d+'/in/a.txt').read()); open(d+'/c.txt','a').write('!')"
    a = spec("A", make, outputs={"out": "out.txt", "tree": "tree"})
    b = spec("B", edit, inputs={"a": ("A", ("out",))}, stage_from={"link": {"in/a.txt": "@A:out"}, "copy": {"c.txt": "@A:out"}})
    c = spec("C", TOUCH, inputs={"a": ("A", ("tree",))}, stage_from={"copy": {"tree": "@A:tree"}})  # the 305/scene case
    result = st.Store(art).execute([a, b, c], st.Ledger())
    link = result["B"].dir / "in/a.txt"
    assert not os.path.isabs(os.readlink(link)) and link.resolve() == (result["A"].dir / "out.txt").resolve()
    assert (result["B"].dir / "c.txt").read_text() == "a!" and (result["A"].dir / "out.txt").read_text() == "a"
    assert result["C"] == "failed"
    assert [p.name for p in (art / "runs").glob("toy-C-*")][0].split("-failed-")[1]


def test_failure_renames_dir_blocks_dependents_and_optional_inputs_drop(tmp_path, art):
    fail = spec("F", "import sys; sys.exit(3)")
    graph = [fail, spec("G", UPPER, "@F:out", inputs={"f": ("F", ("out",))}), spec("H", TOUCH),
             spec("I", ARGV, "--carve", "--opt", "@F:out", "--keep", "1", inputs={"f?": ("F", ("out",))})]
    result = st.Store(art).execute(graph, st.Ledger())
    assert result["F"] == "failed" and result["G"] == "blocked" and isinstance(result["H"], Hit)
    assert json.loads((result["I"].dir / "out.txt").read_text()) == ["--carve", "--keep", "1"]
    failed = list((art / "runs").glob("toy-F-*"))
    assert len(failed) == 1 and "-failed-" in failed[0].name and "exit 3" in (failed[0] / "runner.log").read_text()
    assert {e["stage"] for e in entries(art)} == {"H", "I"}


def test_timeout_kills_the_subprocess(art):
    result = st.Store(art).execute([spec("T", "import time; time.sleep(60)", timeout_s=-299)], st.Ledger())
    assert result["T"] == "failed"


def test_four_workers_run_at_once(tmp_path, art):
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    code = ("import os,sys,time; b=sys.argv[2]; open(os.path.join(b, sys.argv[3]),'w').write(str(time.time())); t=time.time()\n"
            "while len(os.listdir(b)) < 4 and time.time() - t < 20: time.sleep(.02)\n"
            "time.sleep(.3); open(sys.argv[1]+'/out.txt','w').write(f'{t} {time.time()} {len(os.listdir(b))}')")
    graph = [spec(f"W{i}", code, str(barrier), f"w{i}") for i in range(6)]
    result = st.Store(art).execute(graph, st.Ledger(), workers=4)
    spans = [tuple(map(float, (h.dir / "out.txt").read_text().split()[:2])) for h in result.values()]
    overlap = max(sum(a <= t < b for a, b in spans) for t, _ in spans)
    assert overlap == 4


def test_redaction_on_stdout_and_log(art, capsys):
    code = "print('keep me'); print('the capability is c'); print('Token: t'); print('a SECRET')"
    result = st.Store(art).execute([spec("R", code + "; open(__import__('sys').argv[1]+'/out.txt','w').write('x')")], st.Ledger())
    out, log = capsys.readouterr().out, (result["R"].dir / "runner.log").read_text()
    for text in (out, log):
        assert "keep me" in text and not st.REDACT.search(text)


def test_record_scope_lock_path_disjoint_roles_and_latest(tmp_path, art):
    d = art / "runs/adopted-281"
    d.mkdir(parents=True)
    (d / "infer.json").write_text("{}")
    (d / "mesh.ply").write_text("m")
    infer = spec("R16", TOUCH, outputs={"infer": "infer.json"})
    fuse = spec("R17", TOUCH, outputs={"mesh": "mesh.ply"}, inputs={"depth": ("R16", ())})
    s = st.Store(art)
    k = s.record(infer, d, "delivered-only", ["delivered"], {"stage": "R16"}, adopted_from="runs/adopted-281")
    assert json.loads((art / f"runs/report-runner/adopted/locks/{k}.json").read_text())["key"] == k
    s.record(fuse, d, "delivered-only", ["delivered"], {}, adopted_from="runs/adopted-281")  # disjoint roles share the dir
    # one delivered run may stand for two stages (ME340 171 is census and camera): adopted records may share its files
    s.record(spec("X", TOUCH, outputs={"mesh": "mesh.ply"}), d, "content", ["delivered"], {}, adopted_from="x")
    assert st.Store(art, scope="delivered").lookup(spec("X", TOUCH, outputs={"mesh": "mesh.ply"})).outputs["mesh"] == d / "mesh.ply"
    assert st.Store(art, scope="research").lookup(infer) is None
    assert st.Store(art, scope="delivered").lookup(infer).dir == d
    digest = entries(art)[1]["inputs"]["depth"]
    assert s.latest("R17", {"depth": digest}).dir == d and s.latest("R17", {"depth": "0" * 64}) is None


def test_a_whole_directory_output_is_signed_after_its_lock(tmp_path, art):
    """An output role '.' is the stage's whole directory, lock.json included: the lock is written before the signatures are
    taken, so the run is a hit afterwards, by size and mtime and by full sha256."""
    whole = [spec("W", TOUCH, outputs={"all": "."})]
    ran = st.Store(art).execute(whole, st.Ledger())["W"]
    assert (ran.dir / "lock.json").is_file() and ran.verification == "ran"
    assert st.Store(art).lookup(whole[0]).dir == ran.dir and st.Store(art).lookup(whole[0], verify=True).dir == ran.dir


def test_decision_value_and_droid_frame_digests(art):
    s = st.Store(art)
    d = art / "runs/decide"
    d.mkdir(parents=True)
    dec = spec("shots", TOUCH, outputs={"decision": "d.json"})
    use = spec("use", TOUCH, inputs={"d": ("shots", ())}, consumes={"d": "decision-value"})
    keys = []
    for doc in ({"value": [0, 9], "evidence": "a"}, {"value": [0, 9], "evidence": "a longer one"}, {"value": [1, 9], "evidence": "a"}):
        (d / "d.json").write_text(json.dumps(doc))
        s.record(dec, d, "ran", ["research"], {})
        keys.append(s.key(use))
    assert keys[0] == keys[1] != keys[2]

    clip = art / "data/clips/toy"
    (clip / "rgb").mkdir(parents=True)
    (clip / "rgb.txt").write_text("# t f\n" + "".join(f"{i}.0 rgb/{i}.png\n" for i in range(4)))
    make = spec("R01", TOUCH, outputs={"clip": "clip.json", "rgb": "rgb"})
    cam = spec("cam", TOUCH, "--frames", "1:3", inputs={"clip": ("R01", ())}, consumes={"clip": "droid-frames"})

    def key_with(frames, k=500.):
        for i, text in enumerate(frames):
            (clip / f"rgb/{i}.png").write_text(text)
        (clip / "clip.json").write_text(json.dumps({"K": [k, k, 319.5, 239.5], "D": [0, 0, 0, 0]}))
        s.record(make, clip, "content", ["research"], {})
        return s.key(cam)
    base = key_with(["a", "b", "c", "d"])
    assert key_with(["aa", "b", "c", "dd"]) == base  # frames outside [1, 3)
    assert key_with(["a", "bb", "c", "d"]) != base
    assert key_with(["a", "b", "c", "d"], k=610.) != base


def fake_modules(monkeypatch, art, graph):
    monkeypatch.setattr(st, "art_root", lambda: art)
    monkeypatch.setitem(sys.modules, "report_runner.stages", types.SimpleNamespace(graph=graph))
    profiles = {n: types.SimpleNamespace(cache_only=n == "delivered") for n in ("research", "commercial", "delivered")}
    monkeypatch.setitem(sys.modules, "report_runner.profiles", types.SimpleNamespace(PROFILES=profiles))
    monkeypatch.delenv("PANOPTES_PAID_BUDGET_USD", raising=False)


def decided_graph(tmp_path):
    decide = "import sys,json; open(sys.argv[1]+'/decision.json','w').write(json.dumps({'value': 7, 'evidence': 'toy', 'rule': 'toy@1'}))"

    def graph(ctx):
        a, b, c = toy(tmp_path)
        dec = spec("D", decide, inputs={"a": ("A", ())}, outputs={"decision": "decision.json"})
        if "D" not in ctx.decisions:
            raise Pending("D", [a, dec])
        c = spec("C", COUNT, "@B:out", str(ctx.decisions["D"]["value"]), inputs={"b": ("B", ("out",))})
        return [a, dec, b, c]
    return graph


CLI = ["--video", "v.mp4", "--start", "1", "--end", "2", "--site", "toy"]


def test_main_resolves_pending_decisions_then_runs(tmp_path, art, monkeypatch):
    fake_modules(monkeypatch, art, decided_graph(tmp_path))
    assert run_video_report.main(CLI) == 0
    runs = {p.name.split("-")[1]: p for p in (art / "runs").glob("toy-*")}
    assert set(runs) == {"A", "B", "C", "D"} and (runs["C"] / "out.txt").read_text() == "5"
    assert run_video_report.main(CLI) == 0 and len(list((art / "runs").glob("toy-*"))) == 4  # all hits now
    assert run_video_report.main([*CLI, "--profile", "delivered"]) == 2  # research runs are not delivered entries


def test_dry_run_is_inert(tmp_path, art, monkeypatch, capsys):
    fake_modules(monkeypatch, art, decided_graph(tmp_path))
    st.Store(art).execute(toy(tmp_path)[:1], st.Ledger())  # A cached; the decision D is not

    def refuse(*a, **k):
        raise AssertionError("dry run must not start processes or open sockets")
    before = sorted((str(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*"))
    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket, "socket", refuse)
    assert run_video_report.main([*CLI, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "hit" in out and "miss" in out and "stops here: decision D" in out
    assert run_video_report.main([*CLI, "--profile", "delivered"]) == 2  # cache-only refuses the missing decision
    assert sorted((str(p), p.stat().st_mtime_ns) for p in tmp_path.rglob("*")) == before


def test_prices_match_the_code():
    tree = ast.parse((REPO / "modal_apps/splat_train.py").read_text())
    usd = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "USD_PER_S")
    from ehs_spatial.video import MODAL_L4_USD_PER_S
    assert st.PRICES == {**usd, "L4_1cpu_8gib": MODAL_L4_USD_PER_S}


def test_cli_help_and_unknown_flags():
    script = str(REPO / "scripts/run_video_report.py")
    ok = subprocess.run([PY, script, "--help"], capture_output=True, text=True)
    assert ok.returncode == 0 and "--dry-run" in ok.stdout
    for bad in (["--bogus"], ["--dry"]):
        run = subprocess.run([PY, script, *CLI, *bad], capture_output=True, text=True)
        assert run.returncode == 2 and "unrecognized" in run.stderr


def test_profile_refusals_are_planned_and_never_run(tmp_path, art, capsys):
    """A profile's refusal (profiles.refuse) shows in the plan with its reason and stops the stage before any lookup or run."""
    marker = tmp_path / "ran"
    a, b, c = toy(tmp_path)
    b = spec("B", f"import sys; open({str(marker)!r}, 'w').write('x')", "@A:out", inputs={"a": ("A", ("out",))})
    no_b = lambda s: "licence not verified" if s.name == "B" else None
    rows = {r["stage"]: r for r in st.Store(art, refuse=no_b).plan([a, b, c])}
    assert rows["B"]["status"] == "refused" and rows["B"]["why"] == "licence not verified" and rows["C"]["status"] == "unresolved"
    st.print_plan(list(rows.values()))
    assert "refused B: licence not verified" in capsys.readouterr().out
    result = st.Store(art, scope="commercial", refuse=no_b).execute([a, b, c], st.Ledger())
    assert isinstance(result["A"], Hit) and result["B"] == "refused" and result["C"] == "blocked" and not marker.exists()


def test_adopted_runs_may_hold_dangling_links_and_absent_outputs(art):
    """ME340 305/scene/mono dangles and 217's trained splats.splat is gone: hashing skips the dangling link, an adopted role
    recorded as absent is never served, and resolve() substitutes a served stage's command from the hits."""
    d = art / "runs/adopted"
    (d / "scene").mkdir(parents=True)
    (d / "scene/scene.json").write_text("{}")
    (d / "scene/mono").symlink_to("../nowhere")
    s = st.Store(art)
    first = spec("A", TOUCH, outputs={"scene": "scene", "splats": None})
    s.record(first, d, "delivered-only", ["delivered"], {}, adopted_from="S39")
    hit = st.Store(art, scope="delivered").lookup(first)
    assert hit.outputs == {"scene": d / "scene"}
    use = spec("B", TOUCH, "@A:scene", inputs={"a": ("A", ("scene",))})
    s.record(dataclass_replace(use, outputs={"out": "scene/scene.json"}), d, "delivered-only", ["delivered"], {}, adopted_from="x")
    assert s.resolve(use) == [[PY, "-c", TOUCH, str(d), str(d / "scene")]]


def dataclass_replace(spec_, **kw):
    import dataclasses
    return dataclasses.replace(spec_, **kw)
