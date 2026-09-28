"""M2 report runner, Part B: the stage graph, DEFAULTS, normalize(), versions.json and the script fixes the graph needs.

CPU only, no Modal call: the $ART cases read cached runs read-only, write only to tmp_path, and skip without $ART.
"""
import ast
import contextlib
import hashlib
import io
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "modal_apps")]
from report_runner import stages  # noqa: E402
from report_runner.spec import Ctx, Pending  # noqa: E402

ART = stages.art()
RUNS = ART / "runs"
needs_art = pytest.mark.skipif(not RUNS.is_dir(), reason=f"no cached runs at {ART}")
REVIEW = REPO / "docs/phase2/box-review-303"


# ---------------------------------------------------------------- argparse defaults, read from source
def _constants(tree):
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
                elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                    out.update({t.id: v for t, v in zip(target.elts, node.value.elts) if isinstance(t, ast.Name)})
    return out


def _eval(node, consts):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, consts) for e in node.elts)
    if isinstance(node, ast.List):
        return [_eval(e, consts) for e in node.elts]
    if isinstance(node, ast.Name):
        return _eval(consts[node.id], consts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand, consts)
    if isinstance(node, ast.BinOp):
        a, b = _eval(node.left, consts), _eval(node.right, consts)
        return {ast.Div: lambda: a / b, ast.Mult: lambda: a * b, ast.Add: lambda: a + b, ast.Sub: lambda: a - b}[type(node.op)]()
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path":
        return Path(*(_eval(a, consts) for a in node.args))
    raise ValueError(f"cannot evaluate {ast.dump(node)[:120]}")


def script_defaults(path):
    """{--flag: default} of every add_argument option of a script (loops over constant names included), never importing it."""
    tree = ast.parse(Path(path).read_text())
    consts, found = _constants(tree), {}

    def add(call, names):
        kw = {k.arg: k.value for k in call.keywords}
        for name in (n for n in names if n.startswith("--")):
            action = _eval(kw["action"], consts) if "action" in kw else None
            found[name] = False if action == "store_true" else _eval(kw["default"], consts) if "default" in kw else None

    def walk(node, loop=None):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and isinstance(node.iter, (ast.Tuple, ast.List)) \
                and all(isinstance(e, ast.Constant) for e in node.iter.elts):
            for child in node.body:
                walk(child, (node.target.id, [e.value for e in node.iter.elts]))
            return
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant):
                add(node, [a.value for a in node.args if isinstance(a, ast.Constant)])
            elif loop and isinstance(first, ast.BinOp) and isinstance(first.left, ast.Constant) and getattr(first.right, "id", None) == loop[0]:
                add(node, [first.left.value + v for v in loop[1]])
        for child in ast.iter_child_nodes(node):
            walk(child, loop)
    walk(tree)
    return found


def _plain(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and value.startswith("$ART"):
        return value.replace("$ART", str(ART))
    return list(value) if isinstance(value, tuple) else value


@pytest.mark.parametrize("script", sorted(stages.DEFAULTS))
def test_defaults_are_the_scripts_own(script):
    found = {k: _plain(v) for k, v in script_defaults(REPO / script).items()}
    for flag, default in stages.PENDING_FLAGS.get(script, {}).items():  # Part C's flag: absent here, or present with this default
        assert found.pop(flag, default) == default
    assert found == {k: _plain(v) for k, v in stages.DEFAULTS[script].items()}


def test_every_graph_script_has_defaults():
    assert {s for s in stages.SCRIPT.values() if s} <= set(stages.DEFAULTS)


# ---------------------------------------------------------------- the graph under the section 5 decision values
NONE = {"bands": [], "texture_rows": None, "fill_rows": None, "lingbot_rows": "0:0", "splat_captions": [0, 0, 0, 0], "objects_no_captions": True, "refuse": []}
ME340_BAND = {"bands": [[648, 704]], "texture_rows": "648:704", "fill_rows": "648:704", "lingbot_rows": "646:706",
              "splat_captions": [648, 704, 160, 1120], "objects_no_captions": False, "refuse": []}
PLAN = {"recgen_entities": ["object-020", "object-030"], "box_excludes": {"object-003": "accepted by SAM 3D"}, "excluded_all": {"object-156": "on the moving cart"}}
SITES = {  # clip window, and the decision values the rules give on the delivered runs (design section 5)
    "me340": (165, 195, {"shots": {"primary": [226, 899], "others": [[0, 14], [14, 226]], "mapped": True}, "overlay": ME340_BAND,
                         "lens": {"keep": True, "fov_deg": 80.1}, "sam2_frames": sorted(set(range(0, 899, 3)) | {14, 226}),
                         "floor_frames": list(range(0, 899, 75)), "lingbot_stride": 2, "voxel": 0.01399, "lingbot_conf": 1.06,
                         "track_windows": {"shot": [[226, 526], [486, 786], [746, 899]], "stitch": True, "others": [[[0, 14]], [[14, 226]]]},
                         "static_filter": {"moved": [], "cleared": ["object-032"]}, "splat_pick": "negligible_1",
                         "other_shot-0-14": {"accepted": False}, "other_shot-14-226": {"accepted": True}, "generator_plan": PLAN,
                         "generator_plan-box": PLAN, "dense_gate": {"use": "raw", "dir": None}, "inferred_floor": True}),
    "samsclub-a2": (337, 367, {"shots": {"primary": [0, 420], "others": [[420, 750]], "mapped": True}, "overlay": NONE,
                               "lens": {"keep": False, "fov_deg": 55.3194223319173}, "sam2_frames": list(range(0, 750, 3)),
                               "floor_frames": [0, 66, 135, 201, 270, 336, 405], "lingbot_stride": 1, "voxel": 0.00567, "lingbot_conf": 1.74,
                               "track_windows": {"shot": [[0, 300], [260, 420]], "stitch": True, "others": [[[420, 750]]]},
                               "static_filter": {"moved": ["object-900"], "cleared": []}, "splat_pick": "negligible_1",
                               "other_shot-420-750": {"accepted": False}, "generator_plan": PLAN, "generator_plan-box": PLAN,
                               "dense_gate": {"use": "icp", "dir": "icp"}, "inferred_floor": False}),
    "walmart": (190, 220, {"shots": {"primary": [383, 750], "others": [[0, 383]], "mapped": True}, "overlay": NONE,
                           "lens": {"keep": True, "fov_deg": 55.96}, "sam2_frames": list(range(0, 750, 3)), "floor_frames": [405, 471, 540, 606, 675, 747],
                           "lingbot_stride": 1, "voxel": 0.011858, "lingbot_conf": 1.01,
                           "track_windows": {"shot": [[383, 750]], "stitch": False, "others": [[[0, 383]]]},
                           "static_filter": {"moved": [], "cleared": []}, "splat_pick": "negligible_1", "other_shot-0-383": {"accepted": False},
                           "generator_plan": PLAN, "generator_plan-box": PLAN, "dense_gate": {"use": "icp", "dir": "icp"}, "inferred_floor": False}),
}
# the shape of profiles.Profile (Part C): models {role: (hf_id, revision, weights_sha256, licence, commercial)}
GIANT = ("depth-anything/DA3-GIANT-1.1", "72ee9f89", None, "CC BY-NC 4.0", False)
RESEARCH = types.SimpleNamespace(models={"depth": GIANT, "register": GIANT, "camera": ("princeton-vl/DROID-SLAM", "2dfd39f0", None, "verify", None)},
                                 generators=("sam3d", "recgen", "box"), dense_map=True, namer="gemini", caps={"splat_minutes": 58, "generator_usd": 10.},
                                 cache_only=False, rules={"voxel": "voxel@1"})
COMMERCIAL = types.SimpleNamespace(models={"depth": ("depth-anything/DA3-BASE", "f4a6c9b3", None, "Apache-2.0", True),
                                           "register": ("depth-anything/DA3-BASE#camera", "f4a6c9b3", None, "verify", None)},
                                   generators=("sam3d", "box"), dense_map=False, namer="qwen3vl", caps={"splat_minutes": 58, "generator_usd": 10., "sam3d_usd": 5},
                                   cache_only=False, rules={})


def ctx(site, profile=RESEARCH, decisions=None, video=Path("/x/video.mp4"), review=REVIEW):
    start, end, values = SITES[site]
    values = values if decisions is None else decisions
    return Ctx(site, video, start, end, profile, review, ART, None, {k: {"value": v, "evidence": {}, "rule": k} for k, v in values.items()})


def build(site, **kw):
    return {s.name: s for s in stages.graph(ctx(site, **kw))}


def words(spec, i=0):
    return list(spec.commands[i])


def flag(spec, name, i=0):
    """The values after --name in command i (None: the flag is absent)."""
    cmd = words(spec, i)
    if name not in cmd:
        return None
    rest = cmd[cmd.index(name) + 1:]
    return rest[:next((n for n, w in enumerate(rest) if w.startswith("--")), len(rest))]


def check_structure(specs):
    """What store._check and store._key need: unique names off the reserved words, producers built first and declared,
    every token naming a declared output, known digest modes; the import last."""
    names = [s.name for s in specs]
    assert len(set(names)) == len(names) and not set(names) & stages.RESERVED
    by = {}
    for s in specs:
        producers = {p for p, _ in s.inputs.values()}
        assert producers <= set(by), (s.name, producers - set(by))
        texts = [t for c in s.commands for t in c] + [r for refs in s.stage_from.values() for r in refs.values()] + [s.cwd or ""]
        for text in texts:
            for producer, role in stages.TOKEN.findall(text):
                if producer in stages.RESERVED and not role:
                    continue
                assert producer in producers, (s.name, text)
                assert not role or role in by[producer].outputs, (s.name, text)
        for producer, roles in s.inputs.values():
            assert set(roles) <= set(by[producer].outputs), (s.name, producer, roles)
        assert set(s.consumes.values()) <= {"outputs", "droid-frames", "lingbot-source", "decision-value"} and set(s.consumes) <= set(s.inputs)
        for cmd in s.commands:
            if cmd[0] == stages.PY:
                stages.normalize_command(cmd, lambda p: None)  # every flag is one the script declares
        by[s.name] = s
    assert names[-1] == "import"
    return by


@pytest.mark.parametrize("site", sorted(SITES))
def test_graph_resolves_for_each_site(site):
    by = check_structure(stages.graph(ctx(site)))
    assert {"source", "cuts", "census", "camera", "depth", "fuse", "object_map", "names", "static_filter", "sam3d", "recgen", "box", "merge",
            "splat", "splat_final", "lingbot", "lingbot_build", "dense_gate", "floor_infer", "events", "outlines"} <= set(by)
    assert all(by[n].paid for n in ("census", "camera", "depth", "sam3d", "splat", "lingbot", "lingbot_build", "names")) and not by["fuse"].paid
    assert not by["box"].paid and by["lingbot_build"].gpu is None and by["lingbot_build"].worst_usd > 0 and by["depth"].gpu == "A100-80GB"
    assert "PYTHONPATH" in by["voxel"].env and "PYTHONPATH" not in by["fuse"].env
    assert by["voxel"].rules == {"voxel": f"voxel@1#{stages.rule_code('voxel')[1][:12]}"} and by["fuse"].rules == {}, "a rule's version and code"
    assert "scripts/lingbot_icp_refine.py" in by["dense_gate"].deps and by["voxel"].deps == (stages.DECIDE,), "the files a rule runs or imports"
    assert by["camera"].consumes == {("source" if site != "samsclub-a2" else "lens_clip"): "droid-frames"}
    assert by["lingbot"].consumes == {"source": "lingbot-source"} and by["lingbot"].inputs["source"] == ("source", ("source_full",))
    assert by["import"].consumes["lens_gate"] == "decision-value" and by["import"].inputs["lens_gate"] == ("lens_gate", ())
    assert "static_filter" not in by["outlines"].consumes and by["outlines"].inputs["static_filter"] == ("static_filter", ()), "its folder is read: bytes count"
    assert by["shots"].consumes == {} and by["track_windows"].consumes == {"shots": "decision-value"}
    assert flag(by["import"], "--request-suffix") == ["@key"] and by["import"].cwd == "$ART" and flag(by["import"], "--lens") == ["@lens_gate:decision"]


def test_me340_templates():
    by = build("me340")
    assert flag(by["camera"], "--frames") == ["226:899"] and flag(by["camera"], "--clip") == ["@source:out"] and "lens_clip" not in by
    assert flag(by["depth"], "--exclude-frames") == ["0:14", "14:226"] and flag(by["metric"], "--exclude-frames") == ["0:14", "14:226"]
    assert flag(by["fuse"], "--voxel-length-native") == ["0.01399"] and flag(by["sam3d"], "--voxel-native") == ["0.01399"]
    assert flag(by["splat"], "--skip") == ["0-13", "14-225"] and flag(by["splat"], "--captions") == ["648", "704", "160", "1120"]
    assert flag(by["texture"], "--overlay-rows") == ["648:704"] and flag(by["lingbot_build"], "--overlay-rows") == ["646:706"]
    assert flag(by["lingbot"], "--stride") == ["2"] and flag(by["lingbot"], "--frames") == ["226:899"] and flag(by["lingbot_build"], "--conf") == ["1.06"]
    assert flag(by["sam3d"], "--skip-frames") == ["0-13", "14-225"] and flag(by["sam3d"], "--no-captions") is None, "a caption band keeps the caption test on"
    assert [n for n in by if n.startswith(("tracks-", "register-", "otracks-", "movers-"))] == \
        ["tracks-226-526", "tracks-486-786", "tracks-746-899", "register-0-14", "register-14-226", "otracks-14-226", "movers-14-226"]
    assert flag(by["otracks-14-226"], "--text-only") == [] and flag(by["otracks-14-226"], "--motion") is None
    assert flag(by["movers-14-226"], "--merge") is None and flag(by["movers-14-226"], "--person-tracks") is None, "no cross-shot join (D3b)"
    assert flag(by["import"], "--dynamic-scene") == ["@movers-14-226:scene"] and flag(by["import"], "--exclude-frames") == ["0:14", "14:226"]
    assert flag(by["import"], "--dense-points") == ["@lingbot_build:out"] and flag(by["import"], "--inferred-floor") == ["@floor_infer:out"]
    assert flag(by["import"], "--title") == ["me340 02:45–03:15 (imported, not accepted)"]
    assert flag(by["events"], "--marks") == ["@analysis:analysis"] and flag(by["outlines"], "--mesh") == ["@fill:shell"]


def test_samsclub_templates():
    by = build("samsclub-a2")
    assert flag(by["lens_clip"], "--derive") == ["@source:out"] and flag(by["lens_clip"], "--frames") == ["0:420"]
    assert flag(by["lens_clip"], "--fov-deg") == ["55.3194223319173"] and flag(by["lens_clip"], "--output") == ["@clip"]
    assert flag(by["camera"], "--clip") == ["@lens_clip:out"] and flag(by["camera"], "--frames") is None, "the derived clip is the shot"
    assert flag(by["depth"], "--exclude-frames") is None and flag(by["splat"], "--skip") is None and flag(by["sam3d"], "--skip-frames") is None
    assert flag(by["import"], "--exclude-frames") == ["420:750"] and flag(by["import"], "--video") == ["@source:source_rgb"]
    assert flag(by["register-420-750"], "--clip") == ["@source:out"] and not [n for n in by if n.startswith("movers")]
    assert flag(by["sam3d"], "--no-captions") == [] and flag(by["lingbot_diagnose"], "--overlay-rows") == ["0:0"] and flag(by["texture"], "--overlay-rows") is None
    assert flag(by["splat"], "--captions") == ["0", "0", "0", "0"] and flag(by["fuse"], "--video") == ["@lens_clip:source_rgb"]
    assert flag(by["import"], "--dense-points") == ["@dense_gate/icp"] and flag(by["import"], "--inferred-floor") is None
    assert flag(by["floor_infer"], "--dense") == ["@dense_gate/icp"] and flag(by["stitch"], "--tracks") == ["@tracks-0-300:out", "@tracks-260-420:out"]


def test_walmart_templates():
    by = build("walmart")
    assert flag(by["camera"], "--frames") == ["383:750"] and flag(by["motion"], "--exclude-frames") == ["0:383"]
    assert flag(by["sam3d"], "--skip-frames") == ["0-382"] and flag(by["sam3d"], "--voxel-native") == ["0.011858"] and flag(by["sam3d"], "--all") == []
    assert flag(by["splat"], "--skip") == ["0-382"] and flag(by["floor_infer"], "--skip-frames") == ["0-382"] and "stitch" not in by
    assert flag(by["recgen"], "--entities") == ["object-020", "object-030"] and flag(by["recgen"], "--count") == ["2"]
    assert flag(by["recgen"], "--exclude") == ["object-156=on the moving cart"], "review exclusions reach every generator (D17, X11)"
    assert flag(by["box"], "--exclude") == ["object-003=accepted by SAM 3D"] and flag(by["box"], "--invoke") is None
    assert flag(by["merge"], "--review") == [str(REVIEW / "walmart.json")] and by["merge"].leaves == {"review": REVIEW / "walmart.json"}
    assert by["sam3d"].budget_flags == {"--max-usd": "usd"} and flag(by["sam3d"], "--max-usd") == ["10"] and flag(by["sam3d"], "--workers") == ["4"]
    assert by["camera"].models == (("camera", "princeton-vl/DROID-SLAM", "2dfd39f0", None),) and by["splat"].models == () and by["box"].models == ()
    assert by["splat"].budget_flags == {"--max-minutes": ("minutes", stages.USD_PER_S["H100"])} and flag(by["splat"], "--max-minutes") == ["58"]


def test_commercial_profile():
    by = build("walmart", profile=COMMERCIAL)
    check_structure(list(by.values()))
    assert not {"names", "sam3d", "recgen", "box", "merge", "lingbot", "lingbot_build", "dense_gate", "generator_plan"} & set(by)
    assert flag(by["depth"], "--da3-model") == ["depth-anything/DA3-BASE"] and flag(by["register-0-383"], "--model") == ["depth-anything/DA3-BASE"]
    assert by["depth"].models == (("depth", "depth-anything/DA3-BASE", "f4a6c9b3", None),)
    assert by["register-0-383"].models == (("register", "depth-anything/DA3-BASE#camera", "f4a6c9b3", None),), "the pin keeps its licence row"
    assert flag(by["import"], "--models") is None and flag(by["import"], "--dense-points") is None and flag(by["floor_infer"], "--dense") is None
    assert "object_map=@object_map:out" in words(by["static_filter"]) and by["static_filter"].stage_from == {"link": {"surfaces": "@object_map:surfaces"}}
    named = build("walmart", profile=types.SimpleNamespace(**{**vars(COMMERCIAL), "namer": "gemini"}))
    assert "recgen" not in named and flag(named["merge"], "--recgen") is None and flag(named["merge"], "--sam3d") == ["@sam3d:out"]
    assert flag(named["sam3d"], "--max-usd") == ["5"] and flag(named["box"], "--exclude") == ["object-003=accepted by SAM 3D"]


def test_two_caption_bands_leave_those_layers_blank():
    values = {**SITES["me340"][2], "overlay": {**NONE, "bands": [[100, 140], [648, 704]], "lingbot_rows": None, "splat_captions": None,
                                              "objects_no_captions": None, "refuse": ["texture", "fill", "lingbot", "splat", "objects"]}}
    by = build("me340", decisions=values)
    check_structure(list(by.values()))
    assert not {"texture", "fill", "splat", "splat_final", "lingbot", "sam3d", "box", "merge"} & set(by)
    assert flag(by["outlines"], "--mesh") == ["@fuse:predicted"] and flag(by["import"], "--shell-glb") is None and flag(by["import"], "--splats") is None


def test_pending_until_each_decision_resolves():
    """graph raises Pending(stage, specs so far) in batches; the specs always hold that stage; generators wait for the
    static filter, so they can seed their journals from runs with the same inputs (D17)."""
    values, given, order = SITES["walmart"][2], {}, []
    while True:
        try:
            specs = stages.graph(ctx("walmart", decisions=given))
            break
        except Pending as p:
            names = [s.name for s in p.specs]
            assert p.decision in names and p.decision not in given
            if "sam3d" in names:
                assert "static_filter" in given
            order.append(p.decision)
            given[p.decision] = values[p.decision]
    assert order[:2] == ["shots", "lens"] and order.index("voxel") < order.index("static_filter") < order.index("splat_pick")
    assert set(order) == set(values) and specs[-1].name == "import"


def test_part_c_profiles_when_present():
    """With Part C's profiles.py beside this file (after integration): both profiles build, research refuses nothing."""
    profiles = pytest.importorskip("report_runner.profiles")
    for name in ("research", "commercial"):
        specs = stages.graph(ctx("walmart", profile=name))
        check_structure(specs)
        if name == "research":
            assert all(profiles.refuse(name, s) is None for s in specs)


def test_no_mapped_shot_stops():
    with pytest.raises(RuntimeError, match="no shot"):
        stages.graph(ctx("walmart", decisions={"shots": {"primary": [0, 40], "others": [], "mapped": False}}))


def test_generators_seed_from_the_store():
    """D17: a generator stage whose depth, object map and masks digests match an earlier run gets that run's journals (env,
    never the key)."""
    seen = []

    class Store:
        def _status(self, spec):
            return {}

        def _key(self, spec, status):
            return "k", {r: f"digest-{r}" for r, _ in spec.inputs.items()}

        def latest(self, stage, match):
            seen.append((stage, sorted(match)))
            return types.SimpleNamespace(dir=Path(f"/runs/walmart-{stage}-old")) if stage == "sam3d" else None
    c = ctx("walmart")
    specs = {s.name: s for s in stages.graph(Ctx(*[getattr(c, f) for f in ("site", "video", "start", "end", "profile", "review", "art")], Store(), c.decisions))}
    assert specs["sam3d"].env["PANOPTES_SEED_JOURNAL"] == "/runs/walmart-sam3d-old/out" and "PANOPTES_SEED_JOURNAL" not in specs["box"].env
    assert ("sam3d", ["fuse", "mask_root", "static_filter"]) in seen and {s for s, _ in seen} == {"sam3d", "recgen", "box"}


def test_a_republish_keeps_the_published_title(tmp_path):
    """D14: --republish is the site's last imports.jsonl record path (path only) and a republish keeps the published title
    (O1); a first import gets the rule title. Only the delivered profile or an explicit --republish republishes: a research
    or commercial run on a delivered site imports a report of its own, never a new version of the delivered one."""
    c = ctx("walmart")
    first = {s.name: s for s in stages.graph(c)}["import"]
    assert flag(first, "--title") == ["walmart 03:10–03:40 (imported, not accepted)"] and flag(first, "--republish") is None
    (tmp_path / "imports.jsonl").write_text(json.dumps({"site": "walmart", "importRecordPath": "/art/.platform/imports/video-import-p.json", "title": "Walmart aisle"}) + "\n")
    fields = [getattr(c, f) for f in ("site", "video", "start", "end", "profile", "review", "art")]
    research = {s.name: s for s in stages.graph(Ctx(*fields, types.SimpleNamespace(state=tmp_path), c.decisions))}["import"]
    assert flag(research, "--republish") is None and flag(research, "--title") == flag(first, "--title"), "research: a new report of its own"
    for asked in (Ctx(*fields, types.SimpleNamespace(state=tmp_path), c.decisions, republish=True),
                  Ctx(*fields[:4], types.SimpleNamespace(**{**vars(RESEARCH), "name": "delivered"}), *fields[5:], types.SimpleNamespace(state=tmp_path), c.decisions)):
        again = {s.name: s for s in stages.graph(asked)}["import"]
        assert flag(again, "--title") == ["Walmart aisle"] and flag(again, "--republish") == ["/art/.platform/imports/video-import-p.json"]


def test_movers_never_take_the_mapped_shots_people():
    """D3b: an accepted other shot without track windows of its own is left blank; only delivered ME340 (X6) moved the
    mapped shot's analysis into it."""
    values = {**SITES["me340"][2], "track_windows": {**SITES["me340"][2]["track_windows"], "others": []}}
    assert not [n for n in build("me340", decisions=values) if n.startswith(("movers-", "otracks-"))]
    delivered = build("me340", decisions=values, profile=types.SimpleNamespace(**{**vars(RESEARCH), "name": "delivered"}))
    assert flag(delivered["movers-14-226"], "--analysis") == ["@analysis:analysis"]


def test_floor_frames_take_the_operator_list_by_its_bytes(tmp_path):
    """O1 kept in research: REVIEW/<site>.floor-frames.json is a leaf of floor_frames (its bytes, with the clip it names);
    without it the rule's default applies."""
    by = build("walmart")
    operator = REVIEW / "walmart.floor-frames.json"
    assert words(by["floor_frames"])[-3:] == ["shots=@shots:decision", f"operator={operator}", "clip=@source:out"] and by["floor_frames"].leaves == {"operator": operator}
    (tmp_path / "walmart.json").write_bytes((REVIEW / "walmart.json").read_bytes())
    bare = build("walmart", review=tmp_path)
    assert words(bare["floor_frames"])[-1] == "shots=@shots:decision" and bare["floor_frames"].leaves == {}


# ---------------------------------------------------------------- templates against the delivered runs (read-only)
DELIVERED = {
    "me340": {"source": "../data/clips/me340-165", "cuts": "m0-integrate-cuts/me340-165", "census": "droid-me340-165-171", "camera": "droid-me340-165-171",
              "floor_masks": "sam3-me340-floor-174", "sam2": "sam2-me340-everything-184", "mask_root": "me340-masks-194", "motion": "me340-motion-masks-177",
              "stitch": "me340-stitched-186", "analysis": "me340-motion-analysis-187", "dynamic_masks": "me340-dynamic-masks-188",
              "depth": "da3-posed-me340-173", "metric": "da3-posed-me340-173", "fuse": "da3-posed-me340-223-shotc", "dynamic": "me340-dynamic-190",
              "object_map": "me340-object-map-195", "names": "me340-entity-names-200", "texture": "me340-textured-224", "fill": "me340-filled-225",
              "outlines": "me340-frame-selection-201", "events": "me340-events-197", "lingbot": "me340-lingbot-map-222", "lingbot_diagnose": "me340-lingbot-map-222",
              "lingbot_build": "me340-lingbot-map-222", "floor_infer": "me340-inferred-floor-244", "register-14-226": "me340-cutaway-register-304",
              "movers-14-226": "me340-cutaway-dynamic-305", "sam3d": "me340-object-models-241-sam3d", "recgen": "me340-object-models-231",
              "box": "me340-object-models-302-box", "box_test": "m0-box-test-me340", "merge": "me340-object-models-303-merged"},
    "samsclub-a2": {"source": "../data/clips/samsclub-337", "lens_clip": "../data/clips/samsclub-337-a2", "cuts": "m0-integrate-cuts/samsclub-337",
                    "census": "droid-samsclub-337-157", "camera": "droid-samsclub-a2-280", "floor_masks": "sam3-samsclub-floor-159",
                    "sam2": "sam2-samsclub-everything-160", "mask_root": "samsclub-masks-162", "motion": "samsclub-a-motion-masks-262",
                    "tracks-0-300": "samsclub-a-sam31-tracks-263", "tracks-260-420": "samsclub-a-sam31-tracks-265", "stitch": "samsclub-a-stitched-266",
                    "analysis": "samsclub-a-motion-analysis-267", "dynamic_masks": "samsclub-a-dynamic-masks-268", "depth": "da3-posed-samsclub-a2-281",
                    "metric": "da3-posed-samsclub-a2-281", "fuse": "da3-posed-samsclub-a2-281", "dynamic": "samsclub-a2-dynamic-285",
                    "texture": "samsclub-a2-textured-282", "fill": "samsclub-a2-filled-283", "object_map": "samsclub-a2-object-map-284",
                    "names": "samsclub-a2-entity-names-288", "outlines": "samsclub-a2-frame-selection-289", "events": "samsclub-events-198",
                    "splat": "samsclub-a2-splat-287", "splat_clean": "samsclub-a2-splat-287", "splat_final": "samsclub-a2-splat-287",
                    "lingbot": "samsclub-a-lingbot-map-264", "lingbot_diagnose": "samsclub-a-lingbot-map-264", "lingbot_build": "samsclub-a2-lingbot-map-286",
                    "floor_infer": "samsclub-a2-inferred-floor-296-icp", "sam3d": "samsclub-a2-object-models-291-sam3d", "recgen": "samsclub-a2-object-models-292",
                    "box": "samsclub-a2-object-models-301-box", "box_test": "m0-box-test-samsclub-a2", "merge": "samsclub-a2-object-models-303-merged"},
    "walmart": {"source": "../data/clips/walmart-190", "cuts": "m0-integrate-cuts/walmart-190", "census": "droid-walmart-190-168",
                "camera": "droid-walmart-190-shot383-250", "floor_masks": "sam3-walmart-floor-170", "sam2": "sam2-walmart-everything-181",
                "mask_root": "walmart-masks-182", "motion": "walmart-motion-masks-252", "tracks-383-750": "walmart-sam31-tracks-253",
                "analysis": "walmart-motion-analysis-254", "dynamic_masks": "walmart-dynamic-masks-255", "depth": "da3-posed-walmart-251-shot383",
                "metric": "da3-posed-walmart-251-shot383", "fuse": "da3-posed-walmart-251-shot383", "dynamic": "walmart-dynamic-262",
                "texture": "walmart-textured-256", "fill": "walmart-filled-257", "object_map": "walmart-object-map-259", "names": "walmart-entity-names-261",
                "outlines": "walmart-frame-selection-263", "events": "walmart-events-199", "splat": "walmart-splat-260", "splat_clean": "walmart-splat-260",
                "splat_final": "walmart-splat-260", "lingbot": "walmart-lingbot-map-258", "lingbot_diagnose": "walmart-lingbot-map-258",
                "lingbot_build": "walmart-lingbot-map-258", "floor_infer": "walmart-inferred-floor-265", "sam3d": "walmart-object-models-267-sam3d",
                "recgen": "walmart-object-models-264", "box": "walmart-object-models-300-box", "box_test": "m0-box-test-walmart",
                "merge": "walmart-object-models-303-merged"},
}


def delivered_path(site, producer, role, rest, specs):
    """How an adopted run answers @producer:role: the delivered folder holds a tool's output at its top (no 'out/')."""
    folder = (RUNS / DELIVERED[site][producer]).resolve()
    rel = specs[producer].outputs[role] if role else "out"
    rel = rel.removeprefix("out").lstrip("/")
    return folder / rel / rest.lstrip("/") if rest else folder / rel


def resolved(site, text, specs):
    """A template word with its tokens replaced by delivered paths; None when a token names a stage with no delivered folder."""
    out, last = "", 0
    for m in stages.TOKEN.finditer(text):
        producer, role = m.groups()
        if producer in stages.RESERVED and not role:
            return None
        if producer not in DELIVERED[site]:
            return None
        end = m.end()
        rest = re.match(r"/[^\s=]*", text[end:])
        rest = rest.group(0) if rest else ""
        out += text[last:m.start()] + str(delivered_path(site, producer, role, rest, specs))
        last = end + len(rest)
    return out + text[last:]



@needs_art
@pytest.mark.parametrize("site", sorted(SITES))
def test_templates_resolve_on_the_delivered_runs(site):
    """Every @stage:role a stage reads names an existing file or folder of the delivered run standing in for that stage."""
    specs = {s.name: s for s in stages.graph(ctx(site))}
    checked, missing = 0, []
    for s in specs.values():
        for text in [t for c in s.commands for t in c] + [r for refs in s.stage_from.values() for r in refs.values()]:
            if not stages.TOKEN.search(text) or text.startswith(f"{site}-"):
                continue
            path = resolved(site, text.split("=", 1)[1] if "=@" in text else text, specs)
            if path is None:
                continue
            checked += 1
            if not Path(path).exists():
                missing.append((s.name, text, path))
    assert not missing, missing
    assert checked >= 60, checked


@needs_art
def test_walmart_templates_normalize_to_the_delivered_argv():
    """P1 on the stages where M2 changes nothing for Walmart: the runner's argv, resolved on the delivered folders, normalizes
    to the recorded one (walmart stage graph S10-S30); LingBot build differs only by the recomputed --conf (deviation D4)."""
    specs = {s.name: s for s in stages.graph(ctx("walmart"))}
    folders = {(RUNS / d).resolve(): name for name, d in DELIVERED["walmart"].items()}

    def owner_of(path):
        path = Path(path).resolve()
        for folder, name in folders.items():
            if path == folder or folder in path.parents:  # one delivered folder can hold several stages: both sides get the same name
                return name, str(path.relative_to(folder)) if path != folder else "."
        return None
    R, CLIP = str(RUNS), str((RUNS / "../data/clips/walmart-190").resolve())
    delivered = {
        "motion": ["modal_apps/motion_masks.py", "--droid-run", f"{R}/droid-walmart-190-shot383-250", "--output", f"{R}/walmart-motion-masks-252", "--exclude-frames", "0:383"],
        "depth": ["modal_apps/mono_room.py", "infer", "--droid-run", f"{R}/droid-walmart-190-shot383-250", "--output", f"{R}/da3-posed-walmart-251-shot383",
                  "--da3-model", "depth-anything/DA3-GIANT-1.1", "--every", "3", "--exclude-frames", "0:383"],
        "fuse": ["modal_apps/mono_room.py", "fuse", "--droid-run", f"{R}/droid-walmart-190-shot383-250", "--output", f"{R}/da3-posed-walmart-251-shot383",
                 "--voxel-length-native", "0.011858", "--support-relative", "0.02", "--support-all-views", "--edge-jump", "0.03", "--carve",
                 "--dynamic-masks", f"{R}/walmart-dynamic-masks-255/masks", "--video", f"{CLIP}/source-rgb.mp4", "--floor-plane", f"{R}/da3-posed-walmart-251-shot383/metric-scale.json"],
        "texture": ["scripts/texture_fused_mesh.py", "--droid-run", f"{R}/droid-walmart-190-shot383-250", "--fused", f"{R}/da3-posed-walmart-251-shot383",
                    "--output", f"{R}/walmart-textured-256", "--video", f"{CLIP}/source-full.mp4", "--dynamic-masks", f"{R}/walmart-dynamic-masks-255/masks"],
        "object_map": ["scripts/build_video_object_map.py", "--droid-run", f"{R}/droid-walmart-190-shot383-250", "--depth-run", f"{R}/da3-posed-walmart-251-shot383",
                       "--masks", f"{R}/walmart-masks-182", "--floor", f"{R}/da3-posed-walmart-251-shot383/metric-scale.json",
                       "--dynamic-masks", f"{R}/walmart-dynamic-masks-255/masks", "--output", f"{R}/walmart-object-map-259"],
        "splat": ["modal_apps/splat_train.py", "--clip", CLIP, "--droid-run", f"{R}/droid-walmart-190-shot383-250", "--masks", f"{R}/walmart-dynamic-masks-255/masks",
                  "--mesh", f"{R}/da3-posed-walmart-251-shot383/mono-anchored-mesh.ply", "--fill", f"{R}/walmart-filled-257/textured-scene.glb",
                  "--scale", f"{R}/da3-posed-walmart-251-shot383/metric-scale.json", "--captions", "0", "0", "0", "0", "--steps", "60000", "--cap", "2500000",
                  "--pose", "--max-elongation", "4", "--max-minutes", "45", "--skip", "0-382", "--output", f"{R}/walmart-splat-260"],
        "lingbot_build": ["scripts/lingbot_dense_map.py", "build", "--run-id", "walmart-lingbot-map-258", "--droid-run", f"{R}/droid-walmart-190-shot383-250",
                          "--clip", CLIP, "--masks", f"{R}/walmart-dynamic-masks-255/masks", "--depth-run", f"{R}/da3-posed-walmart-251-shot383",
                          "--mesh", f"{R}/da3-posed-walmart-251-shot383/mono-anchored-mesh.ply", "--output", f"{R}/walmart-lingbot-map-258",
                          "--exclude-frames", "0:383", "--overlay-rows", "0:0"]}
    for name, argv in delivered.items():
        runner = [(resolved("walmart", w, specs) or w) if stages.TOKEN.search(w) else w for w in words(specs[name])]
        mine = stages.normalize_command([stages.PY, *runner[1:]], owner_of)
        theirs = stages.normalize(argv[0], argv[1:], owner_of)
        if name == "lingbot_build":
            assert mine["flags"].pop("--conf") == [1.01] and theirs["flags"].pop("--conf") == [1.06]
        assert mine == theirs, name


# ---------------------------------------------------------------- normalize and spans
def test_spans_round_trip_to_end_exclusive():
    for text, style in (("14:226", "colon"), ("14-225", "dash"), ("0:383", "colon"), ("0-382", "dash")):
        assert stages.fmt_span(stages.to_span(text, style), style) == text
    assert stages.to_span("14:226", "colon") == stages.to_span("14-225", "dash") == [14, 226]
    import_ = stages.normalize("scripts/import_video_scene.py", ["--exclude-frames", "0:383"], lambda p: None)["flags"]["--exclude-frames"]
    objects = stages.normalize("scripts/complete_video_objects.py", ["--skip-frames", "0-382"], lambda p: None)["flags"]["--skip-frames"]
    tracks = stages.normalize("modal_apps/sam3_motion_tracks.py", ["--frames", "0", "383"], lambda p: None)["flags"]["--frames"]
    assert import_ == objects == tracks == [[0, 383]]


def test_normalize_fills_defaults_drops_key_free_flags_and_is_idempotent(tmp_path):
    leaf = tmp_path / "review.json"
    leaf.write_text("{}")
    owners = {Path("/r/fuse"): ("fuse", "out"), Path("/r/masks"): ("dynamic_masks", "masks")}
    owner_of = owners.get
    argv = ["--droid-run", "/r/cam", "--depth-run", "/r/fuse", "--dynamic-masks", "/r/masks", "--skip-frames", "0-382", "--voxel-native", "0.0118580",
            "--exclude", "object-156=on a moving cart", "object-080=skylight", "--max-usd", "4", "--invoke", "--workers", "4", "--output", "/r/new"]
    n = stages.normalize("scripts/complete_video_objects.py", argv, owner_of)
    f = n["flags"]
    assert f["--depth-run"] == [{"in": "fuse", "role": "out"}] and f["--dynamic-masks"] == [{"in": "dynamic_masks", "role": "masks"}]
    assert f["--droid-run"] == [{"missing": "/r/cam"}] and f["--voxel-native"] == [0.011858] and f["--skip-frames"] == [[0, 383]]
    assert f["--exclude"] == ["object-156", "object-080"], "ids and their order stay, the reasons go"
    assert not {"--max-usd", "--invoke", "--workers", "--output"} & set(f) and f["--count"] == [10] and f["--all"] is False and f["--generator"] == ["recgen"]
    explicit = ["--generator", "recgen", "--count", "10", "--output", "/elsewhere", "--exclude", "object-156", "object-080", "--voxel-native", "0.011858",
                "--skip-frames", "0-382", "--dynamic-masks", "/r/masks", "--depth-run", "/r/fuse", "--droid-run", "/r/cam", "--max-usd", "30"]
    assert stages.normalize("$REPO/scripts/complete_video_objects.py", explicit, owner_of) == n
    decided = stages.normalize("report_runner.decide", ["generator_plan", "--out", "/r/new", "sam3d=/r/fuse", f"review={leaf}"], owner_of)
    assert decided == {"script": "report_runner.decide", "args": ["generator_plan"],
                       "flags": {"review": {"leaf": hashlib.sha256(b"{}").hexdigest()}, "sam3d": {"in": "fuse", "role": "out"}}}
    with pytest.raises(ValueError, match="unknown flag"):
        stages.normalize("scripts/complete_video_objects.py", ["--frames", "1:2"], owner_of)
    assert stages.normalize("modal_apps/splat_train.py", [], owner_of)["flags"]["--captions"] == [648, 704, 160, 1120], "ME340's defaults are filled in"
    assert stages.normalize("scripts/lingbot_dense_map.py", ["build"], owner_of)["flags"]["--overlay-rows"] == [[646, 706]]


# ---------------------------------------------------------------- versions.json
def test_versions_json_is_current():
    assert stages.check_versions(json.loads((REPO / "tests/versions.json").read_text())) == []


def test_versions_guard_needs_a_bump_when_code_changes():
    recorded = stages.current_versions()
    recorded["fuse"] = {**recorded["fuse"], "depsSha256": "0" * 64}
    assert any(p.startswith("fuse: its code changed") for p in stages.check_versions(recorded))
    recorded["fuse"]["version"] = 0  # the version moved with it: versions.json only needs rewriting
    assert [p for p in stages.check_versions(recorded) if p.startswith("fuse")] == ["fuse: versions.json is stale; run python -m report_runner.stages --versions"]
    assert {"modal_apps/mono_room.py", "modal_apps/droid_room.py", "modal_apps/moge3_app.py"} <= set(stages.deps_of("modal_apps/mono_room.py"))
    assert "scripts/report_runner/spec.py" in stages.deps_of("scripts/report_runner/stages.py"), "relative imports are followed"


def test_a_changed_rule_needs_a_version_bump():
    """Decision rules are guarded like the script kinds: each rule's code (its function in decide.py, every module-level
    name it reaches, the modules it imports, the scripts it runs) is tests/versions.json 'rule:NAME', and is in its key."""
    from report_runner import decide, profiles
    assert profiles.M2_RULES == {n: f"{n}@{v}" for n, v in decide.VERSIONS.items()} and set(decide.VERSIONS) == set(decide.RULES)
    text = (REPO / stages.DECIDE).read_text()
    changed = text.replace("LENS_FRAMES, LENS_KEEP_DEG, MIN_PLANE_INLIERS = 15, 3., .90", "LENS_FRAMES, LENS_KEEP_DEG, MIN_PLANE_INLIERS = 15, 3.5, .90")
    assert changed != text
    moved = {r for r in decide.RULES if stages.rule_code(r, changed)[1] != stages.rule_code(r)[1]}
    assert moved == {"lens"}, "a constant moves the rules that read it, and only those"
    helper = text.replace('return json.loads(_file(path, name).read_text())', 'return json.loads(_file(path, name).read_text() or "{}")')
    assert helper != text and {r for r in decide.RULES if stages.rule_code(r, helper)[1] != stages.rule_code(r)[1]} >= {"shots", "voxel", "lens", "generator_plan"}
    recorded = stages.current_versions()
    recorded["rule:lens"] = {**recorded["rule:lens"], "depsSha256": stages.rule_code("lens", changed)[1]}  # versions.json from before the edit
    assert [p for p in stages.check_versions(recorded) if p.startswith("rule:lens")] == \
        ["rule:lens: its code changed but decide.VERSIONS['lens'] is still 1; bump it, then --versions"]
    assert {"scripts/lingbot_icp_refine.py", "scripts/lingbot_dense_map.py"} <= set(stages.rule_code("dense_gate")[0])
    assert "scripts/register_cut_shot.py" in stages.rule_code("other_shot")[0] and "modal_apps/mono_room.py" in stages.rule_code("static_filter")[0]


# ---------------------------------------------------------------- script fixes
@needs_art
def test_derive_rebuilds_samsclub_a2(tmp_path):
    """prepare_video_clip --derive: samsclub-337 frames 0-419 with the shot's 55.3194 deg gives a2's MP4s byte for byte."""
    parent = ART / "data/clips/samsclub-337"
    source = Path(json.loads((parent / "clip.json").read_text())["source"]["video"])
    if not source.is_file():  # opened by its recorded path, the folder is never listed
        pytest.skip("the source MP4 of samsclub-337 is not on this machine")
    local = tmp_path / "parent"
    local.mkdir()
    subprocess.run(["cp", "-c", "-R", str(parent / "rgb"), str(local / "rgb")], check=True)  # an APFS clone: the delivered frames are untouched
    for name in ("rgb.txt", "clip.json"):
        shutil.copy(parent / name, local / name)
    done = subprocess.run([sys.executable, str(REPO / "scripts/prepare_video_clip.py"), "--derive", str(local), "--frames", "0:420",
                           "--fov-deg", "55.3194223319173", "--output", str(tmp_path / "a2")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(tmp_path / "a2/source-rgb.mp4").startswith("9ffd4381") and sha(tmp_path / "a2/source-full.mp4").startswith("8a5faee4")
    clip = json.loads((tmp_path / "a2/clip.json").read_text())
    assert clip["K"] == [610.5529403686523, 610.5529403686523, 319.5, 239.5] and clip["name"] == "a2" and clip["source"]["frames"] == 420
    assert (tmp_path / "a2/rgb").stat().st_nlink and len(list((tmp_path / "a2/rgb").iterdir())) == 420


def _prepared(run, generator, excludes, tmp_path):
    """prepare() of a delivered complete_video_objects run at HEAD, with its recorded inputs (CPU; nothing written to $ART)."""
    import argparse
    import complete_video_objects as cvo
    m = json.loads((run / "manifest.json").read_text())
    out = tmp_path / "out"
    out.mkdir()
    if (run / "resegmented").is_dir():
        shutil.copytree(run / "resegmented", out / "resegmented")  # SAM 2.1 answers of the run: prepare asks nothing again
    args = argparse.Namespace(**{k: Path(v) for k, v in m["inputs"].items()}, output=out, entities=[], exclude=excludes, count=cvo.COUNT, all=True,
                              generator=generator, reassess=False, invoke=False, function_id=m["route"]["modalFunctionId"], max_usd=0.,
                              skip_frames=["0-382"], voxel_native=0.011858, no_captions=True, workers=1)
    cvo.VOXEL, cvo.OCCLUSION = args.voxel_native, args.voxel_native / 2
    cvo.FIT_GATE["max_fit_median_native"] = cvo.VOXEL
    args.skip = cvo.frame_set(args.skip_frames)
    args.metres_per_native = json.loads((args.depth_run / "metric-scale.json").read_text())["metres_per_native_unit"]
    _, _, _, chosen, _ = cvo.prepare(args)
    tries = lambda c: list(zip(c["tries"], c["payloads"])) + [(c["tries"][0], {**c["payloads"][0], "seed": cvo.EXTRA_SEED})]
    made = {o["entityId"]: o.get("attempts") or [] for o in m["objects"]}
    paid = lambda attempt: not str((attempt.get("record") or {}).get("error", "")).startswith("not requested")  # the cap stopped it: never sent
    return cvo, args, [(c["entity"]["entityId"], view, payload) for c in chosen
                       for (view, payload), attempt in zip(tries(c), made.get(c["entity"]["entityId"], [])) if paid(attempt)]


@needs_art
def test_sam3d_journal_answers_every_call_of_walmart_267(tmp_path):
    """Seeded with run 267's journal, every SAM 3D call that run made is read back by sam3d_obtain itself (no --invoke, so a
    miss would come back empty instead of calling Modal): 0 new calls. Walmart 267 ran with --no-captions (not recorded)."""
    run = RUNS / "walmart-object-models-267-sam3d"
    excludes = [r["entityId"] + "=x" for r in json.loads((run / "manifest.json").read_text())["rejected"] if r["reason"].startswith("excluded at review")]
    cvo, args, calls = _prepared(run, "sam3d", excludes, tmp_path)
    new = [(eid, view["frame"], payload["seed"]) for eid, view, payload in calls
           if cvo.sam3d_obtain(payload, run / "journal-sam3d" / eid, args)[0] is None]
    assert len(calls) == 157 and new == []


@needs_art
def test_recgen_journal_holds_every_call_of_walmart_264(tmp_path):
    """RecGen's read-back goes through the Modal volume, so here only the journal lookup runs: call_identity under
    journal_folder (and its re-dispatch folders) finds a received call for every try run 264 made."""
    run = RUNS / "walmart-object-models-264"
    excludes = [r["entityId"] + "=x" for r in json.loads((run / "manifest.json").read_text())["rejected"] if r["reason"].startswith("excluded at review")]
    cvo, args, calls = _prepared(run, "recgen", excludes, tmp_path)

    def received(eid, view, payload):
        folder = cvo.journal_folder(run / "journal", payload, view["frame"], args.function_id)
        attempts = [(folder, payload)] + [(folder / f"redispatch-{k}", {**payload, "_researchProtocol": {"dispatchAttemptId": f"redispatch-{k}"}})
                                          for k in range(1, cvo.REDISPATCH + 1)]
        return any((f / cvo.call_identity(p, args.function_id) / "manifest.json").exists() for f, p in attempts)
    new = [(eid, view["frame"]) for eid, view, payload in calls if not received(eid, view, payload)]
    assert calls and new == [], new


def _fake_generator_run(tmp_path, monkeypatch, workers, name):
    """complete_video_objects.run with the box generator, prepare, assess and write replaced by fakes: 8 objects, 2 views each."""
    import complete_video_objects as cvo
    live, peak, lock = [0], [0], threading.Lock()

    def box_obtain(payload, args):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(.15)
        with lock:
            live[0] -= 1
        n, view = int(payload["entityId"][-1]), int(payload["anchorObservationId"][-1])
        if payload["seed"] != cvo.SEED or view != n % 3:  # object n is accepted on view n % 3 (never, for view 2)
            return None, {"reason": "no box"}
        return (np.eye(3) * (n + 1), np.array([[0, 1, 2]]), np.full((3, 3), 128.)), {"generator": "fake"}

    def assess(entity, view, crop, mesh, record, rows, clip, args):
        v = {k: 0. for k in ("silhouette_iou", "relative_depth_median", "relative_depth_p95", "fitResidualNative", "observedCoverage", "entityCoverage",
                             "gpuSeconds", "wallSeconds", "estimatedUsd", "fitResidualCm", "observedShare")}
        return {"validation": {**v, "accepted_source_consistency": True, "rejectionReasons": [], "gpuType": None, "triangles": 1, "maskSource": "fake"},
                "vertices": mesh[0], "faces": mesh[1], "colors": mesh[2], "view": view}

    def write(entity, best, args, plan_up):
        folder = args.output / "models" / entity["entityId"]
        folder.mkdir(parents=True)
        (folder / "model.glb").write_bytes(best["vertices"].tobytes())
        return 1

    crop = {"mask": np.zeros((4, 4), bool), "depth": np.zeros((4, 4)), "K": np.eye(3), "cameraToWorld": np.eye(4)}
    chosen = [{"entity": {"entityId": f"object-00{n}", "label": "box", "labelStatus": "clear"}, "why": "rule",
               "tries": [{"frame": 10 * n + v, "observation": f"o{n}{v}"} for v in range(2)],
               "payloads": [{"entityId": f"object-00{n}", "anchorObservationId": f"o{n}{v}", "seed": cvo.SEED, "views": [crop]} for v in range(2)]}
              for n in range(8)]

    def prepare(args):
        args.explicit = set()
        return None, {}, {}, chosen, []
    for attr, fake in (("prepare", prepare), ("sheet", lambda *a: None), ("box_obtain", box_obtain), ("assess", assess), ("write", write),
                       ("decimate", lambda v, f, c, t: (v, f, c))):
        monkeypatch.setattr(cvo, attr, fake)
    inputs = tmp_path / "inputs"
    (inputs / "depth").mkdir(parents=True, exist_ok=True)
    (inputs / "depth/metric-scale.json").write_text(json.dumps({"metres_per_native_unit": 2.}))
    import argparse
    args = argparse.Namespace(droid_run=inputs / "droid", depth_run=inputs / "depth", object_map=inputs / "map", masks=inputs / "masks",
                              dynamic_masks=inputs / "dyn", clip=inputs / "clip", output=tmp_path / name, generator="box", all=False, reassess=False,
                              invoke=False, max_usd=30., skip=set(), workers=workers, function_id="fu", no_captions=True)
    cvo.run(args)
    return args.output, peak[0]


def test_workers_change_speed_not_results(tmp_path, monkeypatch):
    one, peak1 = _fake_generator_run(tmp_path, monkeypatch, 1, "w1")
    four, peak4 = _fake_generator_run(tmp_path, monkeypatch, 4, "w4")
    assert peak1 == 1 and peak4 >= 3, (peak1, peak4)
    m1, m4 = (json.loads((d / "manifest.json").read_text()) for d in (one, four))
    assert m1 == m4 and [o["entityId"] for o in m1["objects"]] == [f"object-00{n}" for n in range(8)], "judged in entity order"
    assert [o.get("accepted") for o in m1["objects"]] == [True, True, None] * 2 + [True, True]
    assert m1["inputs"]["clip"] == str(tmp_path / "inputs/clip"), "the manifest records the clip path, not the Clip object"
    models = lambda d: {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted((d / "models").rglob("*.glb"))}
    assert models(one) == models(four) and len(models(one)) == 6


def test_seed_journal_is_copied_before_the_run(tmp_path, monkeypatch):
    seed = tmp_path / "old/out"
    (seed / "journal-sam3d/object-001/sam3d-abc").mkdir(parents=True)
    (seed / "journal-sam3d/object-001/sam3d-abc/record.json").write_text("{}")
    monkeypatch.setenv("PANOPTES_SEED_JOURNAL", str(seed))
    out, _ = _fake_generator_run(tmp_path, monkeypatch, 2, "seeded")
    assert (out / "journal-sam3d/object-001/sam3d-abc/record.json").read_text() == "{}" and (seed / "journal-sam3d").is_dir()


def test_lingbot_masks_tar_is_keyed_by_content(tmp_path, monkeypatch):
    """Two mask sets under one run id upload two tars; the same set again uploads nothing (it used to reuse the first tar)."""
    import lingbot_dense_map as ldm

    class Volume:
        def __init__(self):
            self.files = {}

        def iterdir(self, run_id):
            return [types.SimpleNamespace(path=p) for p in self.files if p.startswith(run_id + "/")]

        @contextlib.contextmanager
        def batch_upload(self):
            yield types.SimpleNamespace(put_file=lambda f, path: self.files.__setitem__(path, f.read()))
    volume = Volume()
    monkeypatch.setattr(ldm, "volume", volume)
    for name, data in (("a", b"one"), ("b", b"two")):
        (tmp_path / name).mkdir()
        (tmp_path / name / "00001-0.png").write_bytes(data)
    first, second = ldm.upload_masks("run", tmp_path / "a"), ldm.upload_masks("run", tmp_path / "b")
    assert first != second and sorted(volume.files) == sorted([f"run/{first}", f"run/{second}"])
    assert ldm.upload_masks("run", tmp_path / "a") == first and len(volume.files) == 2
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(volume.files[f"run/{second}"])) as archive:
        assert archive.extractfile("00001-0.png").read() == b"two"
    assert "me340-filled-213" not in (REPO / "scripts/lingbot_dense_map.py").read_text()


@needs_art
def test_text_tracks_keep_their_prompt_as_label(tmp_path):
    """ME340 analysis 187 replayed at HEAD: same entities and masks, and the motion track the stitch folded into the presenter's
    'person' track (339 frames, labelled 'moving object' before, fixed by hand with --person-tracks in run 305) is 'person'."""
    r = RUNS
    done = subprocess.run([sys.executable, str(REPO / "scripts/motion_tracks_to_analysis.py"), "--droid-run", str(r / "droid-me340-165-171"),
                           "--tracks", *[str(r / f"me340-sam31-tracks-{n}") for n in (178, 179, 180)], "--stitched", str(r / "me340-stitched-186"),
                           "--output", str(tmp_path / "a")], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]
    now, then = (json.loads(p.read_text()) for p in (tmp_path / "a/analysis.json", r / "me340-motion-analysis-187/analysis.json"))
    shape = lambda d: [(f["sourceFrame"], [(o["entityId"], o["maskUrl"]) for o in f["objects"]]) for f in d["frames"]]
    assert shape(now) == shape(then) and now["entities"] == then["entities"]
    presenter = [o for f in now["frames"] for o in f["objects"] if o["entityId"] == "me340-sam31-tracks-178-text-0"]
    assert len(presenter) == 687 and {(o["label"], o["sourceLabel"]) for o in presenter} == {("person", "person")}
    assert {o["label"] for f in now["frames"] for o in f["objects"] if o["entityId"] == "me340-sam31-tracks-180-motion-4"} == {"moving object"}


@needs_art
def test_text_only_tracking_sends_no_motion_seeds(tmp_path, monkeypatch):
    """--text-only: no --motion, the remote gets no seeds (so no motion session), the run is complete on its text tracks.
    The GPU function is replaced; nothing is sent to Modal."""
    import sam3_motion_tracks as smt
    calls = []

    def remote(jpegs, seeds, text):
        calls.append((len(jpegs), seeds, text))
        mask = np.zeros((480, 640), bool)
        mask[100:300, 200:300] = True
        buffer = io.BytesIO()
        np.savez_compressed(buffer, report=json.dumps({"stages": {"text": {"shape": [480, 640], "frames_with_masks": 1}}}), **{"text/0/1": np.packbits(mask)})
        return buffer.getvalue()
    monkeypatch.setattr(smt, "app", types.SimpleNamespace(run=contextlib.nullcontext))
    monkeypatch.setattr(smt, "track_remote", types.SimpleNamespace(remote=remote))
    args = types.SimpleNamespace(droid_run=RUNS / "droid-walmart-190-shot383-250", video=None, fixed_camera=False, reference=None, motion=None,
                                 text="person", text_only=True, frames=(0, 6), stride=1, output=tmp_path / "t")
    smt.run(args)
    state = json.loads((tmp_path / "t/tracks.json").read_text())
    assert calls == [(6, [], "person")] and state["status"] == "complete" and state["text_only"] and state["motion_run"] is None
    assert state["objects"]["text"][0]["frames"] == 1 and "motion" not in state["objects"]
    with pytest.raises(ValueError, match="text-only"):
        smt.run(types.SimpleNamespace(**{**vars(args), "motion": tmp_path, "output": tmp_path / "u"}))
