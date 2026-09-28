"""M2 report runner, Part B: the stage graph from one MP4 to the import (runner design section 4), DEFAULTS and normalize().

graph(ctx) builds the StageSpecs in dependency order. When it needs a decision's value that ctx.decisions lacks it raises
Pending(stage, specs built so far); the runner (store.main) serves or runs that decision stage and calls graph again.
Decisions it only passes along as files (lens_gate, static_filter's map) are tokens, not Pending.

Conventions of every template (the token grammar is spec.py's):
  - A tool writes to @new/out, which it creates itself (almost every script refuses an existing output folder, and the
    store makes @new first); role 'out' is that folder. Exceptions: detect_shot_cuts writes @new/segments.json,
    lingbot_dense_map diagnose writes into @new, a clip goes to @clip (data/clips, where droid_room registers clips)
    and @new/out links to it, and a decision writes @new/<rule>.json (role 'decision').
  - Spans are end-exclusive [a, b) in code and are written in each script's own syntax: START:END, or inclusive A-B
    for splat_train --skip, complete_video_objects --skip-frames and infer_room_floor --skip-frames.
  - A tool that reads files beside the one it is given gets '@stage:out/<file>' so the whole folder is its input.
  - Only the decision stages carry rule versions; a consumer hashes a decision's value (early cutoff).

  python -m report_runner.stages --versions     # rewrite tests/versions.json {kind: {version, depsSha256}}
  python -m report_runner.stages --self-check
"""
import ast
import dataclasses
import functools
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys

from .spec import Pending, StageSpec

REPO = Path(__file__).resolve().parents[2]
PY = "python"  # resolved through PATH below: the key keeps the word, not this machine's interpreter
ENV = {"PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"}  # env is not in a key
TOKEN = re.compile(r"(?<![\w.])@([A-Za-z][\w-]*)(?::([\w.-]+))?")  # spec.py's grammar (store.TOKEN)
RESERVED = {"new", "key", "clip"}
DROID_BUILD = "$ART/runs/droid-me340-165-171"  # --reuse-build-from: the compiled DROID build every run reuses (not in the key)
WORKERS = 4  # complete_video_objects --workers: generator calls in flight (outputs do not depend on it)
# RecGen runs one call at a time: recgen_transport journals each call under the process-wide PANOPTES_RECGEN_JOURNAL, which
# complete_video_objects sets per call, so parallel calls journal into each other's folders (the first Lightning run failed so)
SERIAL_GENERATORS = {"recgen"}
CAPS = {"generator_usd": 10., "splat_minutes": 58.}  # profiles.CAPS; a profile may also cap one generator (<name>_usd)
MODEL_IDS = {"depth": "depth-anything/DA3-GIANT-1.1", "register": "depth-anything/DA3-GIANT-1.1"}  # argv needs these; the rest are fixed in their scripts
USD_PER_S = {"L4": .000222 + .0000131 + 8 * .00000222, "A100-40GB": .000583, "A100-80GB": .000694, "H100": .001097,
             "cpu16": 4 * .0000131 + 16 * .00000222, "cpu24": 4 * .0000131 + 24 * .00000222}  # store.PRICES + Modal CPU containers

# script of each stage kind (a stage's name is its kind, or kind-A-B for one span); None: staging only
SCRIPT = {"source": "scripts/prepare_video_clip.py", "cuts": "scripts/detect_shot_cuts.py", "census": "modal_apps/droid_room.py",
          "moge": "scripts/prepare_video_clip.py", "lens_clip": "scripts/prepare_video_clip.py", "camera": "modal_apps/droid_room.py",
          "floor_masks": "scripts/discover_video_keyframes.py", "sam2": "modal_apps/sam2_everything.py", "mask_root": None,
          "motion": "modal_apps/motion_masks.py", "tracks": "modal_apps/sam3_motion_tracks.py", "otracks": "modal_apps/sam3_motion_tracks.py",
          "stitch": "scripts/stitch_track_windows.py", "ostitch": "scripts/stitch_track_windows.py", "analysis": "scripts/motion_tracks_to_analysis.py",
          "oanalysis": "scripts/motion_tracks_to_analysis.py", "dynamic_masks": "scripts/assemble_dynamic_masks.py", "depth": "modal_apps/mono_room.py",
          "metric": "modal_apps/mono_room.py", "fuse": "modal_apps/mono_room.py", "dynamic": "modal_apps/mono_room.py",
          "texture": "scripts/texture_fused_mesh.py", "fill": "scripts/fill_scene_holes.py", "object_map": "scripts/build_video_object_map.py",
          "names": "scripts/name_video_entities.py", "outlines": "scripts/project_entities_to_frames.py", "events": "modal_apps/video_events.py",
          "lingbot": "modal_apps/lingbot_room.py", "lingbot_diagnose": "scripts/lingbot_dense_map.py", "lingbot_build": "scripts/lingbot_dense_map.py",
          "floor_infer": "scripts/infer_room_floor.py", "splat": "modal_apps/splat_train.py", "splat_clean": "modal_apps/splat_train.py",
          "splat_final": "modal_apps/splat_train.py", "register": "scripts/register_cut_shot.py", "movers": "scripts/register_cut_shot.py",
          "sam3d": "scripts/complete_video_objects.py", "recgen": "scripts/complete_video_objects.py", "box": "scripts/complete_video_objects.py",
          "box_test": "scripts/box_free_space.py", "merge": "scripts/merge_object_models.py", "import": "scripts/import_video_scene.py"}
# bump a kind's version when its code (tests/versions.json depsSha256) changes; the guard test fails until you do
# 2 at M2 integration: Part C pinned the HF revisions these tools load (MoGe-3: source and moge; DA3: depth; SAM 2.1; SAM 3: floor
# masks) and gave the import --lens and the corrected scale limitation. The other kinds only saw a new contract_scale branch that
# only the import reaches, or pins in code paths they do not run.
# names not bumped at the M2 review: its deps changed only through profiles.M2_RULES (floor_frames@2), which
# name_video_entities never reads (it reads QWEN3VL_NAMING_GATE); nor for U6 (commercial names with Gemini: profiles.CLOUD,
# Profile.cloud, the commercial rows), which only the runner reads; a commercial names key changes through its model pins.
# The delivered profile only serves the adopted runs, which no code of today made: its keys keep the versions they were adopted
# under (ADOPTED), whatever VERSIONS says later.
ADOPTED = {kind: 2 if kind in ("source", "moge", "depth", "sam2", "floor_masks", "import") else 1 for kind in SCRIPT}
# 2 at the Lightning fixes: floor_infer withholds the untested convex outline (no --dense)
VERSIONS = ADOPTED | {"floor_infer": 2}
DECIDE = "scripts/report_runner/decide.py"  # decision stages: versioned by their rule (decide.VERSIONS, rule_code), not here

# every option's default, as each script's argparse declares it (tests/test_report_runner_stages.py re-reads them by AST)
DEFAULTS = {
    "scripts/prepare_video_clip.py": {'--self-check': False, '--video': None, '--start': None, '--end': None, '--name': None, '--output': None, '--fov-deg': None, '--full-video': None, '--fov-of': None, '--derive': None, '--frames': None},
    "scripts/detect_shot_cuts.py": {'--clip': None, '--output': None, '--self-check': False},
    "modal_apps/droid_room.py": {'--output': None, '--run-id': None, '--reuse-build-from': None, '--clip': 'fr1-room', '--frames': None},
    "scripts/discover_video_keyframes.py": {'--video': None, '--frames': None, '--output': None, '--prompt': 'person', '--provider': 'self-hosted'},
    "modal_apps/sam2_everything.py": {'--self-check': False, '--video': None, '--frames': None, '--work-width': None, '--known': None, '--output': None},
    "modal_apps/motion_masks.py": {'--self-check': False, '--droid-run': None, '--video': None, '--fixed-camera': False, '--output': None, '--every': 10, '--gap': 8, '--reference': None, '--exclude-frames': []},
    "modal_apps/sam3_motion_tracks.py": {'--self-check': False, '--stride': 1, '--droid-run': None, '--video': None, '--fixed-camera': False, '--motion': None, '--frames': None, '--text': 'person', '--text-only': False, '--reference': None, '--output': None},
    "scripts/stitch_track_windows.py": {'--self-check': False, '--tracks': None, '--output': None},
    "scripts/motion_tracks_to_analysis.py": {'--self-check': False, '--droid-run': None, '--tracks': None, '--stitched': None, '--output': None},
    "scripts/assemble_dynamic_masks.py": {'--self-check': False, '--droid-run': None, '--tracks': None, '--analysis': None, '--reference': None, '--output': None},
    "modal_apps/mono_room.py": {'--droid-run': None, '--support': None, '--output': None, '--stride': 1, '--every': None, '--midframes': False, '--da3-model': None, '--voxel-length-native': 0.03, '--support-relative': None, '--support-all-views': False, '--conf-percentile': None, '--edge-jump': None, '--carve': False, '--entity-depth-run': None, '--analysis': None, '--floor-masks': None, '--metric-depth-run': None, '--camera-height': None, '--uncalibrated': False, '--dynamic-masks': None, '--video': None, '--floor-plane': None, '--base-scene': None, '--exclude-frames': []},
    "scripts/texture_fused_mesh.py": {'--self-check': False, '--droid-run': None, '--fused': None, '--output': None, '--scene': None, '--texture-width': 1280, '--texture-megapixels': 60, '--video': None, '--dynamic-masks': None, '--overlay-rows': None, '--min-facing': 0.2, '--nearly': 0.3, '--jpeg-quality': 90, '--heldout': None, '--every': 8},
    "scripts/fill_scene_holes.py": {'--self-check': False, '--droid-run': None, '--depth-run': None, '--shell': None, '--dynamic-masks': None, '--every': 2, '--step': 4, '--carve': False, '--carve-tolerance': 0.08, '--video': None, '--overlay-rows': None, '--texture-megapixels': 30, '--jpeg-quality': 90, '--output': None},
    "scripts/build_video_object_map.py": {'--self-check': False, '--droid-run': None, '--depth-run': None, '--support': None, '--masks': None, '--floor': None, '--dynamic-masks': None, '--method': 'overlap', '--output': None},
    "scripts/name_video_entities.py": {'--object-map': None, '--masks': None, '--output': None, '--per-request': 10, '--names': None, '--namer': 'gemini'},
    "scripts/project_entities_to_frames.py": {'--self-check': False, '--droid-run': None, '--scene': None, '--object-map': None, '--mesh': None, '--output': None, '--merge-analysis': None},
    "modal_apps/video_events.py": {'--self-check': False, '--video': None, '--marks': None, '--names': None, '--window': 12.0, '--fps': 2.0, '--width': 640, '--model': 'Qwen/Qwen3-VL-8B-Instruct', '--seed': 0, '--temperature': 0.0, '--max-new-tokens': 900, '--output': None},
    "modal_apps/lingbot_room.py": {'--output': None, '--manifest': None, '--sample': None, '--stride': 3, '--run-id': None, '--video': None, '--frames': None},
    "scripts/lingbot_dense_map.py": {'--run-id': None, '--droid-run': None, '--clip': None, '--masks': None, '--output': None, '--mesh': None, '--surface': None, '--points': None, '--depth-run': None, '--conf': 1.06, '--fill-conf': 1.0, '--sample': 3, '--cell': 0.0075, '--tolerance': 0.04, '--tolerance-floor': 0.015, '--max-points': 4000000, '--exclude-frames': [], '--overlay-rows': (646, 706)},
    "scripts/lingbot_icp_refine.py": {},
    "scripts/infer_room_floor.py": {'--self-check': False, '--fused': None, '--dense': None, '--droid-run': None, '--skip-frames': [], '--legacy-convex': False, '--output': None},
    "modal_apps/splat_train.py": {'--self-check': False, '--clip': '$ART/data/clips/me340-165', '--droid-run': '$ART/runs/droid-me340-165-171', '--masks': '$ART/runs/me340-dynamic-masks-188/masks', '--mesh': '$ART/runs/da3-posed-me340-189-fused-dynamic/mono-anchored-mesh.ply', '--fill': '$ART/runs/me340-filled-213/textured-scene.glb', '--scale': '$ART/runs/da3-posed-me340-189-fused-dynamic/metric-scale.json', '--captions': (648, 704, 160, 1120), '--steps': 30000, '--cap': 1500000, '--pose': False, '--exposure': False, '--ablation': False, '--max-elongation': None, '--depth-weight': 0, '--lingbot': None, '--seed-voxel': 0.005, '--seed-scale': 0.5, '--compare': None, '--max-minutes': 40, '--resume': None, '--clean': None, '--pick': None, '--skip': [], '--output': None},
    "scripts/register_cut_shot.py": {'--self-check': False, '--droid-run': None, '--depth-run': None, '--clip': None, '--output': None, '--mesh': None, '--registration': None, '--scene': None, '--analysis': None, '--merge': [], '--person-tracks': [], '--shot': '14:226', '--cut-frames': [], '--walk-count': 14, '--model': 'depth-anything/DA3-GIANT-1.1', '--invoke': False},
    "scripts/complete_video_objects.py": {'--self-check': False, '--droid-run': None, '--depth-run': None, '--object-map': None, '--masks': None, '--dynamic-masks': None, '--clip': None, '--output': None, '--entities': [], '--exclude': [], '--count': 10, '--all': False, '--generator': 'recgen', '--reassess': False, '--invoke': False, '--function-id': 'fu-Hh2leT3x1kprDaWpWsZ09l', '--max-usd': 30.0, '--skip-frames': [], '--voxel-native': 0.014, '--no-captions': False, '--workers': 1},
    "scripts/box_free_space.py": {'--box': None, '--study': None, '--output': None, '--self-check': False},
    "scripts/merge_object_models.py": {'--recgen': None, '--sam3d': None, '--box': None, '--box-test': None, '--review': None, '--output': None, '--against': None, '--self-check': False},
    "scripts/import_video_scene.py": {'--droid-run': None, '--depth-run': None, '--object-map': None, '--masks': None, '--policy': None, '--models': None, '--video': None, '--exclude-frames': None, '--dense-points': None, '--splats': None, '--full-video': None, '--analysis': None, '--inferred-floor': None, '--shell-glb': None, '--comparison-video': None, '--republish': None, '--dynamic-scene': None, '--video-events': None, '--skeleton-scene': None, '--dynamic-analysis': None, '--title': 'Video workcell (imported, not accepted)', '--output-dir': '.platform/imports', '--request-suffix': '1', '--lens': None},
}
PENDING_FLAGS = {}  # flags another part adds later (the AST test accepts either state); Part C's --lens and --namer have landed

# span flags and their syntax: colon START:END and pair "START END" are end-exclusive, dash A-B inclusive
SPANS = {("scripts/prepare_video_clip.py", "--frames"): "colon", ("modal_apps/droid_room.py", "--frames"): "colon",
         ("modal_apps/motion_masks.py", "--exclude-frames"): "colon", ("modal_apps/sam3_motion_tracks.py", "--frames"): "pair",
         ("modal_apps/mono_room.py", "--exclude-frames"): "colon", ("modal_apps/lingbot_room.py", "--frames"): "colon",
         ("scripts/lingbot_dense_map.py", "--exclude-frames"): "colon", ("scripts/lingbot_dense_map.py", "--overlay-rows"): "colon",
         ("scripts/texture_fused_mesh.py", "--overlay-rows"): "colon", ("scripts/fill_scene_holes.py", "--overlay-rows"): "colon",
         ("scripts/infer_room_floor.py", "--skip-frames"): "dash", ("modal_apps/splat_train.py", "--skip"): "dash",
         ("scripts/register_cut_shot.py", "--shot"): "colon", ("scripts/complete_video_objects.py", "--skip-frames"): "dash",
         ("scripts/import_video_scene.py", "--exclude-frames"): "colon"}
# not in a key: where outputs go, paid-call switches, budgets, run ids, execution knobs (store.KEY_DROP_* plus --workers, --name,
# --output-dir); NOTE_FLAGS keep 'id' and drop '=free text'
KEY_DROPPED = {"--output", "--invoke", "--run-id", "--reuse-build-from", "--max-usd", "--max-minutes", "--republish", "--request-suffix",
               "--workers", "--name", "--output-dir", "--self-check"}
NOTE_FLAGS = {"--entities", "--exclude"}
CLIP_NAMED = {("modal_apps/droid_room.py", "--clip")}  # the tool reads a clip folder as its clip.json name (so do we)
# a mode (a flag, or a positional subcommand) that reads only some flags: the others are not compared. splat_train --pick reads
# the run, the rule and the clip's fps; --clean also the masks, the clip video and the report surfaces; lingbot_dense_map build
# and diagnose never read --mesh/--surface/--points (evaluate does)
_LINGBOT_BUILD = {"--run-id", "--droid-run", "--clip", "--masks", "--output", "--depth-run", "--conf", "--fill-conf", "--sample", "--cell",
                  "--tolerance", "--tolerance-floor", "--max-points", "--exclude-frames", "--overlay-rows"}
MODE_READS = {("modal_apps/splat_train.py", "--pick"): {"--clean", "--pick", "--clip"},
              ("modal_apps/splat_train.py", "--clean"): {"--clean", "--clip", "--masks", "--mesh", "--fill"},
              ("scripts/lingbot_dense_map.py", "build"): _LINGBOT_BUILD, ("scripts/lingbot_dense_map.py", "diagnose"): _LINGBOT_BUILD}


# ---------------------------------------------------------------- spans and normalize()
def to_span(text, style):
    """'14:226' (colon) or '14-225' (dash, inclusive) -> [14, 226]."""
    a, b = (int(v) for v in str(text).split(":" if style == "colon" else "-"))
    return [a, b + 1] if style == "dash" else [a, b]


def fmt_span(span, style):
    a, b = span
    return f"{a}-{b - 1}" if style == "dash" else f"{a}:{b}"


def _number(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return int(v) if v.is_integer() else v


def art():
    """$ART as modal_apps/droid_room.ART sets it, read from the source (never importing Modal)."""
    return Path(re.search(r'^ART = Path\("([^"]+)"\)', (REPO / "modal_apps/droid_room.py").read_text(), re.M).group(1))


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _canon_path(text, owner_of):
    path = Path(text.replace("$ART", str(art())).replace("$REPO", str(REPO)))
    owner = owner_of(path)
    if owner:
        return {"in": owner[0], "role": owner[1]}
    if path.is_file():
        return {"leaf": _sha(path)}
    if path.is_dir():
        return {"leaf": hashlib.sha256(json.dumps([[str(p.relative_to(path)), _sha(p)] for p in sorted(path.rglob("*")) if p.is_file()]).encode()).hexdigest()}
    return {"missing": str(path)}


def _canon(value, owner_of):
    if isinstance(value, (bool, type(None))):
        return value
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else float(value)
    if isinstance(value, dict):  # already canonical (a path's owner)
        return value
    text = str(value)
    number = _number(text)
    if number is not None:
        return number
    if text.startswith(("/", "$ART", "$REPO")):
        return _canon_path(text, owner_of)
    return text


def script_of(command):
    """(script, argv) of a stage command: repo-relative script path, or 'report_runner.decide' for python -m."""
    words = list(command)
    if words[1] == "-m":
        return words[2], words[3:]
    script = words[1].replace("$REPO/", "").replace(str(REPO) + "/", "")
    return script, words[2:]


def normalize(script, argv, owner_of):
    """One tool command in canonical form, to compare a runner argv with a delivered one (P1; the store keys with its own _norm).

    $ART/$REPO expanded; spans -> [a, b); every option present with its default filled in (None dropped); key-dropped flags
    and the free text of --entities/--exclude removed; a path -> {"in": node, "role": role} by owner_of (symlinks are the
    caller's to resolve), else a leaf's sha256; flags sorted, value order kept. An unknown flag raises."""
    script = script.replace("$REPO/", "").replace(str(REPO) + "/", "")
    if script == "report_runner.decide":
        rule, words = argv[0], list(argv[1:])
        roles = {}
        for n, word in enumerate(words):
            if word != "--out" and (n == 0 or words[n - 1] != "--out") and "=" in word:
                role, value = word.split("=", 1)
                roles[role] = _canon(value, owner_of)
        return {"script": script, "args": [rule], "flags": dict(sorted(roles.items()))}
    defaults = {**DEFAULTS[script], **PENDING_FLAGS.get(script, {})}
    positional, given, flag = [], {}, None
    for word in argv:
        if word.startswith("--") and _number(word) is None:
            if word not in defaults:
                raise ValueError(f"{script}: unknown flag {word}")
            flag, given[word] = word, []  # a repeated flag: argparse keeps the last
        elif flag is None:
            positional.append(_canon(word, owner_of))
        else:
            named = Path(word.replace("$ART", str(art()))) / "clip.json" if (script, flag) in CLIP_NAMED else None
            given[flag].append(json.loads(named.read_text())["name"] if named and named.is_file() else word)
    flags = {}
    for name, default in defaults.items():
        if name in KEY_DROPPED:
            continue
        style = SPANS.get((script, name))
        if name in given:
            words = given[name]
            if default is False and not words:
                flags[name] = True
                continue
            if style == "pair":
                words = [f"{words[i]}:{words[i + 1]}" for i in range(0, len(words), 2)]
            values = [to_span(w, "colon" if style == "pair" else style) if style else _canon(w.split("=", 1)[0] if name in NOTE_FLAGS else w, owner_of)
                      for w in words]
        elif default is None:
            continue
        elif isinstance(default, bool):
            flags[name] = default
            continue
        elif isinstance(default, (list, tuple)) and style:  # a pair default of a colon flag is one span
            values = [list(default)] if default and not isinstance(default[0], (list, tuple, str)) else [to_span(d, style) for d in default]
        elif isinstance(default, (list, tuple)):
            values = [_canon(d, owner_of) for d in default]
        else:
            values = [_canon(default, owner_of)]
        flags[name] = values
    mode = next((reads for (tool, flag), reads in MODE_READS.items() if tool == script and (flag in given or flag in positional)), None)
    if mode:
        flags = {k: v for k, v in flags.items() if k in mode}
    return {"script": script, "args": positional, "flags": dict(sorted(flags.items()))}


def normalize_command(command, owner_of):
    return normalize(*script_of(command), owner_of)


# ---------------------------------------------------------------- dependencies and versions
@functools.lru_cache(maxsize=None)
def deps_of(script):
    """The script and every repo module it imports, transitively (scripts/ and modal_apps/ are on each other's path)."""
    seen, todo = set(), [script]
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        if not (REPO / rel).is_file():
            continue  # kept: a missing dependency must show in the lock, never vanish
        here = Path(rel).parent
        for node in ast.walk(ast.parse((REPO / rel).read_text())):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = ".".join(here.parts[:len(here.parts) - node.level + 1]) if node.level else ""
                module = ".".join(p for p in (base, node.module or "") if p)
                names = [module] + [f"{module}.{a.name}" for a in node.names]
            else:
                continue
            todo += [c for name in names for c in _module_files(name) if c not in seen]
    return tuple(sorted(seen))


def _module_files(name):
    """The repo files an import of `name` may load (scripts/ and modal_apps/ are on each other's path)."""
    path = name.replace(".", "/")
    return [c for folder in ("", "scripts/", "modal_apps/") for c in (f"{folder}{path}.py", f"{folder}{path}/__init__.py") if (REPO / c).is_file()]


@functools.lru_cache(maxsize=None)
def rule_code(rule, text=None):
    """(deps, sha256) of one decision rule. sha256 covers the source of its function in decide.py and of every module-level
    name that reaches (the stage entry point `decide` included, the version table not), plus the repo modules those import
    and the scripts they run by name (dense_gate: lingbot_icp_refine.py), transitively. deps: decide.py and those files.
    The sha is in the decision's key (research, commercial); tests/versions.json guards a bump of decide.VERSIONS."""
    text = text or (REPO / DECIDE).read_text()
    top = {}
    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            top[node.name] = node
        elif isinstance(node, ast.Assign):
            top.update({n.id: node for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)})
    seen, todo, files = set(), [rule, "decide"], set()
    while todo:
        name = todo.pop()
        if name in seen or name not in top or name in ("VERSIONS", "RULES"):
            continue
        seen.add(name)
        for n in ast.walk(top[name]):
            if isinstance(n, ast.Name):
                todo.append(n.id)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                files |= {f for m in ([a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]) for f in _module_files(m)}
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.endswith(".py"):  # a script it runs
                files |= set(_module_files(n.value[:-3]))
    external = sorted({d for f in files for d in deps_of(f)} - {DECIDE})
    segments = sorted({ast.get_source_segment(text, top[n]) for n in seen})
    return (DECIDE, *external), hashlib.sha256(json.dumps([segments, deps_sha(external)]).encode()).hexdigest()


def deps_sha(deps):
    """store's lock depsSha256: sha256 of the canonical [[path, file sha256], ...]."""
    rows = [[p, _sha(REPO / p) if (REPO / p).is_file() else None] for p in sorted(deps)]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def current_versions():
    """{kind: {version, depsSha256}} of every script kind and, as 'rule:NAME', of every decision rule (decide.VERSIONS)."""
    from .decide import VERSIONS as RULE_VERSIONS
    rows = {kind: {"version": VERSIONS[kind], "depsSha256": deps_sha(deps_of(script) if script else ())} for kind, script in sorted(SCRIPT.items())}
    return rows | {f"rule:{name}": {"version": v, "depsSha256": rule_code(name)[1]} for name, v in sorted(RULE_VERSIONS.items())}


def check_versions(recorded):
    """Problems with tests/versions.json: a kind whose code changed while its version did not is the one that matters."""
    problems, now = [], current_versions()
    for kind, row in now.items():
        old = recorded.get(kind)
        table = f"decide.VERSIONS['{kind[5:]}']" if kind.startswith("rule:") else f"VERSIONS['{kind}']"
        if old is None:
            problems.append(f"{kind}: not in versions.json; run python -m report_runner.stages --versions")
        elif old["depsSha256"] != row["depsSha256"] and old["version"] == row["version"]:
            problems.append(f"{kind}: its code changed but {table} is still {row['version']}; bump it, then --versions")
        elif old != row:
            problems.append(f"{kind}: versions.json is stale; run python -m report_runner.stages --versions")
    problems += [f"{kind}: in versions.json but not a stage kind" for kind in sorted(set(recorded) - set(now))]
    return problems


# ---------------------------------------------------------------- graph
def kind_of(name):
    return name.split("-")[0]


def _num(x):
    """A float as argv: shortest exact form, no trailing .0 (165.0 -> '165', 0.01399 -> '0.01399')."""
    text = repr(float(x))
    return text[:-2] if text.endswith(".0") else text


def _mmss(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _field(obj, name, default=None):
    value = obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    return default if value is None else value


def profile_of(ctx):
    """Part C's Profile for ctx.profile (a PROFILES key); tests may pass the object itself."""
    if isinstance(ctx.profile, str):
        return importlib.import_module("report_runner.profiles").PROFILES[ctx.profile]
    return ctx.profile


def pins(profile, *roles):
    """(role, hf_id, revision | 'unpinned', weights sha256 | None) of the profile's models (profiles.Profile.models rows
    (hf_id, revision, weights_sha256, licence, commercial)); a role the profile does not list has no pin."""
    rows = _field(profile, "models", {})
    return tuple((role, rows[role][0], rows[role][1] or "unpinned", rows[role][2]) for role in roles if role in rows)


def model_id(profile, role):
    """The model an argv names: the profile's hf_id without its licence-table suffix ('DA3-BASE#camera' is DA3-BASE)."""
    rows = _field(profile, "models", {})
    return rows[role][0].split("#")[0] if role in rows else MODEL_IDS[role]


def delivered(ctx):
    return ctx.profile == "delivered" or _field(ctx.profile, "name") == "delivered"


def previous_import(ctx):
    """The site's last imports.jsonl row: its record path (path only; the record holds a capability and is never opened) and
    the published title, which a republish keeps (O1: titles are operator data). Only the delivered profile or an explicit
    --republish republishes; any other run imports a new report of its own, never a new version of a delivered one."""
    if not (delivered(ctx) or getattr(ctx, "republish", False)):
        return {}
    state = getattr(ctx.store, "state", None)  # the store's state folder ($ART/runs/report-runner); no store, no history
    index = Path(state) / "imports.jsonl" if state else None
    rows = [json.loads(line) for line in index.read_text().splitlines() if line.strip()] if index and index.exists() else []
    return next((r for r in reversed(rows) if r.get("site") == ctx.site and r.get("importRecordPath")), {})


REQUIRED = object()


class _Graph:
    def __init__(self, ctx, profile):
        self.ctx, self.profile, self.specs, self.by, self.decisions = ctx, profile, [], {}, set()

    def need(self, name, absent=REQUIRED):
        """A decision's value. The runner marks a decision {'absent': why} when its stage failed, was blocked or refused:
        an optional layer then gets `absent` (and is left blank); a decision without one stops the graph."""
        found = self.ctx.decisions.get(name)
        if found is None:
            raise Pending(name, self.specs)
        if found.get("absent"):
            if absent is REQUIRED:
                raise RuntimeError(f"{self.ctx.site}: decision {name} could not be made ({found['absent']}), and the report needs it")
            return absent
        return found["value"]

    def add(self, name, commands, outputs, *, link=None, copy=None, optional=(), droid_frames=None, lingbot_source=None,
            leaves=None, env=None, gpu=None, rate=None, compute="cpu", timeout_s=3600, est_usd=0., est_s=60., worst_usd=None,
            budget_flags=None, models=(), rules=None, cwd=None, deps=None):
        """A StageSpec whose inputs are read off its tokens: role = producer name ('?' when optional); a producer consumed
        as droid frames or LingBot source, a decision's value (when only its JSON is read), else the referenced output
        roles (a bare @stage: all; static_filter's and dense_gate's folders are read, so their bytes count)."""
        refs = {}
        for text in [t for c in commands for t in c] + list((link or {}).values()) + list((copy or {}).values()) + [cwd or ""]:
            for producer, role in TOKEN.findall(text):
                if producer in RESERVED and not role:
                    continue
                spec = self.by.get(producer)
                if spec is None or (role and role not in spec.outputs):
                    raise ValueError(f"{name}: @{producer}:{role} names no declared output")
                refs.setdefault(producer, set()).add(role or None)
        inputs, consumes = {}, {}
        for producer, roles in refs.items():
            key = producer + ("?" if producer in optional else "")
            mode = "droid-frames" if producer == droid_frames else "lingbot-source" if producer == lingbot_source else \
                   "decision-value" if producer in self.decisions and roles == {"decision"} else "outputs"
            whole = None in roles or mode in ("droid-frames", "decision-value")
            inputs[key] = (producer, () if whole else tuple(sorted(roles)))
            if mode != "outputs":
                consumes[key] = mode
        kind = kind_of(name)
        rate = rate or gpu  # USD_PER_S key of what the stage pays for: its GPU, or a Modal CPU container
        paid = rate is not None or compute == "cloud"
        spec = StageSpec(name=name, site=self.ctx.site, version=(ADOPTED if delivered(self.ctx) else VERSIONS).get(kind, 1),
                         deps=deps if deps is not None else deps_of(SCRIPT[kind]) if SCRIPT.get(kind) else (),
                         commands=tuple(tuple(c) for c in commands), inputs=inputs, consumes=consumes,
                         leaves={k: Path(v) for k, v in (leaves or {}).items()}, outputs=dict(outputs),
                         stage_from={k: dict(v) for k, v in (("link", link), ("copy", copy)) if v}, compute=compute, gpu=gpu,
                         timeout_s=timeout_s, worst_usd=worst_usd if worst_usd is not None else round(timeout_s * USD_PER_S.get(rate or "", 0), 2),
                         est_usd=est_usd, est_s=est_s, budget_flags=budget_flags or {}, models=tuple(models), rules=rules or {},
                         env={**ENV, **(env or {})}, cwd=cwd, paid=paid)
        self.specs.append(spec)
        self.by[name] = spec
        return spec

    def decide(self, name, rule, extra=None, link=None, leaves=None, **roles):
        """A decision stage: python -m report_runner.decide RULE --out @new role=... -> @new/RULE.json (role 'decision')."""
        versions = _field(self.profile, "rules", {})  # {decision: 'name@v'}, or 'adopted' for the delivered profile
        deps, code = rule_code(rule)
        # a rule that runs keys its version and its code; an adopted value (delivered) was never computed by this code
        rules = {rule: f"{versions.get(rule, f'{rule}@unversioned')}#{code[:12]}" if isinstance(versions, dict) else f"{rule}@{versions}"}
        cmd = [PY, "-m", "report_runner.decide", rule, "--out", "@new", *[f"{r}={v}" for r, v in roles.items() if v is not None]]
        spec = self.add(name, [cmd], {"decision": f"{rule}.json", **(extra or {})}, link=link, leaves=leaves, rules=rules,
                        deps=deps, env={"PYTHONPATH": str(REPO / "scripts")}, est_s=30)
        self.decisions.add(name)
        return spec

    def seeded(self, spec, match):
        """D17: point a generator stage at the newest run of the same stage that consumed the same bytes for `match`
        (Store.latest); complete_video_objects copies that run's journals, so no paid call is made twice. The environment
        is not in a key, so the seed never changes whether the stage is a hit. ponytail: Store has no public digests(),
        _key/_status are used; nothing to match before the producers are hits in this Store."""
        store = self.ctx.store
        if store is None:
            return spec
        try:
            _, digests = store._key(spec, store._status(spec))
        except Exception:
            return spec
        hit = store.latest(spec.name, {r: digests[r] for r in match if r in digests})
        if hit is None:
            return spec
        seeded = dataclasses.replace(spec, env={**spec.env, "PANOPTES_SEED_JOURNAL": str(Path(hit.dir) / "out")})
        self.specs[self.specs.index(spec)] = self.by[spec.name] = seeded
        return seeded


CLIP_OUT = {"out": "out", "clip_json": "out/clip.json", "rgb_txt": "out/rgb.txt", "rgb": "out/rgb", "source_rgb": "out/source-rgb.mp4",
            "source_full": "out/source-full.mp4", "source_full_json": "out/source-full.json"}
DROID_OUT = {"out": "out", "prediction": "out/prediction.npz", "manifest": "out/input-manifest.json", "run": "out/run.json"}
TRACKS_OUT = {"out": "out", "tracks": "out/tracks.json", "npz": "out/tracks.npz"}
ANALYSIS_OUT = {"out": "out", "analysis": "out/analysis.json", "masks": "out/masks"}


def S(script):
    return f"$REPO/scripts/{script}"


def M(script):
    return f"$REPO/modal_apps/{script}"


def graph(ctx):
    """The M2 stage graph for one clip (runner design section 4). The last stage is the import."""
    profile = profile_of(ctx)
    g = _Graph(ctx, profile)
    site, generators, omit = ctx.site, tuple(_field(profile, "generators", ())), set(_field(profile, "omit", ()))
    caps = {**CAPS, **dict(_field(profile, "caps", {}))}
    depth_model, register_model = model_id(profile, "depth"), model_id(profile, "register")
    review = Path(ctx.review) / f"{site}.json" if ctx.review else None
    review = review if review and review.is_file() else None  # no review of this site: no box is approved (blank, never guessed)
    review_json = json.loads(review.read_text()) if review else {}
    review_leaf = {"review": review} if review else None

    # R01-R04: the clip, its cuts, a whole-clip DROID census, the mapped shot (D1, D2)
    clip = [PY, S("prepare_video_clip.py")]
    g.add("source", [[*clip, "--video", str(ctx.video), "--start", _num(ctx.start), "--end", _num(ctx.end), "--output", "@clip"],
                     [*clip, "--full-video", "@clip"], ["ln", "-s", "@clip", "@new/out"]], CLIP_OUT, leaves={"video": ctx.video},
          gpu="L4", compute="modal", timeout_s=900, est_usd=.01, est_s=240, models=pins(profile, "lens"))
    g.add("cuts", [[PY, S("detect_shot_cuts.py"), "--clip", "@source:out", "--output", "@new/segments.json"]], {"segments": "segments.json"}, est_s=30)
    droid = [PY, M("droid_room.py"), "execute"]
    g.add("census", [[*droid, "--clip", "@source:out", "--run-id", f"{site}-census-@key", "--output", "@new/out", "--reuse-build-from", DROID_BUILD]],
          DROID_OUT, droid_frames="source", gpu="A100-40GB", compute="modal", timeout_s=900, est_usd=.03, est_s=120, models=pins(profile, "camera"))
    g.decide("shots", "shots", segments="@cuts:segments", census="@census:out")
    shots = g.need("shots")
    if not shots["mapped"]:
        raise RuntimeError(f"{site}: no shot has >= 60 frames and >= 8 census keyframes; import_video_scene needs a mapped shot")
    (a, b), others = shots["primary"], [tuple(s) for s in shots["others"]]
    registered = {tuple(s) for s in shots.get("registered", others)}  # adopted values name the shots the delivered report registered (D5)
    n_clip = max(e for _, e in [(a, b), *others])

    # cheap decisions on the clip and the shots (one batch): overlay band (D6), lens (D4), frame lists (D12, D18), windows (D10), stride (D7)
    g.decide("overlay", "overlay", clip="@source:out")
    g.add("moge", [[*clip, "--fov-of", "@source:out", "--frames", fmt_span((a, b), "colon"), "--output", "@new/out"]], {"out": "out", "fov": "out/fov.json"},
          gpu="L4", compute="modal", timeout_s=900, est_usd=.02, est_s=120, models=pins(profile, "lens"))
    g.decide("lens", "lens", shots="@shots:decision", clip="@source:out", moge="@moge:fov")
    g.decide("sam2_frames", "sam2_frames", census="@census:out", segments="@cuts:segments")
    # O1: the operator's floor-mask frames (REVIEW/<site>.floor-frames.json) enter the key by their bytes; the rule uses them
    # only on the clip they name, so they never reach another window or video of this site
    operator = Path(ctx.review) / f"{site}.floor-frames.json" if ctx.review else None
    operator = operator if operator and operator.is_file() else None
    g.decide("floor_frames", "floor_frames", leaves={"operator": operator} if operator else None, shots="@shots:decision",
             **({"operator": str(operator), "clip": "@source:out"} if operator else {}))
    g.decide("track_windows", "track_windows", shots="@shots:decision")
    dense_map = bool(_field(profile, "dense_map", False))
    if dense_map:
        g.decide("lingbot_stride", "lingbot_stride", shots="@shots:decision")
    lens, overlay, sam2_frames, floor_frames, windows = (g.need(n) for n in ("lens", "overlay", "sam2_frames", "floor_frames", "track_windows"))
    refused = set(overlay["refuse"])  # two caption bands: those layers stay blank
    dense_map = dense_map and "lingbot" not in refused

    # R06b derived clip when the shot's lens differs; the camera clip holds frames [0, n_cam)
    cam, n_cam = "source", n_clip
    if not lens["keep"]:
        g.add("lens_clip", [[*clip, "--derive", "@source:out", "--frames", f"0:{b}", "--fov-deg", repr(float(lens["fov_deg"])), "--output", "@clip"],
                            ["ln", "-s", "@clip", "@new/out"]], CLIP_OUT, est_s=60)
        cam, n_cam = "lens_clip", b
    others_cam = [(s, min(e, n_cam)) for s, e in others if s < n_cam]  # other shots inside the camera clip: never a view there
    exclude = ["--exclude-frames", *[fmt_span(o, "colon") for o in others_cam]] if others_cam else []
    skip = [fmt_span(o, "dash") for o in others_cam]

    # R07 camera, R08-R10 masks, R11-R15 movers, R16-R17a depth and scale, voxel (D5)
    g.add("camera", [[*droid, "--clip", f"@{cam}:out", *([] if (a, b) == (0, n_cam) else ["--frames", fmt_span((a, b), "colon")]),
                      "--run-id", f"{site}-camera-@key", "--output", "@new/out", "--reuse-build-from", DROID_BUILD]],
          DROID_OUT, droid_frames=cam, gpu="A100-40GB", compute="modal", timeout_s=900, est_usd=.03, est_s=120, models=pins(profile, "camera"))
    g.add("floor_masks", [[PY, S("discover_video_keyframes.py"), "--video", "@source:source_rgb", "--frames", *map(str, floor_frames),
                           "--prompt", "floor", "--provider", "self-hosted", "--output", "@new/out"]], {"out": "out"},
          gpu="L4", compute="modal", timeout_s=900, est_usd=.02, est_s=120, models=pins(profile, "floor"))
    g.add("sam2", [[PY, M("sam2_everything.py"), "--video", "@source:source_rgb", "--frames", *map(str, sam2_frames), "--output", "@new/out"]],
          {"out": "out", "object_a": "out/object-a", "segment": "out/segment.json"}, gpu="L4", compute="modal", timeout_s=1800, est_usd=.17, est_s=750,
          models=pins(profile, "segment"))
    g.add("mask_root", [], {"out": "out", "floor_a": "out/floor-a", "object_a": "out/object-a"},
          link={"out/floor-a": "@floor_masks:out", "out/object-a": "@sam2:object_a"}, est_s=1)
    g.add("motion", [[PY, M("motion_masks.py"), "--droid-run", "@camera:out", "--output", "@new/out", *exclude]], {"out": "out", "motion": "out/motion.json"},
          gpu="L4", compute="modal", timeout_s=900, est_usd=.03, est_s=180)
    tracks = []
    for s, e in windows["shot"]:
        g.add(f"tracks-{s}-{e}", [[PY, M("sam3_motion_tracks.py"), "--droid-run", "@camera:out", "--motion", "@motion:out", "--frames", str(s), str(e),
                                   "--output", "@new/out"]], TRACKS_OUT, gpu="A100-40GB", compute="modal", timeout_s=900, est_usd=.18, est_s=400,
              models=pins(profile, "tracks"))
        tracks.append(f"@tracks-{s}-{e}:out")
    if len(tracks) > 1:
        g.add("stitch", [[PY, S("stitch_track_windows.py"), "--tracks", *tracks, "--output", "@new/out"]], {"out": "out", "stitched": "out/stitched.json"})
    g.add("analysis", [[PY, S("motion_tracks_to_analysis.py"), "--droid-run", "@camera:out", "--tracks", *tracks,
                        *(["--stitched", "@stitch:out"] if len(tracks) > 1 else []), "--output", "@new/out"]], ANALYSIS_OUT)
    g.add("dynamic_masks", [[PY, S("assemble_dynamic_masks.py"), "--droid-run", "@camera:out", "--analysis", "@analysis:analysis", "--output", "@new/out"]],
          {"out": "out", "masks": "out/masks"})
    mono = [PY, M("mono_room.py")]
    g.add("depth", [[*mono, "infer", "--droid-run", "@camera:out", "--output", "@new/out", "--da3-model", depth_model, "--every", "3", *exclude]],
          {"out": "out", "mono": "out/mono", "infer": "out/infer.json"}, gpu="A100-80GB", compute="modal", timeout_s=1200, est_usd=.08, est_s=240,
          models=pins(profile, "depth"))
    g.add("metric", [[*mono, "metric", "--droid-run", "@camera:out", "--output", "@new/out", "--floor-masks", "@floor_masks:out", "--camera-height", "1.6",
                      *exclude]], {"out": "out", "metric_scale": "out/metric-scale.json"}, link={"out/mono": "@depth:mono"},
          copy={"out/infer.json": "@depth:infer"}, est_s=120)
    g.decide("voxel", "voxel", metric="@metric:metric_scale")
    voxel = _num(g.need("voxel"))

    # R17b fuse, R18 moving layer, R21-R22b object map, names, static filter (D20); the lens gate on the mapped shot (D4)
    if "lens_gate" not in omit:
        g.decide("lens_gate", "lens", shots="@shots:decision", clip="@source:out", moge="@moge:fov", metric="@metric:metric_scale")
    g.add("fuse", [[*mono, "fuse", "--droid-run", "@camera:out", "--output", "@new/out", "--voxel-length-native", voxel, "--support-relative", "0.02",
                    "--support-all-views", "--edge-jump", "0.03", "--carve", "--dynamic-masks", "@dynamic_masks:masks", "--video", f"@{cam}:source_rgb",
                    "--floor-plane", "@metric:metric_scale"]],
          {"out": "out", "mesh": "out/mono-anchored-mesh.ply", "predicted": "out/predicted-scene.glb", "points": "out/supported-keyframe-points.glb",
           "scene": "out/scene.json", "metrics": "out/fuse-metrics.json", "metric_scale": "out/metric-scale.json", "support": "out/droid-support.npz"},
          link={"out/mono": "@depth:mono"}, copy={"out/infer.json": "@depth:infer", "out/metric-scale.json": "@metric:metric_scale"}, est_s=600)
    g.add("dynamic", [[*mono, "dynamic", "--droid-run", "@camera:out", "--output", "@new/out", "--analysis", "@analysis:analysis"]],
          {"out": "out", "scene": "out/scene.json", "surfaces": "out/dynamic"},
          link={"out/mono": "@depth:mono", "out/droid-support.npz": "@fuse:support"},
          copy={"out/metric-scale.json": "@fuse:metric_scale", "out/infer.json": "@depth:infer", "out/scene.json": "@fuse:scene"}, est_s=300)
    g.add("object_map", [[PY, S("build_video_object_map.py"), "--droid-run", "@camera:out", "--depth-run", "@fuse:out", "--masks", "@mask_root:out",
                          "--floor", "@fuse:metric_scale", "--dynamic-masks", "@dynamic_masks:masks", "--method", "overlap", "--output", "@new/out"]],
          {"out": "out", "object_map": "out/object-map.json", "surfaces": "out/surfaces"}, est_s=600)
    named = _field(profile, "namer") == "gemini"  # any other namer is gated (Part C): names and models stay blank, the import still runs
    if named:
        g.add("names", [[PY, S("name_video_entities.py"), "--object-map", "@object_map:out", "--masks", "@mask_root:out", "--per-request", "10",
                         "--output", "@new/out"]], {"out": "out", "object_map": "out/object-map.json", "names": "out/names.json", "surfaces": "out/surfaces"},
              compute="cloud", timeout_s=1800, worst_usd=.5, est_usd=.05, est_s=180, models=pins(profile, "names"))
    mapped = "names" if named else "object_map"
    filtered = f"@{mapped}:out"  # the object map every later stage reads
    if "static_filter" not in omit:
        g.decide("static_filter", "static_filter", extra={"object_map": "object-map.json", "surfaces": "surfaces"}, link={"surfaces": f"@{mapped}:surfaces"},
                 object_map=f"@{mapped}:out", masks="@mask_root:out", dynamic_masks="@dynamic_masks:masks", droid="@camera:out")
        g.need("static_filter")  # resolved first so the generator stages below can seed their journals (D17) from runs with the same inputs
        filtered = "@static_filter"

    # R19-R20 room shell, R23 outlines (D16), R24 events (D15), R30-R31a splat, R25-R26 LingBot, R32 registrations (D3), R34s SAM 3D
    if "texture" not in refused:
        g.add("texture", [[PY, S("texture_fused_mesh.py"), "--droid-run", "@camera:out", "--fused", "@fuse:out", "--video", f"@{cam}:source_full",
                           "--dynamic-masks", "@dynamic_masks:masks", *(["--overlay-rows", overlay["texture_rows"]] if overlay["texture_rows"] else []),
                           "--output", "@new/out"]], {"out": "out", "shell": "out/textured-scene.glb", "report": "out/texture-report.json"}, est_s=120)
    fill = "fill" not in refused and "texture" not in refused
    if fill:
        g.add("fill", [[PY, S("fill_scene_holes.py"), "--droid-run", "@camera:out", "--depth-run", "@fuse:out", "--shell", "@texture:out/textured-scene.glb",
                        "--dynamic-masks", "@dynamic_masks:masks", "--step", "2", "--carve", "--video", f"@{cam}:source_full",
                        *(["--overlay-rows", overlay["fill_rows"]] if overlay["fill_rows"] else []), "--output", "@new/out"]],
              {"out": "out", "shell": "out/textured-scene.glb", "fill": "out/fill.json"}, est_s=1800)
    g.add("outlines", [[PY, S("project_entities_to_frames.py"), "--droid-run", "@camera:out", "--scene", "@fuse:scene", "--object-map", filtered,
                        "--mesh", "@fill:shell" if fill else "@fuse:predicted", "--output", "@new/out"]], {"out": "out", "analysis": "out/analysis.json"}, est_s=300)
    g.add("events", [[PY, M("video_events.py"), "--video", "@source:source_rgb", "--marks", "@analysis:analysis", "--output", "@new/out"]],
          {"out": "out", "events": "out/events.json"}, gpu="A100-40GB", compute="modal", timeout_s=1200, est_usd=.1, est_s=300, models=pins(profile, "events"))
    splat = fill and "splat" not in refused
    if splat:
        captions = [str(v) for v in overlay["splat_captions"]]
        inputs = ["--clip", f"@{cam}:out", "--droid-run", "@camera:out", "--masks", "@dynamic_masks:masks", "--mesh", "@fuse:mesh", "--fill", "@fill:shell",
                  "--scale", "@fuse:metric_scale", "--captions", *captions]  # every ME340 default replaced, the --clean calls included
        minutes = float(caps["splat_minutes"])
        g.add("splat", [[PY, M("splat_train.py"), *inputs, "--steps", "60000", "--cap", "2500000", "--pose", "--max-elongation", "4",
                         *(["--skip", *skip] if skip else []), "--max-minutes", _num(minutes), "--output", "@new/out"]],
              {"out": "out", "splats": "out/splats.splat", "json": "out/splats.json", "train": "out/train.json", "launch": "out/launch.json",
               "cameras": "out/refined-cameras.npz"}, gpu="H100", compute="modal", timeout_s=int(minutes * 60) + 600, est_usd=1.4, est_s=minutes * 60,
              worst_usd=round((minutes * 60 + 600) * USD_PER_S["H100"], 2), budget_flags={"--max-minutes": ("minutes", USD_PER_S["H100"])},
              models=())
        trained = {f"out/{n}": f"@splat:{r}" for n, r in (("splats.splat", "splats"), ("splats.json", "json"), ("train.json", "train"),
                                                          ("launch.json", "launch"), ("refined-cameras.npz", "cameras"))}
        g.add("splat_clean", [[PY, M("splat_train.py"), "--clean", "@new/out", *inputs]], {"out": "out", "clean": "out/clean/clean.json"},
              link=trained, gpu="H100", compute="modal", timeout_s=1800, est_usd=.09, est_s=300)
        g.decide("splat_pick", "splat_pick", clean="@splat_clean:out")
    if dense_map:
        stride, run = g.need("lingbot_stride"), f"{site}-lingbot-@key"
        room = [PY, M("lingbot_room.py")]
        g.add("lingbot", [[*room, "prepare", "--video", "@source:source_full", "--run-id", run, "--stride", str(stride), "--frames", fmt_span((a, b), "colon"),
                           "--output", "@new/out"], [*room, "execute", "--run-id", run, "--output", "@new/out"]],
              {"out": "out", "plan": "out/plan.json", "submission": "out/submission.json"}, lingbot_source="source", gpu="H100", compute="modal",
              timeout_s=1800, est_usd=.1, est_s=400, models=pins(profile, "dense"))
        dense = ["--run-id", "@lingbot:out", "--droid-run", "@camera:out", "--clip", f"@{cam}:out", "--masks", "@dynamic_masks:masks", "--depth-run", "@fuse:out"]
        g.add("lingbot_diagnose", [[PY, S("lingbot_dense_map.py"), "diagnose", *dense, "--overlay-rows", overlay["lingbot_rows"], "--output", "@new"]],
              {"diagnose": "diagnose.json"}, rate="cpu16", compute="modal", timeout_s=1800, est_usd=.02, est_s=300)
        g.decide("lingbot_conf", "lingbot_conf", diagnose="@lingbot_diagnose:diagnose")
    for s, e in (o for o in others if o in registered):
        g.add(f"register-{s}-{e}", [[PY, S("register_cut_shot.py"), "--droid-run", "@camera:out", "--depth-run", "@fuse:out", "--clip", "@source:out",
                                     "--shot", fmt_span((s, e), "colon"), "--mesh", "@fuse:mesh", "--model", register_model, "--output", "@new/out", "--invoke"]],
              {"out": "out", "registration": "out/registration.json"}, gpu="A100-80GB", compute="modal", timeout_s=1200, est_usd=.05, est_s=240,
              models=pins(profile, "register"))
        g.decide(f"other_shot-{s}-{e}", "other_shot", registration=f"@register-{s}-{e}:registration")
    objects = named and "objects" not in refused and generators
    common = ["--droid-run", "@camera:out", "--depth-run", "@fuse:out", "--object-map", filtered, "--masks", "@mask_root:out",
              "--dynamic-masks", "@dynamic_masks:masks", "--clip", f"@{cam}:out", "--voxel-native", voxel, *(["--skip-frames", *skip] if skip else []),
              *(["--no-captions"] if overlay["objects_no_captions"] else [])]
    excluded = [f"{k}={v}" for k, v in sorted(review_json.get("entitiesExcluded", {}).items())]  # review excludes apply to every generator (D17)
    match = ("fuse", filtered[1:].split(":")[0], "mask_root")  # D17: the depth, object-map and masks this generator consumed
    learned = []

    def generator(name, flags, usd, est):
        paid = usd is not None
        spec = g.add(name, [[PY, S("complete_video_objects.py"), *common, "--generator", name, *flags, "--workers", str(1 if name in SERIAL_GENERATORS else WORKERS),
                             *(["--max-usd", _num(usd), "--invoke"] if paid else []), "--output", "@new/out"]],
                     {"out": "out", "manifest": "out/manifest.json", "models": "out/models"}, gpu="A100-80GB" if paid else None,
                     compute="modal" if paid else "cpu", timeout_s=int(usd / USD_PER_S["A100-80GB"]) + 3600 if paid else 7200,
                     worst_usd=usd or 0., est_usd=est, est_s=1800, budget_flags={"--max-usd": "usd"} if paid else None, models=pins(profile, name))
        g.seeded(spec, match)
        if paid:
            learned.append(name)

    if objects and "sam3d" in generators:
        generator("sam3d", ["--all", *(["--exclude", *excluded] if excluded else [])], float(caps.get("sam3d_usd", caps["generator_usd"])), 2.2)
        g.decide("generator_plan", "generator_plan", leaves=review_leaf, sam3d="@sam3d:out", review=str(review) if review else None)

    # second round: splat pick (D11), LingBot build, dense gate (D8), other-shot movers (D3b), RecGen (D17)
    if splat:
        rule = g.need("splat_pick", absent=None)
        splat = rule is not None  # no pick (the splat failed or was blocked): no splat layer
    if splat:
        g.add("splat_final", [[PY, M("splat_train.py"), "--clean", "@new/out", "--pick", rule, *inputs]],
              {"out": "out", "splats": "out/splats-clean.splat", "json": "out/splats-clean.json", "cameras": "out/refined-cameras.json"},
              link={f"out/{n}": f"@splat_clean:out/{n}" for n in ("splats.splat", "splats.json", "train.json", "launch.json", "refined-cameras.npz", "clean")},
              est_s=120)
    if dense_map:
        conf = g.need("lingbot_conf", absent=None)
        dense_map = conf is not None  # no confidence (the LingBot run or its diagnosis failed): no dense points
    if dense_map:
        g.add("lingbot_build", [[PY, S("lingbot_dense_map.py"), "build", *dense, "--mesh", "@fuse:mesh", "--conf", _num(conf),
                                 "--overlay-rows", overlay["lingbot_rows"], *exclude, "--output", "@new/out"]],
              {"out": "out", "points": "out/dense-points.glb", "info": "out/points.json", "attributes": "out/point-attributes.npz", "remote": "out/remote.json"},
              copy={"out/plan.json": "@lingbot:plan", "out/diagnose.json": "@lingbot_diagnose:diagnose"}, rate="cpu24", compute="modal",
              timeout_s=3600, est_usd=.05, est_s=900)
        g.decide("dense_gate", "dense_gate", map="@lingbot_build:out", fused="@fuse:out")
    moved, scene = [], "@dynamic:out"
    for i, (s, e) in enumerate(others):
        if (s, e) not in registered or not g.need(f"other_shot-{s}-{e}", absent={"accepted": False})["accepted"] or e > n_cam:  # refused, or outside the camera clip: blank
            continue
        wins = windows["others"][i] if i < len(windows["others"]) else []
        if not wins and not delivered(ctx):  # never the mapped shot's people in another shot: blank
            continue
        otracks, movers_analysis = [], "@analysis:analysis"  # delivered ME340 only (X6): its movers came from the mapped shot's analysis
        for ws, we in wins:
            g.add(f"otracks-{ws}-{we}", [[PY, M("sam3_motion_tracks.py"), "--droid-run", "@camera:out", "--frames", str(ws), str(we), "--text-only",
                                          "--output", "@new/out"]], TRACKS_OUT, gpu="A100-40GB", compute="modal", timeout_s=900, est_usd=.18, est_s=400,
                  models=pins(profile, "tracks"))
            otracks.append(f"@otracks-{ws}-{we}:out")
        if len(otracks) > 1:
            g.add(f"ostitch-{s}-{e}", [[PY, S("stitch_track_windows.py"), "--tracks", *otracks, "--output", "@new/out"]], {"out": "out", "stitched": "out/stitched.json"})
        if otracks:
            g.add(f"oanalysis-{s}-{e}", [[PY, S("motion_tracks_to_analysis.py"), "--droid-run", "@camera:out", "--tracks", *otracks,
                                          *(["--stitched", f"@ostitch-{s}-{e}:out"] if len(otracks) > 1 else []), "--output", "@new/out"]], ANALYSIS_OUT)
            movers_analysis = f"@oanalysis-{s}-{e}:analysis"
        g.add(f"movers-{s}-{e}", [[PY, S("register_cut_shot.py"), "--registration", f"@register-{s}-{e}:out", "--scene", scene,
                                   "--analysis", movers_analysis, "--mesh", "@fuse:mesh", "--droid-run", "@camera:out", "--depth-run", "@fuse:out",
                                   "--clip", "@source:out", "--model", register_model, "--output", "@new/out", "--invoke"]],
              {"out": "out", "scene": "out/scene/scene.json", "scene_dir": "out/scene", "analysis": "out/analysis/analysis.json"},
              gpu="A100-80GB", compute="modal", timeout_s=1200, est_usd=.05, est_s=300, models=pins(profile, "register"))
        scene = f"@movers-{s}-{e}:scene_dir"
        moved.append(f"movers-{s}-{e}")
    box_excludes, unplanned = None, set()  # unplanned: generators whose plan is absent (they failed); the merge goes on without them
    if objects and "sam3d" in generators:
        plan = g.need("generator_plan", absent=None)
        if plan is None:  # SAM 3D failed or was blocked: the box stage keeps the review excludes
            box_excludes, unplanned = dict(sorted(review_json.get("entitiesExcluded", {}).items())), {"sam3d"}
        elif "recgen" in generators and plan["recgen_entities"]:
            ids = plan["recgen_entities"]
            generator("recgen", ["--entities", *ids, "--count", str(len(ids)),
                                 *(["--exclude", *[f"{k}={v}" for k, v in plan["excluded_all"].items()]] if plan["excluded_all"] else [])],
                      float(caps.get("recgen_usd", caps["generator_usd"])), 5.)
            g.decide("generator_plan-box", "generator_plan", leaves=review_leaf, sam3d="@sam3d:out", recgen="@recgen:out",
                     review=str(review) if review else None)
        else:
            box_excludes = plan["box_excludes"]
    elif objects:
        box_excludes = dict(sorted(review_json.get("entitiesExcluded", {}).items()))

    # third round: inferred floor (D9), box (D17), box test, merge (D13); then the import (D14)
    dense_dir = None
    if dense_map:
        use = g.need("dense_gate", absent={"use": None})["use"]
        dense_dir = {"raw": "@lingbot_build:out", "icp": "@dense_gate/icp"}.get(use)
    g.add("floor_infer", [[PY, S("infer_room_floor.py"), "--fused", "@fuse:out", *(["--dense", dense_dir, "--droid-run", "@camera:out"] if dense_dir else []),
                           *(["--skip-frames", *skip] if skip else []), "--output", "@new/out"]], {"out": "out", "floor": "out/inferred-floor.json"}, est_s=300)
    g.decide("inferred_floor", "inferred_floor", floor="@floor_infer:floor", lens=None if "lens_gate" in omit else "@lens_gate:decision")
    if objects and box_excludes is None and "generator_plan-box" in g.by:
        after = g.need("generator_plan-box", absent=None)
        box_excludes, unplanned = (after or plan)["box_excludes"], unplanned | ({"recgen"} if after is None else set())
    if objects and "box" in generators:
        generator("box", ["--all", *(["--exclude", *[f"{k}={v}" for k, v in box_excludes.items()]] if box_excludes else [])], None, 0.)
        g.add("box_test", [[PY, S("box_free_space.py"), "--box", "@box:out", "--output", "@new/out"]], {"out": "out", "box_test": "out/box-test.json"}, est_s=300)
    if objects:
        runs = [f"--{name}" for name in ("recgen", "sam3d", "box") if name in g.by and name not in unplanned]
        g.add("merge", [[PY, S("merge_object_models.py"), *[w for flag in runs for w in (flag, f"@{flag[2:]}:out")],
                         *(["--box-test", "@box_test:box_test"] if "box" in g.by else []), *(["--review", str(review)] if review else []),
                         "--output", "@new/out"]], {"out": "out", "merge": "out/merge.json", "models": "out/models"}, leaves=review_leaf, est_s=120)
    floor_kept = g.need("inferred_floor", absent=False)
    last = moved[-1] if moved else None
    previous = previous_import(ctx)
    title = previous.get("title") or f"{site} {_mmss(ctx.start)}–{_mmss(ctx.end)} (imported, not accepted)"
    layers = [("--shell-glb", "@fill:out/textured-scene.glb", fill), ("--splats", "@splat_final:splats", splat),
              ("--dense-points", dense_dir, dense_dir), ("--inferred-floor", "@floor_infer:out", floor_kept), ("--models", "@merge:models", objects)]
    g.add("import", [[PY, S("import_video_scene.py"), "--droid-run", "@camera:out", "--depth-run", "@fuse:out", "--object-map", filtered,
                      "--masks", "@mask_root:out", "--video", "@source:source_rgb", "--full-video", "@source:out/source-full.json",
                      "--analysis", "@outlines:analysis", "--dynamic-scene", f"@{last}:scene" if last else "@dynamic:scene",
                      "--dynamic-analysis", f"@{last}:analysis" if last else "@analysis:analysis", "--video-events", "@events:events",
                      *[w for flag, token, on in layers if on for w in (flag, token)],
                      *(["--exclude-frames", *[fmt_span(o, "colon") for o in others]] if others else []), "--title", title,
                      *(["--republish", previous["importRecordPath"]] if previous else []), "--request-suffix", "@key",
                      *(["--lens", "@lens_gate:decision"] if "lens_gate" not in omit else [])]],
          {"log": "runner.log"}, optional={"fill", "splat_final", "lingbot_build", "dense_gate", "floor_infer", "merge", "events", *moved},
          cwd="$ART", timeout_s=2 * 3600, est_s=3600)
    return g.specs


def self_check():
    assert to_span("14:226", "colon") == to_span("14-225", "dash") == [14, 226] and fmt_span([14, 226], "dash") == "14-225"
    assert _num(165.) == "165" and _num(.01399) == "0.01399" and _num(55.3194223319173) == "55.3194223319173"
    assert kind_of("tracks-226-526") == "tracks" and kind_of("generator_plan-box") == "generator_plan"
    assert "scripts/lingbot_dense_map.py" in deps_of("scripts/lingbot_dense_map.py") and "modal_apps/lingbot_room.py" in deps_of("scripts/lingbot_dense_map.py")
    print("stages self-check passed: spans round-trip, argv numbers, stage kinds, import closure")


if __name__ == "__main__":
    if "--versions" in sys.argv:
        (REPO / "tests/versions.json").write_text(json.dumps(current_versions(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {REPO / 'tests/versions.json'}")
    else:
        self_check()
