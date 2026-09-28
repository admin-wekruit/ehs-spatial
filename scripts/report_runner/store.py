"""M2 report runner, Part A: a content-addressed stage store, a budgeted executor, and the one-command loop.

A stage's key hashes its name, version, key-normalised argv, the digests of what it consumes, its leaves, stage_from,
model pins and decision-rule versions. Consumers hash their producers' output bytes, never their keys, so a rebuilt
stage with byte-identical outputs leaves everything downstream a hit (early cutoff). Excluded from the key: --output,
--invoke, --run-id, --reuse-build-from, --republish, --request-suffix, --workers, budgets (--max-usd, --max-minutes, worst/est
USD) and the free text after 'id=' in --entities / --exclude.

State: $ART/runs/report-runner/keys.jsonl (append-only; adopted locks in adopted/locks/). A run gets a fresh
$ART/runs/<site>-<stage>-<key[:10]>/ holding its outputs, lock.json and runner.log (the tool's stdout, redacted); a
failed one is renamed ...-failed-<ts> and never indexed. Tests: pytest tests/test_report_runner_store.py
"""
import concurrent.futures as cf
from collections import Counter
import functools
from graphlib import TopologicalSorter
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import threading
import time

from .profiles import hosting
from .spec import Ctx, Hit, Pending

REPO = Path(__file__).resolve().parents[2]
# modal_apps/splat_train.USD_PER_S plus ehs_spatial.video.MODAL_L4_USD_PER_S (L4 + 1 core + 8 GiB); a test keeps them equal
PRICES = {"H100": .001097, "A100-80GB": .000694, "A100-40GB": .000583, "cpu_core": .0000131, "memory_gib": .00000222,
          "L4_1cpu_8gib": .000222 + .0000131 + 8 * .00000222}
REDACT = re.compile(r"capabilit|token|secret", re.I)
TOKEN = re.compile(r"(?<![\w.])@([A-Za-z][\w-]*)(?::([\w.-]+))?")
RESERVED = {"new", "key", "clip"}
KEY_DROP_VALUE = {"--output", "--run-id", "--reuse-build-from", "--max-usd", "--max-minutes", "--republish", "--request-suffix", "--workers"}
KEY_DROP_FLAG = {"--invoke"}
NOTE_FLAGS = {"--entities", "--exclude"}  # 'id=free text': the ids and their order stay in the key, the text does not
ABSENT = ("failed", "blocked", "refused")
MODES = {"outputs", "droid-frames", "lingbot-source", "decision-value"}
_HASHES, _HASH_LOCK = {}, threading.Lock()


class Unresolved(Exception):
    """A producer of this stage has no hit yet, so its key cannot be computed."""


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha(text):
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()


def say(text):
    for line in str(text).splitlines():
        if not REDACT.search(line):
            print(line, flush=True)


def file_sha(path, fresh=False):
    st = os.stat(path)
    cache_key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if fresh or cache_key not in _HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        with _HASH_LOCK:
            _HASHES[cache_key] = h.hexdigest()
    return _HASHES[cache_key]


def _files(root):
    """Files under root, links followed; a dangling link has no content and is left out."""
    return sorted(p for r, _, fs in os.walk(root, followlinks=True) for f in fs if (p := Path(r, f)).exists())


def content_sha(path, fresh=False):
    """A file's sha256; a directory's is over its sorted (relpath, file sha256) pairs, links followed."""
    p = Path(path)
    if p.is_dir():
        return sha(canonical([[f.relative_to(p).as_posix(), file_sha(f, fresh)] for f in _files(p)]))
    return file_sha(p, fresh)


def stat_sig(path):
    """(size, mtime_ns); a directory sums its files' sizes and takes the newest mtime of it and everything in it."""
    p = Path(path)
    if not p.is_dir():
        st = p.stat()
        return st.st_size, st.st_mtime_ns
    size, mtime = 0, p.stat().st_mtime_ns
    for root, _, files in os.walk(p, followlinks=True):
        mtime = max(mtime, os.stat(root).st_mtime_ns)
        for f in files:
            try:
                st = os.stat(os.path.join(root, f))
            except FileNotFoundError:  # a dangling link (e.g. ME340 305/scene/mono)
                continue
            size, mtime = size + st.st_size, max(mtime, st.st_mtime_ns)
    return size, mtime


def _frames(spec):
    """The [a, b) a stage's argv asks for with --frames A:B (or --frames A B); None is the whole clip."""
    for cmd in spec.commands:
        if "--frames" in cmd:
            i = cmd.index("--frames")
            if ":" in cmd[i + 1]:
                a, b = cmd[i + 1].split(":")
                return int(a), int(b)
            return int(cmd[i + 1]), int(cmd[i + 2])
    return None


def _refs(text):
    return {m.group(1) for m in TOKEN.finditer(text)}


def _absent(spec, status):
    return {p for role, (p, _) in spec.inputs.items() if role.endswith("?") and status.get(p) in ABSENT}


def _drop(cmd, absent):
    """cmd without the tokens naming an absent optional producer, nor a flag that is left without a value."""
    out, i = list(cmd), 0
    while i < len(out):
        if _refs(out[i]) & absent:
            del out[i]
            if i and out[i - 1].startswith("--") and (i == len(out) or out[i].startswith("--")):
                del out[i - 1]
                i -= 1
            continue
        i += 1
    return tuple(out)


def _symbolic(spec, status):
    absent = _absent(spec, status)
    commands = tuple(_drop(c, absent) for c in spec.commands)
    stage_from = {kind: {rel: ref for rel, ref in refs.items() if not _refs(ref) & absent} for kind, refs in spec.stage_from.items()}
    return commands, stage_from


def _over_budget(spec, ledger):
    if ledger.cap is None:
        return f"a paid stage (worst case ${spec.worst_usd:.2f}) and PANOPTES_PAID_BUDGET_USD is unset: no paid call"
    return f"worst case ${spec.worst_usd:.2f} does not fit the paid budget (${ledger.cap:.2f}, ${ledger.spent:.2f} booked)"


def booked(spec, reserved, wall_s):
    """What a finished paid stage books against the budget: its GPU's list price x wall seconds x the containers it keeps
    busy at once (--workers), never more than it reserved. A stage without a GPU price (a Modal CPU container, a cloud
    API) books its whole reservation. ponytail: wall time bounds billed container time; parse each tool's spend if it must be exact."""
    rate = {**PRICES, "L4": PRICES["L4_1cpu_8gib"]}.get(spec.gpu)
    if rate is None:
        return reserved
    workers = next((int(c[c.index("--workers") + 1]) for c in spec.commands if "--workers" in c), 1)
    return min(reserved, round(wall_s * rate * workers, 4))


class Ledger:
    """The run's paid budget (PANOPTES_PAID_BUDGET_USD). None: no paid call at all. A paid stage reserves its worst case
    before it starts (one with budget flags may instead be capped to what is left) and books what it cost when it ends
    (booked()). With a path, every booking is appended there and a re-run of the same command starts from their sum: the
    cap holds for the command, not for one process."""

    def __init__(self, cap_usd=None, path=None):
        self.cap, self.spent, self.reserved, self._lock, self.path = cap_usd, 0.0, {}, threading.Lock(), path
        if path and Path(path).is_file():
            self.spent = sum(json.loads(line)["usd"] for line in Path(path).read_text().splitlines() if line.strip())

    @classmethod
    def from_env(cls, path=None):
        value = os.environ.get("PANOPTES_PAID_BUDGET_USD")
        cap = float(value) if value is not None else None
        if cap is not None and not (math.isfinite(cap) and cap >= 0):
            raise ValueError("Invalid PANOPTES_PAID_BUDGET_USD")
        return cls(cap, path)

    def remaining(self):
        return self.cap - self.spent - sum(self.reserved.values())

    def reserve(self, spec):
        if not spec.paid:
            return True
        with self._lock:
            if self.cap is None:
                return False
            room = self.remaining()
            amount = spec.worst_usd if spec.worst_usd <= room else room if spec.budget_flags and room > 0 else None
            if amount is None:
                return False
            self.reserved[spec.name] = amount
            return True

    def settle(self, spec, usd):
        with self._lock:
            self.reserved.pop(spec.name, None)
            self.spent += usd
            if self.path:
                Path(self.path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.path, "a") as f:
                    f.write(json.dumps({"stage": spec.name, "site": spec.site, "usd": usd, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")


class Store:
    def __init__(self, art, scope=None, verify=False, repo=REPO, refuse=None):
        """refuse(spec) -> reason | None: the profile's refusals (profiles.refuse), checked before any lookup or run."""
        self.art, self.scope, self.verify, self.repo, self.refuse = Path(art), scope, verify, Path(repo), refuse
        self.state = self.art / "runs/report-runner"
        self.index = self.state / "keys.jsonl"
        self._resolved = {}  # (site, stage) -> (Hit, index entry)
        self._failed = set()  # keys that failed in this process: not run again by it (a paid failure is paid once)
        self._lock = threading.Lock()

    # ---- index -------------------------------------------------------------------------------------------------
    def _entries(self):
        with self._lock:
            if not self.index.exists():
                return []
            return [json.loads(line) for line in self.index.read_text().splitlines() if line.strip()]

    def _rel(self, path):
        try:
            return Path(path).relative_to(self.art).as_posix()
        except ValueError:
            return str(path)

    def _valid(self, entry, verify):
        d, outputs = self.art / entry["dir"], {}
        for role, (rel, size, mtime, digest) in entry["outputs"].items():
            if rel is None:  # an adopted run's output that is no longer on disk (recorded as absent, never served)
                continue
            try:
                ok = content_sha(d / rel, fresh=True) == digest if verify else stat_sig(d / rel) == (size, mtime)
            except OSError:
                return None
            if not ok:
                return None
            outputs[role] = d / rel
        return Hit(d, outputs, entry["verification"])

    def _find(self, spec, key, verify):
        for entry in reversed(self._entries()):
            if entry["key"] != key or (self.scope and self.scope not in entry["scope"]) or set(spec.outputs) - set(entry["outputs"]):
                continue
            hit = self._valid(entry, verify)
            if hit:
                self._resolved[(spec.site, spec.name)] = (hit, entry)
                return hit, entry
        return None

    # ---- keys --------------------------------------------------------------------------------------------------
    def _digest(self, spec, role, hit, entry):
        producer, roles = spec.inputs[role]
        roles = tuple(roles) or tuple(sorted(entry["outputs"]))
        shas = {r: entry["outputs"][r][3] for r in roles}
        mode = spec.consumes.get(role, "outputs")
        if mode == "outputs":
            return sha(canonical(sorted(shas.items())))
        if mode == "lingbot-source":  # the source bytes and the range, whatever the producer calls them
            return sha(canonical([sorted(shas.values()), _frames(spec)]))
        paths = [hit.outputs[r] for r in roles]
        if mode == "decision-value":
            return sha(canonical([json.loads(Path(p).read_text())["value"] for p in paths]))
        # droid-frames: the frames [a, b) in rgb.txt order, then K and D; frames outside the range and other files do not count
        clip = next(Path(p) for p in paths if Path(p).name == "clip.json")
        meta = json.loads(clip.read_text())
        names = [line.split()[1] for line in (clip.parent / "rgb.txt").read_text().splitlines() if line.strip() and not line.startswith("#")]
        a, b = _frames(spec) or (0, len(names))
        return sha(canonical([[file_sha(clip.parent / n) for n in names[a:b]], meta["K"], meta["D"]]))

    def _norm(self, spec, commands):
        leaves = {str(Path(p)): {"leaf": name} for name, p in spec.leaves.items()}
        out = []
        for cmd in commands:
            norm, skip, notes = [], False, False
            for tok in cmd:
                if skip:
                    skip = False
                    continue
                if tok in KEY_DROP_VALUE or tok in KEY_DROP_FLAG:
                    skip = tok in KEY_DROP_VALUE
                    continue
                if tok.startswith("--"):
                    notes = tok in NOTE_FLAGS
                elif notes:
                    tok = tok.split("=", 1)[0]
                full = self._expand(tok)
                role, eq, value = full.partition("=")
                if eq and not full.startswith("-") and value in leaves:  # a decision's role=PATH: the leaf's bytes, never its path
                    norm.append({role: leaves[value]})
                    continue
                norm.append(leaves.get(full) or full.replace(str(self.art), "$ART").replace(str(self.repo), "$REPO"))
            out.append(norm)
        return out

    def _key(self, spec, status):
        digests = {}
        for role, (producer, _) in sorted(spec.inputs.items()):
            st = status.get(producer)
            if isinstance(st, tuple):
                digests[role] = self._digest(spec, role, *st)
            elif not (role.endswith("?") and st in ABSENT):
                raise Unresolved(producer)
        commands, stage_from = _symbolic(spec, status)
        body = {"stage": spec.name, "version": spec.version, "argv": self._norm(spec, commands), "inputs": digests,
                "leaves": {n: content_sha(p) for n, p in sorted(spec.leaves.items())}, "stage_from": stage_from,
                "models": [list(m) for m in spec.models], "rules": spec.rules}
        return sha(canonical(body)), digests

    def _status(self, spec):
        return {p: self._resolved[(spec.site, p)] for p, _ in spec.inputs.values() if (spec.site, p) in self._resolved}

    def key(self, spec):
        """Raises Unresolved until every producer is a hit (lookup, record or execute) in this Store."""
        return self._key(spec, self._status(spec))[0]

    def lookup(self, spec, verify=False):
        found = self._find(spec, self.key(spec), verify or self.verify)
        return found[0] if found else None

    def latest(self, stage, match):
        """The newest valid run of `stage` whose consumed digests include `match` (generator journal seeding, D17)."""
        for entry in reversed(self._entries()):
            if entry["stage"] == stage and all(entry["inputs"].get(r) == d for r, d in match.items()):
                hit = self._valid(entry, self.verify)
                if hit:
                    return hit
        return None

    def record(self, spec, dir, verification, scope, lock, adopted_from=None):
        key, digests = self._key(spec, self._status(spec))
        return self._record(spec, key, digests, Path(dir), verification, scope, lock, adopted_from)

    def _record(self, spec, key, digests, d, verification, scope, lock, adopted_from=None):
        absent = {role for role, rel in spec.outputs.items() if rel is None and adopted_from}  # e.g. ME340 217's trained splats.splat: gone, its cleaned pick is kept
        for role, rel in spec.outputs.items():
            if role not in absent and not (d / rel).exists():
                raise FileNotFoundError(f"{spec.name}: output {role} is missing: {d / rel}")
        # the lock before the signatures: an output role '.' (the whole directory) is signed with its lock.json in it
        lock_path = self.state / "adopted/locks" / f"{key}.json" if adopted_from else d / "lock.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(dict(lock, key=key), indent=1, default=str))
        outputs = {role: [None, 0, 0, "absent"] if role in absent else [rel, *stat_sig(d / rel), content_sha(d / rel)] for role, rel in spec.outputs.items()}
        rel_dir = self._rel(d)  # several stages may share an adopted run's files (171 is census and camera); a run's own dir is fresh
        entry ={"key": key, "stage": spec.name, "site": spec.site, "dir": rel_dir, "inputs": digests, "outputs": outputs,
                 "outputDigest": sha(canonical(sorted((r, o[3]) for r, o in outputs.items()))), "verification": verification,
                 "scope": list(scope), "lock": self._rel(lock_path), "adoptedFrom": adopted_from, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with self._lock:
            self.index.parent.mkdir(parents=True, exist_ok=True)
            with self.index.open("a") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        self._resolved[(spec.site, spec.name)] = (Hit(d, {r: d / o[0] for r, o in outputs.items() if o[0] is not None}, verification), entry)
        return key

    # ---- running -----------------------------------------------------------------------------------------------
    def _expand(self, text):
        return text.replace("$ART", str(self.art)).replace("$REPO", str(self.repo))

    def clip_dir(self, spec, key):
        return self.art / "data/clips" / f"{spec.site}-{key[:10]}"

    def _sub(self, text, spec, status, d, key):
        def one(m):
            name, role = m.groups()
            if not role and name in RESERVED:
                return {"new": str(d), "key": key[:12], "clip": str(self.clip_dir(spec, key))}[name]
            hit = status[name][0]
            return str(hit.outputs[role] if role else hit.dir)
        return TOKEN.sub(one, self._expand(text))

    def _check(self, specs):
        by_name = {s.name: s for s in specs}
        if len(by_name) != len(specs) or RESERVED & set(by_name):
            raise ValueError("stage names must be unique and not new/key/clip")
        for s in specs:
            producers = {p for p, _ in s.inputs.values()}
            texts = [t for c in s.commands for t in c] + [r for refs in s.stage_from.values() for r in refs.values()] + [s.cwd or ""]
            if missing := producers - set(by_name):
                raise ValueError(f"{s.name}: producers {sorted(missing)} are not in the graph")
            if bad := set().union(*map(_refs, texts)) - producers - RESERVED:
                raise ValueError(f"{s.name}: tokens name stages it does not declare as inputs: {sorted(bad)}")
            if bad := set(s.consumes.values()) - MODES:
                raise ValueError(f"{s.name}: unknown digest modes {sorted(bad)}")
        return by_name

    def _git(self, spec):
        """({commit, dirty}, deps sha256), or None when a dependency is untracked or differs from HEAD."""
        def git(*args):
            return subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True, check=True).stdout
        try:
            if spec.deps:
                tracked = set(git("ls-files", "--", *spec.deps).splitlines())
                if set(spec.deps) - tracked or git("diff", "--name-only", "HEAD", "--", *spec.deps).strip():
                    return None
            deps = sha(canonical([[p, file_sha(self.repo / p)] for p in sorted(spec.deps)]))
            return {"commit": git("rev-parse", "HEAD").strip(), "dirty": False}, deps
        except (OSError, subprocess.CalledProcessError):
            return None

    def _new_dir(self, spec, key):
        base = self.art / "runs" / f"{spec.site}-{spec.name}-{key[:10]}"
        for n in itertools.count(1):
            d = base if n == 1 else base.with_name(f"{base.name}-{n}")
            try:
                d.mkdir(parents=True)
                return d
            except FileExistsError:
                continue

    def _failed_journals(self, spec, key):
        """The out/ of the newest failed run of this key that journaled generator calls (complete_video_objects), or None.
        ponytail: a seed the graph found (a finished run on the same inputs) wins; the two are never merged."""
        runs = [d for d in (self.art / "runs").glob(f"{spec.site}-{spec.name}-{key[:10]}*-failed-*")
                if any((d / "out" / j).is_dir() for j in ("journal", "journal-sam3d"))]
        return max(runs, key=lambda d: d.stat().st_mtime) / "out" if runs else None

    def _stage(self, spec, stage_from, status, d, key):
        for kind in ("link", "copy"):
            for rel, ref in stage_from.get(kind, {}).items():
                src, dst = Path(self._sub(ref, spec, status, d, key)), d / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if kind == "link":
                    dst.symlink_to(os.path.relpath(src, dst.parent))
                elif src.is_dir():
                    shutil.copytree(src, dst, symlinks=True)
                else:
                    shutil.copy2(src, dst)
        dangling = [os.path.join(r, n) for r, ds, fs in os.walk(d) for n in ds + fs
                    if os.path.islink(os.path.join(r, n)) and not os.path.exists(os.path.join(r, n))]
        if dangling:
            raise RuntimeError(f"staged links do not resolve: {dangling}")

    def _argv(self, spec, commands, status, d, key, amount):
        cmds = [[self._sub(t, spec, status, d, key) for t in c] for c in commands]
        for flag, how in spec.budget_flags.items() if spec.paid else ():
            value = math.floor((amount if how == "usd" else amount / how[1] / 60) * 100) / 100
            target = next((c for c in cmds if flag in c), cmds[0])
            if flag in target:
                i = target.index(flag)
                target[i + 1] = str(min(float(target[i + 1]), value))
            else:
                target += [flag, str(value)]
        return cmds

    def _call(self, spec, cmd, env, cwd, log):
        line = "$ " + " ".join(cmd)
        log.write(("$ <command withheld: matches the redaction pattern>" if REDACT.search(line) else line) + "\n")
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                errors="replace", start_new_session=True)
        killed = threading.Event()

        def kill():  # killing the client stops its ephemeral Modal app
            killed.set()
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        timer = threading.Timer(spec.timeout_s + 300, kill)
        timer.start()
        try:
            for line in proc.stdout:
                if not REDACT.search(line):
                    log.write(line)
                    print(f"[{spec.name}] {line}", end="", flush=True)
        finally:
            timer.cancel()
        code = proc.wait()
        if killed.is_set():
            raise TimeoutError(f"timed out after {spec.timeout_s + 300} s: {' '.join(cmd[:3])}")
        if code:
            raise RuntimeError(f"exit {code}: {' '.join(cmd[:3])}")

    def _run(self, spec, key, digests, status, git, ledger):
        amount = ledger.reserved.get(spec.name, 0.0) if spec.paid else 0.0
        started, t0, d, usd = time.strftime("%Y-%m-%dT%H:%M:%S%z"), time.monotonic(), None, None
        try:
            d = self._new_dir(spec, key)
            with (d / "runner.log").open("w") as log:
                try:
                    commands, stage_from = _symbolic(spec, status)
                    self._stage(spec, stage_from, status, d, key)
                    cmds = self._argv(spec, commands, status, d, key, amount)
                    env = {k: v for k, v in os.environ.items() if k != "PANOPTES_PAID_BUDGET_USD"}  # unset: the tool makes no paid call
                    env.update({k: str(v) for k, v in spec.env.items()})
                    journals = self._failed_journals(spec, key)
                    if journals and "PANOPTES_SEED_JOURNAL" not in env:  # D17 for a failed run of these very inputs: its received calls are read back
                        env["PANOPTES_SEED_JOURNAL"] = str(journals)
                    if spec.paid:
                        env["PANOPTES_PAID_BUDGET_USD"] = str(amount)  # a tool that reads it sees its own share only
                    cwd = self._sub(spec.cwd, spec, status, d, key) if spec.cwd else None
                    for cmd in cmds:
                        self._call(spec, cmd, env, cwd, log)
                except Exception as exc:
                    log.write(f"runner: failed: {exc}\n")
                    raise
            wall = round(time.monotonic() - t0, 1)
            usd = booked(spec, amount, wall) if spec.paid else 0.0
            lock = {"stage": spec.name, "site": spec.site, "version": spec.version, "depsSha256": git[1], "git": git[0], "argv": cmds,
                    "models": [dict(zip(("role", "id", "revision", "weightsSha256"), m), hosting=hosting(m[1])) for m in spec.models],  # 'cloud': a third-party API
                    "image": None, "modal": None,  # ponytail: the tool's own run.json records its image and app
                    "gpu": {"requested": spec.gpu}, "usd": {"reserved": amount, "estimate": spec.est_usd, "booked": usd},
                    "wall_s": wall, "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
            self._record(spec, key, digests, d, "ran", [self.scope or "research"], lock)
            return self._resolved[(spec.site, spec.name)]
        except Exception as exc:
            self._failed.add(key)
            failed, stamp = None, time.strftime('%Y%m%dT%H%M%S')
            if d:  # a unique name: the same stage may fail twice within a second (two commands)
                failed = next(f for n in itertools.count(1) if not (f := d.with_name(f"{d.name}-failed-{stamp}" + (f"-{n}" if n > 1 else ""))).exists())
                d.rename(failed)
            say(f"{spec.name}: failed ({exc}); {failed}")
            return "failed"
        finally:
            if spec.paid:  # a failed stage books too
                ledger.settle(spec, usd if usd is not None else booked(spec, amount, time.monotonic() - t0))

    def _ready(self, spec, status, ledger):
        """A ready stage's status, or a job to submit when it must run."""
        mine = {p: status[p] for p, _ in spec.inputs.values()}
        if any(mine[p] in ABSENT for role, (p, _) in spec.inputs.items() if not role.endswith("?")):
            return "blocked"
        why = self.refuse(spec) if self.refuse else None
        if why:
            say(f"{spec.name}: refused by the {self.scope} profile: {why}")
            return "refused"
        key, digests = self._key(spec, mine)
        found = self._find(spec, key, self.verify)
        if found:
            return found
        if key in self._failed:
            return "failed"
        git = self._git(spec)
        if git is None:
            say(f"{spec.name}: refused, a dependency is untracked or differs from HEAD")
            return "refused"
        if not ledger.reserve(spec):
            say(f"{spec.name}: refused, {_over_budget(spec, ledger)}")
            return "refused"
        return functools.partial(self._run, spec, key, digests, mine, git, ledger)

    def execute(self, specs, ledger, workers=4):
        """Run the graph: hits are served, misses run on a pool. -> {name: Hit | 'failed' | 'blocked' | 'refused'}"""
        by_name = self._check(specs)
        order = TopologicalSorter({n: {p for p, _ in s.inputs.values()} for n, s in by_name.items()})
        order.prepare()
        status, running, stop, waiting = {}, {}, False, []
        with cf.ThreadPoolExecutor(workers) as pool:
            while order.is_active():
                ready, waiting = waiting + list(order.get_ready()), []
                for name in ready:
                    spec = by_name[name]
                    if spec.paid and not stop and any(by_name[n].paid for n in running.values()):
                        waiting.append(name)  # one paid stage at a time: it reserves only after the one before has booked its cost
                        continue
                    r = "blocked" if stop else self._ready(spec, status, ledger)
                    stop = stop or (r == "refused" and spec.paid and ledger.cap is None)  # no budget: stop before the first paid miss
                    if callable(r):
                        running[pool.submit(r)] = name
                        continue
                    status[name] = r
                    order.done(name)
                    say(f"{name}: " + (f"hit {r[0].dir}" if isinstance(r, tuple) else r))
                if running:
                    done, _ = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                    for f in done:
                        name = running.pop(f)
                        status[name] = f.result()
                        order.done(name)
                        if isinstance(status[name], tuple):
                            say(f"{name}: ran {status[name][0].dir}")
        for name, st in status.items():
            if not isinstance(st, tuple):
                self._resolved.pop((by_name[name].site, name), None)
        return {name: st[0] if isinstance(st, tuple) else st for name, st in status.items()}

    def resolve(self, spec):
        """A resolved stage's commands as they run or ran: tokens substituted from this Store's hits (its own dir for @new)."""
        hit, entry = self._resolved[(spec.site, spec.name)]
        status = {p: self._resolved.get((spec.site, p), "failed") for p, _ in spec.inputs.values()}
        commands, _ = _symbolic(spec, status)
        return [[self._sub(t, spec, status, hit.dir, entry["key"]) for t in c] for c in commands]

    def plan(self, specs, ledger=None):
        """Dry run: no subprocess, no network, no writes. Rows: stage, status (hit|miss|unresolved|refused), key, dir, usd, s.
        ponytail: the dirty-dependency refusal needs git, so it shows up only when the stage runs."""
        by_name = self._check(specs)
        sim = Ledger(ledger.cap) if ledger else None
        if sim:
            sim.spent = ledger.spent
        status, rows = {}, []
        for name in TopologicalSorter({n: {p for p, _ in s.inputs.values()} for n, s in by_name.items()}).static_order():
            spec = by_name[name]
            mine = {p: status[p] for p, _ in spec.inputs.values()}
            row = {"stage": name, "status": "unresolved", "key": None, "dir": None, "usd": spec.est_usd, "s": spec.est_s}
            why = self.refuse(spec) if self.refuse else None
            if why:
                row.update(status="refused", why=why)
                status[name] = "refused"
            elif all(isinstance(v, tuple) for v in mine.values()):
                key, _ = self._key(spec, mine)
                found = self._find(spec, key, self.verify)
                if found:
                    row.update(status="hit", key=key, dir=self._rel(found[0].dir), usd=0.0, s=0.0)
                else:
                    fits = sim is None or sim.reserve(spec)
                    if sim and fits and spec.paid:
                        sim.settle(spec, min(sim.reserved[name], spec.est_usd))  # paid stages run one at a time
                    row.update(status="miss" if fits else "refused", key=key, dir=f"runs/{spec.site}-{name}-{key[:10]}",
                               **({} if fits else {"why": _over_budget(spec, sim)}))
                status[name] = found or row["status"]
            else:
                status[name] = "unresolved"
            rows.append(row)
        return rows


def print_plan(rows):
    say(f"{'stage':<22} {'status':<10} {'key':<12} {'usd':>7} {'s':>7}  dir")
    for r in rows:
        say(f"{r['stage']:<22} {r['status']:<10} {(r['key'] or '-')[:12]:<12} {r['usd']:>7.2f} {r['s']:>7.0f}  {r['dir'] or '-'}")
    for r in rows:
        if r.get("why"):
            say(f"refused {r['stage']}: {r['why']}")
    n, todo = Counter(r["status"] for r in rows), [r for r in rows if r["status"] != "hit"]
    say(f"total: {n['hit']} hit, {n['miss']} miss, {n['unresolved']} unresolved, {n['refused']} refused; "
        f"est ${sum(r['usd'] for r in todo):.2f}, {sum(r['s'] for r in todo):.0f} s")


def art_root():
    """$ART as modal_apps/droid_room.ART sets it, read from the source so the runner never imports Modal."""
    return Path(re.search(r'^ART = Path\("([^"]+)"\)', (REPO / "modal_apps/droid_room.py").read_text(), re.M).group(1))


def _decision(output):
    return json.loads(Path(output.outputs.get("decision") or next(iter(output.outputs.values()))).read_text())


def record_import(state, art, site, profile, key, log_text, db=None):
    """Append a new import to imports.jsonl, so the next run of the same command republishes it (stages.previous_import) instead of
    making a second report: its ids from the import's own output, the record by path only (never opened), and, when the platform
    database answers (read-only), its published title, fingerprint and counts."""
    adopt = importlib.import_module("report_runner.adopt")
    ids = dict(re.findall(r'"(publicationId|projectId)":\s*"([^"]+)"', log_text))
    row = {"site": site, "profile": profile, "key": key, "publicationId": ids["publicationId"], "projectId": ids.get("projectId"),
           "importRecordPath": adopt.import_record(art, ids["publicationId"])}
    try:  # ponytail: the row without fingerprint still republishes; the fingerprint is for the reproduction proof
        published = (db or adopt.Database()).publication(ids["publicationId"])
        row.update(projectId=published["projectId"], title=published["title"], fingerprint=adopt.fingerprint(published["document"], published["title"]),
                   counts=adopt.counts(published["document"]))
    except Exception as error:
        say(f"imports.jsonl: {ids['publicationId']} recorded without its fingerprint ({type(error).__name__})")
    row["at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(state).mkdir(parents=True, exist_ok=True)
    with open(Path(state) / "imports.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def main(args):
    """The loop behind scripts/run_video_report.py. graph(ctx) raises Pending(decision, specs) until ctx.decisions holds
    what it needs; the decision stage is served or run, and graph is called again. The graph's last stage is the import."""
    stages = importlib.import_module("report_runner.stages")
    profiles = importlib.import_module("report_runner.profiles")
    profile = profiles.PROFILES[args.profile]
    cache_only = getattr(profile, "cache_only", False)
    art = art_root()
    refuse = functools.partial(profiles.refuse, args.profile) if hasattr(profiles, "refuse") else None
    store = Store(art, scope=args.profile, verify=args.verify, refuse=refuse)
    command = sha(canonical([args.site, str(Path(args.video).resolve()), args.start, args.end, args.profile]))[:16]
    ledger = Ledger.from_env(store.state / "ledgers" / f"{args.site}-{command}.jsonl")  # a re-run continues this command's budget
    # absolute paths, so a token names its leaf exactly (a key holds a leaf's bytes, never its path)
    ctx = Ctx(args.site, Path(args.video).resolve(), args.start, args.end, args.profile, Path(args.review).resolve() if args.review else None, art, store,
              republish=getattr(args, "republish", False))
    while True:
        try:
            specs = stages.graph(ctx)
            break
        except RuntimeError as error:  # a decision the report needs could not be made (or no mapped shot)
            say(f"stops: {error}")
            return 1
        except Pending as p:
            chosen = next((s for s in p.specs if s.name == p.decision), None)
            if chosen is None or p.decision in ctx.decisions:
                raise RuntimeError(f"graph asked for decision {p.decision!r} without a stage that makes it") from p
            rows = store.plan(p.specs, ledger)
            if next(r for r in rows if r["stage"] == p.decision)["status"] != "hit":
                if args.dry_run or cache_only:
                    print_plan(rows)
                    say(f"stops here: decision {p.decision} is not in the cache" + ("" if args.dry_run else "; the profile is cache-only"))
                    return 0 if args.dry_run else 2
                done = store.execute(p.specs, ledger).get(p.decision)
                if not isinstance(done, Hit):  # the graph leaves the layers that need it blank, or stops when the report needs it
                    say(f"decision {p.decision} did not resolve ({done}); the layers that need it are left blank")
                    ctx.decisions[p.decision] = {"value": None, "absent": done or "missing"}
                    continue
            ctx.decisions[p.decision] = _decision(store.lookup(chosen))
    rows = store.plan(specs, ledger)
    if args.dry_run or (cache_only and any(r["status"] != "hit" for r in rows)):
        print_plan(rows)
        if not args.dry_run:
            say(f"refused: the {args.profile} profile is cache-only")
        return 0 if args.dry_run else 2
    final = store.execute(specs, ledger)[specs[-1].name]
    if not isinstance(final, Hit):
        say(f"{specs[-1].name}: {final}")
        return 1
    log = final.dir / "runner.log"
    found = re.search(r'"publicationId":\s*"([^"]+)"', log.read_text()) if log.is_file() else None
    if final.verification != "ran":  # served from the cache: nothing was imported again; show what the import was given
        say(f"{specs[-1].name}: served from the cache ({final.verification}); no new import. Resolved command:")
        for cmd in store.resolve(specs[-1]):
            say("  " + " ".join(cmd))
    elif found:
        record_import(store.state, art, args.site, args.profile, store._resolved[(args.site, specs[-1].name)][1]["key"], log.read_text())
    if args.publish:
        if not found:
            say("no publicationId in the import's output; not publishing")
            return 1
        importlib.import_module("report_runner.publish").publish(found.group(1), dry_run=False)
    return 0
