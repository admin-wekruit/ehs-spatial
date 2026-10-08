"""One run = one config x one benchmark item: L1..L7 in order, each layer's outputs written under runs/<run_id>/<layer>/ (contract
models through their `dump`, dicts / lists as sorted JSON), `reuse: <run_id>` loads a layer's outputs from that run instead of
computing them, and one ledger line per run in runs/ledger.jsonl."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import yaml

from ehs_spatial.verdict import plugins
from ehs_spatial.verdict.contracts import Facts, Provenance, RulePack, Scene, Signature, VerdictSet

PKG = Path(__file__).resolve().parents[1]                      # ehs_spatial/verdict
NEEDS = {"L1": ["source"], "L2": ["scene"], "L3": ["scene", "facts"], "L4": ["spec_dir", "signature", "scene"],
         "L5": ["clauses", "alignment", "retrieved", "signature"], "L6": ["facts", "rule_pack", "scene"],
         "L7": ["verdicts", "facts", "scene", "workdir"]}                        # the layer I/O table of plugins.py
MODELS = {"scene": Scene, "facts": Facts, "rule_pack": RulePack, "verdicts": VerdictSet}


def new_run_id() -> str:
    """<YYYYmmdd-HHMMSS>-<git short sha>; 'nogit' when there is no repository."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=PKG, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        sha = "nogit"
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{sha}"


def items(benchmark: Path) -> list[dict]:
    return yaml.safe_load((Path(benchmark) / "items.yaml").read_text())["items"]


def tags(config: dict) -> dict[str, str]:
    """{layer: 'name@version'} from the registry; a `version` in the config must match the registered one."""
    out = {}
    for layer in plugins.LAYERS:
        cls = plugins.get(layer, config[layer]["plugin"])
        want = config[layer].get("version")
        if want is not None and str(want) != cls.version:
            raise KeyError(f"{layer}.{cls.name}: config wants version {want}, registered {cls.version}")
        out[layer] = plugins.tag(cls)
    return out


def write(outputs: dict, layer_dir: Path) -> None:
    for key, value in outputs.items():
        if hasattr(value, "dump"):
            value.dump(layer_dir / f"{key}.json")
        elif not isinstance(value, Path):          # a Path is a file the plugin already wrote into its workdir (L7's report)
            (layer_dir / f"{key}.json").write_text(json.dumps(value, indent=1, sort_keys=True, ensure_ascii=False))


def read(layer_dir: Path) -> dict:
    return {f.stem: MODELS[f.stem].load(f) if f.stem in MODELS else json.loads(f.read_text()) for f in sorted(layer_dir.glob("*.json"))}


def run(config: dict, runs_dir: str | Path, run_id: str | None = None) -> str:
    run_id, runs_dir, started = run_id or new_run_id(), Path(runs_dir), time.time()
    benchmark = Path(config["benchmark"])
    by_id = {str(i["id"]): i for i in items(benchmark)}
    item_id = str(config.get("item") or next(iter(by_id)))
    item = by_id[item_id]
    plugin_tags = tags(config)
    prov = Provenance(run_id=run_id, benchmark=f"{benchmark.name}/{item_id}", plugins=plugin_tags)
    root = runs_dir / run_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({**config, "item": item_id, "run_id": run_id}, indent=1, sort_keys=True))
    state: dict = {"source": benchmark / item["source"], "signature": Signature.load(config.get("signature") or PKG / "signature-v1.json"),
                   "spec_dir": Path(config.get("spec_dir") or benchmark / "spec")}
    llm_calls, reused = 0, {}
    for layer in plugins.LAYERS:
        layer_cfg = dict(config[layer])
        layer_dir = root / layer
        layer_dir.mkdir(exist_ok=True)
        state["workdir"] = layer_dir
        if layer_cfg.get("reuse"):
            outputs = read(runs_dir / layer_cfg["reuse"] / layer)
            reused[layer] = layer_cfg["reuse"]
        else:
            cls = plugins.get(layer, layer_cfg.pop("plugin"))
            layer_cfg.pop("version", None)
            outputs = cls().run({**{k: state[k] for k in NEEDS[layer]}, "provenance": prov}, layer_cfg, layer_dir)
            llm_calls += int(outputs.pop("llm_calls", 0))
        write(outputs, layer_dir)
        state.update(outputs)
    with (runs_dir / "ledger.jsonl").open("a") as f:    # ponytail: concurrent matrix tasks rely on O_APPEND of one short line each
        f.write(json.dumps({"run_id": run_id, "config_file": config.get("config_file", ""), "benchmark": str(benchmark), "item": item_id,
                            "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), "seconds": round(time.time() - started, 3),
                            "plugins": plugin_tags, "reused": reused, "llm_calls": llm_calls}, sort_keys=True) + "\n")
    return run_id
