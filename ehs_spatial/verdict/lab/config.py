"""Run configs: one YAML per run: `benchmark: <dir>` (relative to the config file), optional `item` (benchmark item id), and one
entry per layer `{plugin: name, version?: ..., params..., reuse?: <run_id>}`. `extends: <relative path>` composes a base config
with overrides (layer dicts merge key by key). No Hydra."""
from __future__ import annotations

from pathlib import Path

import yaml


def load(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    if "benchmark" in cfg and not Path(cfg["benchmark"]).is_absolute():
        cfg["benchmark"] = str((path.parent / cfg["benchmark"]).resolve())
    base = cfg.pop("extends", None)
    if base:
        merged = load(path.parent / base)
        for key, value in cfg.items():
            merged[key] = {**merged[key], **value} if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        cfg = merged
    cfg["config_file"] = str(path)
    return cfg
