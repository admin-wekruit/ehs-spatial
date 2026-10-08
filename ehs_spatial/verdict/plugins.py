"""Plugin interface and registry of the verdict layer.

Every layer is a plugin with the same shape; a run picks one plugin per layer from a YAML config. Plugins import only
`ehs_spatial.verdict.contracts` (and their own package); a test enforces it. Third-party plugins register through the entry-point
group `panoptes.verdict.plugins` (each entry point is a module that calls `register` at import).

Layer I/O (all values are contract models or plain dicts; files are written by the runner, not by plugins):
  L1  inputs: {'source': <path or dict the producer understands>}            -> {'scene': Scene}
  L2  inputs: {'scene': Scene}                                               -> {'facts': Facts}
  L3  inputs: {'scene': Scene, 'facts': Facts}                               -> {'facts': Facts}   (annotated: quality + 'untrusted' flags)
  L4  inputs: {'spec_dir': Path, 'signature': Signature, 'scene': Scene|None} -> {'clauses': ClauseGraph, 'alignment': Alignment, 'retrieved': list[str]}
  L5  inputs: {'clauses': ClauseGraph, 'alignment': Alignment, 'retrieved': list[str], 'signature': Signature} -> {'rule_pack': RulePack}
  L6  inputs: {'facts': Facts, 'rule_pack': RulePack, 'scene': Scene}         -> {'verdicts': VerdictSet}
  L7  inputs: {'verdicts': VerdictSet, 'facts': Facts, 'scene': Scene, 'workdir': Path} -> {'report': Path}
"""
from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6", "L7")


@runtime_checkable
class Plugin(Protocol):
    layer: ClassVar[str]
    name: ClassVar[str]
    version: ClassVar[str]

    def run(self, inputs: dict[str, Any], cfg: dict[str, Any], workdir: Path) -> dict[str, Any]: ...


REGISTRY: dict[str, dict[str, type]] = {layer: {} for layer in LAYERS}


def register(layer: str, name: str, version: str):
    """Class decorator: `@register('L6', 'clingo', '1')`. The class gets layer/name/version attributes and lands in REGISTRY."""
    if layer not in LAYERS:
        raise ValueError(f"unknown layer {layer}")

    def deco(cls):
        cls.layer, cls.name, cls.version = layer, name, version
        if name in REGISTRY[layer] and REGISTRY[layer][name] is not cls:
            raise ValueError(f"plugin {layer}.{name} registered twice")
        REGISTRY[layer][name] = cls
        return cls
    return deco


def tag(plugin: type | Plugin) -> str:
    """'name@version' as it appears in provenance and the scorecard."""
    return f"{plugin.name}@{plugin.version}"


def get(layer: str, name: str) -> type:
    if name not in REGISTRY[layer]:
        load_all()
    try:
        return REGISTRY[layer][name]
    except KeyError:
        raise KeyError(f"no plugin {layer}.{name}; known: {sorted(REGISTRY[layer])}") from None


def load_all() -> None:
    """Import the built-in plugin packages and every entry point of group `panoptes.verdict.plugins`."""
    import importlib
    import pkgutil

    import ehs_spatial.verdict.layers as layers
    for mod in pkgutil.walk_packages(layers.__path__, layers.__name__ + "."):
        importlib.import_module(mod.name)
    for ep in entry_points(group="panoptes.verdict.plugins"):
        ep.load()


def available() -> dict[str, list[str]]:
    load_all()
    return {layer: sorted(tag(c) for c in REGISTRY[layer].values()) for layer in LAYERS}
