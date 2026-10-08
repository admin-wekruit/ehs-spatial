"""Lab rule 1: a layer plugin imports only ehs_spatial.verdict.contracts (and .plugins, for @register), the stdlib, numpy / shapely /
scipy / clingo / yaml, and its own package. Never another layer, never the rest of ehs_spatial. AST scan, no imports executed."""
import ast
import sys
from pathlib import Path

LAYERS = Path(__file__).resolve().parents[2] / "ehs_spatial/verdict/layers"
ALLOWED_TOP = set(sys.stdlib_module_names) | {"numpy", "shapely", "scipy", "clingo", "yaml"}
ALLOWED_VERDICT = {"ehs_spatial.verdict.contracts", "ehs_spatial.verdict.plugins"}
SHARED = ("ehs_spatial.verdict.synth",)   # lab-shared test-support packages (synthetic scenes); not a layer


def imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        if node.level:
            return [] if node.level == 1 else [f"<relative import, level {node.level}>"]
        if node.module == "ehs_spatial.verdict":
            return [f"ehs_spatial.verdict.{a.name}" for a in node.names]
        return [node.module or ""]
    return []


def violations(path: Path) -> list[str]:
    own = f"ehs_spatial.verdict.layers.{path.parent.name}"
    bad = []
    for node in ast.walk(ast.parse(path.read_text())):
        for name in imported_names(node):
            if not (name.split(".")[0] in ALLOWED_TOP or name in ALLOWED_VERDICT or name == own or name.startswith(own + ".") or any(name == m or name.startswith(m + ".") for m in SHARED)):
                bad.append(f"{path.relative_to(LAYERS)}: {name}")
    return bad


def test_layer_plugins_import_only_contracts_and_their_own_package():
    files = sorted(LAYERS.glob("l[1-7]_*/**/*.py"))
    assert files, "no layer modules found"
    bad = [v for p in files for v in violations(p)]
    assert not bad, "\n".join(bad)
