"""Clause retrieval anchored on what the scene contains: a clause is retrieved when its `tags` share a class or zone kind with the
scene's objects / zones; every clause when there is no scene. Graph order, no ranking, no embeddings (taxonomy-style baseline)."""
from __future__ import annotations

from ehs_spatial.verdict.contracts import ClauseGraph, Scene


def retrieve(clauses: ClauseGraph, scene: Scene | None) -> list[str]:
    if scene is None:
        return [c.id for c in clauses.clauses]
    present = {o.cls for o in scene.objects} | {z.kind for z in scene.zones}
    return [c.id for c in clauses.clauses if present.intersection(c.tags)]
