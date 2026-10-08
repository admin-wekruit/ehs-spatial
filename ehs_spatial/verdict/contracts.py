"""The five contracts of the verdict layer (docs/research/verdict-layer-architecture-2026-10-08.md).

C1 Scene      what any reconstruction / mapping producer hands over (metres, one world frame)
C2 Facts      typed predicates computed from a Scene (what rules consume); plus the plan occupancy grid
C3 Signature  the vocabulary rules may reference (classes, zones, predicates, attributes, declared inputs)
C4 RulePack   versioned, reviewed rules (ASP text) with their clause ids and tests
C5 Verdict    one verdict per (rule, subjects) with measurement, uncertainty, evidence and provenance

Every layer plugin imports only this module. Nothing here computes geometry or runs rules.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

SCHEMA_VERSION = "verdict/1"
Status = Literal["PASS", "FAIL", "NEEDS_MEASUREMENT", "NEEDS_INPUT", "CANNOT_DETERMINE"]
Confidence = Literal["high", "medium", "low", "unverified"]
ObjSource = Literal["observed", "believed", "declared"]   # believed = inferred (e.g. occluded, WorldSGG-style); declared = given by a human / datasheet
ZoneKind = Literal["hazard_zone", "restricted_space", "operator_station", "aisle", "other"]


class _Model(BaseModel):
    def dump(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=1))

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate_json(Path(path).read_text())


# ---------------------------------------------------------------- C1 Scene
class Obj(_Model):
    id: str
    cls: str                       # a class of the Signature (semantic input, with confidence); 'other' when unknown
    label: str = ""
    center_m: list[float]          # [x, y, z] world metres
    axes: list[list[float]]        # [l, w, up] unit vectors; up = floor normal
    size_m: list[float]            # [L, W, H]
    bottom_m: float                # lowest point above the floor (negative = below the fitted floor)
    top_m: float
    sigma_m: dict[str, float | None] = Field(default_factory=dict)   # 1-sigma per quantity: 'L','W','H','bottom' (None = unmeasured)
    confidence: Confidence = "unverified"
    floor_contact: bool = False
    views: list[str] = Field(default_factory=list)                   # evidence refs (photo / frame ids); opaque to the engine
    t: float | None = None                                           # seconds, for 4D producers; None for a static scene
    source: ObjSource = "observed"


class Zone(_Model):
    id: str
    kind: ZoneKind
    polygon_xy_m: list[list[float]]   # plan-view polygon in the scene's plan basis (see Coverage.basis / Grid.basis)
    source: Literal["declared", "derived", "reviewer"] = "declared"
    note: str = ""


class Coverage(_Model):
    """What the sensors actually observed, on the plan grid: a cell is 'observed' when some view saw through or onto it.
    None (no Coverage) means unknown everywhere -> topology rules can only answer CANNOT_DETERMINE."""
    cell_m: float
    origin_xy: list[float]
    basis: list[list[float]]          # [e1, e2] unit vectors spanning the floor plane (world frame)
    shape: list[int]                  # [nx, ny]
    observed: list[list[int]]         # cells [ix, iy]
    method: str = ""


class Scene(_Model):
    schema: str = SCHEMA_VERSION
    scene_id: str
    objects: list[Obj]
    ground_normal: list[float]
    scale_rel_unc: float | None = None          # relative scale uncertainty of the metric calibration
    zones: list[Zone] = Field(default_factory=list)
    coverage: Coverage | None = None
    declared_inputs: dict[str, float | str] = Field(default_factory=dict)   # stop_time_ms, resolution_mm, risk_level, ...
    producer: str = ""                           # plugin name@version that produced the scene


# ---------------------------------------------------------------- C2 Facts
class Fact(_Model):
    pred: str                        # predicate name of the Signature, e.g. 'min_distance_3d', 'floor_gap', 'top_height'
    args: list[str]                  # object / zone ids (order as the Signature declares)
    value: float | None = None       # mm (or the predicate's unit)
    unit: str | None = None
    u: float | None = None           # expanded uncertainty (k=2), same unit
    views: list[str] = Field(default_factory=list)
    t: float | None = None
    flags: list[str] = Field(default_factory=list)   # e.g. 'default_sigma', 'default_scale_unc', 'untrusted'


class Grid(_Model):
    cell_m: float
    origin_xy: list[float]
    basis: list[list[float]]
    shape: list[int]
    blocked: list[list[int]]
    hazard: list[list[int]]
    outside: list[list[int]]
    observed: list[list[int]] | None = None    # copied from Scene.coverage when present


class Facts(_Model):
    schema: str = SCHEMA_VERSION
    scene_id: str
    signature_version: str
    facts: list[Fact]
    grid: Grid | None = None
    quality: dict[str, object] = Field(default_factory=dict)   # L3's findings (untrusted objects, coverage ratio, ...)
    producer: str = ""


# ---------------------------------------------------------------- C3 Signature
class Predicate(_Model):
    args: list[str]                  # arg kinds: 'object', 'zone', 'floor', 'cell'
    unit: str | None = None
    note: str = ""


class Signature(_Model):
    schema: str = SCHEMA_VERSION
    version: str
    classes: list[str]
    zones: list[str]
    predicates: dict[str, Predicate]
    attributes: list[str]
    declared_inputs: list[str]


# ---------------------------------------------------------------- L4 output: clause graph + alignment (consumed by L5)
class Table(_Model):
    id: str                          # e.g. 'ISO13857:2019/Table2'
    standard: str
    edition: str
    function: str                    # import path 'ehs_spatial.verdict.layers.l4_spec.tables:iso13857_table2'
    inputs: list[str]                # argument names, e.g. ['hazard_height_mm', 'structure_height_mm', 'risk_level']
    output_unit: str = "mm"
    verified: bool = False           # True only when checked against the purchased text
    source_url: str = ""


class Clause(_Model):
    id: str                          # 'ISO13857:2019/4.4'
    standard: str
    edition: str
    title: str
    paraphrase: str                  # our words, not the text (short quotes only inside rules' source_text)
    rule_class: Literal["geometry", "topology", "semantic", "procedural"]
    selection: dict[str, list[str]] = Field(default_factory=dict)      # variable -> Signature classes / zones
    applicability: list[str] = Field(default_factory=list)             # Signature predicates / zones that must hold, e.g. 'perimeter_of(F, Z)'
    requirement: dict[str, object] = Field(default_factory=dict)       # {predicate, args, operator, threshold|table|formula, unit, inputs}
    exceptions: list[str] = Field(default_factory=list)                # Signature attributes that switch the clause off
    definitions: list[str] = Field(default_factory=list)               # terms defined elsewhere (keys of ClauseGraph.definitions)
    tags: list[str] = Field(default_factory=list)                      # retrieval anchors: classes / zones the clause is about
    photo_checkable: bool = True
    verified: bool = False
    source_url: str = ""


class ClauseGraph(_Model):
    schema: str = SCHEMA_VERSION
    version: str
    standards: list[dict[str, str]] = Field(default_factory=list)      # [{id, title, edition, url}]
    clauses: list[Clause]
    tables: list[Table] = Field(default_factory=list)
    definitions: dict[str, list[str]] = Field(default_factory=dict)    # term -> aliases (used by alignment)


class AlignmentRow(_Model):
    term: str                        # a word / phrase of the clauses
    kind: Literal["class", "zone", "predicate", "attribute", "input"]
    target: str                      # Signature entry
    confidence: float
    source: str = ""                 # 'exact' | 'synonym' | 'embedding' | 'classifier' | 'reviewer'


class Alignment(_Model):
    schema: str = SCHEMA_VERSION
    signature_version: str
    rows: list[AlignmentRow]


# ---------------------------------------------------------------- C4 RulePack
RuleStatus = Literal["compiled", "needs_input", "vocabulary_gap", "refused"]


class Rule(_Model):
    rule_id: str
    version: str
    clause: str                      # e.g. 'ISO 13857:2019 4.4'
    standard: str
    edition: str
    rule_class: Literal["geometry", "topology", "semantic"]
    source_text: str = ""
    spec: dict[str, object] = Field(default_factory=dict)   # machine-readable semantics every engine evaluates: selection {var: [classes]},
                                                            # applicability [predicate(args)], requirement {predicate, args, operator, threshold|table|formula, unit, inputs}, exceptions [attributes]
    asp: str = ""                    # the rule's ASP text rendered from spec (clingo engine); must only use Signature predicates
    inputs: list[str] = Field(default_factory=list)      # declared inputs it needs
    thresholds: dict[str, float] = Field(default_factory=dict)
    status: RuleStatus = "compiled"
    unsupported_reason: str | None = None
    tests: list[str] = Field(default_factory=list)       # ids of synthetic scenes with expected outcomes
    review: dict[str, str] = Field(default_factory=dict) # reviewer, date, decision
    provenance: dict[str, str] = Field(default_factory=dict)   # how it was produced: plugin@version, model, ...


class RulePack(_Model):
    schema: str = SCHEMA_VERSION
    pack_id: str
    version: str
    signature_version: str
    common_asp: str = ""             # shared text: decision rule, helpers
    rules: list[Rule]


# ---------------------------------------------------------------- C5 Verdict
class Provenance(_Model):
    run_id: str
    benchmark: str = ""
    plugins: dict[str, str] = Field(default_factory=dict)   # {'L1': 'photo-v2@1.3', ..., 'L6': 'clingo-v1@5.8.2'}


class Verdict(_Model):
    rule_id: str
    rule_version: str
    status: Status
    subjects: list[str]
    labels: list[str] = Field(default_factory=list)
    measured: float | None = None
    u: float | None = None
    threshold: float | None = None
    unit: str | None = None
    margin: float | None = None
    unknown_inputs: list[str] = Field(default_factory=list)
    evidence: dict[str, object] = Field(default_factory=dict)   # facts used (pred/args/value/u), views, grid cells
    notes: list[str] = Field(default_factory=list)
    provenance: Provenance


class VerdictSet(_Model):
    schema: str = SCHEMA_VERSION
    scene_id: str
    rule_pack: str                   # pack_id@version
    decision_rule: str               # e.g. 'guard_band_k2'
    verdicts: list[Verdict]
    coverage: dict[str, object] = Field(default_factory=dict)   # rules evaluated / skipped and why
    gaps: dict[str, object] = Field(default_factory=dict)       # unknown inputs, untrusted objects, vocabulary gaps
    provenance: Provenance


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
