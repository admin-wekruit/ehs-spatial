"""Shared types of the M2 report runner. Part A owns this file; Parts B, C and D may commit a byte-identical copy.

Command tokens, substituted when a stage runs (the key keeps them symbolic): '@new' the stage's fresh directory,
'@clip' a derived clip directory $ART/data/clips/<site>-<key[:10]>, '@key' key[:12], '@STAGE' a producer's directory,
'@STAGE:ROLE' a producer's output path; '$ART' and '$REPO' expand. A flag and its value are separate tokens.
An input role ending in '?' is optional: when its producer failed, was blocked or was refused, the consumer still runs,
every token naming that producer is dropped, and so is the flag right before it when that flag is left without a value.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StageSpec:
    name: str
    site: str
    version: int
    deps: tuple[str, ...]  # repo-relative files whose sha goes in the lock; one that differs from HEAD refuses the stage
    commands: tuple[tuple[str, ...], ...]
    inputs: dict[str, tuple[str, tuple[str, ...]]]  # role -> (producer stage name, producer roles; () means all)
    consumes: dict[str, str]  # role -> 'outputs' (default) | 'droid-frames' | 'lingbot-source' | 'decision-value'
    leaves: dict[str, Path]  # MP4, review JSON, docs files: no stage makes them, their sha256 enters the key
    outputs: dict[str, str]  # role -> relpath in the stage directory
    stage_from: dict  # {'link': {rel: '@stage:role'}, 'copy': {rel: '@stage:role'}}
    compute: str
    gpu: str | None
    timeout_s: int  # the Modal function timeout x calls; each subprocess gets this + 300 s
    worst_usd: float
    est_usd: float
    est_s: float
    budget_flags: dict  # {'--max-usd': 'usd'} | {'--max-minutes': ('minutes', usd_per_s)}
    models: tuple[tuple[str, str, str, str | None], ...]  # (role, hf_id, revision | 'unpinned', weights_sha256 | None)
    rules: dict[str, str]  # decision -> 'name@v'
    env: dict
    cwd: str | None = None
    paid: bool = False
    usd_per_s: float = 0.  # list price of the Modal container it keeps busy (GPU + requested CPU and memory); 0: none (a cloud API)


class Pending(Exception):
    """graph(ctx) needs a decision ctx.decisions lacks. specs: the stages built so far (closed under producers), including
    the stage named `decision`; its output (role 'decision' when it has several) is the JSON {value, evidence, rule}."""

    def __init__(self, decision: str, specs=()):
        super().__init__(decision)
        self.decision, self.specs = decision, tuple(specs)


@dataclass
class Hit:
    dir: Path
    outputs: dict[str, Path]
    verification: str  # ran | replayed | recorded-rev | content | delivered-only


@dataclass
class Ctx:
    site: str
    video: Path
    start: float
    end: float
    profile: str  # a key of profiles.PROFILES
    review: Path | None
    art: Path
    store: object
    decisions: dict[str, dict] = field(default_factory=dict)  # decision -> {value, evidence, rule}
    republish: bool = False  # the import becomes the next version of the site's last import (implied by the delivered profile)
