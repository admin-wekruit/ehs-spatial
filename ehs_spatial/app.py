import json
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any
from uuid import uuid4

import gradio as gr
from pydantic import ValidationError

from .contracts import (
    Assessment,
    CaptureRun,
    GroundedAnswer,
    PolicySpec,
    ReviewDisposition,
    SceneMap,
)
from .pipeline import EHSAssessmentPipeline
from .providers.base import ProviderError
from .video import (
    BAND_M,
    COST_PER_SAM_CALL_USD,
    RULE_NAMES,
    run_video_assessment,
    video_paths,
)


DEMO_RULE_COPY = "0.6 m demo rule — not an official EHS standard"
# Compilation from prose is an offline, reviewed step (scripts/
# policy_compile.py); the UI only offers specs a reviewer could have read.
POLICIES_DIR = "outputs/policies/compiled"
POLICY_REVIEW_COPY = (
    "Compiled specs are reviewed before use — this UI selects reviewed "
    "specs and never compiles prose live."
)
PIPELINE_CONCURRENCY_ID = "ehs-provider-pipeline"
# History/disposition handlers only touch local run artifacts; their own lane
# keeps them responsive while a multi-minute provider analysis holds the
# provider lane.
LOCAL_CONCURRENCY_ID = "ehs-local-ui"
# Report rendering is read-only and must appear instantly on page load; its
# own lane keeps it from queueing behind the History table's full-runs scan.
REPORT_CONCURRENCY_ID = "ehs-report-ui"


def load_policy_specs(policies_dir: str | Path = POLICIES_DIR) -> list[PolicySpec]:
    """Precompiled PolicySpecs found at app build time. An unreadable file is
    skipped so one bad cache entry never blocks startup."""
    specs = []
    for path in sorted(Path(policies_dir).glob("*.json")):
        try:
            specs.append(
                PolicySpec.model_validate_json(path.read_text(encoding="utf-8"))
            )
        except Exception:
            continue
    return specs


def _policy_label(spec: PolicySpec) -> str:
    return f"{spec.policy_id} — {spec.source_text[:60]}"


def _provider_error_copy(error: ProviderError) -> str:
    return (
        f"{error.provider} {error.operation} failed. Check provider credentials, "
        "quota, and service status, then retry."
    )


def _analysis_error_outputs(error_copy: str) -> tuple[object, ...]:
    return (
        None,
        gr.update(
            value=(
                "### RUN ERROR\n\n"
                "Analysis failed. No assessment was produced.\n\n"
                f"{error_copy}\n\n"
                "No fallback result was generated."
            ),
            elem_classes=["result-status", "status-error"],
        ),
        None,
        None,
        {
            "status": "RUN_ERROR",
            "error": error_copy,
            "assessment": None,
            "scene_map": None,
        },
        [],
        gr.update(value="", interactive=False),
        gr.update(interactive=False),
    )


_STATUS_ORDER = {
    "FAIL": 0,
    "NEEDS_REVIEW": 1,
    "INSUFFICIENT_EVIDENCE": 2,
    "PASS": 3,
}


def _status_copy(
    assessment: Assessment,
    scene: SceneMap | None = None,
    policy_results: list[dict] | None = None,
    policy_specs: dict[str, dict] | None = None,
) -> str:
    # Never print a bare decimal: the band is part of the measurement.
    if assessment.approximate_distance_m is None:
        distance = "unavailable from the evidence"
    elif assessment.distance_error_budget_m is not None:
        distance = (
            f"{assessment.approximate_distance_m:.2f} m "
            f"± {assessment.distance_error_budget_m:.2f} m"
        )
    else:
        distance = f"{assessment.approximate_distance_m:.2f} m"
    lines = [
        f"### {assessment.status.value}",
        f"Approximate boundary clearance: **{distance}**.",
        f"**{DEMO_RULE_COPY}.**",
    ]
    if scene is not None and scene.scale_source:
        confidence = (
            f", confidence {scene.scale_confidence:.2f}"
            if scene.scale_confidence is not None
            else ""
        )
        lines.append(f"Scale source: `{scene.scale_source}`{confidence}.")
    if policy_results:
        ordered = sorted(
            policy_results,
            key=lambda r: _STATUS_ORDER.get(str(r.get("status")), 9),
        )
        lines.append("**Policies:**")
        for result in ordered:
            spec = (policy_specs or {}).get(result["policy_id"], {})
            rule = (
                f" — {spec['predicate']} {spec['threshold']} {spec['unit']}"
                if spec.get("predicate")
                else ""
            )
            worst = result.get("violations") or []
            detail = (
                f" — worst {worst[0]['measured']}{worst[0]['unit']} "
                f"(limit {worst[0]['threshold']}{worst[0]['unit']})"
                if worst
                else ""
            )
            source = ""
            if spec.get("source_text"):
                excerpt = spec["source_text"]
                if len(excerpt) > 90:
                    excerpt = excerpt[:87] + "..."
                source = f' — "{excerpt}"'
            lines.append(
                f"- `{result['status']}` {result['policy_id']}{rule}{detail}{source}"
            )
    if scene is not None and scene.warnings:
        shown = scene.warnings[:3]
        extra = len(scene.warnings) - len(shown)
        lines.append("**Warnings:**")
        lines.extend(f"- {warning}" for warning in shown)
        if extra > 0:
            lines.append(f"- (+{extra} more in the structured output)")
    climb = assessment.climb_review
    climb_copy = "Climb: **REVIEW only**"
    if climb is not None:
        rationale = climb.rationale.removeprefix("REVIEW only: ")
        climb_copy += f" — {climb.verdict.upper()}. {rationale}"
    lines.append(climb_copy)
    return "\n\n".join(lines)


def _scripts_on_path() -> None:
    import sys as _sys

    scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts_dir not in _sys.path:
        _sys.path.insert(0, scripts_dir)


def current_analysis_version() -> str:
    _scripts_on_path()
    from scene_inventory import ANALYSIS_VERSION

    return ANALYSIS_VERSION


def analysis_state(run_dir: Path) -> str:
    """'current' | 'stale' | 'missing' — one product, one generation of
    analysis: a run whose inventory predates the current rules is stale
    and gets upgraded on open instead of being shown as-is."""
    inventory = run_dir / "inventory" / "inventory.json"
    if not inventory.exists():
        return "missing"
    try:
        stamp = json.loads(inventory.read_text(encoding="utf-8")).get(
            "analysis_version"
        )
    except Exception:
        return "stale"
    return "current" if stamp == current_analysis_version() else "stale"


def _chain_running(run_dir: Path) -> bool:
    status_path = run_dir / "deep_report.status"
    if not status_path.exists():
        return False
    return status_path.read_text(encoding="utf-8").strip() in {
        "detect", "inventory", "report",
    }


def _run_deep_report_chain(run_id: str) -> None:
    """Device detection -> inventory refinement (order matters: the
    inventory's detection-sync reads detections.json) -> interactive
    report. Fail-soft stage by stage; status file keeps the 报告 tab
    honest while this grinds. Re-entrant for upgrades: a run that already
    has detections skips straight to the inventory rules."""
    import traceback

    run_dir = Path("runs") / run_id
    status_path = run_dir / "deep_report.status"

    def _mark(state: str) -> None:
        try:
            status_path.write_text(state, encoding="utf-8")
        except Exception:
            pass

    _scripts_on_path()
    if not (run_dir / "detection" / "detections.json").exists():
        try:
            _mark("detect")
            import detect_devices as _detect

            _detect.main(["--run", run_id])
        except Exception:
            traceback.print_exc()
    try:
        _mark("inventory")
        import scene_inventory as _inventory

        _inventory.main(["--run", run_id, "--live"])
        _mark("report")
        # one pipeline, one instance set: the 3D viewer must show the same
        # objects the report and CAD show, so rebuild it from the inventory
        try:
            from .viewer import build_viewer_html

            build_viewer_html(run_dir)
        except Exception:
            traceback.print_exc()
        from .interactive_report import build_interactive_run_report

        build_interactive_run_report(run_id)
        _mark("done")
    except Exception:
        traceback.print_exc()
        _mark("failed")


def layer_summary_markdown(run_dir: Path) -> object:
    """First-eye page view of the VLM + detection layers: every phrase the
    VLM enumerated with where it went (instances or an unresolved reason)
    and the taxonomy checklist (found / missing / rejected). Returns a
    gr.update() no-op until the deep chain has produced the files."""
    run_dir = Path(run_dir)
    inventory_path = run_dir / "inventory" / "inventory.json"
    detections_path = run_dir / "detection" / "detections.json"
    if not inventory_path.exists() and not detections_path.exists():
        return gr.update()
    lines = []
    if inventory_path.exists():
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except Exception:
            inventory = {}
        phrases = inventory.get("phrases") or []
        if isinstance(phrases, dict):
            phrases = list(phrases.keys())
        objects = inventory.get("objects") or []
        unresolved = {}
        try:
            for item in json.loads(
                (run_dir / "inventory" / "unresolved.json").read_text(encoding="utf-8")
            ):
                unresolved[str(item.get("phrase"))] = str(item.get("reason", ""))
        except Exception:
            pass
        lines.append(f"**VLM 枚举 {len(phrases)} 项 → 去向**（枚举必有交代）")
        for phrase in phrases:
            count = sum(1 for o in objects if str(o.get("label")) == str(phrase))
            if count:
                lines.append(f"- {phrase}: {count} 个实例")
            else:
                lines.append(
                    f"- {phrase}: 未成实例 — {unresolved.get(str(phrase), '未记录原因')}"
                )
    if detections_path.exists():
        try:
            detections = json.loads(detections_path.read_text(encoding="utf-8"))
        except Exception:
            detections = {}
        found = detections.get("detections") or []
        missing = detections.get("missing") or []
        rejected = detections.get("rejected") or []
        lines.append("")
        lines.append(
            f"**装置检测清单**：检出 {len(found)} · 缺失 {len(missing)} · 拒绝 {len(rejected)}"
        )
        for det in found[:20]:
            lines.append(
                f"- #{det.get('number', '?')} [{det.get('category', '?')}] "
                f"{det.get('zh') or det.get('label') or det.get('item_id')} "
                f"(SAM {det.get('sam_score', det.get('score', '—'))})"
            )
        if missing:
            lines.append(
                "- 缺失/需现场核实: " + "、".join(
                    str(m.get("zh") or m.get("item_id")) for m in missing
                )
            )
        for rej in rejected[:5]:
            lines.append(
                f"- 拒绝 {rej.get('zh') or rej.get('item_id')}: {str(rej.get('reason', ''))[:120]}"
            )
    return gr.update(value="\n".join(lines))


def _review_markdown(run_dir: Path) -> str:
    review_path = Path(run_dir) / "review.json"
    if not review_path.exists():
        return "_未审核_"
    try:
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except Exception:
        return "_审核记录无法读取_"
    decision = review.get("decision", "?")
    status = review.get("overridden_status") or review.get("status") or ""
    return (
        f"**已审核** — {review.get('reviewer', '?')} · {decision}"
        + (f" → {status}" if decision == "overridden" and status else "")
        + (f"\n\n理由：{review.get('reason')}" if review.get("reason") else "")
        + (f"\n\n{review.get('reviewed_at') or review.get('created_at') or ''}")
    )


def _start_deep_report_chain(run_id: str) -> None:
    import threading

    threading.Thread(
        target=_run_deep_report_chain, args=(run_id,), daemon=True
    ).start()


def resume_interrupted_chains(runs_root: str | Path = "runs") -> list[str]:
    """A chain runs inside the app process; a restart mid-chain used to
    strand the run at the thin report forever. On startup every run whose
    status is still in-flight (or failed) is picked up again — the chain
    is re-entrant, so finished stages are skipped, not redone."""
    resumed = []
    root = Path(runs_root)
    if not root.exists():
        return resumed
    for status_path in root.glob("*/deep_report.status"):
        try:
            state = status_path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if state in {"detect", "inventory", "report", "failed"}:
            _start_deep_report_chain(status_path.parent.name)
            resumed.append(status_path.parent.name)
    return resumed


def analyze_run(
    pipeline: Any,
    image_1: str | None,
    image_2: str | None,
    image_3: str | None,
    image_4: str | None,
    camera_height_m: float,
    selected_policies: list[str] | None = None,
    available_policies: dict[str, PolicySpec] | None = None,
) -> tuple[object, ...]:
    image_paths = [
        path for path in (image_1, image_2, image_3, image_4) if path
    ]
    if not image_paths:
        return _analysis_error_outputs(
            "Upload at least one workcell view before analysis."
        )

    try:
        run_id = uuid4().hex
        capture = CaptureRun(
            run_id=run_id,
            image_paths=[str(path) for path in image_paths],
            camera_height_m=camera_height_m,
            # Only reviewed, build-time specs are selectable; a stale or
            # unknown id from the client degrades to "not selected".
            policies=[
                available_policies[policy_id]
                for policy_id in selected_policies or []
                if policy_id in (available_policies or {})
            ],
        )
        assessment = pipeline.run_assessment(capture)
        # the quick verdict returns now; the deep chain (device detection
        # -> inventory refinement -> interactive report) runs behind it so
        # the 报告 tab upgrades from summary to full dossier on its own
        _start_deep_report_chain(run_id)
        paths = pipeline.store.paths(run_id)
        scene = pipeline.store.load_json(paths.scene_json, SceneMap)
        policy_results = None
        policy_specs = None
        if paths.policies_json.exists():
            import json as _json

            payload = _json.loads(
                paths.policies_json.read_text(encoding="utf-8")
            )
            policy_results = payload["results"]
            policy_specs = {
                spec["policy_id"]: spec for spec in payload.get("specs", [])
            }
        status_class = assessment.status.value.lower().replace("_", "-")
        return (
            run_id,
            gr.update(
                value=_status_copy(assessment, scene, policy_results, policy_specs)
                + "\n\n⏳ **完整交互报告后台生成中（实测约 5–8 分钟）** — "
                "到 报告 tab 点「生成/查看报告」，好了会自动显示完整版。",
                elem_classes=["result-status", f"status-{status_class}"],
            ),
            str(paths.point_cloud_glb),
            str(paths.topdown_png),
            {
                "assessment": assessment.model_dump(mode="json"),
                "scene_map": scene.model_dump(mode="json"),
                "policy_results": policy_results,
                "policy_specs": policy_specs,
            },
            [],
            gr.update(value="", interactive=True),
            gr.update(interactive=True),
        )
    except ProviderError as exc:
        return _analysis_error_outputs(_provider_error_copy(exc))
    except Exception as exc:
        return _analysis_error_outputs(
            f"Local processing failed ({type(exc).__name__}). Check the uploaded "
            "files and local artifacts, then retry."
        )


def load_run_evidence(
    pipeline: Any,
    run_id: str | None,
) -> tuple[list[tuple[str, str]], str | None]:
    """Reviewer evidence for the current run: mask-overlay gallery items and
    the viewer.html download. Reads only files the pipeline already wrote;
    both artifacts are fail-soft in the pipeline, so either may be absent —
    and a cleared run (run_id None) empties the section."""
    if not run_id:
        return [], None
    paths = pipeline.store.paths(run_id)
    overlays = [
        (str(path), path.stem.removesuffix("_overlay"))
        for path in sorted(paths.evidence_dir.glob("*_overlay.png"))
    ]
    viewer = str(paths.viewer_html) if paths.viewer_html.is_file() else None
    return overlays, viewer


def answer_run_question(
    pipeline: Any,
    question: str,
    history: list[dict[str, object]] | None,
    run_id: str | None,
) -> tuple[list[dict[str, object]], str]:
    if not run_id:
        raise gr.Error("Analyze a workcell before asking about this run.")
    normalized_question = question.strip()
    if not normalized_question:
        raise gr.Error("Enter a question about the current run.")
    try:
        answer = GroundedAnswer.model_validate(
            pipeline.answer_question(run_id, normalized_question)
        )
    except ProviderError as exc:
        raise gr.Error(
            _provider_error_copy(exc), print_exception=False
        ) from None

    fact_ids = ", ".join(answer.fact_ids) if answer.fact_ids else "none"
    updated_history = list(history or [])
    updated_history.extend(
        [
            {"role": "user", "content": normalized_question},
            {
                "role": "assistant",
                "content": f"{answer.answer}\n\nFact IDs: `{fact_ids}`",
            },
        ]
    )
    return updated_history, ""


HISTORY_HEADERS = [
    "run_id",
    "created_at",
    "operator",
    "tier",
    "status",
    "policies",
    "distance",
    "disposition",
]
OVERRIDE_STATUSES = ["PASS", "FAIL", "INSUFFICIENT_EVIDENCE"]


def _load_policy_payload(
    paths: Any,
) -> tuple[list[dict] | None, dict[str, dict] | None]:
    """Policy results + specs-by-id from policies.json for the history card.
    Handles the current {"specs", "results"} envelope and older bare-list
    files; anything unreadable degrades to (None, None) so a card still
    renders without its policy section."""
    try:
        payload = json.loads(paths.policies_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if isinstance(payload, list):
        return _sanitize_policy_results(payload), {}
    if isinstance(payload, dict):
        specs = {
            spec["policy_id"]: spec
            for spec in payload.get("specs", [])
            if isinstance(spec, dict) and "policy_id" in spec
        }
        results = payload.get("results")
        return (
            _sanitize_policy_results(results)
            if isinstance(results, list)
            else None
        ), specs
    return None, None


def _sanitize_policy_results(results: list) -> list[dict]:
    """Same tolerance as report.py's _policy_rows: skip non-dict rows and
    default missing policy_id/status to '?' so the card renders instead of
    crashing on a malformed policies.json."""
    return [
        {"policy_id": "?", "status": "?", **row}
        for row in results
        if isinstance(row, dict)
    ]


def _disposition_copy(review: ReviewDisposition | None) -> str:
    if review is None:
        return "_No disposition recorded for this run._"
    if review.decision == "confirmed":
        ruling = "**confirmed** the machine verdict"
    else:
        target = (
            review.overridden_status.value if review.overridden_status else "?"
        )
        ruling = f"**overrode** the machine verdict to **{target}**"
    reason = f" — {review.reason}" if review.reason else ""
    return f"{review.reviewer} {ruling} at {review.created_at}{reason}."


def _load_disposition(pipeline: Any, paths: Any) -> ReviewDisposition | None:
    try:
        return pipeline.store.load_json(paths.review_json, ReviewDisposition)
    except Exception:
        return None


def _history_distance_copy(
    distance: float | None, budget: float | None
) -> str | None:
    # Never print a bare decimal when a band exists: it is part of the
    # measurement (same rule as the verdict card and report).
    if distance is None:
        return None
    if budget is not None:
        return f"{distance:.2f} ± {budget:.2f} m"
    return f"{distance:.2f} m"


def list_history(pipeline: Any) -> list[list[object]]:
    """Rows for the History table, straight from the artifact index."""
    return [
        [
            run["run_id"],
            run["created_at"],
            run["operator"],
            run["capture_tier"],
            run["status"],
            run["worst_policy"],
            _history_distance_copy(
                run["distance"], run["distance_error_budget_m"]
            ),
            run["disposition"],
        ]
        for run in pipeline.store.list_runs()
    ]


def load_history_run(
    pipeline: Any,
    rows: list[list[object]] | None,
    evt: gr.SelectData,
) -> tuple[object, ...]:
    """Selecting a History row loads that run's verdict card, top-down
    evidence and current disposition, and resets the disposition form so a
    ruling composed against the previous run cannot be saved onto this one.
    Every artifact is optional: a partial run renders whatever it has
    instead of erroring the whole tab."""
    try:
        run_id = str(rows[evt.index[0]][0] or "")
    except (TypeError, IndexError, KeyError):
        run_id = ""
    if not run_id:
        raise gr.Error("Select a run row from the history table.")
    paths = pipeline.store.paths(run_id)
    card = "### NO ASSESSMENT\n\nThis run has no readable assessment.json."
    status_class = "status-idle"
    try:
        assessment = pipeline.store.load_json(paths.assessment_json, Assessment)
    except Exception:
        assessment = None
    if assessment is not None:
        try:
            scene = pipeline.store.load_json(paths.scene_json, SceneMap)
        except Exception:
            scene = None
        policy_results, policy_specs = _load_policy_payload(paths)
        card = _status_copy(assessment, scene, policy_results, policy_specs)
        status_class = (
            f"status-{assessment.status.value.lower().replace('_', '-')}"
        )
    return (
        run_id,
        gr.update(
            value=card,
            elem_classes=["result-status", status_class],
        ),
        str(paths.topdown_png) if paths.topdown_png.is_file() else None,
        gr.update(value=_disposition_copy(_load_disposition(pipeline, paths))),
        gr.update(value=""),
        gr.update(value="confirmed"),
        gr.update(value=None),
        gr.update(value=""),
    )


def save_disposition(
    pipeline: Any,
    run_id: str | None,
    reviewer: str,
    decision: str,
    overridden_status: str | None,
    reason: str,
) -> tuple[object, list[list[object]]]:
    """Write the reviewer's ruling beside the machine verdict. The original
    assessment.json is never rewritten: the disposition is a separate,
    auditable layer in review.json. Returns the disposition copy plus fresh
    History rows so the table never shows a stale disposition column."""
    if not run_id:
        raise gr.Error("Select a run in the History table before saving.")
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise gr.Error("Enter the reviewer name.")
    reason = (reason or "").strip()
    if decision == "overridden" and not reason:
        raise gr.Error("Overriding a machine verdict requires a reason.")
    if decision == "overridden" and not overridden_status:
        raise gr.Error("Pick the status the override asserts.")
    # A disposition rules on a machine verdict, so one must exist — this also
    # stops a save from recreating a deleted run directory as a ghost run.
    try:
        assessment = pipeline.store.load_json(
            pipeline.store.paths(run_id).assessment_json, Assessment
        )
    except Exception:
        raise gr.Error(
            "This run has no machine assessment to rule on."
        ) from None
    if (
        decision == "overridden"
        and overridden_status == assessment.status.value
    ):
        raise gr.Error(
            "The override matches the machine verdict "
            f"({assessment.status.value}); confirm it instead."
        )
    try:
        disposition = ReviewDisposition(
            run_id=run_id,
            reviewer=reviewer,
            decision=decision,
            overridden_status=(
                overridden_status if decision == "overridden" else None
            ),
            reason=reason,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    except ValidationError as exc:
        raise gr.Error(f"Disposition is invalid: {exc}") from None
    pipeline.store.save_json(
        pipeline.store.paths(run_id).review_json, disposition
    )
    return gr.update(value=_disposition_copy(disposition)), list_history(pipeline)


VIDEO_MAX_FRAMES = 60
VIDEO_SAMPLE_RATES = [("0.5 fps", 0.5), ("1 fps", 1.0), ("2 fps", 2.0)]
VIDEO_DEFAULT_LABELS = "person, forklift"
_VIDEO_RULE_COPY = {
    "R1_zone": "R1 keep-clear zone",
    "R2_min_distance": "R2 person-to-vehicle ≥ 2.0 m",
    "R3_speed": "R3 person speed ≤ 1.5 m/s",
}
_VIDEO_STATUS_CLASS = {
    "FAIL": "status-fail",
    "NEEDS_REVIEW": "status-needs-review",
    "PASS": "status-pass",
    "NO_DATA": "status-insufficient-evidence",
}
_VIDEO_STATUS_COUNT_ORDER = ("FAIL", "NEEDS_REVIEW", "PASS", "NO_DATA")


def _parse_video_labels(labels_text: str) -> tuple[str, ...]:
    return tuple(
        label.strip() for label in (labels_text or "").split(",") if label.strip()
    )


def video_cost_copy(sample_fps: float, labels_text: str) -> str:
    """The explicit spend line: SAM is a paid call per sampled frame per
    label, capped by the frame budget."""
    n_labels = max(1, len(_parse_video_labels(labels_text)))
    calls = VIDEO_MAX_FRAMES * n_labels
    seconds = VIDEO_MAX_FRAMES / float(sample_fps)
    return (
        f"~{calls} SAM calls ≈ ${calls * COST_PER_SAM_CALL_USD:.2f} — live "
        f"provider spend (≤{VIDEO_MAX_FRAMES} frames × {n_labels} label(s), "
        f"plus 3 MoGe keyframe calls). At {sample_fps:g} fps that covers the "
        f"first {seconds:.0f} s of footage."
    )


def _video_status_copy(report: dict) -> tuple[str, str]:
    """(card markdown, status css class) for a video report. Abstention is a
    first-class outcome: when the floor fit or masks fail there is no
    verdict and no fallback."""
    tier = report.get("tier", {})
    tier_line = (
        f"Uncalibrated video-mono tier (±{tier.get('band_m', BAND_M)} m band); "
        f"{tier.get('calibrated_reference', '')}"
    )
    abstained = report.get("abstained")
    if abstained:
        return (
            "### NO VERDICT\n\n"
            f"{abstained}. States and trajectories require a metric floor "
            "and detected objects; no fallback result was generated.\n\n"
            f"{tier_line}",
            "status-insufficient-evidence",
        )
    verdicts = report.get("verdicts", {})
    overall = verdicts.get("overall", "NO_DATA")
    timelines = report.get("timelines", {})
    lines = [
        f"### {overall}",
        f"Worst state across {len(report.get('sampled_frame_ids', []))} "
        f"sampled frames ({report.get('step_seconds', '?')} s step).",
    ]
    for rule in RULE_NAMES:
        counts = {status: 0 for status in _VIDEO_STATUS_COUNT_ORDER}
        for status in timelines.get(rule, {}).values():
            counts[status] = counts.get(status, 0) + 1
        detail = ", ".join(
            f"{count} {status}"
            for status, count in counts.items()
            if count
        )
        lines.append(
            f"- `{verdicts.get(rule, 'NO_DATA')}` {_VIDEO_RULE_COPY[rule]} — "
            f"{detail or 'no frames'}"
        )
    floor = report.get("floor", {})
    if floor.get("fitted"):
        spread = floor.get("height_spread_m")
        spread_copy = (
            f", height spread {spread} m across keyframes"
            if spread is not None
            else ""
        )
        lines.append(
            f"Floor: fitted from MoGe keyframes — camera height "
            f"{floor.get('camera_height_m')} m, inlier fraction "
            f"{floor.get('inlier_fraction')}{spread_copy}."
        )
    spend = report.get("spend", {})
    lines.append(
        f"Spend: {spend.get('sam_calls', 0)} SAM calls ≈ "
        f"${spend.get('sam_cost_usd', 0):.2f} + "
        f"{spend.get('moge_keyframes', 0)} MoGe keyframes "
        "(cached rerun/replay is free)."
    )
    lines.append(tier_line)
    return "\n\n".join(lines), _VIDEO_STATUS_CLASS.get(overall, "status-idle")


def list_video_runs(pipeline: Any) -> list[tuple[str, str]]:
    """(label, run_id) choices for the replay dropdown: every cached run
    with a video_report.json, newest first. Same degradation contract as
    list_runs: a missing or corrupt manifest never hides the run."""
    choices: list[tuple[str, str, str]] = []
    root = pipeline.store.root
    if not root.is_dir():
        return []
    for run_dir in root.iterdir():
        if not run_dir.is_dir():
            continue
        try:
            paths = video_paths(pipeline.store, run_dir.name)
        except ValueError:
            continue
        if not paths.report_json.is_file():
            continue
        created = ""
        try:
            manifest = json.loads(
                paths.manifest_json.read_text(encoding="utf-8")
            )
            if isinstance(manifest.get("created_at"), str):
                created = manifest["created_at"]
        except (OSError, ValueError):
            pass
        label = f"{run_dir.name} — {created[:19]}" if created else run_dir.name
        choices.append((created, label, run_dir.name))
    choices.sort(key=lambda entry: (entry[0], entry[2]), reverse=True)
    return [(label, run_id) for _, label, run_id in choices]


def _video_error_outputs(error_copy: str) -> tuple[object, ...]:
    return (
        gr.update(
            value=(
                "### RUN ERROR\n\n"
                "Video analysis failed. No assessment was produced.\n\n"
                f"{error_copy}\n\n"
                "No fallback result was generated."
            ),
            elem_classes=["result-status", "status-error"],
        ),
        None,
        None,
        {"status": "RUN_ERROR", "error": error_copy, "report": None},
        gr.update(),
    )


def analyze_video(
    pipeline: Any,
    video_path: str | None,
    sample_fps: float,
    labels_text: str,
    zone_wkt: str,
) -> tuple[object, ...]:
    """Run a paid video assessment and render its artifacts. Mirrors
    analyze_run's error discipline: provider failures surface as a RUN
    ERROR card, never a fallback result."""
    if not video_path:
        return _video_error_outputs("Upload a video before analysis.")
    labels = _parse_video_labels(labels_text)
    if not labels:
        return _video_error_outputs("Enter at least one object label.")
    try:
        run_id = uuid4().hex
        report = run_video_assessment(
            video_path,
            store=pipeline.store,
            run_id=run_id,
            sample_fps=float(sample_fps),
            max_frames=VIDEO_MAX_FRAMES,
            labels=labels,
            zone_wkt=(zone_wkt or "").strip() or None,
        )
    except ProviderError as exc:
        return _video_error_outputs(_provider_error_copy(exc))
    except Exception as exc:
        return _video_error_outputs(
            f"Video processing failed ({type(exc).__name__}: {exc}). Check "
            "the uploaded file and settings, then retry."
        )
    paths = video_paths(pipeline.store, run_id)
    card, status_class = _video_status_copy(report)
    return (
        gr.update(value=card, elem_classes=["result-status", status_class]),
        str(paths.overlay_gif) if paths.overlay_gif.is_file() else None,
        str(paths.topdown_png) if paths.topdown_png.is_file() else None,
        report,
        gr.update(choices=list_video_runs(pipeline), value=run_id),
    )


def load_video_run(pipeline: Any, run_id: str | None) -> tuple[object, ...]:
    """Replay a processed video run from its cached artifacts — no provider
    spend. Every artifact is optional: a partial run renders what it has."""
    if not run_id:
        return (
            gr.update(
                value=(
                    "### No run selected\n\nPick a processed video run to "
                    "replay its artifacts without provider spend."
                ),
                elem_classes=["result-status", "status-idle"],
            ),
            None,
            None,
            None,
        )
    paths = video_paths(pipeline.store, run_id)
    try:
        report = json.loads(paths.report_json.read_text(encoding="utf-8"))
        card, status_class = _video_status_copy(report)
    except (OSError, ValueError):
        report = None
        card = (
            "### NO REPORT\n\nThis run has no readable video_report.json."
        )
        status_class = "status-idle"
    return (
        gr.update(value=card, elem_classes=["result-status", status_class]),
        str(paths.overlay_gif) if paths.overlay_gif.is_file() else None,
        str(paths.topdown_png) if paths.topdown_png.is_file() else None,
        report,
    )


def build_app(
    pipeline: Any | None = None,
    policies_dir: str | Path = POLICIES_DIR,
) -> gr.Blocks:
    service = pipeline if pipeline is not None else EHSAssessmentPipeline()
    specs = load_policy_specs(policies_dir)
    supported = {
        spec.policy_id: spec for spec in specs if not spec.unsupported_reason
    }
    refused = [spec for spec in specs if spec.unsupported_reason]
    analyze = partial(analyze_run, service, available_policies=supported)
    ask = partial(answer_run_question, service)

    with gr.Blocks(
        analytics_enabled=False,
        title="EHS Spatial Inspection Workbench",
        fill_width=True,
    ) as demo:
        run_id = gr.State(None)
        gr.Markdown(
            "# EHS Spatial Inspection Workbench\n"
            "Photo-based spatial evidence (1-4 views) for a deterministic "
            "fence-clearance demo.",
            elem_classes="workbench-title",
        )
        gr.Markdown(
            f"**{DEMO_RULE_COPY}.** Distances are approximate; "
            "climb output is review-only.",
            elem_classes="rule-note",
        )

        with gr.Tabs():
            with gr.Tab("Workbench"):
                with gr.Row(elem_classes="workbench-layout"):
                    with gr.Column(scale=4, min_width=300, elem_classes="capture-rail"):
                        gr.Markdown("## Capture", elem_classes="section-heading")
                        uploads = [
                            gr.Image(
                                label=label,
                                type="filepath",
                                sources=["upload"],
                                interactive=True,
                                height=148,
                                buttons=["fullscreen"],
                            )
                            for label in (
                                "View 1 — workcell front (required)",
                                "View 2 — workcell right (optional)",
                                "View 3 — workcell rear (optional)",
                                "View 4 — workcell left (optional)",
                            )
                        ]
                        camera_height = gr.Number(
                            value=1.5,
                            label="Camera height (m)",
                            info="Measured lens height above the factory floor.",
                            minimum=0.1,
                            step=0.1,
                            precision=2,
                        )
                        with gr.Accordion(
                            "Policies", open=False, elem_classes="policy-rail"
                        ):
                            gr.Markdown(POLICY_REVIEW_COPY)
                            policy_selector = gr.CheckboxGroup(
                                choices=[
                                    (_policy_label(spec), policy_id)
                                    for policy_id, spec in supported.items()
                                ],
                                value=[],
                                label="Compiled policy specs",
                                info=(
                                    "Selected specs are evaluated "
                                    "deterministically against the "
                                    "reconstructed scene."
                                ),
                            )
                            # A refusal is a feature: the compiler said why it
                            # cannot measure this rule, so the operator sees a
                            # disabled entry with that reason, not silence.
                            for spec in refused:
                                gr.Checkbox(
                                    value=False,
                                    interactive=False,
                                    label=_policy_label(spec),
                                    info=(
                                        "Compiler refusal: "
                                        f"{spec.unsupported_reason}"
                                    ),
                                )
                        analyze_button = gr.Button(
                            "Analyze workcell",
                            variant="primary",
                            elem_classes="analyze-action",
                        )

                    with gr.Column(scale=7, min_width=420, elem_classes="evidence-canvas"):
                        gr.Markdown("## Evidence", elem_classes="section-heading")
                        status = gr.Markdown(
                            "### Awaiting capture\n\nUpload 1-4 views to begin "
                            "(more views = stronger evidence).",
                            elem_classes=["result-status", "status-idle"],
                        )
                        with gr.Row(elem_classes="evidence-views"):
                            point_cloud = gr.Model3D(
                                label="3D point cloud",
                                display_mode="point_cloud",
                                interactive=False,
                                height=420,
                            )
                            topdown = gr.Image(
                                label="Top-down evidence",
                                type="filepath",
                                interactive=False,
                                height=420,
                                buttons=["fullscreen"],
                            )
                        # VLM enumeration + taxonomy detection: what the
                        # models said is in the picture and where each
                        # item went — filled in as the deep chain lands
                        layer_summary = gr.Markdown(
                            "_VLM 枚举与装置检测：完整报告链跑完后在此显示_"
                        )
                        structured_result = gr.JSON(
                            label="Assessment + SceneMap",
                            open=False,
                            height=320,
                        )
                        with gr.Accordion(
                            "Reviewer evidence", open=False, elem_classes="evidence-extras"
                        ):
                            overlay_gallery = gr.Gallery(
                                value=[],
                                label="Mask overlays on the captured views",
                                columns=2,
                                height=320,
                                interactive=False,
                            )
                            viewer_file = gr.File(
                                label=(
                                    "Interactive 3D viewer — download viewer.html "
                                    "and open it in a browser"
                                ),
                                interactive=False,
                            )

                with gr.Column(elem_classes="chat-zone"):
                    gr.Markdown("## Ask about this run", elem_classes="section-heading")
                    chatbot = gr.Chatbot(
                        value=[],
                        label="Grounded answers",
                        height=260,
                        buttons=["copy", "copy_all"],
                        placeholder="Analysis enables questions grounded in SceneMap facts.",
                    )
                    with gr.Row(elem_classes="chat-controls"):
                        question = gr.Textbox(
                            label="Question",
                            placeholder="Ask about a measured fact",
                            interactive=False,
                            scale=5,
                        )
                        ask_button = gr.Button(
                            "Ask about this run",
                            interactive=False,
                            scale=1,
                            elem_classes="ask-action",
                        )

            with gr.Tab("Video"):
                with gr.Row(elem_classes="workbench-layout"):
                    with gr.Column(scale=4, min_width=300, elem_classes="capture-rail"):
                        gr.Markdown("## Video capture", elem_classes="section-heading")
                        video_input = gr.Video(
                            label="Fixed-camera video",
                            sources=["upload"],
                            interactive=True,
                        )
                        video_sample_rate = gr.Dropdown(
                            choices=list(VIDEO_SAMPLE_RATES),
                            value=1.0,
                            label="Sample rate (fps)",
                            info="Sampled frames drive SAM spend and the speed band.",
                        )
                        video_labels = gr.Textbox(
                            value=VIDEO_DEFAULT_LABELS,
                            label="Object labels",
                            info=(
                                "Comma-separated SAM prompts. Rules are "
                                "person-centric: 'person' plus the machinery "
                                "to keep apart from."
                            ),
                        )
                        video_zone = gr.Textbox(
                            value="",
                            label="Keep-clear zone WKT (optional)",
                            placeholder="POLYGON ((x y, ...)) in floor-plane metres",
                            info=(
                                "Author it from a previous run's top-down "
                                "plot; leave empty to skip the zone rule."
                            ),
                        )
                        video_cost_line = gr.Markdown(
                            video_cost_copy(1.0, VIDEO_DEFAULT_LABELS),
                            elem_classes="rule-note",
                        )
                        video_analyze_button = gr.Button(
                            "Analyze video",
                            variant="primary",
                            elem_classes="analyze-action",
                        )
                        gr.Markdown(
                            "## Processed runs", elem_classes="section-heading"
                        )
                        video_runs_dropdown = gr.Dropdown(
                            choices=list_video_runs(service),
                            value=None,
                            label="Replay a processed video run",
                            info="Loads cached artifacts — no provider spend.",
                        )
                        video_refresh_button = gr.Button(
                            "Refresh processed runs"
                        )
                    with gr.Column(scale=7, min_width=420, elem_classes="evidence-canvas"):
                        gr.Markdown("## Video evidence", elem_classes="section-heading")
                        video_status = gr.Markdown(
                            "### Awaiting video\n\nUpload a fixed-camera clip "
                            "or replay a processed run.",
                            elem_classes=["result-status", "status-idle"],
                        )
                        with gr.Row(elem_classes="evidence-views"):
                            video_overlay = gr.Image(
                                label="Overlay (masks, tracks, per-frame verdicts)",
                                type="filepath",
                                interactive=False,
                                height=420,
                                buttons=["fullscreen"],
                            )
                            video_topdown = gr.Image(
                                label="Top-down trajectories",
                                type="filepath",
                                interactive=False,
                                height=420,
                                buttons=["fullscreen"],
                            )
                        video_report = gr.JSON(
                            label="Video report",
                            open=False,
                            height=320,
                        )

            with gr.Tab("补测 Refine"):
                gr.Markdown(
                    "## 框选补测\n"
                    "词表漏检的物体（透明护板、光幕柱、掠射角侧板…）："
                    "选 run → 在照片上**点两下**（左上角、右下角）画框 → "
                    "起标签 → SAM 分割并用该 run 自己的几何量尺寸；勾选回灌则"
                    "写入 scene 并重评该 run 的全部 policy。",
                    elem_classes="section-heading",
                )
                with gr.Row():
                    refine_run = gr.Dropdown(
                        label="Run", choices=[], allow_custom_value=True
                    )
                    refine_refresh = gr.Button("刷新 run 列表")
                refine_image = gr.Image(
                    label="点两下画框（左上 → 右下）", type="filepath",
                    interactive=False,
                )
                with gr.Row():
                    refine_x1 = gr.Number(label="x1", precision=0)
                    refine_y1 = gr.Number(label="y1", precision=0)
                    refine_x2 = gr.Number(label="x2", precision=0)
                    refine_y2 = gr.Number(label="y2", precision=0)
                with gr.Row():
                    refine_label = gr.Textbox(
                        label="标签（英文，如 safety fence / safety sensor）",
                        value="safety fence",
                    )
                    refine_apply = gr.Checkbox(
                        label="回灌判定（写入 scene + 重评 policies）", value=True
                    )
                    refine_go = gr.Button("SAM 补测", variant="primary")
                refine_result = gr.JSON(label="测量结果")
                refine_overlay = gr.Image(label="mask 证据", interactive=False)
                with gr.Row():
                    agent_say = gr.Textbox(
                        label="对话补测：一句话描述漏检物体（中文可）",
                        placeholder="例：入口右侧红色的斜坡挡板",
                    )
                    agent_go = gr.Button("让 agent 定位并补测")

                def _refine_runs() -> gr.Dropdown:
                    names = [r.get("run_id") for r in service.store.list_runs()]
                    return gr.Dropdown(choices=[n for n in names if n])

                def _refine_pick_image(name: str | None):
                    if not name:
                        return None
                    run_dir = service.store.paths(name).root
                    images = sorted((run_dir / "input").glob("image_*"))
                    return str(images[0]) if images else None

                def _refine_click(x1, y1, x2, y2, evt: gr.SelectData):
                    # Corner parity without extra state: an incomplete box
                    # (x2 empty) means this click is the second corner.
                    x, y = evt.index
                    if x1 is None or x2 is not None:
                        return x, y, None, None
                    return x1, y1, x, y

                def _refine_go(name, label, x1, y1, x2, y2, apply_it):
                    from .refine import RefineError, refine_region

                    if not name:
                        raise gr.Error("先选一个 run")
                    if not all(v is not None for v in (x1, y1, x2, y2)):
                        raise gr.Error("先在照片上点两下画框")
                    try:
                        result = refine_region(
                            str(name), str(label),
                            (int(x1), int(y1), int(x2), int(y2)),
                            runs_root=service.store.root,
                            apply=bool(apply_it),
                        )
                    except RefineError as exc:
                        raise gr.Error(str(exc)) from exc
                    overlay = result.pop("overlay_path", None)
                    return result, overlay

                refine_refresh.click(
                    _refine_runs, outputs=[refine_run],
                    concurrency_id=LOCAL_CONCURRENCY_ID,
                    concurrency_limit=1,
                )
                refine_run.change(
                    _refine_pick_image, inputs=[refine_run],
                    outputs=[refine_image],
                    concurrency_id=LOCAL_CONCURRENCY_ID,
                    concurrency_limit=1,
                )
                refine_image.select(
                    _refine_click,
                    inputs=[refine_x1, refine_y1, refine_x2, refine_y2],
                    outputs=[refine_x1, refine_y1, refine_x2, refine_y2],
                    concurrency_id=LOCAL_CONCURRENCY_ID,
                    concurrency_limit=1,
                )
                refine_go.click(
                    _refine_go,
                    inputs=[refine_run, refine_label, refine_x1, refine_y1,
                            refine_x2, refine_y2, refine_apply],
                    outputs=[refine_result, refine_overlay],
                    concurrency_id=PIPELINE_CONCURRENCY_ID,
                    concurrency_limit=1,
                )

                def _agent_go(name, instruction, apply_it):
                    from .agent import agent_refine
                    from .refine import RefineError

                    if not name:
                        raise gr.Error("先选一个 run")
                    if not (instruction or "").strip():
                        raise gr.Error("先描述要补测的物体")
                    try:
                        result = agent_refine(
                            str(name), instruction.strip(),
                            runs_root=service.store.root,
                            apply=bool(apply_it),
                        )
                    except (RefineError, ProviderError) as exc:
                        raise gr.Error(str(exc)) from exc
                    overlay = result.pop("overlay_path", None)
                    return result, overlay

                agent_go.click(
                    _agent_go,
                    inputs=[refine_run, agent_say, refine_apply],
                    outputs=[refine_result, refine_overlay],
                    concurrency_id=PIPELINE_CONCURRENCY_ID,
                    concurrency_limit=1,
                )

            with gr.Tab("报告 Report"):
                gr.Markdown(
                    "## 单 run 报告\n选 run → 生成自包含 HTML（判定+检测清单+"
                    "测量+回投+补测证据），页面内直接看，也可下载转发。",
                    elem_classes="section-heading",
                )
                _EXCLUDED_RUN_PREFIXES = (
                    "gen-", "real-anno-", "demo-", "poc-", "video-", "phase",
                )

                def _report_run_choices() -> list[tuple[str, str]]:
                    # operator submissions + the real-photo test set only,
                    # newest first, submission time in the label — this
                    # list IS the run history
                    from datetime import datetime

                    candidates = [
                        p for p in Path("runs").glob("*")
                        if (
                            (p / "manifest.json").exists()
                            or (p / "scene.json").exists()
                        )
                        and not p.name.startswith(_EXCLUDED_RUN_PREFIXES)
                    ]
                    candidates.sort(
                        key=lambda p: p.stat().st_mtime, reverse=True
                    )
                    choices = []
                    for path in candidates:
                        stamp = datetime.fromtimestamp(
                            path.stat().st_mtime
                        ).strftime("%m-%d %H:%M")
                        # history row = time · verdict · review state · id
                        verdict = "—"
                        try:
                            verdict = json.loads(
                                (path / "assessment.json").read_text(encoding="utf-8")
                            ).get("status", "—")
                        except Exception:
                            pass
                        reviewed = "已审核" if (path / "review.json").exists() else "未审核"
                        choices.append(
                            (f"{stamp}  ·  {verdict}  ·  {reviewed}  ·  {path.name}", path.name)
                        )
                    return choices

                _seed_runs = _report_run_choices()
                gr.Markdown(
                    "提交后：快速判定约 1 分钟（单图）/ 3-4 分钟（4 图）；"
                    "完整交互报告随后自动生成，再等约 5-8 分钟。"
                    "列表按提交时间排序，就是全部历史。"
                )
                with gr.Row():
                    report_run = gr.Dropdown(
                        label="Run（按时间倒序 = 历史）",
                        choices=_seed_runs,
                        value=_seed_runs[0][1] if _seed_runs else None,
                        allow_custom_value=True,
                    )
                    report_refresh = gr.Button("刷新 run 列表")
                    report_go = gr.Button("生成/查看报告", variant="primary")
                # the real thing is one standalone page served by the app —
                # this tab is the index into it plus an inline preview
                report_link = gr.HTML(
                    "<p>选 run 后点「生成/查看报告」，这里会出现"
                    "<b>完整报告页</b>链接（一个页面：报告 + 3D + CAD + 审核 + Agent）。</p>"
                )
                report_file = gr.File(label="下载", interactive=False)
                report_view = gr.HTML()
                report_cloud = gr.Model3D(
                    label="3D 点云（同一 run 的重建结果，可旋转）",
                    interactive=False,
                    height=520,
                )
                report_cad = gr.Image(
                    label="CAD 平面图（inventory 完整版：全实体 · 墙线 · cell 矩形 · 尺寸）",
                    type="filepath",
                    interactive=False,
                )
                # the report page is the one place: review and the agent
                # live beside the report they change
                with gr.Accordion("Review 审核 — 对本 run 签字", open=True):
                    review_summary = gr.Markdown("_未审核_")
                    with gr.Row():
                        review_reviewer = gr.Textbox(label="审核员")
                        review_decision = gr.Radio(
                            choices=["confirmed", "overridden"],
                            value="confirmed",
                            label="决定",
                        )
                        review_override = gr.Dropdown(
                            choices=list(OVERRIDE_STATUSES),
                            label="推翻后的状态（仅推翻时）",
                        )
                    review_reason = gr.Textbox(label="理由（推翻必填）", lines=2)
                    review_save = gr.Button("保存审核", variant="primary")
                with gr.Accordion(
                    "Agent 对话 — 追问 / 补测 / 纠错 / 调整策略（全部写入 run，进报告）",
                    open=True,
                ):
                    hub_chat = gr.Chatbot(label="对话记录", height=320)
                    with gr.Row():
                        hub_say = gr.Textbox(
                            label="说一句",
                            placeholder=(
                                "例：围栏离机器人多远？ / 右边黄色柱子帮我量 / "
                                "这块不是围栏，是导向挡板 / p01 改成 0.8m"
                            ),
                            scale=4,
                        )
                        hub_apply = gr.Checkbox(label="回灌判定", value=True)
                        hub_send = gr.Button("发送", variant="primary")

                def _report_runs() -> gr.Dropdown:
                    choices = _report_run_choices()
                    return gr.Dropdown(
                        choices=choices,
                        value=choices[0][1] if choices else None,
                    )

                def _report_go(name):
                    if not name:
                        # zero-friction default: newest run
                        choices = _report_run_choices()
                        if not choices:
                            return None, "<p>还没有任何 run。</p>"
                        name = choices[0][1]
                    run_dir = Path("runs") / name
                    state = analysis_state(run_dir)
                    running = _chain_running(run_dir)
                    # one generation of analysis for the whole product:
                    # a run without the current rules gets upgraded the
                    # moment someone opens it (skips detection when the
                    # run already has it), and the page says so
                    stage_now = (
                        (run_dir / "deep_report.status").read_text(encoding="utf-8").strip()
                        if (run_dir / "deep_report.status").exists()
                        else ""
                    )
                    # not current, or a chain that died mid-way (app restart,
                    # provider outage): kick it again — re-entrant, so only
                    # the missing stages run
                    if (state != "current" or stage_now == "failed") and not running:
                        _start_deep_report_chain(name)
                        running = True
                    stage = (
                        (run_dir / "deep_report.status").read_text(
                            encoding="utf-8"
                        ).strip()
                        if (run_dir / "deep_report.status").exists()
                        else ""
                    )
                    notice = (
                        "<p style='padding:8px 12px;background:#fff3cd;"
                        "border:1px solid #ffe08a;border-radius:6px'>{}</p>"
                    )
                    if state == "missing":
                        from .report import build_run_report

                        path = build_run_report(name)
                        banner = notice.format(
                            "⏳ 正在用最新分析生成完整交互报告（检测清单 / 矩形约束 / "
                            f"精修测量），当前阶段: {stage or 'detect'}。实测约 5-8 分钟，"
                            "期间重新点击「生成/查看报告」即可。下面先显示快速摘要。"
                        )
                    else:
                        from .interactive_report import (
                            build_interactive_run_report,
                        )

                        path = build_interactive_run_report(name)
                        banner = ""
                        if state == "stale":
                            banner = notice.format(
                                "⏳ 此 run 的分析早于当前规则，正在后台用最新分析重算"
                                f"（当前阶段: {stage or 'inventory'}，约 2-4 分钟）。"
                                "下面是升级前的版本，稍后重新点击即为最新。"
                            )
                        elif stage == "failed":
                            banner = notice.format(
                                "深度报告链失败，显示的是已有版本（服务器日志有 traceback）。"
                            )
                    html = path.read_text(encoding="utf-8")
                    if banner:
                        html = banner + html
                    framed = (
                        '<iframe style="width:100%;height:900px;border:1px '
                        'solid #ccc;border-radius:6px" srcdoc="'
                        + html.replace("&", "&amp;").replace('"', "&quot;")
                        + '"></iframe>'
                    )
                    cloud = run_dir / "geometry" / "point_cloud.glb"
                    cad = run_dir / "inventory" / "floor_plan.png"
                    if not cad.exists():
                        cad = run_dir / "topdown.png"
                    from .agent_hub import chat_history

                    link = (
                        f'<p style="font-size:16px"><a href="/report/{name}" target="_blank">'
                        f"▶ 打开完整报告页 /report/{name[:12]}…</a>"
                        "　·　一个页面：报告 + 3D + CAD + 审核签字 + Agent 对话"
                        f'　·　<a href="/report/{name}/file">下载 HTML</a></p>'
                    )
                    return (
                        str(path),
                        framed,
                        str(cloud) if cloud.exists() else None,
                        str(cad) if cad.exists() else None,
                        _review_markdown(run_dir),
                        chat_history(run_dir),
                        link,
                    )

                def _hub_save_review(name, reviewer, decision, override, reason):
                    if not name:
                        raise gr.Error("先选一个 run")
                    save_disposition(
                        service, name, reviewer, decision, override, reason
                    )
                    return _review_markdown(Path("runs") / name)

                def _hub_agent(name, message, apply_it, history):
                    from .agent import agent_refine
                    from .agent_hub import agent_turn
                    from .refine import RefineError

                    if not name:
                        raise gr.Error("先选一个 run")
                    if not (message or "").strip():
                        raise gr.Error("先说一句")

                    def _answer(question: str) -> str:
                        answer = GroundedAnswer.model_validate(
                            service.answer_question(name, question)
                        )
                        facts = ", ".join(answer.fact_ids) or "none"
                        return f"{answer.answer}\n\nFact IDs: `{facts}`"

                    def _refine(rid: str, instruction: str, apply: bool) -> dict:
                        try:
                            return agent_refine(
                                rid, instruction, runs_root=service.store.root,
                                apply=bool(apply),
                            )
                        except (RefineError, ProviderError) as exc:
                            return {"message": f"补测失败：{exc}"}

                    try:
                        out = agent_turn(
                            name, message.strip(), apply=bool(apply_it),
                            answer_fn=_answer, refine_fn=_refine,
                        )
                    except ProviderError as exc:
                        raise gr.Error(_provider_error_copy(exc)) from exc
                    history = list(history or [])
                    history.append({"role": "user", "content": f"[{out['intent']}] {message.strip()}"})
                    history.append({"role": "assistant", "content": out["reply"]})
                    return history, ""

                report_refresh.click(
                    _report_runs,
                    outputs=[report_run],
                    concurrency_id=REPORT_CONCURRENCY_ID,
                    concurrency_limit=1,
                )
                review_save.click(
                    _hub_save_review,
                    inputs=[report_run, review_reviewer, review_decision,
                            review_override, review_reason],
                    outputs=[review_summary],
                    concurrency_id=LOCAL_CONCURRENCY_ID,
                    concurrency_limit=1,
                )
                hub_send.click(
                    _hub_agent,
                    inputs=[report_run, hub_say, hub_apply, hub_chat],
                    outputs=[hub_chat, hub_say],
                    concurrency_id=PIPELINE_CONCURRENCY_ID,
                    concurrency_limit=1,
                )
                report_go.click(
                    _report_go,
                    inputs=[report_run],
                    outputs=[report_file, report_view, report_cloud, report_cad,
                             review_summary, hub_chat, report_link],
                    concurrency_id=REPORT_CONCURRENCY_ID,
                    concurrency_limit=1,
                )

        # First-eye page keeps up with the deep chain: once the inventory's
        # full CAD exists for the run on screen, it replaces the quick
        # top-down without the operator doing anything.
        cad_timer = gr.Timer(30)

        def _cad_refresh(current_run):
            if not current_run:
                return gr.update(), gr.update()
            run_dir = Path("runs") / str(current_run)
            full = run_dir / "inventory" / "floor_plan.png"
            cad = (
                gr.update(
                    value=str(full),
                    label="CAD 平面图（完整版 · 全实体 · cell 矩形）",
                )
                if full.exists()
                else gr.update()
            )
            return cad, layer_summary_markdown(run_dir)

        cad_timer.tick(
            _cad_refresh,
            inputs=[run_id],
            outputs=[topdown, layer_summary],
            concurrency_id=LOCAL_CONCURRENCY_ID,
            concurrency_limit=1,
        )

        # Evidence follows the run id rather than extending the analyze
        # tuple: the 8-output analyze contract stays stable, and a failed
        # re-analysis (run_id -> None) clears the evidence section too.
        run_id.change(
            partial(load_run_evidence, service),
            inputs=[run_id],
            outputs=[overlay_gallery, viewer_file],
            api_name="load_run_evidence",
            api_visibility="private",
            concurrency_id=LOCAL_CONCURRENCY_ID,
            concurrency_limit=1,
        )
        analyze_button.click(
            analyze,
            inputs=[*uploads, camera_height, policy_selector],
            outputs=[
                run_id,
                status,
                point_cloud,
                topdown,
                structured_result,
                chatbot,
                question,
                ask_button,
            ],
            api_name="analyze_workcell",
            concurrency_id=PIPELINE_CONCURRENCY_ID,
            concurrency_limit=1,
        )
        ask_button.click(
            ask,
            inputs=[question, chatbot, run_id],
            outputs=[chatbot, question],
            api_name="ask_about_run",
            concurrency_id=PIPELINE_CONCURRENCY_ID,
            concurrency_limit=1,
        )
        question.submit(
            ask,
            inputs=[question, chatbot, run_id],
            outputs=[chatbot, question],
            api_name="ask_about_run_from_enter",
            api_visibility="private",
            concurrency_id=PIPELINE_CONCURRENCY_ID,
            concurrency_limit=1,
        )


        # Video tab: analysis is paid, so it shares the provider lane;
        # cost-copy updates and cached-run replay are local-only.
        video_analyze_button.click(
            partial(analyze_video, service),
            inputs=[video_input, video_sample_rate, video_labels, video_zone],
            outputs=[
                video_status,
                video_overlay,
                video_topdown,
                video_report,
                video_runs_dropdown,
            ],
            api_name="analyze_video",
            concurrency_id=PIPELINE_CONCURRENCY_ID,
            concurrency_limit=1,
        )
        for cost_trigger in (video_sample_rate.change, video_labels.change):
            cost_trigger(
                video_cost_copy,
                inputs=[video_sample_rate, video_labels],
                outputs=[video_cost_line],
                api_visibility="private",
                concurrency_id=LOCAL_CONCURRENCY_ID,
                concurrency_limit=1,
            )
        video_refresh_button.click(
            lambda: gr.update(choices=list_video_runs(service)),
            outputs=[video_runs_dropdown],
            api_name="list_video_runs",
            api_visibility="private",
            concurrency_id=LOCAL_CONCURRENCY_ID,
            concurrency_limit=1,
        )
        # .input, not .change: replay only on user selection, so the
        # post-analysis dropdown refresh does not immediately reload.
        video_runs_dropdown.input(
            partial(load_video_run, service),
            inputs=[video_runs_dropdown],
            outputs=[video_status, video_overlay, video_topdown, video_report],
            api_name="load_video_run",
            api_visibility="private",
            concurrency_id=LOCAL_CONCURRENCY_ID,
            concurrency_limit=1,
        )

    return demo.queue(api_open=False, default_concurrency_limit=1)


APP_CSS = """
:root {
  --ehs-surface: oklch(96% 0.014 82);
  --ehs-panel: oklch(92% 0.018 82);
  --ehs-paper: oklch(98% 0.009 82);
  --ehs-ink: oklch(23% 0.026 252);
  --ehs-muted: oklch(43% 0.025 252);
  --ehs-line: oklch(73% 0.022 78);
  --ehs-navy: oklch(34% 0.08 252);
  --ehs-amber: oklch(78% 0.15 78);
  --ehs-amber-dark: oklch(54% 0.13 68);
  --ehs-pass: oklch(43% 0.09 151);
  --ehs-pass-bg: oklch(93% 0.04 151);
  --ehs-fail: oklch(45% 0.15 27);
  --ehs-fail-bg: oklch(94% 0.035 27);
  --ehs-warn: oklch(50% 0.11 73);
  --ehs-warn-bg: oklch(94% 0.045 80);
  --ehs-focus: oklch(58% 0.14 252);
}

.gradio-container {
  background: var(--ehs-surface) !important;
  color: var(--ehs-ink) !important;
  font-family: "Avenir Next", "Segoe UI Variable", "Segoe UI", sans-serif !important;
  font-size: 1rem !important;
  padding: clamp(16px, 3vw, 40px) !important;
}

.main {
  max-width: 1500px !important;
  margin-inline: auto !important;
}

.workbench-title h1 {
  color: var(--ehs-ink) !important;
  font-size: clamp(1.9rem, 3.5vw, 3.25rem) !important;
  line-height: 1.05 !important;
  letter-spacing: -0.035em !important;
  max-width: 18ch;
  margin: 0 !important;
}

.workbench-title p,
.rule-note p {
  color: var(--ehs-muted) !important;
  max-width: 68ch;
}

.rule-note {
  border-block: 1px solid var(--ehs-line);
  margin-block: 16px 28px !important;
  padding-block: 12px !important;
}

.workbench-layout {
  align-items: stretch !important;
  gap: clamp(20px, 3vw, 44px) !important;
}

.capture-rail,
.evidence-canvas,
.chat-zone {
  border: 1px solid var(--ehs-line) !important;
  border-radius: 4px !important;
  box-shadow: none !important;
}

.capture-rail {
  background: var(--ehs-panel) !important;
  padding: 20px !important;
}

.evidence-canvas,
.chat-zone {
  background: var(--ehs-paper) !important;
  padding: clamp(16px, 2vw, 28px) !important;
}

.section-heading h2 {
  color: var(--ehs-navy) !important;
  font-size: 0.8rem !important;
  font-weight: 750 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase;
  margin: 0 0 12px !important;
}

.capture-rail .image-container,
.evidence-canvas .model3d,
.evidence-canvas .image-container,
.evidence-canvas .json-holder,
.chat-zone .chatbot {
  border-color: var(--ehs-line) !important;
  border-radius: 3px !important;
  box-shadow: none !important;
}

.result-status {
  border: 1px solid var(--ehs-line) !important;
  border-radius: 3px !important;
  padding: 16px 20px !important;
  margin-bottom: 20px !important;
}

.result-status h3 {
  font-size: clamp(1.25rem, 2vw, 1.8rem) !important;
  letter-spacing: 0.03em !important;
  margin: 0 0 8px !important;
}

.status-idle { background: var(--ehs-panel) !important; }
.status-pass {
  background: var(--ehs-pass-bg) !important;
  border-color: var(--ehs-pass) !important;
}
.status-pass h3 { color: var(--ehs-pass) !important; }
.status-fail {
  background: var(--ehs-fail-bg) !important;
  border-color: var(--ehs-fail) !important;
}
.status-fail h3 { color: var(--ehs-fail) !important; }
.status-needs-review, .status-insufficient-evidence {
  background: var(--ehs-warn-bg) !important;
  border-color: var(--ehs-warn) !important;
}
.status-needs-review h3,
.status-insufficient-evidence h3 { color: var(--ehs-warn) !important; }
.status-error {
  background: var(--ehs-warn-bg) !important;
  border-color: var(--ehs-warn) !important;
}
.status-error h3 { color: var(--ehs-warn) !important; }

.analyze-action,
.ask-action,
button {
  min-height: 44px !important;
  border-radius: 3px !important;
  font-weight: 700 !important;
}

.analyze-action {
  background: var(--ehs-amber) !important;
  background-image: none !important;
  border-color: var(--ehs-amber-dark) !important;
  color: var(--ehs-ink) !important;
}

.analyze-action:hover { background: oklch(73% 0.15 75) !important; }
.analyze-action:active { background: oklch(68% 0.14 73) !important; }

.ask-action {
  background: var(--ehs-navy) !important;
  background-image: none !important;
  border-color: var(--ehs-navy) !important;
  color: var(--ehs-paper) !important;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible,
[tabindex]:focus-visible {
  outline: 3px solid var(--ehs-focus) !important;
  outline-offset: 2px !important;
}

.evidence-views,
.chat-controls { gap: 16px !important; }

.chat-zone {
  margin-top: clamp(24px, 4vw, 48px) !important;
}

@media (max-width: 860px) {
  .workbench-layout,
  .evidence-views,
  .chat-controls {
    flex-direction: column !important;
  }

  .capture-rail,
  .evidence-canvas {
    min-width: 0 !important;
  }

  .ask-action { width: 100% !important; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}
"""


__all__ = [
    "APP_CSS",
    "analyze_run",
    "analyze_video",
    "answer_run_question",
    "build_app",
    "list_history",
    "list_video_runs",
    "load_history_run",
    "load_policy_specs",
    "load_run_evidence",
    "load_video_run",
    "save_disposition",
    "video_cost_copy",
]
