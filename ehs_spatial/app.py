from functools import partial
from typing import Any
from uuid import uuid4

import gradio as gr

from .contracts import Assessment, CaptureRun, GroundedAnswer, SceneMap
from .pipeline import EHSAssessmentPipeline
from .providers.base import ProviderError


DEMO_RULE_COPY = "0.6 m demo rule — not an official EHS standard"
PIPELINE_CONCURRENCY_ID = "ehs-provider-pipeline"


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
            worst = result.get("violations") or []
            detail = (
                f" — worst {worst[0]['measured']}{worst[0]['unit']} "
                f"(limit {worst[0]['threshold']}{worst[0]['unit']})"
                if worst
                else ""
            )
            lines.append(f"- `{result['status']}` {result['policy_id']}{detail}")
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


def analyze_run(
    pipeline: Any,
    image_1: str | None,
    image_2: str | None,
    image_3: str | None,
    image_4: str | None,
    camera_height_m: float,
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
        )
        assessment = pipeline.run_assessment(capture)
        paths = pipeline.store.paths(run_id)
        scene = pipeline.store.load_json(paths.scene_json, SceneMap)
        policy_results = None
        if paths.policies_json.exists():
            import json as _json

            policy_results = _json.loads(
                paths.policies_json.read_text(encoding="utf-8")
            )
        status_class = assessment.status.value.lower().replace("_", "-")
        return (
            run_id,
            gr.update(
                value=_status_copy(assessment, scene, policy_results),
                elem_classes=["result-status", f"status-{status_class}"],
            ),
            str(paths.point_cloud_glb),
            str(paths.topdown_png),
            {
                "assessment": assessment.model_dump(mode="json"),
                "scene_map": scene.model_dump(mode="json"),
                "policy_results": policy_results,
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


def build_app(pipeline: Any | None = None) -> gr.Blocks:
    service = pipeline if pipeline is not None else EHSAssessmentPipeline()
    analyze = partial(analyze_run, service)
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
                structured_result = gr.JSON(
                    label="Assessment + SceneMap",
                    open=False,
                    height=320,
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

        analyze_button.click(
            analyze,
            inputs=[*uploads, camera_height],
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


__all__ = ["APP_CSS", "analyze_run", "answer_run_question", "build_app"]
