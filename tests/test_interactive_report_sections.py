"""The interactive run report carries all 14 sections of PRODUCT_PLAN §3,
each tagged with a stable data-section marker, and every section degrades
to a one-line 无 state instead of crashing when its source file is missing."""

import json
import re
from pathlib import Path

import pytest

from ehs_spatial.interactive_report import build_interactive_run_report

REPO = Path(__file__).parents[1]
SECTIONS = [
    "header", "verdicts", "phrases", "detections", "photo", "plan", "viewer",
    "measurements", "reprojection", "gallery", "refinements", "chat", "review",
    "appendix",
]


def _markers(html: str) -> list[str]:
    return re.findall(r'data-section="([a-z]+)"', html)


@pytest.mark.skipif(
    not (REPO / "runs" / "user-bor1-02" / "inventory" / "inventory.json").exists(),
    reason="BOR1 run not present",
)
def test_bor1_report_has_all_14_sections(monkeypatch):
    monkeypatch.chdir(REPO)
    out = build_interactive_run_report("user-bor1-02")
    html = out.read_text(encoding="utf-8")
    found = _markers(html)
    assert found == SECTIONS, found  # plan order, each exactly once
    # the new sections carry real content on this run, not just markers
    assert "枚举必有交代" in html
    assert "拒绝项" in html and "crop check" in html
    assert "cell 矩形约束" in html and "开口" in html and "邻 cell 结构" in html
    assert "尺度来源与置信" in html and "moge_anchor" in html
    assert "frame_0004_overlay" in html  # every evidence frame, not just the first
    assert "阈值已按本 run 调整" not in html.split('data-section="verdicts"')[1].split("<h4")[0]
    assert out.stat().st_size < 12 * 1024 * 1024


def test_minimal_run_degrades_gracefully(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "run-min"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "run_id": "run-min", "created_at": "2026-09-01T00:00:00+00:00",
        "operator": "t", "capture_tier": "mono",
        "providers": {"gemini_model": "g", "code_version": "abc"},
    }))
    (run / "scene.json").write_text(json.dumps({
        "scale_source": "camera_height", "scale_factor": 1.0, "scale_confidence": None,
        "entities": [], "warnings": ["w1"],
    }))
    (run / "policies.json").write_text(json.dumps({
        "specs": [{"policy_id": "p01", "predicate": "min_separation", "threshold": 0.6,
                   "unit": "m", "adjusted_for_run": True}],
        "results": [{"policy_id": "p01", "status": "PASS", "warnings": [], "violations": []}],
    }))
    monkeypatch.chdir(tmp_path)
    out = build_interactive_run_report("run-min")
    html = out.read_text(encoding="utf-8")
    assert _markers(html) == SECTIONS
    assert "无判定" in html  # no assessment.json -> neutral badge
    assert "无审核记录" in html and "无对话记录" in html
    assert "阈值已按本 run 调整" in html
    assert "<li class=\"mono\" style=\"font-size:12.5px\">w1</li>" in html


def test_chat_and_review_render_both_entry_shapes(tmp_path, monkeypatch):
    run = tmp_path / "runs" / "run-chat"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"run_id": "run-chat", "created_at": "t0", "providers": {}}))
    (run / "scene.json").write_text(json.dumps({"entities": [], "warnings": []}))
    (run / "policies.json").write_text(json.dumps({"specs": [], "results": []}))
    (run / "review.json").write_text(json.dumps({
        "run_id": "run-chat", "reviewer": "adam", "decision": "overridden",
        "overridden_status": "PASS", "reason": "fence is a neighbour cell", "created_at": "2026-09-08T10:00:00+00:00",
    }))
    lines = [
        {"type": "gemini_cursor", "interaction_id": "x"},  # cursor rows are not turns
        {"type": "chat_turn", "user": {"role": "user", "content": "围栏离机器人多远？"},
         "assistant": {"role": "assistant", "content": "0.42 m", "fact_ids": ["fact-1"]}, "interaction_id": "y"},
        {"type": "agent_turn", "ts": "2026-09-08T10:05:00+00:00", "intent": "relabel",
         "user": "把 3 号改成 bollard", "assistant": "已改 1 处", "changed": True},
    ]
    (run / "chat.jsonl").write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n")
    monkeypatch.chdir(tmp_path)
    html = build_interactive_run_report("run-chat").read_text(encoding="utf-8")
    chat = html.split('data-section="chat"')[1].split("<h4")[0]
    assert chat.count('<div class="policy">') == 2  # cursor row skipped
    assert "围栏离机器人多远？" in chat and "0.42 m" in chat and "facts: fact-1" in chat
    assert ">relabel<" in chat and "已改动 run" in chat and "2026-09-08T10:05:00+00:00" in chat
    review = html.split('data-section="review"')[1].split("<h4")[0]
    assert "adam" in review and "overridden" in review and "PASS" in review and "fence is a neighbour cell" in review
    assert "推翻机器判定 → PASS" in html  # header 审核结论 line
