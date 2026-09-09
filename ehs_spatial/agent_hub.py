"""One conversation surface per run, five verbs, everything written back.

The operator types one sentence about the run they are looking at; the
hub decides what kind of request it is and routes it:

  ask            grounded question over the SceneMap facts
  refine         locate + segment + measure something the pipeline missed
  relabel        "this is not a fence, it is a guide deflector" -> entity
                 label changes, policies re-evaluate
  policy_adjust  "use 0.8 m for p01 in this cell" -> per-run threshold,
                 policies re-evaluate

Every turn is appended to the run's chat.jsonl (type "agent_turn") so the
interactive report's 对话记录 section shows exactly what was asked and
what changed. Intent detection is deterministic keyword routing on
purpose: it is testable, explainable in the log, and never spends a VLM
call to decide what to do.
"""

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

INTENTS = ("ask", "refine", "relabel", "policy_adjust")

_RELABEL_SPLIT = re.compile(
    r"(不是|其实是|应该是|改成|标成|is not|should be|relabel(?:ed)? (?:as|to))",
    re.IGNORECASE,
)
_POLICY_ID = re.compile(r"\bp0?(\d)\b", re.IGNORECASE)
_NUMBER_M = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m\b|米|公尺)", re.IGNORECASE)
_POLICY_KEYWORDS = (
    "policy", "规则", "阈值", "改用", "threshold", "间距规则", "标准",
)
_ASK_MARKERS = ("?", "？", "多远", "多少", "是否", "有没有", "为什么", "哪", "吗")
_REFINE_MARKERS = (
    "量", "测", "补测", "定位", "找", "没识别", "漏", "missing", "measure",
    "locate", "还有", "没看到",
)

# operator vocabulary -> the English labels policies reason over; anything
# not in here is kept verbatim (the label is still recorded and shown)
_ZH_LABELS = {
    "围栏": "safety fence", "护栏": "safety fence", "栅栏": "safety fence",
    "铁网": "safety fence", "透明护板": "safety fence",
    "导向挡板": "sloped surface", "斜坡": "sloped surface", "挡板": "sloped surface",
    "小车": "material cart", "料车": "material cart", "容器": "material cart",
    "托盘": "pallet", "箱子": "crate", "货架": "storage rack",
    "梯子": "step ladder", "平台": "portable work platform",
    "机器人": "industrial robot arm", "机械臂": "industrial robot arm",
    "光幕": "light curtain", "急停": "emergency stop button",
    "防撞柱": "bollard", "立柱": "bollard", "警示带": "floor marking",
    "地面标线": "floor marking", "标线": "floor marking",
}
_POLICY_HINTS = {
    "间距": "p01", "clearance": "p01", "走道": "p03", "walkway": "p03",
    "灭火器": "p04", "托盘高": "p05", "梯子": "p06", "货架": "p07",
    "安全帽": "p08", "急停": "p09",
}


def classify_intent(message: str) -> str:
    text = (message or "").strip()
    lowered = text.lower()
    # "p01 改成 0.8m" carries a relabel verb too — a threshold with a
    # policy reference wins, so test this before relabel
    if (
        _POLICY_ID.search(lowered) or any(k in lowered for k in _POLICY_KEYWORDS)
    ) and _NUMBER_M.search(lowered):
        return "policy_adjust"
    if _RELABEL_SPLIT.search(text):
        return "relabel"
    if any(m in text for m in _ASK_MARKERS) and not any(
        m in text for m in _REFINE_MARKERS
    ):
        return "ask"
    return "refine"


def _english_label(text: str) -> str:
    cleaned = re.sub(r"^(这块|这个|那个|把|它|这|那|is|it)\s*", "", text.strip())
    cleaned = cleaned.strip(" ，,。.、是的").strip()
    for zh, en in _ZH_LABELS.items():
        if zh in cleaned:
            return en
    return cleaned.lower()


_NEGATION_VERBS = ("不是", "is not")
_NEW_SPLIT = re.compile(r"(?:，|,|、|\s)*(?:而是|其实是|应该是|是|it is|it's|but)\s*", re.IGNORECASE)


def parse_relabel(message: str) -> tuple[str, str] | None:
    """Two operator shapes:
      "这块不是围栏，是导向挡板"  -> old and new both follow the negation
      "把小车改成托盘" / "X should be Y" -> old before the verb, new after
    """
    match = _RELABEL_SPLIT.search(message)
    if not match:
        return None
    verb = match.group(1).lower()
    after = message[match.end():]
    if verb in _NEGATION_VERBS:
        parts = _NEW_SPLIT.split(after, maxsplit=1)
        if len(parts) < 2:
            return None
        old, new = _english_label(parts[0]), _english_label(parts[1])
    else:
        old, new = _english_label(message[: match.start()]), _english_label(after)
    if not old or not new or old == new:
        return None
    return old, new


def parse_policy_adjust(message: str) -> tuple[str | None, float] | None:
    number = _NUMBER_M.search(message)
    if not number:
        return None
    threshold = float(number.group(1))
    pid = _POLICY_ID.search(message.lower())
    if pid:
        return f"p{int(pid.group(1)):02d}", threshold
    for hint, policy in _POLICY_HINTS.items():
        if hint in message.lower():
            return policy, threshold
    return None, threshold


def _reevaluate(run_dir: Path) -> list[dict]:
    from .contracts import PolicySpec, SceneMap
    from .policy import evaluate_policies

    scene_path = run_dir / "scene.json"
    policies_path = run_dir / "policies.json"
    if not scene_path.exists() or not policies_path.exists():
        return []
    scene = SceneMap.model_validate(json.loads(scene_path.read_text()))
    envelope = json.loads(policies_path.read_text())
    specs = [PolicySpec.model_validate(s) for s in envelope.get("specs", [])]
    frames = len({f for e in scene.entities for f in e.evidence_frame_ids}) or 1
    results = evaluate_policies(specs, scene, capture_frame_count=frames)
    envelope["results"] = [r.model_dump(mode="json") for r in results]
    policies_path.write_text(json.dumps(envelope, indent=2) + "\n")
    return envelope["results"]


def _verdict_lines(results: list[dict]) -> str:
    return "\n".join(
        f"- {r['policy_id']}: {r['status']}" for r in results
    ) or "- （本 run 无 policy）"


def apply_relabel(run_dir: Path, old: str, new: str) -> tuple[int, list[dict]]:
    """Rename every entity whose label matches `old` in scene.json and the
    inventory, then re-evaluate policies. Returns (changed, results)."""
    changed = 0
    scene_path = run_dir / "scene.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_text())
        for entity in scene.get("entities", []):
            label = str(entity.get("label", ""))
            if old in label or label in old:
                entity["label"] = new
                changed += 1
        scene_path.write_text(json.dumps(scene, indent=2) + "\n")
    inventory_path = run_dir / "inventory" / "inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        for obj in inventory.get("objects", []):
            label = str(obj.get("label", ""))
            if old in label or label in old:
                obj["label"] = new
                obj["relabelled_from"] = label
                changed += 1
        inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    results = _reevaluate(run_dir) if changed else []
    return changed, results


def apply_policy_adjust(
    run_dir: Path, policy: str | None, threshold: float
) -> tuple[list[str], list[dict]]:
    """Per-run threshold override. The original envelope is kept once as
    policies.original.json so the change is auditable and reversible."""
    policies_path = run_dir / "policies.json"
    if not policies_path.exists():
        return [], []
    original = run_dir / "policies.original.json"
    if not original.exists():
        original.write_text(policies_path.read_text())
    envelope = json.loads(policies_path.read_text())
    touched = []
    for spec in envelope.get("specs", []):
        pid = str(spec.get("policy_id", ""))
        if policy is None or pid.startswith(policy):
            spec["threshold"] = threshold
            spec["adjusted_for_run"] = True
            touched.append(pid)
    if not touched:
        return [], []
    policies_path.write_text(json.dumps(envelope, indent=2) + "\n")
    return touched, _reevaluate(run_dir)


def agent_turn(
    run_id: str,
    message: str,
    *,
    runs_root: str | Path = "runs",
    apply: bool = True,
    answer_fn: Callable[[str], str] | None = None,
    refine_fn: Callable[[str, str, bool], dict] | None = None,
) -> dict:
    """Route one operator sentence, do it, log it. Returns
    {"intent", "reply", "changed", "overlay_path"}."""
    run_dir = Path(runs_root) / run_id
    intent = classify_intent(message)
    reply, changed, overlay = "", False, None
    if intent == "ask":
        reply = answer_fn(message) if answer_fn else "（问答后端未接入）"
    elif intent == "relabel":
        parsed = parse_relabel(message)
        if parsed is None:
            reply = "没听懂要改哪个：请说「X 不是 A，是 B」或「把 A 改成 B」。"
        else:
            old, new = parsed
            if apply:
                count, results = apply_relabel(run_dir, old, new)
                changed = count > 0
                reply = (
                    f"已把 {count} 个「{old}」实体改为「{new}」，policy 重评：\n"
                    + _verdict_lines(results)
                    if count
                    else f"没找到标签含「{old}」的实体，未改动。"
                )
            else:
                reply = f"（预览）会把「{old}」改为「{new}」；勾选回灌判定后生效。"
    elif intent == "policy_adjust":
        parsed = parse_policy_adjust(message)
        if parsed is None:
            reply = "没听到阈值数字：例如「p01 改成 0.8m」。"
        else:
            policy, threshold = parsed
            if apply:
                touched, results = apply_policy_adjust(run_dir, policy, threshold)
                changed = bool(touched)
                reply = (
                    f"已把 {', '.join(touched)} 的阈值改为 {threshold} m（仅本 run，"
                    "原值留在 policies.original.json），重评：\n"
                    + _verdict_lines(results)
                    if touched
                    else f"本 run 没有匹配 {policy or '任何'} 的 policy。"
                )
            else:
                reply = f"（预览）会把 {policy or '全部'} 阈值改为 {threshold} m。"
    else:
        if refine_fn is None:
            reply = "（补测后端未接入）"
        else:
            result = refine_fn(run_id, message, apply) or {}
            overlay = result.get("overlay_path")
            if result.get("label"):
                reply = (
                    f"定位到「{result.get('label')}」：高 {result.get('height_m')} m，"
                    f"占地 {result.get('extent_m')}，距相机 {result.get('camera_dist_m')} m，"
                    f"SAM {result.get('sam_score')}。"
                    + ("已回灌判定。" if apply else "未回灌（预览）。")
                )
                changed = bool(apply)
            else:
                reply = result.get("message") or "没定位到，换个说法或框选补测。"
    entry = {
        "type": "agent_turn",
        "ts": datetime.now(timezone.utc).isoformat(),
        "intent": intent,
        "user": message,
        "assistant": reply,
        "changed": changed,
    }
    chat = run_dir / "chat.jsonl"
    chat.parent.mkdir(parents=True, exist_ok=True)
    with chat.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"intent": intent, "reply": reply, "changed": changed, "overlay_path": overlay}


def chat_history(run_dir: Path) -> list[dict]:
    """chat.jsonl -> chatbot messages (both grounded QA turns and hub turns)."""
    path = Path(run_dir) / "chat.jsonl"
    if not path.exists():
        return []
    messages = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except Exception:
            continue
        kind = entry.get("type")
        if kind == "agent_turn":
            messages.append({"role": "user", "content": f"[{entry.get('intent')}] {entry.get('user', '')}"})
            messages.append({"role": "assistant", "content": entry.get("assistant", "")})
        elif kind == "chat_turn":
            q = entry.get("question") or entry.get("user") or ""
            a = entry.get("answer") or entry.get("assistant") or ""
            if q or a:
                messages.append({"role": "user", "content": f"[ask] {q}"})
                messages.append({"role": "assistant", "content": str(a)})
    return messages


__all__ = [
    "INTENTS", "agent_turn", "apply_policy_adjust", "apply_relabel",
    "chat_history", "classify_intent", "parse_policy_adjust", "parse_relabel",
]
