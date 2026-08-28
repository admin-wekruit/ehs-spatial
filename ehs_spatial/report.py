"""Per-run report: one self-contained HTML page a run can carry with it.

The workbench generates it after analysis (or on demand) so submitting a
photo and reading the findings live in one place: verdicts with plain-
language reasons, the numbered device-detection overlay, entity
measurements, refinement gallery, and the reprojection check — everything
embedded, so the file can be sent to anyone and opened offline.
"""

import base64
import io
import json
from pathlib import Path

CATEGORY_META = {
    "A": ("感知防护 SENSING/AOPD", "#39c5cf"),
    "B": ("控制防护 CONTROL", "#f25c8a"),
    "C": ("防护罩/围护 GUARDS", "#4ad07a"),
    "D": ("阻挡与引导 IMPEDING", "#e8b93c"),
    "E": ("信息标识 INFO", "#c9a0ff"),
}

STATUS_META = {
    "FAIL": ("#e5484d", "FAIL 违规"),
    "NEEDS_REVIEW": ("#f5a524", "NEEDS REVIEW 待复核"),
    "INSUFFICIENT_EVIDENCE": ("#8f8f8f", "证据不足"),
    "PASS": ("#30a46c", "PASS"),
}


def _jpeg_uri(path: Path, max_width: int = 1100, quality: int = 74) -> str:
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.width > max_width:
            image = image.resize(
                (max_width, round(image.height * max_width / image.width))
            )
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(
        buffer.getvalue()
    ).decode()


def _section(title: str, body: str) -> str:
    return f"<h2>{title}</h2>{body}" if body else ""


def build_run_report(run_id: str, *, runs_root: str | Path = "runs") -> Path:
    """Render runs/<run>/report.html and return its path."""
    run = Path(runs_root) / run_id
    parts: list[str] = []

    inputs = sorted((run / "input").glob("image_*"))
    if inputs:
        figs = "".join(
            f'<figure><img src="{_jpeg_uri(p, 700, 70)}"></figure>'
            for p in inputs[:4]
        )
        parts.append(_section("输入照片", f'<div class="row">{figs}</div>'))

    policies_path = run / "policies.json"
    if policies_path.exists():
        rows = []
        for result in json.loads(policies_path.read_text()).get("results", []):
            status = str(result.get("status", "")).split(".")[-1]
            colour, zh = STATUS_META.get(status, ("#8f8f8f", status))
            why = "; ".join(
                str(w) for w in (result.get("warnings") or [])
            )[:300]
            rows.append(
                f'<tr><td class="mono">{result.get("policy_id", "")}</td>'
                f'<td><span class="pill" style="background:{colour}">{zh}'
                f"</span></td><td>{why}</td></tr>"
            )
        if rows:
            parts.append(
                _section(
                    "判定",
                    '<table><tr><th>policy</th><th>结论</th><th>说明</th>'
                    f'</tr>{"".join(rows)}</table>',
                )
            )

    detections_path = run / "detection" / "detections.json"
    if detections_path.exists():
        envelope = json.loads(detections_path.read_text())
        overlay = run / "detection" / "overlay.png"
        body = ""
        if overlay.exists():
            body += f'<figure><img src="{_jpeg_uri(overlay)}"></figure>'
        groups: dict[str, list[dict]] = {}
        for det in envelope.get("detections", []):
            if "rle" in det:
                groups.setdefault(det["category"], []).append(det)
        for category in "ABCDE":
            if category not in groups:
                continue
            name, colour = CATEGORY_META[category]
            items = " ".join(
                f'<span class="lg"><b style="color:{colour}">#{d["number"]}'
                f"</b> {d['zh']}"
                + (f' <span class="dim">{d["iso"]}</span>' if d.get("iso") else "")
                + "</span>"
                for d in groups[category]
            )
            body += f'<div class="cat"><b>{name}</b><div>{items}</div></div>'
        missing = envelope.get("missing", [])
        body += (
            '<div class="miss">未见/需现场核实：'
            + "、".join(m["zh"] for m in missing)
            + "</div>"
            if missing
            else '<div class="ok">检测清单全部检出</div>'
        )
        parts.append(_section("装置检测清单", body))

    inventory_path = run / "inventory" / "inventory.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text())
        rows = "".join(
            f'<tr><td>{o["label"]}</td><td>{o["height_m"]} m</td>'
            f'<td>{o.get("size_m", "")}</td>'
            f'<td>{o.get("camera_dist_m", "")} m</td>'
            f'<td class="dim">{o.get("footprint_method") or "hull"}</td></tr>'
            for o in inventory.get("objects", [])
            if not o.get("off_plan_reason")
        )
        plan = run / "inventory" / "floor_plan.png"
        body = (
            f'<figure><img src="{_jpeg_uri(plan)}"></figure>' if plan.exists() else ""
        )
        body += (
            "<table><tr><th>物体</th><th>高</th><th>尺寸</th>"
            f"<th>距相机</th><th>方法</th></tr>{rows}</table>"
        )
        parts.append(_section("实体测量 + 平面图", body))

    reprojection_path = run / "inventory" / "reprojection.json"
    if reprojection_path.exists():
        scores = json.loads(reprojection_path.read_text()).get("scores", [])
        chips = " ".join(
            f'<span class="lg">#{s.get("instance")} '
            + (
                f'{round(s["mean_dv_frac"] * 100, 1)}%'
                if s.get("mean_dv_frac") is not None
                else s.get("status", "—")
            )
            + "</span>"
            for s in scores
        )
        figure = run / "inventory" / "reprojection.png"
        body = (
            f'<figure><img src="{_jpeg_uri(figure)}"></figure>'
            if figure.exists()
            else ""
        ) + f"<div>{chips}</div>"
        parts.append(_section("回投验证（红点应压结构接地线 · 偏差=图高占比）", body))

    refinements_path = run / "refinements.json"
    if refinements_path.exists():
        figs = ""
        for item in json.loads(refinements_path.read_text()):
            if "height_m" not in item:
                continue
            slug = (
                item["label"].replace(" ", "_")
                + "_"
                + "_".join(str(v) for v in item["box"])
            )
            overlay = run / "refinements" / f"{slug}.png"
            if not overlay.exists():
                continue
            figs += (
                f'<figure><img src="{_jpeg_uri(overlay, 520, 66)}">'
                f'<figcaption>{item["label"]} · SAM {item["sam_score"]} · '
                f'{item["height_m"]} m</figcaption></figure>'
            )
        if figs:
            parts.append(_section("人工/agent 补测", f'<div class="row">{figs}</div>'))

    html = f"""<meta charset="utf-8"><title>Panoptes · {run_id}</title>
<style>
body{{background:#141a1f;color:#dde3e8;font:15px/1.55 -apple-system,'PingFang SC',sans-serif;max-width:1080px;margin:0 auto;padding:28px}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #2a333b;padding-bottom:6px}}
figure{{margin:8px 0}} img{{max-width:100%;border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px}} td,th{{border:1px solid #2a333b;padding:5px 9px;text-align:left}}
.pill{{color:#fff;border-radius:10px;padding:2px 9px;font-size:12.5px}}
.row{{display:flex;flex-wrap:wrap;gap:10px}} .row figure{{flex:1 1 240px;margin:0}}
.lg{{display:inline-block;margin:2px 10px 2px 0;font-size:13px}} .dim{{color:#7d8790;font-size:12px}}
.cat{{margin:7px 0}} .miss{{color:#e5484d;margin-top:8px}} .ok{{color:#30a46c;margin-top:8px}}
.mono{{font-family:ui-monospace,monospace;font-size:12.5px}}
</style>
<h1>Panoptes · {run_id}</h1>
<p class="dim">自包含单文件报告 · 交互 3D 见 runs/{run_id}/viewer.html · 深度交互版见测试集总报告</p>
{"".join(parts)}"""
    out = run / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


__all__ = ["build_run_report"]
