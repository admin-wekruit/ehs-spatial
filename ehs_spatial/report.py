"""One printable HTML page per run, built only from the run's own artifact
files. Stdlib only on purpose: an inspector must be able to print a report
from a cached run directory with no providers, no gradio, and no network.
Every artifact is optional — a partial run still yields a partial report."""

import argparse
import base64
import json
import mimetypes
from html import escape
from pathlib import Path


# Kept in sync with ehs_spatial.app.DEMO_RULE_COPY by test_report; duplicated
# here so the report never imports the gradio-heavy app module.
DEMO_RULE_COPY = "0.6 m demo rule — not an official EHS standard"

_CSS = """
body { font-family: Georgia, 'Times New Roman', serif; color: #1a2233;
       max-width: 60rem; margin: 2rem auto; padding: 0 1rem; background: #fff; }
h1 { font-size: 1.6rem; border-bottom: 3px solid #1a2233; padding-bottom: .4rem; }
h2 { font-size: 1.1rem; text-transform: uppercase; letter-spacing: .08em;
     margin-top: 2rem; }
table { border-collapse: collapse; width: 100%; font-size: .9rem; }
th, td { border: 1px solid #9aa2b1; padding: .4rem .6rem; text-align: left;
         vertical-align: top; }
th { background: #eef0f4; }
.verdict { border: 2px solid #1a2233; padding: 1rem; font-size: 1.1rem; }
.verdict .status { font-size: 1.5rem; font-weight: bold; letter-spacing: .05em; }
.meta dt { font-weight: bold; }
.meta dd { margin: 0 0 .4rem 0; }
.disposition { border: 2px dashed #6b7280; padding: 1rem; margin-top: 1rem; }
.missing { color: #6b7280; font-style: italic; }
img.evidence { max-width: 100%; border: 1px solid #9aa2b1; margin: .5rem 0; }
@media print {
  body { margin: 0; max-width: none; }
  h2 { break-after: avoid; }
  img.evidence, .verdict, .disposition, table { break-inside: avoid; }
}
"""


def _load_json(path: Path) -> object | None:
    """Fail-soft artifact read: a report renders around whatever is absent."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _load_dict(path: Path) -> dict:
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else {}


def _image_tag(path: Path, caption: str) -> str:
    """Embed one PNG as a base64 data URI so the page is self-contained;
    skip silently when the file is absent or unreadable."""
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError:
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return (
        f'<figure><img class="evidence" alt="{escape(caption)}" '
        f'src="data:{mime};base64,{encoded}">'
        f"<figcaption>{escape(caption)}</figcaption></figure>"
    )


def _policy_rows(policies: object) -> list[dict]:
    """Normalize policies.json: the current envelope is
    {"specs": [...], "results": [...]}; older runs stored a bare result
    list. Returns one row dict per result, spec fields merged in."""
    if isinstance(policies, list):
        specs: dict[str, dict] = {}
        results = policies
    elif isinstance(policies, dict):
        specs = {
            spec.get("policy_id"): spec
            for spec in policies.get("specs", [])
            if isinstance(spec, dict)
        }
        results = policies.get("results", [])
    else:
        return []
    rows = []
    for result in results:
        if not isinstance(result, dict):
            continue
        spec = specs.get(result.get("policy_id"), {})
        rule = (
            f"{spec['predicate']} {spec['threshold']} {spec['unit']}"
            if spec.get("predicate")
            else ""
        )
        violations = result.get("violations") or []
        worst = violations[0] if violations else {}
        rows.append(
            {
                "policy_id": result.get("policy_id", "?"),
                "status": result.get("status", "?"),
                "rule": rule,
                "source_text": spec.get("source_text", ""),
                "worst": (
                    f"{worst['measured']} {worst['unit']} "
                    f"(limit {worst['threshold']} {worst['unit']})"
                    if worst
                    else ""
                ),
                "warnings": "; ".join(result.get("warnings") or []),
            }
        )
    return rows


def _distance_copy(assessment: dict) -> str:
    # Never print a bare decimal: the band is part of the measurement.
    distance = assessment.get("approximate_distance_m")
    if distance is None:
        return "unavailable from the evidence"
    budget = assessment.get("distance_error_budget_m")
    if budget is None:
        return f"{distance:.2f} m"
    return f"{distance:.2f} m ± {budget:.2f} m"


def build_report_html(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    manifest = _load_dict(run_dir / "manifest.json")
    assessment = _load_dict(run_dir / "assessment.json")
    scene = _load_dict(run_dir / "scene.json")
    review = _load_dict(run_dir / "review.json")
    policy_rows = _policy_rows(_load_json(run_dir / "policies.json"))

    run_id = manifest.get("run_id") or run_dir.name
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>EHS run report — {escape(run_id)}</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<h1>EHS spatial run report — {escape(run_id)}</h1>",
    ]

    parts.append('<h2>Run</h2><dl class="meta">')
    for label, value in (
        ("Created", manifest.get("created_at")),
        ("Operator", manifest.get("operator")),
        ("Capture tier", manifest.get("capture_tier")),
    ):
        shown = escape(str(value)) if value else "<span class='missing'>unknown</span>"
        parts.append(f"<dt>{label}</dt><dd>{shown}</dd>")
    providers = manifest.get("providers")
    if isinstance(providers, dict):
        pins = ", ".join(
            f"{escape(str(name))}={escape(str(pin))}"
            for name, pin in sorted(providers.items())
        )
        parts.append(f"<dt>Provider pins</dt><dd>{pins}</dd>")
    parts.append("</dl>")

    parts.append("<h2>Verdict</h2>")
    if assessment:
        parts.append(
            '<div class="verdict">'
            f'<div class="status">{escape(str(assessment.get("status", "?")))}</div>'
            "<p>Approximate boundary clearance: "
            f"<strong>{escape(_distance_copy(assessment))}</strong>.</p>"
            f"<p><strong>{escape(DEMO_RULE_COPY)}.</strong></p></div>"
        )
    else:
        parts.append('<p class="missing">No readable assessment.json.</p>')

    parts.append("<h2>Policies</h2>")
    if policy_rows:
        parts.append(
            "<table><tr><th>Policy</th><th>Status</th><th>Rule</th>"
            "<th>Worst violation</th><th>Source</th><th>Warnings</th></tr>"
        )
        for row in policy_rows:
            parts.append(
                "<tr>"
                f"<td>{escape(str(row['policy_id']))}</td>"
                f"<td>{escape(str(row['status']))}</td>"
                f"<td>{escape(row['rule'])}</td>"
                f"<td>{escape(row['worst'])}</td>"
                f"<td>{escape(row['source_text'])}</td>"
                f"<td>{escape(row['warnings'])}</td>"
                "</tr>"
            )
        parts.append("</table>")
    else:
        parts.append('<p class="missing">No compiled policies for this run.</p>')

    warnings = scene.get("warnings")
    parts.append("<h2>Warnings</h2>")
    if isinstance(warnings, list) and warnings:
        parts.append("<ul>")
        parts.extend(f"<li>{escape(str(warning))}</li>" for warning in warnings)
        parts.append("</ul>")
    else:
        parts.append('<p class="missing">None recorded.</p>')

    if review:
        parts.append('<h2>Review disposition</h2><div class="disposition">')
        decision = review.get("decision", "?")
        line = f"<p><strong>{escape(str(review.get('reviewer', '?')))}</strong> "
        if decision == "overridden":
            line += (
                "overrode the machine verdict to "
                f"<strong>{escape(str(review.get('overridden_status', '?')))}</strong>"
            )
        else:
            line += f"{escape(str(decision))} the machine verdict"
        line += f" at {escape(str(review.get('created_at', '?')))}.</p>"
        parts.append(line)
        if review.get("reason"):
            parts.append(f"<p>Reason: {escape(str(review['reason']))}</p>")
        parts.append("</div>")

    parts.append("<h2>Evidence images</h2>")
    overlays = sorted((run_dir / "evidence").glob("*_overlay.png"))
    images = "".join(
        [
            _image_tag(run_dir / "topdown.png", "Top-down evidence"),
            _image_tag(run_dir / "plan_view.png", "Plan view"),
            _image_tag(overlays[0], f"Mask overlay — {overlays[0].stem}")
            if overlays
            else "",
        ]
    )
    parts.append(images or '<p class="missing">No evidence images on disk.</p>')

    parts.append("</body></html>")
    return "".join(parts)


def write_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    destination = run_dir / "report.html"
    destination.write_text(build_report_html(run_dir), encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="one runs/<id> directory")
    args = parser.parse_args(argv)
    if not args.run_dir.is_dir():
        parser.error(f"not a run directory: {args.run_dir}")
    print(write_report(args.run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ---- per-run workbench report (self-contained, embedded images) ----
import io  # noqa: E402

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

    assessment_path = run / "assessment.json"
    if assessment_path.exists():
        try:
            assessment = json.loads(assessment_path.read_text())
        except ValueError:
            assessment = {}
        status = str(assessment.get("status", "")).split(".")[-1]
        colour, zh = STATUS_META.get(status, ("#8f8f8f", status or "?"))
        distance = assessment.get("approximate_distance_m")
        budget = assessment.get("distance_error_budget_m")
        line = ""
        if distance is not None:
            line = f"边界间距约 {distance:.2f} m"
            if budget is not None:
                line += f" ± {budget:.2f} m"
        parts.append(
            _section(
                "总判定",
                f'<p><span class="pill" style="background:{colour};'
                f'font-size:15px">{zh}</span> {line}</p>',
            )
        )

    evidence_dir = run / "evidence"
    if evidence_dir.exists():
        figs = "".join(
            f'<figure><img src="{_jpeg_uri(p, 700, 70)}"></figure>'
            for p in sorted(evidence_dir.glob("*_overlay.png"))[:4]
        )
        if figs:
            parts.append(
                _section("检出证据叠加", f'<div class="row">{figs}</div>')
            )

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



