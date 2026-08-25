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
