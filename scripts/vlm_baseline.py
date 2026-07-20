"""Harness-vs-VLM spatial understanding benchmark.

Same inputs both sides: the 4 photos + stated camera height. The harness runs
its full geometry pipeline; the VLM (Gemini 3.5 Flash, no tools, no facts) is
asked the same metric questions directly. Ground truth: laser scans (real
packs) / analytic mesh (synthetic).

Usage: uv run --env-file .env python scripts/vlm_baseline.py --live
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

SCENES = [
    {
        "name": "redwood_boardroom (VGA real, laser GT)",
        "images_dir": "outputs/redwood_v2/input",
        "annotations": "outputs/redwood_v2/annotations.json",
        "harness_report": "outputs/redwood_v2/v2_report.json",
        "harness_mode": "camera_height",
    },
    {
        "name": "eth3d_office (DSLR real, laser GT)",
        "images_dir": "outputs/eth3d_v2/input",
        "annotations": "outputs/eth3d_v2/annotations.json",
        "harness_report": "outputs/eth3d_v2/v2_report.json",
        "harness_mode": "camera_height",
    },
]

SYNTHETIC = [
    {"case": "ladder_050", "pair": "step ladder-safety fence", "gt": 0.5},
    {"case": "ladder_070", "pair": "step ladder-safety fence", "gt": 0.7},
]


def _image_parts(paths: list[Path]):
    from google.genai import types

    parts = []
    for path in paths:
        suffix = path.suffix.lower().lstrip(".")
        mime = "image/jpeg" if suffix in ("jpg", "jpeg") else "image/png"
        parts.append(types.Part.from_bytes(data=path.read_bytes(), mime_type=mime))
    return parts


def _ask_vlm(client, images: list[Path], camera_height: float, pairs: list[str]) -> dict:
    prompt = (
        "These photos show one scene from four viewpoints, taken at a camera "
        f"height of {camera_height:.2f} m. Estimate the horizontal ground "
        "distance in metres between the nearest base points of each object "
        "pair below. Respond with ONLY a JSON object mapping each pair id to "
        "a number in metres, no other text.\n"
        + "\n".join(f"- {p}" for p in pairs)
    )
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=[*_image_parts(images), prompt],
    )
    text = response.text or ""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    out = {}
    for key, value in raw.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            m = re.search(r"[\d.]+", str(value))
            if m:
                out[key] = float(m.group(0))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    if not args.live:
        print("pass --live (a few Gemini Flash calls, cents)", file=sys.stderr)
        return 2

    from google import genai

    client = genai.Client()
    rows = []

    for scene in SCENES:
        ann = json.loads(Path(scene["annotations"]).read_text())
        report = json.loads(Path(scene["harness_report"]).read_text())
        harness = {
            r["pair"]: r["predicted_m"]
            for r in report["results"]
            if r["mode"] == scene["harness_mode"]
        }
        images = sorted(Path(scene["images_dir"]).iterdir())
        pair_ids = [p["pair_id"] for p in ann["pairs"]]
        vlm = _ask_vlm(client, images, float(ann["camera_height_m"]), pair_ids)
        for pair in ann["pairs"]:
            pid, gt = pair["pair_id"], pair["gt_distance_m"]
            h = harness.get(pid)
            v = vlm.get(pid)
            rows.append(
                {
                    "scene": scene["name"],
                    "pair": pid,
                    "gt_m": gt,
                    "harness_m": h,
                    "vlm_m": v,
                    "harness_err_cm": None if h is None else round(abs(h - gt) * 100, 1),
                    "vlm_err_cm": None if v is None else round(abs(v - gt) * 100, 1),
                }
            )

    manifest = json.loads(Path("outputs/ehs_v1/manifest.json").read_text())
    offline = json.loads(Path("outputs/ehs_v1/offline_report.json").read_text())
    for spec in SYNTHETIC:
        case = spec["case"]
        images = sorted(Path(f"outputs/ehs_v1/{case}").glob("frame_0*_rgb.png"))
        vlm = _ask_vlm(
            client, images, float(manifest["camera_height_m"]), [spec["pair"]]
        )
        h = offline["cases"][case]["actual_distance_m"]
        v = vlm.get(spec["pair"])
        rows.append(
            {
                "scene": f"synthetic {case} (analytic GT; harness = deterministic core on oracle geometry)",
                "pair": spec["pair"],
                "gt_m": spec["gt"],
                "harness_m": h,
                "vlm_m": v,
                "harness_err_cm": round(abs(h - spec["gt"]) * 100, 1),
                "vlm_err_cm": None if v is None else round(abs(v - spec["gt"]) * 100, 1),
            }
        )

    def _mae(key):
        errs = [r[key] for r in rows if r[key] is not None]
        return round(sum(errs) / len(errs), 1) if errs else None

    summary = {
        "rows": rows,
        "harness_mae_cm": _mae("harness_err_cm"),
        "vlm_mae_cm": _mae("vlm_err_cm"),
        "vlm_model": "gemini-3.5-flash (images + camera height only, no tools)",
    }
    out = Path("outputs/vlm_baseline.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for r in rows:
        print(r)
    print("MAE cm — harness:", summary["harness_mae_cm"], "| vlm:", summary["vlm_mae_cm"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
