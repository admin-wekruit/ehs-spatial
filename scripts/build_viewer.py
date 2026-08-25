"""Thin CLI shell over ehs_spatial.viewer.build_viewer_html.

Rebuilds `runs/<run>/viewer.html` for any cached run. The pipeline emits the
same artifact automatically; this exists for iterating on old runs.

Usage:
  uv run python scripts/build_viewer.py --run demo-real-factory
"""

import argparse
import sys
from pathlib import Path

from ehs_spatial.viewer import build_viewer_html


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument(
        "--exclude",
        default="floor,ceiling,wall,ground,roof",
        help="comma-separated labels kept as scene, not as selectable objects",
    )
    args = parser.parse_args(argv)
    excluded = {token.strip() for token in args.exclude.split(",") if token.strip()}

    try:
        summary = build_viewer_html(
            Path("runs") / args.run,
            camera_height_m=args.camera_height,
            exclude=excluded,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    out = summary["path"]
    print(f"{summary['points']} points | {len(summary['objects'])} selectable objects")
    for obj in summary["objects"]:
        print(f"  [{obj['id']:>2}] {obj['label']:<20} H={obj['height_m']:>6.2f}m "
              f"{obj['size']:>16}  d={obj['camera_dist_m']:>6.2f}m  n={obj['points']}")
    print("wrote", out, f"({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
