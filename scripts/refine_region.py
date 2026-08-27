"""CLI wrapper over ehs_spatial.refine — see that module for the design.

Usage (pixel coordinates in the original image):
  uv run --env-file .env python scripts/refine_region.py \
      --run gen-01 --label "safety sensor" --box 289,228,434,705 [--apply]
"""

import argparse
import json
import sys

from ehs_spatial.refine import RefineError, refine_region


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--box", required=True, help="x1,y1,x2,y2 pixels")
    parser.add_argument("--camera-height", type=float, default=1.5)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = refine_region(
            args.run,
            args.label,
            tuple(int(v) for v in args.box.split(",")),
            camera_height=args.camera_height,
            apply=args.apply,
        )
    except RefineError as exc:
        print(f"refine failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.apply and result.get("policies"):
        print("\npolicy statuses with the correction applied:")
        for row in result["policies"]:
            print(f"  {row['policy_id']}: {row['status']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
