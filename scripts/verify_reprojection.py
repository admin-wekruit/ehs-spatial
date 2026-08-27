"""CLI for the reprojection loop — see ehs_spatial.reproject.

  uv run python scripts/verify_reprojection.py --run real-clean-01
"""

import argparse
import json

from ehs_spatial.reproject import verify_reprojection


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args(argv)
    out = verify_reprojection(args.run)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
