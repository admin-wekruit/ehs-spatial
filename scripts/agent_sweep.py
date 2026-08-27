"""Standard reviewer sweep as an agent — see ehs_spatial/agent.py.

Usage:
  uv run --env-file .env python scripts/agent_sweep.py --run real-clean-01
"""

import argparse
import json
import sys

from ehs_spatial.agent import agent_sweep


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--no-apply", action="store_true")
    args = parser.parse_args(argv)
    outcomes = agent_sweep(args.run, apply=not args.no_apply)
    print(json.dumps(outcomes, indent=2, ensure_ascii=False))
    measured = sum(1 for o in outcomes if o["status"] == "measured")
    print(f"\n{measured}/{len(outcomes)} items measured", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
