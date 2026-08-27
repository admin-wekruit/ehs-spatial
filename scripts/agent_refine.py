"""CLI for the conversational correction turn — see ehs_spatial/agent.py.

Usage:
  uv run --env-file .env python scripts/agent_refine.py \
      --run real-clean-03 --say "入口右侧红色的斜坡挡板" [--no-apply]
"""

import argparse
import json
import sys

from ehs_spatial.agent import ProviderError, RefineError, agent_refine


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--say", required=True)
    parser.add_argument("--no-apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = agent_refine(args.run, args.say, apply=not args.no_apply)
    except (RefineError, ProviderError) as exc:
        print(f"agent refine failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
