"""One command from an MP4 and a time window to a platform import (M2 report runner).

  python scripts/run_video_report.py --video MP4 --start S --end E --site NAME
         [--profile research|commercial|delivered] [--review DIR] [--dry-run] [--publish] [--verify] [--republish]

Stages are content-addressed (scripts/report_runner/store.py): a stage whose key is already in
$ART/runs/report-runner/keys.jsonl is served from its directory; the rest run, 4 at a time, paid stages one at a time.
PANOPTES_PAID_BUDGET_USD caps the run's paid calls; unset, the run stops before its first paid cache miss. A re-run of the
same command continues its budget ($ART/runs/report-runner/ledgers/): each paid stage books list price x wall seconds.
PANOPTES_DATABASE_URL and PANOPTES_BLOB_ROOT are needed by the import only.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from report_runner import store  # noqa: E402


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    p.add_argument("--video", type=Path, required=True, help="the source MP4")
    p.add_argument("--start", type=float, required=True, help="window start, seconds into the video")
    p.add_argument("--end", type=float, required=True, help="window end, seconds into the video")
    p.add_argument("--site", required=True, help="site name; names the run directories")
    p.add_argument("--profile", choices=("research", "commercial", "delivered"), default="research",
                   help="delivered is cache-only and proves reproduction of the delivered reports")
    p.add_argument("--review", type=Path, help="eye review directory holding <site>.json (the only human input)")
    p.add_argument("--dry-run", action="store_true", help="print the plan (key, hit/miss/unresolved/refused, dir, USD, s); run and write nothing")
    p.add_argument("--publish", action="store_true", help="after a successful import: export, check, prepare, deploy")
    p.add_argument("--republish", action="store_true", help="import as the next version of the site's last import (imports.jsonl); "
                                                             "implied by --profile delivered, otherwise every run imports a new report")
    p.add_argument("--verify", action="store_true", help="check every hit by full sha256 instead of size and mtime")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.end <= args.start:
        parser().error("--end must be after --start")
    return store.main(args)


if __name__ == "__main__":
    sys.exit(main())
