"""CLI for the taxonomy detection layer — see ehs_spatial.detect.

  uv run --env-file .env python scripts/detect_devices.py --run real-clean-01
  (add --fresh to discard the cached detections and re-detect)
"""

import argparse
import json
import shutil
from pathlib import Path

from ehs_spatial.detect import detect_devices


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    if args.fresh:
        cache = Path("runs") / args.run / "detection" / "detections.json"
        if cache.exists():
            # archive, don't discard: the new round merges every verified
            # detection from this file so re-detects are monotonic
            cache.rename(cache.with_name("detections.prev.json"))
    envelope = detect_devices(args.run)
    found = [d for d in envelope["detections"] if "rle" in d]
    print(f"{len(found)} devices masked, {len(envelope['missing'])} missing, "
          f"{len(envelope['rejected'])} rejected")
    for det in found:
        print(f"  #{det['number']:2d} [{det['category']}] {det['zh']:12s} "
              f"SAM {det['sam_score']}  box {det['box']}")
    for item in envelope["missing"]:
        print(f"  MISSING {item['item_id']} {item['zh']}")
    for item in envelope["rejected"]:
        print(f"  REJECTED {item['item_id']}: {item['reason']}")
    print("overlay:", Path("runs") / args.run / "detection" / "overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
