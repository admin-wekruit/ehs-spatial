"""CLI for the taxonomy detection layer — see ehs_spatial.detect.

  uv run --env-file .env python scripts/detect_devices.py --run real-clean-01
  (add --fresh to repeat each frame's sweep; SAM caches remain reusable)
"""

import argparse
from pathlib import Path

from ehs_spatial.detect import detect_devices


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)
    envelope = detect_devices(args.run, fresh=args.fresh)
    found = [d for d in envelope["detections"] if "rle" in d]
    print(f"{len(found)} devices masked, {len(envelope['missing'])} missing, "
          f"{len(envelope['rejected'])} rejected")
    for det in found:
        print(f"  {det['frame_id']} #{det['number']:2d} [{det['category']}] {det['zh']:12s} "
              f"SAM {det['sam_score']}  box {det['box']}")
    for item in envelope["missing"]:
        print(f"  MISSING {item['frame_id']} {item['item_id']} {item['zh']}")
    for item in envelope["rejected"]:
        print(f"  REJECTED {item['frame_id']} {item['item_id']}: {item['reason']}")
    for frame in envelope['frames']:
        print("overlay:", Path("runs") / args.run / frame['overlay_path'])
    print("this execution:", envelope['last_execution'])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
