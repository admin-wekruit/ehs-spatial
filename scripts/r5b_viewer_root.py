"""r5b integrate: a viewer root (the layout of layers.serve: reports/<id> and blobs/sha256/<hex>) of symlinks into bench mirrors, so
the local viewer serves several benches' reports at once without copying a byte.

    python scripts/r5b_viewer_root.py OUT MIRROR:REPORT [MIRROR:REPORT ...]
"""
import sys
from pathlib import Path


def build(out, pairs):
    out = Path(out)
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "blobs" / "sha256").mkdir(parents=True, exist_ok=True)
    for mirror, report in pairs:
        mirror = Path(mirror)
        link = out / "reports" / report
        if not link.exists():
            link.symlink_to(mirror / "reports" / report)
        for b in (mirror / "blobs" / "sha256").iterdir():
            dst = out / "blobs" / "sha256" / b.name
            if not dst.exists() and not b.name.startswith("."):
                dst.symlink_to(b)
    return out


if __name__ == "__main__":
    build(sys.argv[1], [a.rsplit(":", 1) for a in sys.argv[2:]])
    print("viewer root", sys.argv[1])
