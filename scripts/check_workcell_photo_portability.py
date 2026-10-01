"""No-GPU check of portable Modal invocation and frozen viewer dependencies."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import workcell_photo_oneshot as report


def check():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        built = root / "web/dist-photo"
        built.mkdir(parents=True)
        (built / "photo.html").write_text("<main>report</main>")
        assets = root / "three"
        hashes = {}
        for name in report.VIEWER_ASSET_SHA256:
            path = assets / name
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = name.encode()
            path.write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        with patch.object(report, "REPO", root), patch.dict(report.VIEWER_ASSET_SHA256, hashes):
            report._freeze_report_ui(root / "output", assets)
            frozen = root / "output/report-ui"
            assert (frozen / "photo.html").read_text() == "<main>report</main>"
            for name in hashes:
                assert (frozen / "viewer-assets" / name).read_bytes() == name.encode()
            (assets / "three.core.js").write_text("changed")
            assert (frozen / "viewer-assets/three.core.js").read_text() == "three.core.js"
            try:
                report._freeze_report_ui(root / "invalid", assets)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched Three.js asset accepted")
            assert not (root / "invalid").exists()
            (assets / "LICENSE").unlink()
            try:
                report._freeze_report_ui(root / "missing", assets)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Missing Three.js license accepted")
            (frozen / "viewer-assets/LICENSE").unlink()
            try:
                report._build_page(root / "output", {})
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("Incomplete frozen viewer packaged")
            assert not (root / "output/page").exists()
        result = subprocess.CompletedProcess([], 0, "https://modal.com/apps/ap-test123")
        with patch.object(report.subprocess, "run", return_value=result) as run:
            record = report._job("check", ["modal_apps/workcell_photo_all.py"], root)
            assert run.call_args.args[0] == [sys.executable, "-m", "modal", "run", "modal_apps/workcell_photo_all.py"]
            assert run.call_args.kwargs["cwd"] == report.REPO
            assert record["runIds"] == ["ap-test123"]
    print("photo portability: current interpreter, complete pinned assets and frozen copies passed")


if __name__ == "__main__":
    check()
