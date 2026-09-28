"""Publish one imported report: export -> check -> prepare -> deploy (docs/platform/OPERATIONS.md, video site variant).

Runs only under `run_video_report.py --publish`. `publish(id, dry_run=True)` prints the four commands and runs none.

  python -m report_runner.publish PUBLICATION_ID            # print only
  python -m report_runner.publish PUBLICATION_ID --publish  # export, check, prepare, then `modal deploy`
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8792"


def commands(publication_id, catalog=None, http=None, python=sys.executable):
    """The four argvs, each as (env overrides, argv); paths absolute so `modal deploy` packs the right catalog."""
    catalog = Path(catalog or os.environ.get("PANOPTES_PUBLICATION_CATALOG") or REPO / ".platform/video-publication-catalog").resolve()
    http = Path(http or os.environ.get("PANOPTES_PUBLICATION_HTTP") or REPO / ".platform/video-publication-http").resolve()
    modal = str(Path(python).with_name("modal"))
    return [({}, [python, str(REPO / "scripts/export_platform_publication.py"), "--api", API, "--publication", publication_id, "--output", str(catalog / publication_id)]),
            ({}, [python, str(REPO / "tests/check_publication_site.py"), "--catalog", str(catalog), "--source-api", API]),
            ({}, [python, str(REPO / "scripts/prepare_publication_site.py"), "--catalog", str(catalog), "--output", str(http)]),
            ({"PANOPTES_PUBLICATION_CATALOG": str(catalog), "PANOPTES_PUBLICATION_HTTP": str(http)}, [modal, "deploy", str(REPO / "modal_apps/video_publication_site.py")])]


def publish(publication_id: str, *, dry_run: bool, catalog=None, http=None, run=subprocess.run) -> list[list[str]]:
    """Returns the argvs (env overrides written as leading NAME=value words). dry_run prints them and runs nothing."""
    steps = commands(publication_id, catalog, http)
    shown = [[f"{k}={v}" for k, v in env.items()] + argv for env, argv in steps]
    for words in shown:
        print(" ".join(words))
    if not dry_run:
        catalog_dir = Path(steps[1][1][3])
        if not catalog_dir.is_dir() or not any(catalog_dir.iterdir()):  # the deploy serves only this catalog: a fresh one would drop every other report
            raise SystemExit(f"no live catalog at {catalog_dir}: set PANOPTES_PUBLICATION_CATALOG (and _HTTP) to the catalog the site is deployed from")
        for env, argv in steps:  # stop at the first failure: a failed check must never reach the deploy
            run(argv, check=True, cwd=REPO, env={**os.environ, **env})
    return shown


def self_check():
    calls = []
    argvs = publish("pub-1", dry_run=True, catalog="/c", http="/h", run=lambda *a, **k: calls.append(a))
    assert calls == [] and len(argvs) == 4, "a dry run runs nothing"
    assert argvs[0][-2:] == ["--output", "/c/pub-1"] and "--api" in argvs[0] and API in argvs[0]
    assert argvs[1][-4:] == ["--catalog", "/c", "--source-api", API] and argvs[2][-4:] == ["--catalog", "/c", "--output", "/h"]
    assert argvs[3][:2] == ["PANOPTES_PUBLICATION_CATALOG=/c", "PANOPTES_PUBLICATION_HTTP=/h"] and argvs[3][-2:-1] == ["deploy"]

    try:
        publish("pub-1", dry_run=False, catalog="/nonexistent-catalog", http="/h", run=lambda *a, **k: calls.append(a))
        raise AssertionError("published into an empty catalog")
    except SystemExit:
        assert calls == [], "an empty catalog is refused before anything runs"
    import tempfile
    live = Path(tempfile.mkdtemp())
    (live / "older-publication").mkdir()

    def failing(argv, **_):
        calls.append(argv)
        if "check_publication_site.py" in argv[1]:
            raise subprocess.CalledProcessError(1, argv)
    try:
        publish("pub-1", dry_run=False, catalog=live, http="/h", run=failing)
    except subprocess.CalledProcessError:
        pass
    assert len(calls) == 2 and not any("deploy" in c for c in calls), "a failed check stops before prepare and deploy"
    print("publish self-check passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("publication", nargs="?")
    parser.add_argument("--publish", action="store_true", help="really run the four steps (default: print them only)")
    parser.add_argument("--self-check", action="store_true")
    a = parser.parse_args()
    if a.self_check or not a.publication:
        self_check()
    else:
        publish(a.publication, dry_run=not a.publish)
