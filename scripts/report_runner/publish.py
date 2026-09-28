"""Publish one imported report to the shared publication site, the way it was proven by hand (2026-09-27, publish-4.sh):

  1. the local report API (127.0.0.1:8792) answers, or is started for this call (its report-server command) and stopped after
  2. export the publication from that API into the shared catalog, CAT/<id> (scripts/export_platform_publication.py)
  3. every blob the previous publication of the same project already holds becomes an APFS clone of it (cp -c): no second copy
  4. a new HTTP dir from the whole catalog (scripts/prepare_publication_site.py)
  5. checks (tests/check_publication_site.py): the new publication against the API (--source-api), then the full catalog; each must
     print its two PASS lines
  6. only when all pass: deploy from the platform repo, PANOPTES_PUBLICATION_CATALOG / PANOPTES_PUBLICATION_HTTP set,
     .venv/bin/modal deploy modal_apps/publication_site.py (app panoptes-publications, which the GitHub Pages viewer reads)
  7. remove the previous HTTP dir

Runs only under `run_video_report.py --publish`. `publish(id, dry_run=True)` prints the steps and runs none.

  python -m report_runner.publish PUBLICATION_ID            # print only
  python -m report_runner.publish PUBLICATION_ID --publish  # the seven steps
"""
import argparse
import contextlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request

REPO = Path(__file__).resolve().parents[2]
API = "http://127.0.0.1:8792"
PLATFORM = Path("/Users/adam/Desktop/Tesla/panoptes-platform")  # the shared site's repo, venv and live catalog
DATABASE_URL = "postgresql://panoptes@127.0.0.1:54329/panoptes_video"


def say(text):
    from .store import say as redacted
    redacted(text)


def api_up(api=API):
    try:
        with urllib.request.urlopen(api + "/api/publications", timeout=3):
            return True
    except OSError:
        return False


@contextlib.contextmanager
def report_server(art, platform=PLATFORM, up=api_up, wait_s=90):
    """The local report API for the export and the source check: the running one, or one started here (as report-server.sh:
    uvicorn ehs_spatial.platform.runtime on 127.0.0.1:8792 over this repo, the video database and $ART's blobs) and stopped after."""
    if up():
        yield
        return
    env = {**os.environ, "PANOPTES_DATABASE_URL": os.environ.get("PANOPTES_DATABASE_URL", DATABASE_URL),
           "PANOPTES_BLOB_ROOT": os.environ.get("PANOPTES_BLOB_ROOT", str(Path(art) / ".platform/blobs")), "PANOPTES_WEB_ROOT": str(REPO / "web/dist")}
    server = subprocess.Popen([str(platform / ".venv/bin/uvicorn"), "ehs_spatial.platform.runtime:application", "--factory", "--host", "127.0.0.1",
                               "--port", API.rsplit(":", 1)[1]], cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + wait_s
        while not up():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("the local report API did not come up")
            time.sleep(1)
        yield
    finally:
        server.terminate()
        server.wait(timeout=30)


def previous_publication(publication_id, catalog, db):
    """The newest other publication of the same project that the catalog holds (its blobs are the ones to clone), or None."""
    project = db.publication(publication_id)["projectId"]
    return next((p for p in db.project_publications(project) if p != publication_id and (Path(catalog) / p).is_dir()), None)


def steps(publication_id, *, platform=PLATFORM, catalog=None, previous=None, stamp=None, python=sys.executable):
    """The seven steps, each (name, argv or None, cwd, env overrides). The HTTP dirs live beside the catalog: the new one is
    publication-http-<stamp>, the previous one the newest before it."""
    catalog = Path(catalog or platform / ".platform/publication-catalog")
    root = catalog.parent
    stamp = stamp or time.strftime("%Y%m%dT%H%M%S")
    http = root / f"publication-http-{stamp}"
    old = sorted((d for d in root.glob("publication-http-*") if d.is_dir() and d != http), key=lambda d: d.stat().st_mtime)
    single = Path(tempfile.gettempdir()) / f"publication-check-{publication_id}"  # a catalog of the new publication alone (a link)
    return [("export", [python, str(REPO / "scripts/export_platform_publication.py"), "--api", API, "--publication", publication_id, "--output",
                        str(catalog / publication_id)], REPO, {}),
            ("clone blobs", ["clone", str(catalog / publication_id), str(catalog / previous)] if previous else None, None, {}),
            ("prepare", [python, str(REPO / "scripts/prepare_publication_site.py"), "--catalog", str(catalog), "--output", str(http)], REPO, {}),
            ("check new", [python, str(REPO / "tests/check_publication_site.py"), "--catalog", str(single), "--source-api", API], REPO, {"single": str(single)}),
            ("check all", [python, str(REPO / "tests/check_publication_site.py"), "--catalog", str(catalog)], REPO, {}),
            ("deploy", [str(platform / ".venv/bin/modal"), "deploy", "modal_apps/publication_site.py"], platform,
             {"PANOPTES_PUBLICATION_CATALOG": str(catalog), "PANOPTES_PUBLICATION_HTTP": str(http)}),
            ("remove old http", ["rm", "-rf", str(old[-1])] if old else None, None, {})]


def clone_blobs(new, previous):
    """Each blob of `new` that `previous` holds under the same name (its sha256) becomes an APFS clone of that file."""
    count = 0
    for blob in sorted((Path(new) / "blobs").iterdir()):
        same = Path(previous) / "blobs" / blob.name
        if same.is_file():
            subprocess.run(["cp", "-c", str(same), str(blob) + ".clone"], check=True)
            os.replace(str(blob) + ".clone", blob)
            count += 1
    return count


def publish(publication_id, *, dry_run, art=None, platform=PLATFORM, catalog=None, db=None, run=subprocess.run, server=None, stamp=None):
    """Returns 0 when deployed (or planned: dry_run), 1 when a check failed (nothing deployed). dry_run prints the steps."""
    if dry_run:
        for name, argv, cwd, env in steps(publication_id, platform=platform, catalog=catalog, previous="<previous publication of the project>", stamp=stamp):
            say(f"{name}: " + (" ".join([f"{k}={v}" for k, v in env.items() if k.isupper()] + argv) if argv else "(nothing)") + (f"  [in {cwd}]" if cwd else ""))
        return 0
    catalog = Path(catalog or platform / ".platform/publication-catalog")
    if not catalog.is_dir() or not any(catalog.iterdir()):  # the deploy serves this catalog: a fresh one would drop every other report
        raise SystemExit(f"no live catalog at {catalog}")
    if db is None:
        from .adopt import Database
        db = Database()
    previous = previous_publication(publication_id, catalog, db)
    plan = steps(publication_id, platform=platform, catalog=catalog, previous=previous, stamp=stamp)
    with (server or report_server)(art):
        for name, argv, cwd, env in plan:
            if argv is None:
                continue
            if name == "export" and (catalog / publication_id / "bundle.json").is_file():
                say(f"export: {publication_id} is already in the catalog; the source check verifies it")
                continue
            if name == "clone blobs":
                say(f"clone blobs: {clone_blobs(argv[1], argv[2])} blobs of {Path(argv[1]).name} are clones of {Path(argv[2]).name}'s")
                continue
            if name == "check new":
                single = Path(env["single"])
                single.mkdir(parents=True, exist_ok=True)
                link = single / publication_id
                if link.is_symlink():  # an earlier check's link, to another catalog or dangling: this check reads this catalog's copy
                    link.unlink()
                if not link.exists():
                    link.symlink_to(catalog / publication_id)
            done = run(argv, cwd=cwd, env={**os.environ, **{k: v for k, v in env.items() if k.isupper()}}, capture_output=True, text=True)
            passes = [line for line in (done.stdout or "").splitlines() if line.startswith("PASS")]
            say(f"{name}: exit {done.returncode}" + "".join(f"\n  {line[:110]}" for line in passes))
            if name.startswith("check") and (done.returncode or len(passes) != 2):
                say(f"{name} did not pass: not deployed")
                return 1
            if done.returncode:
                raise RuntimeError(f"{name} failed (exit {done.returncode}): {(done.stderr or '')[-800:]}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("publication")
    parser.add_argument("--publish", action="store_true", help="really run the seven steps (default: print them only)")
    a = parser.parse_args()
    from .store import art_root
    sys.exit(publish(a.publication, dry_run=not a.publish, art=art_root()))
