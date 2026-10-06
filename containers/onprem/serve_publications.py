"""On-prem publication service: the read-only app that modal_apps/publication_site.py deploys (create_app over an exported
catalog, prepared HTTP bodies, CORS for the report website, per-object visitor feedback in SQLite), under plain uvicorn.

  /app/.venv/bin/uvicorn serve_publications:app --factory --host 0.0.0.0 --port 8793

PANOPTES_PUBLICATION_CATALOG  catalog dir (one export per publication id; scripts/export_platform_publication.py), default /catalog
PANOPTES_PUBLICATION_HTTP     prepared bodies (scripts/prepare_publication_site.py); empty, or no index.json there yet = verify and
                              serve the catalog directly
PANOPTES_PUBLICATION_ORIGINS  extra report website origins allowed by CORS (comma-separated); empty = same origin only, as behind
                              the nginx /api/ proxy (nginx.conf)
PANOPTES_FEEDBACK_DB          SQLite file for visitor feedback, default /feedback/feedback.sqlite

A fresh install (no publication in the catalog yet) starts and serves an empty report library; restart after the first publish.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from uuid import UUID

from ehs_spatial.platform.feedback import FeedbackService
from ehs_spatial.platform.publication_site import create_app


def empty_library():
    """Prepared bodies for a catalog without publications: the library's list routes, each {"items": []}; everything else 404."""
    out, body = Path(tempfile.mkdtemp(prefix="panoptes-empty-library-")), b'{"items":[]}'
    sha = hashlib.sha256(body).hexdigest()
    (out / f"{sha}.json.gz").write_bytes(gzip.compress(body, mtime=0))
    entry = {"file": f"{sha}.json.gz", "sha256": sha, "sizeBytes": len(body)}
    routes = {route: entry for route in ("/api/publications", "/api/projects", "/api/policies")}
    (out / "index.json").write_text(json.dumps({"schemaVersion": 1, "routes": routes, "assets": {}}))
    return out


def publications_only(catalog):
    """The catalog's publication exports (UUID-named dirs), linked into a fresh dir: an export's staging dir or any other entry
    in the catalog must not stop the service from starting (read_catalog refuses non-UUID names)."""
    out = Path(tempfile.mkdtemp(prefix="panoptes-catalog-"))
    for path in Path(catalog).iterdir():
        try:
            if path.is_dir() and str(UUID(path.name)) == path.name:
                (out / path.name).symlink_to(path.resolve(), target_is_directory=True)
        except ValueError:
            continue
    return out


def app():
    catalog = publications_only(os.environ.get("PANOPTES_PUBLICATION_CATALOG", "/catalog"))
    prepared = os.environ.get("PANOPTES_PUBLICATION_HTTP") or None
    if not any(catalog.iterdir()):
        prepared = empty_library()   # nothing published yet
    elif prepared and not (Path(prepared) / "index.json").exists():
        prepared = None              # exported but not prepared yet: verify and serve the catalog itself
    origins = [origin.strip() for origin in os.environ.get("PANOPTES_PUBLICATION_ORIGINS", "").split(",") if origin.strip()]
    # ponytail: the feedback assistant model stays off on-prem (it is a SaaS call); feedback is stored only. One uvicorn
    # process is the single SQLite writer; more workers need the shared-SQL store noted in feedback.persistent_feedback_app.
    feedback = FeedbackService(os.environ.get("PANOPTES_FEEDBACK_DB", "/feedback/feedback.sqlite"))
    return create_app(catalog, allowed_origins=origins, feedback=feedback, prepared_dir=prepared)
