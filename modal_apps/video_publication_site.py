"""A standalone public copy of the phase-2 video reports: the frozen publication API and the report viewer on one origin.

The shared panoptes-publications app stays as it is: its viewer lives on GitHub Pages, built from main, without the
video, dynamic-layer and video-memory views. This app serves only the catalog it is deployed with, read-only, plus this
branch's web/dist build, which calls /api on its own origin (VITE_API_ORIGIN empty). No platform database, no project
credentials, no feedback writes, no model calls. Export, verify and prepare the catalog exactly as OPERATIONS.md says
for the shared site, into their own directories:

  PANOPTES_PUBLICATION_CATALOG=/abs/catalog PANOPTES_PUBLICATION_HTTP=/abs/http modal deploy modal_apps/video_publication_site.py
  -> https://<workspace>--panoptes-video-publications-web.modal.run/app.html#/reports/<publication id>
"""
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
catalog = Path(os.environ.get("PANOPTES_PUBLICATION_CATALOG", ROOT / ".platform/video-publication-catalog"))
prepared = Path(os.environ.get("PANOPTES_PUBLICATION_HTTP", ROOT / ".platform/video-publication-http"))
app = modal.App("panoptes-video-publications")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1", "libgl1")
    .pip_install("open3d==0.19.0", "fastapi==0.139.0", "starlette==1.3.1", "pydantic==2.13.4", "google-genai==2.11.0", "numpy==2.5.1", "shapely==2.1.2")
    .env({"PYTHONPATH": "/app", "OMP_NUM_THREADS": "1"})
    .add_local_file(ROOT / "ehs_spatial/__init__.py", "/app/ehs_spatial/__init__.py")
    .add_local_dir(catalog, "/publications")
    .add_local_dir(prepared, "/publication-http")
    .add_local_dir(ROOT / "web/dist", "/web")
)
for module in ("__init__", "publication_site", "publication_view", "scene_measurements", "planar_surfaces", "spatial", "blender_export", "feedback", "contracts", "agent_service", "repository", "identity"):
    image = image.add_local_file(ROOT / f"ehs_spatial/platform/{module}.py", f"/app/ehs_spatial/platform/{module}.py")


def viewer_app(catalog_dir, prepared_dir, web_dir):
    """The read-only publication API with the report viewer's static build mounted behind it."""
    from fastapi.staticfiles import StaticFiles
    from ehs_spatial.platform.publication_site import create_app
    server = create_app(catalog_dir, allowed_origins=[], prepared_dir=prepared_dir)
    server.mount("/", StaticFiles(directory=web_dir, html=True), name="viewer")  # after the API routes, so /api wins
    return server


@app.function(image=image, cpu=1, memory=3072, timeout=180, scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    return viewer_app("/publications", "/publication-http", "/web")
