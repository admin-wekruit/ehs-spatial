"""Deploy an immutable, verified publication catalog for public report viewing.

Export each publication into CATALOG/PUBLICATION_ID with
scripts/export_platform_publication.py, preserving previous directories, then run:
  PANOPTES_PUBLICATION_CATALOG=/absolute/catalog .venv/bin/modal deploy modal_apps/publication_site.py

The website remains on GitHub Pages. This service supplies only frozen report
metadata and assets; it has no database, model credentials, or write capability.
"""
import os
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
catalog = Path(os.environ.get("PANOPTES_PUBLICATION_CATALOG", ROOT / ".platform/publication-catalog"))
app = modal.App("panoptes-publications")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.139.0", "starlette==1.3.1")
    .env({"PYTHONPATH": "/app"})
    .add_local_file(ROOT / "ehs_spatial/platform/publication_site.py", "/app/publication_server.py")
    .add_local_dir(catalog, "/publications")
)


@app.function(image=image, cpu=1, memory=1024, scaledown_window=300, max_containers=3)
@modal.concurrent(max_inputs=32)
@modal.asgi_app()
def web():
    from publication_server import create_app

    return create_app("/publications", allowed_origins=["https://admin-wekruit.github.io"])
