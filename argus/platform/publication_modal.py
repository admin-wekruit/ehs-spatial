"""Deploy an immutable, verified publication catalog for public report viewing.

Export each publication into CATALOG/PUBLICATION_ID with
python -m argus.platform.export_platform_publication, preserving previous directories.
Prepare verified HTTP files with python -m argus.platform.prepare_publication_site
--catalog CATALOG --output PANOPTES_DATA_ROOT/publications/http, then run:
  modal deploy argus/platform/publication_modal.py

The website remains on GitHub Pages. Frozen metadata/assets remain immutable;
visitor feedback uses a separate persistent volume and never gains scene writes.
Model calls remain disabled unless model, per-call reservation and total budget
are explicitly configured. Set GEMINI_API_KEY in the environment when enabling
the text-only assistant.
"""
import os
from pathlib import Path
from decimal import Decimal

import modal


publications = Path(os.environ["PANOPTES_DATA_ROOT"]).resolve() / "publications" if modal.is_local() else Path("/publications")
catalog = Path(os.environ.get("PANOPTES_PUBLICATION_CATALOG", publications / "catalog"))
prepared = Path(os.environ.get("PANOPTES_PUBLICATION_HTTP", publications / "http"))
# A separately deployed catalog (e.g. one new report) takes its own app and feedback volume, leaving the main service untouched.
app = modal.App(os.environ.get("PANOPTES_PUBLICATION_APP", "panoptes-publications"))
feedback_data = modal.Volume.from_name(os.environ.get("PANOPTES_PUBLICATION_FEEDBACK_VOLUME", "panoptes-publication-feedback"), create_if_missing=True)
feedback_config = {key: os.environ.get(key, "") for key in (
    "PANOPTES_FEEDBACK_MODEL", "PANOPTES_FEEDBACK_TOTAL_BUDGET_USD", "PANOPTES_FEEDBACK_CALL_RESERVATION_USD")}
for key in ("PANOPTES_FEEDBACK_TOTAL_BUDGET_USD", "PANOPTES_FEEDBACK_CALL_RESERVATION_USD"):
    if feedback_config[key] and (not Decimal(feedback_config[key]).is_finite() or Decimal(feedback_config[key]) < 0):
        raise ValueError("Feedback budget must be finite and nonnegative")
model_enabled = bool(feedback_config["PANOPTES_FEEDBACK_MODEL"] and
                     Decimal(feedback_config["PANOPTES_FEEDBACK_TOTAL_BUDGET_USD"] or "0") > 0 and
                     Decimal(feedback_config["PANOPTES_FEEDBACK_CALL_RESERVATION_USD"] or "0") > 0)
feedback_secrets = []
if modal.is_local() and model_enabled:
    values = {"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY", "")}
    if not values.get("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is required for enabled feedback")
    feedback_secrets = [modal.Secret.from_dict(values)]
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1", "libgl1")
    .pip_install("open3d==0.19.0", "fastapi==0.139.0", "starlette==1.3.1", "pydantic==2.13.4", "google-genai==2.11.0", "numpy==2.5.1", "shapely==2.1.2")
    .env({"PYTHONPATH": "/app", "OMP_NUM_THREADS":"1", **feedback_config})
    .add_local_python_source("argus")
    .add_local_dir(catalog, "/publications")
    .add_local_dir(prepared, "/publication-http")
)


@app.function(image=image, volumes={"/feedback": feedback_data}, secrets=feedback_secrets,
              cpu=1, memory=6144, timeout=180, scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    from argus.platform.publication_site import create_app, FEEDBACK_PATH, IDENTITY_SUGGESTIONS_PATH
    from argus.platform.feedback import FeedbackService, FEEDBACK_INSTRUCTION, persistent_feedback_app
    from argus.platform.agent_service import GeminiAgentProvider

    model = os.environ.get("PANOPTES_FEEDBACK_MODEL")
    total = os.environ.get("PANOPTES_FEEDBACK_TOTAL_BUDGET_USD") or None
    reservation = os.environ.get("PANOPTES_FEEDBACK_CALL_RESERVATION_USD") or None
    enabled = bool(model and total and reservation and Decimal(total) > 0 and Decimal(reservation) > 0)
    provider = GeminiAgentProvider(model=model, max_call_cost=float(reservation), instruction=FEEDBACK_INSTRUCTION,
        http_options={"timeout": 120000, "retry_options": {"attempts": 1}}) if enabled else None
    feedback = FeedbackService("/feedback/feedback.sqlite", provider=provider, total_budget=total,
                               call_reservation=reservation, checkpoint=feedback_data.commit)

    server = create_app("/publications", allowed_origins=["https://admin-wekruit.github.io"], feedback=feedback, prepared_dir="/publication-http")
    return persistent_feedback_app(server, feedback_data, lambda path: FEEDBACK_PATH.fullmatch(path) or IDENTITY_SUGGESTIONS_PATH.fullmatch(path))
