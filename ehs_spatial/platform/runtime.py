"""Configuration boundary. Neither domain documents nor URLs encode a vendor."""
from __future__ import annotations

import os
from pathlib import Path

from .config import PlatformConfig
from .postgres import PostgresRepository
from .storage import LocalBlobStore, S3BlobStore


def services(config=None):
    config = config or PlatformConfig.from_env()
    repository = PostgresRepository(config.database_url, paid_budget=config.paid_budget)
    if config.blob_backend == 'local':
        blobs = LocalBlobStore(config.blob_root)
    else:
        blobs = S3BlobStore(config.s3_bucket)
    repository.blobs = blobs
    from .reconstruction import provider_snapshot_from_env
    repository.execution_config = {"providerManifest": provider_snapshot_from_env()}
    return repository, blobs


def application():
    from fastapi.staticfiles import StaticFiles
    from .api import create_app
    from .agent_service import AgentService, GeminiAgentProvider
    from .executor import LocalJobExecutor, ModalJobExecutor
    from .policy_service import PolicyService
    from .policy_repository import PostgresPolicyRepository
    config = PlatformConfig.from_env()
    repository, blobs = services(config)
    repository.migrate()
    if config.executor_backend == 'modal':
        executor = ModalJobExecutor(config.modal_app, config.modal_function)
    else:
        # Updated LocalJobExecutor launches the same module as Modal.
        executor = LocalJobExecutor()
    policies = PolicyService(PostgresPolicyRepository(repository), blobs)
    model, cost = os.environ.get('PANOPTES_AGENT_MODEL'), os.environ.get('PANOPTES_AGENT_CALL_BUDGET_USD')
    provider = GeminiAgentProvider(model=model, max_call_cost=float(cost)) if model and cost else None
    agents = AgentService(repository, provider, policies)
    app = create_app(repository=repository, blobs=blobs, executor=executor, agent_service=agents, policy_service=policies)
    web = Path(os.environ.get('PANOPTES_WEB_ROOT', Path(__file__).resolve().parents[2] / 'web' / 'dist'))
    if web.is_dir():
        app.mount('/', StaticFiles(directory=web, html=True), name='website')
    return app
