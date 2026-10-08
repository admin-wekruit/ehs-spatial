"""Configuration boundary. Neither domain documents nor URLs encode a vendor."""
from __future__ import annotations

from argus import ROOT
import os
import json
from pathlib import Path

from argus.platform.config import MONGO_SCHEMES, PlatformConfig
from argus.platform.storage import LocalBlobStore


def repository_from_url(url, **kwargs):
    """PANOPTES_DATABASE_URL scheme selects the repository: mongodb:// | mongodb+srv:// or postgres:// | postgresql://."""
    if url.startswith(MONGO_SCHEMES):
        from argus.platform.mongo import MongoRepository
        return MongoRepository(url, **kwargs)
    from argus.platform.postgres import PostgresRepository
    return PostgresRepository(url, **kwargs)


def blob_store(root):
    """PANOPTES_BLOB_ROOT selects the blob store: s3://bucket/prefix or a directory."""
    if str(root).startswith("s3://"):
        from argus.platform.s3_storage import S3BlobStore
        return S3BlobStore.from_url(str(root))
    return LocalBlobStore(root)


def policy_repository(repository):
    """The policy repository matching a repository's backend."""
    from argus.platform.mongo import MongoRepository
    if isinstance(repository, MongoRepository):
        from argus.platform.mongo_policy import MongoPolicyRepository
        return MongoPolicyRepository(repository)
    from argus.platform.policy_repository import PostgresPolicyRepository
    return PostgresPolicyRepository(repository)


def services(config=None):
    config = config or PlatformConfig.from_env()
    repository = repository_from_url(config.database_url, paid_budget=config.paid_budget)
    blobs = blob_store(config.blob_root)
    repository.blobs = blobs
    from argus.platform.reconstruction import provider_snapshot_from_env
    repository.execution_config = {"providerManifest": provider_snapshot_from_env()}
    preparation_path = os.environ.get('PANOPTES_RESEARCH_PREPARATION')
    if preparation_path:
        preparation = json.loads(Path(preparation_path).read_text())
        allowed = {'authority','budgetAtPreparation','runtimeManifest','callLimits',
                   'metricDefinitions','policyThresholds','split','purpose','maskErosionEnabled'}
        if not isinstance(preparation,dict) or set(preparation)-allowed:
            raise ValueError('Invalid server research preparation')
        repository.execution_config['researchPreparation'] = preparation
    return repository, blobs


def application():
    from fastapi.staticfiles import StaticFiles
    from argus.platform.api import create_app
    from argus.platform.agent_service import AgentService, GeminiAgentProvider
    from argus.platform.executor import LocalJobExecutor, ModalJobExecutor
    from argus.platform.policy_service import PolicyService
    config = PlatformConfig.from_env()
    repository, blobs = services(config)
    repository.migrate()
    if config.executor_backend == 'modal':
        executor = ModalJobExecutor(config.modal_app, config.modal_function)
    else:
        # Updated LocalJobExecutor launches the same module as Modal.
        executor = LocalJobExecutor()
    policies = PolicyService(policy_repository(repository), blobs)
    model, cost = os.environ.get('PANOPTES_AGENT_MODEL'), os.environ.get('PANOPTES_AGENT_CALL_BUDGET_USD')
    provider = GeminiAgentProvider(model=model, max_call_cost=float(cost)) if model and cost else None
    agents = AgentService(repository, provider, policies)
    app = create_app(repository=repository, blobs=blobs, executor=executor, agent_service=agents, policy_service=policies)
    web = Path(os.environ.get('PANOPTES_WEB_ROOT', ROOT / 'web' / 'dist'))
    if web.is_dir():
        app.mount('/', StaticFiles(directory=web, html=True), name='website')
    return app
