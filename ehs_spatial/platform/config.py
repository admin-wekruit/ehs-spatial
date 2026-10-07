"""Backend selection by URL. Missing paid budget disables paid execution.

PANOPTES_DATABASE_URL: postgres:// | postgresql:// -> PostgreSQL, mongodb:// | mongodb+srv:// -> MongoDB.
PANOPTES_BLOB_ROOT: a directory -> LocalBlobStore, s3://bucket/prefix -> S3BlobStore.
"""
from dataclasses import dataclass
from decimal import Decimal
import os

POSTGRES_SCHEMES = ("postgres://", "postgresql://")
MONGO_SCHEMES = ("mongodb://", "mongodb+srv://")


@dataclass(frozen=True)
class PlatformConfig:
    database_url: str
    blob_root: str = ".platform/blobs"
    executor_backend: str = "local"
    modal_app: str | None = None
    modal_function: str | None = None
    paid_budget: Decimal | None = None

    @property
    def database_backend(self):
        return "mongo" if self.database_url.startswith(MONGO_SCHEMES) else "postgres"

    @property
    def blob_backend(self):
        return "s3" if str(self.blob_root).startswith("s3://") else "local"

    @classmethod
    def from_env(cls):
        dsn = os.environ.get("PANOPTES_DATABASE_URL")
        if not dsn:
            raise ValueError("PANOPTES_DATABASE_URL is required")
        if not dsn.startswith(POSTGRES_SCHEMES + MONGO_SCHEMES):
            raise ValueError("PANOPTES_DATABASE_URL must start with postgres://, postgresql://, mongodb:// or mongodb+srv://")
        value = os.environ.get("PANOPTES_PAID_BUDGET_USD")
        budget = Decimal(value) if value is not None else None
        if budget is not None and (not budget.is_finite() or budget < 0):
            raise ValueError("Invalid PANOPTES_PAID_BUDGET_USD")
        root = os.environ.get("PANOPTES_BLOB_ROOT", ".platform/blobs")
        if os.environ.get("PANOPTES_BLOB_BACKEND") == "s3" and not root.startswith("s3://"):
            raise ValueError("PANOPTES_BLOB_BACKEND is retired: set PANOPTES_BLOB_ROOT=s3://bucket/prefix")
        executor = os.environ.get("PANOPTES_EXECUTOR_BACKEND", "local")
        if executor not in ("local", "modal"):
            raise ValueError("Unsupported platform backend")
        return cls(dsn, root, executor, os.environ.get("PANOPTES_MODAL_APP"), os.environ.get("PANOPTES_MODAL_FUNCTION"), budget)
