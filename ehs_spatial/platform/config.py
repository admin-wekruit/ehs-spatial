"""Explicit backend selection. Missing paid budget disables paid execution."""
from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path


@dataclass(frozen=True)
class PlatformConfig:
    database_url: str
    blob_backend: str = "local"
    blob_root: Path = Path(".platform/blobs")
    s3_bucket: str | None = None
    executor_backend: str = "local"
    modal_app: str | None = None
    modal_function: str | None = None
    paid_budget: Decimal | None = None

    @classmethod
    def from_env(cls):
        dsn = os.environ.get("PANOPTES_DATABASE_URL")
        if not dsn:
            raise ValueError("PANOPTES_DATABASE_URL is required")
        value = os.environ.get("PANOPTES_PAID_BUDGET_USD")
        budget = Decimal(value) if value is not None else None
        if budget is not None and (not budget.is_finite() or budget < 0):
            raise ValueError("Invalid PANOPTES_PAID_BUDGET_USD")
        blob = os.environ.get("PANOPTES_BLOB_BACKEND", "local")
        executor = os.environ.get("PANOPTES_EXECUTOR_BACKEND", "local")
        if blob not in ("local", "s3") or executor not in ("local", "modal"):
            raise ValueError("Unsupported platform backend")
        return cls(dsn, blob, Path(os.environ.get("PANOPTES_BLOB_ROOT", ".platform/blobs")),
                   os.environ.get("PANOPTES_S3_BUCKET"), executor, os.environ.get("PANOPTES_MODAL_APP"),
                   os.environ.get("PANOPTES_MODAL_FUNCTION"), budget)
