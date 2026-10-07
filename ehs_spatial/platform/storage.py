"""Content addressed immutable blobs; bytes are verified before DB registration."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import re
import tempfile
from typing import Protocol

from .contracts import PlatformError


def _key(storage_key):
    if not re.fullmatch(r"sha256/[a-f0-9]{64}", storage_key):
        raise PlatformError("invalid_storage_key")
    return storage_key


def _metadata(data, media_type):
    sha = hashlib.sha256(data).hexdigest()
    return {"storageKey": "sha256/" + sha, "sha256": sha, "sizeBytes": len(data), "mediaType": media_type}


def _verify(data, sha256, size_bytes):
    if len(data) != size_bytes or hashlib.sha256(data).hexdigest() != sha256:
        raise PlatformError("blob_integrity_error", 409)
    return data


def _source_bytes(source):
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    return source.read()


class BlobStore(Protocol):
    def put_immutable(self, key: str, source, sha256: str, byte_size: int, media_type: str) -> dict: ...
    def head(self, key: str) -> dict | None: ...
    def open(self, key: str): ...
    def public_url(self, key: str) -> str: ...
    def put(self, data: bytes, media_type: str) -> dict: ...
    def get(self, storage_key: str, sha256: str, size_bytes: int) -> bytes: ...
    def url(self, storage_key: str, asset_id: str) -> str: ...


class LocalBlobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        (self.root / "sha256").mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes, media_type: str = "application/octet-stream") -> dict:
        metadata = _metadata(data, media_type)
        return self.put_immutable(metadata["storageKey"], data, metadata["sha256"], metadata["sizeBytes"], media_type)

    def put_immutable(self, key, source, sha256, byte_size, media_type):
        data = _verify(_source_bytes(source), sha256, byte_size)
        if _key(key) != "sha256/" + sha256:
            raise PlatformError("blob_integrity_error", 409)
        metadata = _metadata(data, media_type)
        destination = self.root / key
        handle, temporary = tempfile.mkstemp(prefix=".upload-", dir=destination.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError:
                pass
        finally:
            os.unlink(temporary)
        self.get(metadata["storageKey"], metadata["sha256"], metadata["sizeBytes"])
        return metadata

    def get(self, storage_key, sha256, size_bytes):
        try:
            data = (self.root / _key(storage_key)).read_bytes()
        except FileNotFoundError:
            raise PlatformError("blob_not_found", 404) from None
        return _verify(data, sha256, size_bytes)

    def url(self, storage_key, asset_id):
        _key(storage_key)
        return f"/api/assets/{asset_id}/content"

    def head(self, key):
        path = self.root / _key(key)
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
        _verify(data, key.removeprefix("sha256/"), len(data))
        return _metadata(data, "application/octet-stream")

    def open(self, key):
        metadata = self.head(key)
        if metadata is None:
            raise PlatformError("blob_not_found", 404)
        return io.BytesIO(self.get(key, metadata["sha256"], metadata["sizeBytes"]))

    def public_url(self, key):
        return "/api/blobs/" + _key(key)


def __getattr__(name):
    if name == "S3BlobStore":  # moved to s3_storage; lazy so boto3 is only imported for s3:// roots
        from .s3_storage import S3BlobStore
        return S3BlobStore
    raise AttributeError(name)
