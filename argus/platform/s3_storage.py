"""S3-compatible content-addressed blobs; keys are <prefix>sha256/<hex> like LocalBlobStore.

PANOPTES_BLOB_ROOT=s3://bucket/prefix selects this store. The boto3 client reads AWS_ENDPOINT_URL
(or PANOPTES_S3_ENDPOINT_URL), AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY and AWS_DEFAULT_REGION
(or PANOPTES_S3_REGION) from the environment, uses path-style addressing for MinIO-like servers and
only the basic API (PUT, GET, HEAD, multipart upload, presigned GET).
"""
from __future__ import annotations

from argus import ROOT
import io
import os

from argus.platform.contracts import PlatformError
from argus.platform.storage import _key, _metadata, _source_bytes, _verify

MULTIPART_THRESHOLD = 8 * 1024 * 1024
MIN_PART_SIZE = 5 * 1024 * 1024


def s3_client(**overrides):
    import boto3
    from botocore.config import Config
    options = {"endpoint_url": os.environ.get("PANOPTES_S3_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL") or None,
               "region_name": os.environ.get("PANOPTES_S3_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1",
               "config": Config(signature_version="s3v4", s3={"addressing_style": "path"},
                                request_checksum_calculation="when_required", response_checksum_validation="when_required")}
    return boto3.client("s3", **{**options, **overrides})


def parse_s3_url(url):
    """'s3://bucket/some/prefix' -> ('bucket', 'some/prefix/'); 's3://bucket' -> ('bucket', '')."""
    if not isinstance(url, str) or not url.startswith("s3://"):
        raise ValueError("Expected s3://bucket[/prefix]")
    bucket, _, prefix = url[5:].partition("/")
    if not bucket:
        raise ValueError("S3 bucket is required")
    prefix = prefix.strip("/")
    return bucket, prefix + "/" if prefix else ""


def _missing_object(exc):
    response = getattr(exc, "response", {})
    return response.get("Error", {}).get("Code") in {"NoSuchKey", "NotFound", "404"}


class S3BlobStore:
    def __init__(self, bucket: str, *, client=None, prefix="panoptes/", multipart_threshold=MULTIPART_THRESHOLD):
        if not bucket:
            raise ValueError("S3 bucket is required")
        self.bucket, self.client, self.prefix = bucket, client if client is not None else s3_client(), prefix
        self.multipart_threshold = multipart_threshold

    @classmethod
    def from_url(cls, url, **kwargs):
        bucket, prefix = parse_s3_url(url)
        return cls(bucket, prefix=prefix, **kwargs)

    def put(self, data: bytes, media_type: str = "application/octet-stream") -> dict:
        metadata = _metadata(data, media_type)
        return self.put_immutable(metadata["storageKey"], data, metadata["sha256"], metadata["sizeBytes"], media_type)

    def put_immutable(self, key, source, sha256, byte_size, media_type):
        data = _verify(_source_bytes(source), sha256, byte_size)
        if _key(key) != "sha256/" + sha256:
            raise PlatformError("blob_integrity_error", 409)
        metadata = _metadata(data, media_type)
        # ponytail: content-addressed application writes, not provider WORM.
        # Concurrent valid writers for this key can only supply identical bytes.
        try:
            self.get(key, sha256, byte_size)
            return metadata
        except PlatformError as exc:
            if exc.code != "blob_not_found":
                raise
        try:
            if len(data) > self.multipart_threshold:
                from boto3.s3.transfer import TransferConfig
                config = TransferConfig(multipart_threshold=self.multipart_threshold, multipart_chunksize=max(self.multipart_threshold, MIN_PART_SIZE))
                self.client.upload_fileobj(io.BytesIO(data), self.bucket, self.prefix + metadata["storageKey"], Config=config,
                                           ExtraArgs={"ContentType": media_type, "Metadata": {"sha256": sha256}})
            else:
                self.client.put_object(Bucket=self.bucket, Key=self.prefix + metadata["storageKey"], Body=data,
                                       ContentType=media_type, Metadata={"sha256": sha256})
        except Exception:
            raise PlatformError("blob_upload_failed", 502) from None
        self.get(metadata["storageKey"], metadata["sha256"], metadata["sizeBytes"])
        return metadata

    def get(self, storage_key, sha256, size_bytes):
        key = self.prefix + _key(storage_key)
        try:
            result = self.client.get_object(Bucket=self.bucket, Key=key)
            with result["Body"] as stream:
                data = stream.read()
        except Exception as exc:
            if _missing_object(exc):
                raise PlatformError("blob_not_found", 404) from None
            raise PlatformError("blob_read_failed", 502) from None
        return _verify(data, sha256, size_bytes)

    def url(self, storage_key, asset_id):
        return self.public_url(storage_key)

    def public_url(self, storage_key):
        key = self.prefix + _key(storage_key)
        base = os.environ.get("PANOPTES_BLOB_PUBLIC_BASE")
        if base:
            return base.rstrip("/") + "/" + key
        expiry = int(os.environ.get("PANOPTES_S3_URL_EXPIRY_S", "3600"))
        return self.client.generate_presigned_url("get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expiry)

    def head(self, key):
        storage_key = self.prefix + _key(key)
        try:
            result = self.client.head_object(Bucket=self.bucket, Key=storage_key)
        except Exception as exc:
            if _missing_object(exc):
                return None
            raise PlatformError("blob_read_failed", 502) from None
        sha = key.removeprefix("sha256/")
        # HEAD is metadata only; get/open verify actual bytes against the key.
        return {"storageKey": key, "sha256": sha, "sizeBytes": result["ContentLength"], "mediaType": result.get("ContentType", "application/octet-stream")}

    def open(self, key):
        metadata = self.head(key)
        if metadata is None:
            raise PlatformError("blob_not_found", 404)
        return io.BytesIO(self.get(key, metadata["sha256"], metadata["sizeBytes"]))
