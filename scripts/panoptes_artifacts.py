"""Sync a local artifact directory with an S3 prefix, content addressed.

  python scripts/panoptes_artifacts.py push DIR s3://bucket/prefix/<name>
  python scripts/panoptes_artifacts.py pull s3://bucket/prefix/<name> DIR

Every object lives at <prefix>/<name>/<relative path>; <prefix>/<name>/manifest.json maps each
relative path to {sha256, size}. push uploads only files whose sha256 differs from the remote
manifest (multipart for large files) and writes the manifest last; pull downloads only files whose
local sha256 differs and verifies every downloaded file against the manifest. The boto3 client is
the one S3BlobStore uses (AWS_ENDPOINT_URL, AWS_* credentials, path-style addressing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ehs_spatial.platform.s3_storage import parse_s3_url, s3_client  # noqa: E402

MANIFEST = "manifest.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_manifest(root):
    root = Path(root)
    return {path.relative_to(root).as_posix(): {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(root.rglob("*")) if path.is_file() and path.name != MANIFEST}


def remote_manifest(client, bucket, prefix):
    try:
        body = client.get_object(Bucket=bucket, Key=prefix + MANIFEST)["Body"].read()
    except client.exceptions.NoSuchKey:
        return {}
    return json.loads(body)


def push(client, root, url):
    bucket, prefix = parse_s3_url(url)
    manifest, remote = local_manifest(root), remote_manifest(client, bucket, prefix)
    uploaded = 0
    for relative, entry in manifest.items():
        if remote.get(relative) == entry:
            continue
        client.upload_file(str(Path(root) / relative), bucket, prefix + relative, ExtraArgs={"Metadata": {"sha256": entry["sha256"]}})
        uploaded += 1
    client.put_object(Bucket=bucket, Key=prefix + MANIFEST, Body=json.dumps(manifest, indent=1, sort_keys=True).encode(), ContentType="application/json")
    return {"files": len(manifest), "uploaded": uploaded, "skipped": len(manifest) - uploaded}


def pull(client, url, root):
    bucket, prefix = parse_s3_url(url)
    root = Path(root)
    manifest = remote_manifest(client, bucket, prefix)
    if not manifest:
        raise SystemExit(f"no {MANIFEST} under {url}")
    downloaded = 0
    for relative, entry in manifest.items():
        target = root / relative
        if target.is_file() and target.stat().st_size == entry["size"] and sha256_file(target) == entry["sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        client.download_file(bucket, prefix + relative, str(partial))
        if sha256_file(partial) != entry["sha256"] or partial.stat().st_size != entry["size"]:
            partial.unlink()
            raise SystemExit(f"sha256 mismatch for {relative}")
        partial.replace(target)
        downloaded += 1
    (root / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return {"files": len(manifest), "downloaded": downloaded, "skipped": len(manifest) - downloaded}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("push"); up.add_argument("directory"); up.add_argument("url")
    down = commands.add_parser("pull"); down.add_argument("url"); down.add_argument("directory")
    args = parser.parse_args(argv)
    client = s3_client()
    result = push(client, args.directory, args.url) if args.command == "push" else pull(client, args.url, args.directory)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
