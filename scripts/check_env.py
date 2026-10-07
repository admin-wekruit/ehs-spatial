#!/usr/bin/env python3
"""Validate a Panoptes .env against env.template, the one configuration surface. Stdlib only, no network unless --live.

  python scripts/check_env.py [.env] [--template env.template] [--strict] [--live]

Errors (exit 1): a required key missing or empty, an unknown *_BACKEND value, an http backend without its URL(s), a URL / database
URL / blob root that does not parse, a non-integer count, HF_TOKEN stored in the file.
Warnings: placeholder values (change-me, example), paths that do not exist, keys the template does not know, a cloud default left
in place. --strict turns warnings into errors (what `make check-env` runs on the customer's filled .env).
--live: GET <root>/healthz with X-API-Key on every *_HTTP_URLS entry and *_HTTP_URL root; any non-200 is an error.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("PANOPTES_SERVICE_API_KEY", "PANOPTES_DATABASE_URL", "PANOPTES_BLOB_ROOT", "PANOPTES_SERVING", "PANOPTES_WORKCELL", "WEIGHTS")
# backend key -> (allowed values, code default, url key, url key holds a comma list of roots)
BACKENDS = {
    "SAM3D_BACKEND": (("http", "local", "modal"), "http", "SAM3D_HTTP_URLS", True),
    "GEOMETRY_MVS_BACKEND": (("http", "local", "modal"), "http", "GEOMETRY_MVS_HTTP_URLS", True),
    "SAM3_BACKEND": (("fal", "modal", "http"), "fal", "SAM3_HTTP_URL", False),
    "GEOMETRY_BACKEND": (("replicate", "modal", "http"), "replicate", "GEOMETRY_HTTP_URL", False),
    "MOGE_BACKEND": (("modal", "replicate", "http"), "modal", "MOGE_HTTP_URL", False),
}
INTS = ("PANOPTES_SERVICE_MAX_QUEUE", "PANOPTES_SERVICE_TIMEOUT_S", "PANOPTES_SERVICE_RETRIES", "EHS_SAM_CONCURRENCY")
FLAGS = ("PANOPTES_FAKE_MODEL", "HF_HUB_OFFLINE", "PANOPTES_ONPREM")
PATHS = ("PANOPTES_SERVING", "PANOPTES_WORKCELL", "PANOPTES_PLATFORM", "WEIGHTS", "SWAP_SCRATCH", "PANOPTES_RUNS", "PANOPTES_PAGES", "PY", "PG",
         "PGDATA_DIR", "PANOPTES_PUBLICATION_CATALOG", "PANOPTES_PUBLICATION_HTTP", "PANOPTES_WEB_ROOT", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
URLS = ("PANOPTES_PUBLIC_BASE_URL", "AWS_ENDPOINT_URL", "PANOPTES_CONTRACT_URL")
DB_SCHEMES = ("postgresql", "postgres", "mongodb", "mongodb+srv")
PLACEHOLDER = re.compile(r"change-me|example|<[^>]+>", re.I)
LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines; `export` prefix, single/double quotes and ` # comments` after unquoted values are accepted."""
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE.match(line)
        if not m:
            continue
        key, value = m.group(1), m.group(2).strip()
        if value[:1] in ("'", '"') and value[-1:] == value[:1] and len(value) >= 2:
            value = value[1:-1]
        else:
            value = re.split(r"\s+#", value, 1)[0].strip()
        out[key] = value
    return out


def template_keys(text: str) -> set[str]:
    """Every key the template documents, set or commented out (`# KEY=`)."""
    return {m.group(1) for m in re.finditer(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, re.M)}


def healthz_root(url: str, is_root: bool) -> str:
    """v1 roots are used as given (a nginx prefix such as /sam3d stays); v0 endpoint URLs lose their last path segment (/sam3)."""
    url = url.strip().rstrip("/")
    return url if is_root else url.rsplit("/", 1)[0]


def check(env: dict[str, str], known: set[str], *, live: bool = False) -> tuple[list[str], list[str]]:
    errors, warnings = [], []

    def url_ok(key, value):
        p = urlsplit(value)
        if p.scheme not in ("http", "https") or not p.netloc:
            errors.append(f"{key}: not an http(s) URL: {value!r}")
            return False
        return True

    for key in REQUIRED:
        if not env.get(key):
            errors.append(f"{key}: required, missing or empty")
    if env.get("HF_TOKEN"):
        errors.append("HF_TOKEN: never store it in .env (export it in the shell for the one-time weights fetch, then unset)")
    for key, value in env.items():
        if key not in known:
            warnings.append(f"{key}: not in env.template (there is only one configuration surface: document it there or drop it)")
        if value and PLACEHOLDER.search(value):
            warnings.append(f"{key}: placeholder value {value!r}")
    live_targets = []
    for key, (allowed, default, url_key, is_list) in BACKENDS.items():
        value = env.get(key)
        if value is None:
            if default != "http":
                warnings.append(f"{key}: unset, code default is the cloud backend {default!r}; set http for on-prem")
            continue
        if value.strip().lower() not in allowed:
            errors.append(f"{key}: {value!r} is not one of {', '.join(allowed)}")
            continue
        if value.strip().lower() == "http":
            urls = env.get(url_key, "")
            parts = [u.strip() for u in urls.split(",") if u.strip()] if is_list else [urls.strip()]
            if not parts or not parts[0]:
                errors.append(f"{url_key}: required when {key}=http")
                continue
            for u in parts:
                if url_ok(url_key, u):
                    live_targets.append((url_key, healthz_root(u, is_list)))
    for key in INTS:
        if env.get(key) and not env[key].isdigit():
            errors.append(f"{key}: must be a non-negative integer, got {env[key]!r}")
    for key in FLAGS:
        if env.get(key) and env[key] not in ("0", "1"):
            errors.append(f"{key}: must be 0 or 1, got {env[key]!r}")
    db = env.get("PANOPTES_DATABASE_URL", "")
    if db:
        p = urlsplit(db)
        if p.scheme not in DB_SCHEMES:
            errors.append(f"PANOPTES_DATABASE_URL: scheme must be one of {', '.join(DB_SCHEMES)}, got {db!r}")
        elif not (p.netloc or "host=" in p.query):
            errors.append("PANOPTES_DATABASE_URL: no host (user:pass@host:port/db or ?host=/socket/dir)")
        if p.scheme.startswith("mongodb") and not env.get("PANOPTES_MONGO_PREFIX"):
            warnings.append("PANOPTES_MONGO_PREFIX: unset with a mongodb:// database; default panoptes_ applies")
    blob = env.get("PANOPTES_BLOB_ROOT", "")
    if blob:
        if blob.startswith("s3://"):
            p = urlsplit(blob)
            if not p.netloc:
                errors.append(f"PANOPTES_BLOB_ROOT: s3://bucket/prefix needs a bucket, got {blob!r}")
            if not (env.get("AWS_ACCESS_KEY_ID") and env.get("AWS_SECRET_ACCESS_KEY")):
                warnings.append("PANOPTES_BLOB_ROOT is s3:// but AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are unset (fine only with an instance role)")
        elif not blob.startswith("/"):
            errors.append(f"PANOPTES_BLOB_ROOT: an absolute directory or s3://bucket/prefix, got {blob!r}")
        elif not Path(blob).is_dir():
            warnings.append(f"PANOPTES_BLOB_ROOT: directory does not exist: {blob}")
    bucket = env.get("PANOPTES_RESULT_BUCKET", "")
    if bucket and (not bucket.startswith("s3://") or not urlsplit(bucket).netloc):
        errors.append(f"PANOPTES_RESULT_BUCKET: s3://bucket/prefix, got {bucket!r}")
    for key in URLS:
        if env.get(key) and url_ok(key, env[key]) and env[key].endswith("/"):
            warnings.append(f"{key}: drop the trailing slash")
    for key in PATHS:
        if env.get(key) and not Path(env[key]).exists():
            warnings.append(f"{key}: path does not exist: {env[key]}")
    if not env.get("PANOPTES_PUBLIC_BASE_URL"):
        warnings.append("PANOPTES_PUBLIC_BASE_URL: unset; `panoptes run` cannot print the report address")
    if live:
        key_header = {"X-API-Key": env.get("PANOPTES_SERVICE_API_KEY", "")}
        for key, root in live_targets:
            try:
                with urllib.request.urlopen(urllib.request.Request(root + "/healthz", headers=key_header), timeout=10) as r:
                    body = json.loads(r.read().decode() or "{}")
                print(f"live {key} {root}/healthz -> {r.status} ok={body.get('ok')} model={body.get('model')} gpu={body.get('gpu')} "
                      f"queue_depth={body.get('queue_depth')} vram_free_mb={body.get('vram_free_mb')}")
            except urllib.error.HTTPError as e:
                errors.append(f"{key}: {root}/healthz -> HTTP {e.code} ({'wrong PANOPTES_SERVICE_API_KEY' if e.code == 401 else 'service not ready'})")
            except Exception as e:  # connection refused, DNS, TLS, timeout
                errors.append(f"{key}: {root}/healthz unreachable: {e}")
    return errors, warnings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("env_file", nargs="?", default=".env")
    ap.add_argument("--template", default=str(ROOT / "env.template"))
    ap.add_argument("--strict", action="store_true", help="warnings are errors")
    ap.add_argument("--live", action="store_true", help="GET /healthz on every http service root")
    a = ap.parse_args(argv)
    env_path = Path(a.env_file)
    if not env_path.is_file():
        print(f"check_env: {env_path} not found (make env)", file=sys.stderr)
        return 1
    env = parse_env(env_path.read_text())
    errors, warnings = check(env, template_keys(Path(a.template).read_text()), live=a.live)
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    failed = bool(errors) or (a.strict and bool(warnings))
    print(f"check_env: {env_path}: {len(env)} keys, {len(errors)} errors, {len(warnings)} warnings{' (strict)' if a.strict else ''} -> {'FAIL' if failed else 'OK'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
