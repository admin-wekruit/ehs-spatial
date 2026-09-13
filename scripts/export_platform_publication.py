#!/usr/bin/env python3
"""Export one publication through public GET endpoints, without provider credentials.

Run: python scripts/export_platform_publication.py --api http://127.0.0.1:8792 \
    --publication PUBLICATION_ID --output .platform/publication-site
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit
from urllib.request import build_opener, HTTPRedirectHandler
from uuid import UUID


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("The public API must serve content directly; redirects are not exported")


def export_publication(api: str, publication_id: str, output: Path):
    parsed = urlsplit(api)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("api must be an HTTP(S) origin or base path without credentials")
    publication_id = str(UUID(publication_id))
    api, output = api.rstrip("/"), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
    opener = build_opener(NoRedirect)

    def get(path):
        with opener.open(api + path, timeout=90) as response:
            return json.load(response)

    publication = get("/api/publications/" + publication_id)
    if publication["id"] != publication_id:
        raise ValueError("publication identity mismatch")
    snapshot, project_id = publication["snapshot"], publication["projectId"]
    if snapshot.get("assetManifest") is None:
        raise ValueError("Publication must contain a frozen asset manifest")
    revision = snapshot["revision"]
    if revision["id"] != publication["sceneRevisionId"] or revision["projectId"] != project_id:
        raise ValueError("publication revision mismatch")
    summaries = [item for item in get("/api/publications")["items"] if item["id"] == publication_id]
    if len(summaries) != 1:
        raise ValueError("publication summary not found")
    detail = get("/api/projects/" + project_id)
    if detail["project"]["id"] != project_id:
        raise ValueError("project identity mismatch")
    revisions = [revision]
    if snapshot.get("reconstructionRevision"):
        revisions.append(snapshot["reconstructionRevision"])
    branches = []
    for frozen in revisions:
        if frozen["projectId"] != project_id:
            raise ValueError("foreign revision in publication")
        branch = deepcopy(next(item for item in detail["branches"] if item["id"] == frozen["branchId"]))
        # The project's public view is the publication's frozen branch/revision projection.
        branch["headRevisionId"] = frozen["id"]
        if frozen is revision:
            branch["kind"] = snapshot.get("branchKind") or branch["kind"]
            branch["title"] = snapshot.get("branchTitle") or branch["title"]
        branches.append(branch)
    project = {**detail["project"], "defaultBranchId": revision["branchId"]}
    responses = {
        "/api/publications": {"items": summaries},
        "/api/publications/" + publication_id: publication,
        "/api/projects": {"items": [project]},
        "/api/projects/" + project_id: {"project": project, "branch": branches[0], "branches": branches, "revision": revision},
    }
    for frozen in revisions:
        responses["/api/revisions/" + frozen["id"]] = frozen
    for route, field in (("jobs", "jobs"), ("edits", "editBatches"), ("evaluations", "evaluations"), ("reviews", "reviews")):
        responses[f"/api/projects/{project_id}/{route}"] = {"items": snapshot.get(field) or []}
    responses[f"/api/projects/{project_id}/revisions"] = {"items": revisions}
    for job in snapshot.get("jobs", []):
        responses["/api/jobs/" + job["id"]] = job
    policies = [item for item in get("/api/policies")["items"] if item["projectId"] == project_id]
    responses["/api/policies"] = {"items": policies}
    for policy in policies:
        value = get("/api/policies/" + policy["id"])
        if value["policy"]["projectId"] != project_id:
            raise ValueError("foreign policy in publication export")
        responses["/api/policies/" + policy["id"]] = value

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.name + "-", dir=output.parent) as temporary:
        staging = Path(temporary) / "bundle"
        (staging / "blobs").mkdir(parents=True)
        assets, total = {}, 0
        for index, entry in enumerate(snapshot["assetManifest"], 1):
            asset_id, sha = str(UUID(entry["assetId"])), entry["sha256"]
            if asset_id in assets or not re.fullmatch(r"[0-9a-f]{64}", sha) or type(entry["sizeBytes"]) is not int or entry["sizeBytes"] < 0:
                raise ValueError("invalid frozen asset manifest")
            record = get("/api/assets/" + asset_id)
            if record["id"] != asset_id or record["projectId"] != project_id or any(record[key] != entry[key] for key in ("sha256", "sizeBytes", "mediaType")):
                raise ValueError(f"Frozen asset metadata mismatch: {asset_id}")
            record["url"] = f"/api/assets/{asset_id}/content"
            destination = staging / "blobs" / sha
            digest, size = hashlib.sha256(), 0
            with opener.open(api + record["url"], timeout=90) as response, destination.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if digest.hexdigest() != sha or size != entry["sizeBytes"]:
                raise ValueError(f"Frozen asset byte integrity mismatch: {asset_id}")
            assets[asset_id] = record
            responses["/api/assets/" + asset_id] = record
            total += size
            if index % 20 == 0 or index == len(snapshot["assetManifest"]):
                print(f"Verified {index}/{len(snapshot['assetManifest'])} assets, {total:,} bytes", flush=True)
        responses[f"/api/projects/{project_id}/assets"] = {"items": [{key: value for key, value in record.items() if key != "url"} for record in assets.values()]}
        bundle = {"schemaVersion": 1, "publicationId": publication_id, "projectId": project_id,
                  "exportedAt": datetime.now(timezone.utc).isoformat(), "responses": responses}
        (staging / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
        staging.rename(output)
    return {"publicationId": publication_id, "assetCount": len(assets), "sizeBytes": total, "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", required=True)
    parser.add_argument("--publication", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_publication(args.api, args.publication, args.output), indent=2))


if __name__ == "__main__":
    main()
