"""Freeze the complete local identity input inventory using a read-only DB snapshot.

No credentials, conversations, model calls, or production writes enter the output.
Run with PANOPTES_DATABASE_URL / PANOPTES_BLOB_ROOT configured privately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_ref(path):
    path = Path(path).resolve()
    if not path.is_file():
        return {"path": str(path), "available": False}
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": checksum, "sizeBytes": path.stat().st_size, "available": True}


def scan(output, catalog, run_roots=()):
    output, catalog = Path(output).resolve(), Path(catalog).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "documents").mkdir()

    def freeze(value):
        raw = encoded(value)
        path = output / "documents" / (sha(raw) + ".json")
        if not path.exists():
            path.write_bytes(raw)
        return {"path": str(path), "sha256": sha(raw), "sizeBytes": len(raw)}

    queries = {
        "projects": "SELECT id,title,default_branch_id,fork_source_revision_id,created_at FROM projects ORDER BY id",
        "branches": "SELECT id,project_id,kind,title,source_revision_id,head_revision_id,created_at FROM scene_branches ORDER BY id",
        "revisions": "SELECT id,project_id,branch_id,parent_revision_id,source_revision_id,document,document_sha256,label,created_at FROM scene_revisions ORDER BY id",
        "publications": "SELECT id,project_id,scene_revision_id,title,evaluation_ids,review_ids,snapshot,created_at FROM publications ORDER BY id",
        "assets": "SELECT id,project_id,storage_key,sha256,size_bytes,media_type,metadata FROM assets ORDER BY id",
        "captures": "SELECT id,project_id,branch_id,base_revision_id,revision_id,target,images,task,created_at FROM captures ORDER BY id",
    }
    with psycopg.connect(os.environ["PANOPTES_DATABASE_URL"], row_factory=dict_row) as connection:
        connection.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        tables = {name: json.loads(encoded(connection.execute(sql).fetchall())) for name, sql in queries.items()}
    manifest = {"schemaVersion": 1, "createdAt": datetime.now(timezone.utc).isoformat(),
                "databaseReadOnly": True, "newModelCalls": 0, "counts": {k: len(v) for k, v in tables.items()},
                "projects": tables["projects"], "branches": tables["branches"], "revisions": [], "publications": [],
                "assets": [], "captures": [], "catalog": [], "evaluationRevisions": [], "runInventory": []}
    cache = {}
    blob_root = Path(os.environ.get("PANOPTES_BLOB_ROOT", ".platform/blobs")).resolve()
    assets = {}
    for row in tables["assets"]:
        key = row.pop("storage_key")
        path = (blob_root / key).resolve()
        if not path.is_relative_to(blob_root):
            raise ValueError("Asset storage path escaped blob root")
        actual = cache.setdefault(str(path), None)
        if actual is None:
            actual = cache[str(path)] = file_ref(path)
        row["file"] = actual
        row["integrityVerified"] = actual.get("sha256") == row["sha256"].strip() and actual.get("sizeBytes") == row["size_bytes"]
        assets[row["id"]] = row
        manifest["assets"].append(row)

    def document_summary(doc):
        visible = [e for e in doc["entities"] if not e.get("sourceContext")]
        image_ids = {c["imageId"] for c in doc["cameras"]}
        observed_images = {o["imageId"] for o in doc["observations"]}
        owners = {o["id"]: [e["id"] for e in doc["entities"] if o["id"] in e.get("observationRefs", [])] for o in doc["observations"]}
        geometry = doc.get("geometryEvidence", {})
        geometry_images = {c["imageId"] for c in doc["cameras"] if c["id"] in {f["cameraId"] for f in geometry.get("frames", [])}}
        return {"schemaVersion": doc["schemaVersion"], "captureId": doc.get("captureId"),
                "entities": len(doc["entities"]), "businessEntities": len(visible), "observations": len(doc["observations"]),
                "multiImageEntities": sum(len({o["imageId"] for o in doc["observations"] if o["id"] in e.get("observationRefs", [])}) > 1 for e in visible),
                "confirmedBusinessEntities": sum(e["associationState"] == "confirmed" for e in visible),
                "sourceImageShas": sorted({assets[a]["sha256"].strip() for a in image_ids | observed_images if a in assets}),
                "geometryEvidence": geometry, "coordinateFrames": doc["coordinateFrames"],
                "observationManifest": [{"id": o["id"], "revision": o.get("revision"), "sha256": sha(encoded(o)),
                    "imageId": o["imageId"], "maskAssetId": o.get("maskAssetId"), "entityIds": owners[o["id"]],
                    "pixelMapping": o.get("pixelMapping"), "nativeGeometryAvailable": o["imageId"] in geometry_images,
                    "missingEvidence": o.get("missingEvidence", [])} for o in doc["observations"]],
                "assetManifest": [{"id": a["id"], "sha256": a.get("sha256"), "sizeBytes": a.get("sizeBytes"),
                    "available": assets.get(a["id"], {}).get("integrityVerified", False)} for a in doc["assets"]]}

    revision_index = {}
    for row in tables["revisions"]:
        doc = row.pop("document")
        frozen = freeze(doc)
        if frozen["sha256"] != row["document_sha256"].strip():
            raise ValueError("Stored scene document hash mismatch: " + row["id"])
        row.update(document=frozen, summary=document_summary(doc))
        revision_index[row["id"]] = row
        manifest["revisions"].append(row)
    for row in tables["publications"]:
        row["snapshot"] = freeze(row["snapshot"])
        manifest["publications"].append(row)
    for row in tables["captures"]:
        row["document"] = freeze({"images": row.pop("images"), "task": row.pop("task")})
        manifest["captures"].append(row)
    for path in sorted(catalog.glob("*/bundle.json")):
        bundle = json.loads(path.read_bytes())
        publication = bundle["responses"]["/api/publications/" + bundle["publicationId"]]
        checks = [file_ref(path.parent / "blobs" / a["sha256"]) for a in publication["snapshot"]["assetManifest"]]
        for expected, actual in zip(publication["snapshot"]["assetManifest"], checks):
            if actual.get("sha256") != expected["sha256"] or actual.get("sizeBytes") != expected["sizeBytes"]:
                raise ValueError("Catalog asset integrity mismatch")
        database_publication = next((p for p in tables["publications"] if p["id"] == publication["id"]), None)
        if database_publication and database_publication["snapshot"]["sha256"] != sha(encoded(publication["snapshot"])):
            raise ValueError("Catalog and database publication diverge")
        manifest["catalog"].append({"publicationId": publication["id"], "revisionId": publication["sceneRevisionId"],
            "projectId": publication["projectId"], "bundle": file_ref(path), "snapshot": freeze(publication["snapshot"]), "assets": checks})
        rid = publication["sceneRevisionId"]
        if rid not in revision_index:
            doc = publication["snapshot"]["revision"]["document"]
            row = {"id": rid, "project_id": publication["projectId"], "document": freeze(doc), "summary": document_summary(doc)}
            revision_index[rid] = row
            manifest["revisions"].append(row)
    selected = {b["head_revision_id"] for b in tables["branches"]} | {p["scene_revision_id"] for p in tables["publications"]} | {p["revisionId"] for p in manifest["catalog"]}
    for rid in sorted(selected):
        row = revision_index[rid]
        manifest["evaluationRevisions"].append({"revisionId": rid, "projectId": row["project_id"], "document": row["document"],
            "branchIds": [b["id"] for b in tables["branches"] if b["head_revision_id"] == rid],
            "publicationIds": [p["id"] for p in tables["publications"] if p["scene_revision_id"] == rid],
            "catalogPublicationIds": [p["publicationId"] for p in manifest["catalog"] if p["revisionId"] == rid],
            "observations": row["summary"]["observations"], "nativeGeometryFrames": len(row["summary"]["geometryEvidence"].get("frames", []))})
    for root in run_roots:
        for run in sorted(Path(root).iterdir()):
            if not run.is_dir() or run.name.startswith((".", "_")):
                continue
            # ponytail: inventory only documented source artifacts, not arbitrary credentials or caches.
            paths = [p for pattern in ("manifest.json", "scene.json", "input/*.jpg", "input/*.png", "frames/*.jpg", "frames/*.png", "geometry/frames/*/*.npy", "geometry/frames/*/meta.json") for p in run.glob(pattern)]
            if paths:
                manifest["runInventory"].append({"path": str(run.resolve()), "files": [file_ref(p) for p in sorted(set(paths))]})
    manifest["counts"].update(catalogPublications=len(manifest["catalog"]), evaluationRevisions=len(selected),
        invalidAssets=sum(not a["integrityVerified"] for a in manifest["assets"]))
    (output / "manifest.json").write_bytes(encoded(manifest) + b"\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path(".platform/publication-catalog"))
    parser.add_argument("--run-root", action="append", type=Path, default=[])
    args = parser.parse_args()
    try:
        result = scan(args.output, args.catalog, args.run_root)
    except psycopg.Error as exc:
        raise SystemExit("Database snapshot failed: " + type(exc).__name__) from None
    print(json.dumps({"counts": result["counts"], "manifest": str(args.output.resolve() / "manifest.json")}))
