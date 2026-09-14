"""Run the real reassociation entry point against frozen files and an isolated writer.

Run: python -m scripts.research.evaluate_object_identity --manifest PREPARED_MANIFEST --output NEW_DIR
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ehs_spatial.platform.storage import LocalBlobStore
from scripts.research.prepare_object_identity_inputs import checked
from scripts.research.scan_object_identity import encoded, file_ref


class FrozenBlobs:
    def __init__(self, assets, output):
        self.sources = {a["sha256"]: a["file"] for a in assets}
        self.output = LocalBlobStore(output)

    def get(self, key, checksum, size):
        if key != "sha256/" + checksum:
            raise ValueError("Unexpected blob key")
        if checksum not in self.sources:
            return self.output.get(key, checksum, size)
        raw = checked(self.sources[checksum])
        if len(raw) != size:
            raise ValueError("Frozen asset size mismatch")
        return raw

    def put(self, data, media_type):
        return self.output.put(data, media_type)


class FrozenRepository:
    """Only the methods needed by reassociation; registrations are memory-local."""
    def __init__(self, target, document, assets, captures):
        self.target, self.document = target, document
        self.assets = {a["id"]: {**a, "storageKey": "sha256/" + a["sha256"]} for a in assets}
        self.captures = [{**c, **json.loads(checked(c["document"]))} for c in captures if c["project_id"] == target["projectId"]]

    def get_revision(self, identity):
        if identity != self.target["revisionId"]:
            raise ValueError("Revision escaped frozen target")
        return {"id": identity, "projectId": self.target["projectId"], "document": deepcopy(self.document)}

    def get_asset(self, identity):
        return deepcopy(self.assets[identity])

    def list_project_records(self, project_id, kind):
        if project_id != self.target["projectId"] or kind not in ("assets", "captures"):
            raise ValueError("Operation escaped frozen project")
        return {"items": deepcopy(list(self.assets.values()) if kind == "assets" else self.captures)}

    def register_asset(self, project_id, blob, job_id):
        if project_id != self.target["projectId"]:
            raise ValueError("Asset escaped frozen project")
        identity = str(uuid5(NAMESPACE_URL, "identity-evaluation:" + job_id + ":" + blob["sha256"]))
        asset = {**blob, "id": identity, "projectId": project_id, "jobId": job_id}
        self.assets[identity] = asset
        return deepcopy(asset)


def quality_metrics(association, labels):
    groups = (association or {}).get("groups", [])
    membership = {oid: i for i, group in enumerate(groups) for oid in group}
    scored = [x for x in labels if x.get("truth") in ("same", "different") and x.get("reviewer") and len(x.get("observationIds", [])) == 2]
    counts = {"truePositive": 0, "falsePositive": 0, "falseNegative": 0, "trueNegative": 0}
    failures = []
    for label in scored:
        a, b = label["observationIds"]
        same = a in membership and b in membership and membership[a] == membership[b]
        actual = label["truth"] == "same"
        counts["truePositive" if same and actual else "falsePositive" if same else "falseNegative" if actual else "trueNegative"] += 1
        if same != actual:
            failures.append({"observationIds": [a, b], "expected": label["truth"], "actual": "same" if same else "unlinked"})
    tp, fp, fn = counts["truePositive"], counts["falsePositive"], counts["falseNegative"]
    return {"humanLabeledPairs": len(scored), "unknownLabels": len(labels) - len(scored),
        "counts": counts if scored else None, "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None, "groupAccuracy": None, "failures": failures}


def evaluate(manifest_path, output):
    from ehs_spatial.platform.contracts import PlatformError, validate_document
    from ehs_spatial.platform.reconstruction import run_reassociation
    from scripts.research.reprocess_object_identity import assert_source_conserved
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_bytes())
    output.mkdir(parents=True, exist_ok=False)
    blobs = FrozenBlobs(manifest["assets"], output / "blobs")
    results = []
    for target in manifest["evaluationRevisions"]:
        source = json.loads(checked(target["document"]))
        repository = FrozenRepository(target, source, manifest["assets"], manifest["captures"])
        job = {"id": str(uuid5(NAMESPACE_URL, "offline-identity:" + target["revisionId"])), "projectId": target["projectId"],
            "baseRevisionId": target["revisionId"], "inputs": {}, "config": {"associationConfigVersion": "workcell-identity-v2"}}
        try:
            document, result = run_reassociation(repository, blobs, job, {})
        except PlatformError as exc:
            document = deepcopy(source)
            result = {"status": "not_comparable", "newModelCalls": 0,
                "errors": [{"stage": "migration", "code": exc.code, "params": exc.params}],
                "evaluatedObservationCount": 0, "observationCount": len(source["observations"])}
        validate_document(document)
        assert_source_conserved(source, document)
        assert result["newModelCalls"] == 0
        path = output / (target["revisionId"] + ".json")
        path.write_bytes(encoded(document))
        results.append({"revisionId": target["revisionId"], "projectId": target["projectId"],
            "publicationIds": target["publicationIds"], "catalogPublicationIds": target["catalogPublicationIds"],
            "sourceDocument": target["document"], "resultDocument": file_ref(path), "result": result,
            "validation": {"validDocument": True, "sourceObservationsMasksCamerasAssetsConserved": True},
            "comparability": "native_geometry_available" if source.get("geometryEvidence") and result["evaluatedObservationCount"] else "not_comparable",
            "quality": quality_metrics(result.get("association"), manifest.get("pairLabels", {}).get(target["revisionId"], []))})
        print(json.dumps({"revisionId": target["revisionId"], "status": result["status"], "evaluatedObservations": result["evaluatedObservationCount"]}), flush=True)
    report = {"schemaVersion": 1, "inputManifest": file_ref(manifest_path), "results": results, "newModelCalls": 0,
        "productionWrites": 0, "evaluationRevisionCount": len(results), "physicalIdentityGate": "unverified_without_independent_labels"}
    root = Path(__file__).resolve().parents[2]
    report["codePins"] = [file_ref(root / name) for name in ("ehs_spatial/platform/contracts.py", "ehs_spatial/platform/reconstruction.py", "ehs_spatial/platform/identity.py", "ehs_spatial/platform/spatial.py", "scripts/research/evaluate_object_identity.py")]
    (output / "results.json").write_bytes(encoded(report) + b"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.output)
