"""Read-only report projection; full edit payloads remain independently addressable."""


def edit_summary(edit):
    return {key: edit[key] for key in ("id", "createdAt", "baseRevisionId", "revisionId")} | {
        "operationTypes": [operation["type"] for operation in edit["operations"]]}


def publication_view(publication, detail):
    snapshot = publication["snapshot"]
    return {
        "publication": {**publication, "snapshot": {**snapshot, "editBatches": None}},
        "project": detail["project"], "branch": detail["branch"], "branches": detail["branches"],
        "edits": [edit_summary(edit) for edit in snapshot.get("editBatches") or []],
    }
