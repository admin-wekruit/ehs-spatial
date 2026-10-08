"""Real PostgreSQL checks for atomic, revision-bound internal continuations."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from uuid import UUID, uuid5

import psycopg
import pytest

from argus.platform.contracts import PlatformError, canonical, digest
from argus.platform.storage import LocalBlobStore
from test_platform_backend import edit_body, entity, identity, make_job, project, repo


def continuation(**overrides):
    return {"kind": "reconstruct_scene", "inputs": {"entityIds": ["next-object"]},
            "config": {"pipeline": {"step": "review"}}, **overrides}


def children(repo, parent):
    with repo._connect() as connection:
        return connection.execute("SELECT * FROM jobs WHERE project_id=%s AND id<>%s",
                                  (parent["projectId"], parent["id"])).fetchall()


def test_finish_enqueues_one_revision_bound_child_with_frozen_config(repo):
    cap, scene = project(repo)
    repo.execution_config = {"providerManifest": {"pin": "frozen"}, "serverFlag": True}
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    repo.execution_config = {"providerManifest": {"pin": "new-server-value"}}
    command = continuation()
    finished = repo.finish_job(parent["id"], parent["attemptToken"], "incomplete",
                               document=scene["revision"]["document"], result={"saved": True}, continuation=command)
    child = repo.get_job(finished["result"]["continuationJobId"])
    assert child["id"] == str(uuid5(UUID(parent["id"]), "continuation"))
    assert child["requestId"] == child["id"] and child["status"] == "pending_dispatch"
    assert child["baseRevisionId"] == finished["resultRevisionId"] != parent["baseRevisionId"]
    assert child["projectId"] == parent["projectId"] and child["branchId"] == parent["branchId"]
    assert child["config"] == {**parent["config"], **command["config"]}
    assert child["inputs"] == command["inputs"] and finished["headAdvanced"]
    assert finished["result"]["saved"] is True


def test_finish_without_document_uses_parent_base_and_does_not_invent_revision(repo):
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    finished = repo.finish_job(parent["id"], parent["attemptToken"], "succeeded", continuation=continuation())
    child = repo.get_job(finished["result"]["continuationJobId"])
    assert child["baseRevisionId"] == parent["baseRevisionId"]
    assert finished["resultRevisionId"] is None and finished["headAdvanced"] is False


def test_two_sequential_model_revisions_preserve_the_first_object(repo):
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    first_document = deepcopy(scene["revision"]["document"])
    first_document["entities"].append(entity())
    first = repo.finish_job(parent["id"], parent["attemptToken"], "succeeded",
                            document=first_document, continuation=continuation())
    child = repo.claim_job(first["result"]["continuationJobId"])
    second_document = repo.get_revision(child["baseRevisionId"])["document"]
    second_document["entities"].append(entity())
    second = repo.finish_job(child["id"], child["attemptToken"], "succeeded", document=second_document)
    current = repo.get_project(parent["projectId"])["revision"]
    assert current["id"] == second["resultRevisionId"]
    assert [e["id"] for e in current["document"]["entities"]] == [e["id"] for e in second_document["entities"]]
    assert len(current["document"]["entities"]) == 2
    assert repo.get_revision(first["resultRevisionId"])["document"] == first_document


def test_failed_child_insert_rolls_back_parent_revision_and_branch_head(repo):
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    with repo._connect() as connection:
        connection.execute("""CREATE FUNCTION reject_pipeline_child() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN IF NEW.kind='reconstruct_scene' THEN RAISE EXCEPTION 'fixture_child_rejected'; END IF;
            RETURN NEW; END $$""")
        connection.execute("CREATE TRIGGER reject_pipeline_child BEFORE INSERT ON jobs FOR EACH ROW EXECUTE FUNCTION reject_pipeline_child()")
    with pytest.raises(psycopg.errors.RaiseException, match="fixture_child_rejected"):
        repo.finish_job(parent["id"], parent["attemptToken"], "succeeded",
                        document=scene["revision"]["document"], continuation=continuation())
    assert repo.get_job(parent["id"])["status"] == "running"
    assert repo.get_job(parent["id"])["resultRevisionId"] is None
    assert repo.get_project(parent["projectId"])["revision"]["id"] == parent["baseRevisionId"]
    assert not children(repo, parent)
    with repo._connect() as connection:
        assert connection.execute("SELECT count(*) AS n FROM scene_revisions").fetchone()["n"] == 1
        connection.execute("DROP TRIGGER reject_pipeline_child ON jobs")
    finished = repo.finish_job(parent["id"], parent["attemptToken"], "succeeded",
                               document=scene["revision"]["document"], continuation=continuation())
    assert finished["result"]["continuationJobId"] and len(children(repo, parent)) == 1


def test_concurrent_duplicate_finish_never_inserts_a_second_child(repo):
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    def finish(_):
        return repo.finish_job(parent["id"], parent["attemptToken"], "succeeded",
                               document=scene["revision"]["document"], result={}, continuation=continuation())
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(finish, range(2)))
    assert results[0]["result"]["continuationJobId"] == results[1]["result"]["continuationJobId"]
    assert len(children(repo, parent)) == 1


@pytest.mark.parametrize("has_document", [False, True])
def test_branch_change_stops_continuation_and_preserves_result_evidence(repo, has_document):
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    winner = repo.commit_edits(parent["projectId"], cap, edit_body(scene))
    finished = repo.finish_job(parent["id"], parent["attemptToken"], "succeeded", result={"evidence": "kept"},
        document=scene["revision"]["document"] if has_document else None, continuation=continuation())
    assert finished["result"] == {"evidence": "kept", "continuationStopped": "branch_changed"}
    assert finished["headAdvanced"] is False and bool(finished["resultRevisionId"]) == has_document
    assert repo.get_project(parent["projectId"])["revision"]["id"] == winner["revision"]["id"]
    assert not children(repo, parent)


@pytest.mark.parametrize("condition", ["cancelled", "old_attempt", "expired", "failed", "outcome_unknown"])
def test_cancelled_stale_and_unsuccessful_attempts_cannot_enqueue(repo, condition):
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    token, status = parent["attemptToken"], "succeeded"
    if condition == "cancelled":
        repo.cancel_job(parent["id"], cap)
    elif condition == "old_attempt":
        token = identity()
    elif condition == "expired":
        with repo._connect() as connection:
            connection.execute("UPDATE jobs SET lease_expires_at=now()-interval '1 second' WHERE id=%s", (parent["id"],))
    else:
        status = condition
    finished = repo.finish_job(parent["id"], token, status, document=scene["revision"]["document"], continuation=continuation())
    assert not children(repo, parent)
    assert not (finished.get("result") or {}).get("continuationJobId")
    assert repo.get_project(parent["projectId"])["revision"]["id"] == parent["baseRevisionId"]


@pytest.mark.parametrize("kind", ["validate_model", "reconstruct_scene"])
def test_clients_cannot_create_internal_pipeline_job_kinds(repo, kind):
    cap, scene = project(repo)
    with pytest.raises(PlatformError, match="admin_job_required"):
        repo.create_job(scene["project"]["id"], cap, {"requestId": identity(), "branchId": scene["branch"]["id"],
            "baseRevisionId": scene["revision"]["id"], "kind": kind, "inputs": {}, "config": {}})


def test_client_config_cannot_supply_pipeline_authority_even_without_server_config(repo):
    cap, scene = project(repo)
    repo.execution_config = {}
    authority = {"providerManifest": {"malicious": True}, "researchPreparation": {"authority": True},
                 "researchProtocolSha256": "x", "pipeline": {"next": "paid"}, "continuation": {},
                 "continuationJobId": identity(), "parentJobId": identity(), "pipelineRootJobId": identity()}
    job = repo.create_job(scene["project"]["id"], cap, {"requestId": identity(), "branchId": scene["branch"]["id"],
        "baseRevisionId": scene["revision"]["id"], "kind": "generate_object", "inputs": {},
        "config": {**authority, "seed": 7}})
    assert job["config"] == {"seed": 7}


@pytest.mark.parametrize("command", [[], {"kind": "generate_object"}, {"kind": []}, continuation(inputs=[]),
                                      continuation(config=[]), continuation(baseRevisionId="client-selected")])
def test_invalid_continuation_command_rejects_without_committing_parent(repo, command):
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    with pytest.raises(PlatformError, match="invalid_job_continuation"):
        repo.finish_job(parent["id"], parent["attemptToken"], "succeeded", continuation=command)
    assert repo.get_job(parent["id"])["status"] == "running" and not children(repo, parent)


def research_command(repo, parent, tmp_path, *, target_overrides=None, asset_kind="recgen_validation_input"):
    repo.blobs = LocalBlobStore(tmp_path)
    frozen = {key: parent[key] for key in ("projectId", "branchId", "baseRevisionId")}
    frozen.update(protocol={"fixture": "authority validation is exercised at the shared boundary"},
                  providerManifest={"generation": {"pins": {"model": "TRI-ML/RecGen"}}})
    frozen.update(target_overrides or {})
    blob = repo.blobs.put(canonical(frozen), "application/json")
    asset = repo.register_asset(parent["projectId"], {**blob, "metadata": {"kind": asset_kind}})
    command = continuation(kind="validate_model", inputs={"validationAssetId": asset["id"], "validationSha256": asset["sha256"]},
                           config={"researchProtocolSha256": digest(frozen["protocol"])})
    return command, frozen


def test_research_child_revalidates_authority_inside_parent_transaction(repo, tmp_path, monkeypatch):
    from argus.platform import research_authority
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    command, frozen = research_command(repo, parent, tmp_path)
    calls = []
    def validate(repository, connection, value, sha):
        assert repository is repo and value == frozen and sha == command["inputs"]["validationSha256"]
        assert connection.execute("SELECT status FROM jobs WHERE id=%s", (parent["id"],)).fetchone()["status"] == "running"
        calls.append(value)
        return {"source": "database_admin", "databaseRole": "fixture_live_authority"}
    monkeypatch.setattr(research_authority, "validate_prepared", validate)
    finished = repo.finish_job(parent["id"], parent["attemptToken"], "succeeded", continuation=command)
    child = repo.get_job(finished["result"]["continuationJobId"])
    assert calls == [frozen] and child["kind"] == "validate_model"
    assert child["config"]["submittedBy"]["databaseRole"] == "fixture_live_authority"
    assert child["baseRevisionId"] == frozen["baseRevisionId"]


@pytest.mark.parametrize("bad", ["projectId", "branchId", "baseRevisionId", "input_hash", "protocol_hash", "asset_kind", "missing_blobs"])
def test_research_child_checks_source_and_automatic_target_before_authority(repo, tmp_path, monkeypatch, bad):
    from argus.platform import research_authority
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    command, _ = research_command(repo, parent, tmp_path,
        target_overrides={bad: identity()} if bad in {"projectId", "branchId", "baseRevisionId"} else None,
        asset_kind="unrelated_asset" if bad == "asset_kind" else "recgen_validation_input")
    if bad == "input_hash":
        command["inputs"]["validationSha256"] = "0" * 64
    elif bad == "protocol_hash":
        command["config"]["researchProtocolSha256"] = "0" * 64
    elif bad == "missing_blobs":
        repo.blobs = None
    monkeypatch.setattr(research_authority, "validate_prepared", lambda *_args: pytest.fail("invalid envelope reached authority boundary"))
    with pytest.raises(PlatformError, match="research_input_hash_mismatch|blob_store_not_configured"):
        repo.finish_job(parent["id"], parent["attemptToken"], "succeeded", continuation=command)
    assert repo.get_job(parent["id"])["status"] == "running" and not children(repo, parent)


def test_research_budget_rejection_cannot_commit_parent_or_child(repo, tmp_path, monkeypatch):
    from argus.platform import research_authority
    cap, scene = project(repo)
    parent = repo.claim_job(make_job(repo, cap, scene)["id"])
    command, _ = research_command(repo, parent, tmp_path)
    def exhausted(*_args):
        raise PlatformError("paid_budget_exceeded", 409)
    monkeypatch.setattr(research_authority, "validate_prepared", exhausted)
    with pytest.raises(PlatformError, match="paid_budget_exceeded"):
        repo.finish_job(parent["id"], parent["attemptToken"], "succeeded", continuation=command)
    assert repo.get_job(parent["id"])["status"] == "running" and not children(repo, parent)
