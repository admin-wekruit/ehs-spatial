"""Offline checks for installed platform paths and explicit publication configuration."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import types
import pytest

from argus import ROOT
from argus.platform.config import PlatformConfig
from argus.platform.import_public_scene import converter_identity
from argus.platform.executor import LocalJobExecutor


def test_platform_uses_package_sources_and_data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PANOPTES_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PANOPTES_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.delenv("PANOPTES_BLOB_ROOT", raising=False)
    assert Path(PlatformConfig.from_env().blob_root) == tmp_path / "blobs"
    assert converter_identity()["sourceCadConverterSha256"] == hashlib.sha256((ROOT / "argus/platform/source_cad.py").read_bytes()).hexdigest()


def test_publisher_never_loads_credentials_from_checkout(tmp_path):
    hidden = tmp_path / ".platform/identity-runtime-env.json"
    hidden.parent.mkdir()
    hidden.write_text('{"PANOPTES_DATABASE_URL":"ignored-file-value"}')
    env = {key: value for key, value in os.environ.items() if key != "PANOPTES_DATABASE_URL"}
    env.update(PYTHONPATH=str(ROOT), PANOPTES_DATA_ROOT=str(tmp_path / "data"))
    result = subprocess.run([sys.executable, "-m", "argus.platform.publish_capture", "090-v2-mvs-fill-sam3d", "run", "scale", "title"],
                            cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode != 0 and "PANOPTES_DATABASE_URL is not set" in result.stderr
    assert "ignored-file-value" not in result.stdout + result.stderr


@pytest.mark.parametrize("cell", ["090", "030"])
@pytest.mark.parametrize("missing", [None, "comparisons", "evidence", "generation", "mesh", "incomplete"])
def test_publisher_writes_variant_result_and_catalog_under_data_root(tmp_path, monkeypatch, cell, missing):
    from argus.platform import api, export_platform_publication, import_public_scene, policy_repository, policy_service, runtime
    from fastapi import testclient

    config = json.loads((ROOT / "argus/pipeline/cells" / f"{cell}.json").read_text())
    variant = config["variant"]
    objects = config["objects"]
    run = tmp_path / "run"
    (run / "public").mkdir(parents=True)
    scene = run / "public/scene.json"
    scene.write_text("{}")
    for name in ("result/comparisons.json", "evidence/objects.json"):
        path = run / name
        path.parent.mkdir()
        rows = objects[:-1] if missing == ("comparisons" if name.startswith("result/") else "evidence") else objects
        path.write_text(json.dumps({"objects": [{"object_id": oid} for oid in rows]}))
    for oid in objects:
        folder = run / "generation" / oid
        folder.mkdir(parents=True)
        if missing == "generation" and oid == objects[-1]:
            continue
        folder.joinpath("output.json").write_text(json.dumps({"object_id": oid,
            "status": "failed" if missing == "incomplete" and oid == objects[-1] else "complete",
            "paths": {"mesh": "object.ply", "posed_mesh": "posed-object.ply"}}))
        for name in ("object.ply", "posed-object.ply"):
            if not (missing == "mesh" and oid == objects[-1] and name == "posed-object.ply"):
                folder.joinpath(name).write_bytes(b"fixture")
    scale = tmp_path / "scale.json"
    scale.write_text(json.dumps({"nativeToMeters": .25, "method": "fixture", "limit": .05,
                                "maxDeviation": .01, "passed": True, "features": []}))
    data = tmp_path / "data"
    monkeypatch.setenv("PANOPTES_DATA_ROOT", str(data))
    monkeypatch.setenv("PANOPTES_DATABASE_URL", "postgresql://localhost/test")
    monkeypatch.setattr(sys, "argv", ["publish_capture", variant, str(run), str(scale), "fixture"])
    repo = types.SimpleNamespace(migrate=lambda: None,
        get_revision=lambda revision: {"document": {"coordinateFrames": [{"id": "frame"}], "cameras": [{"imageId": "image"}]}},
        commit_edits=lambda project, cap, edit: {"revision": {"id": "scaled", "document": {"schemaVersion": 2,
            "coordinateFrames": [{"scale": edit["operations"][1]["scale"]}]}}},
        create_publication=lambda *args: {"id": "publication"})
    calls = []
    monkeypatch.setattr(runtime, "services", lambda config: calls.append("services") or (repo, None))

    def imported(source, repository, blobs, output, **kwargs):
        calls.append("import")
        assert output == data / "publications/imports" and kwargs["geometry_root"] == run
        output.mkdir(parents=True)
        (output / (hashlib.sha256(source.read_bytes()).hexdigest() + ".management.json")).write_text('{"capability":"fixture"}')
        return {"projectId": "project", "branchId": "branch", "sceneRevisionId": "native"}

    monkeypatch.setattr(import_public_scene, "run_import", imported)
    monkeypatch.setattr(policy_repository, "PostgresPolicyRepository", lambda repo: None)
    monkeypatch.setattr(policy_service, "PolicyService", lambda *args: None)
    monkeypatch.setattr(api, "create_app", lambda **kwargs: None)
    response = types.SimpleNamespace(content=b'{"publicationId":"publication"}', raise_for_status=lambda: None)
    monkeypatch.setattr(testclient, "TestClient", lambda app: types.SimpleNamespace(get=lambda path: response))
    monkeypatch.setattr(export_platform_publication, "build_opener", export_platform_publication.build_opener)
    monkeypatch.setattr(export_platform_publication, "export_publication", lambda api, pid, output: output.mkdir(parents=True))
    if missing:
        with pytest.raises(RuntimeError, match=objects[-1]):
            runpy.run_module("argus.platform.publish_capture", run_name="__main__")
        assert calls == []
        return
    runpy.run_module("argus.platform.publish_capture", run_name="__main__")
    assert calls == ["services", "import"]
    result = json.loads((data / "publications" / variant / "result.json").read_text())
    assert result["publicationId"] == "publication" and result["nativeToMeters"] == .25
    assert Path(result["export"]) == data / "publications/catalog/publication"
    assert json.loads(Path(result["view"]).read_text())["publicationId"] == "publication"


def test_modal_publication_mounts_package_and_data(tmp_path, monkeypatch):
    calls = []

    class Image:
        def __getattr__(self, name):
            def call(*args, **kwargs):
                calls.append((name, args))
                return self
            return call

    modal = types.SimpleNamespace(Image=types.SimpleNamespace(debian_slim=lambda **kwargs: Image()),
          App=lambda name: types.SimpleNamespace(function=lambda **kwargs: lambda f: f),
          Volume=types.SimpleNamespace(from_name=lambda *args, **kwargs: None),
          concurrent=lambda **kwargs: lambda f: f, asgi_app=lambda: lambda f: f, is_local=lambda: True)
    monkeypatch.setitem(sys.modules, "modal", modal)
    monkeypatch.delitem(sys.modules, "argus.platform.publication_modal", raising=False)
    monkeypatch.setenv("PANOPTES_DATA_ROOT", str(tmp_path))
    for key in ("PANOPTES_PUBLICATION_CATALOG", "PANOPTES_PUBLICATION_HTTP", "PANOPTES_FEEDBACK_MODEL"):
        monkeypatch.delenv(key, raising=False)
    publication = importlib.import_module("argus.platform.publication_modal")
    assert publication.catalog == tmp_path / "publications/catalog"
    assert publication.prepared == tmp_path / "publications/http"
    assert ("add_local_python_source", ("argus",)) in calls
    assert all("ehs_spatial" not in str(args) for name, args in calls)


def test_local_executor_starts_installed_worker(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: calls.append(argv) or types.SimpleNamespace(wait=lambda: 0))
    executor = LocalJobExecutor()
    try:
        assert executor._run("job", "reference") == 0
        assert calls == [[sys.executable, "-m", "argus.platform.worker", "--job-id", "job"]]
    finally:
        executor.close()
