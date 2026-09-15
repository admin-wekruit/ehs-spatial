"""Exercise the actual SAM3D loader and error boundary without GPU/model calls."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


def load_worker(monkeypatch, tmp_path):
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"generation": {
        "runtimeImage": "registry.example/sam3d@sha256:" + "a" * 64,
        "pins": {"model": "facebook/sam-3d-objects", "codeRevision": "b" * 40, "modelRevision": "c" * 40},
        "distribution": "sam3d_objects", "checkpointConfig": "checkpoints/pipeline.yaml"}}))
    monkeypatch.setenv("PANOPTES_MODEL_RUNTIME_MANIFEST", str(manifest))
    decorator = lambda **kwargs: lambda value: value
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(
        App=lambda name: SimpleNamespace(cls=decorator), enter=decorator, method=decorator,
        Image=SimpleNamespace(from_registry=lambda name: SimpleNamespace(env=lambda values: None)),
        Secret=SimpleNamespace(from_name=lambda name: name)))
    spec = importlib.util.spec_from_file_location("sam3d_worker_test", Path(__file__).parents[1] / "modal_apps/platform_models.py")
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.setattr(worker, "_verify_distribution", lambda stage: None)
    return worker


def test_loader_removes_depth_factory_before_hydra_recurses(monkeypatch, tmp_path):
    worker = load_worker(monkeypatch, tmp_path)
    constructed = []

    def depth_factory():
        constructed.append(True)
        raise AssertionError("An internal depth model would be constructed/downloaded")

    settings = SimpleNamespace(depth_model=depth_factory)

    def instantiate(config, **overrides):
        # Hydra resolves overrides before recursively constructing nested targets.
        depth = overrides.get("depth_model", config.depth_model)
        return SimpleNamespace(depth_model=depth() if callable(depth) else depth)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: str(tmp_path)))
    monkeypatch.setitem(sys.modules, "omegaconf", SimpleNamespace(OmegaConf=SimpleNamespace(load=lambda path: settings)))
    monkeypatch.setitem(sys.modules, "hydra.utils", SimpleNamespace(instantiate=instantiate))
    instance = worker.SAM3DObjects()
    instance.load()
    assert constructed == [] and instance.depth_calls == 0
    assert settings.rendering_engine == "pytorch3d" and settings.compile_model is False
    with pytest.raises(RuntimeError, match="External pointmap required"):
        instance.pipeline.depth_model()
    assert instance.depth_calls == 1
    with pytest.raises(AssertionError, match="constructed/downloaded"):
        instantiate(settings)  # Negative control: original construction fails.


def test_external_pointmap_is_preserved_and_failure_returns_telemetry(monkeypatch, tmp_path):
    worker = load_worker(monkeypatch, tmp_path)

    class Event:
        def __init__(self, **kwargs): pass
        def record(self): pass
        def elapsed_time(self, end): return 1250

    class Tensor:
        def __init__(self, data): self.data = data
        def float(self): return self
        def cuda(self): return self

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        from_numpy=Tensor, cuda=SimpleNamespace(Event=Event, synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "pytorch3d.transforms", SimpleNamespace(quaternion_to_matrix=None))
    monkeypatch.setitem(sys.modules, "sam3d_objects.data.dataset.tdfy.transforms_3d", SimpleNamespace(compose_transform=None))
    received = {}

    def run(image, mask, **kwargs):
        received.update(image=image, mask=mask, **kwargs)
        raise RuntimeError("GPU inference failed after receiving the pointmap")

    instance = worker.SAM3DObjects()
    instance.depth_calls = 0
    instance.pipeline = SimpleNamespace(run=run)
    pointmap = np.arange(18, dtype=np.float32).reshape(2, 3, 3) / 10
    payload = {"image": np.full((2, 3, 3), 123, np.uint8), "mask": np.ones((2, 3), bool),
        "pointmap": pointmap, "seed": 7, "decode_formats": ["mesh"], "with_mesh_postprocess": False,
        "with_texture_baking": False, "with_layout_postprocess": False, "use_vertex_color": True}
    result = instance.run(payload)
    np.testing.assert_array_equal(received["pointmap"].data, pointmap)
    np.testing.assert_array_equal(received["image"][..., :3], payload["image"])
    assert received["seed"] == 7 and received["estimate_plane"] is False
    assert received["decode_formats"] == ["mesh"] and instance.depth_calls == 0
    assert result["providerError"]["code"] == "gpu_inference_failed"
    assert result["telemetry"]["gpuElapsedSeconds"] == 1.25
    assert result["telemetry"]["workerElapsedSeconds"] >= 0
    assert result["telemetry"]["actualCostUsd"] is None
