"""Small executable check of exported mesh units and floor alignment."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.workcell_photo_oneshot import _export_metric_scene


def check():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Floor y=2, up=-Y. This beam's lowest face has native clearance .4.
        for name in ("robot-v4", "cart-single", "posts", "fence-fitted", "floor-fitted"):
            mesh = trimesh.creation.box([1, .2, .1])
            mesh.apply_translation([0, 1.5, 0])
            (root / f"{name}.glb").write_bytes(mesh.export(file_type="glb"))
        previous = None
        for scale in (1, 2):
            geometry = {"anchor": {"mPerNative": scale, "assumedHeightM": .2 * scale,
                                   "assumedWidthM": .2 * scale},
                        "floor": {"normal": [0, -1, 0], "offset": 2}}
            _export_metric_scene(root, geometry)
            loaded = trimesh.load(root / "workcell-metric.glb", force="scene").to_geometry()
            assert np.isclose(loaded.bounds[0, 1], .4 * scale, atol=1e-6), loaded.bounds
            assert np.isclose(loaded.extents[0], scale, atol=1e-6), loaded.extents
            if previous is not None:
                assert np.allclose(loaded.bounds, previous * 2, atol=1e-6)
            previous = loaded.bounds
        geometry["anchor"]["mPerNative"] = float("nan")
        try:
            _export_metric_scene(root, geometry)
        except ValueError:
            pass
        else:
            raise AssertionError("NaN scale accepted")
    print("metric export: scale, clearance, Y-up round-trip and invalid input checks passed")


if __name__ == "__main__":
    check()
