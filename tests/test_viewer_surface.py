"""Minimal runtime checks for the two independently positioned 3D modes."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from ehs_spatial.viewer import _surface_data


def test_surface_rejects_inventory_or_asset_changes(tmp_path):
    (tmp_path / "inventory").mkdir()
    inventory = tmp_path / "inventory" / "inventory.json"
    inventory.write_text('{"objects":[{}]}')
    directory = tmp_path / "surface"
    directory.mkdir()
    manifest = {"supported_inv":[0],"inventory_sha256":hashlib.sha256(inventory.read_bytes()).hexdigest()}
    for name, key in [("surface.glb","asset_sha256"),("face-inv.bin","face_map_sha256")]:
        (directory / name).write_bytes(b"verified")
        manifest[key] = hashlib.sha256(b"verified").hexdigest()
    (directory / "surface.json").write_text(json.dumps(manifest))
    assert _surface_data(tmp_path,[0])["asset_url"].endswith("/surface/surface.glb")
    with pytest.raises(ValueError,match="不可交互"):
        _surface_data(tmp_path,[])
    inventory.write_text('{"objects":[{"changed":true}]}')
    with pytest.raises(ValueError,match="过期"):
        _surface_data(tmp_path,[0])
    inventory.write_text('{"objects":[{}]}')
    (directory / "face-inv.bin").write_bytes(b"changed")
    with pytest.raises(ValueError,match="资产"):
        _surface_data(tmp_path,[0])


def test_native_surface_camera_javascript():
    result = subprocess.run(["node",str(Path(__file__).with_suffix(".mjs"))],capture_output=True,text=True)
    assert result.returncode == 0, result.stdout + result.stderr
