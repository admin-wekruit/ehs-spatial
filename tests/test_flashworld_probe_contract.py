"""Known pixels/K and the pinned upstream crop; no model or performance mock."""

import ast
import hashlib
import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from modal_apps.flashworld_probe import CODE_REV, input_contract


def source(width=518, height=392):
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[..., 0] = np.arange(width, dtype=np.uint16)[None] % 256
    pixels[..., 1] = np.arange(height, dtype=np.uint16)[:, None] % 256
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    raw = stream.getvalue()
    K = [[400.0, 0, 259.0], [0, 420.0, 196.0], [0, 0, 1]]
    manifest = {"source_sha256": hashlib.sha256(raw).hexdigest(), "intrinsics": K,
                "canonical_size_wh": [width, height], "seed": 0}
    return raw, K, manifest


def test_known_crop_and_camera_projection():
    raw, K, manifest = source()
    request, transform = input_contract(raw, K, manifest)
    assert transform["crop_box_ltrb"] == [0, 19, 518, 372]
    A = np.array(transform["actual_image_edge_transform"])
    np.testing.assert_allclose(A @ [0, 19, 1], [0, 0, 1])
    np.testing.assert_allclose(A @ [518, 372, 1], [704, 480, 1])
    # Exact pinhole projection under the CLI's scalar K transform, independent of image rounding.
    xyz = np.array([0.3, -0.2, 2.0])
    source_uv = np.array(K) @ xyz
    source_uv /= source_uv[2]
    expected = (source_uv[:2] - [0, 19]) * (704 / 518)
    projected = np.array(transform["native_cli_processed_K"]) @ xyz
    np.testing.assert_allclose(projected[:2] / projected[2], expected)
    assert transform["native_K_scale_minus_actual_xy"][1] != 0  # Never label this crop identity.
    assert len(request["cameras"]) == 24
    assert request["cameras"][0]["position"] == [0, 0, 0]
    assert request["cameras"][-1]["position"] == [0.2, 0, 0]
    assert all(c["quaternion"] == [1, 0, 0, 0] for c in request["cameras"])


def test_contract_matches_actual_pinned_cli_preprocess(tmp_path):
    """Execute only upstream image/camera preprocessing, stopping before model invocation."""
    torch = pytest.importorskip("torch")
    upstream = Path(__file__).parents[1] / "outputs/candidate-evaluation/vendor/flashworld"
    if not (upstream / "cli.py").exists():
        pytest.skip("Optional upstream preprocessing check needs the pinned local source checkout")
    import subprocess
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip() == CODE_REV
    tree = ast.parse((upstream / "cli.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "process_generation_request")
    prefix = []
    for node in function.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "start_time" for t in node.targets):
            break
        prefix.append(node)
    prefix.append(ast.Return(ast.Tuple([ast.Name("image", ast.Load()), ast.Name("cameras", ast.Load())], ast.Load())))
    function.body, function.decorator_list = prefix, []
    namespace = {"os": __import__("os"), "Image": Image, "torch": torch, "np": np}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(upstream / "cli.py"), "exec"), namespace)
    raw, K, manifest = source()
    request, transform = input_contract(raw, K, manifest)
    path = tmp_path / "source.png"
    path.write_bytes(raw)
    request["image_prompt"] = str(path)
    processed, cameras = namespace["process_generation_request"](request, None, "unused")
    native_K = np.array(transform["native_cli_processed_K"])
    np.testing.assert_allclose(cameras[0, 7:].numpy() * [704, 480, 704, 480],
                               native_K[[0, 1, 0, 1], [0, 1, 2, 2]], atol=4e-5)
    expected = np.asarray(Image.open(path).crop(tuple(transform["crop_box_ltrb"])).resize((704, 480)))
    actual = processed.add(1).mul(127.5).round().byte().permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, expected)


def test_refuse_input_and_calibration_drift():
    raw, K, manifest = source()
    with pytest.raises(ValueError, match="Source bytes"):
        input_contract(raw + b"changed", K, manifest)
    changed = [row[:] for row in K]
    changed[0][0] += 1
    with pytest.raises(ValueError, match="Intrinsics do not match"):
        input_contract(raw, changed, manifest)
