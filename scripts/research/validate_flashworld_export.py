"""Correct only the pinned FlashWorld probe's serialized Gaussian layout, on CPU.

Example (no model load):
  .venv/bin/python scripts/research/validate_flashworld_export.py RESULT_DIR \
    --spz-module-dir /tmp/flashworld-export-audit/python-deps
  .venv/bin/python scripts/research/validate_flashworld_export.py --self-test \
    --spz-module-dir /tmp/flashworld-export-audit/python-deps

SPZ must be built from SPZ_REV, e.g. in a temporary target directory:
  CMAKE_ARGS='-DCMAKE_CXX_STANDARD=17' uv pip install --python .venv/bin/python \
    --target /tmp/flashworld-export-audit/python-deps --no-deps \
    git+https://github.com/nianticlabs/spz.git@a4fc69e7948c7152e807e6501d73ddc9c149ce37
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

CODE_REV = "424db4487de308a39e5137afbe85e7568a96193f"
SPZ_REV = "a4fc69e7948c7152e807e6501d73ddc9c149ce37"
FIELDS = (["x", "y", "z"] + [f"f_dc_{i}" for i in range(27)] + ["opacity"]
          + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
ARRAYS = ("positions", "scales", "rotations", "alphas", "colors", "sh")
OUTPUTS = ("corrected-gaussians.ply", "corrected-gaussians.spz", "export-validation.json")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_spz(module_dir):
    if module_dir:
        sys.path.insert(0, str(Path(module_dir).resolve()))
    import spz

    provenance = json.loads(importlib.metadata.distribution("spz").read_text("direct_url.json") or "{}")
    if provenance.get("vcs_info", {}).get("commit_id") != SPZ_REV:
        raise ValueError("SPZ distribution must identify the exact pinned Git commit")
    return spz


def native_cloud(path, spz):
    # ponytail: only the frozen N×38 binary PLY contract, not a general PLY importer.
    with path.open("rb") as stream:
        header = []
        while True:
            line = stream.readline(4096)
            if not line or not line.endswith(b"\n") or stream.tell() > 16384:
                raise ValueError("Invalid or oversized native PLY header")
            line = line.decode("ascii").rstrip("\r\n")
            header.append(line)
            if line == "end_header":
                break
        offset = stream.tell()
    if header[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("Expected native binary little-endian PLY")
    count_line = header[2].split()
    if count_line[:2] != ["element", "vertex"] or len(count_line) != 3:
        raise ValueError("Expected one vertex element")
    n = int(count_line[2])
    if not 0 < n <= 10 * 1024 * 1024:
        raise ValueError("Point count is outside the pinned official PLY reader's supported range")
    if header[3:-1] != [f"property float {name}" for name in FIELDS]:
        raise ValueError("Native fields differ from the pinned FlashWorld exporter")
    if path.stat().st_size != offset + n * 38 * 4:
        raise ValueError("Native PLY body size does not match its declared rows")
    raw = np.memmap(path, mode="r", dtype="<f4", offset=offset, shape=(n, 38))
    if np.isnan(raw).any() or not np.isfinite(raw[:, :30]).all() or not np.isfinite(raw[:, 31:]).all():
        raise ValueError("Native geometry/SH/scales/rotations must be finite; only opacity logits may be infinite")
    if not np.allclose(np.linalg.norm(raw[:, 34:38], axis=1), 1, atol=1e-4, rtol=0):
        raise ValueError("Native Gaussian rotations are not unit quaternions")
    cloud = spz.GaussianCloud()
    cloud.sh_degree = 2
    cloud.positions = raw[:, :3].ravel()
    cloud.scales = raw[:, 31:34].ravel()
    cloud.rotations = raw[:, 34:38][:, [1, 2, 3, 0]].ravel()  # wxyz -> xyzw
    cloud.alphas = raw[:, 30].ravel()
    cloud.colors = raw[:, 3:6].ravel()
    cloud.sh = raw[:, 6:30].ravel()  # native coefficient-major RGB triplets
    return cloud


def spz_errors(expected, actual):
    if actual.num_points != expected.num_points or actual.sh_degree != 2:
        raise AssertionError("SPZ lost points or changed SH degree")
    result = {}
    for name in ("positions", "scales", "colors", "sh"):
        error = np.abs(np.asarray(getattr(actual, name)) - np.asarray(getattr(expected, name)))
        result[name] = {"max_abs": float(error.max()), "mean_abs": float(error.mean(dtype=np.float64))}
    a = np.asarray(expected.rotations).reshape(-1, 4).astype(np.float64)
    b = np.asarray(actual.rotations).reshape(-1, 4).astype(np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    angle = np.degrees(2 * np.arccos(np.clip(np.abs((a * b).sum(1)), 0, 1)))
    result["rotation_degrees_sign_invariant"] = {"max": float(angle.max()), "mean": float(angle.mean())}
    opacity = lambda logits: 0.5 * (1 + np.tanh(np.asarray(logits, dtype=np.float64) / 2))
    alpha_error = np.abs(opacity(expected.alphas) - opacity(actual.alphas))
    result["opacity_probability"] = {"max_abs": float(alpha_error.max()), "mean_abs": float(alpha_error.mean())}
    result["note"] = "SPZ is lossy; measured errors are reported without asserting equivalence to the float PLY. Scales are log-scales; colors are SH DC coefficients."
    return result


def validate(result_dir, spz):
    root = Path(result_dir).resolve()
    if any((root / name).exists() for name in OUTPUTS):
        raise FileExistsError("Refusing to overwrite a previous corrected export or validation report")
    source, metadata_path = root / "gaussians.ply", root / "metadata.json"
    before = {p.name: digest(p) for p in (source, metadata_path)}
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("code_revision") != CODE_REV or metadata.get("metric_scale_known") is not False:
        raise ValueError("Metadata must identify the pinned, uncalibrated FlashWorld probe")
    if metadata.get("camera_convention") != "OpenGL camera-to-world, quaternion wxyz, +x right, +y up, -z forward":
        raise ValueError("Source coordinate convention must explicitly be the probe's OpenGL/RUB convention")
    if metadata.get("scale_rule") != "native translations / (T_norm + 0.01); native PLY/SPZ xyz/scales * T_norm":
        raise ValueError("Unknown native scale transformation")
    radius = np.asarray(metadata["T_norm"], dtype=float).ravel()
    if radius.size != 1 or not np.isfinite(radius[0]) or radius[0] <= 0:
        raise ValueError("Expected a finite, positive native T_norm")
    cloud = native_cloud(source, spz)
    if cloud.num_points != metadata["gaussian_count"]:
        raise ValueError("Native PLY point count differs from the probe metadata")
    pack, unpack = spz.PackOptions(), spz.UnpackOptions()
    pack.from_coord, unpack.to_coord = spz.CoordinateSystem.RUB, spz.CoordinateSystem.RUB
    report = {
        "status": "complete", "upstream_probe_status": metadata.get("status"),
        "source_sha256": before, "code_revision": CODE_REV, "spz_revision": SPZ_REV,
        "spz_module": str(Path(spz.__file__).resolve()), "gaussian_count": cloud.num_points,
        "sh_degree": 2, "source_coordinates": "RUB", "corrected_spz_coordinates": "RUB",
        "corrected_ply_coordinates": "RDF (official SPZ PLY convention)",
        "source_to_corrected_ply_xyz": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
        "T_norm": float(radius[0]), "metric_scale_known": False,
        "units": "Preserve native exported units: xyz already equals normalized scene xyz * T_norm. No second scale multiplication, no source-camera/world registration.",
        "scale_encoding": "Preserve native log(scale * T_norm + 1e-8) exactly; no epsilon subtraction or re-activation/re-encoding.",
        "corrected_ply_rotation_fields": "wxyz; official coordinate conversion also changes signs",
        "cloud_and_spz_rotation_order": "xyzw",
        "validation_scope": "Lossless roundtrip to native PLY's six recovered arrays, not to the pre-export GPU tensor. Original PLY already applied T_norm and log(scale + 1e-8).",
    }
    with tempfile.TemporaryDirectory(prefix=".flashworld-export-", dir=root) as temporary:
        stage = Path(temporary)
        if not spz.save_splat_to_ply(cloud, pack, str(stage / OUTPUTS[0])):
            raise RuntimeError("Official PLY writer failed")
        loaded = spz.load_splat_from_ply(str(stage / OUTPUTS[0]), unpack)
        if loaded.num_points != cloud.num_points or loaded.sh_degree != 2:
            raise AssertionError("Corrected PLY point count or SH degree changed")
        exact = {name: bool(np.array_equal(getattr(cloud, name), getattr(loaded, name))) for name in ARRAYS}
        if not all(exact.values()):
            raise AssertionError(f"Official PLY roundtrip changed recovered arrays: {exact}")
        report["ply_exact_roundtrip"] = exact
        if not spz.save_spz(cloud, pack, str(stage / OUTPUTS[1])):
            raise RuntimeError("Official SPZ writer failed")
        report["spz_lossy_roundtrip"] = spz_errors(cloud, spz.load_spz(str(stage / OUTPUTS[1]), unpack))
        report["outputs"] = {name: {"bytes": (stage / name).stat().st_size, "sha256": digest(stage / name)} for name in OUTPUTS[:2]}
        if before != {p.name: digest(p) for p in (source, metadata_path)}:
            raise AssertionError("Source artifacts changed during conversion")
        (stage / OUTPUTS[2]).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        for name in OUTPUTS:
            if (root / name).exists():
                raise FileExistsError(root / name)
            (stage / name).rename(root / name)
    return report


def self_test(spz):
    with tempfile.TemporaryDirectory(prefix="flashworld-export-selftest-") as temporary:
        root = Path(temporary)
        raw = np.zeros((2, 38), dtype="<f4")
        raw[:, :3] = [[.123, .456, -.789], [1.23, -.45, -2.67]]
        raw[:, 3:30] = np.arange(54, dtype=np.float32).reshape(2, 27) / 128 - .2
        raw[:, 30] = [-.7, .9]
        raw[:, 31:34] = [[-4, -3, -2], [-3.1, -2.2, -1.3]]
        raw[:, 34:] = [[1, 0, 0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)]]
        header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
        header += "".join(f"property float {name}\n" for name in FIELDS) + "end_header\n"
        (root / "gaussians.ply").write_bytes(header.encode() + raw.tobytes())
        (root / "metadata.json").write_text(json.dumps({
            "status": "synthetic-self-test", "code_revision": CODE_REV,
            "metric_scale_known": False, "T_norm": [[[.2]]], "gaussian_count": 2,
            "camera_convention": "OpenGL camera-to-world, quaternion wxyz, +x right, +y up, -z forward",
            "scale_rule": "native translations / (T_norm + 0.01); native PLY/SPZ xyz/scales * T_norm",
        }))
        # The official importer must demonstrate why the native PLY cannot be used directly.
        options = spz.UnpackOptions()
        options.to_coord = spz.CoordinateSystem.RUB
        broken = spz.load_splat_from_ply(str(root / "gaussians.ply"), options)
        assert broken.sh_degree == 0 and len(broken.sh) == 0
        report = validate(root, spz)
        assert all(report["ply_exact_roundtrip"].values())
        assert report["spz_lossy_roundtrip"]["positions"]["max_abs"] <= 1 / 8192 + 1e-6
        assert report["spz_lossy_roundtrip"]["rotation_degrees_sign_invariant"]["max"] < .2
        recovered = spz.load_splat_from_ply(str(root / OUTPUTS[0]), options)
        assert np.array_equal(np.asarray(recovered.positions).reshape(2, 3), raw[:, :3])
        assert np.array_equal(np.asarray(recovered.rotations).reshape(2, 4)[0], [0, 0, 0, 1])
        try:
            validate(root, spz)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Overwrite must fail")
        print(json.dumps({"self_test": "passed", "ply_exact_roundtrip": report["ply_exact_roundtrip"],
                          "spz_lossy_roundtrip": report["spz_lossy_roundtrip"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("result_dir", nargs="?")
    parser.add_argument("--spz-module-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test == bool(args.result_dir):
        parser.error("Provide exactly one result directory or --self-test")
    spz = load_spz(args.spz_module_dir)
    if args.self_test:
        self_test(spz)
    else:
        result = validate(args.result_dir, spz)
        print(json.dumps({"status": result["status"], "gaussian_count": result["gaussian_count"],
                          "report": str(Path(args.result_dir).resolve() / OUTPUTS[2])}, indent=2))
