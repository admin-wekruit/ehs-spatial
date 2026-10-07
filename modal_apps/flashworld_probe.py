"""One private, ephemeral FlashWorld experiment; no endpoint or production adapter.

Run only when authorized: modal run modal_apps/flashworld_probe.py \
  --input-dir <frozen input> --output-dir <new result directory>
Root's separate CPU preparation must finish before the A100-80GB invocation.
"""

import hashlib
import io
import json
import math
from pathlib import Path

import modal

CODE_REV = "424db4487de308a39e5137afbe85e7568a96193f"
MODEL_ID = "imlixinyang/FlashWorld"
MODEL_REV = "6a8e88c6f88678ac098e4c82675f0aee555d6e5d"
MODEL_SHA256 = "610a09a11917ae112ea71f2f42291d5c8f11f54ae501ec14a8740f22da2540bd"
BASE_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
BASE_REV = "b8fff7315c768468a5333511427288870b2e9635"
RESOLUTION = [24, 480, 704]
SEED = 0

app = modal.App("flashworld-private-probe")
volume = modal.Volume.from_name("panoptes-flashworld-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential", "cmake", "ninja-build", "libgl1", "libglib2.0-0", "ffmpeg")
    .env({"TORCH_CUDA_ARCH_LIST": "8.0", "MAX_JOBS": "4", "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HOME": "/cache/huggingface"})
    .run_commands(
        "git clone https://github.com/imlixinyang/FlashWorld.git /opt/flashworld",
        f"git -C /opt/flashworld checkout --detach {CODE_REV}",
    )
    .pip_install("torch==2.6.0", "torchvision==0.21.0", index_url="https://download.pytorch.org/whl/cu124")
    .apt_install("zlib1g-dev")
    .run_commands(
        "python -c \"from pathlib import Path; s=Path('/opt/flashworld/requirements.txt').read_text(); g='git+https://github.com/nerfstudio-project/gsplat.git@32f2a54d21c7ecb135320bb02b136b7407ae5712'; assert s.count('uvicorn==0.37.9') == 1 and s.count(g) == 1; Path('/tmp/flashworld-install-requirements.txt').write_text(s.replace('uvicorn==0.37.9', 'uvicorn==0.37.0').replace(g, ''))\"",
        "CC=gcc CXX=g++ python -m pip install -r /tmp/flashworld-install-requirements.txt",
        "CC=gcc CXX=g++ python -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@32f2a54d21c7ecb135320bb02b136b7407ae5712",
    )
    .pip_install("huggingface_hub==0.36.0", "sentencepiece==0.2.1", "imageio-ffmpeg==0.6.0")
    .run_commands("cd /opt/flashworld && python -c \"import cli; assert cli.mode == 'Local'\"")
)


def sha256_file(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_contract(image_bytes, intrinsics, manifest):
    """Mirror pinned CLI crop arithmetic; report its K/image rounding difference."""
    from PIL import Image

    digest = hashlib.sha256(image_bytes).hexdigest()
    if digest != manifest["source_sha256"]:
        raise ValueError("Source bytes do not match frozen manifest")
    if intrinsics != manifest["intrinsics"]:
        raise ValueError("Intrinsics do not match frozen manifest")
    if len(intrinsics) != 3 or any(len(row) != 3 for row in intrinsics):
        raise ValueError("Expected 3x3 source K")
    if not all(math.isfinite(v) for row in intrinsics for v in row):
        raise ValueError("K must be finite")
    if intrinsics[0][0] <= 0 or intrinsics[1][1] <= 0 or intrinsics[2] != [0, 0, 1] or intrinsics[0][1] != 0 or intrinsics[1][0] != 0:
        raise ValueError("Native input supports positive focal lengths and zero skew")
    with Image.open(io.BytesIO(image_bytes)) as source:
        width, height = source.size
        if source.format != "PNG" or source.mode != "RGB":
            raise ValueError("This bounded probe requires the frozen RGB canonical PNG")
    if [width, height] != manifest["canonical_size_wh"] or manifest["seed"] != SEED:
        raise ValueError("Frozen image size or seed mismatch")
    frames, out_h, out_w = RESOLUTION
    scale = max(out_h / height, out_w / width)
    crop_h, crop_w = int(out_h / scale), int(out_w / scale)
    left, top = (width - crop_w) // 2, (height - crop_h) // 2
    sx, sy = out_w / crop_w, out_h / crop_h
    processed_k = [[intrinsics[0][0] * scale, 0, (intrinsics[0][2] - left) * scale],
                   [0, intrinsics[1][1] * scale, (intrinsics[1][2] - top) * scale], [0, 0, 1]]
    cameras = [{"quaternion": [1, 0, 0, 0], "position": [0.2 * i / (frames - 1), 0, 0],
                "fx": intrinsics[0][0], "fy": intrinsics[1][1],
                "cx": intrinsics[0][2], "cy": intrinsics[1][2]} for i in range(frames)]
    request = {"image_prompt": "source.png", "text_prompt": "", "resolution": RESOLUTION,
               "image_index": 0, "cameras": cameras}
    transform = {
        "source_size_wh": [width, height], "processed_size_wh": [out_w, out_h],
        "crop_box_ltrb": [left, top, left + crop_w, top + crop_h],
        "native_cli_scalar_scale": scale, "native_cli_processed_K": processed_k,
        "actual_image_edge_transform": [[sx, 0, -sx * left], [0, sy, -sy * top], [0, 0, 1]],
        "actual_integer_pixel_center_transform": [[sx, 0, sx * (0.5 - left) - 0.5],
                                                   [0, sy, sy * (0.5 - top) - 0.5], [0, 0, 1]],
        "native_K_scale_minus_actual_xy": [scale - sx, scale - sy],
        "image_resize": "Pinned CLI PIL RGB crop then resize; default Pillow BICUBIC",
        "K_pixel_convention": "Source K passed exactly as supplied; no invented half-pixel correction",
    }
    return request, transform


@app.function(image=image, gpu="A100-80GB", cpu=8, memory=131072,
              volumes={"/cache": volume}, timeout=1800, max_containers=1, retries=0,
              block_network=True, single_use_containers=True)
def probe(image_bytes: bytes, intrinsics: list, source_manifest: dict):
    import contextlib
    import os
    import random
    import subprocess
    import sys
    import time
    import traceback
    import uuid

    volume.reload()
    request, transform = input_contract(image_bytes, intrinsics, source_manifest)
    output = Path("/cache/probes") / uuid.uuid4().hex
    output.mkdir(parents=True)
    (output / "source.png").write_bytes(image_bytes)
    (output / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2))
    weights = json.loads(Path("/cache/flashworld-weight-preparation.json").read_text())
    models = {record["repo"]: record for record in weights["models"]}
    if models[MODEL_ID]["revision"] != MODEL_REV or models[BASE_ID]["revision"] != BASE_REV or weights["flashworld_sha256"] != MODEL_SHA256:
        raise ValueError("CPU weight preparation does not match the fixed model revisions/checksum")
    if Path(models[MODEL_ID]["snapshot"]).name != MODEL_REV or Path(models[BASE_ID]["snapshot"]).name != BASE_REV:
        raise ValueError("Weight snapshot paths must name the immutable revisions")
    if (Path(models[MODEL_ID]["snapshot"]) / "model.ckpt").stat().st_size != 20936494840:
        raise ValueError("Checkpoint size differs from the SHA-verified CPU preparation")
    (output / "weights-manifest.json").write_text(json.dumps(weights, indent=2))
    request["image_prompt"] = str(output / "source.png")
    (output / "input.json").write_text(json.dumps(request, indent=2))
    metadata = {"status": "failed", "code_revision": CODE_REV, "seed": SEED,
                "source_sha256": source_manifest["source_sha256"], "preprocess": transform,
                "source_camera_registration": "unknown; diagnostic render only", "metric_scale_known": False,
                "camera_convention": "OpenGL camera-to-world, quaternion wxyz, +x right, +y up, -z forward",
                "trajectory": "24 fixed views, pure +x translation from 0 to 0.2 model coordinate units",
                "scale_rule": "native translations / (T_norm + 0.01); native PLY/SPZ xyz/scales * T_norm",
                "source_K_provenance": source_manifest["intrinsics_origin"],
                "code_license": "Apache-2.0", "checkpoint_license": "CC-BY-NC-SA-4.0",
                "base_license": "Apache-2.0", "inference_network_blocked": True,
                "environment_overrides": {
                    "uvicorn": "0.37.9 upstream pin does not exist on PyPI; install 0.37.0 for unused web import, source requirements untouched",
                    "gsplat_build": "Disable build isolation only for gsplat so setup.py can use the already pinned torch installation",
                    "compiler": "Use installed gcc/g++ for native packages; base image CC=clang names an absent compiler"}}
    started = time.monotonic()
    with (output / "runtime.log").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        try:
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
            os.chdir("/opt/flashworld")
            sys.path.insert(0, "/opt/flashworld")
            if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != CODE_REV:
                raise ValueError("Upstream source revision mismatch")
            (output / "pip-freeze.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
            # The official literal model_id also accepts this pinned local directory.
            local_base = Path(BASE_ID)
            local_base.parent.mkdir(exist_ok=True)
            local_base.symlink_to(models[BASE_ID]["snapshot"], target_is_directory=True)
            import numpy as np
            import torch
            import imageio
            from PIL import Image
            from app import GenerationSystem
            from cli import process_generation_request
            from gsplat import rasterization
            from utils import normalize_cameras, quaternion_to_matrix

            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            metadata["hardware"] = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                                    "cuda": torch.version.cuda, "pillow": Image.__version__}
            t0 = time.monotonic()
            system = GenerationSystem(str(Path(models[MODEL_ID]["snapshot"]) / "model.ckpt"), device="cuda")
            torch.cuda.synchronize()
            metadata["model_load_seconds"] = time.monotonic() - t0
            captured = {}

            class RecordedGeneration:
                # ponytail: observe the official CLI seam; no model or preprocess replacement.
                def generate(self, *args, **kwargs):
                    captured["cameras"] = args[0].detach().clone()
                    captured["image"] = args[2].detach().clone()
                    begin = time.monotonic()
                    result = system.generate(*args, **kwargs)
                    torch.cuda.synchronize()
                    metadata["native_generate_seconds"] = time.monotonic() - begin
                    captured["result"] = result
                    return result

            # Keep exactly one scene; render only its 24 anchor views below, not 346 interpolated frames.
            process_generation_request(request, RecordedGeneration(), str(output), video=False, spz=True, ply=True)
            scene, ref_w2c, radius = captured["result"]
            cameras, _, _ = normalize_cameras(captured["cameras"].cuda()[None], return_meta=True)
            metadata["ref_w2c"] = ref_w2c.cpu().tolist()
            metadata["T_norm"] = radius.cpu().tolist()
            metadata["normalized_cameras"] = cameras.cpu().tolist()
            metadata["cli_processed_cameras"] = captured["cameras"].cpu().tolist()
            metadata["gaussian_count"] = int(scene.shape[0])
            condition = captured["image"].permute(1, 2, 0).cpu().add(1).mul(127.5).round().clamp(0, 255).byte().numpy()
            Image.fromarray(condition).save(output / "condition.png")
            with torch.no_grad():
                rgb, depth = system.recon_decoder.render([scene], cameras, RESOLUTION[1], RESOLUTION[2], bg_mode="white")
                rgb_u8 = rgb[0].clamp(-1, 1).add(1).div(2).permute(0, 2, 3, 1).cpu().mul(255).round().byte().numpy()
                Image.fromarray(rgb_u8[0]).save(output / "source-render.png")
                Image.fromarray(rgb_u8[-1]).save(output / "novel-render.png")
                imageio.mimwrite(output / "trajectory.mp4", rgb_u8, fps=6, quality=8, macro_block_size=1)
                np.save(output / "source-depth.npy", depth[0, 0, 0].cpu().numpy())
                # Upstream renderer uses gsplat RGB+D but drops alpha; request that same native result once.
                sh_degree = system.recon_decoder.gs_head.sh_degree
                coefficients = (sh_degree + 1) ** 2
                xyz, opacity, scale, rotation, feature = scene.float().split([3, 1, 3, 4, 3 * coefficients], dim=-1)
                metadata["spherical_harmonics_degree"] = sh_degree
                c2w = torch.eye(4, device="cuda")
                c2w[:3, :3] = quaternion_to_matrix(cameras[0, 0, :4])
                c2w[:3, 3] = cameras[0, 0, 4:7]
                c2w[:3, 1:3] *= -1
                K = torch.eye(3, device="cuda")
                K[0, 0], K[1, 1], K[0, 2], K[1, 2] = cameras[0, 0, 7:] * cameras.new_tensor([RESOLUTION[2], RESOLUTION[1], RESOLUTION[2], RESOLUTION[1]])
                rendered, alpha, _ = rasterization(xyz, rotation, scale, opacity[:, 0], feature.reshape(-1, coefficients, 3),
                    c2w.inverse()[None], K[None], RESOLUTION[2], RESOLUTION[1], sh_degree=sh_degree,
                    near_plane=0.01, far_plane=1000, render_mode="RGB+D",
                    backgrounds=torch.ones(1, 3, device="cuda"), rasterize_mode="classic")
                np.save(output / "source-alpha.npy", alpha[0, :, :, 0].cpu().numpy())
                metadata["source_render_crosscheck_max_abs"] = float((rendered[0, :, :, :3].clamp(0, 1) - rgb[0, 0].permute(1, 2, 0).add(1).div(2)).abs().max())
            metadata["depth_semantics"] = "Native gsplat RGB+D accumulated depth in normalized model coordinates; not expected surface depth or meters"
            metadata["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            metadata["status"] = "complete"
        except Exception:
            metadata["error"] = traceback.format_exc()
            traceback.print_exc()
    metadata["gpu_function_seconds"] = time.monotonic() - started
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    files = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(output.iterdir()) if p.is_file()]
    volume.commit()
    return {"status": metadata["status"], "volume_path": str(output.relative_to("/cache")), "files": files}


@app.local_entrypoint()
def main(input_dir: str, output_dir: str):
    source, output = Path(input_dir), Path(output_dir)
    if output.exists():
        raise ValueError("Output directory must be new; refusing to overwrite prior results")
    image_bytes = (source / "canonical.png").read_bytes()
    intrinsics = json.loads((source / "intrinsics.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    input_contract(image_bytes, intrinsics, manifest)  # Validate before any remote work.
    result = probe.remote(image_bytes, intrinsics, manifest)
    output.mkdir(parents=True, exist_ok=False)
    for item in result["files"]:
        destination = output / item["name"]
        with destination.open("wb") as stream:
            for chunk in volume.read_file(f"{result['volume_path']}/{item['name']}"):
                stream.write(chunk)
        if sha256_file(destination) != item["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {item['name']}")
    (output / "download-manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"status": result["status"], "output": str(output), "files": len(result["files"])}))
    if result["status"] != "complete":
        raise RuntimeError("Native probe failed; inspect returned runtime.log and metadata.json")
