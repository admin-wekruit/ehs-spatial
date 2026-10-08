"""Self-hosted SAM 3D Objects (facebook/sam-3d-objects, SAM License) on Modal for research runs: one masked image and our own
pointmap in, one posed mesh out. On-prem capable: the same image runs on any Linux box with a >= 32 GB NVIDIA GPU.

The runtime is the platform's governed one (modal_apps/platform_models.py SAM3DObjects: mesh only, the reviewed
mesh-only patch, internal depth disabled so only the pointmap we pass places the object), built here as a Modal image
from the pinned code instead of an audited registry digest: research evidence, not a release. Weights come from the
gated HF repo at a pinned revision into a Modal volume (the huggingface secret; access verified with
scripts/research/check_sam3d_access.py).

Input pointmap: H x W x 3 in PyTorch3D camera coordinates (OpenCV camera x and y negated), NaN where there is no depth;
upstream compute_pointmap takes it as is and infers the intrinsics from it. Output: the mesh in object coordinates and
objectToCamera, the 4x4 that places it in that same PyTorch3D camera frame (upstream get_mesh: z-up -> y-up basis, then
scale, wxyz rotation and translation).

  modal run modal_apps/sam3d_research.py::probe          # image imports and GPU, no weights, no model call
  python scripts/complete_video_objects.py ... --generator sam3d --invoke
"""
from pathlib import Path

import modal

CODE_REVISION = "f91db411c50efee93d8db7aeb323885650f6f722"   # facebookresearch/sam-3d-objects
MODEL_REVISION = "2e73555018d2741ccd486e56c24fac41155a1dc6"  # facebook/sam-3d-objects on HF
UPSTREAM = f"https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/{CODE_REVISION}"
PYTORCH3D = "75ebeeaea0908c5527e7b1e305fbc7681382db47"  # upstream requirements.p3d.txt
GSPLAT = "2323de5905d5e90e035f792fe65bad0fedd413e7"     # upstream requirements.inference.txt
HYDRA_UTILS = "https://raw.githubusercontent.com/gleize/hydra/78f00766b5f37672aa7232ebbf01bdd74246bd60/hydra/core/utils.py"  # upstream patching/hydra
FLASH_ATTN = ("https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/"
              "flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")  # the pinned 2.8.3, prebuilt
from argus import ROOT
GPU = "A100-80GB"  # upstream needs >= 32 GB
USD_PER_SECOND = .000694 + 4 * .0000131 + 32 * .00000222  # Modal list price: A100-80GB + 4 cores + 32 GiB

image = (modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11")
         .apt_install("git", "curl", "build-essential", "ninja-build", "libgl1", "libglib2.0-0", "libegl1", "libxrender1", "libxext6",
                      "libsm6", "libxi6", "libxkbcommon0", "libxxf86vm1", "libx11-6", "ffmpeg")
         .env({"TORCH_CUDA_ARCH_LIST": "8.0;9.0", "FORCE_CUDA": "1", "MAX_JOBS": "8",
               "CC": "gcc", "CXX": "g++",  # Modal's Python was built with clang, which this image lacks: extensions build with gcc
               "PIP_EXTRA_INDEX_URL": "https://download.pytorch.org/whl/cu121",  # upstream also lists pypi.ngc.nvidia.com, which no longer resolves
               "PIP_FIND_LINKS": "https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"})
         .pip_install("torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1", index_url="https://download.pytorch.org/whl/cu121")
         .pip_install(FLASH_ATTN, "hatchling", "hatch-requirements-txt", "setuptools", "wheel", "ninja")
         # the two CUDA extensions compile against this torch (no isolated build env), each in its own cached layer
         .run_commands(f"pip install --no-build-isolation 'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@{PYTORCH3D}'")
         .run_commands(f"pip install --no-build-isolation 'gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@{GSPLAT}'")
         # upstream's own requirement files at the pinned revision, less nvidia-pyindex (a shim that only adds the dead NGC
         # index) and the two extensions just built; then the package itself without dependencies, so pip keeps its git provenance
         .run_commands(f"for f in requirements.txt requirements.p3d.txt requirements.inference.txt; do curl -fsSL {UPSTREAM}/$f; echo; done "
                       "| grep -v -e '^nvidia-pyindex' -e '^pytorch3d' -e '^gsplat' -e '^flash_attn' > /tmp/sam3d-requirements.txt && "
                       "pip install --no-build-isolation -r /tmp/sam3d-requirements.txt")
         .run_commands(f"pip install --no-build-isolation --no-deps 'sam3d_objects @ git+https://github.com/facebookresearch/sam-3d-objects.git@{CODE_REVISION}'")
         .run_commands(f"python -c \"import hydra; assert hydra.__version__ == '1.3.2'\" && "
                       f"curl -fsSL {HYDRA_UTILS} -o $(python -c 'import hydra, os; print(os.path.join(os.path.dirname(hydra.__file__), \"core\", \"utils.py\"))')")
         .add_local_file(ROOT / "argus/providers/prepare_sam3d_mesh_source.py", "/opt/prep/scripts/prepare_sam3d_mesh_source.py", copy=True)
         .add_local_file(ROOT / "argus/providers/sam3d_mesh_only.patch", "/opt/prep/scripts/sam3d_mesh_only.patch", copy=True)
         .run_commands("python /opt/prep/scripts/prepare_sam3d_mesh_source.py")  # the reviewed mesh-only patch, once, with its receipt
         .env({"HF_HOME": "/weights/huggingface", "CUDA_HOME": "/usr/local/cuda",
               "LIDRA_SKIP_INIT": "true"}))  # as upstream notebook/inference.py: its package init imports a module the public repo lacks
app = modal.App("panoptes-sam3d-objects-research")
weights = modal.Volume.from_name("panoptes-sam3d-weights", create_if_missing=True)


@app.function(image=image, gpu=GPU, timeout=600, retries=0)
def probe():
    """Imports and the GPU only: no weights, no model call."""
    import importlib.metadata
    import json
    import torch
    import pytorch3d, kaolin, gsplat, flash_attn, spconv  # noqa: F401  the extensions upstream builds on
    import sam3d_objects.pipeline.inference_pipeline_pointmap  # noqa: F401
    distribution = importlib.metadata.distribution("sam3d_objects")
    return {"cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0), "torch": str(torch.__version__),
            "installedRevision": json.loads(distribution.read_text("direct_url.json"))["vcs_info"]["commit_id"],
            "meshOnlyReceipt": distribution.locate_file("sam3d_objects/panoptes_mesh_build.json").exists()}


@app.cls(image=image, gpu=GPU, cpu=4, memory=32768, timeout=900, retries=0, max_containers=1,
         volumes={"/weights": weights}, secrets=[modal.Secret.from_name("huggingface")])
class SAM3DObjects:
    @modal.enter()
    def load(self):
        """As platform_models.SAM3DObjects.load: mesh decoder only, no depth or Gaussian models built."""
        from huggingface_hub import snapshot_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        root = Path(snapshot_download("facebook/sam-3d-objects", revision=MODEL_REVISION))
        weights.commit()
        settings = OmegaConf.load(root / "checkpoints/pipeline.yaml")
        settings.rendering_engine, settings.compile_model, settings.workspace_dir = "pytorch3d", False, str(root / "checkpoints")
        self.pipeline = instantiate(settings, depth_model=None, decode_formats=["mesh"],
                                    slat_decoder_gs_config_path=None, slat_decoder_gs_ckpt_path=None,
                                    slat_decoder_gs_4_config_path=None, slat_decoder_gs_4_ckpt_path=None)

        def forbidden_depth(*args, **kwargs):
            raise RuntimeError("external pointmap required; internal depth disabled")
        self.pipeline.depth_model = forbidden_depth

    @modal.method()
    def run(self, rgb, mask, pointmap, seed):
        """rgb H x W x 3 uint8, mask H x W bool, pointmap H x W x 3 float32 (PyTorch3D camera, NaN = no depth)."""
        import time
        import numpy as np
        import torch
        from pytorch3d.transforms import quaternion_to_matrix
        from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
        started = time.monotonic()
        rgb, mask, pointmap = np.asarray(rgb, np.uint8), np.asarray(mask, bool), np.asarray(pointmap, np.float32)
        if rgb.shape[:2] != mask.shape or pointmap.shape != rgb.shape[:2] + (3,):
            raise ValueError("image, mask and pointmap grids differ")
        rgba = np.concatenate([rgb[..., :3], (mask.astype(np.uint8) * 255)[..., None]], -1)
        try:
            result = self.pipeline.run(rgba, None, seed=seed, pointmap=torch.from_numpy(pointmap).cuda(), estimate_plane=False,
                                       decode_formats=["mesh"], with_mesh_postprocess=False, with_texture_baking=False,
                                       with_layout_postprocess=False, use_vertex_color=True)
        except Exception:  # the traceback comes back as data: the caller journals it and never re-sends this input
            import traceback
            return {"error": traceback.format_exc()[-6000:], "seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(0)}
        mesh = result["glb"]
        basis = np.eye(4, dtype=np.float32)
        basis[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # upstream get_mesh: z-up -> y-up before the pose
        pose = compose_transform(scale=result["scale"], rotation=quaternion_to_matrix(result["rotation"]), translation=result["translation"])
        return {"vertices": np.asarray(mesh.vertices, np.float32), "faces": np.asarray(mesh.faces, np.uint32),
                "colors": np.asarray(mesh.visual.vertex_colors)[:, :3].astype(np.uint8),
                "objectToCamera": (pose.get_matrix()[0].detach().cpu().numpy().T @ basis).astype(np.float64),  # row-vector pose, as a column matrix
                "pins": {"model": "facebook/sam-3d-objects", "modelRevision": MODEL_REVISION, "codeRevision": CODE_REVISION},
                "gpu": torch.cuda.get_device_name(0), "seconds": time.monotonic() - started}


@app.local_entrypoint()
def debug():
    """One synthetic call (a grey square on a plane two units away) to see the runtime work end to end."""
    import numpy as np
    rgb = np.full((256, 256, 3), 90, np.uint8)
    rgb[88:168, 88:168] = (200, 60, 40)
    mask = np.zeros((256, 256), bool)
    mask[88:168, 88:168] = True
    v, u = np.indices((256, 256), dtype=np.float64)
    z = np.full((256, 256), 2.)
    z[88:168, 88:168] = 1.8
    pointmap = np.stack([-(u - 127.5) / 250 * z, -(v - 127.5) / 250 * z, z], -1).astype(np.float32)
    out = SAM3DObjects().run.remote(rgb, mask, pointmap, 42)
    print(out["error"] if "error" in out else {k: (v.shape if hasattr(v, "shape") else v) for k, v in out.items()})
