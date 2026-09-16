"""Pinned model workers for the platform, distinct from historical NC services.

Deploy only with PANOPTES_MODEL_RUNTIME_MANIFEST: stage -> pins, runtimeImage
(registry @sha256 digest) or modalImageId, distribution, and for SAM3D checkpointConfig.
Geometry/depth also require adapterSourceSha256 of this exact deployed file.
Images must already contain the audited pinned GPU dependencies. This file does
not claim that an image, HF access, native-pose fixture or quality gate has passed.
"""
import base64
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import tempfile
import time
from functools import wraps

import modal

app = modal.App("panoptes-platform-models")
manifest_path = os.environ.get("PANOPTES_MODEL_RUNTIME_MANIFEST")
# Remote workers import this module without the deployer's local filesystem.
frozen_config = os.environ.get("PANOPTES_MODEL_RUNTIME_CONFIG")
CONFIG = json.loads(frozen_config) if frozen_config else json.loads(Path(manifest_path).read_text()) if manifest_path else {}


def _timed_gpu(function):
    """CUDA stream interval, not GPU billing time; cold-start/billing remain unknown."""
    @wraps(function)
    def measured(*args,**kwargs):
        import torch
        started = time.monotonic()
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            result = function(*args,**kwargs)
        except Exception:
            result = {"providerError":{"code":"gpu_inference_failed"}}
        gpu_seconds = None
        try:
            end.record()
            torch.cuda.synchronize()
            gpu_seconds = start.elapsed_time(end)/1000
        except Exception:
            pass  # A lost device still leaves wall time; GPU time is unknown.
        result["telemetry"] = {"gpuElapsedSeconds":gpu_seconds,"workerElapsedSeconds":time.monotonic()-started,
                               "timingMethod":"cuda_event_stream_interval","actualCostUsd":None,"usage":None}
        return result
    return measured


def _verify_adapter_source(stage):
    if (stage in ("geometry", "depth") and CONFIG[stage].get("adapterSourceSha256") !=
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
        raise RuntimeError("Deployed adapter source differs from frozen runtime")


def _runtime_image(stage):
    _verify_adapter_source(stage)
    config = CONFIG[stage]
    for key in ("modelRevision","codeRevision"):
        if not re.fullmatch(r"[a-f0-9]{40}",config["pins"][key]):
            raise ValueError("Model and code require immutable revision pins")
    if ("runtimeImage" in config) == ("modalImageId" in config):
        raise ValueError("Specify one immutable runtime image")
    if "modalImageId" in config:
        if stage not in {"geometry", "depth"} or not re.fullmatch(r"im-[A-Za-z0-9]{22}", config["modalImageId"]):
            raise ValueError("Invalid immutable Modal image ID")
        image = modal.Image.from_id(config["modalImageId"])
    else:
        if not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}",config["runtimeImage"]):
            raise ValueError("runtimeImage must use a content digest")
        image = modal.Image.from_registry(config["runtimeImage"])
    return image.env({"HF_HOME":"/cache/huggingface",
                      "PANOPTES_MODEL_RUNTIME_CONFIG":json.dumps(CONFIG,sort_keys=True,separators=(",",":"))})


def _runtime_digest(stage, expected):
    _verify_adapter_source(stage)
    actual = hashlib.sha256(json.dumps(CONFIG[stage],sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
    if expected is not None and expected != actual:
        raise ValueError("Frozen runtime manifest differs from deployed runtime")
    return actual


def _verify_distribution(stage):
    _verify_adapter_source(stage)
    config = CONFIG[stage]
    distribution = importlib.metadata.distribution(config["distribution"])
    provenance = distribution.read_text("direct_url.json")
    actual = json.loads(provenance or "{}").get("vcs_info",{}).get("commit_id")
    if actual != config["pins"]["codeRevision"]:
        raise RuntimeError("Installed source revision is not the configured revision")
    if stage == "generation":
        receipt = distribution.locate_file("sam3d_objects/panoptes_mesh_build.json").read_bytes()
        if hashlib.sha256(receipt).hexdigest() != config.get("meshSourceBuildSha256"):
            raise RuntimeError("SAM3D mesh source build differs from configuration")
        build = json.loads(receipt)
        if build["codeRevision"] != actual:
            raise RuntimeError("SAM3D mesh source build revision differs")
        for relative, hashes in build["files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "sam3d_objects":
                raise RuntimeError("SAM3D mesh source build path is invalid")
            if hashlib.sha256(distribution.locate_file(relative).read_bytes()).hexdigest() != hashes["patchedSha256"]:
                raise RuntimeError("SAM3D patched source differs from recorded build")


def _image(record):
    from PIL import Image
    import numpy as np
    prefix,encoded = record["dataUri"].split(",",1)
    if prefix != "data:image/png;base64":
        raise ValueError("Expected normalized PNG bytes")
    raw = base64.b64decode(encoded,validate=True)
    import hashlib
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Source image hash differs")
    with Image.open(io.BytesIO(raw)) as image:
        rgb = np.asarray(image.convert("RGB"))
    if rgb.shape[:2] != (record["height"],record["width"]):
        raise ValueError("Source image dimensions differ")
    return raw,rgb


def _views_with_mapping(paths):
    """Reuses the prior exact upstream raster replay, independent of any floor."""
    import numpy as np
    from PIL import Image,ImageOps
    from mapanything.utils.image import load_images,rgb
    from mapanything.utils.cropping import rescale_image_and_other_optional_info,crop_image_and_other_optional_info
    views = load_images(paths)
    if len(views) != len(paths):
        raise ValueError("Upstream skipped an image")
    metadata = []
    for path,view in zip(paths,views):
        with Image.open(path) as image:
            original = ImageOps.exif_transpose(image).convert("RGB")
        h,w = map(int,view["true_shape"][0])
        resized = rescale_image_and_other_optional_info(original,np.array([w,h]))[0]
        rw,rh = resized.size
        left,top = (rw-w)//2,(rh-h)//2
        canonical = np.asarray(crop_image_and_other_optional_info(resized,[left,top,left+w,top+h])[0])
        loaded = np.rint(rgb(view["img"][0],view["data_norm_type"][0])*255).astype(np.uint8)
        if not np.array_equal(canonical,loaded):
            raise ValueError("Preprocessing raster replay differs")
        sx,sy = rw/original.width,rh/original.height
        metadata.append((canonical,[[sx,0,(sx-1)/2-left],[0,sy,(sy-1)/2-top],[0,0,1]]))
    return views,metadata


if "geometry" in CONFIG:
    if CONFIG["geometry"]["pins"]["model"] != "facebook/map-anything-apache":
        raise ValueError("Public platform requires Apache MapAnything weights")
    @app.cls(image=_runtime_image("geometry"),gpu="A100",cpu=(2,2),memory=(8192,8192),
             timeout=300,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
    class MapAnythingApache:
        @modal.enter()
        def load(self):
            _verify_distribution("geometry")
            from mapanything.models import MapAnything
            pins = CONFIG["geometry"]["pins"]
            self.model = MapAnything.from_pretrained(pins["model"],revision=pins["modelRevision"]).to("cuda").eval()

        @modal.method()
        @_timed_gpu
        def run(self,payload,expectedRuntimeManifestSha256=None):
            runtime_sha = _runtime_digest("geometry", expectedRuntimeManifestSha256)
            import numpy as np
            import torch
            images = payload["images"]
            if not 1 <= len(images) <= 4:
                raise ValueError("Expected one to four images")
            with tempfile.TemporaryDirectory() as root:
                paths = []
                for i,image in enumerate(images):
                    path = Path(root)/f"{i}.png"
                    path.write_bytes(_image(image)[0])
                    paths.append(str(path))
                views,metadata = _views_with_mapping(paths)
                with torch.inference_mode():
                    predictions = self.model.infer(views,memory_efficient_inference=False,use_amp=True,amp_dtype="bf16",apply_mask=False,mask_edges=True)
            if len(predictions) != len(images):
                raise ValueError("Prediction count differs")
            frames = []
            for source,prediction,(rgb,mapping) in zip(images,predictions,metadata):
                def array(key):
                    return prediction[key][0].float().cpu().numpy()
                predicted_rgb = np.clip(array("img_no_norm")*255+.5,0,255).astype(np.uint8)
                if not np.array_equal(predicted_rgb,rgb):
                    raise ValueError("Returned image changed the input raster")
                depth = array("depth_z").squeeze()
                gy,gx = np.gradient(depth)
                valid = array("non_ambiguous_mask").astype(bool) & (array("conf") >= .1)
                valid &= np.hypot(gx,gy)/np.maximum(depth,1e-6) < .08
                frames.append({"imageId":source["imageId"],"points":array("pts3d"),"valid":valid,"rgb":rgb,"K":array("intrinsics"),
                    "cameraToWorld":array("camera_poses"),"inputToCanonical":mapping})
            return {"frames":frames,"pins":CONFIG["geometry"]["pins"],"runtimeManifestSha256":runtime_sha}


if "depth" in CONFIG:
    if CONFIG["depth"]["pins"]["model"] != "Ruicheng/moge-3-vitl":
        raise ValueError("Depth stage requires MoGe3")
    @app.cls(image=_runtime_image("depth"),gpu="L4",cpu=(2,2),memory=(8192,8192),
             timeout=300,startup_timeout=300,retries=0,max_containers=1,scaledown_window=2)
    class MoGe3:
        @modal.enter()
        def load(self):
            _verify_distribution("depth")
            from moge.model.v3 import MoGeModel
            pins = CONFIG["depth"]["pins"]
            self.model = MoGeModel.from_pretrained(pins["model"],revision=pins["modelRevision"]).to("cuda").eval()

        @modal.method()
        @_timed_gpu
        def run(self,payload,expectedRuntimeManifestSha256=None):
            runtime_sha = _runtime_digest("depth", expectedRuntimeManifestSha256)
            import numpy as np
            import torch
            _,rgb = _image(payload["image"])
            # Preserve the exact source grid; no run-level cached scale reuse.
            tensor = torch.from_numpy(rgb.copy()).float().permute(2,0,1).cuda()/255
            with torch.inference_mode():
                result = self.model.infer(tensor,use_fp16=True,apply_mask=False)
            arrays = {k:result[k].detach().cpu().numpy() for k in ("points","mask","intrinsics","depth")}
            return {**arrays,"imageId":payload["image"]["imageId"],"imageSha256":payload["image"]["sha256"],"inputToCanonical":np.eye(3),"pins":CONFIG["depth"]["pins"],"runtimeManifestSha256":runtime_sha}


if "generation" in CONFIG:
    @app.cls(image=_runtime_image("generation"),gpu="A100-80GB",timeout=900,retries=0,max_containers=1,
             secrets=[modal.Secret.from_name("huggingface")])
    class SAM3DObjects:
        @modal.enter()
        def load(self):
            _verify_distribution("generation")
            from huggingface_hub import snapshot_download
            from omegaconf import OmegaConf
            from hydra.utils import instantiate
            config = CONFIG["generation"]
            pins = config["pins"]
            if pins["model"] != "facebook/sam-3d-objects":
                raise ValueError("Only the audited SAM3D Objects runtime is allowed")
            root = Path(snapshot_download(pins["model"],revision=pins["modelRevision"]))
            relative = Path(config["checkpointConfig"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Checkpoint config must remain inside snapshot")
            settings = OmegaConf.load(root/relative)
            settings.rendering_engine = "pytorch3d"
            settings.compile_model = False
            settings.workspace_dir = str((root/relative).parent)
            # Hydra recursively builds configured depth models before __init__.
            self.pipeline = instantiate(settings, depth_model=None, decode_formats=["mesh"],
                slat_decoder_gs_config_path=None, slat_decoder_gs_ckpt_path=None,
                slat_decoder_gs_4_config_path=None, slat_decoder_gs_4_ckpt_path=None)
            self.depth_calls = 0
            def forbidden_depth(*args,**kwargs):
                self.depth_calls += 1
                raise RuntimeError("External pointmap required; internal depth disabled")
            self.pipeline.depth_model = forbidden_depth

        @modal.method()
        @_timed_gpu
        def run(self,payload,expectedRuntimeManifestSha256=None):
            runtime_sha = _runtime_digest("generation", expectedRuntimeManifestSha256)
            import numpy as np
            import torch
            from pytorch3d.transforms import quaternion_to_matrix
            from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
            flags = {"decode_formats":["mesh"],"with_mesh_postprocess":False,"with_texture_baking":False,
                     "with_layout_postprocess":False,"use_vertex_color":True}
            if any(payload.get(k) != v for k,v in flags.items()):
                raise ValueError("Mesh-only runtime flags differ")
            image,mask,pointmap = np.asarray(payload["image"]),np.asarray(payload["mask"]),np.asarray(payload["pointmap"])
            if image.shape[:2] != mask.shape or pointmap.shape != image.shape[:2]+(3,) or not np.isfinite(pointmap).all():
                raise ValueError("External pointmap grid differs")
            rgba = np.concatenate((image[...,:3],(mask.astype(np.uint8)*255)[...,None]),axis=-1)
            start_depth_calls = self.depth_calls
            result = self.pipeline.run(rgba,None,seed=payload["seed"],pointmap=torch.from_numpy(pointmap).float().cuda(),estimate_plane=False,**flags)
            mesh = result["glb"]
            raw = np.asarray(mesh.vertices,dtype=np.float32)
            # Official get_mesh uses this z-up -> y-up basis before compose_transform.
            basis = np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=np.float32)
            pose = compose_transform(scale=result["scale"],rotation=quaternion_to_matrix(result["rotation"]),translation=result["translation"])
            official = pose.transform_points(torch.from_numpy(raw@basis.T).to(result["translation"].device)[None])[0].detach().cpu().numpy()
            column_pose = pose.get_matrix()[0].detach().cpu().numpy().T
            basis4 = np.eye(4,dtype=np.float32)
            basis4[:3,:3] = basis
            colors = np.asarray(mesh.visual.vertex_colors)[:,:3].astype(np.float32)/255
            return {"vertices":raw,"faces":np.asarray(mesh.faces,dtype=np.uint32),"colors":colors,"officialPosedVertices":official,
                    "objectToProvider":column_pose@basis4,"decodeFormats":["mesh"],"internalDepthCalls":self.depth_calls-start_depth_calls,
                    "runtimeManifestSha256":runtime_sha,
                    "pins":CONFIG["generation"]["pins"]}
