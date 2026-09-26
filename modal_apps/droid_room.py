"""One bounded, pretrained official DROID-SLAM RGB experiment (no remote GT).

Run once: python modal_apps/droid_room.py execute --run-id NEW_ID --output ABS_NEW_RUN --reuse-build-from ABS_COMPLETED_RUN
Recover without resubmitting: python modal_apps/droid_room.py collect --run-id ID --output ABS_RUN
Check locally: python modal_apps/droid_room.py self-check
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import modal

REV = "2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99"
LIETORCH = "7f687644fcea81ab337831749445224503c1290d"
SCATTER = "6cf77c420f837a427b0d57965e414d68c1bd89ec"
ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
DATASET = ART / "data/rgbd_dataset_freiburg1_room"
SOURCE_K = [517.306408, 516.469215, 318.643040, 255.313989]
SOURCE_D = [0.262383, -0.953104, -0.005358, 0.002628, 1.163314]
# Calibrated 640x480 TUM-format clips. fr1-room keeps its original volume-root archive; others live under clips/NAME.
CLIPS = {"fr1-room": {"dataset": DATASET, "K": SOURCE_K, "D": SOURCE_D},
         "fr3-walking-xyz": {"dataset": ART / "data/tum-fr3-walking-xyz/rgbd_dataset_freiburg3_walking_xyz",
                             "K": [535.4, 539.2, 320.1, 247.6], "D": [0., 0., 0., 0., 0.]}}  # official tum3 calibration, already undistorted
# clips made from ordinary video by scripts/prepare_video_clip.py register themselves; their intrinsics are estimates, not calibrations
for _spec in sorted((ART / "data/clips").glob("*/clip.json")):
    _clip = json.loads(_spec.read_text())
    CLIPS.setdefault(_clip["name"], {"dataset": Path(_clip["dataset"]), "K": _clip["K"], "D": _clip["D"]})
CONTRACT_VERSION = "droid-final-upsampling-v2"
ROOT = Path("/artifact")
app = modal.App("panoptes-droid-room-once")
image = modal.Image.from_registry("pytorch/pytorch:2.7.0-cuda12.6-cudnn9-devel").entrypoint([])
volume = modal.Volume.from_name("panoptes-droid-room-001", create_if_missing=True)
attempts = modal.Dict.from_name("panoptes-droid-room-attempts", create_if_missing=True)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def patch_lowmem(source):
    """Fix the shared derived-output ordering; BA/tracking operations are unchanged."""
    replacements = [
        ("                s = 8\n", "                upsample_updates = []\n                s = 8\n"),
        ("                            self.video.upsample(torch.unique(iis), upmask)",
         "                            upsample_updates.append((torch.unique(iis), upmask))"),
        ("                self.video.dirty[:t] = True\n", """                if self.upsample:
                    if step == steps - 1:
                        self.video.phase2_final_lowres_before = self.video.disps[:t].clone()
                        self.video.phase2_final_poses_before = self.video.poses[:t].clone()
                    with autocast(enabled=True):
                        for source_ix, learned_mask in upsample_updates:
                            self.video.upsample(source_ix, learned_mask)
                    if step == steps - 1:
                        self.video.phase2_final_upsample_masks = [
                            (ix.clone(), mask.detach().clone()) for ix, mask in upsample_updates]
                        self.video.phase2_final_backend_steps = steps
                        assert torch.equal(self.video.phase2_final_lowres_before, self.video.disps[:t])
                        assert torch.equal(self.video.phase2_final_poses_before, self.video.poses[:t])
                self.video.dirty[:t] = True
""")]
    for before, after in replacements:
        assert source.count(before) == 1, "Pinned update_lowmem source differs"
        source = source.replace(before, after)
    compile(source, "factor_graph.py", "exec")
    return source


def prepare_image(image, calibration, resolution_scale=1):
    """Official TUM raster operations, full-precision source calibration."""
    import cv2
    import numpy as np
    if type(resolution_scale) is not int or resolution_scale not in (1, 2):
        raise ValueError('Resolution scale must be 1 or 2')
    if image.shape != (480, 640, 3):
        raise ValueError('Calibration requires a 640x480 source raster')
    s = resolution_scale
    fx, fy, cx, cy = calibration["source_K_fx_fy_cx_cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    image = cv2.undistort(image, K, np.asarray(calibration["source_distortion"]))
    image = cv2.resize(image, (352*s, 256*s))[8*s:-8*s, 16*s:-16*s]
    intrinsics = np.array([fx*352/640, fy*256/480, cx*352/640-16, cy*256/480-8], np.float32) * s
    return image, intrinsics


@app.function(image=image, gpu="A100-40GB", cpu=(4, 4), memory=(16384, 16384),
              timeout=840, startup_timeout=45, retries=0, max_containers=1,
              min_containers=0, scaledown_window=2, volumes={"/artifact": volume})
def infer(run_id, deadline, expected_archive_sha, expected_manifest_sha, expected_build_sha, resolution_scale=1, clip=None):
    if type(resolution_scale) is not int or resolution_scale not in (1, 2):
        raise ValueError('Resolution scale must be 1 or 2')
    s = resolution_scale
    # Platform preemption can restart inputs despite retries=0. Never re-execute inference.
    if not attempts.put(run_id, {"claimed_at": time.time()}, skip_if_exists=True):
        raise RuntimeError("GPU attempt already claimed; inspect artifacts, never resubmit")
    started = time.time()
    if deadline-started < 30:
        raise TimeoutError("GPU submission deadline already exhausted")
    import signal
    def expired(*_):
        raise TimeoutError("Single GPU experiment deadline reached")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(min(810, max(1, int(deadline-started)-30)))
    volume.reload()
    namespace = ROOT / "runs" / run_id
    namespace.mkdir(parents=True, exist_ok=False)
    out = namespace / "result"
    out.mkdir()
    save(out / "remote-run.json", {"status": "gpu_running", "started_unix": started})
    volume.commit()
    try:
        import difflib
        import shutil
        code = ROOT / "DROID-SLAM"
        assert sha(ROOT / "build.json") == expected_build_sha
        # The base GPU image has no git; the successful build manifest binds all pins.
        original = code / "droid_slam/factor_graph.py"
        assert sha(original) == "22403512df0b778b4265549919df9d4650dc5acf438754399a9dea2211952aba"
        before = original.read_text()
        copied = namespace / "droid_slam"
        shutil.copytree(code / "droid_slam", copied, ignore=shutil.ignore_patterns("__pycache__"))
        after = patch_lowmem(before)
        (copied / "factor_graph.py").write_text(after)
        (out / "native-source.patch").write_text("".join(difflib.unified_diff(
            before.splitlines(True), after.splitlines(True),
            fromfile="a/droid_slam/factor_graph.py", tofile="b/droid_slam/factor_graph.py")))
        save(out / "native-source.json", {"contract_version": CONTRACT_VERSION, "source_revision": REV,
             "shared_function": "FactorGraph.update_lowmem", "unmodified_source_sha256": sha(original),
             "modified_source_sha256": sha(copied / "factor_graph.py"),
             "patch_sha256": sha(out / "native-source.patch"), "cached_build_sha256": expected_build_sha,
             "changes": "Move original learned upsampling after same BA; retain last masks and invariance evidence",
             "python_source_sha256": {str(p.relative_to(copied)): sha(p) for p in copied.rglob("*.py")}})
        sys.path[:0] = [str(ROOT / "site"), str(copied)]
        os.environ["MPLBACKEND"] = "Agg"
        import cv2
        import numpy as np
        import torch
        import lietorch
        from droid import Droid
        from droid_net import cvx_upsample
        import factor_graph
        assert Path(factor_graph.__file__).resolve() == (copied / "factor_graph.py").resolve()
        torch.multiprocessing.set_start_method("spawn", force=True)
        build_info = json.loads((ROOT / "build.json").read_text())
        assert build_info["status"] == "cpu_build_complete"
        assert build_info["source_revision"] == REV and build_info["lietorch_revision"] == LIETORCH
        assert build_info["scatter_revision"] == SCATTER
        assert sha(ROOT / "droid.pth") == build_info["weights_sha256"]
        inputs = ROOT if clip is None else ROOT / "clips" / clip
        assert sha(inputs / "input-rgb.tar") == expected_archive_sha
        assert sha(inputs / "input-manifest.json") == expected_manifest_sha
        manifest = json.loads((inputs / "input-manifest.json").read_text())
        frame_count = manifest["frame_count"]
        assert frame_count == len(manifest["frames"]) and 0 < frame_count <= 2000
        input_dir = Path("/tmp/droid-room-rgb"); input_dir.mkdir()
        with tarfile.open(inputs / "input-rgb.tar") as archive:
            expected = {f["relative_path"] for f in manifest["frames"]}
            members = archive.getmembers()
            assert len(members) == len(expected) and {m.name for m in members} == expected
            for member in members:
                assert member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
                destination = input_dir / member.name; destination.parent.mkdir(exist_ok=True)
                with archive.extractfile(member) as source:
                    destination.write_bytes(source.read())
        images = []; image_hashes = []
        for record in manifest["frames"]:
            path = input_dir / record["relative_path"]
            assert sha(path) == record["sha256"]
            bgr = cv2.imread(str(path)); assert bgr.shape == (480, 640, 3)
            canonical, intrinsics = prepare_image(bgr, manifest, resolution_scale=s)
            image_hashes.append(hashlib.sha256(canonical.tobytes()).hexdigest())
            images.append(torch.from_numpy(canonical.copy()).permute(2, 0, 1)[None])
        # Official test_tum defaults; capacity covers all inputs plus filler batch16 (1400 for the 1362-frame room clip).
        args = argparse.Namespace(weights=str(ROOT / "droid.pth"), buffer=frame_count+38, image_size=[240*s, 320*s],
            disable_vis=True, beta=.3, filter_thresh=1.5, warmup=12, keyframe_thresh=2.0,
            frontend_thresh=12.0, frontend_window=25, frontend_radius=2, frontend_nms=1,
            backend_thresh=20.0, backend_radius=2, backend_nms=3, motion_damping=.5,
            upsample=True, stereo=False, asynchronous=False, frontend_device="cuda:0", backend_device="cuda:0")
        save(out / "inference-config.json", vars(args))
        K = torch.from_numpy(intrinsics)
        droid = Droid(args)
        with (out / "frames.jsonl").open("w") as ledger:
            for index, image_tensor in enumerate(images):
                before = droid.video.counter.value
                tick = time.monotonic()
                droid.track(index, image_tensor, intrinsics=K)
                torch.cuda.synchronize()
                row = {"source_index": index, "source_timestamp_text": manifest["frames"][index]["timestamp_text"],
                       "rgb_sha256": manifest["frames"][index]["sha256"],
                       "model_bgr_sha256": image_hashes[index], "native_timestamp": index,
                       "keyframe_count_before": before, "keyframe_count_after": droid.video.counter.value,
                       "tracking_seconds": time.monotonic()-tick}
                ledger.write(json.dumps(row)+"\n"); ledger.flush()
                if index % 100 == 0:
                    print(json.dumps({"frame": index, "keyframes": droid.video.counter.value, "elapsed": time.time()-started}), flush=True)
        stream = ((i, tensor, K) for i, tensor in enumerate(images))
        trajectory = droid.terminate(stream)
        assert trajectory.shape == (frame_count, 7) and np.isfinite(trajectory).all()
        poses = lietorch.SE3(torch.as_tensor(trajectory, device="cuda")).matrix().cpu().numpy()
        n = droid.video.counter.value
        indices = droid.video.tstamp[:n].cpu().numpy()
        assert np.array_equal(indices, np.round(indices)) and np.all(np.diff(indices)>0)
        key_c2w = lietorch.SE3(droid.video.poses[:n]).inv().matrix().cpu().numpy()
        key_indices = indices.astype(np.int64)
        agreement = np.abs(poses[key_indices] - key_c2w)
        mask_chunks = droid.video.phase2_final_upsample_masks
        mask_ix = torch.cat([ix for ix, mask in mask_chunks])
        coverage = torch.equal(mask_ix, torch.arange(n, device=mask_ix.device))
        low_unchanged = torch.equal(droid.video.phase2_final_lowres_before, droid.video.disps[:n])
        poses_unchanged = torch.equal(droid.video.phase2_final_poses_before, droid.video.poses[:n])
        assert coverage and low_unchanged and poses_unchanged
        assert droid.video.phase2_final_backend_steps == 12
        max_recompute_error = 0.0
        with torch.autocast(device_type="cuda", enabled=True):
            for ix, mask in mask_chunks:
                recomputed = cvx_upsample(droid.video.disps[ix].unsqueeze(-1), mask).squeeze(-1)
                error = (recomputed-droid.video.disps_up[ix]).abs().max().item()
                max_recompute_error = max(max_recompute_error, error)
        assert max_recompute_error == 0.0
        mask_logits = torch.cat([mask.squeeze(0) for ix, mask in mask_chunks]).cpu().numpy()
        assert mask_logits.shape == (n, 576, 30*s, 40*s) and np.isfinite(mask_logits).all()
        def array_sha(tensor):
            return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        save(out / "final-upsampling-validation.json", {
            "contract_version": CONTRACT_VERSION, "final_lowres_unchanged": low_unchanged,
            "final_native_poses_unchanged": poses_unchanged, "full_keyframe_mask_coverage": coverage,
            "max_abs_recompute_error": max_recompute_error, "last_backend_steps": 12,
            "mask_logits_dtype": str(mask_logits.dtype), "mask_logits_shape": list(mask_logits.shape),
            "mask_logits_sha256": hashlib.sha256(mask_logits.tobytes()).hexdigest(),
            "mask_source_indices_sha256": hashlib.sha256(key_indices.tobytes()).hexdigest(),
            "final_lowres_before_upsample_sha256": array_sha(droid.video.phase2_final_lowres_before),
            "final_lowres_after_terminate_sha256": array_sha(droid.video.disps[:n]),
            "final_native_poses_before_upsample_sha256": array_sha(droid.video.phase2_final_poses_before),
            "final_native_poses_after_terminate_sha256": array_sha(droid.video.poses[:n]),
            "recompute": "Official cvx_upsample(final video.disps, last matching learned logits), CUDA autocast enabled",
            "quality_scope": "State and numeric validity only; depth quality remains unvalidated"})
        low_disparity = droid.video.disps[:n].cpu().numpy()
        low_valid = np.isfinite(low_disparity) & (low_disparity > 0)
        low_depth = np.divide(1.0, low_disparity, out=np.zeros_like(low_disparity), where=low_valid)
        disparity = droid.video.disps_up[:n].cpu().numpy()
        valid = np.isfinite(disparity) & (disparity > 0)
        depth = np.divide(1.0, disparity, out=np.zeros_like(disparity), where=valid)
        intrinsics_full = droid.video.intrinsics[:n].cpu().numpy() * 8
        assert np.isfinite(poses).all() and np.isfinite(key_c2w).all()
        assert np.allclose(intrinsics_full, intrinsics, atol=1e-5)
        np.savez(out / "prediction.npz", poses_c2w=poses, poses_xyzw=trajectory,
                 keyframe_source_indices=key_indices, keyframe_c2w=key_c2w,
                 keyframe_final_fullres_depth=depth, keyframe_final_fullres_inverse_depth=disparity,
                 keyframe_final_fullres_valid=valid, keyframe_final_fullres_intrinsics=intrinsics_full,
                 keyframe_final_upsample_mask_logits=mask_logits,
                 keyframe_final_upsample_mask_source_indices=key_indices,
                 keyframe_final_lowres_inverse_depth=low_disparity,
                 keyframe_final_lowres_depth=low_depth, keyframe_final_lowres_valid=low_valid,
                 keyframe_final_lowres_intrinsics=intrinsics_full/8,
                 keyframe_model_bgr=np.stack([images[i][0].permute(1, 2, 0).numpy() for i in key_indices]))
        save(out / "pose-contract.json", {
            "full_vs_keyframe_max_matrix_difference": float(agreement.max()),
            "full_vs_keyframe_max_translation_difference": float(np.linalg.norm(poses[key_indices,:3,3]-key_c2w[:,:3,3],axis=1).max()),
            "within_float_tolerance_1e-5": bool(agreement.max() <= 1e-5),
            "interpretation": "Official filler motion-optimizes all supplied images, including original keyframe images; differences are preserved, never overwritten",
            "full_pose": "camera-to-world, inverse of official filler output",
            "keyframe_pose": "camera-to-world, inverse of final native video.poses"})
        assert (out / "prediction.npz").stat().st_size < s*s*1024**3
        save(out / "preprocessing.json", {"source_K": manifest["source_K_fx_fy_cx_cy"],
             "source_distortion": manifest["source_distortion"], "undistort_new_K": "same as source K",
             "resolution_scale": s, "resize_wh": [352*s, 256*s], "crop_xyxy": [16*s, 8*s, 336*s, 248*s], "model_wh": [320*s, 240*s],
             "model_intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
             "intrinsics_convention": "Official TUM scale/crop convention, not half-pixel-adjusted resize K",
             "color": "OpenCV BGR; native MotionFilter converts BGR to normalized RGB",
             "depth_scope": "Final keyframes only, native monocular scale; finite positive is numeric validity, not confidence",
             "contract_version": CONTRACT_VERSION,
             "upsampled_depth_state": "Original learned convex upsampling of final post-BA video.disps with last matching learned masks; exact GPU recompute verified",
             "official_offline_viewer_raster": {"wh": [160*s, 120*s], "rgb_and_disparity": "[..., ::2, ::2] stride, no resizing or pixel offset", "K": "model K/2 = 4*stored video.intrinsics"},
             "final_lowres_depth_state": f"Final video.disps after backend BA; same optimization state as keyframe_c2w; raster {40*s}x{30*s}, K=model_K/8",
             "keyframe_model_bgr": f"Exact undistorted resized cropped source raster, {320*s}x{240*s}, matching full-resolution depth pixel domain",
             "full_trajectory_scope": "All frames optimized by official PoseTrajectoryFiller; not per-frame tracking success"})
        result = {"status": "inference_complete", "run_id": run_id, "contract_version": CONTRACT_VERSION, "frames_processed": frame_count, "clip": clip or "fr1-room", "full_pose_count": len(poses),
             "keyframe_count": n, "elapsed_seconds": time.time()-started,
             "weights_sha256": build_info["weights_sha256"], "source_revision": REV,
             "torch_version": str(torch.__version__), "torch_cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(), "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
             "groundtruth_uploaded": False, "sensor_depth_uploaded": False,
             "scale": "uncalibrated_monocular", "groundtruth_alignment_applied": False,
             "full_poses_are_filler_output": True,
             "artifacts_sha256": {p.name: sha(p) for p in out.iterdir() if p.name != "remote-run.json"}}
    except Exception as error:
        result = {"status": "inference_failed", "error": repr(error), "elapsed_seconds": time.time()-started}
    save(out / "remote-run.json", result)
    signal.alarm(0)
    volume.commit()
    return result


def evaluate(output, dataset=DATASET):
    """Local posthoc only; no geometry is rescaled or rewritten."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from reconstruct_tum_room import associate, metric_ate, read_rows
    data = np.load(output / "prediction.npz")
    manifest = json.loads((output / "input-manifest.json").read_text())
    if not (dataset / "groundtruth.txt").exists():  # an ordinary video: nothing to measure the cameras against
        return {"evaluation_only": True, "status": "no_ground_truth", "frames": len(manifest["frames"])}
    gt = read_rows(dataset / "groundtruth.txt", 8)
    times = [float(f["timestamp_text"]) for f in manifest["frames"]]
    pairs = associate(times, [float(r[0]) for r in gt], .02)
    x = data["poses_c2w"][[i for i, j in pairs], :3, 3].astype(float)
    y = np.array([gt[j][1:4] for i, j in pairs], float)
    xc, yc = x-x.mean(0), y-y.mean(0)
    u, values, vt = np.linalg.svd(xc.T @ yc)
    correction = np.ones(3); correction[-1] = np.linalg.det(vt.T @ u.T)
    scale = float(np.sum(values*correction)/np.sum(xc**2))
    sim3 = metric_ate(scale*x, y)
    sim3.update(alignment="Sim3 fitted scale and rigid transform, posthoc evaluation only", scale_fit=True, scale=scale)
    result = {"evaluation_only": True, "groundtruth_sha256": sha(dataset / "groundtruth.txt"),
        "matched_poses": len(pairs), "input_frames": len(times), "pose_output_fraction": len(data["poses_c2w"])/len(times),
        "groundtruth_match_fraction": len(x)/len(times),
        "tracking_success_fraction": None, "full_poses_are_filler_output": True,
        "Sim3_fitted_meters_per_native_unit": sim3,
        "SE3_scale1_native_units_diagnostic": metric_ate(x, y),
        "native_geometry_rescaled": False, "native_scale": "uncalibrated_monocular",
        "whole_room_accepted": False, "dense_geometry_review": "pending",
        "associations": [{"source_index": i, "timestamp": times[i], "gt_timestamp": float(gt[j][0])} for i,j in pairs]}
    save(output / "evaluation.json", result)
    return {k: v for k,v in result.items() if k != "associations"}


def collect(output, run_id, dataset=DATASET):
    """Read persisted namespaced artifacts; never dispatches remote compute."""
    for name in ["build.json", "build.log"]:
        with (output / name).open("wb") as stream:
            for block in volume.read_file(name): stream.write(block)
    remote = "runs/"+run_id+"/result"
    for entry in volume.iterdir(remote, recursive=False):
        name = Path(entry.path).name
        with (output / name).open("wb") as stream:
            for block in volume.read_file(remote+"/"+name): stream.write(block)
    state = json.loads((output / "remote-run.json").read_text())
    if state["status"] == "inference_complete":
        assert state["run_id"] == run_id and state["contract_version"] == CONTRACT_VERSION
        for name, checksum in state["artifacts_sha256"].items():
            assert sha(output / name) == checksum
        print(json.dumps(evaluate(output, dataset), indent=2))
    return state


def clip_archive(output, clip, frames=None):
    """RGB-only archive and hash manifest of another calibrated clip; same layout as the original room upload.
    frames (START, END): only clip frames START..END-1 (one shot of an edited clip) are archived and tracked; the whole
    clip's manifest is kept beside it as clip-input-manifest.json for clip_index."""
    dataset, records = CLIPS[clip]["dataset"], []
    for line in (dataset / "rgb.txt").read_text().splitlines():
        if not line or line.startswith("#"): continue
        stamp, relative = line.split()
        assert relative.startswith("rgb/") and (dataset / relative).resolve().is_relative_to(dataset.resolve())
        records.append({"source_index": len(records), "timestamp_text": stamp, "relative_path": relative, "sha256": sha(dataset / relative)})
    manifest = lambda rows: {"frame_count": len(rows), "rgb_index_sha256": sha(dataset / "rgb.txt"),
        "source_K_fx_fy_cx_cy": CLIPS[clip]["K"], "source_distortion": CLIPS[clip]["D"], "frames": rows,
        "groundtruth_included": False, "depth_included": False}
    if frames:
        save(output / "clip-input-manifest.json", manifest(records))
        records = records[frames[0]:frames[1]]
    save(output / "input-manifest.json", manifest(records))
    with tarfile.open(output / "input-rgb.tar", "w") as archive:
        for record in records:
            archive.add(dataset / record["relative_path"], arcname=record["relative_path"], recursive=False)


def shot_to_clip(data, start, count):
    """A one-shot prediction in clip frame numbers: one pose per clip frame, keyframe indices shifted by START. A frame outside
    the shot holds the shot's first or last camera, a placeholder that is never evidence (consumers exclude those frames)."""
    import numpy as np
    at = np.clip(np.arange(count) - start, 0, len(data["poses_c2w"]) - 1)
    return {**data, "poses_c2w": data["poses_c2w"][at], "poses_xyzw": data["poses_xyzw"][at],
            "keyframe_source_indices": data["keyframe_source_indices"] + start,
            "keyframe_final_upsample_mask_source_indices": data["keyframe_final_upsample_mask_source_indices"] + start}


def clip_index(output, frames):
    """Rewrite a finished one-shot run the way every consumer reads a run (prediction.npz and input-manifest.json in clip
    frames); the remote's own prediction.npz, frames.jsonl and the shot's input-manifest.json move unchanged to shot/."""
    import numpy as np
    (output / "shot").mkdir()
    for name in ("prediction.npz", "frames.jsonl", "input-manifest.json"):
        (output / name).rename(output / "shot" / name)
    (output / "clip-input-manifest.json").rename(output / "input-manifest.json")
    count = len(json.loads((output / "input-manifest.json").read_text())["frames"])
    with np.load(output / "shot" / "prediction.npz") as data:
        assert len(data["poses_c2w"]) == frames[1] - frames[0] <= count
        np.savez(output / "prediction.npz", **shot_to_clip(dict(data), frames[0], count))
    state = json.loads((output / "run.json").read_text())
    state.update(shot_frames=list(frames), clip_indexed={"from": "shot/prediction.npz (the remote's, hash-checked by collect)", "outside_shot_poses":
                 "placeholder: the shot's first/last camera; never evidence, exclude those frames downstream", "prediction_sha256": sha(output / "prediction.npz")})
    save(output / "run.json", state)


def execute(output, run_id, reuse_run, clip="fr1-room", frames=None):
    """One GPU-only run using the already successful pinned build; the room clip also reuses its uploaded RGB archive."""
    assert run_id and all(c.isalnum() or c in "-_" for c in run_id)
    previous = json.loads((reuse_run / "run.json").read_text())
    assert previous["status"] == "inference_complete" and previous["source_revision"] == REV
    dataset = CLIPS[clip]["dataset"]
    if clip == "fr1-room":
        assert sha(reuse_run / "input-rgb.tar") == previous["archive_sha256"]
        assert sha(reuse_run / "input-manifest.json") == previous["input_manifest_sha256"]
        output.mkdir(parents=True, exist_ok=False)
        (output / "input-manifest.json").write_bytes((reuse_run / "input-manifest.json").read_bytes())
        archive_path = reuse_run / "input-rgb.tar"
    else:
        output.mkdir(parents=True, exist_ok=False)
        clip_archive(output, clip, frames)
        archive_path = output / "input-rgb.tar"
    assert not frames or clip != "fr1-room", "one-shot runs are for video clips"
    uploaded = f"{clip}-{frames[0]}-{frames[1]}" if frames else clip  # the volume folder of this upload
    manifest = json.loads((output / "input-manifest.json").read_text())
    assert not manifest["groundtruth_included"] and not manifest["depth_included"]
    assert manifest["rgb_index_sha256"] == sha(dataset / "rgb.txt")
    for record in manifest["frames"]:
        assert sha(dataset / record["relative_path"]) == record["sha256"]
    (output / "runner-at-execution.py").write_bytes(Path(__file__).read_bytes())
    save(output / "reservation.json", {"run_id": run_id, "max_additional_usd": 1.5,
        "inside_existing_droid_reservation_usd": 3, "total_root_reserved_usd": 27.36,
        "cpu_build_max_seconds": 0, "gpu_submission_max_seconds": 900,
        "gpu_attempts": 1, "gpu_retries": 0, "rates_usd_hour": {"A100-40GB": 2.10, "cpu_core": .0473, "ram_GiB": .008}})
    state = {"status": "preparing", "run_id": run_id, "clip": clip, "contract_version": CONTRACT_VERSION,
        "source_revision": REV, "script_sha256": sha(Path(__file__)),
        "archive_sha256": sha(archive_path), "input_manifest_sha256": sha(output / "input-manifest.json"),
        "cached_build_sha256": sha(reuse_run / "build.json"), "reuse_successful_run": str(reuse_run),
        "reuse_successful_run_sha256": sha(reuse_run / "run.json"),
        "input_archive_path": str(archive_path), "cpu_build_invoked": False,
        "remote_namespace": "runs/"+run_id, "groundtruth_uploaded": False, "sensor_depth_uploaded": False}
    save(output / "run.json", state)
    with app.run():
        if clip != "fr1-room":  # upload is outside the GPU deadline
            with volume.batch_upload(force=True) as batch:
                batch.put_file(archive_path, f"/clips/{uploaded}/input-rgb.tar")
                batch.put_file(output / "input-manifest.json", f"/clips/{uploaded}/input-manifest.json")
        deadline = time.time()+900
        state.update(status="gpu_submitting", app_id=app.app_id, gpu_deadline_unix=deadline)
        save(output / "run.json", state)
        call = infer.spawn(run_id, deadline, state["archive_sha256"], state["input_manifest_sha256"], state["cached_build_sha256"],
                           clip=None if clip == "fr1-room" else uploaded)
        state.update(status="gpu_running", gpu_call_id=call.object_id)
        save(output / "run.json", state); print(json.dumps(state), flush=True)
        try:
            result = call.get(timeout=max(1, deadline-time.time()))
        except BaseException:
            call.cancel(terminate_containers=True)
            state["status"] = "gpu_result_unknown_or_timeout"; save(output / "run.json", state)
            raise
        print(json.dumps(result), flush=True)
        state["status"] = result["status"]; save(output / "run.json", state)
        collected = collect(output, run_id, dataset)
        state["remote_run_sha256"] = sha(output / "remote-run.json")
        state["status"] = collected["status"]; save(output / "run.json", state)
    if frames and state["status"] == "inference_complete":
        clip_index(output, frames)
    return state


def self_check():
    import cv2
    import numpy as np
    image = np.zeros((480, 640, 3), np.uint8)
    image[100:300, 150:350] = [20, 50, 100]
    result, K = prepare_image(image, {"source_K_fx_fy_cx_cy": SOURCE_K, "source_distortion": SOURCE_D})
    fx,fy,cx,cy=SOURCE_K
    official = cv2.undistort(image, np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]), np.array(SOURCE_D))
    official = cv2.resize(official, (352,256))[8:-8,16:-16]
    assert np.array_equal(result,official) and result.shape==(240,320,3)
    assert np.allclose(K,[284.5185244,275.450248,159.253672,128.1674608])
    assert 1400 >= 1362+16
    # Exact pinned source test catches wrong anchors and preserves every BA invocation.
    import urllib.request
    source = urllib.request.urlopen("https://raw.githubusercontent.com/princeton-vl/DROID-SLAM/"+REV+"/droid_slam/factor_graph.py").read().decode()
    patched = patch_lowmem(source)
    import ast
    def calls(text):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(text))
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "ba"]
    assert calls(source) == calls(patched)
    lowmem = patched.split("    def update_lowmem", 1)[1].split("    def add_neighborhood_factors", 1)[0]
    assert lowmem.index("self.video.ba(") < lowmem.index("self.video.upsample(")
    assert "self.video.phase2_final_upsample_masks" in lowmem
    shot = {"poses_c2w": np.arange(3.), "poses_xyzw": np.arange(3.), "keyframe_source_indices": np.array([0, 2]), "keyframe_final_upsample_mask_source_indices": np.array([0, 2])}
    clip = shot_to_clip(shot, 2, 6)
    assert clip["poses_c2w"].tolist() == [0, 0, 0, 1, 2, 2] and clip["keyframe_source_indices"].tolist() == [2, 4], "one-shot run in clip frames"
    print("DROID raster/calibration/capacity and pinned BA-before-upsampling self-check passed; no GPU invoked")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["execute","collect","self-check"])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--reuse-build-from", type=Path)
    parser.add_argument("--clip", choices=sorted(CLIPS), default="fr1-room")
    parser.add_argument("--frames", type=lambda s: tuple(int(v) for v in s.split(":")), metavar="START:END",
                        help="track only clip frames START..END-1 (one continuous shot of an edited clip); the run is then rewritten in clip frame numbers")
    args=parser.parse_args()
    if args.mode=="self-check": self_check()
    else:
        assert args.run_id and args.output, "Explicit --run-id and --output required"
        if args.mode=="collect":
            print(json.dumps(collect(args.output, args.run_id, CLIPS[args.clip]["dataset"]),indent=2))
            if args.frames: clip_index(args.output, args.frames)
        else:
            assert args.reuse_build_from, "This entry reuses a successful build; --reuse-build-from required"
            print(json.dumps(execute(args.output, args.run_id, args.reuse_build_from, args.clip, args.frames),indent=2))
