"""One bounded, pretrained official DROID-SLAM RGB experiment (no remote GT).

Run once: python modal_apps/droid_room.py execute --output ABS_NEW_RUN
Recover without resubmitting: python modal_apps/droid_room.py collect --output ABS_RUN
Check locally: python modal_apps/droid_room.py self-check
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import modal

REV = "2dfd39f0dcad44012ca7bbb8aa70b55edbfa9c99"
LIETORCH = "7f687644fcea81ab337831749445224503c1290d"
SCATTER = "6cf77c420f837a427b0d57965e414d68c1bd89ec"
WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh&export=download&confirm=t"
WEIGHTS_BYTES = 16061701
ART = Path("/Users/adam/Desktop/panoptes-public/research-notes/phase2")
DATASET = ART / "data/rgbd_dataset_freiburg1_room"
SOURCE_K = [517.306408, 516.469215, 318.643040, 255.313989]
SOURCE_D = [0.262383, -0.953104, -0.005358, 0.002628, 1.163314]
RUN_ID = "droid-fr1-room-001"
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


@app.function(image=image, cpu=(4, 4), memory=(16384, 16384), timeout=1750,
              startup_timeout=45, retries=0, max_containers=1, scaledown_window=2,
              volumes={"/artifact": volume})
def build(deadline):
    """CPU-only build; CUDA toolkit cross-compiles extensions without a GPU."""
    import shutil
    import urllib.request
    started = time.time()
    ROOT.mkdir(exist_ok=True)
    logfile = ROOT / "build.log"
    env = dict(os.environ, PYTHONPATH=str(ROOT / "site"), CUDA_HOME="/usr/local/cuda",
               FORCE_CUDA="1", TORCH_CUDA_ARCH_LIST="8.0", MAX_JOBS="4",
               OMP_NUM_THREADS="4", MPLBACKEND="Agg")

    def command(argv, cwd=None):
        remaining = min(1700, int(deadline-time.time())-10)
        if remaining <= 0:
            raise TimeoutError("CPU build deadline exhausted")
        with logfile.open("a") as log:
            log.write(json.dumps(argv) + "\n"); log.flush()
            subprocess.run(argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=remaining)

    try:
        if not shutil.which("git"):
            command(["apt-get", "update"])
            command(["apt-get", "install", "-y", "git", "build-essential"])
        code = ROOT / "DROID-SLAM"
        if not code.exists():
            command(["git", "clone", "--no-checkout", "https://github.com/princeton-vl/DROID-SLAM.git", str(code)])
        command(["git", "checkout", REV], code)
        command(["git", "submodule", "update", "--init", "--recursive"], code)
        for folder, revision in [(code, REV), (code / "thirdparty/lietorch", LIETORCH),
                                 (code / "thirdparty/pytorch_scatter", SCATTER)]:
            assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=folder, text=True).strip() == revision
        command([sys.executable, "-m", "pip", "install", "--target", str(ROOT / "site"),
                 "setuptools==75.8.0", "wheel", "ninja", "numpy==1.26.4", "scipy==1.15.2",
                 "opencv-python-headless==4.11.0.86", "matplotlib==3.10.1", "tqdm==4.67.1"])
        for project in [code / "thirdparty/lietorch", code / "thirdparty/pytorch_scatter", code]:
            command([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps",
                     "--target", str(ROOT / "site"), str(project)])
        weight = ROOT / "droid.pth"
        if not weight.exists():
            with urllib.request.urlopen(WEIGHTS_URL, timeout=60) as response, weight.open("wb") as dest:
                shutil.copyfileobj(response, dest)
        assert weight.stat().st_size == WEIGHTS_BYTES
        command([sys.executable, "-c", "import torch,lietorch,droid_backends,torch_scatter; "
                 "assert torch.__version__.startswith('2.7.0'); "
                 "assert torch.version.cuda.startswith('12.6'); "
                 "w=torch.load('/artifact/droid.pth',map_location='cpu',weights_only=True); "
                 "assert isinstance(w,dict) and len(w)>50; print(torch.__version__,torch.version.cuda,len(w))"])
        result = {"status": "cpu_build_complete", "source_revision": REV,
                  "lietorch_revision": LIETORCH, "scatter_revision": SCATTER,
                  "weights_sha256": sha(weight), "elapsed_seconds": time.time()-started,
                  "python": sys.version, "gpu_allocated": False}
    except Exception as error:
        result = {"status": "cpu_build_failed", "error": repr(error), "elapsed_seconds": time.time()-started}
    save(ROOT / "build.json", result)
    volume.commit()
    return result


def prepare_image(image, calibration):
    """Official TUM raster operations, full-precision source calibration."""
    import cv2
    import numpy as np
    fx, fy, cx, cy = calibration["source_K_fx_fy_cx_cy"]
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    image = cv2.undistort(image, K, np.asarray(calibration["source_distortion"]))
    image = cv2.resize(image, (352, 256))[8:-8, 16:-16]
    intrinsics = np.array([fx*352/640, fy*256/480, cx*352/640-16, cy*256/480-8], np.float32)
    return image, intrinsics


@app.function(image=image, gpu="A100-40GB", cpu=(4, 4), memory=(16384, 16384),
              timeout=840, startup_timeout=45, retries=0, max_containers=1,
              min_containers=0, scaledown_window=2, volumes={"/artifact": volume})
def infer(deadline, expected_archive_sha, expected_manifest_sha):
    # Platform preemption can restart inputs despite retries=0. Never re-execute inference.
    if not attempts.put(RUN_ID, {"claimed_at": time.time()}, skip_if_exists=True):
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
    out = ROOT / "result"
    out.mkdir(exist_ok=False)
    save(out / "remote-run.json", {"status": "gpu_running", "started_unix": started})
    volume.commit()
    try:
        sys.path[:0] = [str(ROOT / "site"), str(ROOT / "DROID-SLAM/droid_slam")]
        os.environ["MPLBACKEND"] = "Agg"
        import cv2
        import numpy as np
        import torch
        import lietorch
        from droid import Droid
        torch.multiprocessing.set_start_method("spawn", force=True)
        build_info = json.loads((ROOT / "build.json").read_text())
        assert build_info["status"] == "cpu_build_complete"
        assert sha(ROOT / "droid.pth") == build_info["weights_sha256"]
        assert sha(ROOT / "input-rgb.tar") == expected_archive_sha
        assert sha(ROOT / "input-manifest.json") == expected_manifest_sha
        manifest = json.loads((ROOT / "input-manifest.json").read_text())
        assert manifest["frame_count"] == 1362 and len(manifest["frames"]) == 1362
        input_dir = Path("/tmp/droid-room-rgb"); input_dir.mkdir()
        with tarfile.open(ROOT / "input-rgb.tar") as archive:
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
            canonical, intrinsics = prepare_image(bgr, manifest)
            image_hashes.append(hashlib.sha256(canonical.tobytes()).hexdigest())
            images.append(torch.from_numpy(canonical.copy()).permute(2, 0, 1)[None])
        # Official test_tum defaults; capacity covers all inputs plus filler batch16.
        args = argparse.Namespace(weights=str(ROOT / "droid.pth"), buffer=1400, image_size=[240, 320],
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
        assert trajectory.shape == (1362, 7) and np.isfinite(trajectory).all()
        poses = lietorch.SE3(torch.as_tensor(trajectory, device="cuda")).matrix().cpu().numpy()
        n = droid.video.counter.value
        indices = droid.video.tstamp[:n].cpu().numpy()
        assert np.array_equal(indices, np.round(indices)) and np.all(np.diff(indices)>0)
        key_c2w = lietorch.SE3(droid.video.poses[:n]).inv().matrix().cpu().numpy()
        key_indices = indices.astype(np.int64)
        agreement = np.abs(poses[key_indices] - key_c2w)
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
                 keyframe_depth_native=depth, keyframe_inverse_depth_native=disparity,
                 keyframe_depth_valid=valid, keyframe_intrinsics_fx_fy_cx_cy=intrinsics_full,
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
        assert (out / "prediction.npz").stat().st_size < 1024**3
        save(out / "preprocessing.json", {"source_K": manifest["source_K_fx_fy_cx_cy"],
             "source_distortion": manifest["source_distortion"], "undistort_new_K": "same as source K",
             "resize_wh": [352, 256], "crop_xyxy": [16, 8, 336, 248], "model_wh": [320, 240],
             "model_intrinsics_fx_fy_cx_cy": intrinsics.tolist(),
             "intrinsics_convention": "Official TUM scale/crop convention, not half-pixel-adjusted resize K",
             "color": "OpenCV BGR; native MotionFilter converts BGR to normalized RGB",
             "depth_scope": "Final keyframes only, native monocular scale; finite positive is numeric validity, not confidence",
             "upsampled_depth_state": "Official disps_up snapshot before last low-memory backend BA; not guaranteed identical optimization state as final keyframe pose",
             "final_lowres_depth_state": "Final video.disps after backend BA; same optimization state as keyframe_c2w; raster 40x30, K=model_K/8",
             "keyframe_model_bgr": "Exact undistorted resized cropped source raster, 320x240, matching full-resolution depth pixel domain",
             "full_trajectory_scope": "All frames optimized by official PoseTrajectoryFiller; not per-frame tracking success"})
        result = {"status": "inference_complete", "frames_processed": 1362, "full_pose_count": len(poses),
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


def evaluate(output):
    """Local posthoc only; no geometry is rescaled or rewritten."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from reconstruct_tum_room import associate, metric_ate, read_rows
    data = np.load(output / "prediction.npz")
    manifest = json.loads((output / "input-manifest.json").read_text())
    gt = read_rows(DATASET / "groundtruth.txt", 8)
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
    result = {"evaluation_only": True, "groundtruth_sha256": sha(DATASET / "groundtruth.txt"),
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


def collect(output):
    """Read existing persisted artifacts; this never invokes either remote function."""
    for prefix in ["build.json", "build.log"]:
        with (output / prefix).open("wb") as stream:
            for block in volume.read_file(prefix): stream.write(block)
    entries = list(volume.iterdir("result", recursive=False))
    for entry in entries:
        name = Path(entry.path).name
        with (output / name).open("wb") as stream:
            for block in volume.read_file("result/"+name): stream.write(block)
    state = json.loads((output / "remote-run.json").read_text())
    if state["status"] == "inference_complete":
        for name, checksum in state["artifacts_sha256"].items():
            assert sha(output / name) == checksum
        print(json.dumps(evaluate(output), indent=2))
    return state


def execute(output):
    output.mkdir(parents=True, exist_ok=False)
    save(output / "reservation.json", {"run_id": RUN_ID, "max_additional_usd": 3,
        "prior_root_reservation_usd": 24.36, "total_reserved_usd": 27.36,
        "cpu_build_max_seconds": 1800, "gpu_submission_max_seconds": 900,
        "gpu_attempts": 1, "gpu_retries": 0, "rates_usd_hour": {"A100-40GB": 2.10, "cpu_core": .0473, "ram_GiB": .008}})
    records = []
    for line in (DATASET / "rgb.txt").read_text().splitlines():
        if not line or line.startswith("#"): continue
        stamp, relative = line.split()
        path = DATASET / relative
        assert relative.startswith("rgb/") and path.resolve().is_relative_to(DATASET)
        records.append({"source_index": len(records), "timestamp_text": stamp,
                        "relative_path": relative, "sha256": sha(path)})
    assert len(records) == 1362
    manifest = {"frame_count": len(records), "rgb_index_sha256": sha(DATASET / "rgb.txt"),
        "source_K_fx_fy_cx_cy": SOURCE_K, "source_distortion": SOURCE_D, "frames": records,
        "groundtruth_included": False, "depth_included": False}
    save(output / "input-manifest.json", manifest)
    with tarfile.open(output / "input-rgb.tar", "w") as archive:
        for record in records:
            archive.add(DATASET / record["relative_path"], arcname=record["relative_path"], recursive=False)
    (output / "runner-at-execution.py").write_bytes(Path(__file__).read_bytes())
    state = {"status": "preparing", "run_id": RUN_ID, "source_revision": REV,
             "script_sha256": sha(Path(__file__)), "archive_sha256": sha(output / "input-rgb.tar"),
             "input_manifest_sha256": sha(output / "input-manifest.json")}
    save(output / "run.json", state)
    with app.run():
        state["app_id"] = app.app_id
        cpu_deadline = time.time()+1800
        call = build.spawn(cpu_deadline)
        state.update(status="cpu_build_running", cpu_call_id=call.object_id, cpu_deadline_unix=cpu_deadline)
        save(output / "run.json", state); print(json.dumps(state), flush=True)
        try:
            built = call.get(timeout=max(1, cpu_deadline-time.time()))
        except BaseException:
            call.cancel(terminate_containers=True)
            state["status"] = "cpu_result_unknown_or_timeout"; save(output / "run.json", state)
            raise
        save(output / "build-result.json", built)
        if built["status"] != "cpu_build_complete":
            state["status"] = "cpu_build_failed"; save(output / "run.json", state)
            for name in ["build.json", "build.log"]:
                with (output/name).open("wb") as stream:
                    for block in volume.read_file(name): stream.write(block)
            raise RuntimeError(built)
        print(json.dumps(built), flush=True)
        state["status"] = "uploading_verified_rgb"; save(output / "run.json", state)
        with volume.batch_upload() as batch:
            batch.put_file(output / "input-rgb.tar", "/input-rgb.tar")
            batch.put_file(output / "input-manifest.json", "/input-manifest.json")
        deadline = time.time()+900
        state.update(status="gpu_submitting", gpu_deadline_unix=deadline)
        save(output / "run.json", state)
        call = infer.spawn(deadline, state["archive_sha256"], state["input_manifest_sha256"])
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
        collect(output)
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
    print("DROID source raster/calibration/capacity self-check passed; no GPU invoked")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["execute","collect","self-check"])
    parser.add_argument("--output", type=Path, default=ART / "runs" / RUN_ID)
    args=parser.parse_args()
    if args.mode=="self-check": self_check()
    elif args.mode=="collect": print(json.dumps(collect(args.output),indent=2))
    else: print(json.dumps(execute(args.output),indent=2))
