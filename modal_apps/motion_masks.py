"""Class-agnostic moving-object masks from the cameras a run already has: image motion no static point could produce.

A static point seen from two posed cameras must land, in the second image, on a segment: between where it would be at
infinite depth and where it would be at the nearest plausible depth. How far the measured optical flow ends from that
segment is motion the camera cannot explain, whatever the point's depth. No depth map enters, so a depth model's bias
can neither fake nor hide motion, and a camera that only rotates is handled (the segment shrinks to a point).
Residual blobs become SAM 2.1 box prompts (sam2_everything.box_masks), so the result is object masks, not blobs.
Flow is torchvision's pretrained RAFT-large (BSD-3), both directions; pixels failing the forward-backward check are
occlusions and do not vote. Two GPU calls (flow, boxes), retries 0.
An explicitly fixed camera needs no poses or depth: its residual is optical-flow length.

  python modal_apps/motion_masks.py --droid-run RUN --output NEW_DIR [--every 10] [--gap 8] [--reference MASK_DIR]
  python modal_apps/motion_masks.py --video CLIP --fixed-camera --output NEW_DIR [--every 10] [--gap 8]
  python modal_apps/motion_masks.py --self-check
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import sam2_everything  # noqa: E402  its image already holds torch, torchvision and OpenCV; its box prompts finish the masks

RESIDUAL_PX, NOISE_MADS, CYCLE_PX = 2., 5., 1.5  # motion floor on the 640x480 raster; a frame's own flow/pose noise raises it; forward-backward agreement
MIN_BLOB, BLOB_IN_MASK = .003, .3  # blob area as a share of the frame; share of a SAM mask that must itself be moving
MAX_VIDEO_BYTES = 512 * 1024 ** 2
app = sam2_everything.app
image = sam2_everything.image.add_local_python_source("sam2_everything")  # this file imports it at the top, in the container too


@app.function(image=image, gpu="L4", volumes={"/cache": sam2_everything.volume}, timeout=600, retries=0, max_containers=1)
def flow_remote(pairs):
    """pairs: [(key, jpg_a, jpg_b)] -> [(key, npz of forward flow a->b (float16) and its forward-backward error)]."""
    import os
    os.environ["TORCH_HOME"] = "/cache/torch"
    import cv2
    import torch
    from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    model = raft_large(weights=Raft_Large_Weights.DEFAULT).eval().cuda()
    prepare = lambda data: torch.from_numpy(cv2.cvtColor(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float().cuda() / 127.5 - 1
    results = []
    for key, first, second in pairs:
        a, b = prepare(first), prepare(second)
        with torch.inference_mode():
            forward, backward = model(a, b)[-1][0], model(b, a)[-1][0]
            h, w = forward.shape[1:]
            y, x = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
            grid = torch.stack([(x + forward[0]) / (w - 1) * 2 - 1, (y + forward[1]) / (h - 1) * 2 - 1], -1)[None]
            cycle = (forward + torch.nn.functional.grid_sample(backward[None], grid, align_corners=True)[0]).norm(dim=0)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, flow=forward.permute(1, 2, 0).half().cpu().numpy(), cycle=cycle.half().cpu().numpy())
        results.append((key, buffer.getvalue()))
    return results


def fixed_video_frames(path):
    """Source-order BGR frames, uniformly resized to <=640px and padded to multiples of eight; no camera estimate."""
    import cv2
    with path.open('rb') as stream:
        source_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    cap = cv2.VideoCapture(str(path))
    frames, times, shape, raster_hash = [], [], None, hashlib.sha256()
    # ponytail: buffer at most 512 MiB of <=640px frames; stream selected frames to support longer clips.
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        expected_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError('Video must be readable with a positive frame rate')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if shape is None:
                shape = frame.shape
                h, w = shape[:2]
                scale = min(1., 640 / max(h, w))
                width, height = max(1, round(w * scale)), max(1, round(h * scale))
                right, bottom = -width % 8, -height % 8
            elif frame.shape != shape:
                raise ValueError('Video changes pixel dimensions')
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) if scale < 1 else frame
            padded = cv2.copyMakeBorder(small, 0, bottom, 0, right, cv2.BORDER_REPLICATE)
            if (len(frames) + 1) * padded.nbytes > MAX_VIDEO_BYTES:
                raise ValueError('Decoded video exceeds the 512 MiB memory limit')
            frames.append(padded)
            times.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000)
            raster_hash.update(padded.tobytes())
    finally:
        cap.release()
    if not frames:
        raise ValueError('Video has no decoded frames')
    if np.isfinite(expected_count) and expected_count > 0 and len(frames) != int(expected_count):
        raise ValueError('Decoded frame count differs from the video frame count; input may be truncated')
    return frames, {'video': str(path.resolve()), 'sha256': source_sha, 'camera_mode': 'fixed_asserted',
                    'frame_count': len(frames), 'fps': fps, 'frame_timestamps_seconds': times,
                    'source_size_wh': [w, h], 'resized_size_wh': [width, height],
                    'raster_size_wh': [width + right, height + bottom], 'resize_factors_xy': [width / w, height / h],
                    'padding_ltrb': [0, 0, right, bottom], 'padding_mode': 'replicate',
                    'raster_sha256': raster_hash.hexdigest()}


def residual(flow, k=None, c2w_a=None, c2w_b=None, nearest=None, *, fixed_camera=False):
    """Pixels between each flow endpoint and the segment a static point of depth >= nearest could reach. NaN where it cannot be said."""
    if fixed_camera:
        return np.linalg.norm(flow, axis=-1)
    h, w = flow.shape[:2]
    fx, fy, cx, cy = k
    v, u = np.indices((h, w), float)
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones((h, w))], -1)
    b_from_a = np.linalg.inv(c2w_b) @ c2w_a
    turned = rays @ b_from_a[:3, :3].T
    def project(points):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(points[..., 2:] > 1e-6, np.stack([points[..., 0] / points[..., 2] * fx + cx, points[..., 1] / points[..., 2] * fy + cy], -1), np.nan)
    far, near = project(turned), project(turned * nearest + b_from_a[:3, 3])
    end = np.stack([u, v], -1) + flow
    along = near - far
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.clip(np.nan_to_num(((end - far) * along).sum(-1) / (along * along).sum(-1)), 0, 1)
    return np.linalg.norm(end - (far + t[..., None] * along), axis=-1)


def blobs(distance, cycle):
    """Connected motion regions, adaptive floor, and the trusted moving pixels before morphology."""
    import cv2
    trusted = np.isfinite(distance) & (cycle <= CYCLE_PX)
    calm = distance[trusted]
    floor = max(RESIDUAL_PX, float(np.median(calm) + NOISE_MADS * 1.4826 * np.median(np.abs(calm - np.median(calm))))) if calm.size else np.inf
    trusted_moving = trusted & (distance > floor)
    moving = cv2.morphologyEx(trusted_moving.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(moving)
    keep = [n for n in range(1, count) if stats[n, cv2.CC_STAT_AREA] >= MIN_BLOB * distance.size]
    return [(labels == n, [int(stats[n, 0]), int(stats[n, 1]), int(stats[n, 0] + stats[n, 2]), int(stats[n, 1] + stats[n, 3])]) for n in keep], floor, trusted_moving


def run(args):
    import cv2
    if bool(args.droid_run) == bool(args.video) or args.fixed_camera != bool(args.video):
        raise ValueError('Choose --droid-run or --video with --fixed-camera')
    if args.video and args.reference:
        raise ValueError('--reference is supported only with --droid-run')
    if args.every <= 0 or args.gap <= 0:
        raise ValueError('--every and --gap must be positive')
    source, nearest = None, None
    if args.video:
        images, source = fixed_video_frames(args.video)
        frame_count = len(images)
        picture = lambda index: (images[index], None)
    else:
        import mono_room as M
        M.use_clip(args.droid_run)
        manifest = json.loads((args.droid_run / "input-manifest.json").read_text())
        data = np.load(args.droid_run / "prediction.npz")
        poses, depth = data["poses_c2w"].astype(float), data["keyframe_final_fullres_depth"]
        nearest = float(np.percentile(depth[depth > 0], 1))  # one loose bound for the whole clip, in the cameras' own units: not a per-pixel depth
        frame_count = len(poses)
        picture = lambda index: M.prepare_image(cv2.imread(str(M.DATASET / manifest["frames"][index]["relative_path"])), M.CALIBRATION, 2)
    if args.gap >= frame_count:
        raise ValueError('--gap must leave at least one source frame pair')
    args.output.mkdir(parents=True, exist_ok=False)
    firsts = [i for i in range(0, frame_count - args.gap, args.every)]
    frames = {i: picture(i) for i in sorted(set(firsts) | {i + args.gap for i in firsts})}
    jpg = lambda image: cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    state = {"status": "gpu_running", "flow": "torchvision raft_large DEFAULT", "gap_frames": args.gap, "every": args.every, "pairs": len(firsts), "nearest_depth_native": nearest,
             "gpu": "L4", "timeout_s": 600, "retries": 0, "cameras": str(args.droid_run) if args.droid_run else None, "reference": str(args.reference) if args.reference else None}
    if source:
        state['source'] = source
    (args.output / f"motion-{int(time.time())}.json").write_text(json.dumps(state, indent=1))
    started = time.time()
    with app.run():
        answers = dict(flow_remote.remote([(i, jpg(frames[i][0]), jpg(frames[i + args.gap][0])) for i in firsts]))
    state["flow_wall_seconds"] = time.time() - started
    found, requests = {}, {}
    for i in firsts:
        answer = np.load(io.BytesIO(answers[i]))
        if answer['flow'].shape != (*frames[i][0].shape[:2], 2) or answer['cycle'].shape != frames[i][0].shape[:2]:
            raise ValueError('Optical flow raster differs from its source frames')
        distance = (residual(answer['flow'].astype(float), fixed_camera=True) if args.video else
                    residual(answer['flow'].astype(float), frames[i][1], poses[i], poses[i + args.gap], nearest))
        height, width = distance.shape
        content_width, content_height = source['resized_size_wh'] if source else (width, height)
        y, x = np.indices(distance.shape)
        end_x, end_y = x + answer['flow'][..., 0], y + answer['flow'][..., 1]
        valid = ((x < content_width) & (y < content_height) & (end_x >= 0) & (end_x <= content_width - 1)
                 & (end_y >= 0) & (end_y <= content_height - 1))
        found[i], floor, trusted_moving = blobs(distance, np.where(valid, answer['cycle'].astype(float), np.inf))
        requests.update({(i, n): (frames[i][0], box) for n, (_, box) in enumerate(found[i])})
        np.savez_compressed(args.output / f"{i:05d}-residual.npz", residual=distance.astype(np.float16), floor=floor, trusted_moving=trusted_moving)
    started = time.time()
    masks = sam2_everything.box_masks(requests, work_width=640) if requests else {}
    state["box_wall_seconds"] = time.time() - started
    rows = []
    for i in firsts:
        moving = np.zeros(frames[i][0].shape[:2], bool)
        for n, (blob, _) in enumerate(found[i]):
            mask = masks[(i, n)][0]
            if (mask & blob).sum() >= BLOB_IN_MASK * mask.sum():  # a box on a false blob returns the desk under it: an object that moves is mostly moving
                moving |= mask
        cv2.imwrite(str(args.output / f"{i:05d}-moving.png"), moving.astype(np.uint8) * 255)
        row = {"frame": i, "blobs": len(found[i]), "moving_share": float(moving.mean())}
        if args.reference:
            known, later = M.moving_mask(args.reference, i), M.moving_mask(args.reference, i + args.gap)
            row.update(reference_share=float(known.mean()), reference_moved=bool((known | later).sum() and (known & later).sum() / (known | later).sum() < .8),
                       iou=float((moving & known).sum() / max((moving | known).sum(), 1)), hit=int((moving & known).sum()), said=int(moving.sum()), there=int(known.sum()))
        rows.append(row)
    summary = {}
    if args.reference:
        for name, chosen in (("reference_moved", [r for r in rows if r["there"] and r["reference_moved"]]), ("reference_still", [r for r in rows if r["there"] and not r["reference_moved"]]),
                             ("no_reference", [r for r in rows if not r["there"]])):
            summary[name] = {"frames": len(chosen), "median_iou": float(np.median([r["iou"] for r in chosen])) if chosen and name != "no_reference" else None,
                             "pixel_recall": sum(r["hit"] for r in chosen) / max(sum(r["there"] for r in chosen), 1), "pixel_precision": sum(r["hit"] for r in chosen) / max(sum(r["said"] for r in chosen), 1),
                             "moving_share_of_frame": float(np.mean([r["moving_share"] for r in chosen])) if chosen else None}
    state.update(status="complete", frames=rows, summary=summary, rule=f"residual > max({RESIDUAL_PX}px, median+{NOISE_MADS} MAD) where forward-backward flow agrees within {CYCLE_PX}px; "
                 f"blobs >= {MIN_BLOB:.1%} of the frame; SAM 2.1 box mask kept if >= {BLOB_IN_MASK:.0%} of it is moving")
    (args.output / "motion.json").write_text(json.dumps(state, indent=1))
    print(json.dumps({"pairs": len(firsts), "flow_s": round(state["flow_wall_seconds"]), "boxes": len(requests), "box_s": round(state["box_wall_seconds"]), "summary": summary}, indent=1))


def self_check():
    rng = np.random.default_rng(0)
    k, h, w = np.array([500., 500., 320., 240.]), 480, 640
    angle = .05
    b = np.eye(4); b[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]; b[:3, 3] = [.2, 0, .05]
    depth = rng.uniform(1, 6, (h, w))
    v, u = np.indices((h, w), float)
    points = np.stack([(u - k[2]) / k[0], (v - k[3]) / k[1], np.ones((h, w))], -1) * depth[..., None]
    def seen_from_b(world):
        local = (world - b[:3, 3]) @ b[:3, :3]
        return np.stack([local[..., 0] / local[..., 2] * k[0] + k[2], local[..., 1] / local[..., 2] * k[1] + k[3]], -1)
    flow = seen_from_b(points) - np.stack([u, v], -1)
    assert np.nanmax(residual(flow, k, np.eye(4), b, 1.)) < 1e-6, "a static scene of any depth has no residual"
    moved = points.copy(); moved[200:280, 300:360] += [0, .25, 0]  # a box-sized region slides sideways to the camera's motion
    distance = residual(seen_from_b(moved) - np.stack([u, v], -1), k, np.eye(4), b, 1.)
    assert np.nanmin(distance[200:280, 300:360]) > 5 and np.nanmax(np.delete(distance, np.s_[200:280], 0)) < 1e-6, "only what moved stands out"
    turn = np.eye(4); turn[:3, :3] = b[:3, :3]  # a camera that only turns: the segment is a point, depth still plays no part
    local = points @ turn[:3, :3]
    spun = np.stack([local[..., 0] / local[..., 2] * k[0] + k[2], local[..., 1] / local[..., 2] * k[1] + k[3]], -1) - np.stack([u, v], -1)
    assert np.nanmax(residual(spun, k, np.eye(4), turn, 1.)) < 1e-6, "pure rotation is explained too"
    found, floor, trusted = blobs(distance, np.zeros((h, w)))
    assert len(found) == 1 and floor == RESIDUAL_PX and found[0][1] == [300, 200, 360, 280], found
    assert trusted.sum() == 80 * 60
    print("motion mask check passed: static depth-free, moved region found, pure rotation explained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--droid-run", type=Path)
    inputs.add_argument("--video", type=Path, help="original video, requires --fixed-camera; resized longest side <=640 and padded to multiples of eight")
    parser.add_argument("--fixed-camera", action="store_true", help="explicitly assert that --video has no camera motion; do not estimate poses or depth")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--every", type=int, default=10, help="start a pair at every Nth source frame")
    parser.add_argument("--gap", type=int, default=8, help="frames between the two images of a pair: slow things need a wider one")
    parser.add_argument("--reference", type=Path, help="SOURCEINDEX-*.png masks of things known to move, to measure against (never an input)")
    a = parser.parse_args()
    self_check() if a.self_check else run(a)
