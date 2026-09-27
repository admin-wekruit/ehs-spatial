"""Warm per-frame LingBot-Map probe: load once, feed frames one at a time; latency, GPU memory, drift.

The plain-video fallback of the live tier (docs/phase2/STREAMING-PLAN.md §10, M0). Only the source video
goes up; per-frame poses, timings and memory samples come back. The weights licence is undeclared, so
nothing derived from them is redistributed.
  stream   the official inference_streaming loop body with one KV cache for the whole run (64-frame
           sliding window + 8 scale frames). Its 3D RoPE table must cover every keyframe, so max_frame_num
           is raised to fit the run; the official 1024 would stop the stream at keyframe 1024.
  windowed the official inference_windowed fixed-interval schedule (window = 64 keyframes), driven frame
           by frame: a full window restarts the KV cache on its last 8 frames (the new scale block) and is
           chained by the model's own _pairwise_alignment Sim3, so positions and memory reset per window.
The shot plays forward, backward, forward... so every pass sees the same source frames with no jump at the
turn: pass p against pass 0 is drift, not a failed relocalisation.

  modal run modal_apps/lingbot_stream_probe.py --mode windowed --out RUN_DIR [--gpu L4 --frames 300]
  python modal_apps/lingbot_stream_probe.py --analyze RUN_DIR
  python modal_apps/lingbot_stream_probe.py --self-check
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lingbot_room import image, volume, digest, save, REV, WEIGHTS_REV, WEIGHTS_SHA
import modal

CLIP = Path('/Users/adam/Desktop/panoptes-public/research-notes/phase2/data/clips/samsclub-337-a2')
SCALE, WINDOW = 8, 64  # official demo defaults: scale frames, KV window (streaming) / keyframes per window (windowed)
app = modal.App('panoptes-lingbot-stream-probe')
image = image.add_local_file(Path(__file__).with_name('lingbot_room.py'), '/root/lingbot_room.py')


def schedule(n_src, stride, count):
    """(source frame, pass) of each fed frame: the shot forward, then backward, and so on."""
    ids, out, p = list(range(0, n_src, stride)), [], 0
    while len(out) < count:
        out += [(s, p) for s in (ids if p % 2 == 0 else ids[::-1])]
        p += 1
    return out[:count]


def window_frames(keyframe_interval):
    """Actual frames per window, as inference_windowed counts them (window_size counts keyframes)."""
    return SCALE + (WINDOW - SCALE) * keyframe_interval


@app.function(image=image, gpu=['H100!', 'A100-80GB'], cpu=(4, 4), memory=(16384, 16384), timeout=1200,
              retries=0, max_containers=1, min_containers=0, scaledown_window=2, volumes={'/artifact': volume})
def serve(video, mode, frames, stride, keyframe_interval, submitted, budget_s, prefetch=True):
    entered = time.time()
    import collections
    import cv2
    import torch
    from PIL import Image
    from torchvision.transforms import functional as F
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, '/opt/lingbot')
    from demo import load_model
    from lingbot_map.utils.load_fn import load_and_preprocess_images
    from lingbot_map.models.gct_stream_window import GCTStream as Windowed
    from types import SimpleNamespace
    stages = {'queue_and_boot_s': entered - submitted, 'imports_s': time.time() - entered}

    t = time.time(); Path('/tmp/source.mp4').write_bytes(video)
    cap, rgb, n_src = cv2.VideoCapture('/tmp/source.mp4'), {}, 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if n_src % stride == 0: rgb[n_src] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        n_src += 1
    cap.release(); stages['decode_s'] = time.time() - t
    order = schedule(n_src, stride, frames)

    def preprocess(frame):
        """load_and_preprocess_images(mode='crop', 518, 14) on an already decoded frame (checked below)."""
        h, w = frame.shape[:2]; nh = round(h * 518 / w / 14) * 14
        x = F.to_tensor(Image.fromarray(frame).resize((518, nh), Image.Resampling.BICUBIC))
        return x[:, (nh - 518) // 2:(nh - 518) // 2 + 518] if nh > 518 else x

    cv2.imwrite('/tmp/check.png', cv2.cvtColor(rgb[0], cv2.COLOR_RGB2BGR))
    preprocess_diff = float((preprocess(rgb[0]) - load_and_preprocess_images(['/tmp/check.png'], mode='crop', image_size=518, patch_size=14)[0]).abs().max())
    assert preprocess_diff < 1e-6, preprocess_diff

    t = time.time()
    weight = hf_hub_download('robbyant/lingbot-map', 'lingbot-map.pt', revision=WEIGHTS_REV, cache_dir='/artifact/hf')
    stages['weights_fetch_s'] = time.time() - t; t = time.time()
    assert digest(weight) == WEIGHTS_SHA
    stages['weights_sha_s'] = time.time() - t; t = time.time()
    K = keyframe_interval
    # stream: 3D RoPE positions advance once per keyframe and never reset; windowed resets them every window
    rope = max(1024, SCALE + -(-(frames - SCALE) // K) + 16) if mode == 'stream' else 1024
    args = SimpleNamespace(mode='windowed' if mode == 'windowed' else 'streaming', image_size=518, patch_size=14,
        enable_3d_rope=True, max_frame_num=rope, kv_cache_sliding_window=WINDOW, num_scale_frames=SCALE,
        use_sdpa=True, camera_num_iterations=4, model_path=weight)
    model = load_model(args, 'cuda'); model.aggregator = model.aggregator.to(dtype=torch.bfloat16); model.eval()
    torch.cuda.synchronize(); stages['model_load_s'] = time.time() - t
    model_ready = time.time()

    n = len(order); dev = torch.device('cuda')
    pose = np.full((n, 9), np.nan, np.float32); dmed = np.full(n, np.nan, np.float32)
    t_pre, t_in, t_fwd = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    window_id = np.full(n, -1, np.int32); kf = np.zeros(n, bool); boundary = np.zeros(n, bool)
    mem, wins = [], []
    L = window_frames(K) if mode == 'windowed' else None
    tail = collections.deque(maxlen=SCALE)  # (image, global pose_enc, depth, keyframe) of the newest frames
    pending, sim3, in_window, win, first_out = [], None, 0, 0, None

    def forward(block):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return model.forward(block, num_frame_for_scale=SCALE, num_frame_per_block=block.shape[1], causal_inference=True)

    def emit(rows, out, keyframes):
        pe, depth = out['pose_enc'].float(), out['depth'].float()
        if sim3 is not None:
            warped = Windowed._warp_predictions({'pose_enc': pe, 'depth': depth}, sim3[1], sim3[2], sim3[0], 1)
            pe, depth = warped['pose_enc'], warped['depth']
        pose[rows] = pe[0].cpu().numpy(); dmed[rows] = depth[0, ..., 0].flatten(1).median(1).values.cpu().numpy()
        _ = depth.cpu()  # a live service ships every depth map to the mapper
        for k, r in enumerate(rows):
            tail.append((xs[k], pe[:, k:k + 1], depth[:, k:k + 1], keyframes[k]))
            window_id[r], kf[r] = win, keyframes[k]

    def prep(src):
        a = time.perf_counter(); x = preprocess(rgb[src]); return x, time.perf_counter() - a

    # prefetch: the next frame is resized on CPU while the GPU runs this one, as a live decoder thread would
    pool = __import__('concurrent.futures').futures.ThreadPoolExecutor(1)
    nxt = pool.submit(prep, order[0][0])
    loop_start = time.time(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for j, (src, _) in enumerate(order):
            if time.time() - entered > budget_s: break
            a = time.perf_counter()
            x, t_pre[j] = nxt.result() if prefetch else prep(src)
            if prefetch and j + 1 < n: nxt = pool.submit(prep, order[j + 1][0])
            x = x.to(dev)[None, None]
            b = time.perf_counter(); t_in[j] = b - a
            if j < SCALE:  # the first scale block needs 8 frames; earlier ones wait (recorded as NaN)
                pending.append(x)
                if j < SCALE - 1: continue
                xs = pending; emit(list(range(SCALE)), forward(torch.cat(pending, 1)), [True] * SCALE)
                in_window = SCALE; first_out = time.time()
            else:
                if L is not None and in_window == L:  # window full: its last 8 frames become the next scale block
                    model.clean_kv_cache()
                    xs_prev = [f[0] for f in tail]
                    prev = {'pose_enc': torch.cat([f[1] for f in tail], 1), 'depth': torch.cat([f[2] for f in tail], 1),
                            'is_keyframe': torch.tensor([[f[3] for f in tail]], device=dev)}
                    out = forward(torch.cat(xs_prev, 1))
                    curr = {'pose_enc': out['pose_enc'].float(), 'depth': out['depth'].float(),
                            'is_keyframe': torch.ones(1, SCALE, dtype=torch.bool, device=dev)}
                    sim3 = Windowed._pairwise_alignment(prev, curr, SCALE, 1, dev, torch.float32)
                    win += 1; in_window = SCALE; boundary[j] = True
                    wins.append({'frame': j, 'scale': float(sim3[0][0]), 'R': sim3[1][0].cpu().tolist(), 't': sim3[2][0].cpu().tolist()})
                key = K <= 1 or (in_window - SCALE) % K == 0
                if not key: model._set_skip_append(True)
                out = forward(x)
                if not key: model._set_skip_append(False)
                xs = [x]; emit([j], out, [key]); in_window += 1
            t_fwd[j] = time.perf_counter() - b
            if j % 20 == 0 or boundary[j]:
                free, total = torch.cuda.mem_get_info()
                mem.append([j, torch.cuda.memory_allocated(), torch.cuda.memory_reserved(), torch.cuda.max_memory_allocated(),
                            total - free, model.get_kv_cache_info()['cache_memory_mb']])
                torch.cuda.reset_peak_memory_stats()
    done = int(np.isfinite(t_fwd).sum()) + SCALE - 1 if first_out else 0
    buf = io.BytesIO()
    np.savez_compressed(buf, src=np.array([s for s, _ in order]), pass_id=np.array([p for _, p in order]), pose=pose,
        depth_median=dmed, t_pre=t_pre, t_in=t_in, t_fwd=t_fwd, window=window_id, keyframe=kf, boundary=boundary,
        mem=np.array(mem, np.float64).reshape(-1, 6), win_scale=np.array([w['scale'] for w in wins]))
    stages['first_pose_after_submit_s'] = first_out - submitted if first_out else None
    stages['first_pose_after_model_ready_s'] = first_out - model_ready if first_out else None
    summary = {'mode': mode, 'gpu': torch.cuda.get_device_name(), 'torch': str(torch.__version__), 'code_revision': REV,
        'weights_sha256': WEIGHTS_SHA, 'video_sha256': digest('/tmp/source.mp4'),
        'config': {'image_size': 518, 'scale_frames': SCALE, 'window': WINDOW, 'keyframe_interval': K,
                   'frames_per_window': L, 'overlap': SCALE if L else None, 'max_frame_num': rope, 'backend': 'sdpa',
                   'camera_iterations': 4, 'prefetch': prefetch, 'dtype': 'bf16 aggregator, fp32 heads', 'stride': stride,
                   'source_frames': n_src, 'input_hw': list(x.shape[-2:])},
        'frames_requested': frames, 'frames_done': done, 'stopped_by_budget': done < frames,
        'preprocess_max_abs_diff_vs_official': preprocess_diff, 'cold_start': stages,
        'loop_seconds': time.time() - loop_start, 'container_seconds': time.time() - entered, 'windows': wins}
    return {'summary': summary, 'arrays': buf.getvalue()}


def umeyama(a, b):
    """Least-squares Sim3 with a ~ s R b + t for (N, 3) point sets."""
    ma, mb = a.mean(0), b.mean(0); A, B = a - ma, b - mb
    U, S, Vt = np.linalg.svd(A.T @ B / len(a))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt; s = float((S * np.diag(D)).sum() / (B ** 2).sum(1).mean())
    return s, R, ma - s * R @ mb


def angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def analyze(a, warm=50):
    """Latency, memory and pass-against-pass-0 drift from the arrays serve() returns (E-free: all measured)."""
    from scipy.spatial.transform import Rotation
    # t_in: main thread waiting for the frame + upload (= preprocessing when not prefetched; runs before t_in existed were not)
    t_in = a['t_in'] if 't_in' in a else a['t_pre']
    t = t_in + a['t_fwd']; n = len(t); idx = np.arange(n); fed = np.isfinite(t)
    lat = np.maximum(a['t_pre'], t_in) + a['t_fwd']  # frame handed over -> pose and depth on the host
    steady = fed & (idx >= warm) & ~a['boundary']
    ms = lambda x, p: round(float(np.percentile(x, p)) * 1000, 1)
    tenth = max(1, int(fed.sum() - warm) // 10)
    after = np.flatnonzero(fed & (idx >= warm))
    fps = lambda rows: round(len(rows) / float(t[rows].sum()), 2)
    latency = {'warm_frames': int(steady.sum()), 'p50_ms': ms(lat[steady], 50), 'p95_ms': ms(lat[steady], 95),
               'p99_ms': ms(lat[steady], 99), 'max_ms': ms(lat[steady], 100),
               'preprocess_p50_ms': ms(a['t_pre'][steady], 50), 'model_p50_ms': ms(a['t_fwd'][steady], 50),
               'boundary_frames': int(a['boundary'].sum()),
               **({'boundary_p50_ms': ms(lat[a['boundary']], 50), 'boundary_max_ms': ms(lat[a['boundary']], 100)} if a['boundary'].any() else {}),
               'fps_all_frames': fps(np.flatnonzero(fed)), 'fps_first_tenth': fps(after[:tenth]), 'fps_last_tenth': fps(after[-tenth:])}
    m = a['mem'][a['mem'][:, 0] >= warm]; mb = 1 / 2 ** 20
    k = max(1, len(m) // 10)
    memory = {'samples': len(m), 'allocated_mb_first_last': [round(m[0, 1] * mb), round(m[-1, 1] * mb)],
              'peak_alloc_mb_first_tenth': round(m[:k, 3].max() * mb), 'peak_alloc_mb_last_tenth': round(m[-k:, 3].max() * mb),
              'device_used_mb_first_last': [round(m[0, 4] * mb), round(m[-1, 4] * mb)], 'device_used_mb_max': round(m[:, 4].max() * mb),
              'kv_cache_mb_first_last': [round(m[0, 5], 1), round(m[-1, 5], 1)],
              'alloc_slope_mb_per_1000_frames': round(float(np.polyfit(m[:, 0], m[:, 1] * mb, 1)[0]) * 1000, 2) if len(m) > 1 else None}
    memory['flat'] = memory['peak_alloc_mb_last_tenth'] - memory['peak_alloc_mb_first_tenth'] <= 256

    ok = np.isfinite(a['pose']).all(1)
    rows = {int(p): {int(s): i for i, s in zip(idx[ok & (a['pass_id'] == p)], a['src'][ok & (a['pass_id'] == p)])}
            for p in np.unique(a['pass_id'][ok])}
    base = rows[0]; srcs = sorted(base)
    c0 = a['pose'][[base[s] for s in srcs], :3].astype(float)
    path0 = float(np.linalg.norm(np.diff(c0, axis=0), axis=1).sum())
    passes = []
    for p, r in rows.items():
        common = [s for s in srcs if s in r]
        if len(common) < 0.9 * len(srcs): continue  # partial last pass
        i0, ip = [base[s] for s in common], [r[s] for s in common]
        A, B = a['pose'][i0, :3].astype(float), a['pose'][ip, :3].astype(float)
        s, R, tr = umeyama(A, B)
        rot = Rotation.from_quat(a['pose'][i0, 3:7]).inv() * Rotation.from_quat(a['pose'][ip, 3:7])
        passes.append({'pass': int(p), 'first_frame': int(min(ip)),
                       'center_error_over_path': round(float(np.median(np.linalg.norm(A - B, axis=1))) / path0, 4),
                       'camera_rotation_error_deg': round(float(np.degrees(np.median(rot.magnitude()))), 2),
                       'path_scale_vs_pass0': round(1 / s, 4), 'fit_rotation_deg': round(angle_deg(R), 2),
                       'fit_residual_over_path': round(float(np.sqrt(((A - (s * B @ R.T + tr)) ** 2).sum(1).mean())) / path0, 4),
                       'depth_scale_vs_pass0': round(float(np.median(a['depth_median'][ip] / a['depth_median'][i0])), 4)})
    drift = {'pass0_path_length_model_units': round(path0, 4), 'passes': passes}
    if len(a['win_scale']):
        ws = a['win_scale']  # each window's Sim3 scale into the first window's frame
        step = ws / np.concatenate([[1], ws[:-1]])
        drift['window_scales'] = {'count': len(ws), 'last_window_vs_first': round(float(ws[-1]), 4),
                                  'geometric_mean_step': round(float(np.exp(np.log(step).mean())), 4),
                                  'step_min_max': [round(float(step.min()), 4), round(float(step.max()), 4)]}
    go = latency['fps_all_frames'] >= 10 and latency['fps_last_tenth'] >= 10 and memory['flat']
    return {'latency': latency, 'memory': memory, 'drift': drift,
            'go_10fps_flat_memory': bool(go), 'measured': 'all numbers measured (M) from frames.npz'}


def load(run_dir):
    with np.load(Path(run_dir) / 'frames.npz') as z: return {k: z[k] for k in z.files}


@app.local_entrypoint()
def main(out: str, mode: str = 'windowed', frames: int = 7560, stride: int = 2, keyframe_interval: int = 2,
         gpu: str = '', budget: int = 1000, prefetch: bool = True):
    """7560 frames at stride 2 of a 25 fps shot = 12.5 fps for 10.1 min of stream."""
    assert mode in ('stream', 'windowed') and 0 < budget <= 1100
    video = CLIP / 'source-full.mp4'; out = Path(out); out.mkdir(parents=True, exist_ok=False)
    fn = serve.with_options(gpu=gpu) if gpu else serve
    submitted = time.time()
    result = fn.remote(video.read_bytes(), mode, frames, stride, keyframe_interval, submitted, budget, prefetch)
    wall = time.time() - submitted
    (out / 'frames.npz').write_bytes(result['arrays'])
    (out / 'runner-at-execution.py').write_bytes(Path(__file__).read_bytes())
    save(out / 'run.json', {**result['summary'], 'client_wall_seconds': wall, 'source_video': str(video)})
    report = analyze(load(out)); save(out / 'report.json', report)
    print(json.dumps({'client_wall_seconds': round(wall, 1), 'gpu': result['summary']['gpu'],
                      'cold_start': result['summary']['cold_start'], **report, 'drift': report['drift']['passes'][-3:]}), flush=True)


def self_check():
    s = schedule(420, 2, 500)
    assert [x[0] for x in s[:3]] == [0, 2, 4] and s[209] == (418, 0) and s[210] == (418, 1) and s[419] == (0, 1) and s[420] == (0, 2)
    assert len(schedule(420, 2, 7560)) == 7560 and schedule(420, 2, 7560)[-1] == (0, 35)
    assert window_frames(1) == 64 and window_frames(2) == 120
    rng = np.random.default_rng(0); b = rng.normal(size=(50, 3))
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler('xyz', [10, -20, 30], degrees=True).as_matrix()
    s, R2, t = umeyama(1.7 * b @ R.T + [1, 2, 3], b)
    assert abs(s - 1.7) < 1e-9 and np.allclose(R2, R) and np.allclose(t, [1, 2, 3]) and abs(angle_deg(R2) - angle_deg(R)) < 1e-6
    # synthetic run: a curved walk; pass p drawn 1 + 0.01 p larger and turned p degrees about z; depth scales with it
    order = schedule(40, 2, 120); n = len(order)
    src = np.array([o[0] for o in order]); pas = np.array([o[1] for o in order])
    walk = np.stack([np.sin(src / 10), src / 20, np.cos(src / 7)], 1)
    pose = np.zeros((n, 9), np.float32); dm = np.zeros(n, np.float32)
    for i, p in enumerate(pas):
        Rz = Rotation.from_euler('z', p, degrees=True)
        pose[i, :3] = (1 + 0.01 * p) * Rz.apply(walk[i]); pose[i, 3:7] = Rz.as_quat(); dm[i] = 2 * (1 + 0.01 * p)
    t_fwd = np.full(n, 0.08); t_fwd[:7] = np.nan; boundary = np.zeros(n, bool); boundary[60] = True; t_fwd[60] = 0.5
    mem = np.array([[j, 5e9, 6e9, 5.5e9 + (j > 100) * 1e8, 7e9, 100] for j in range(0, n, 10)], float)
    rep = analyze({'src': src, 'pass_id': pas, 'pose': pose, 'depth_median': dm, 't_pre': np.full(n, 0.002), 't_fwd': t_fwd,
                   'boundary': boundary, 'mem': mem, 'win_scale': np.array([1.0, 1.1])}, warm=10)
    last = rep['drift']['passes'][-1]
    assert last['pass'] == 5 and abs(last['path_scale_vs_pass0'] - 1.05) < 1e-3 and abs(last['fit_rotation_deg'] - 5) < 0.01
    assert abs(last['camera_rotation_error_deg'] - 5) < 0.01 and abs(last['depth_scale_vs_pass0'] - 1.05) < 1e-3
    assert last['fit_residual_over_path'] < 1e-3 and rep['drift']['passes'][0]['center_error_over_path'] == 0
    assert rep['latency']['p50_ms'] == 82.0 and rep['latency']['boundary_frames'] == 1 and rep['latency']['fps_all_frames'] < 12.2
    assert rep['memory']['flat'] and rep['go_10fps_flat_memory'] and rep['drift']['window_scales']['geometric_mean_step'] == round(1.1 ** .5, 4)
    rep = analyze({'src': src, 'pass_id': pas, 'pose': pose, 'depth_median': dm, 't_pre': np.full(n, 0.07), 't_in': np.full(n, 0.001),
                   't_fwd': t_fwd, 'boundary': boundary, 'mem': mem, 'win_scale': np.array([1.0, 1.1])}, warm=10)
    assert rep['latency']['p50_ms'] == 150.0 and rep['latency']['fps_last_tenth'] == round(1 / 0.081, 2)  # prefetched: latency > 1 / fps
    mem[:, 3] += mem[:, 0] * 1e7; rep = analyze({**{'src': src, 'pass_id': pas, 'pose': pose, 'depth_median': dm}, 't_pre': np.full(n, 0.002),
                                                  't_fwd': t_fwd, 'boundary': boundary, 'mem': mem, 'win_scale': np.array([])}, warm=10)
    assert not rep['memory']['flat'] and not rep['go_10fps_flat_memory'] and 'window_scales' not in rep['drift']
    print('self-check ok')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--self-check', action='store_true'); p.add_argument('--analyze', type=Path, metavar='RUN_DIR')
    a = p.parse_args()
    if a.self_check: self_check()
    elif a.analyze: report = analyze(load(a.analyze)); save(a.analyze / 'report.json', report); print(json.dumps(report))
    else: p.print_help()
