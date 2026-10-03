"""Explicit detector-free observation ablation; fixed cameras and physical gates.

run(root, out, sources, cameras) requires two CUDA devices, torch and kornia on
the remote runner. Importing this module does not import torch or run a model.
Six source-image pairs are split across two separate model instances/devices.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import time
import urllib.request

import cv2
import numpy as np

from workcell_guard_joint import _inputs, _belongs, _merge_tracks, _triangulate, build
from workcell_photo_oneshot import _frame

MODEL_REPO = 'kornia/loftr'
REQUESTED_REVISION = 'bd2587b2acecf13bb5be0f8b4290c12afe335fb3'
MODEL_FILE = 'loftr_indoor_ds_new.ckpt'
MODEL_URL = f'https://cmp.felk.cvut.cz/~mishkdmy/models/{MODEL_FILE}'
MODEL_SHA256 = 'be9ff88b323ec27889114719f668ae41aff7034b56a4c4acbd46b8b180b87ed3'


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _checkpoint(out):
    file = out / MODEL_FILE
    temporary = out / (MODEL_FILE + '.part')
    digest = hashlib.sha256()
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open('wb') as stream:
        while chunk := response.read(1024 * 1024):
            stream.write(chunk); digest.update(chunk)
    checksum = digest.hexdigest()
    if MODEL_SHA256 is not None and checksum != MODEL_SHA256:
        raise ValueError('Official checkpoint SHA256 does not match the pinned artifact')
    temporary.replace(file)
    return file, {'provider': 'Kornia LoFTR', 'checkpoint': MODEL_FILE, 'checkpointUrl': MODEL_URL,
                  'sha256': checksum, 'source': 'official author mirror referenced by Kornia v0.8.2',
                  'rejectedHfSource': {'repo': MODEL_REPO, 'revision': REQUESTED_REVISION, 'reason': 'revision contains README and .gitattributes only; no weights'},
                  'pretrainedVariant': 'indoor_new', 'coarseTempBugFix': True,
                  'confidenceScope': 'LoFTR learned correspondence score; not a calibrated physical accuracy probability'}


def _crops(root, out, sources, boards):
    crops, records = {}, []
    for photo in range(1, 5):
        raw = cv2.imread(str(sources[photo - 1]))
        saved = _frame(root, photo)
        if raw is None or list(raw.shape[:2]) != [saved['original_image']['height'], saved['original_image']['width']]:
            raise ValueError('Raw source dimensions disagree with the recorded pixel transform')
        A = np.asarray(saved['input_mask_transform']['input_to_canonical_pixel_centres'], float)
        union = np.logical_or.reduce([board['masks'][photo] for board in boards.values() if photo in board['masks']])
        yy, xx = np.nonzero(union)
        if len(xx) < 16:
            raise ValueError('Assembly crop has insufficient mask support')
        lo, hi = np.array([xx.min(), yy.min()]) - .5, np.array([xx.max(), yy.max()]) + .5
        margin = np.maximum(12., .2 * (hi - lo))
        ends = (np.c_[np.stack([lo - margin, hi + margin]), np.ones(2)] @ np.linalg.inv(A).T)[:, :2]
        x0, y0 = np.maximum(0, np.floor(ends.min(0))).astype(int)
        x1, y1 = np.minimum(raw.shape[1::-1], np.ceil(ends.max(0))).astype(int)
        width, height = x1 - x0, y1 - y0
        if min(width, height) < 8:
            raise ValueError('Assembly crop is too small')
        factor = min(1., 768 / max(width, height))
        nw, nh = max(8, int(width * factor) // 8 * 8), max(8, int(height * factor) // 8 * 8)
        sx, sy = nw / width, nh / height
        raw_to_network = np.array([[sx, 0, (sx - 1) / 2 - sx * x0],
                                   [0, sy, (sy - 1) / 2 - sy * y0], [0, 0, 1.]])
        network_to_canonical = A @ np.linalg.inv(raw_to_network)
        color = cv2.resize(raw[y0:y1, x0:x1], (nw, nh), interpolation=cv2.INTER_AREA)
        file = f'crop-{photo}.png'; cv2.imwrite(str(out / file), color)
        crops[photo] = {'gray': cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), 'color': color,
                        'networkToCanonical': network_to_canonical, 'networkToRaw': np.linalg.inv(raw_to_network)}
        records.append({'photo': photo, 'file': file, 'rawShape': list(raw.shape[:2]),
                        'cropRawXYXY': [int(x0), int(y0), int(x1), int(y1)], 'networkShape': [nh, nw],
                        'networkToCanonicalPixelCentres': network_to_canonical.tolist(),
                        'networkToRawPixelCentres': np.linalg.inv(raw_to_network).tolist(),
                        'sourceMaskPixels': int(union.sum()), 'contextFraction': .2})
    _write(out / 'crops.json', records)
    return crops


def _worker(device, pairs, crops, checkpoint):
    import torch
    import kornia
    from kornia.feature import LoFTR
    from kornia.feature.loftr.loftr import default_cfg

    started = time.monotonic()
    config = copy.deepcopy(default_cfg)
    config['coarse']['temp_bug_fix'] = True  # official updated indoor checkpoint config
    with torch.cuda.device(device):
        model = LoFTR(pretrained=None, config=config)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['state_dict'], strict=True)
        model = model.eval().to(f'cuda:{device}')
        results = []
        for a, b in pairs:
            first = torch.from_numpy(crops[a]['gray']).to(device=f'cuda:{device}', dtype=torch.float32)[None, None] / 255.
            second = torch.from_numpy(crops[b]['gray']).to(device=f'cuda:{device}', dtype=torch.float32)[None, None] / 255.
            tick = time.monotonic()
            with torch.inference_mode():
                matched = model({'image0': first, 'image1': second})
            torch.cuda.synchronize(device)
            results.append({'photos': [a, b], 'device': device, 'seconds': time.monotonic() - tick,
                            'keypoints0': matched['keypoints0'].cpu().numpy(),
                            'keypoints1': matched['keypoints1'].cpu().numpy(),
                            'confidence': matched['confidence'].cpu().numpy()})
        del model
    return results, {'device': device, 'pairs': [list(pair) for pair in pairs], 'wallSeconds': time.monotonic() - started,
                     'torchVersion': torch.__version__, 'korniaVersion': kornia.__version__,
                     'configuration': config, 'strictStateDictionaryLoad': True}


def _epipolar(first, second, a, b):
    relative = second['pose'][:3, :3].T @ first['pose'][:3, :3]
    t = second['pose'][:3, :3].T @ (first['pose'][:3, 3] - second['pose'][:3, 3])
    cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    F = np.linalg.inv(second['K']).T @ cross @ relative @ np.linalg.inv(first['K'])
    ah, bh = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    line_b, line_a = ah @ F.T, bh @ F
    numerator = np.sum(bh * line_b, axis=1) ** 2
    denominator = np.sum(line_a[:, :2] ** 2 + line_b[:, :2] ** 2, axis=1)
    return np.sqrt(numerator / np.maximum(denominator, 1e-20))


def _collect(out, matched, crops, frames, boards):
    nodes = {side: {} for side in boards}; links = {side: [] for side in boards}
    positions = {side: {photo: [] for photo in frames} for side in boards}
    pair_records = []
    def node(side, photo, network, canonical, raw, confidence, pair):
        previous = positions[side][photo]
        distances = np.linalg.norm(np.asarray(previous) - network, axis=1) if previous else np.empty(0)
        if len(distances) and distances.min() <= 2.:
            ident = (photo, int(distances.argmin()))
            observation = nodes[side][ident]
            observation['confidence'] = min(observation['confidence'], confidence)
            observation['sourcePairs'] = sorted(set(observation['sourcePairs'] + [pair]))
            return ident
        ident = (photo, len(previous)); previous.append(network.copy())
        nodes[side][ident] = {'photo': photo, 'uv': canonical.tolist(), 'networkUv': network.tolist(),
                             'rawUv': raw.tolist(), 'confidence': confidence, 'sourcePairs': [pair]}
        return ident
    for result in sorted(matched, key=lambda value: value['photos']):
        a, b = result['photos']; pair = f'{a}-{b}'
        p0, p1, confidence = result['keypoints0'], result['keypoints1'], result['confidence']
        if p0.shape != p1.shape or p0.shape != (len(confidence), 2):
            raise ValueError('LoFTR returned inconsistent match arrays')
        uv0 = (np.c_[p0, np.ones(len(p0))] @ crops[a]['networkToCanonical'].T)[:, :2]
        uv1 = (np.c_[p1, np.ones(len(p1))] @ crops[b]['networkToCanonical'].T)[:, :2]
        raw0 = (np.c_[p0, np.ones(len(p0))] @ crops[a]['networkToRaw'].T)[:, :2]
        raw1 = (np.c_[p1, np.ones(len(p1))] @ crops[b]['networkToRaw'].T)[:, :2]
        epi = _epipolar(frames[a], frames[b], uv0, uv1)
        counts = {key: 0 for key in ('networkMatches', 'nonfinite', 'confidence', 'epipolar', 'instanceMask', 'triangulation', 'accepted')}
        counts['networkMatches'] = len(confidence)
        rows, accepted = [], []
        for index in np.argsort(-confidence):
            score = float(confidence[index]); reason, side = None, None
            if not np.isfinite([score, epi[index], *uv0[index], *uv1[index]]).all():
                counts['nonfinite'] += 1; continue
            if score < .5:
                reason = 'confidence'
            elif epi[index] > 1.5:
                reason = 'epipolar'
            else:
                owners = [name for name, board in boards.items() if a in board['masks'] and b in board['masks']
                          and _belongs(uv0[index], board['masks'][a]) and _belongs(uv1[index], board['masks'][b])]
                if len(owners) != 1:
                    reason = 'instanceMask'
                else:
                    side = owners[0]
                    tri = _triangulate([{'photo': a, 'uv': uv0[index]}, {'photo': b, 'uv': uv1[index]}], frames)
                    if tri is None or tri[1] > 2.5 or tri[2] < .25:
                        reason = 'triangulation'
            counts[reason or 'accepted'] += 1
            row = {'networkUv0': p0[index].tolist(), 'networkUv1': p1[index].tolist(),
                   'canonicalUv0': uv0[index].tolist(), 'canonicalUv1': uv1[index].tolist(),
                   'rawUv0': raw0[index].tolist(), 'rawUv1': raw1[index].tolist(), 'confidence': score,
                   'sampsonErrorCanonicalPx': float(epi[index]), 'side': side, 'rejection': reason}
            rows.append(row)
            if reason is None:
                x = node(side, a, p0[index], uv0[index], raw0[index], score, pair)
                y = node(side, b, p1[index], uv1[index], raw1[index], score, pair)
                links[side].append((x, y)); accepted.append(row)
        _write(out / f'pair-{pair}.json', {'photos': [a, b], 'counts': counts, 'matches': rows})
        first, second = crops[a]['color'], crops[b]['color']
        canvas = np.zeros((max(len(first), len(second)), first.shape[1] + second.shape[1], 3), np.uint8)
        canvas[:len(first), :first.shape[1]] = first; canvas[:len(second), first.shape[1]:] = second
        colors = {'left': (255, 170, 50), 'center': (40, 230, 230), 'right': (200, 80, 240)}
        for row in accepted[:100]:
            start = tuple(np.rint(row['networkUv0']).astype(int))
            end = tuple(np.rint(np.asarray(row['networkUv1']) + [first.shape[1], 0]).astype(int))
            cv2.line(canvas, start, end, colors[row['side']], 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f'matches-{pair}.jpg'), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        pair_records.append({'photos': [a, b], 'device': result['device'], 'seconds': result['seconds'],
                             'counts': counts, 'matches': f'pair-{pair}.json', 'visualization': f'matches-{pair}.jpg'})
    tracks, diagnostics = [], []
    for side in boards:
        merged, conflicts = _merge_tracks(nodes[side], links[side], side)
        kept, duplicate, invalid = [], 0, 0
        for candidate in sorted(merged, key=lambda t: (-len(t['observations']), -min(o['confidence'] for o in t['observations']))):
            # A source location belongs to one track before train/test splitting.
            # ponytail: six pairs permit a bounded quadratic dedup; use a spatial
            # index if captures grow. Dense neighbors within six input pixels are
            # withheld to avoid near-identical samples in train and holdout sets.
            if any(any(a['photo'] == b['photo'] and np.linalg.norm(np.asarray(a['networkUv']) - b['networkUv']) < 6.
                       for a in candidate['observations'] for b in existing['observations']) for existing in kept):
                duplicate += 1; continue
            tri = _triangulate(candidate['observations'], frames)
            if tri is None or tri[1] > 2.5 or tri[2] < .25:
                invalid += 1; continue
            candidate.update(xyz=tri[0].tolist(), triangulationResidualPx=tri[1], parallaxDeg=tri[2],
                             confidence=min(o['confidence'] for o in candidate['observations']), provider='kornia-loftr-indoor-new')
            kept.append(candidate)
        tracks.extend(kept)
        diagnostics.append({'side': side, 'pairLinks': len(links[side]), 'mergedTracks': len(merged),
                            'conflictingComponentsRejected': conflicts, 'nearDuplicatesRejected': duplicate,
                            'multiviewTriangulationRejected': invalid, 'retainedDistinctTracks': len(kept)})
    for index, track in enumerate(tracks):
        track['id'] = index
    return tracks, pair_records, diagnostics


def run(root, out, sources, cameras):
    root, out = Path(root), Path(out); sources = [Path(source) for source in sources]
    out.mkdir(parents=True, exist_ok=True)
    if len(sources) != 4 or cameras is None:
        raise ValueError('Dense experiment requires four raw photos and explicit calibrated cameras')
    started = time.monotonic()
    frames, boards, association = _inputs(root, cameras, sources)
    crops = _crops(root, out, sources, boards)
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Dense experiment requires two CUDA GPUs; no local or CPU fallback')
    torch.set_num_threads(2)
    pairs = list(itertools.combinations(range(1, 5), 2))
    matched, workers = [], []
    with tempfile.TemporaryDirectory(prefix='workcell-loftr-') as checkpoint_directory:
        checkpoint, model_source = _checkpoint(Path(checkpoint_directory))
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(_worker, device, pairs[device::2], crops, checkpoint) for device in range(2)]
            for job in jobs:
                results, worker = job.result(); matched.extend(results); workers.append(worker)
    tracks, pair_records, merge = _collect(out, matched, crops, frames, boards)
    provenance = {'model': model_source, 'workers': workers, 'pairResults': pair_records, 'trackMerge': merge,
                  'association': association, 'sourceCameraFile': str(cameras),
                  'cameraOptimization': False, 'optimizedModelWeights': False,
                  'gates': {'confidenceMin': .5, 'sampsonMaxCanonicalPx': 1.5, 'triangulationMaxCanonicalPx': 2.5,
                            'parallaxMinDeg': .25, 'mergeNetworkPx': 2., 'dedupNetworkPx': 6.},
                  'limitations': ['Tests only whether sparse feature detection was a bottleneck.',
                                  'Learned matches and confidence are not geometric ground truth; repeated stripes can remain ambiguous.',
                                  'Guard rigidity is assumed. Robot/cart may move; those objects are excluded by board instance masks.',
                                  'Cameras, global scale and physical acceptance gates are unchanged. No physical precision claim.']}
    _write(out / 'tracks.json', {'model': model_source, 'tracks': tracks})
    _write(out / 'dense-results.json', provenance)
    # Exactly one existing solver call consumes supplied source correspondences.
    routes = build(root, out / 'joint', cameras=cameras, tracks=out / 'tracks.json', sources=sources)
    for route, result in routes.items():
        result['observationProvider'] = {'model': model_source, 'denseEvidence': '../../dense-results.json',
                                         'sourceTracks': '../../tracks.json'}
        _write(out / 'joint' / route / 'results.json', result)
    provenance['wallSeconds'] = time.monotonic() - started
    provenance['jointResults'] = {route: [{'id': row['id'], 'status': row['status'], 'reasons': row['reasons'],
                                          'candidateAngleDeg': row.get('candidateAngleDeg')} for row in result['objects']]
                                  for route, result in routes.items()}
    _write(out / 'dense-results.json', provenance)
    return provenance


def _self_check():
    """CPU-only pixel-transform and cross-view track checks; no model import."""
    import gzip
    from workcell_photo_objects import _project
    with tempfile.TemporaryDirectory() as directory:
        out = Path(directory)
        A = np.array([[.25, 0, -.375], [0, .25, -.375], [0, 0, 1.]])
        mask = np.zeros((40, 60), bool); mask[17:25, 20:40] = True
        sources = []
        for photo in range(1, 5):
            source = out / f'source-{photo}.jpg'; cv2.imwrite(str(source), np.full((160, 240, 3), 127, np.uint8)); sources.append(source)
            with gzip.open(out / f'frame_{photo:04d}.json.gz', 'wt') as stream:
                json.dump({'original_image': {'height': 160, 'width': 240},
                           'input_mask_transform': {'input_to_canonical_pixel_centres': A.tolist()}}, stream)
        cropped = _crops(out, out, sources, {'left': {'masks': {p: mask for p in range(1, 5)}}})
        for record in json.loads((out / 'crops.json').read_text()):
            h, w = record['networkShape']; x0, y0, x1, y1 = record['cropRawXYXY']
            assert h % 8 == w % 8 == 0 and max(h, w) <= 768
            uv = np.array([11., 9., 1.])
            expected = A @ [(uv[0] + .5) * (x1 - x0) / w + x0 - .5,
                            (uv[1] + .5) * (y1 - y0) / h + y0 - .5, 1.]
            assert np.allclose(cropped[record['photo']]['networkToCanonical'] @ uv, expected)
        K = np.array([[100., 0, 64.], [0, 100., 64.], [0, 0, 1.]])
        points = np.array([[x, y, 2.] for x in (-.4, -.2, 0, .2) for y in (-.3, -.1, .1, .3)])
        frames, crops = {}, {}
        for photo in range(1, 5):
            pose = np.eye(4); pose[0, 3] = .2 * (photo - 1)
            frames[photo] = {'K': K, 'pose': pose}
            crops[photo] = {'color': np.zeros((128, 128, 3), np.uint8), 'networkToCanonical': np.eye(3), 'networkToRaw': np.eye(3)}
        matched = []
        for index, (a, b) in enumerate(itertools.combinations(range(1, 5), 2)):
            p0, _ = _project(points, frames[a]); p1, _ = _project(points, frames[b])
            # A nearby repeated pair must merge; low confidence must be rejected.
            matched.append({'photos': [a, b], 'device': index % 2, 'seconds': 0.,
                            'keypoints0': np.vstack([p0, p0[:2]]), 'keypoints1': np.vstack([p1, p1[:2]]),
                            'confidence': np.r_[np.full(len(points) + 1, .9), .1]})
        tracks, pairs, _ = _collect(out, matched, crops, frames, {'left': {'masks': {p: np.ones((128, 128), bool) for p in frames}}})
        assert len(tracks) == len(points) and all(len(t['observations']) == 4 for t in tracks)
        assert all(p['counts']['confidence'] == 1 for p in pairs)
        assert max(t['triangulationResidualPx'] for t in tracks) < 1e-8
    print('PASS: crop pixel-center transforms, six-pair track merging/dedup, confidence rejection and fixed-camera triangulation; no GPU/model invoked')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--root', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--images', type=Path, nargs=4)
    parser.add_argument('--cameras', type=Path)
    args = parser.parse_args()
    if args.self_check:
        _self_check()
    elif None in (args.root, args.out, args.images, args.cameras):
        parser.error('--root, --out, --images and --cameras are required for the real experiment')
    else:
        print(json.dumps(run(args.root, args.out, args.images, args.cameras)['jointResults'], indent=2))
