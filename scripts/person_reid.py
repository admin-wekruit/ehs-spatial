"""Offline OSNet-AIN descriptors from observed person masks; never assign identities.

Download provenance belongs in model-dir/manifest.json. This entry uses the
official standalone model source, existing Torch/Torchvision, and no training.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


SOURCE_SHA = 'a43b495866a4b1645cd134a30c3011592e8c6826ba4781955a135588cf46cd25'
WEIGHTS_SHA = '8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f'


def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def person_crop(rgb, mask):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError('Expected RGB uint8 image and mask in the same source pixel domain')
    if mask.dtype != bool or not mask.any():
        raise ValueError('A nonempty observed boolean mask is required')
    y, x = np.nonzero(mask)
    x0, y0, x1, y1 = int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)
    px, py = int(np.ceil((x1 - x0) * .1)), int(np.ceil((y1 - y0) * .1))
    box = [max(0, x0 - px), max(0, y0 - py), min(rgb.shape[1], x1 + px), min(rgb.shape[0], y1 + py)]
    return Image.fromarray(rgb).crop(box), box


class PersonReID:
    def __init__(self, model_dir: Path, device='mps'):
        import torch
        import torchvision.transforms as transforms

        source, weights = model_dir / 'osnet_ain.py', model_dir / 'weights.pth'
        if sha(source) != SOURCE_SHA or sha(weights) != WEIGHTS_SHA:
            raise ValueError('OSNet source or weights differ from the reviewed official revision')
        spec = importlib.util.spec_from_file_location('official_osnet_ain', source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        state = torch.load(weights, map_location='cpu', weights_only=True)
        if not isinstance(state, dict) or not all(key.startswith('module.') for key in state):
            raise ValueError('Expected the pinned official ReID state_dict')
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        self.model = module.osnet_ain_x1_0(num_classes=state['classifier.weight'].shape[0], pretrained=False)
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        # Official FeatureExtractor preprocessing; crop is derived from the mask.
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 128)), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        self.device = device

    def describe(self, rgb, mask):
        import torch

        cropped, box = person_crop(rgb, mask)
        with torch.inference_mode():
            vector = self.model(self.preprocess(cropped)[None].to(self.device)).cpu().numpy()[0]
        norm = np.linalg.norm(vector)
        if vector.shape != (512,) or not np.isfinite(vector).all() or not norm > 0:
            raise ValueError('OSNet returned an invalid descriptor')
        return vector / norm, box


def compare(features, samples):
    features = np.asarray(features)
    if features.ndim != 2 or len(features) != len(samples) or not np.isfinite(features).all():
        raise ValueError('Expected one finite descriptor per observation')
    if not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-5):
        raise ValueError('Descriptors must be L2 normalized')
    references = [i for i, sample in enumerate(samples) if sample['role'] == 'reference']
    ids = sorted(set(samples[i]['reference_id'] for i in references))
    if len(ids) < 2:
        raise ValueError('Comparison requires at least two competing reference identities')
    cosine = np.clip(features @ features.T, -1, 1)
    results = []
    for i, sample in enumerate(samples):
        if sample['role'] != 'query':
            continue
        scores = [max(float(cosine[i, j]) for j in references if samples[j]['reference_id'] == identity) for identity in ids]
        order = np.argsort(scores)[::-1]
        results.append({'query': sample['name'], 'scores_to_references': {
            samples[j]['name']: float(cosine[i, j]) for j in references},
            'reference_ids': ids, 'max_reference_cosine': scores,
            'min_reference_l2_distance': [float(np.sqrt(max(0, 2 - 2 * score))) for score in scores],
            'candidate_by_score': ids[int(order[0])] if scores[order[0]] > scores[order[1]] else None,
            'candidate_competition_margin': scores[order[0]] - scores[order[1]],
            'identity_assignment': None, 'status': 'candidate_only_not_calibrated'})
    return results, cosine


def run(args):
    start = time.monotonic()
    manifest = json.loads(args.samples.read_text())
    samples = manifest['samples']
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing experiment')
    model = PersonReID(args.model_dir, args.device)
    descriptors, observations = [], []
    for sample in samples:
        image_path, mask_path = Path(sample['rgb_path']), Path(sample['mask_path'])
        if sha(image_path) != sample['rgb_file_sha256'] or sha(mask_path) != sample['mask_file_sha256']:
            raise ValueError('Source image or observed mask changed')
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        mask = np.asarray(Image.open(mask_path).convert('L')) > 0
        vector, box = model.describe(rgb, mask)
        descriptors.append(vector)
        observations.append(sample | {'crop_xyxy': box, 'mask_area_pixels': int(mask.sum())})
    queries, cosine = compare(descriptors, samples)
    result = {'method': 'Official OSNet-AIN x1.0 MSMT17 pretrained person descriptor',
              'weights_sha256': WEIGHTS_SHA, 'model_source_sha256': SOURCE_SHA,
              'script_sha256': sha(__file__), 'input_manifest_sha256': sha(args.samples),
              'model_provenance': json.loads((args.model_dir / 'manifest.json').read_text()),
              'training_performed': False, 'device': args.device, 'descriptor_dimension': 512,
              'crop_padding_fraction': .1, 'input_size_height_width': [256, 128],
              'preprocessing': 'Observed mask bounding crop with 10% padding; official Resize/ToTensor/ImageNet normalization',
              'samples': observations, 'queries': queries, 'all_pair_cosine': cosine.tolist(),
              'elapsed_seconds': time.monotonic() - start,
              'identity_scope': 'video_session_only',
              'limitations': ['Scores are not calibrated identity probabilities',
                             'No automatic identity assignment, tracking correction, or cross-video identity claim']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix('.npz'), features=descriptors, cosine_similarity=cosine,
             names=[sample['name'] for sample in samples])
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps({'elapsed_seconds': result['elapsed_seconds'], 'queries': queries}, indent=2))


def self_check():
    rgb = np.zeros((10, 8, 3), np.uint8)
    rgb[2:8, 1:5] = [255, 0, 0]
    mask = np.zeros((10, 8), bool)
    mask[2:8, 1:5] = True
    crop, box = person_crop(rgb, mask)
    assert box == [0, 1, 6, 9] and crop.size == (6, 8)
    assert tuple(np.asarray(crop)[1, 1]) == (255, 0, 0)  # Preserve RGB channel order.
    samples = [{'name': 'a', 'role': 'reference', 'reference_id': 0},
               {'name': 'b', 'role': 'reference', 'reference_id': 1},
               {'name': 'q', 'role': 'query'}]
    exact, _ = compare(np.array([[1, 0], [0, 1], [1, 0.]]), samples)
    assert exact[0]['candidate_by_score'] == 0 and exact[0]['identity_assignment'] is None
    tied, _ = compare(np.array([[1, 0], [0, 1], [1, 1]]) / [[1], [1], [np.sqrt(2)]], samples)
    assert tied[0]['candidate_by_score'] is None and tied[0]['candidate_competition_margin'] == 0
    for bad in [mask[:3], np.zeros_like(mask)]:
        try:
            person_crop(rgb, bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid mask accepted')
    print('person ReID self-check passed: RGB crop domain, empty mask, normalized scores and unknown/tied identity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path)
    parser.add_argument('--samples', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['mps', 'cpu'], default='mps')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.model_dir and args.samples and args.output:
        run(args)
    else:
        parser.error('--model-dir, --samples and --output are required')
