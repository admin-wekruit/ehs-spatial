"""Refresh source-linked metrology inputs after a new depth reconstruction.

The three target identities reuse frozen original-image SAM observations. Native
positions, observed extents, floor, fence and button anchor are recomputed. No
previous mesh or observed 3D corner is reused in the new world frame.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time

import numpy as np

from ehs_spatial.measurements import measure_observed_points
from scripts.workcell_photo_oneshot import _array, _frame, _mask, _response
from scripts.workcell_photo_objects import _observation


def depth_run_path(mount, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 2 or relative.parts[0] != 'workcell-depth-resolution':
        raise ValueError('Invalid dedicated depth experiment Volume path')
    root = Path(mount) / 'workcell-depth-resolution'
    candidate = Path(mount) / relative
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError('Depth run escapes the dedicated Volume directory')
    return candidate


def match_fence_plane(old, new):
    def keys(beam):
        return {(beam['sourcePhoto'], *sorted(tuple(round(float(v), 2) for v in p) for p in edge))
                for edge in beam['rawEdges']}
    frozen = set().union(*(keys(b) for b in old['fence']['beams'] if b['plane'] == 0))
    matches = {}
    for beam in new['fence']['beams']:
        matches.setdefault(beam['plane'], set()).update(keys(beam) & frozen)
    ranked = sorted(matches, key=lambda plane: len(matches[plane]), reverse=True)
    if (not ranked or not matches[ranked[0]] or
            (len(ranked) > 1 and len(matches[ranked[0]]) == len(matches[ranked[1]]))):
        raise ValueError('New fence planes lack an unambiguous source-edge identity match')
    selected = matches[ranked[0]]
    photos = sorted({row[0] for row in selected})
    if len(photos) < 2:
        raise ValueError('Fence identity requires matching raw edges in at least two photos')
    return {'planeIndex': ranked[0], 'sourcePhotos': photos, 'matchedRawEdges': len(selected),
            'method': 'same frozen original-image line segments, independent of depth/world frame'}


def match_reference(old, new):
    frozen = {row['photo']: row['boxRaw'] for row in old['anchor']['views']}
    views = new['anchor']['views']
    if len(views) < 2 or any(frozen.get(row['photo']) != row['boxRaw'] for row in views):
        raise ValueError('Depth branch changed the physical reference component or lacks multiple source views')
    return {'sourcePhotos': sorted(row['photo'] for row in views),
            'method': 'same original-image component boxes as the frozen physical reference'}


def refresh_catalog(baseline, root):
    geometry = json.loads((root / 'geometry.json').read_text())
    old_geometry = json.loads((baseline / 'geometry.json').read_text())
    old = {item['id']: item for item in json.loads((baseline / 'objects.json').read_text())['objects']}
    segmentation = json.loads((root / 'sam3.json').read_text())
    identity = match_fence_plane(old_geometry, geometry)
    reference_identity = match_reference(old_geometry, geometry)
    floor = geometry['floor']
    scene = {'floor_plane': [*floor['normal'], floor['offset']]}
    catalog = []
    for ident in ('fence-0', 'post-box-1', 'post-box-2'):
        item = {'id': ident, 'label': old[ident]['label'], 'kind': old[ident]['kind'],
                'observations': [], 'representation': 'fresh depth support linked by frozen source identity'}
        clouds = []
        if ident == 'fence-0':
            item['geometryPlaneIndex'] = identity['planeIndex']
            item['identityEvidence'] = identity
        for previous in old[ident]['observations']:
            photo = previous['photo']
            raw = _frame(root, photo)
            points = _array(raw['pts3d'])
            valid = _array(raw['non_ambiguous_mask']).astype(bool) & np.isfinite(points).all(2)
            if ident == 'fence-0':
                plane = geometry['fence']['planes'][identity['planeIndex']]
                mask = _mask(raw, _response(segmentation, photo, 'safety fence'))
                mask &= abs(points @ np.asarray(plane['normal']) + plane['offset']) < max(floor['residualP95Native'] * 5, 1e-6)
            else:
                match = re.search(r'instance (\d+)', previous['source'])
                if match is None:
                    raise ValueError('Frozen post identity lacks its source SAM instance')
                response = _response(segmentation, photo, 'yellow safety post')
                mask = _mask(raw, {'rle': [response['rle'][int(match[1])]]})
            observation = _observation(mask, photo, previous['source'])
            if observation is None:
                continue
            support = points[mask & valid]
            observation['observedMeasurements'] = measure_observed_points(
                support, scene, mask_pixels=int(mask.sum()),
                source={'photo': photo, 'evidence': previous['source'], 'geometry': 'fresh depth reconstruction'})
            item['observations'].append(observation)
            if len(support):
                clouds.append(support)
        if not clouds or len(item['observations']) < 2:
            raise ValueError(f'{ident} has insufficient fresh source-linked depth support')
        item['modelDimensionsNative'] = np.ptp(np.concatenate(clouds), axis=0).tolist()
        catalog.append(item)
    (root / 'objects.json').write_text(json.dumps({'objects': catalog,
        'referenceIdentity': reference_identity,
        'scope': 'three fixed metrology targets; newly derived geometry, not the complete report catalog'}, indent=2))
    return identity


def run(baseline, root, out, sources, reference, joint_max_nfev=100):
    from scripts.workcell_photo_geometry import build as geometry_build
    from scripts.workcell_guard_controls import colmap
    from scripts.workcell_photo_metrology import build as measure
    from scripts.workcell_button_bundle import build as joint
    from scripts.workcell_photo_metrology import _reference
    reference = _reference(reference)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    stages = {}
    def stage(name, function):
        start = time.monotonic()
        try:
            value = function()
            stages[name] = {'status': 'completed', 'seconds': time.monotonic() - start}
            return value
        except Exception as error:
            stages[name] = {'status': 'failed', 'seconds': time.monotonic() - start,
                            'error': f'{type(error).__name__}: {error}'}
            raise
        finally:
            (out / 'stages.json').write_text(json.dumps(stages, indent=2))
    shutil.copyfile(baseline / 'sam3.json', root / 'sam3.json')
    dimensions = reference['features']
    try:
        stage('freshGeometry', lambda: geometry_build(root, sources, dimensions['mainBodyDiameterM'], dimensions['wholeComponentHeightM']))
        identity = stage('sourceIdentity', lambda: refresh_catalog(baseline, root))
        stage('originalMetrology', lambda: measure(root, out / 'original-cameras', sources, reference))
        stage('cameraTracks', lambda: colmap(root, out / 'control', sources, square_pixels=True))
        joint_result = stage('jointButton', lambda: joint(root, out / 'joint-button', sources, reference,
                                          out / 'control/cameras.json', out / 'control/tracks.json', workers=1, max_nfev=joint_max_nfev))
        if (out / 'joint-button/cameras.json').is_file():
            stage('jointMetrology', lambda: measure(root, out / 'joint-cameras', sources, reference,
                                                   out / 'joint-button/cameras.json', out / 'joint-button/joint-reference.json'))
        else:
            stages['jointMetrology'] = {'status': 'unsupported', 'reason': joint_result['reason']}
        manifest = {'status': 'completed', 'fenceIdentity': identity,
                    'seconds': time.monotonic() - started, 'reference': reference,
                    'reused': 'original-image segmentation and target IDs only; no old native geometry or mesh',
                    'freshFrames': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('frame_*.json.gz'))}}
        (out / 'bridge.json').write_text(json.dumps(manifest, indent=2))
        return manifest
    finally:
        (out / 'stages.json').write_text(json.dumps(stages, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'root', 'out', 'reference'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=4, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.baseline, args.root, args.out, args.sources,
                         json.loads(args.reference.read_text())), ensure_ascii=False))
