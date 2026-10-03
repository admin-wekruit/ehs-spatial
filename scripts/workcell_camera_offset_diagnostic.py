"""Is the light-curtain housing gate failure a photo camera offset or a face mismatch?

Diagnostic only; nothing is written back to cameras, point maps or models.
For each held-out photo H, each curtain is refit from the other three photos
with the unchanged housing-face fitter. One rigid correction of camera H
(rotation only by default, or six parameters) is then solved from ONE curtain's observed photo-H boundaries and
applied to the OTHER curtain, which did not enter that correction. If the
same small correction also brings the independent curtain inside the
pre-registered pixel gate, a camera offset is supported; if not, face identity
(front versus wing) or a non-rectangular part remains the explanation.

python scripts/workcell_camera_offset_diagnostic.py --root RUN --sources a.jpg b.jpg c.jpg d.jpg --out NEW_DIR
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from workcell_photo_metrology import _legacy, _load, _object_edges, TARGETS
from workcell_post_faces import _face_errors, _raw_tolerance, _source_initializations, complete_face_observations, fit_multiview_face

CURTAINS = ('post-box-1', 'post-box-2')


def _corrected(frame, parameters):
    """Camera-frame rotation then translation applied to the saved camera-to-world pose."""
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    delta[:3, 3] = parameters[3:]
    return {**frame, 'pose': frame['pose'] @ delta}


def _errors(corners, match, frame):
    _, row, flipped = match
    lengths = np.linalg.norm(np.concatenate(_face_errors(corners, row, frame, flipped, True)), axis=1)
    return {'rmsRawPx': float(np.sqrt(np.mean(lengths ** 2))), 'maxRawPx': float(lengths.max())}


def run(root, sources, out, dof=3):
    started = time.monotonic()
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    geometry, catalog, segmentation, frames, _, _ = _load(root, [Path(p) for p in sources], None)
    ground = geometry['floor']
    up = np.asarray(ground['normal'], float); up /= np.linalg.norm(up)
    legacy = {ident: _legacy(catalog[ident], geometry) for ident in TARGETS}
    drawings = {photo: {'candidates': [], 'selected': []} for photo in frames}
    _, diagnostics = _object_edges(catalog, segmentation, geometry, frames, up, legacy, drawings,
                                   targets=CURTAINS, match_edges=False)
    observed = {row['id']: complete_face_observations(row, frames)[0] for row in diagnostics if row['id'] in CURTAINS}
    full = {}
    for ident in CURTAINS:
        initial = _source_initializations(observed[ident], frames, up)
        upright = fit_multiview_face(observed[ident], frames, ground, initial_corners=initial, orientation='upright')
        extra = [{'photo': p, 'observationId': row['observationId'], 'cornersNative': upright[0].tolist()} for p, row, _ in upright[2] if row.get('rawCorners') is not None]
        full[ident] = fit_multiview_face(observed[ident], frames, ground, initial_corners=initial + extra, orientation='free', fixed_matches=upright[2])
    scale = np.linalg.norm(np.ptp(np.concatenate([corners for corners, _, _ in full.values()]), axis=0))
    report = {'schemaVersion': 1, 'scope': 'Diagnostic only; saved cameras, point maps and models are unchanged.',
              'correctionDegreesOfFreedom': dof, 'byHeldOutPhoto': []}
    for held in sorted(frames):
        threshold = _raw_tolerance(frames[held])
        rows = {}
        for ident in CURTAINS:
            corners, gate, matches = full[ident]
            match = next((m for m in matches if m[0] == held), None)
            training = {p: rows_ for p, rows_ in observed[ident].items() if p != held and rows_}
            fixed = [m for m in matches if m[0] != held]
            if match is None or len(training) < 2:
                rows[ident] = None
                continue
            initial = [row for row in _source_initializations(training, frames, up)]
            # Same fitter, same fixed per-photo associations, held-out photo removed.
            refit, refit_gate, _ = fit_multiview_face(training, frames, ground, initial_corners=initial + [
                {'photo': m[0], 'observationId': m[1]['observationId'], 'cornersNative': corners.tolist()} for m in fixed if m[1].get('rawCorners') is not None],
                orientation='free', fixed_matches=fixed)
            rows[ident] = {'corners': refit, 'match': match, 'trainingGate': refit_gate,
                           'before': _errors(refit, match, frames[held])}
        if any(row is None for row in rows.values()):
            report['byHeldOutPhoto'].append({'photo': held, 'status': 'unsupported', 'reason': 'Both curtains need this photo and two training photos'})
            continue
        entry = {'photo': held, 'thresholdRawPx': threshold, 'curtains': {}}
        for source, target in (CURTAINS, CURTAINS[::-1]):
            full_parameters = (lambda p: p) if dof == 6 else (lambda p: np.r_[p, 0., 0., 0.])
            residual = lambda p, ident=source: np.concatenate(_face_errors(rows[ident]['corners'], rows[ident]['match'][1], _corrected(frames[held], full_parameters(p)), rows[ident]['match'][2], True)).ravel()
            solved = least_squares(residual, np.zeros(dof), loss='soft_l1', f_scale=1.5, max_nfev=200)
            solved.x = full_parameters(solved.x)
            camera = _corrected(frames[held], solved.x)
            entry['curtains'][f'{source}->{target}'] = {
                'correctionFrom': source, 'appliedTo': target,
                'rotationDeg': float(np.degrees(np.linalg.norm(solved.x[:3]))),
                'translationNative': float(np.linalg.norm(solved.x[3:])), 'translationRelativeToCurtainExtent': float(np.linalg.norm(solved.x[3:]) / scale),
                'fitted': {'before': rows[source]['before'], 'after': _errors(rows[source]['corners'], rows[source]['match'], camera)},
                'independent': {'before': rows[target]['before'], 'after': _errors(rows[target]['corners'], rows[target]['match'], camera)},
                'independentInsideGate': _errors(rows[target]['corners'], rows[target]['match'], camera)['maxRawPx'] <= threshold,
                'converged': bool(solved.success)}
        entry['trainingGates'] = {ident: {'accepted': rows[ident]['trainingGate']['accepted'], 'reprojectionByPhoto': rows[ident]['trainingGate']['reprojectionByPhoto']} for ident in CURTAINS}
        report['byHeldOutPhoto'].append(entry)
    report['wallSeconds'] = time.monotonic() - started
    report['reading'] = ('A camera offset is supported only where a correction solved from one curtain brings the other, '
                         'independent curtain inside the gate. Any correction is a diagnostic hypothesis; adopting it would '
                         'require recomputing every geometry that depends on that camera, never mixing it with the old point map.')
    (out / 'camera-offset-diagnostic.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--sources', type=Path, nargs=4, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dof', type=int, choices=(3, 6), default=3, help='3: camera rotation only; 6: rotation and translation')
    args = parser.parse_args()
    result = run(args.root, args.sources, args.out, args.dof)
    for row in result['byHeldOutPhoto']:
        for key, value in row.get('curtains', {}).items():
            print(row['photo'], key, f"rot {value['rotationDeg']:.2f} deg", 'independent', value['independent']['before'], '->', value['independent']['after'], 'gate', row['thresholdRawPx'])
