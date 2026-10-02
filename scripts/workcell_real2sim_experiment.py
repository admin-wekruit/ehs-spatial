"""Bounded cached-input real2sim ablations; execute in the existing Modal app."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import traceback


def run(root, out, sources, *, max_nfev=100, details_only=False):
    from scripts.workcell_button_bundle import build as fit_button
    from scripts.workcell_guard_controls import colmap, incremental_colmap
    from scripts.workcell_metrology_models import export_reference_candidate
    from scripts.workcell_photo_texture import texture_existing_guards
    from scripts.workcell_post_faces import build as fit_faces

    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    reference = json.loads((root/'reference.json').read_text())
    started = time.monotonic()
    records = {}

    def stage(name, function):
        tick = time.monotonic()
        try:
            result = function()
            record = {'status': 'completed', 'seconds': time.monotonic()-tick}
            if isinstance(result, dict):
                record['resultStatus'] = result.get('status')
        except Exception as error:
            result = None
            record = {'status': 'failed', 'seconds': time.monotonic()-tick,
                      'error': f'{type(error).__name__}: {error}'}
            (out/f'{name}-error.txt').write_text(traceback.format_exc())
        records[name] = record
        print(json.dumps({name: record}), flush=True)
        return result

    def button(name, cameras, tracks, selection):
        result = fit_button(root, out/name, sources, reference, cameras, tracks,
                            max_nfev=max_nfev, workers=1, track_selection=selection)
        export_reference_candidate(result, out/name/'model')
        return result

    # Shared camera experiments and independent appearance/visible-face tests
    # read the frozen baseline. No branch overwrites another branch's geometry.
    with ThreadPoolExecutor(max_workers=3) as pool:
        texture = pool.submit(stage, 'textures', lambda: texture_existing_guards(root, sources, out/'textures'))
        faces = pool.submit(stage, 'post-faces', lambda: fit_faces(root, sources, out/'post-faces'))
        if not details_only:
            control = out/'camera-control'
            stage('camera-control', lambda: colmap(root, control, sources, square_pixels=True))
            if (control/'cameras.json').is_file():
                old = pool.submit(stage, 'old-selection', lambda: button('old-selection', control/'cameras.json', control/'tracks.json', 'spatial_round_robin'))
                balanced = pool.submit(stage, 'balanced-selection', lambda: button('balanced-selection', control/'cameras.json', control/'tracks.json', 'connectivity_balanced'))
                native = out/'independent-poses'
                stage('independent-poses', lambda: incremental_colmap(control, native, max_seconds=180))
                old.result(); balanced.result()
                if (native/'cameras.json').is_file():
                    stage('independent-button', lambda: button('independent-button', native/'cameras.json', native/'tracks.json', 'connectivity_balanced'))
        texture.result(); faces.result()
    result = {'status': 'completed' if all(r['status']=='completed' for r in records.values()) else 'partial',
              'records': records, 'wallSeconds': time.monotonic()-started,
              'scope': 'Cached upstream geometry; includes controlled fitting and texture experiments, not a fresh full photo-to-model runtime.',
              'groundTruthUsedForFitting': False}
    (out/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    if result['status'] == 'partial':
        raise RuntimeError('A real2sim experiment stage failed; saved results include the other stages.')
    return result
