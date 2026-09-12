"""Rebuild a correction's derived report artifacts before publishing any of them."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def _snapshot(run: Path) -> dict:
    hashes = {}
    for path in run.rglob('*'):
        if path.is_file() and path.name != '.edit.lock':
            with path.open('rb') as stream:
                hashes[path.relative_to(run).as_posix()] = hashlib.file_digest(stream,'sha256').hexdigest()
    return hashes


def _build_stage(run_id: str) -> dict:
    from scripts import scene_inventory
    from scripts.import_workcell_surface import refresh_surface_associations
    from .interactive_report import build_interactive_run_report
    from .object_evidence import write_object_evidence
    from .observed_scene import build_observed_scene
    from .viewer import build_viewer_html

    run = Path('runs')/run_id
    if scene_inventory.main(['--run',run_id]) != 0:
        raise ValueError('Cached inventory cannot be rebuilt from the saved evidence')
    surface = refresh_surface_associations(run) if (run/'surface/surface.json').is_file() else None
    evidence = json.loads(write_object_evidence(run).read_text())
    observed = build_observed_scene(run)
    build_viewer_html(run)
    build_interactive_run_report(run_id)
    inventory = json.loads((run/'inventory/inventory.json').read_text())
    return {'updated':True, 'report_preserved':False,
            'inventory_count':len(inventory['objects']),
            'evidence_count':len(evidence['candidates']),
            'spatial_status':observed['status'], 'spatial_revision':observed['revision'],
            'surface_supported_inv':surface['supported_inv'] if surface else [],
            'surface_mapped_faces':surface['mapped_faces'] if surface else 0}


def refresh_report_after_correction(run: Path) -> dict:
    """A failed rebuild keeps the last readable inventory/surface/viewer/report.

    The accepted correction already lives in refinements.json. It remains
    there, and in the 2D evidence registry, even if derived geometry fails.
    """
    from .object_evidence import write_object_evidence

    run = Path(run).resolve()
    try:
        write_object_evidence(run)
    except Exception as error:
        return {'updated':False,'report_preserved':True,
                'error':f'Object evidence could not be refreshed: {type(error).__name__}: {error}'}
    before = _snapshot(run)
    repository = Path(__file__).resolve().parents[1]
    # ponytail: copying this run gives a real rollback boundary without a
    # second artifact store. For multi-GB runs use immutable revision assets.
    with tempfile.TemporaryDirectory(prefix=f'.{run.name}-refresh-', dir=run.parent) as temporary:
        workspace = Path(temporary)
        staged = workspace/'runs'/run.name
        shutil.copytree(run, staged, ignore=shutil.ignore_patterns('.edit.lock'))
        environment = dict(os.environ)
        environment['PYTHONPATH'] = str(repository)
        process = subprocess.run([sys.executable,'-m','ehs_spatial.report_refresh',run.name],
                                 cwd=workspace,env=environment,capture_output=True,text=True)
        result_path = staged/'report-refresh.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            'updated':False,'error':f'Report rebuild process exited with code {process.returncode}'}
        if not result.get('updated') or process.returncode:
            return {**result,'updated':False,'report_preserved':True}
        if _snapshot(run) != before:
            return {'updated':False,'report_preserved':True,
                    'error':'The report changed during refresh; the newer source artifacts were preserved'}
        backup = workspace/'previous'
        backup.mkdir()
        promoted = []
        try:
            # Keep the run directory and the caller's .edit.lock inode fixed.
            # Every promoted artifact was validated against the same snapshot.
            for name in ('inventory','surface','refinements','refinements.json','scene.json','policies.json',
                         'object-evidence.json','observed','observed-scene.json',
                         'viewer.html','report.html','report-refresh.json'):
                if not (staged/name).exists():
                    continue
                if (run/name).exists():
                    (run/name).rename(backup/name)
                promoted.append(name)
                (staged/name).rename(run/name)
        except BaseException as error:
            for name in reversed(promoted):
                if (run/name).is_dir():
                    shutil.rmtree(run/name)
                elif (run/name).exists():
                    (run/name).unlink()
                if (backup/name).exists():
                    (backup/name).rename(run/name)
            if not isinstance(error, Exception):
                raise
            return {'updated':False,'report_preserved':True,
                    'error':f'Report publication was rolled back: {type(error).__name__}: {error}'}
        return result


if __name__ == '__main__':
    run_id = sys.argv[1]
    try:
        result = _build_stage(run_id)
    except Exception as error:
        result = {'updated':False,'error':f'{type(error).__name__}: {error}'.replace(str(Path.cwd()),'<staging>')}
    (Path('runs')/run_id/'report-refresh.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    raise SystemExit(0 if result['updated'] else 1)
