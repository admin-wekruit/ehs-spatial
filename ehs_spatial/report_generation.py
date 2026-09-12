"""Run one explicitly selected evidence candidate using the research pipeline."""
import hashlib
import json
import os
from pathlib import Path
import sys

from fastapi import HTTPException
from fastapi.responses import FileResponse

from .artifacts import _read_json_dict
from .report_workspace import write_json


def job_root(runs: Path, run_id: str, candidate_id: str) -> Path:
    return runs / '.generations' / run_id / candidate_id


def job_source(state: dict):
    return state.get('result', {}).get('source_sha256') or state.get('source_sha256')


def source_revision(source) -> str | None:
    return hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:24] if source is not None else None


def result_revision(state: dict) -> str:
    return source_revision({'source':job_source(state), 'artifacts':state.get('result', {}).get('artifacts')})


def generation_state(run_id: str, candidate_id: str, job: Path, current_source) -> dict:
    state = _read_json_dict(job / 'status.json') or {'state':'not_started'}
    stale = state['state'] == 'done' and job_source(state) != current_source
    error = ('Saved result belongs to an earlier source revision; the selected model input may be unchanged. Generate explicitly to validate and reuse its cache.'
             if stale else state.get('error') or state.get('result', {}).get('reason'))
    result = {'state':'stale' if stale else state['state'], 'error':error,
              'source_revision':source_revision(job_source(state)), 'current_source_revision':source_revision(current_source)}
    if state['state'] == 'done':
        revision = result_revision(state)
        base = f'/api/reports/{run_id}/generations/{candidate_id}/revisions/{revision}/assets'
        result.update(result_revision=revision, viewer_url='/published/viewer.html?scene='+base+'/scene.json&object='+candidate_id,
                      metrics_url=base+'/metrics.html')
    return result


def run_generation(runs: Path, run_id: str, candidate_id: str) -> dict:
    from .object_generation import generate_candidate
    job = job_root(runs, run_id, candidate_id)
    state = dict(_read_json_dict(job / 'status.json'), run_id=run_id, candidate_id=candidate_id, state='running')
    write_json(job / 'status.json', state)
    try:
        result = generate_candidate(runs / run_id, candidate_id, job / 'artifact',
            Path(os.environ['PANOPTES_RESEARCH_ROOT']),
            gpu_budget_seconds=630,
            environment_record=Path(os.environ['PANOPTES_RECGEN_ENVIRONMENT']),
            python_executable=os.environ.get('PANOPTES_RESEARCH_PYTHON', sys.executable),
            viewer_repo=Path(os.environ['PANOPTES_PUBLISHED_ROOT']),
            cached_generation_roots=[p/'artifact/run/generation' for p in sorted(
                (job.parent/'.attempts'/candidate_id).glob('*')) if p.is_dir()])
        state.update(state='done' if result.get('state') == 'complete' else 'failed', result=result)
    except Exception as error:
        import logging
        logging.getLogger(__name__).exception('Object generation failed for %s/%s', run_id, candidate_id)
        state.update(state='failed', error=type(error).__name__)
    write_json(job / 'status.json', state)
    return state


def register_generation(api, service):
    from . import report_workspace
    runs = service.store.root

    def resolve(run_id, candidate_id):
        from .path_safety import validate_safe_path_segment
        for value in (run_id, candidate_id):
            try:
                validate_safe_path_segment(value, 'id')
            except ValueError as error:
                raise HTTPException(400, str(error)) from error
            if value.startswith('.'):
                raise HTTPException(404, 'Unknown object')
        run = runs / run_id
        if not run.is_dir() or not run.resolve().is_relative_to(runs.resolve()):
            raise HTTPException(404, 'Report not found')
        evidence = _read_json_dict(run / 'object-evidence.json')
        candidate = next((c for c in evidence.get('candidates', []) if c['id'] == candidate_id), None)
        if candidate is None:
            raise HTTPException(404, 'Evidence candidate not found')
        return run, candidate, job_root(runs, run_id, candidate_id)

    def public_state(run_id, candidate_id, job):
        source = _read_json_dict(runs/run_id/'object-evidence.json').get('source_sha256')
        return generation_state(run_id, candidate_id, job, source)

    @api.get('/api/reports/{run_id}/generations/{candidate_id}')
    def generation_status(run_id: str, candidate_id: str):
        _, _, job = resolve(run_id, candidate_id)
        return public_state(run_id, candidate_id, job)

    @api.post('/api/reports/{run_id}/generations/{candidate_id}')
    def generate(run_id: str, candidate_id: str):
        import fcntl
        run, candidate, job = resolve(run_id, candidate_id)
        with (run / '.edit.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            run, candidate, job = resolve(run_id, candidate_id)
            if _read_json_dict(run / 'workspace.json').get('state') in {'queued', 'analyzing'}:
                raise HTTPException(409, 'Wait for the source report to complete')
            if candidate['generation']['status'] == 'blocked':
                raise HTTPException(422, candidate['generation']['reason'])
            # The evidence-derived ID is also the idempotency key. Retrying an
            # HTTP request must never create a second paid model invocation.
            source_sha = _read_json_dict(run / 'object-evidence.json').get('source_sha256')
            previous = _read_json_dict(job / 'status.json')
            if previous:
                if previous['state'] not in {'failed','done'} or job_source(previous) == source_sha:
                    return public_state(run_id, candidate_id, job)
            if not os.environ.get('PANOPTES_RESEARCH_ROOT') or not os.environ.get('PANOPTES_RECGEN_ENVIRONMENT'):
                raise HTTPException(503, 'Object generation is not configured')
            if any(_read_json_dict(p).get('state') in {'queued', 'running'} for p in job.parent.glob('*/status.json')):
                raise HTTPException(409, 'This report already has an object generation in progress')
            if previous:
                # New evidence permits an explicit new attempt. Retain the old
                # inputs, failure and GPU ledger instead of overwriting them.
                from uuid import uuid4
                archived = job.parent / '.attempts' / candidate_id / uuid4().hex
                archived.parent.mkdir(parents=True, exist_ok=True)
                job.rename(archived)
            write_json(job / 'status.json', {'state': 'queued', 'run_id': run_id, 'candidate_id': candidate_id, 'source_sha256':source_sha})
            try:
                report_workspace.JOBS.submit(run_generation, runs, run_id, candidate_id)
            except Exception as error:
                write_json(job / 'status.json', {'state': 'failed', 'error': type(error).__name__,
                           'run_id':run_id, 'candidate_id':candidate_id, 'source_sha256':source_sha})
                raise HTTPException(503, 'Could not start generation') from error
            return public_state(run_id, candidate_id, job)

    @api.get('/api/reports/{run_id}/generations/{candidate_id}/assets/{asset:path}')
    @api.get('/api/reports/{run_id}/generations/{candidate_id}/revisions/{revision}/assets/{asset:path}')
    def generation_asset(run_id: str, candidate_id: str, asset: str, revision: str | None = None):
        _, _, job = resolve(run_id, candidate_id)
        if revision is not None:
            choices = [job, *(job.parent/'.attempts'/candidate_id).glob('*')]
            job = next((p for p in choices if _read_json_dict(p/'status.json').get('state') == 'done'
                        and result_revision(_read_json_dict(p/'status.json')) == revision), None)
            if job is None:
                raise HTTPException(404, 'Generated result revision not found')
        if _read_json_dict(job / 'status.json').get('state') != 'done':
            raise HTTPException(404, 'Generation is not complete')
        root = (job / 'artifact/run/result').resolve()
        path = (root / asset).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.suffix.lower() not in {'.json','.glb','.gz','.bin','.png','.jpg','.jpeg','.webp','.html','.js'}:
            raise HTTPException(404, 'Generated asset not found')
        return FileResponse(path)
