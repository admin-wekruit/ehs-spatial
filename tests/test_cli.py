"""Offline delivery CLI boundaries and current two-cell paths."""
import os
import json
from pathlib import Path
import subprocess
import sys
import pytest
from argus import ROOT
from argus.pipeline import cli

def run(*args, env=None):
    return subprocess.run([sys.executable, '-m', 'argus.pipeline.cli', *args], cwd=ROOT, capture_output=True, text=True,
                          env={**os.environ, **(env or {})})

def test_env_uses_one_data_root_and_preserves_explicit_settings(tmp_path, monkeypatch):
    monkeypatch.setenv('PANOPTES_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('PANOPTES_DATABASE_URL', 'postgresql://customer-db/panoptes')
    env = cli.load_env()
    assert env['PANOPTES_DATA_ROOT'] == str(tmp_path) and env['PANOPTES_RUNS'] == str(tmp_path / 'runs')
    assert env['PANOPTES_DATABASE_URL'] == 'postgresql://customer-db/panoptes' and env['PY'] == sys.executable
    assert env['PANOPTES_PAGES'] == str(tmp_path / 'measurement-layer')

@pytest.mark.parametrize('cell', ['090', '030'])
def test_fresh_clone_dry_run_traces_every_stage_without_running_models(tmp_path, cell):
    result = run('run', '--cell', cell, '--dry-run', env={'PANOPTES_DATA_ROOT': str(tmp_path)})
    assert result.returncode == 0, result.stderr
    out = result.stdout
    for stage in cli.STEPS:
        assert stage in out
    for module in ('mvs_route', 'geometry_evaluator', 'fill_geometry', 'completion', 'select_sam3d', 'swap_generation',
                   'pin_run', 'assemble_scene', 'build_capture_report', 'serve_export', 'run_stages', 'build_swap_layer'):
        assert 'argus.pipeline.' + module in out or module == 'completion' and '/argus/pipeline/completion.py' in out
    assert 'argus.platform.publish_capture' in out and f'cmp-{cell}-mvs-fill' in out
    assert f'{cell}-v2-mvs-fill-sam3d' in out and 'AB_FRAME=all' in out
    assert not (tmp_path / 'swap-runs').exists()
    assert 'research/' not in out and 'run_stage.py' not in out
    assert 'pg_ctl' not in out and 'pg_isready' not in out and '55432' not in out

def test_step_selection_and_status(tmp_path):
    env = {'PANOPTES_DATA_ROOT': str(tmp_path)}
    only = run('run', '--cell', '090', '--dry-run', '--only', 'S4c', env=env)
    assert only.returncode == 0 and 'S4c generation' in only.stdout and 'S5' not in only.stdout
    status = run('status', '--cell', '030', env=env)
    assert status.returncode == 0 and 'S2a' in status.stdout and 'todo' in status.stdout and 'always' in status.stdout
    assert run('run', '--cell', '090', '--dry-run', '--only', 'S99', env=env).returncode != 0

def test_http_sam3d_is_a_python_call(tmp_path):
    out = run('run', '--cell', '090', '--dry-run', '--only', 'S4a', env={'PANOPTES_DATA_ROOT': str(tmp_path), 'SAM3D_BACKEND': 'http'}).stdout
    assert '-m argus.pipeline.completion --stage sam3d' in out and 'modal run' not in out

def test_default_geometry_route_uses_shared_provider_outputs(tmp_path, monkeypatch):
    from argus.providers import geometry_mvs
    env = {**cli.load_env(), 'PANOPTES_DATA_ROOT': str(tmp_path)}
    ctx = cli.Ctx('090', env)
    frames = [{'frame_id': 'frame_0001'}]
    monkeypatch.delenv('GEOMETRY_MVS_BACKEND', raising=False)
    monkeypatch.setattr(geometry_mvs, 'frames_for', lambda run: frames)
    def route(cell, selected, *, dest):
        assert cell == '090' and selected is frames and dest == tmp_path
        for path in geometry_mvs.outputs(cell, dest):
            path.mkdir(parents=True)
        (dest / 'checks/clean-gpu/090/moge-frame_0001.npz').touch()
    monkeypatch.setattr(geometry_mvs, 'run', route)
    cli.s2a_route(ctx)
    assert ctx.backend == 'modal' and cli.ITEMS[0].done(ctx)
    assert all(p.exists() for p in cli.ITEMS[1].inputs(ctx))


def test_publish_uses_configured_database_without_provisioning(tmp_path, monkeypatch):
    env = {**cli.load_env(), 'PANOPTES_DATA_ROOT': str(tmp_path), 'PANOPTES_DATABASE_URL': 'mongodb://customer-db/panoptes'}
    ctx = cli.Ctx('090', env)
    calls = []
    def execute(argv, **kwargs):
        assert argv[:3] == [sys.executable, '-m', 'argus.platform.publish_capture']
        assert kwargs['env']['PANOPTES_DATABASE_URL'] == 'mongodb://customer-db/panoptes'
        calls.append(argv)
    monkeypatch.setattr(cli.subprocess, 'run', execute)
    cli.s7_publish(ctx)
    assert len(calls) == 1


def test_default_data_root_belongs_to_calling_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv('PANOPTES_DATA_ROOT', raising=False)
    monkeypatch.chdir(tmp_path)
    assert cli.load_env()['PANOPTES_DATA_ROOT'] == str(tmp_path / 'data')


def test_relative_paths_are_resolved_before_child_cwd_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key, value in {'PANOPTES_DATA_ROOT': './data', 'PANOPTES_RUNS': './captures', 'PANOPTES_PAGES': './pages'}.items():
        monkeypatch.setenv(key, value)
    env = cli.load_env()
    assert [env[key] for key in ('PANOPTES_DATA_ROOT', 'PANOPTES_RUNS', 'PANOPTES_PAGES')] == [str(tmp_path / name) for name in ('data', 'captures', 'pages')]


def test_sam3d_done_requires_a_candidate_for_every_configured_object(tmp_path):
    import numpy as np
    ctx = cli.Ctx('030', {**cli.load_env(), 'PANOPTES_DATA_ROOT': str(tmp_path)})
    item = next(i for i in cli.ITEMS if i.name == 'sam3d')
    dest = ctx.AB / 'sam3d';dest.mkdir(parents=True)
    (dest / 'record.json').write_text(json.dumps({'objects': {oid: {'error': 'provider failed'} for oid in ctx.OBJS}}))
    assert not item.done(ctx)
    alternative = ctx.AB / ctx.FRAMES[-1];alternative.mkdir()
    for oid in ctx.OBJS[:-1]:
        np.savez(alternative / f'{oid}.npz', vertices=np.ones((3, 3)))
    assert not item.done(ctx)
    np.savez(alternative / f'{ctx.OBJS[-1]}.npz', vertices=np.ones((3, 3)))
    assert item.done(ctx)


@pytest.mark.parametrize('cache', ['current', 'old', 'missing', 'invalid'])
def test_selection_cache_requires_current_messages_for_every_object(tmp_path, cache):
    ctx = cli.Ctx('030', {**cli.load_env(), 'PANOPTES_DATA_ROOT': str(tmp_path)})
    item = next(i for i in cli.ITEMS if i.name == 'compare')
    assert not item.done(ctx)
    results = {oid: {'decision': {'code': 'measurement.selection.unchanged'}} for oid in ctx.OBJS}
    if cache == 'old':
        results[ctx.OBJS[0]]['decision'] = 'old English decision'
    elif cache == 'missing':
        results.pop(ctx.OBJS[0])
    ctx.CMP.mkdir(parents=True)
    (ctx.CMP / 'results.json').write_text('{' if cache == 'invalid' else json.dumps(results))
    assert bool(item.done(ctx)) is (cache == 'current')
    assert next(i for i in cli.ITEMS if i.name == 'generation').done is None
