"""A failed correction refresh never publishes half a report or loses its source."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ehs_spatial import report_refresh
from ehs_spatial import object_evidence


@pytest.mark.parametrize('outcome', ['success', 'failure', 'source_changed'])
def test_refresh_publication_is_snapshot_checked_and_keeps_lock(tmp_path, monkeypatch, outcome):
    run = tmp_path/'runs/example'
    run.mkdir(parents=True)
    for name, content in [('report.html','previous report'), ('viewer.html','previous viewer'),
                          ('refinements.json','[{"label":"tiny 2D correction"}]'), ('.edit.lock','')]:
        (run/name).write_text(content)
    (run/'inventory').mkdir()
    (run/'inventory/inventory.json').write_text('{"objects":[]}')
    (run/'observed/old').mkdir(parents=True)
    (run/'observed/old/scene.json').write_text('historical scene')
    (run/'observed-scene.json').write_text('{"revision":"old"}')
    inode = (run/'.edit.lock').stat().st_ino
    def registry(path):
        output = path/'object-evidence.json'
        output.write_text('{"candidates":[{"label":"tiny 2D correction","geometry":{"status":"unmeasured"}}]}')
        return output
    monkeypatch.setattr(object_evidence, 'write_object_evidence', registry)
    def rebuild(command, *, cwd, **kwargs):
        stage = Path(cwd)/'runs/example'
        assert not (stage/'.edit.lock').exists()
        assert json.loads((stage/'refinements.json').read_text())[0]['label'] == 'tiny 2D correction'
        (stage/'report.html').write_text('new report')
        (stage/'viewer.html').write_text('new viewer')
        (stage/'inventory/inventory.json').write_text('{"objects":[{"inv":0}]}')
        (stage/'observed/new').mkdir()
        (stage/'observed/new/scene.json').write_text('new observed scene')
        (stage/'observed-scene.json').write_text('{"revision":"new"}')
        result = {'updated':outcome != 'failure','error':'missing source support'}
        (stage/'report-refresh.json').write_text(json.dumps(result))
        if outcome == 'source_changed':
            (run/'refinements.json').write_text('[{"label":"newer concurrent correction"}]')
        return SimpleNamespace(returncode=1 if outcome == 'failure' else 0)
    monkeypatch.setattr(report_refresh.subprocess, 'run', rebuild)
    result = report_refresh.refresh_report_after_correction(run)
    assert result['updated'] == (outcome == 'success')
    assert (run/'.edit.lock').stat().st_ino == inode
    assert (run/'observed/old/scene.json').read_text() == 'historical scene'
    assert json.loads((run/'object-evidence.json').read_text())['candidates'][0]['geometry']['status'] == 'unmeasured'
    if outcome == 'success':
        assert (run/'report.html').read_text() == 'new report'
        assert (run/'viewer.html').read_text() == 'new viewer'
        assert len(json.loads((run/'inventory/inventory.json').read_text())['objects']) == 1
        assert (run/'observed/new/scene.json').read_text() == 'new observed scene'
        assert json.loads((run/'observed-scene.json').read_text())['revision'] == 'new'
    else:
        assert result['report_preserved']
        assert (run/'report.html').read_text() == 'previous report'
        assert (run/'viewer.html').read_text() == 'previous viewer'
        assert json.loads((run/'inventory/inventory.json').read_text())['objects'] == []
        assert not (run/'observed/new').exists()
        assert json.loads((run/'observed-scene.json').read_text())['revision'] == 'old'
        if outcome == 'source_changed':
            assert 'newer concurrent correction' in (run/'refinements.json').read_text()
