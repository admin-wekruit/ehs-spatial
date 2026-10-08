"""The display-schema migration must preserve the published physical measurements."""

import copy
import contextlib
import io
import json
import importlib
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.check_measurement_regression import geometry


FIXTURES = Path(__file__).parent / "fixtures" / "measurements"


def test_actual_selection_messages_and_missing_candidate(tmp_path, monkeypatch):
    """Selection branches localize their numeric evidence; a missing object cannot reach publication."""
    run = tmp_path / 'capture'
    (run / 'evidence').mkdir(parents=True)
    (run / 'evidence/objects.json').write_text(json.dumps({'objects': [{'object_id': 'guard', 'views': []}]}))
    (run / 'evidence/floor.json').write_text(json.dumps({'plane_native': [0, 0, 1, 0], 'estopNativeToMeters': 1}))
    for key, value in {'PANOPTES_DATA_ROOT': tmp_path, 'AB_RUN': run, 'AB_OUT': tmp_path / 'candidates', 'CMP_BOXES': 'none'}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.delenv('CMP_SCALE', raising=False)
    selection = importlib.import_module('argus.pipeline.select_sam3d')
    correction = {'code': 'measurement.selection.cut', 'params': {'depth': '1.5'}}
    gate = {'code': 'measurement.selection.depth', 'params': {'depth': '6.2'}}
    cases = [selection.decide({'gates': []}, {'gates': [], 'fix': []}),
             selection.decide({'gates': [gate]}, {'gates': [], 'fix': [correction]}),
             selection.decide({'gates': [gate]}, {'gates': [gate], 'fix': []})]
    assert [case['code'] for case in cases] == ['measurement.selection.' + key for key in ('unchanged', 'corrected', 'measuredBox')]
    assert '1.5' in selection.diagnostic(cases[1]) and '6.2' in selection.diagnostic(cases[2])
    monkeypatch.setattr(sys, 'argv', ['select_sam3d', 'guard'])
    monkeypatch.delitem(sys.modules, 'argus.pipeline.select_sam3d')
    with pytest.raises(RuntimeError, match='required object: guard'):
        runpy.run_module('argus.pipeline.select_sam3d', run_name='__main__')
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is unavailable; selection checks passed, renderer boundary requires Node')
    script = """
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {renderMessage} from './web/src/measurement-layer.ts';
const cases=JSON.parse(readFileSync(0,'utf8'));
for (const lang of ['en','zh','nl']) {
  const rows=cases.map(candidate=>renderMessage(lang,{code:'measurement.selected',params:{candidate,photo:'1'}}));
  assert.match(rows[1],/1\\.5/); assert.match(rows[2],/6\\.2/);
  if(lang!=='en') for(const text of rows) assert.doesNotMatch(text,/unchanged|floor correction|use measured box|depth differs from photo/);
}
"""
    subprocess.run([node, '--experimental-strip-types', '--input-type=module', '-e', script],
                   cwd=Path(__file__).parents[1], input=json.dumps(cases), capture_output=True, text=True, check=True)


def test_box_computation_matches_original_selfcheck(monkeypatch):
    """The retained deterministic scenes must match the pre-migration numeric outputs."""
    from argus.checks import box_faces
    original_run = box_faces.run
    results = []
    fields = ('centerNative', 'axes', 'faceNormals', 'sizeM', 'bottomM', 'topM',
              'floorContact', 'dims', 'highlight', 'confidence')

    def capture(*args, **kwargs):
        result = original_run(*args, **kwargs)
        results.append({entity: {**{key: box[key] for key in fields},
                                'faces': {face: {key: value[key] for key in ('photos', 'status', 'confidence')}
                                          for face, value in box['faces'].items()}}
                        for entity, box in result['boxes'].items()})
        return result

    monkeypatch.setattr(box_faces, 'run', capture)
    with contextlib.redirect_stdout(io.StringIO()):
        box_faces._check()
    assert results == json.loads((FIXTURES / 'box-selfcheck.json').read_text())


@pytest.mark.parametrize("cell,count", [("090", 9), ("030", 8)])
def test_frozen_geometry_contract(cell, count):
    frozen = json.loads((FIXTURES / f"{cell}.json").read_text())
    assert len(frozen["boxes"]) == count
    layer = {**frozen, "confidence": {entity: {"level": level} for entity, level in frozen["confidence"].items()}}
    assert geometry(layer) == frozen
    entity = next(iter(layer["boxes"]))
    changed = copy.deepcopy(layer)
    changed["boxes"][entity]["dims"]["L"]["valueM"] += 0.01
    assert geometry(changed) != frozen
    changed = copy.deepcopy(layer)
    changed["scale"]["nativeToMeters"] += 0.01
    assert geometry(changed) != frozen
    changed = copy.deepcopy(layer)
    changed["boxes"][entity]["dims"]["L"]["sigmaCm"] = 99
    assert geometry(changed) != frozen


@pytest.mark.parametrize('cell', ['090', '030'])
def test_published_geometry_through_schema2_builder(cell, tmp_path, monkeypatch):
    """Replay frozen physical outputs through the real builder; this does not rerun GPU reconstruction."""
    monkeypatch.setenv('PANOPTES_DATA_ROOT', str(tmp_path))
    from argus.pipeline import build_swap_layer as builder
    frozen = json.loads((FIXTURES / f'{cell}.json').read_text())
    source = json.loads((FIXTURES / 'sources.json').read_text())[cell]
    entities = source['entities']; variant = f'{cell}-v2-mvs-fill-sam3d'
    run = tmp_path / 'swap-runs' / cell / 'mvs-fill-sam3d'
    stage = tmp_path / 'swap-runs' / f'{variant}-stages'
    pipeline = tmp_path / 'pipeline'; pages = tmp_path / 'measurement-layer'
    monkeypatch.setattr(builder, 'SP', tmp_path)
    monkeypatch.setattr(builder, 'HERE', pipeline)
    monkeypatch.setattr(builder, 'PAGES', pages)

    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path

    view = write('view.json', {'publication': {'snapshot': {'revision': {'document': {'coordinateFrames': [{'id': source['coordinateFrameId']}]}}}}})
    write(f'publications/{variant}/result.json', dict(view=str(view), run=str(run), nativeToMeters=source['nativeToMeters'], publicationId=source['publicationId'], revisionId=source['revisionId']))
    write(stage.relative_to(tmp_path) / 'entity-map.json', entities)
    boxes = copy.deepcopy(frozen['boxes'])
    for entity, box in boxes.items():
        box['label'] = builder.message('measurement.name', name=entity)
        box['snapNote'] = builder.message('measurement.snap.still') if box['floorContact'] else None
        box['highlightReasons'] = [builder.message('measurement.reason.layer', level=builder.message('box.level.' + box['confidence']))] if box['highlight'] else []
        for face in box['faces'].values():
            face['need'] = None if face['confidence'] == 'high' else builder.message('measurement.need.second', photos='1')
    # G5 preserves the guard's already-low report confidence, while exercising structured gate messages.
    failures = {entity: dict(failed=[dict(gate='G5', iou={'frame_0001': .1})]) for entity, level in frozen['confidence'].items() if level == 'low' and boxes[entity]['confidence'] != 'low'}
    results = {'box_faces': {'boxes': boxes}, 'floor': {'objects': {}}, 'shape': {'rows': []}, 'obvious_errors': {'objects': failures}, 'lower_edge': {}}
    for name, result in results.items():
        write(stage.relative_to(tmp_path) / name / 'results.json', {'result': result})
    write(run.relative_to(tmp_path) / 'evidence/floor.json', {'plane_native': frozen['ground']['plane']})
    write(run.relative_to(tmp_path) / 'result/comparisons.json', {'objects': [dict(object_id=oid, views=[dict(frame_id='frame_0001', generated_refined=dict(visible_iou=.8, relative_depth_p50=.01))], refinement={'final': {'floor': {'lowest_native': 0, 'penalty': .001}}}) for oid in entities]})
    for oid in entities:
        write(run.relative_to(tmp_path) / f'generation/{oid}/output.json', {'selection': {'decision': builder.message('measurement.selection.unchanged'), 'generationPhoto': 'frame_0001'}})
    write('pipeline/field-values-mvs-fill.json', {'mvs-fill': {f'housing{cell}R_cm': 24.6, f'housing{cell}R_err': .6}})
    # A historical row must never replace the explicitly selected geometry field values.
    write('pipeline/results.json', {'fieldValues': {'mvs-fill': {f'housing{cell}R_cm': 999}}})
    builder.build(variant)
    layer = json.loads((pages / (source['publicationId'] + '.json')).read_text())
    assert layer['schemaVersion'] == 2
    assert geometry(layer) == frozen
    right = entities['right_light_curtain']
    field = next(f for f in layer['facts'][right] if f['kind'] == 'field')
    assert field['text']['params']['first']['params']['value'] == 24.6
    catalogs = {lang: json.loads((Path(__file__).parents[1] / 'web/src/locales' / f'{lang}.json').read_text()) for lang in ('en', 'zh', 'nl')}

    def check_messages(value):
        if isinstance(value, dict):
            assert not any(k.endswith('En') for k in value)
            if 'code' in value:
                assert all(value['code'] in catalog for catalog in catalogs.values()), value['code']
            for nested in value.values():
                check_messages(nested)
        elif isinstance(value, list):
            for nested in value:
                check_messages(nested)
    check_messages(layer)
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is unavailable; Python geometry checks passed, viewer boundary check requires Node')
    subprocess.run([node, '--experimental-strip-types', str(Path(__file__).parents[1] / 'web/tests/producer-layer-check.ts'),
                    str(pages / (source['publicationId'] + '.json'))], check=True, capture_output=True, text=True)
