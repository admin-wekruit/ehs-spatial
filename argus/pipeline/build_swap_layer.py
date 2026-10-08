"""Publish schema-2 measurement messages from the retained MVS+fill/SAM3D stage outputs.
Geometry, uncertainty and capture thresholds remain in their original check modules.
Run: python -m argus.pipeline.build_swap_layer 090-v2-mvs-fill-sam3d 030-v2-mvs-fill-sam3d
"""
import json
import os
from pathlib import Path
import shutil
import sys

from argus import ROOT

SP = Path(os.environ['PANOPTES_DATA_ROOT'])
HERE = SP / 'pipeline'
PLATFORM = ROOT
PAGES = Path(os.environ.get('PANOPTES_PAGES', SP / 'measurement-layer'))
OBJECTS = {'left_light_curtain', 'right_light_curtain', 'left_fence', 'right_fence', 'left_post', 'right_post', 'robot', 'cart', 'guard', 'observed_floor'}
PARTS = {'recessed yellow housing': 'recessedHousing', 'yellow front plate': 'frontPlate', 'bottom rail': 'bottomRail'}
FIELD = {'right_light_curtain': ('housing090R', 'housingRight', 24), 'right_fence': ('fence090R', 'fenceRight', 20),
         'left_light_curtain': ('housing090L', 'housingLeftOccluded', 24)}


def message(code, **params):
    return dict(code=code, params=params) if params else dict(code=code)


def joined(messages):
    result = messages[0]
    for item in messages[1:]:
        result = message('message.join', first=result, second=item)
    return result


def failed_message(item):
    gate = item['gate']
    if gate == 'G2':
        return message('measurement.gate.G2', depth=f"{-item['bottomCm']:.1f}")
    if gate == 'G3':
        return message('measurement.gate.G3', height=f"{item['bottomCm']:.1f}")
    if gate == 'G4':
        return message('measurement.gate.G4', other=item['other'][:8], ratio=f"{100 * item['ratio']:.0f}")
    if gate == 'G5':
        return message('measurement.gate.G5', values=' / '.join(f'{v:.2f}' for v in item['iou'].values()))
    if gate == 'G6' and 'fragments' in item:
        return message('measurement.gate.fragments', count=item['fragments'], ratio=f"{100 * item['fragmentFaces']:.1f}")
    if gate == 'G6':
        return message('measurement.gate.exploded', axis=item['axis'], size=f"{item['fullM']:.2f}", factor=3)
    if gate == 'G7':
        return message('measurement.gate.G7', dimension=message(f"box.dim.{item['dim']}"), model=f"{100 * item['modelM']:.1f}",
                       box=f"{100 * item['boxM']:.1f}", error=f"{100 * item['rel']:+.0f}")
    raise ValueError(f'unsupported object failure gate: {gate}')


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def cmp_dir(v):
    return HERE / f"cmp-{v.split('-', 1)[0]}-mvs-fill"


def build(v):
    if v not in ('090-v2-mvs-fill-sam3d', '030-v2-mvs-fill-sam3d'):
        raise ValueError(f'unsupported delivery variant: {v}')
    PAGES.mkdir(parents=True, exist_ok=True)
    res = load(SP / 'publications' / v / 'result.json')
    doc = load(PLATFORM / res['view'] if not Path(res['view']).is_absolute() else res['view'])['publication']['snapshot']['revision']['document']
    run = Path(res['run']); st = SP / 'swap-runs' / f'{v}-stages'
    ents = load(st / 'entity-map.json')                       # object_id -> entity id
    stage = lambda name: json.loads((st / name / 'results.json').read_text())['result']
    boxes, floor, shape, obvious, lower = stage('box_faces').get('boxes') or {}, stage('floor').get('objects') or {}, stage('shape'), stage('obvious_errors'), stage('lower_edge')
    shape_rows = {r['entityId']: r for r in shape.get('rows', [])}
    comps = {c['object_id']: c for c in load(run / 'result/comparisons.json')['objects']}
    required = set(json.loads((ROOT / 'argus/pipeline/cells' / f"{v.split('-', 1)[0]}.json").read_text())['objects'])
    missing = required - comps.keys() | required - ents.keys()
    if missing:
        raise RuntimeError(f'Missing required workcell objects in measurements: {sorted(missing)}')
    fl = load(run / 'evidence/floor.json'); S = res['nativeToMeters']
    fv = json.loads((HERE / 'field-values-mvs-fill.json').read_text())['mvs-fill']
    field = {'right_light_curtain': ('housing030R', 'housingRight', 24), 'left_light_curtain': ('housing030L', 'housingLeft', 24)} if v.startswith('030-') else FIELD
    cmp = load(cmp_dir(v) / 'results.json')
    version = v.split('-')[1].removeprefix('v')
    frame = doc['coordinateFrames'][0]
    layer = {'schemaVersion': 2, 'publicationId': res['publicationId'], 'revisionId': res['revisionId'], 'coordinateFrameId': frame['id'],
             'scale': {'nativeToMeters': S, 'status': 'operator_anchored', 'source': message('measurement.scale')},
             'ground': {'normal': fl['plane_native'][:3], 'offset': fl['plane_native'][3], 'plane': fl['plane_native'], 'source': message('measurement.ground')},
             'variant': {'id': v, 'label': message('measurement.variant', variant=v, version=version)},
             'labels': {}, 'confidence': {}, 'facts': {}, 'boxes': {}, 'pipelines': {}, 'assets': []}
    for oid, eid in ents.items():
        if oid not in comps:
            raise RuntimeError(f'Measurement entity has no assembly comparison: {oid}')
        label = message('measurement.object.' + oid) if oid in OBJECTS else message('measurement.name', name=oid)
        layer['labels'][eid] = label
        c = comps[oid]; views = {x['frame_id']: x['generated_refined'] for x in c['views']}
        ious = [x['visible_iou'] for x in views.values()]; deps = [x['relative_depth_p50'] for x in views.values() if x['relative_depth_p50'] is not None]
        assembly = [message('measurement.iou', values=' / '.join(f"{views[f]['visible_iou']:.2f}" for f in sorted(views)), mean=f"{sum(ious) / len(ious):.2f}")]
        if deps:
            assembly.append(message('measurement.depthResidual', depth=f"{100 * sum(deps) / len(deps):.1f}"))
        fin = ((c['refinement'].get('final') or {}).get('floor') or {})
        if fin.get('lowest_native') is not None:
            assembly.append(message('measurement.lowest', height=f"{100 * S * fin['lowest_native']:+.1f}"))
        ob = (obvious.get('objects') or {}).get(eid) or {}; fails = [failed_message(x) for x in ob.get('failed', [])]
        fo = floor.get(eid) or {}; sh = shape_rows.get(eid) or {}; bx = boxes.get(eid)
        rec = json.loads((run / 'generation' / oid / 'output.json').read_text()); sel = rec['selection']
        if not isinstance(sel['decision'], dict) or 'code' not in sel['decision']:
            raise ValueError(f'{oid}: selection must use the current message schema; rerun S4b and S4c')
        cm = (cmp or {}).get(oid) or {}
        geometry = [message('measurement.geometry', scale=f'{100 * S:.1f}')]
        if cm.get('sam3d', {}).get('coverage') is not None:
            geometry.append(message('measurement.coverage', coverage=f"{100 * cm['sam3d']['coverage']:.0f}"))
        stages = [{'label': message('measurement.stage.geometry'), 'text': joined(geometry)}]
        completion = [message('measurement.completion')]
        for key, t in (sel.get('tried') or {}).items():
            photo = str(sel.get('generationPhoto', '?')) if key == 'sam3d' else key.removeprefix('sam3d-')
            completion.append(message('measurement.candidate', photo=photo.removeprefix('frame_000'), iou=f"{t['meanIou']:.2f}",
                                      depth=f"{100 * t['meanDepthP50']:.1f}", gates=joined(t['gates']) if t.get('gates') else '—'))
        completion.append(message('measurement.selected', candidate=sel['decision'], photo=str(sel['generationPhoto']).removeprefix('frame_000')))
        stages.append({'label': message('measurement.stage.completion'), 'text': joined(completion)})
        assembled = [message('measurement.assembly', result=joined(assembly))]
        if fin.get('penalty') is not None:
            assembled.append(message('measurement.penalty', penalty=f"{fin['penalty']:.4f}"))
        stages.append({'label': message('measurement.stage.assembly'), 'text': joined(assembled)})
        if sh:
            photo_shape = [message('measurement.shape', verdict=message('measurement.shape.' + sh['verdict']))]
            if sh.get('bestScale'):
                photo_shape.append(message('measurement.bestDepth', depth=f"{100 * (sh.get('bestScale', 1) - 1) * sh.get('distanceToReference', 0) * S:+.0f}"))
            stages.append({'label': message('measurement.stage.shape'), 'text': joined(photo_shape)})
        if fo:
            stages.append({'label': message('measurement.stage.floor'), 'text': message('measurement.floor', height=f"{100 * fo['bottomM']:+.1f}", status=message('measurement.floor.' + fo['status']))})
        le_items = (lower.get('targets') or lower.get('objects') or lower.get('results') or []) if lower else []
        for x in (le_items.values() if isinstance(le_items, dict) else le_items):
            if not isinstance(x, dict) or x.get('entityId') != eid:
                continue
            part = message('measurement.part.' + PARTS[x['part']]) if x.get('part') in PARTS else message('measurement.name', name=x['part']) if x.get('part') else label
            if x.get('valueCm') is not None:
                lower_text = [message('measurement.lower.value', part=part, value=f"{x['valueCm']:.1f}")]
            else:
                lower_text = [message('measurement.lower.unknown', part=part, reason=message('measurement.lower.reason.' + x['reason']))]
                if x.get('estimateCm'):
                    lower_text.append(message('measurement.lower.estimate', value=f"{x['estimateCm']:.1f}", sigma=f"{x['sigmaCm']:.1f}"))
            stages.append({'label': message('measurement.stage.lower'), 'text': joined(lower_text)})
        if bx:
            d = bx['dims']; bx['label'] = label
            confidence = joined([message('measurement.reason.dimension', dimension=message('box.dim.' + k), level=message('box.level.' + d[k]['confidence'])) for k in ('L', 'W', 'H', 'bottom')])
            box_text = [message('measurement.box', length=f"{100 * bx['sizeM'][0]:.0f}", width=f"{100 * bx['sizeM'][1]:.0f}", height=f"{100 * bx['sizeM'][2]:.0f}", bottom=f"{100 * bx['bottomM']:.1f}", confidence=confidence)]
            if bx.get('highlight'):
                box_text.extend(bx.get('highlightReasons') or [])
            stages.append({'label': message('measurement.stage.box'), 'text': joined(box_text)})
        stages.append({'label': message('measurement.stage.gates'), 'text': joined(fails) if fails else message('measurement.passed')})
        pipeline = {'stages': stages, 'caption': message('measurement.caption' if cmp else 'measurement.captionOriginal')}
        sheet = cmp_dir(v) / f'{oid}.jpg'
        if sheet.exists():
            shutil.copy(sheet, PAGES / f'pipeline-{eid[:8]}.jpg'); pipeline['url'] = f'measurement-layer/pipeline-{eid[:8]}.jpg'
        layer['pipelines'][eid] = pipeline
        level = 'low' if fails else (bx or {}).get('confidence', 'unverified')
        layer['confidence'][eid] = {'level': level, 'label': message('box.level.' + level), 'reasons': list((bx or {}).get('highlightReasons') or []), 'missing': fails}
        facts = [{'label': message('measurement.fact.gates'), 'kind': 'check', 'text': joined(fails) if fails else message('measurement.passedDetail')},
                 {'label': message('measurement.fact.assembly'), 'kind': 'note', 'text': joined(assembly)}]
        if oid in field and fv:
            key, name, field_cm = field[oid]; val = fv.get(f'{key}_cm'); err = fv.get(f'{key}_err')
            if val is not None:
                field_text = [message('measurement.field', part=message('measurement.part.' + name), value=val, field=field_cm)]
                if err is not None:
                    field_text.append(message('measurement.field.error', error=f'{err:+}'))
                facts.append({'label': message('measurement.fact.field'), 'kind': 'field', 'text': joined(field_text)})
        layer['facts'][eid] = facts
        if bx:
            layer['boxes'][eid] = bx
    PAGES.mkdir(parents=True, exist_ok=True)
    out = PAGES / f"{res['publicationId']}.json"
    out.write_text(json.dumps(layer, ensure_ascii=False, indent=1) + '\n')
    print(v, res['publicationId'], 'entities', len(layer['pipelines']), 'boxes', len(layer['boxes']), 'low', sum(1 for c in layer['confidence'].values() if c['level'] == 'low'), '->', out.name)


if __name__ == '__main__':
    for v in sys.argv[1:]:
        build(v)
