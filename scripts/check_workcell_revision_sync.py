"""One revision, one set of facts. Run: PYTHONPATH=.:scripts:modal_apps python scripts/check_workcell_revision_sync.py

Synthetic four-photo run through the actual finalize -> estimate -> build ->
semantic bind tail and the candidate-model install. Counterexamples: same ID
with another GLB hash, floor-only change, scale-only change and null scale,
main versus candidate revision, and changed semantic sources. Stale endpoints
must be rejected; after finalize, card values, model vertices, feet, semantic
binding and export metadata must all name the same revision.
"""
import base64
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import trimesh

from ehs_spatial.measurements import measure_observed_points
from scripts.workcell_bottom_models import install_candidate_models
from scripts.workcell_photo_oneshot import _export_metric_scene
from scripts.workcell_photo_report import build, finalize
from scripts.workcell_semantic_match import ENCODERS, analyze, prepare, sha256, write_json

NORMAL = np.array([.12, -.2, 1.]); NORMAL /= np.linalg.norm(NORMAL)
OFFSET = -.3
FLOOR = trimesh.geometry.align_vectors(NORMAL, [0, 0, 1]); FLOOR[2, 3] = OFFSET
NATIVE = np.linalg.inv(FLOOR)  # report floor frame (Z-up, z=0 ground) -> native world
SHAPE = (24, 24)
K = np.array([[20., 0, 12], [0, 20, 12], [0, 0, 1]])
# Every photo looks along report -Y from y=+5: report -X is image right.
CAMERA = np.eye(4); CAMERA[:3, :3] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]; CAMERA[:3, 3] = [0, 5, .8]
BOXES = {'box-1': ([-1., 0, .36], [.1, .05, 1.6]), 'box-2': ([1., 0, .42], [.1, .05, 1.6])}
RAILS = {'section-0-continued-3': ([-1.8, .05, .27], [1.4, .02, .05]),
         'section-1-continued-45': ([1.8, .05, .355], [1.4, .02, .05])}
POLYGONS = {'post-box-1': [[14, 2], [18, 2], [18, 10], [14, 10]], 'post-box-2': [[5, 2], [9, 2], [9, 10], [5, 10]],
            'fence-0': [[17, 14], [21, 14], [21, 20], [17, 20]], 'fence-1': [[2, 14], [6, 14], [6, 20], [2, 20]]}


def encoded(array):
    array = np.ascontiguousarray(array)
    return {'shape': list(array.shape), 'dtype': str(array.dtype), 'data': base64.b64encode(array.tobytes()).decode()}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def native_box(bottom_center, extents):
    mesh = trimesh.creation.box(extents)
    mesh.apply_translation([bottom_center[0], bottom_center[1], bottom_center[2] + extents[2] / 2])
    mesh.apply_transform(NATIVE)
    return mesh


def write_scene(path, boxes):
    scene = trimesh.Scene()
    for node, (center, extents) in boxes.items():
        scene.add_geometry(native_box(center, extents), node_name=node, geom_name=node)
    scene.export(path)


def must_fail(action, message):
    try:
        action()
    except ValueError as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('Accepted: ' + message)


def base_run(root):
    root.mkdir(parents=True)
    pose = NATIVE @ CAMERA
    yy, xx = np.indices(SHAPE)
    local = np.stack([(xx - K[0, 2]) / K[0, 0] * 5, (yy - K[1, 2]) / K[1, 1] * 5, np.full(SHAPE, 5.)], -1)
    points = (local @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    for photo in range(1, 5):
        rgb = np.full((*SHAPE, 3), 120 + 10 * photo, np.uint8)
        for color, polygon in zip(((240, 200, 30), (230, 190, 40), (90, 90, 90), (70, 80, 90)), POLYGONS.values()):
            cv2.fillPoly(rgb, [np.asarray(polygon, np.int32)], color)
        cv2.imwrite(str(root / f'photo-{photo}.png'), rgb[..., ::-1])
        frame = {'image': encoded(rgb), 'pts3d': encoded(points), 'non_ambiguous_mask': encoded(np.ones(SHAPE, bool)),
                 'camera_poses': encoded(pose.astype(np.float32)), 'intrinsics': encoded(K.astype(np.float32)),
                 'input_mask_transform': {'input_to_canonical_pixel_centres': np.eye(3).tolist()}}
        with gzip.GzipFile(root / f'frame_{photo:04d}.json.gz', 'wb', mtime=0) as stream:
            stream.write(json.dumps(frame).encode())
    write_scene(root / 'posts.glb', BOXES)
    write_scene(root / 'fence-fitted.glb', RAILS)
    ground = {'normal': NORMAL.tolist(), 'offset': OFFSET}
    geometry = {'floor': {**ground, 'status': 'fixture floor'},
                'anchor': {'assumedHeightM': .1, 'assumedWidthM': .085, 'mPerNative': None, 'referenceFit': {'status': 'unsupported', 'mPerNative': None}},
                'fence': {'continuations': [{'id': 'section-0-continued-3', 'plane': 0, 'role': 'lower-rail continuation'},
                                            {'id': 'section-1-continued-45', 'plane': 1, 'role': 'observed lower-envelope hypothesis'}]},
                'physicalClearances': {'ground': ground, 'objects': []}}
    (root / 'geometry.json').write_text(json.dumps(geometry))
    (root / 'physical-clearances.json').write_text(json.dumps({'ground': ground, 'objects': []}))
    models = {'post-box-1': ('yellow safety post', 'posts.glb', ['box-1']), 'post-box-2': ('yellow safety post', 'posts.glb', ['box-2']),
              'fence-0': ('safety fence', 'fence-fitted.glb', ['section-0-continued-3']), 'fence-1': ('safety fence', 'fence-fitted.glb', ['section-1-continued-45'])}
    objects = []
    for ident, (kind, file, nodes) in models.items():
        polygon = POLYGONS[ident]
        support = np.zeros(SHAPE, np.uint8); cv2.fillPoly(support, [np.asarray(polygon, np.int32)], 1)
        # Objects-stage visible measurement on the objects-stage floor, as workcell_photo_objects records it.
        measured = measure_observed_points(points[support.astype(bool)].astype(float), {'floor_plane': [*NORMAL, OFFSET]}, mask_pixels=int(support.sum()))
        objects.append({'id': ident, 'kind': kind, 'label': kind, 'model': {'file': file, 'nodes': nodes}, 'representation': 'fixture model',
                        'notes': [], 'measurements': {},
                        'observations': [{'photo': photo, 'source': f'SAM: {kind}; instance {ident}', 'polygons': [polygon],
                                          'box': [polygon[0][0], polygon[0][1], polygon[2][0] + 1, polygon[2][1] + 1],
                                          'observedMeasurements': {**measured, 'source': {'photo': photo}}} for photo in range(1, 5)]})
    (root / 'objects.json').write_text(json.dumps({'objects': objects, 'coverage': {}}))


def add_semantics(root):
    """Real prepare/analyze over the fixture with fixed synthetic embeddings (no neural inference)."""
    out = root / 'semantic-experiment'
    config = {'classes': [{'id': 'light_curtain', 'label': 'Light curtain', 'phrase': 'a light curtain'},
                          {'id': 'fence', 'label': 'Fence', 'phrase': 'a fence'}],
              'queries': [{'id': 'curtain', 'label': '寻找光幕', 'phrase': 'a light curtain'}],
              'referenceClasses': {'post-box-1': 'light_curtain', 'fence-0': 'fence'},
              'overlapNative': .04, 'overlapThreshold': .35, 'semanticThreshold': .75}
    prepare(root, out, config)
    manifest = json.loads((out / 'manifest.json').read_text())
    ids = [row['observationId'] for row in manifest['observations']]
    vectors = np.array([[1., 0.] if row['entityId'].startswith('post') else [0., 1.] for row in manifest['observations']])
    for key in ENCODERS:
        np.savez_compressed(out / f'embeddings-{key}.npz', plain=vectors, masked=vectors, **{'global': np.array([[.7, .7]] * 4)},
                            classText=np.eye(2), queryText=np.array([[1., 0.]]), observationIds=np.array(ids),
                            photoIds=np.array([1, 2, 3, 4]), classPhrases=np.array(['a light curtain', 'a fence']),
                            queryPhrases=np.array(['a light curtain']), manifestSha256=np.array(sha256(out / 'manifest.json')))
        write_json(out / f'timing-{key}.json', {'syntheticSelfCheck': True})
    analyze(root, out, config)
    write_json(out / 'spend-ledger.json', {'status': 'completed', 'functionSeconds': 1., 'callSeconds': 2., 'estimateUsd': 0.,
                                           'callWindowEstimateUsd': 0., 'actualBilledUsd': None})


def copy(source, target):
    shutil.copytree(source, target)
    return target


def endpoints(report):
    return {row['objectId']: row for row in report['endpointEstimation']['endpoints']}


def consistent(root, report):
    """Card values, model vertices, feet, semantic binding and export name one revision."""
    revision, rows = report['revision'], endpoints(report)
    assets = {asset['id']: asset['sha256'] for asset in revision['document']['assets']}
    entities = {entity['id']: entity for entity in revision['document']['entities']}
    for row in rows.values():
        rep = next(r for r in entities[row['objectId']]['representations'] if r['id'] == entities[row['objectId']]['activeModelRepresentationId'])
        assert (row['representationId'], row['assetId'], row['assetSha256']) == (rep['id'], rep['assetId'], assets[rep['assetId']])
        assert sha(root / report['assetURLs'][rep['assetId']]) == row['assetSha256'] and sha(root / row['modelFile']) == row['modelSha256']
        assert abs(row['footNative'][2]) < 1e-9 and np.allclose(np.subtract(row['pointNative'], row['footNative']), [0, 0, row['heightNative']])
        if row['measurementScope'] != 'model_lower_rail_near_curtain':
            mesh = trimesh.load(root / report['assetURLs'][rep['assetId']], force='mesh', process=False)
            heights = mesh.vertices[:, 2] + rep['transform']['position'][2]
            low, high = row['bottomFaceHeightRangeNative']
            assert np.isclose(heights.min(), low, atol=1e-6) and low - 1e-9 <= row['heightNative'] <= high + 1e-9, 'exported curtain mesh reads back the endpoint face'
            if high - low < 1e-9:  # a level face: one exported vertex carries the endpoint height exactly
                assert np.min(np.abs(heights - row['heightNative'])) < 1e-6, 'exported curtain vertex reads back the endpoint'
    tilts = 0
    for entity in revision['document']['entities']:
        for evidence in entity.get('measurements', {}).get('orientationEvidence', {}).values():
            if evidence.get('axisNative') is not None:
                axis = np.asarray(evidence['axisNative']); tilts += 1
                assert np.isclose(np.degrees(np.arccos(min(1., abs(axis[2]) / np.linalg.norm(axis)))), evidence['valueDeg'], atol=1e-6), \
                    "orientation is measured against this revision's floor"
                assert evidence['floor'] == "this revision's floor"
    assert tilts, 'the fixture exercises orientation evidence'
    floor = json.loads((root / 'geometry.json').read_text())['floor']
    up = np.asarray(floor['normal'], float) / np.linalg.norm(floor['normal'])
    for item in report['objects']:
        used = [np.asarray(o['observedMeasurements']['basis']['axes_native'][2]) for o in item['observations'] if (o.get('observedMeasurements') or {}).get('basis')]
        if used:
            angle = max(np.degrees(np.arccos(np.clip(abs(u @ up) / np.linalg.norm(u), 0, 1))) for u in used)
            assert np.isclose(item['visibleExtentFloor']['angleToRevisionFloorDeg'], angle, atol=1e-9), 'visible extents state the floor they were measured on'
    text = json.dumps(report['endpointEstimation']) + json.dumps(report.get('semanticExperiment', {}))
    assert 'estimateCm' not in text and 'valueCm' not in text and ' cm' not in text, 'no centimetre snapshot is stored'
    if 'semanticExperiment' in report:
        binding = report['semanticExperiment']['binding']
        assert (binding['revisionId'], binding['documentSha256']) == (revision['id'], revision['documentSha256'])
        assert all('facts' not in row['policyContext'] for row in report['semanticExperiment']['objects'])
    from scripts.workcell_policy_evidence import policy_evidence
    evidence = policy_evidence(report, json.loads((root / 'objects.json').read_text())['objects'])
    assert (evidence['revisionId'], evidence['documentSha256']) == (revision['id'], revision['documentSha256'])
    assert all(item['machineResult'] is None and item['applicability'] == 'unknown' for item in evidence['items']), 'no verdict without applicability'
    _export_metric_scene(root, report)
    name = next(n for n in ('workcell-conditional.glb', 'workcell-metric.glb', 'workcell-native.glb') if (root / n).is_file())
    with open(root / name, 'rb') as stream:
        data = stream.read()
    head = json.loads(data[20:20 + int.from_bytes(data[12:16], 'little')])
    extras = head['scenes'][head.get('scene', 0)].get('extras', {})
    assert (extras['reportRevision'], extras['documentSha256']) == (revision['id'], revision['documentSha256']), extras
    return rows


def check():
    with tempfile.TemporaryDirectory(prefix='workcell-revision-sync-') as directory:
        directory = Path(directory)
        base = directory / 'base'
        base_run(base)
        finalize(base)
        add_semantics(base)
        main = copy(base, directory / 'workcell-main')
        report = finalize(main)
        rows = consistent(main, report)
        assert report['semanticExperiment']['binding']['reuse'] == 'reused on a revision with identical semantic inputs'
        assert report['semanticExperiment']['sourceRevisionId'] == 'base', 'the experiment keeps the revision it ran on'
        assert np.isclose(rows['post-box-1']['heightNative'], .36) and np.isclose(rows['post-box-2']['heightNative'], .42)
        assert np.isclose(rows['fence-0']['heightNative'], .27) and np.isclose(rows['fence-1']['heightNative'], .355)
        assert rows['post-box-1']['side'] == 'right' and rows['fence-1']['side'] == 'left'
        assert report['modelMeasurementScale']['nativeToMeters'] is None, 'null scale builds without a centimetre crash'

        # (a) Same object ID, another GLB hash: the old endpoint is refused until re-measured.
        moved = copy(main, directory / 'same-id-other-hash')
        write_scene(moved / 'posts.glb', {**BOXES, 'box-1': ([-1., 0, .4], BOXES['box-1'][1])})
        files = lambda root: {path.relative_to(root): sha(path) for path in root.rglob('*') if path.is_file()}
        previous = files(moved)
        must_fail(lambda: build(moved), 'stale: posts.glb')
        assert files(moved) == previous, 'a refused build leaves the report and every asset it hashed untouched'
        partial = copy(main, directory / 'finalize-refused')
        write_scene(partial / 'posts.glb', {**BOXES, 'box-1': ([-1., 0, .4], BOXES['box-1'][1])})
        data = json.loads((partial / 'objects.json').read_text())
        data['objects'][2]['observations'][0]['polygons'] = [[[1, 1], [2, 2]]]
        (partial / 'objects.json').write_text(json.dumps(data))
        previous = files(partial)
        must_fail(lambda: finalize(partial), 'Invalid saved source polygon')
        assert files(partial) == previous, 'a refused finalize leaves the endpoint table and every derived output untouched'
        (moved / 'entity-retired-object.glb').write_bytes(b'stale mesh of an earlier build')
        changed = finalize(moved)
        assert not (moved / 'entity-retired-object.glb').exists(), 'meshes the revision no longer references are removed'
        assert np.isclose(endpoints(changed)['post-box-1']['heightNative'], .4)
        assert endpoints(changed)['post-box-1']['assetSha256'] != rows['post-box-1']['assetSha256']
        assert changed['revision']['documentSha256'] != report['revision']['documentSha256']
        assert 'semanticExperiment' in changed, 'model geometry is not a semantic input'
        consistent(moved, changed)

        # (b) Floor only: heights and feet are recomputed; semantics on another transform are not reused.
        floor = copy(main, directory / 'floor-only')
        for name in ('geometry.json', 'physical-clearances.json'):
            data = json.loads((floor / name).read_text())
            for target in ([data['floor'], data['physicalClearances']['ground']] if name == 'geometry.json' else [data['ground']]):
                target['offset'] = OFFSET - .02
            (floor / name).write_text(json.dumps(data))
        must_fail(lambda: build(floor), 'stale')
        lowered = finalize(floor)
        lowered_rows = consistent(floor, lowered)
        for ident, row in rows.items():
            assert np.isclose(lowered_rows[ident]['heightNative'], row['heightNative'] - .02, atol=1e-6), ident
        assert 'semanticExperiment' not in lowered and lowered['semanticBinding']['status'] == 'not_bound'
        assert 'floor transform' in lowered['semanticBinding']['reason']
        # A tilted floor re-measures visible extents and orientation; a stale catalog value is never shown.
        tilted = copy(main, directory / 'floor-tilted')
        rotation = trimesh.transformations.rotation_matrix(np.radians(2.), [1, 0, 0])[:3, :3]
        for name in ('geometry.json', 'physical-clearances.json'):
            data = json.loads((tilted / name).read_text())
            for target in ([data['floor'], data['physicalClearances']['ground']] if name == 'geometry.json' else [data['ground']]):
                target['normal'] = (rotation @ NORMAL).tolist()
            (tilted / name).write_text(json.dumps(data))
        retilted = finalize(tilted)
        consistent(tilted, retilted)
        angles = [item['visibleExtentFloor']['angleToRevisionFloorDeg'] for item in retilted['objects'] if item['visibleHeightByPhoto']]
        expected = np.degrees(np.arccos(np.clip(NORMAL @ (rotation @ NORMAL), -1, 1)))
        assert angles and np.allclose(angles, expected, atol=1e-6) and expected > 1.9, 'extents keep their objects-stage support and state its floor against this one'
        # consistent() checked each angle against this floor exactly; here the objects-stage values must actually differ.
        evidence = [row for entity in retilted['revision']['document']['entities'] for row in entity.get('measurements', {}).get('orientationEvidence', {}).values()
                    if row.get('valueDeg') is not None]
        assert evidence and any(abs(row['valueDeg'] - row['objectsStageValueDeg']) > 1. for row in evidence), 'orientation is re-referenced to the tilted floor'

        # (c) Scale only: native endpoints identical, the revision's scale changes, nothing stored in cm.
        scaled = copy(main, directory / 'scale-only')
        geometry = json.loads((scaled / 'geometry.json').read_text())
        geometry['anchor'].update(mPerNative=.5, referenceFit={'status': 'available', 'mPerNative': .5, 'camerasFixed': True})
        (scaled / 'geometry.json').write_text(json.dumps(geometry))
        must_fail(lambda: build(scaled), 'stale: geometry.json')
        rescaled = finalize(scaled)
        assert rescaled['modelMeasurementScale']['nativeToMeters'] == .5
        strip = lambda value: {key: item for key, item in value.items() if key != 'sourceFiles'}
        assert strip(rescaled['endpointEstimation']) == strip(report['endpointEstimation'])
        consistent(scaled, rescaled)
        assert [name for name in ('workcell-conditional.glb', 'workcell-metric.glb', 'workcell-native.glb') if (scaled / name).is_file()] == ['workcell-metric.glb'], \
            'the export of the earlier scale status is removed'

        # (d) Main versus candidate: the candidate is its own revision; main files never change.
        candidate_dir = directory / 'candidate-models'; candidate_dir.mkdir()
        lower = trimesh.Trimesh(vertices=[[-1.05, 0, .30], [-.95, 0, .305], [-.95, 0, 2.], [-1.05, 0, 2.],
                                          [-1.05, -.02, .30], [-.95, -.02, .305], [-.95, -.02, 2.], [-1.05, -.02, 2.]],
                                faces=[[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1], [3, 2, 6], [3, 6, 7], [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2]],
                                process=False)
        lower.apply_transform(NATIVE)
        scene = trimesh.Scene(); scene.add_geometry(lower, node_name='post-box-1-closed-extrusion-hypothesis', geom_name='candidate')
        scene.export(candidate_dir / 'post-box-1-volume-candidate.glb')
        candidate = {'ground': {'normal': NORMAL.tolist(), 'offset': OFFSET},
                     'sourceFiles': {name: sha(main / name) for name in ('geometry.json', 'objects.json', 'posts.glb')},
                     'items': [{'id': 'post-box-1', 'status': 'visual_volume_hypothesis', 'physicalValidation': 'none',
                                'model': {'file': 'post-box-1-volume-candidate.glb', 'sha256': sha(candidate_dir / 'post-box-1-volume-candidate.glb'),
                                          'nodes': ['post-box-1-closed-extrusion-hypothesis']},
                                'sourceFaceGateAccepted': False, 'sourceFaceFitGate': {'terminalPartAmbiguity': {'resolved': False}},
                                'sourceObservations': [{'photo': 1, 'source': 'SAM: yellow safety post; instance post-box-1'}],
                                'visibleLowerEdgeVertices': [0, 1], 'visibleLowerEdgeHeightNative': .30,
                                'geometryScope': 'fixture closed extrusion hypothesis', 'thicknessIdentifiable': False, 'searchAtBound': False}]}
        main_hashes = {path.relative_to(main): sha(path) for path in main.rglob('*') if path.is_file()}
        for broken, message in ((lambda value: value['items'][0].update(sourceFaceGateAccepted=True), 'unaccepted'),
                                (lambda value: value['sourceFiles'].update({'objects.json': '0' * 64}), 'stale: objects.json'),
                                (lambda value: value['items'][0]['model'].update(sha256='0' * 64), 'hash differs'),
                                (lambda value: value['items'][0].update(visibleLowerEdgeVertices=[0, 99]), 'does not belong'),
                                (lambda value: value['items'][0].update(visibleLowerEdgeHeightNative=.31), 'recorded candidate height'),
                                (lambda value: value['items'][0]['sourceObservations'][0].update(source='another instance'), 'differs from the catalog')):
            trial = copy(main, directory / f'rejected-{message[:6].replace(" ", "-").replace(":", "")}')
            invalid = deepcopy(candidate); broken(invalid)
            catalog = json.loads((trial / 'objects.json').read_text())
            must_fail(lambda: install_candidate_models(trial, invalid, candidate_dir, catalog), message)
            assert sha(trial / 'objects.json') == main_hashes[Path('objects.json')], 'a rejected candidate writes nothing'
        trial = copy(main, directory / 'rejected-catalog-argument')
        must_fail(lambda: install_candidate_models(trial, candidate, candidate_dir, {'objects': [], 'coverage': {}}), 'catalog argument differs')
        shutil.copyfile(candidate_dir / 'post-box-1-volume-candidate.glb', candidate_dir / 'posts.glb')
        shared = deepcopy(candidate); shared['items'][0]['model']['file'] = 'posts.glb'
        must_fail(lambda: install_candidate_models(trial, shared, candidate_dir, json.loads((trial / 'objects.json').read_text())), 'would overwrite')
        (candidate_dir / 'posts.glb').unlink()
        for generated in ('entity-new-candidate.glb', 'workcell-metric.glb'):
            shutil.copyfile(candidate_dir / 'post-box-1-volume-candidate.glb', candidate_dir / generated)
            named = deepcopy(candidate); named['items'][0]['model'].update(file=generated)
            must_fail(lambda: install_candidate_models(trial, named, candidate_dir, json.loads((trial / 'objects.json').read_text())), 'generated file')
            (candidate_dir / generated).unlink()
        assert files(trial) == main_hashes, 'rejected candidates write nothing'
        accepted = copy(main, directory / 'rejected-over-accepted')
        data = json.loads((accepted / 'objects.json').read_text())
        next(row for row in data['objects'] if row['id'] == 'post-box-1')['physicalBottom'] = {'modelFile': 'posts.glb', 'acceptedFixture': True}
        (accepted / 'objects.json').write_text(json.dumps(data))
        over = deepcopy(candidate); over['sourceFiles']['objects.json'] = sha(accepted / 'objects.json')
        must_fail(lambda: install_candidate_models(accepted, over, candidate_dir, json.loads((accepted / 'objects.json').read_text())), 'accepted physical bottom')
        branch = copy(main, directory / 'workcell-candidate')
        catalog = json.loads((branch / 'objects.json').read_text())
        manifest = install_candidate_models(branch, candidate, candidate_dir, catalog)
        again = copy(branch, directory / 'rejected-second-install')
        must_fail(lambda: install_candidate_models(again, {**candidate, 'sourceFiles': {**candidate['sourceFiles'], 'objects.json': sha(again / 'objects.json')}},
                                                   candidate_dir, json.loads((again / 'objects.json').read_text())), 'would overwrite')
        assert manifest['acceptedForPhysicalUse'] is False and manifest['modelUpdates'] == 1
        (branch / 'revision-lineage.json').write_text(json.dumps({'branchId': 'candidate', 'parentRevisionId': 'workcell-main', 'role': 'candidate', 'label': 'fixture candidate'}))
        alternative = finalize(branch)
        alternative_rows = consistent(branch, alternative)
        assert np.isclose(alternative_rows['post-box-1']['heightNative'], .30) and alternative_rows['post-box-1']['measurementScope'] == 'visible_face_lower_terminal'
        assert alternative_rows['post-box-1']['label'] == '右侧光幕可见面下沿'
        assert np.isclose(alternative_rows['post-box-2']['heightNative'], .42)
        assert alternative['revision']['parentRevisionId'] == 'workcell-main' and alternative['revision']['branchId'] == 'candidate'
        assert alternative['semanticExperiment']['binding']['revisionId'] == 'workcell-candidate'
        assert sha(branch / 'physical-clearances.json') == main_hashes[Path('physical-clearances.json')], 'candidate never becomes a physical clearance'
        item = next(row for row in json.loads((branch / 'objects.json').read_text())['objects'] if row['id'] == 'post-box-1')
        assert item['modelTerminal']['acceptedForPhysicalUse'] is False and 'physicalBottom' not in item
        assert main_hashes == {path.relative_to(main): sha(path) for path in main.rglob('*') if path.is_file()}, 'main revision untouched'
        from scripts.workcell_photo_revisions import build as build_revisions
        must_fail(lambda: build_revisions(branch, directory / 'revisions-from-candidate', directory, main_id='main'), 'candidate revision')
        assert not (directory / 'revisions-from-candidate').exists(), 'a candidate is never relabelled as the main model'

        # (f) Rail pairing: plane identity from geometryPlaneIndex, an adjacency limit, and no left/right of one rail.
        swapped = copy(main, directory / 'plane-identity')
        data = json.loads((swapped / 'objects.json').read_text())
        fences = {row['id']: row for row in data['objects'] if row['kind'] == 'safety fence'}
        fences['fence-0']['model'], fences['fence-1']['model'] = fences['fence-1']['model'], fences['fence-0']['model']
        fences['fence-0']['geometryPlaneIndex'], fences['fence-1']['geometryPlaneIndex'] = 1, 0
        (swapped / 'objects.json').write_text(json.dumps(data))
        swapped_report = finalize(swapped)
        swapped_rows = {row['id']: row for row in consistent(swapped, swapped_report).values()}
        assert np.isclose(swapped_rows['fence-1:near:post-box-1']['heightNative'], .27) and np.isclose(swapped_rows['fence-0:near:post-box-2']['heightNative'], .355)
        far = copy(main, directory / 'rail-not-adjacent')
        write_scene(far / 'posts.glb', {**BOXES, 'box-1': ([-4., 0, .36], BOXES['box-1'][1])})
        far_report = finalize(far)
        light = next(row for row in far_report['endpointEstimation']['endpoints'] if row['id'] == 'post-box-1:terminal')
        assert light['pairingStatus'] == 'no_adjacent_lower_rail' and light['pairedEndpointId'] is None
        ids = {row['id'] for row in far_report['endpointEstimation']['differences']}
        assert ids == {'post-box-2:terminal-minus-rail', 'curtain-left-minus-right'}, ids
        one_rail = copy(main, directory / 'one-rail-both-curtains')
        write_scene(one_rail / 'fence-fitted.glb', {**RAILS, 'section-0-continued-3': ([0., .05, .27], [2.4, .02, .05])})
        one_report = finalize(one_rail)
        rails = [row for row in one_report['endpointEstimation']['endpoints'] if row['measurementScope'] == 'model_lower_rail_near_curtain']
        assert {(row['objectId'], row['node']) for row in rails} == {('fence-0', 'section-0-continued-3')} and len(rails) == 2
        ids = {row['id'] for row in one_report['endpointEstimation']['differences']}
        assert 'rail-left-minus-right' not in ids and 'curtain-left-minus-right' in ids, ids

        # Oneshot semantic stage: the experiment runs on this finished revision, then the tail binds it.
        from unittest.mock import patch
        import scripts.workcell_photo_oneshot as oneshot
        staged = copy(main, directory / 'workcell-staged')
        shutil.rmtree(staged / 'semantic-experiment')
        finalize(staged)  # the local tail always rebuilds under the run's own name first
        ledger = {'runs': []}
        with patch.object(oneshot, '_job', lambda name, argv, out: (add_semantics(staged), {'stage': name})[1]):
            bound = oneshot._semantic_stage(staged, Path('protocol.json'), ledger)
        assert bound['semanticExperiment']['binding']['reuse'] == 'identical semantic inputs verified'
        assert bound['semanticExperiment']['sourceRevisionId'] == 'workcell-staged' and ledger['semanticLedger']['status'] == 'completed'
        failed = copy(main, directory / 'workcell-semantic-failed')
        shutil.rmtree(failed / 'semantic-experiment')
        def failing(name, argv, out):
            write_json(out / 'spend-ledger.json', {'status': 'failed', 'functionSeconds': 3., 'callSeconds': 4., 'estimateUsd': .01,
                                                    'callWindowEstimateUsd': .01, 'actualBilledUsd': None})
            raise RuntimeError('semantic failed (1)')
        ledger = {'runs': []}
        with patch.object(oneshot, '_job', failing):
            unbound = oneshot._semantic_stage(failed, Path('protocol.json'), ledger)
        assert 'semanticExperiment' not in unbound and 'failed experiment' in unbound['semanticBinding']['reason']
        assert ledger['semanticFailure'] == 'semantic failed (1)' and ledger['semanticLedger']['status'] == 'failed', 'failed spend stays recorded'

        # (e) Semantic sources changed: never reuse the earlier result.
        def translate(root, dx, dy):
            # Same shape and pixel count: normalized crops and masks match, only the absolute polygon moved.
            data = json.loads((root / 'objects.json').read_text())
            item = next(row for row in data['objects'] if row['id'] == 'post-box-1')
            item['observations'][1]['polygons'] = [[[x + dx, y + dy] for x, y in polygon] for polygon in item['observations'][1]['polygons']]
            (root / 'objects.json').write_text(json.dumps(data))
        def edit_manifest(root, change, name='manifest.json'):
            path = root / 'semantic-experiment' / name
            manifest = json.loads(path.read_text()); change(manifest); path.write_text(json.dumps(manifest))
        for name, edit, message in (
                ('polygon', lambda root: (root / 'objects.json').write_text((root / 'objects.json').read_text().replace('[[14, 2], [18, 2]', '[[13, 2], [18, 2]', 1)), 'observation input changed'),
                ('translated-1px', lambda root: translate(root, 1, 0), 'input changed (polygon)'),
                ('translated-diagonal', lambda root: translate(root, -2, 3), 'input changed (polygon)'),
                ('catalog-not-carried', lambda root: (root / 'semantic-experiment' / 'source-objects.json').unlink(), 'does not carry the catalog'),
                ('catalog-edited', lambda root: (root / 'semantic-experiment' / 'source-objects.json').write_text('{"objects": []}'), 'does not carry the catalog'),
                ('manifest-polygon-edited', lambda root: edit_manifest(root, lambda m: m['observations'][1].update(polygons=[[[0, 0], [1, 0], [1, 1]]])), 'manifest polygons disagree'),
                ('manifest-unreadable', lambda root: edit_manifest(root, lambda m: m['observations'][0].pop('sourceObservationIndex')), 'missing or unreadable'),
                ('index-null', lambda root: edit_manifest(root, lambda m: m['observations'][0].update(sourceObservationIndex=None)), 'missing or unreadable'),
                ('transform-null', lambda root: edit_manifest(root, lambda m: m.update(sceneTransformNative=None)), 'missing or unreadable'),
                ('observations-null', lambda root: edit_manifest(root, lambda m: m.update(observations=None)), 'missing or unreadable'),
                ('timing-null', lambda root: edit_manifest(root, lambda m: m.update(timing=None), 'semantic-experiment.json'), 'missing or unreadable'),
                ('photo', lambda root: cv2.imwrite(str(root / 'photo-2.png'), np.zeros((*SHAPE, 3), np.uint8)), 'photo changed'),
                ('frame', lambda root: shutil.copyfile(root / 'frame_0001.json.gz', root / 'frame_0003.json.gz'), 'frame changed')):
            stale = copy(main, directory / f'semantic-{name}')
            edit(stale)
            result = finalize(stale)
            assert 'semanticExperiment' not in result and message in result['semanticBinding']['reason'], (name, result.get('semanticBinding'))
        # An experiment recorded before manifests listed polygons binds through the catalog it hashed.
        legacy = copy(main, directory / 'semantic-legacy-manifest')
        edit_manifest(legacy, lambda m: [row.pop('polygons') for row in m['observations']])
        assert finalize(legacy)['semanticExperiment']['binding']['revisionId'] == 'semantic-legacy-manifest'
    print('PASS: same ID/new GLB hash refused then re-measured; floor-only change recomputes heights/feet and unbinds semantics; '
          'oneshot semantic stage binds on its own revision and records failed spend; refused builds leave every file untouched; '
          'tilt is re-referenced to the revision floor and visible extents state their objects-stage floor; refused finalize changes nothing; '
          'rails pair by plane identity within the adjacency limit, one rail is never compared left/right; '
          'candidates never overwrite shared files, accepted bottoms or an earlier candidate; '
          'scale-only change and null scale keep native endpoints; candidate revision self-consistent, never accepted, main untouched; '
          'changed or translated polygon, photo, frame, uncarried catalog or edited manifest never reuses semantics; card/vertex/foot/semantic/export share one revision')


if __name__ == '__main__':
    check()
