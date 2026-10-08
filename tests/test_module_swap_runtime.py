"""Delivery subprocess error boundaries and workcell output consumers."""
import importlib
import json
import os
from pathlib import Path
import sys
import pytest
from argus import ROOT
from argus.pipeline import cli

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

@pytest.mark.parametrize('exit_code,results', [(0, True), (7, True), (0, False)])
def test_checks_require_successful_process_and_results(tmp_path, monkeypatch, exit_code, results):
    monkeypatch.setenv('PANOPTES_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('STAGES_CELL', '030')
    stages = importlib.reload(importlib.import_module('argus.pipeline.run_stages'))
    variant = '030-v2-mvs-fill-sam3d'
    view = tmp_path / 'view.json'
    write(view, {'publication': {'snapshot': {'revision': {'document': {'entities': []}}}}})
    write(tmp_path / f'swap-runs/{variant}-served.json', {'view': str(view), 'photosDir': str(tmp_path), 'photos': '', 'served': str(tmp_path)})
    class Process:
        def __init__(self, command, **kwargs):
            assert command[:3] == [sys.executable, '-m', 'argus.checks.workcell_layer_trial']
            out = Path(command[command.index('--out') + 1]);out.mkdir(parents=True)
            if results:
                write(out / 'results.json', {})
            self.returncode = exit_code
        def wait(self):
            return self.returncode
    monkeypatch.setattr(stages.subprocess, 'Popen', Process)
    if exit_code or not results:
        with pytest.raises(SystemExit):stages.main(variant, ['shape', 'floor', 'lines', 'transfer'])
    else:
        stages.main(variant, ['shape', 'floor', 'lines', 'transfer'])

@pytest.mark.parametrize("cell", ["090", "030"])
def test_generation_and_builder_share_current_workcell_paths(tmp_path, monkeypatch, cell):
    scratch, pages = tmp_path / "data", tmp_path / "pages"
    pages.mkdir()
    for key, value in {"PANOPTES_DATA_ROOT": scratch, "PANOPTES_PAGES": pages,
                       "PANOPTES_RUNS": scratch / "runs", "PY": sys.executable}.items():
        monkeypatch.setenv(key, str(value))
    ctx = cli.Ctx(cell, cli.load_env())
    generation = importlib.reload(importlib.import_module("argus.pipeline.swap_generation"))
    builder = importlib.reload(importlib.import_module("argus.pipeline.build_swap_layer"))
    # Selection is consumed even when generation outputs already exist; no mesh inference is needed for this path check.
    spec = generation.VARIANTS[f"{cell}/mvs-fill-sam3d"]
    assert spec["cands"] == ctx.AB and spec["sel"] == ctx.CMP / "results.json"
    src = spec["src"]
    write(src / "manifest.json", {})
    write(src / "evidence/objects.json", {"objects": [{"object_id": "guard"}]})
    write(src / "evidence/floor.json", {"plane_native": [0, 1, 0, 0]})
    for folder in ("input", "geometry/frames", "evidence/canonical", "evidence/objects"):
        (src / folder).mkdir(parents=True)
    decision = {"code": "measurement.selection.unchanged"}
    write(ctx.CMP / "results.json", {"guard": {"variant": "sam3d", "sam3d": {"coverage": 0.73}, "decision": decision}})
    write(ctx.RUN / "generation/guard/output.json", {"anchor_frame": "frame_0001", "selection": {"candidate": "other-candidate", "decision": "obsolete cached prose"}})
    with pytest.raises(RuntimeError, match='Missing SAM3D selections for required objects'):
        generation.main(f"{cell}/mvs-fill-sam3d")
    # This path fixture contains one object; complete workcell inventories are frozen in the measurement tests.
    monkeypatch.setitem(spec, 'objects', ['guard'])
    with pytest.raises(RuntimeError, match='selected candidate changed'):
        generation.main(f"{cell}/mvs-fill-sam3d")
    write(ctx.RUN / "generation/guard/output.json", {"anchor_frame": "frame_0001", "selection": {"candidate": "sam3d", "decision": "obsolete cached prose"}})
    generation.main(f"{cell}/mvs-fill-sam3d")
    refreshed = json.loads((ctx.RUN / 'generation/guard/output.json').read_text())
    assert refreshed['selection']['decision'] == decision and refreshed['selection']['generationPhoto'] == 'frame_0001'
    assert (ctx.RUN / "generation/swap-record.json").is_file()
    assert builder.cmp_dir(ctx.V) == ctx.CMP
    view = tmp_path / "view.json"
    write(view, {"publication": {"snapshot": {"revision": {"document": {"coordinateFrames": [{"id": "world"}]}}}}})
    write(scratch / f"publications/{ctx.V}/result.json", {"view": str(view), "run": str(ctx.RUN), "nativeToMeters": 1,
                                                                 "publicationId": f"pub-{cell}", "revisionId": "revision"})
    write(ctx.RUN / "result/comparisons.json", {"objects": [{"object_id": "guard", "views": [{"frame_id": "frame_0001",
          "generated_refined": {"visible_iou": 0.9, "relative_depth_p50": None}}], "refinement": {}}]})
    write(scratch / f"swap-runs/{ctx.V}-stages/entity-map.json", {"guard": "guard-eid"})
    for stage in ('box_faces', 'floor', 'shape', 'obvious_errors', 'lower_edge'):
        write(scratch / f'swap-runs/{ctx.V}-stages/{stage}/results.json', {'result': {}})
    write(scratch / 'pipeline/field-values-mvs-fill.json', {'mvs-fill': {}})
    with pytest.raises(RuntimeError, match='Missing required workcell objects in measurements'):
        builder.build(ctx.V)
    write(tmp_path / f'argus/pipeline/cells/{cell}.json', {'objects': ['guard']})
    monkeypatch.setattr(builder, 'ROOT', tmp_path)
    builder.build(ctx.V)
    layer = json.loads((pages / f"pub-{cell}.json").read_text())
    assert layer["schemaVersion"] == 2
    assert layer["variant"]["label"]["code"] == "measurement.variant"
    assert "73" in json.dumps(layer["pipelines"]["guard-eid"]["stages"][0]["text"])

def test_capture_report_uses_pack_and_geometry_helpers(tmp_path):
    import gzip
    import hashlib
    import numpy as np
    from PIL import Image
    from argus.pipeline.build_capture_report import build
    frame = 'frame_0001'
    native = tmp_path / 'geometry/frames' / frame
    native.mkdir(parents=True)
    result = tmp_path / 'result';result.mkdir()
    Image.new('RGB', (8, 8), '#608088').save(native / 'canonical.png')
    image = native / 'canonical.png'
    K, C = np.eye(3), np.eye(4);C[:3, 2] = [0, 1, 0];C[:3, 1] = [0, 0, -1]
    np.save(native / 'intrinsics.npy', K);np.save(native / 'camera_to_world.npy', C)
    mask = np.ones((8, 8), bool)
    (tmp_path / 'evidence').mkdir();np.save(tmp_path / 'evidence/mask.npy', mask)
    write(tmp_path / 'manifest.json', {'experiment': 'synthetic', 'frames': [{'frame_id': frame, 'input': str(image.relative_to(tmp_path)),
          'canonical': str(image.relative_to(tmp_path)), 'width': 8, 'height': 8, 'sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
          'input_to_canonical_pixel_centres': K.tolist()}]})
    write(tmp_path / 'evidence/objects.json', {'objects': [{'object_id': 'guard', 'views': [{'frame_id': frame, 'canonical_mask_path': 'evidence/mask.npy'}]}]})
    write(tmp_path / 'evidence/floor.json', {'plane_native': [0, 0, 1, 0]})
    vertices = np.array([[0, 0, 0, 0, 0, 1, 1, 0, 0], [1, 0, 0, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 0, 1, 0, 0, 1]], '<f4')
    raw = vertices.tobytes() + np.array([0, 1, 2], '<u4').tobytes();(result / 'scene.bin').write_bytes(raw)
    write(result / 'scene.json', {'run_id': 'synthetic', 'cameras': [{'id': frame, 'width': 8, 'height': 8, 'K': K.tolist(), 'camera_to_world': C.tolist(), 'image': f'images/{frame}.png'}],
          'objects': [{'id': 'guard', 'label': 'Guard', 'source': 'generated', 'frame_ids': [frame], 'transform': {'position': [0, 0, 0], 'rotation_deg': [0, 0, 90], 'scale': [1, 1, 1]},
          'mesh': {'byte_offset': 0, 'vertex_count': 3, 'stride': 9, 'index_byte_offset': vertices.nbytes, 'index_count': 3, 'index_type': 'uint32'}}]})
    build(tmp_path, 'Synthetic report')
    scene = json.loads((tmp_path / 'public/scene.json').read_text());report = json.loads((tmp_path / 'public/report.json').read_text())
    assert gzip.decompress((tmp_path / 'public' / scene['objects'][0]['mesh']['asset']['path']).read_bytes()) == raw
    assert report['objects'][0]['views'][0]['bbox'] == [0, 0, 8, 8]
    assert np.allclose(report['objects'][0]['plan']['center'], [-1/3, 1/3])

@pytest.mark.parametrize('code,capped', [('measurement.cap.coverage', True), ('measurement.cap.residual', True), ('measurement.reason.dimension', False)])
def test_obvious_stage_consumes_language_neutral_box_caps(monkeypatch, code, capped):
    import numpy as np
    from argus.checks import obvious_errors as ob
    box = {'centerNative': [0, 0, .1], 'sizeM': [.2, .2, .2], 'axes': np.eye(3).tolist(), 'floorContact': True,
           'bottomM': 0, 'dims': {k: {'confidence': 'high'} for k in ('L', 'W', 'H', 'bottom')}, 'highlightReasons': [{'code': code, 'params': {}}]}
    ctx = {'floor': (np.array([0, 0, 1]), 0), 'S': 1, 'doc': {'entities': [{'id': 'guard'}]}, 'objects': [{'id': 'guard', 'label': 'Guard', 'mesh': ob._box([0, 0, 0], [.2, .2, .2])}], 'layer': {'boxes': {'guard': box}}}
    monkeypatch.setattr(ob, 'g1_floor', lambda *args: {'failed': [], 'warnings': []})
    monkeypatch.setattr(ob, 'g8_scale', lambda *args: {'failed': [], 'warnings': []})
    monkeypatch.setattr(ob, 'g4_pairs', lambda *args: ([], {}))
    monkeypatch.setattr(ob, 'g5_iou', lambda *args, **kwargs: ({'guard': {1: 1}}, {'guard': {}}, {}))
    out = ob.run(ctx, {'draw': False})
    assert out['objects']['guard']['box']['capped'] is capped


@pytest.mark.parametrize('alternative_valid', [False, True])
def test_sam3d_all_views_require_one_real_candidate_per_object(tmp_path, monkeypatch, alternative_valid):
    import numpy as np
    source, output = tmp_path / 'capture', tmp_path / 'candidates'
    write(source / 'manifest.json', {'frames': [{'frame_id': name} for name in ('frame_0001', 'frame_0002')]})
    write(source / 'evidence/objects.json', {'objects': [{'object_id': oid, 'views': [
        {'frame_id': 'frame_0001', 'mask_pixels': 20}, {'frame_id': 'frame_0002', 'mask_pixels': 10}]} for oid in ('guard', 'robot')]})
    for key, value in {'PANOPTES_DATA_ROOT': tmp_path, 'AB_RUN': source, 'AB_OUT': output, 'AB_OBJECTS': 'guard,robot'}.items():
        monkeypatch.setenv(key, str(value))
    completion = importlib.reload(importlib.import_module('argus.pipeline.completion'))
    from argus.providers import sam3d
    monkeypatch.setattr(completion, 'sam3d_inputs', lambda obj, manifest, frame_id=None:
                        (frame_id, None, None, {'frame_id': frame_id or 'frame_0001'}))
    def generate(rgb, mask, pointmap, seed):
        assert seed == 42
        if not alternative_valid or rgb != 'frame_0002':
            return {'error': 'provider failed', 'seconds': 0}
        return {'vertices': np.eye(3), 'faces': np.array([[0, 1, 2]]), 'colors': np.ones((3, 3)),
                'object_to_camera_p3d': np.eye(4), 'seconds': 0, 'gpu': 'test', 'pins': {}}
    monkeypatch.setattr(sam3d, 'generate', generate)
    if alternative_valid:
        completion.stage_sam3d('all')
        for oid in ('guard', 'robot'):
            assert (output / 'sam3d-frame_0002' / f'{oid}.npz').is_file()
    else:
        with pytest.raises(RuntimeError, match='guard, robot'):
            completion.stage_sam3d('all')
        assert (output / 'sam3d/record.json').is_file()
        assert not list(output.glob('sam3d*/*.npz'))
