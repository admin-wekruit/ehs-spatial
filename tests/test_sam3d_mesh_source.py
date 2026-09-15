"""CPU regression on exact prepared upstream code; no model inference or fake GS modules.

First run scripts/prepare_sam3d_mesh_source.py --fetch --source
.platform/model-delivery-20260915/sam3d-mesh-source, or set PANOPTES_SAM3D_TEST_SOURCE.
"""
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

preparation = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/prepare_sam3d_mesh_source.py'))
CODE_REVISION, PATCH, RECEIPT, SOURCE_HASHES, prepare = (preparation[key] for key in
    ('CODE_REVISION', 'PATCH', 'RECEIPT', 'SOURCE_HASHES', 'prepare'))


def test_pinned_prepared_mesh_path_preserves_real_vertex_colors_topology_and_axes():
    torch = pytest.importorskip('torch')
    pytest.importorskip('trimesh')
    root = Path(os.environ.get('PANOPTES_SAM3D_TEST_SOURCE',
        '.platform/model-delivery-20260915/sam3d-mesh-source'))
    if not (root / RECEIPT).exists():
        pytest.skip('Run the documented pinned source preparation first')
    receipt = json.loads((root / RECEIPT).read_text())
    assert receipt['codeRevision'] == CODE_REVISION
    assert receipt['patchSha256'] == hashlib.sha256(PATCH.read_bytes()).hexdigest()
    for relative, original_hash in SOURCE_HASHES.items():
        assert receipt['files'][relative]['upstreamSha256'] == original_hash
        assert receipt['files'][relative]['patchedSha256'] == hashlib.sha256((root / relative).read_bytes()).hexdigest()
    tree = ast.parse((root / 'sam3d_objects/pipeline/inference_pipeline.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and any(a.name == 'postprocessing_utils' for a in n.names) for n in tree.body)
    pipeline = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'InferencePipeline')
    method = next(n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name == 'postprocess_slat_output')
    # Execute the exact prepared method with real CPU tensors and trimesh. The
    # remaining pipeline requires a separately audited Linux GPU image.
    scope = {'np': np, 'logger': SimpleNamespace(info=lambda message: None)}
    decoder_init = next(n for n in pipeline.body if isinstance(n, ast.FunctionDef) and n.name == 'init_slat_decoder_gs')
    exec(compile(ast.Module(body=[method, decoder_init], type_ignores=[]), str(root), 'exec'), scope)
    assert scope['init_slat_decoder_gs'](object(), None, None) is None
    vertices = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]])
    faces = torch.tensor([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    colors = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.2, .4, .6]])
    decoded = SimpleNamespace(vertices=vertices, faces=faces, vertex_attrs=colors)
    output = scope['postprocess_slat_output'](SimpleNamespace(rendering_engine='pytorch3d'),
        {'mesh': [decoded]}, False, False, True)
    mesh = output['glb']
    basis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    np.testing.assert_array_equal(mesh.vertices, vertices.numpy() @ basis)
    np.testing.assert_array_equal(mesh.vertices @ basis.T, vertices.numpy())
    np.testing.assert_array_equal(mesh.faces, faces.numpy())
    np.testing.assert_array_equal(mesh.visual.vertex_colors[:, :3], np.rint(colors.numpy() * 255).astype(np.uint8))
    assert 'gaussian' not in output and 'gs' not in output
    layout = ast.parse((root / 'sam3d_objects/pipeline/layout_post_optimization_utils.py').read_text())
    assert not any(isinstance(n, ast.ImportFrom) and 'gaussian_render' in (n.module or '') for n in layout.body)
    gs = next(n for n in layout.body if isinstance(n, ast.FunctionDef) and n.name == 'get_gs_mask_renderer')
    assert any(isinstance(n, ast.ImportFrom) and 'gaussian_render' in (n.module or '') for n in gs.body)


def test_source_preparation_rejects_different_code_before_patching(tmp_path):
    for relative in SOURCE_HASHES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('unreviewed source\n')
    with pytest.raises(ValueError, match='Source hash differs'):
        prepare(tmp_path)
    assert not (tmp_path / RECEIPT).exists()
    assert all((tmp_path / relative).read_text() == 'unreviewed source\n' for relative in SOURCE_HASHES)
