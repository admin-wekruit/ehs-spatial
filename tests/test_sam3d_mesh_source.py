"""CPU regression on exact prepared upstream code; no model inference or fake GS modules.

First run scripts/prepare_sam3d_mesh_source.py --fetch --source
data/model-sources/sam3d-mesh-source, or set PANOPTES_SAM3D_TEST_SOURCE.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

from argus.providers.prepare_sam3d_mesh_source import CODE_REVISION, PATCH, RECEIPT, SOURCE_HASHES, prepare


def test_pinned_prepared_mesh_path_preserves_real_vertex_colors_topology_and_axes():
    torch = pytest.importorskip('torch')
    pytest.importorskip('trimesh')
    root = Path(os.environ.get('PANOPTES_SAM3D_TEST_SOURCE',
        'data/model-sources/sam3d-mesh-source'))
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


def eager_imports(root, entries):
    """Conservative static Python imports, including package initializers; no model loading."""
    seen, external, pending = set(), set(), list(entries)
    def path(module):
        candidate = root / (module.replace('.', '/') + '.py')
        return candidate if candidate.is_file() else root / module.replace('.', '/') / '__init__.py'
    def nodes(tree):
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            yield node
            yield from nodes(node)
    while pending:
        module = pending.pop()
        if module in seen:
            continue
        source = path(module)
        if not source.is_file():
            external.add(module)
            continue
        seen.add(module)
        package = module if source.name == '__init__.py' else module.rpartition('.')[0]
        pending.extend('.'.join(module.split('.')[:i]) for i in range(1, len(module.split('.'))))
        for node in nodes(ast.parse(source.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                if imported:
                    pending.append(imported)
                    pending.extend(imported + '.' + alias.name for alias in node.names if path(imported + '.' + alias.name).is_file())
    return seen, external


def test_pinned_mesh_import_closure_excludes_optional_gaussian_code():
    root = Path(os.environ.get('PANOPTES_SAM3D_TEST_SOURCE', 'data/model-sources/sam3d-mesh-source'))
    upstream = Path(os.environ.get('PANOPTES_SAM3D_TEST_UPSTREAM', 'data/model-sources/sam3d-upstream'))
    entries = ['sam3d_objects.pipeline.inference_pipeline_pointmap',
        'sam3d_objects.model.backbone.tdfy_dit.models.structured_latent_vae.decoder_mesh']
    if not all((base / 'sam3d_objects/pipeline/inference_pipeline_pointmap.py').exists() for base in (root, upstream)):
        pytest.skip('Full pinned public source and its prepared copy are required for the import-closure check')
    receipt = json.loads((root / RECEIPT).read_text())
    assert receipt['patchSha256'] == hashlib.sha256(PATCH.read_bytes()).hexdigest()
    for relative, expected in SOURCE_HASHES.items():
        assert hashlib.sha256((upstream / relative).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == receipt['files'][relative]['patchedSha256']
    def forbidden(modules):
        return {m for m in modules if any(part in m for part in (
            'representations.gaussian', 'renderers.gaussian_render', 'decoder_gs', 'postprocessing_utils',
            'gsplat', 'diff_gaussian_rasterization'))}
    original, original_external = eager_imports(upstream, entries)
    prepared, prepared_external = eager_imports(root, entries)
    assert 'sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.general_utils' in forbidden(original)
    assert 'sam3d_objects.model.backbone.tdfy_dit.renderers.gaussian_render' in forbidden(original)
    assert not forbidden(prepared | prepared_external)
    assert set(entries) <= prepared
    assert 'sam3d_objects.model.backbone.tdfy_dit.representations.mesh.cube2mesh' in prepared


def test_gaussian_exports_are_deferred_until_explicitly_requested():
    root = Path(os.environ.get('PANOPTES_SAM3D_TEST_SOURCE', 'data/model-sources/sam3d-mesh-source'))
    if not (root / RECEIPT).exists():
        pytest.skip('Prepare the pinned source first')
    for relative, name in (
        ('sam3d_objects/model/backbone/tdfy_dit/models/__init__.py', 'SLatGaussianDecoder'),
        ('sam3d_objects/model/backbone/tdfy_dit/models/structured_latent_vae/__init__.py', 'SLatGaussianDecoder'),
        ('sam3d_objects/model/backbone/tdfy_dit/representations/__init__.py', 'Gaussian'),
    ):
        tree = ast.parse((root / relative).read_text())
        assert not any(isinstance(node, ast.ImportFrom) and name in {a.name for a in node.names} for node in tree.body)
        method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '__getattr__')
        scope, requests = {}, []
        def rejected_import(module, globals=None, locals=None, fromlist=(), level=0):
            requests.append((module, fromlist, level))
            raise ImportError('Optional Gaussian implementation was explicitly requested')
        scope['__builtins__'] = {**vars(__import__('builtins')), '__import__': rejected_import}
        exec(compile(ast.Module(body=[method], type_ignores=[]), relative, 'exec'), scope)
        assert requests == []
        with pytest.raises(AttributeError): scope['__getattr__']('unknown_export')
        assert requests == []
        with pytest.raises(ImportError, match='explicitly requested'): scope['__getattr__'](name)
        assert len(requests) == 1 and name in requests[0][1]
