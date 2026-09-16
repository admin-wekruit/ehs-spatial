import io

import numpy as np
import pytest
from PIL import Image

from ehs_spatial.platform.blender_export import mesh_from_asset
from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.reconstruction import _mesh
from ehs_spatial.platform.spatial import FrameGeometry, MeshData, project_native, unproject_pixels
from scripts.research.build_observed_surface_models import PLANAR_METHOD, planar_surface, surface_glb, textured_surface


def test_source_surface_uv_texture_roundtrip_keeps_holes_and_exact_topology(tmp_path):
    y, x = np.mgrid[:5, :6]
    points = np.stack((x * .1, y * .1, 2 + x * .01), axis=-1)
    mask = np.ones((5, 6), bool)
    mask[2, 2] = False
    frame = FrameGeometry('photo', 'native', '0' * 64, points, np.ones_like(mask), np.eye(3), np.eye(4))
    record = {'rgb': np.zeros((5, 6, 3), np.uint8), 'inputToCanonical': np.diag([.5, .5, 1.]), 'originalShape': (10, 12)}
    source = _mesh(points, frame.support(), record['rgb'], mask)
    stream = io.BytesIO()
    image = Image.new('RGB', (12, 10), '#ff0000')
    image.putpixel((0, 0), (0, 0, 255))
    image.save(stream, format='PNG')
    texture = stream.getvalue()
    mesh, metrics = textured_surface(frame, record, mask, source, texture)
    assert np.array_equal(mesh.vertices, source.vertices) and np.array_equal(mesh.faces, source.faces)
    np.testing.assert_allclose(mesh.uv, (np.column_stack((x[mask], y[mask])) * 2 + .5) / [12, 10])
    assert metrics['newVertices'] == metrics['newTriangles'] == 0
    assert metrics['supportedPixels'] == int(mask.sum()) == len(mesh.vertices)
    assert metrics['boundaryEdgeCount'] > 2 * (5 + 6 - 2)
    assert mesh.colors is None and mesh.texture_bytes == texture
    assert not np.shares_memory(mesh.vertices, source.vertices)
    pose = {'coordinateFrameId': 'native', 'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'scale': [1, 1, 1]}
    document = {'coordinateFrames': [{'id': 'native', 'scale': {'status': 'uncalibrated'}}], 'cameras': []}
    rep = {'id': 'model', 'kind': 'generated_mesh', 'coordinateFrameId': 'native', 'transform': pose,
        'placementState': 'unconfirmed', 'placementReason': 'imported_proposal', 'coverage': 'observed_visible_surface_only',
        'shapeStatus': 'source_derived_surface', 'sourceRefs': [{'observationId': 'observation', 'revision': 1}]}
    path = tmp_path / 'surface.glb'
    result = surface_glb(document, 'revision', {'id': 'entity', 'label': 'Surface', 'observationRefs': ['observation']}, rep, mesh, path)
    restored = mesh_from_asset(path.read_bytes(), {'sha256': result['sha256']})
    assert np.array_equal(restored.vertices, source.vertices) and np.array_equal(restored.faces, source.faces)
    assert np.array_equal(restored.uv, mesh.uv) and restored.texture_bytes == texture
    perturbed = MeshData(source.vertices + .01, source.faces)
    with pytest.raises(PlatformError, match='surface_source_topology_mismatch'):
        textured_surface(frame, record, mask, perturbed, texture)
    with pytest.raises(PlatformError, match='surface_texture_image_mismatch'):
        textured_surface(frame, {**record, 'originalShape': (5, 6)}, mask, source, texture)


def test_explicit_plane_model_preserves_mask_holes_and_rejects_inconsistent_depth(tmp_path):
    y, x = np.mgrid[:30, :30]
    camera = {'K': np.array([[25., 0., 15.], [0., 25., 15.], [0., 0., 1.]]), 'cameraToWorld': np.eye(4)}
    pixels = np.column_stack((x.ravel(), y.ravel()))
    points = unproject_pixels(pixels, 2 + np.sin(x.ravel()) * .001, camera).reshape(30, 30, 3)
    mask = (x > 2) & (x < 27) & (y > 2) & (y < 27) & ~((x > 10) & (x < 16) & (y > 10) & (y < 16))
    frame = FrameGeometry('photo', 'native', '0' * 64, points, np.ones_like(mask), camera['K'], camera['cameraToWorld'])
    record = {'inputToCanonical': np.eye(3), 'originalShape': (30, 30)}
    constraint = {'constraint': 'Reviewed visible transparent sheet is planar', 'appearance': 'neutral_translucent'}
    before = points.copy()
    mesh, report = planar_surface(frame, record, mask, b'', constraint)
    assert report['modelingCheckStatus'] == 'supported_approximation' and report['failedChecks'] == []
    assert report['sourcePositionsIdentical'] is False and report['manualConstraint'] == constraint
    assert len(mesh.faces) == 2 * mask.sum()
    assert mesh.texture_bytes is None and mesh.uv is None and mesh.colors is None
    assert mesh.material['alphaMode'] == 'BLEND' and mesh.material['baseColorFactor'][3] == .22
    centers, _ = project_native(mesh.vertices[mesh.faces].mean(axis=1), camera)
    indices = np.floor(centers + .5).astype(int)
    assert mask[indices[:, 1], indices[:, 0]].all(), 'No triangles span the source mask hole'
    np.testing.assert_allclose(mesh.vertices @ report['plane'][:3] + report['plane'][3], 0, atol=1e-6)
    assert np.array_equal(points, before)
    pose = {'coordinateFrameId': 'native', 'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1], 'scale': [1, 1, 1]}
    source = {'coordinateFrames': [{'id': 'native', 'scale': {'status': 'uncalibrated'}}], 'cameras': []}
    rep = {'id': 'model', 'kind': 'generated_mesh', 'coordinateFrameId': 'native', 'transform': pose,
        'sourceDerivation': {'methodVersion': PLANAR_METHOD}}
    path = tmp_path / 'transparent.glb'
    surface_glb(source, 'revision', {'id': 'entity', 'label': 'Sheet', 'observationRefs': []}, rep, mesh, path)
    from ehs_spatial.platform.blender_export import _read_glb
    assert _read_glb(path.read_bytes())[0]['extras']['methodVersion'] == PLANAR_METHOD
    restored = mesh_from_asset(path.read_bytes(), {})
    assert restored.material['alphaMode'] == 'BLEND' and restored.material['baseColorFactor'][3] == .22
    noisy = unproject_pixels(pixels, 2 + .4 * np.sin(x.ravel()), camera).reshape(30, 30, 3)
    bad_frame = FrameGeometry('photo', 'native', '0' * 64, noisy, np.ones_like(mask), camera['K'], camera['cameraToWorld'])
    _, failed = planar_surface(bad_frame, record, mask, b'', constraint)
    assert failed['modelingCheckStatus'] == 'rejected' and failed['failedChecks']
    with pytest.raises(PlatformError, match='surface_manual_planar_constraint_required'):
        planar_surface(frame, record, mask, b'', {})
