import hashlib
import json

import numpy as np
import pytest

from scripts.research.audit_model_correspondence import audit_catalog, audit_document
from ehs_spatial.platform.storage import LocalBlobStore


def test_corrupt_asset_cannot_count_as_loaded_model(tmp_path):
    store = LocalBlobStore(tmp_path)
    asset = {'id': 'mesh', **store.put(b'not a mesh')}
    doc = {'assets': [asset], 'observations': [], 'entities': [
        {'id': 'object', 'activeModelRepresentationId': 'representation', 'representations': [
            {'id': 'representation', 'kind': 'generated_mesh', 'assetId': 'mesh'}]},
        {'id': 'context', 'sourceContext': True}]}
    result = audit_document(doc, tmp_path)
    assert result['objectRecords'] == 1 and result['sourceContextIds'] == ['context']
    assert result['modelDeclaredCount'] == 1 and result['modelDecodedCount'] == 0
    assert result['allAssetsVerified']  # Correct bytes do not prove a valid model.
    (tmp_path / asset['storageKey']).write_bytes(b'changed')
    result = audit_document(doc, tmp_path)
    assert not result['allAssetsVerified']
    assert result['assetChecks'][0]['actualSha256'] == hashlib.sha256(b'changed').hexdigest()
    assert result['modelDecodedCount'] == 0


def test_observed_geometry_never_counts_as_generated_model(tmp_path):
    result = audit_document({'assets': [], 'observations': [], 'entities': [
        {'id': 'seen', 'representations': [{'id': 'r', 'kind': 'observed_surface'}]},
        {'id': 'primitive', 'activeModelRepresentationId': 'p', 'representations': [
            {'id': 'p', 'kind': 'primitive', 'primitive': {'type': 'box', 'dimensions': [1, 2, 3]}}]}]}, tmp_path)
    assert result['modelDeclaredCount'] == result['modelDecodedCount'] == 1
    assert result['objects'][1]['triangleCount'] == 12


def test_packed_model_uses_complete_scene_asset_metadata(tmp_path):
    from ehs_spatial.platform.spatial import primitive_mesh
    mesh = primitive_mesh({'type': 'box', 'dimensions': [1, 2, 3]})
    rows = np.column_stack((mesh.vertices, np.zeros_like(mesh.vertices), np.ones_like(mesh.vertices))).astype('<f4')
    faces = mesh.faces.astype('<u4')
    store = LocalBlobStore(tmp_path)
    asset = {'id': 'mesh', **store.put(rows.tobytes() + faces.tobytes()), 'format': 'panoptes-mesh-v1',
        'byteLayout': {'byteOffset': 0, 'vertexCount': len(rows), 'indexByteOffset': rows.nbytes, 'indexCount': faces.size}}
    document = {'assets': [asset], 'observations': [], 'entities': [
        {'id': 'object', 'activeModelRepresentationId': 'model', 'representations': [
            {'id': 'model', 'kind': 'generated_mesh', 'assetId': 'mesh'}]}]}
    result = audit_document(document, tmp_path)
    assert result['allAssetsVerified'] and result['modelDeclaredCount'] == result['modelDecodedCount'] == 1
    assert result['objects'][0]['triangleCount'] == len(faces)


@pytest.mark.parametrize('manifest_state', ['valid', 'omitted', 'mismatched'])
def test_catalog_checks_scene_references_against_asset_manifest(tmp_path, manifest_state):
    data = b'exact source asset bytes'
    sha = hashlib.sha256(data).hexdigest()
    publication_root = tmp_path / 'publication'
    (publication_root / 'blobs').mkdir(parents=True)
    (publication_root / 'blobs' / sha).write_bytes(data)
    asset = {'id': 'source', 'sha256': sha, 'sizeBytes': len(data)}
    manifest = [] if manifest_state == 'omitted' else [{'assetId': asset['id'], 'sha256': sha, 'sizeBytes': len(data)}]
    if manifest_state == 'mismatched':
        asset['sha256'] = 'a' * 64
    publication = {'id': 'publication', 'projectId': 'project', 'snapshot': {
        'revision': {'id': 'revision', 'document': {'assets': [asset], 'entities': []}}, 'assetManifest': manifest}}
    (publication_root / 'bundle.json').write_text(json.dumps({'publicationId': 'publication',
        'responses': {'/api/publications/publication': publication}}))
    result = audit_catalog(tmp_path)[0]
    assert result['failures'] == ({'valid': [],
        'omitted': [{'assetId': 'source', 'errorCode': 'scene_asset_missing_from_manifest'}],
        'mismatched': [{'assetId': 'source', 'errorCode': 'scene_asset_manifest_mismatch'}]}[manifest_state])
