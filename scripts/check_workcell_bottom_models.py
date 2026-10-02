"""Run: PYTHONPATH=.:scripts python scripts/check_workcell_bottom_models.py."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
from PIL import Image
import trimesh

from workcell_bottom_models import apply_bottom_models, _world
import workcell_physical_bottoms as physical_bottoms
from workcell_physical_bottoms import _items, _ordered_bottom


def check():
    strong = {'photo': 1, 'identityEvidence': {'independentlySupported': True}}
    weak = {'photo': 2, 'identityEvidence': {'independentlySupported': False}}
    unrelated = {'photo': 3}
    matched = physical_bottoms._matched_fence_items(
        [{'id': 'fence-0', 'observations': [strong, weak, unrelated]}],
        [{'id': 'fence-0', 'bottomEdge': {'observations': [strong, weak]}}])
    assert matched[0]['observations'] == [strong, weak]
    assert matched[0]['requireIdentityAnchor'] is True
    try:
        physical_bottoms._matched_fence_items(matched, [{'id': 'fence-0', 'bottomEdge': None, 'reason': 'disconnected'}])
    except ValueError as error:
        assert 'no common finite source edge' in str(error)
    else:
        raise AssertionError('Unmatched source fragments entered model fitting')
    normal = np.array([.2, -.3, 1.]); normal /= np.linalg.norm(normal)
    floor = trimesh.geometry.align_vectors([0., 0., 1.], normal)
    floor[:3, 3] = .4 * normal
    ground = {'normal': normal.tolist(), 'offset': -.4}
    parent = trimesh.transformations.rotation_matrix(.3, [1., 2., 3.])
    parent[:3, 3] = [2., -.7, .3]
    node_transform = trimesh.transformations.rotation_matrix(-.4, [2., 1., 3.])
    node_transform[:3, 3] = [-.3, .8, 1.2]

    def add(scene, node, size, position, *, textured=False):
        mesh = trimesh.creation.box(size)
        mesh.apply_translation(position)
        mesh.apply_transform(np.linalg.inv(node_transform) @ floor)
        colors = np.array([[60+10*i, 150, 210, 255] for i in range(len(mesh.vertices))], np.uint8)
        if textured:
            mesh.visual = trimesh.visual.texture.TextureVisuals(uv=np.zeros((len(mesh.vertices), 2)),
                material=trimesh.visual.material.PBRMaterial(name='fixture-material',
                    baseColorTexture=Image.new('RGB', (2, 2), (30, 70, 110)), metallicFactor=.2, roughnessFactor=.7))
            mesh.visual.vertex_attributes['color'] = colors
        else:
            mesh.visual.vertex_colors = colors
        scene.add_geometry(mesh, node_name=node, geom_name=node+'-mesh', parent_node_name='group',
                           transform=np.linalg.inv(parent) @ node_transform)

    with tempfile.TemporaryDirectory(prefix='workcell-bottom-model-check-') as directory:
        root, output = Path(directory)/'source', Path(directory)/'preview'
        root.mkdir()
        posts, fences = trimesh.Scene(), trimesh.Scene()
        for scene in (posts, fences):
            scene.graph.update(frame_to='group', matrix=parent)
        add(posts, 'box-1', [.2, .15, 2.], [0., 0., 1.5], textured=True)
        # A non-target node shares the same geometry. Editing the target must
        # split its geometry use while preserving hierarchy and node matrices.
        posts.graph.update(frame_to='untouched-shared', frame_from='group', geometry='box-1-mesh',
                           matrix=np.linalg.inv(parent) @ node_transform @ trimesh.transformations.translation_matrix([3., 0., 0.]))
        add(fences, 'original-lower', [.8, .12, .1], [0., 0., .45])
        add(fences, 'lower-left', [1.1, .12, .1], [-.95, 0., .45])
        add(fences, 'lower-right', [1.1, .12, .1], [.95, 0., .45])
        add(fences, 'top', [3., .12, .1], [0., 0., 2.05])
        add(fences, 'upright', [.1, .12, 1.7], [-1.45, 0., 1.25])
        add(fences, 'other-section', [.1, .12, 1.7], [4., 0., 1.25])
        # Reproduce a fence built under an earlier floor normal. Every rail
        # remains coplanar; fixing this must be a rigid section initialization,
        # not independently projecting lower vertices into a distorted box.
        stale_floor = trimesh.transformations.rotation_matrix(.08, floor[:3, 1], point=floor[:3, 3])
        for mesh in fences.geometry.values():
            mesh.apply_transform(np.linalg.inv(node_transform) @ stale_floor @ node_transform)
        geometry = {'floor': ground, 'fence': {'planes': [{'normal': [0., 1., 0.], 'offset': 0.}],
            'beams': [{'id': 'original-lower', 'plane': 0, 'horizontal': True}],
            'continuations': [{'id': node, 'plane': plane, 'role': role} for node, plane, role in (
                ('lower-left', 0, 'lower-rail continuation'), ('lower-right', 0, 'lower-rail continuation'),
                ('top', 0, 'top-frame'), ('upright', 0, 'end-frame'), ('other-section', 1, 'end-frame'))]},
            'clearances': [{'id': 'fence-plane-0-lower-rail', 'meshNode': 'original-lower'}]}
        posts.export(root/'posts.glb'); fences.export(root/'fence-fitted.glb')
        (root/'geometry.json').write_text(json.dumps(geometry))
        (root/'sam3.json').write_text('{}')
        items = _items(root, geometry, {'fence-0': {'id': 'fence-0', 'geometryPlaneIndex': 0}},
                       [{'id': ident, 'bottomCandidates': []} for ident in ('post-box-1', 'fence-0')])
        initialized = items[1]['initializer']
        assert np.isclose(initialized['angleDegrees'], np.degrees(.08), atol=1e-5)
        source_lower = np.asarray(initialized['sourceBottomVerticesNative'])
        aligned_lower = np.asarray(initialized['initializedBottomVerticesNative'])
        assert initialized['heightSpreadBeforeNative'] > .1 and initialized['heightSpreadAfterNative'] < 1e-6
        assert np.allclose(np.linalg.norm(source_lower[:, None]-source_lower, axis=2),
                           np.linalg.norm(aligned_lower[:, None]-aligned_lower, axis=2), atol=1e-8)
        warped = source_lower.copy(); warped[0] += .01*normal
        try:
            _ordered_bottom(warped, normal, ground['offset'])
        except ValueError as error:
            assert 'not coplanar' in str(error)
        else:
            raise AssertionError('Noncoplanar lower vertices were silently flattened')

        # A fresh depth reconstruction can reorder the two planes without
        # changing physical identities. The explicit mapping must choose the
        # same GLB footprint; a missing opposite footprint remains in results.
        reordered = deepcopy(geometry)
        reordered['fence']['planes'] = [deepcopy(geometry['fence']['planes'][0]) for _ in range(2)]
        for member in reordered['fence']['beams'] + reordered['fence']['continuations']:
            member['plane'] = 1-member['plane']
        reordered['clearances'][0]['id'] = 'fence-plane-1-lower-rail'
        catalog = {ident: {'id': ident} for ident in physical_bottoms.TARGETS}
        catalog['fence-0']['geometryPlaneIndex'] = 1
        catalog['fence-1']['geometryPlaneIndex'] = 0
        diagnostics = [{'id': ident, 'bottomCandidates': []} for ident in ('fence-0', 'fence-1')]
        refreshed = _items(root, reordered, catalog, diagnostics)
        assert [row['id'] for row in refreshed] == ['fence-0', 'fence-1']
        assert refreshed[0]['geometryPlaneIndex'] == 1
        assert refreshed[0]['modelNodes'] == items[1]['modelNodes']
        assert np.allclose(refreshed[0]['bottomVerticesNative'], items[1]['bottomVerticesNative'])
        assert refreshed[1]['status'] == 'unsupported' and refreshed[1]['geometryPlaneIndex'] == 0
        assert 'No existing lower-rail footprint' in refreshed[1]['reason']
        assert 'bottomVerticesNative' not in refreshed[1]
        # The real experiment caller must reject the incomplete pair before
        # optimization, rather than announcing a one-object shared-height fit.
        with patch.object(physical_bottoms, '_load', return_value=(reordered, catalog, {}, {}, [], [])), \
             patch.object(physical_bottoms, '_legacy', return_value={'pointNative': [0., 0., 0.]}), \
             patch.object(physical_bottoms, '_object_edges', return_value=([], diagnostics)), \
             patch('workcell_bottom_fit.fit_bottoms', side_effect=AssertionError('Incomplete pair reached optimizer')):
            rejected = physical_bottoms.build(root, Path(directory)/'missing-footprint', [])
        for name in ('fences-independent', 'fences-shared'):
            assert rejected['fits'][name]['status'] == 'unsupported'
            assert 'fence-1: No existing lower-rail footprint' in rejected['fits'][name]['reason']
        fit = {'status': 'conditional_fit', 'units': 'native', 'ground': ground, 'sharedHeight': False, 'items': []}
        for item, height in zip(items, (.65, .72)):
            old = np.array(item['bottomVerticesNative'])
            movement = floor[:3, :3] @ [.15, -.08, 0.] + (height - item['baselineHeightNative']) * normal
            fit['items'].append({'id': item['id'], 'originalBottomVerticesNative': old.tolist(),
                                  'bottomVerticesNative': (old + movement).tolist(), 'heightNative': height})
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}
        for path in root.iterdir():
            path.chmod(0o444)
        manifest = apply_bottom_models(root, fit, items, out=output)
        assert manifest['status'] == 'preview_candidate' and not manifest['acceptedForPhysicalUse']
        assert manifest['physicalValidation'] == 'none' and manifest['mPerNative'] is None
        assert all(row['exportVerified'] for row in manifest['items'])
        fence = next(row for row in manifest['items'] if row['id'] == 'fence-0')
        assert set(fence['modelNodes']) == {'original-lower', 'lower-left', 'lower-right', 'top', 'upright'}
        assert 'original-lower' in fence['lowerFaceNodes']
        assert fence['initializerMaxMovementNative'] > .1
        assert abs(fence['initializerTopHeightChangeNative']) > .05
        assert np.allclose(np.asarray(fence['bottomAffineNative']) @ np.asarray(fence['initializerWorldNative']),
                           fence['worldAffineNative'])
        for record in manifest['items']:
            scene = trimesh.load(output/record['candidateModelFile'], force='scene', process=False)
            source = trimesh.load(root/record['sourceModelFile'], force='scene', process=False)
            affine = np.asarray(record['worldAffineNative'])
            for node in record['modelNodes']:
                assert np.allclose(_world(scene, node), trimesh.transform_points(_world(source, node), affine), atol=2e-6)
                assert np.allclose(scene.graph.transforms.edge_data[('group', node)]['matrix'],
                                   source.graph.transforms.edge_data[('group', node)]['matrix'], atol=1e-9)
            untouched = 'untouched-shared' if record['id'].startswith('post-') else 'other-section'
            assert np.allclose(_world(scene, untouched), _world(source, untouched), atol=1e-8)
            assert np.allclose(record['exportedBottomHeightRangeNative'], record['heightNative'], atol=2e-6)
        assert hashes == {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in root.iterdir()}

        for invalid in ('collapsed', 'inconsistent-corners', 'tilted-input', 'source-output', 'unsupported'):
            broken, source_items = deepcopy(fit), deepcopy(items)
            target = Path(directory)/invalid
            if invalid == 'collapsed':
                row = broken['items'][0]
                row['bottomVerticesNative'] = (np.asarray(row['bottomVerticesNative']) + (3.-row['heightNative']) * normal).tolist()
                row['heightNative'] = 3.
            elif invalid == 'inconsistent-corners':
                broken['items'][0]['bottomVerticesNative'][0][0] += .1
            elif invalid == 'tilted-input':
                source_items[0]['bottomVerticesNative'][0][2] += .1
                broken['items'][0]['originalBottomVerticesNative'] = source_items[0]['bottomVerticesNative']
                broken['items'][0]['bottomVerticesNative'][0][2] += .1
            elif invalid == 'source-output':
                target = root/'new-candidate'
            else:
                broken['status'] = 'unsupported'
            try:
                apply_bottom_models(root, broken, source_items, out=target)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid candidate was exported: '+invalid)
            assert not target.exists(), 'Rejected candidate left a published output directory'
        print('PASS: stale-floor rigid initialization preserves distances and rejects noncoplanar faces; reordered fence identity; missing section rejected before fitting; tilted shared ground; nested transforms; complete section and original rail; initialized top preserved; texture/colors and untouched nodes; readonly source preserved')


def cached_smoke(root):
    """Optional readonly real-GLB round trip; zero fit translation is not accuracy."""
    root = Path(root)
    geometry = json.loads((root/'geometry.json').read_text())
    catalog = {row['id']: row for row in json.loads((root/'objects.json').read_text())['objects']}
    items = _items(root, geometry, catalog, [{'id': ident, 'bottomCandidates': []}
                                            for ident in ('fence-0', 'fence-1')])
    assert len(items) == 2 and all(row.get('status') != 'unsupported' for row in items), items
    files = ('posts.glb', 'fence-fitted.glb', 'geometry.json', 'objects.json')
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
    fit = {'status': 'conditional_fit', 'units': 'native', 'ground': geometry['floor'], 'sharedHeight': False,
           'items': [{'id': row['id'], 'originalBottomVerticesNative': row['bottomVerticesNative'],
                      'bottomVerticesNative': row['bottomVerticesNative'], 'heightNative': row['baselineHeightNative']}
                     for row in items]}
    with tempfile.TemporaryDirectory(prefix='workcell-cached-model-smoke-') as directory:
        result = apply_bottom_models(root, fit, items, out=Path(directory)/'preview')
        assert all(row['exportVerified'] and abs(row['verticalScale']-1.) < 1e-12 for row in result['items'])
        assert not result['acceptedForPhysicalUse'] and result['physicalValidation'] == 'none'
        print(json.dumps({'check': 'readonly actual GLB zero-translation initializer/export smoke; not an RGB fit or accuracy result',
            'items': [{'id': row['id'], 'angleDegrees': row['initializer']['angleDegrees'],
                       'coplanarityResidualNative': row['initializer']['maxCoplanarResidualNative'],
                       'initializerMaxMovementNative': row['initializerMaxMovementNative'],
                       'topChangeNative': row['initializerTopHeightChangeNative'],
                       'exportedBottomHeightRangeNative': row['exportedBottomHeightRangeNative']}
                      for row in result['items']]}, indent=2))
    assert hashes == {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, help='Optional frozen real run for readonly small-GLB smoke')
    args = parser.parse_args()
    check()
    if args.baseline:
        cached_smoke(args.baseline)
