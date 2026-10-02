"""Write explicit native-world bottom-fit previews; never replace source models."""
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import trimesh

from workcell_endpoint_estimate import _bottom_face
from workcell_photo_metrology import _fence_plane_index


def _world(scene, node):
    transform, name = scene.graph[node]
    return trimesh.transform_points(scene.geometry[name].vertices, transform)


def _bottom(scene, nodes, normal, offset, tolerance):
    points = np.concatenate([_bottom_face(scene, node, normal) for node in nodes])
    heights = points @ normal + offset
    if not np.isfinite(points).all() or np.ptp(heights) > tolerance:
        raise ValueError('Selected physical lower faces are not one horizontal plane')
    return points


def _check_footprint(points, corners, normal, tolerance):
    center = corners.mean(0)
    basis = np.linalg.svd(corners - center, full_matrices=False)[2][:2].T
    if np.max(abs(basis.T @ normal)) > 1e-5:
        raise ValueError('Bottom footprint is not parallel to the fitted ground')
    uv, expected = (points - center) @ basis, (corners - center) @ basis
    # A rectangular bounding footprint may span gaps between observed rails.
    # Its corners describe the model envelope, not additional RGB evidence.
    if not np.allclose(np.r_[uv.min(0), uv.max(0)], np.r_[expected.min(0), expected.max(0)], atol=tolerance, rtol=0):
        raise ValueError('Selected mesh lower-face envelope differs from the fitted input footprint')
    if not np.allclose(points @ normal, corners[0] @ normal, atol=tolerance, rtol=0):
        raise ValueError('Selected mesh lower-face plane differs from the fitted input footprint')


def _appearance_equal(a, b):
    if a.visual.kind != b.visual.kind:
        return False
    if a.visual.kind != 'texture':
        return np.array_equal(a.visual.vertex_colors, b.visual.vertex_colors)
    if not np.allclose(a.visual.uv, b.visual.uv, atol=1e-7, rtol=0):
        return False
    for name in ('color',):
        left, right = a.visual.vertex_attributes.get(name), b.visual.vertex_attributes.get(name)
        if (left is None) != (right is None) or (left is not None and not np.array_equal(left, right)):
            return False
    for name in ('name', 'baseColorFactor', 'metallicFactor', 'roughnessFactor', 'emissiveFactor',
                 'alphaMode', 'alphaCutoff', 'doubleSided', 'baseColorTexture', 'normalTexture',
                 'occlusionTexture', 'emissiveTexture', 'metallicRoughnessTexture'):
        left, right = getattr(a.visual.material, name, None), getattr(b.visual.material, name, None)
        if (left is None) != (right is None):
            return False
        if left is not None and not np.array_equal(np.asarray(left), np.asarray(right)):
            return False
    return True


def apply_bottom_models(root, fit, items, *, out):
    """Export fitted candidates into a new sibling directory and read them back.

    A single world-space affine acts on every node of each selected section:
    rigidly align its source lower plane to the shared ground, translate the
    footprint, preserve the initialized top height, and fit the bottom height.
    These are explicit display assumptions, not accepted physical geometry.
    """
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists() or out == root or root in out.parents or out in root.parents:
        raise ValueError('Preview output must be a new directory outside the source run')
    if fit.get('status') != 'conditional_fit' or fit.get('units') != 'native':
        raise ValueError('Only an explicit conditional native fit can be previewed')
    normal = np.asarray(fit['ground']['normal'], float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) <= 0 or not np.isfinite(fit['ground']['offset']):
        raise ValueError('Fit must provide a finite ground plane')
    norm = np.linalg.norm(normal)
    normal, offset = normal / norm, float(fit['ground']['offset']) / norm
    source_items = {row['id']: row for row in items}
    if len(source_items) != len(items) or not fit.get('items') or len({row['id'] for row in fit['items']}) != len(fit['items']):
        raise ValueError('Source and fitted items require unique identities')
    geometry_path = root / 'geometry.json'
    geometry = json.loads(geometry_path.read_text())
    hashes = {'geometry.json': hashlib.sha256(geometry_path.read_bytes()).hexdigest()}
    originals, scenes, records, changed = {}, {}, [], {}
    for fitted in fit['items']:
        ident = fitted['id']
        if ident not in source_items:
            raise ValueError(f'{ident}: fitted item lacks source model identity')
        item = source_items[ident]
        file = item['modelFile']
        expected_file = 'posts.glb' if ident.startswith('post-box-') else 'fence-fitted.glb' if ident.startswith('fence-') else None
        if file != expected_file:
            raise ValueError(f'{ident}: unexpected source model file')
        if file not in scenes:
            hashes[file] = hashlib.sha256((root / file).read_bytes()).hexdigest()
            originals[file] = trimesh.load(root / file, force='scene', process=False)
            scenes[file] = originals[file].copy()
            for name, mesh in originals[file].geometry.items():
                if mesh.visual.kind == 'texture' and 'color' in mesh.visual.vertex_attributes:
                    scenes[file].geometry[name].visual.vertex_attributes['color'] = mesh.visual.vertex_attributes['color'].copy()
            changed[file] = set()
        scene = scenes[file]
        old, new = np.asarray(item['bottomVerticesNative'], float), np.asarray(fitted['bottomVerticesNative'], float)
        if old.shape != (4, 3) or new.shape != (4, 3) or not np.isfinite([old, new]).all():
            raise ValueError(f'{ident}: bottom corners must be finite 4x3')
        tolerance = 2e-5 * max(1., float(np.linalg.norm(np.ptp(old, axis=0))))
        if not np.allclose(old, fitted['originalBottomVerticesNative'], atol=tolerance, rtol=0):
            raise ValueError(f'{ident}: fit belongs to another source footprint')
        delta = new - old
        if not np.allclose(delta, delta[0], atol=tolerance, rtol=0):
            raise ValueError(f'{ident}: one footprint translation cannot align all four fitted corners')
        old_height, new_height = float(np.mean(old @ normal + offset)), float(fitted['heightNative'])
        if (not np.isfinite(new_height) or new_height < 0 or
                not np.allclose(new @ normal + offset, new_height, atol=tolerance, rtol=0)):
            raise ValueError(f'{ident}: fitted bottom height and corners disagree')
        lower_nodes = list(item['modelNodes'])
        if not lower_nodes or len(set(lower_nodes)) != len(lower_nodes):
            raise ValueError(f'{ident}: distinct source lower-face nodes are required')
        nodes = lower_nodes
        if ident.startswith('fence-'):
            plane = _fence_plane_index(item)
            members = geometry['fence']['beams'] + geometry['fence']['continuations']
            nodes = list(dict.fromkeys(row['id'] for row in members
                                      if row['plane'] == plane and row['id'] in scene.graph.nodes_geometry))
            if not set(lower_nodes) <= set(nodes):
                raise ValueError(f'{ident}: source lower nodes do not belong to the selected fence section')
            original_rail = next((row.get('meshNode') for row in geometry.get('clearances', [])
                                 if row['id'] == f'fence-plane-{plane}-lower-rail'), None)
            if original_rail:
                if original_rail not in nodes:
                    raise ValueError(f'{ident}: the original observed lower rail is absent from this section')
                lower_nodes = list(dict.fromkeys([*lower_nodes, original_rail]))
        if not nodes or any(node not in scene.graph.nodes_geometry for node in nodes) or changed[file].intersection(nodes):
            raise ValueError(f'{ident}: missing or multiply assigned model nodes')
        initializer = np.asarray(item.get('initializerWorldNative'), float)
        if (initializer.shape != (4, 4) or not np.isfinite(initializer).all() or
                not np.allclose(initializer[3], [0., 0., 0., 1.], atol=1e-9) or
                not np.allclose(initializer[:3, :3].T @ initializer[:3, :3], np.eye(3), atol=1e-8) or
                not np.isclose(np.linalg.det(initializer[:3, :3]), 1., atol=1e-8)):
            raise ValueError(f'{ident}: a finite rigid shared-ground initializer is required')
        source_normal = initializer[:3, :3].T @ normal
        source_bottom = np.concatenate([_bottom_face(scene, node, source_normal) for node in lower_nodes])
        bottom = trimesh.transform_points(source_bottom, initializer)
        _check_footprint(bottom, old, normal, tolerance)
        source_points = np.concatenate([_world(scene, node) for node in nodes])
        points = trimesh.transform_points(source_points, initializer)
        original_top = float(np.max(source_points @ normal + offset))
        top_height = float(np.max(points @ normal + offset))
        if top_height <= max(old_height, new_height) + tolerance:
            raise ValueError(f'{ident}: fitted bottom must remain strictly below the preserved top')
        if ident.startswith('post-box-'):
            top_points = points[abs(points @ normal + offset - top_height) <= tolerance]
            _check_footprint(top_points, old + (top_height - old_height) * normal, normal, tolerance)
        stretch = (top_height - new_height) / (top_height - old_height)
        world = np.eye(4)
        world[:3, :3] += (stretch - 1) * np.outer(normal, normal)
        world[:3, 3] = delta[0] - (delta[0] @ normal) * normal + (stretch - 1) * (offset - top_height) * normal
        if not np.allclose(trimesh.transform_points(old, world), new, atol=tolerance, rtol=0):
            raise ValueError(f'{ident}: one upright affine cannot align the fitted bottom corners')
        bottom_affine = world.copy()
        world = bottom_affine @ initializer
        for node in nodes:
            transform, name = scene.graph[node]
            if abs(np.linalg.det(transform[:3, :3])) < 1e-12:
                raise ValueError(f'{ident}: singular source node transform')
            mesh = scene.geometry[name].copy()
            if scene.geometry[name].visual.kind == 'texture' and 'color' in scene.geometry[name].visual.vertex_attributes:
                mesh.visual.vertex_attributes['color'] = scene.geometry[name].visual.vertex_attributes['color'].copy()
            mesh.vertices = trimesh.transform_points(trimesh.transform_points(_world(scene, node), world), np.linalg.inv(transform))
            users = [other for other in scene.graph.nodes_geometry if scene.graph[other][1] == name]
            if len(users) > 1:
                replacement = f'{name}-bottom-{node}'
                if replacement in scene.geometry:
                    raise ValueError('Preview geometry name collides with an existing mesh')
                scene.geometry[replacement] = mesh
                parent = scene.graph.transforms.parents[node]
                edge = dict(scene.graph.transforms.edge_data[(parent, node)])
                edge['geometry'] = replacement
                scene.graph.update(frame_to=node, frame_from=parent, **edge)
            else:
                scene.geometry[name] = mesh
        changed[file].update(nodes)
        records.append({'id': ident, 'sourceModelFile': file, 'candidateModelFile': file.removesuffix('.glb') + '-candidate.glb',
                        'modelNodes': nodes, 'lowerFaceNodes': lower_nodes, 'worldAffineNative': world.tolist(),
                        'initializerWorldNative': initializer.tolist(), 'bottomAffineNative': bottom_affine.tolist(),
                        'initializer': item.get('initializer'),
                        'initializerMaxMovementNative': float(np.max(np.linalg.norm(points - source_points, axis=1))),
                        'originalBottomVerticesNative': trimesh.transform_points(old, np.linalg.inv(initializer)).tolist(),
                        'initializedBottomVerticesNative': old.tolist(), 'bottomVerticesNative': new.tolist(),
                        'originalBottomHeightRangeNative': [float(np.min(source_bottom @ normal + offset)), float(np.max(source_bottom @ normal + offset))],
                        'initializedHeightNative': old_height, 'heightNative': new_height,
                        'originalTopHeightNative': original_top, 'preservedTopHeightNative': top_height,
                        'initializerTopHeightChangeNative': top_height - original_top,
                        'verticalScale': stretch, 'verificationToleranceNative': tolerance})
    manifest = {'schemaVersion': 1, 'status': 'preview_candidate', 'inputFitStatus': fit['status'],
                'baselineModified': False, 'acceptedForPhysicalUse': False, 'physicalValidation': 'none',
                'units': 'native', 'mPerNative': None, 'ground': {'normal': normal.tolist(), 'offset': offset},
                'sharedHeightConstraint': bool(fit.get('sharedHeight')), 'sourceFiles': hashes, 'items': records,
                'assumptions': ['Existing coplanar lower faces are rigidly aligned to the shared inferred ground before bottom fitting; this changes section orientation and may change its top height.',
                    'Initialized footprint and horizontal thickness are retained; the whole section translates in the ground plane.',
                    'Initialized section top height is held fixed; every member is stretched vertically by the same positive affine, including the observed lower rail.',
                    'Top positions move with the footprint; intermediate heights and vertical member thicknesses change. These are display assumptions, not new physical measurements.',
                    'Equal height, when enabled, is an input prior and is not evidence of accuracy.']}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.bottom-preview-', dir=out.parent) as temporary:
        temporary = Path(temporary)
        for file, scene in scenes.items():
            candidate = file.removesuffix('.glb') + '-candidate.glb'
            scene.export(temporary / candidate)
            loaded = trimesh.load(temporary / candidate, force='scene', process=False)
            if set(loaded.graph.nodes_geometry) != set(scene.graph.nodes_geometry):
                raise ValueError('Candidate export changed model node identities')
            for node in scene.graph.nodes_geometry:
                expected = scene if node in changed[file] else originals[file]
                T, name = expected.graph[node]; saved_T, saved_name = loaded.graph[node]
                if not np.allclose(T, saved_T, atol=1e-9, rtol=0):
                    raise ValueError(f'{node}: node transform changed during export')
                tolerance = 2e-5 * max(1., float(np.linalg.norm(np.ptp(_world(expected, node), axis=0))))
                if not np.allclose(_world(loaded, node), _world(expected, node), atol=tolerance, rtol=0):
                    raise ValueError(f'{node}: exported vertices differ from the intended candidate')
                if not np.array_equal(loaded.geometry[saved_name].faces, expected.geometry[name].faces):
                    raise ValueError(f'{node}: exported topology differs from the intended candidate')
                if not _appearance_equal(expected.geometry[name], loaded.geometry[saved_name]):
                    raise ValueError(f'{node}: exported appearance differs from the intended candidate')
            for record in [row for row in records if row['sourceModelFile'] == file]:
                tolerance = record['verificationToleranceNative']
                bottom = _bottom(loaded, record['lowerFaceNodes'], normal, offset, tolerance)
                _check_footprint(bottom, np.asarray(record['bottomVerticesNative']), normal, tolerance)
                top = max(float(np.max(_world(loaded, node) @ normal + offset)) for node in record['modelNodes'])
                if abs(top - record['preservedTopHeightNative']) > tolerance:
                    raise ValueError('Candidate export moved the preserved section top height')
                record['exportedBottomHeightRangeNative'] = [float(np.min(bottom @ normal + offset)), float(np.max(bottom @ normal + offset))]
                record['exportVerified'] = True
            manifest.setdefault('candidateFiles', {})[candidate] = hashlib.sha256((temporary / candidate).read_bytes()).hexdigest()
        if any(hashlib.sha256((root / file).read_bytes()).hexdigest() != digest for file, digest in hashes.items()):
            raise ValueError('Source files changed during candidate export')
        (temporary / 'bottom-models.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
        temporary.rename(out)
    return manifest
