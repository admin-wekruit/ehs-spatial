"""Export standalone source-fit button candidates, never fuse different camera worlds."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


DIMENSIONS = ('wholeComponentHeightM', 'mainBodyDiameterM', 'redActuatorDiameterM')


def export_reference_candidate(result, out):
    """Write a native-world GLB plus explicit conditional/unsupported metadata.

    Uses only the saved joint-fit candidate. No images, depth, ground truth,
    existing scene geometry or measurement targets are read or inferred here.
    """
    status = result.get('status')
    if status not in ('available', 'unsupported'):
        raise ValueError('Joint-fit status must be available or unsupported')
    out = Path(out)
    shape = result.get('fittedNuisanceParameters')
    manifest = {
        'schemaVersion': 1, 'status': status, 'units': 'native',
        'worldFrame': 'saved joint-fit MapAnything native; use its fitted cameras only',
        'modelFile': None, 'metricScaleMPerNative': None,
        'candidateScaleForPreviewOnly': None, 'previewOnly': True,
        'measurementStatus': 'unsupported', 'reason': result.get('reason'),
        'knownDimensions': result.get('knownDimensions', {}),
        'jointCameraSha256': result.get('cameraSha256'), 'sourceHash': result.get('sourceHash'),
        'shapeAssumptions': [
            'Red actuator is modeled as a finite cylinder; the observed rounded cap is not an exact CAD shape.',
            'Yellow body is a fitted frustum with the supplied maximum diameter.',
            'Gray housing is an elliptic cylinder inferred from its hypothesized bottom rim and fitted height; unseen side shape is not measured.',
            'The gray bottom may include a mounting lip. Endpoint identity remains unverified even when projection checks pass.',
        ],
        'displayScope': 'Standalone candidate only; do not overlay geometry reconstructed with different cameras.',
        'parts': [],
    }
    model = None
    if shape is None:
        if status == 'available':
            raise ValueError('An available fit must contain fitted reference geometry')
    else:
        vectors = {key: np.asarray(shape[key], float) for key in ('base', 'u', 'v', 'axis')}
        if any(value.shape != (3,) or not np.isfinite(value).all() for value in vectors.values()):
            raise ValueError('Reference axes and base must be finite 3-vectors')
        basis = np.c_[vectors['u'], vectors['v'], vectors['axis']]
        if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(basis), 1., atol=1e-6):
            raise ValueError('Reference basis must be orthonormal and right handed')
        names = ('height', 'grayHeight', 'yellowHeight', 'redHeight', 'yellowRadius', 'redRadius',
                 'grayWidth', 'grayDepth', 'yellowTopRadiusFraction', 'mPerNative')
        values = [shape[key] for key in names]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Reference dimensions must be finite positive numbers')
        if shape['yellowTopRadiusFraction'] > 1.00001:
            raise ValueError('Yellow maximum diameter must remain the lower-rim diameter')
        scale = shape['mPerNative']
        if not np.isclose(result.get('candidateMPerNative', np.nan), scale, rtol=1e-8, atol=0):
            raise ValueError('Candidate and geometry scales disagree')
        if not np.isclose(sum(shape[key] for key in ('grayHeight', 'yellowHeight', 'redHeight')), shape['height'], rtol=1e-8, atol=0):
            raise ValueError('Section heights do not sum to total axial height')
        supplied = manifest['knownDimensions']
        dimensions = np.array([supplied.get(key, np.nan) for key in DIMENSIONS], float)
        actual = scale * np.array([shape['height'], 2 * shape['yellowRadius'], 2 * shape['redRadius']])
        if not np.isfinite(dimensions).all() or np.any(dimensions <= 0) or not np.allclose(actual, dimensions, rtol=1e-7, atol=1e-10):
            raise ValueError('Candidate violates one of the three supplied dimensions')
        if status == 'available':
            if not np.isclose(result.get('mPerNative', np.nan), scale, rtol=1e-8, atol=0):
                raise ValueError('Accepted metric scale disagrees with reference geometry')
            manifest.update(metricScaleMPerNative=scale, previewOnly=False, measurementStatus='conditional_fit')
        elif result.get('mPerNative') is not None:
            raise ValueError('Unsupported fit cannot carry a measurement scale')
        transform = np.eye(4); transform[:3, :3] = basis; transform[:3, 3] = vectors['base']
        scene = trimesh.Scene()
        gray_top = shape['grayHeight']; yellow_top = gray_top + shape['yellowHeight']
        profiles = {
            'gray-housing': [[0., 0.], [1., 0.], [1., gray_top], [0., gray_top]],
            'yellow-body': [[0., gray_top], [shape['yellowRadius'], gray_top],
                            [shape['yellowRadius'] * shape['yellowTopRadiusFraction'], yellow_top], [0., yellow_top]],
            'red-actuator': [[0., yellow_top], [shape['redRadius'], yellow_top],
                             [shape['redRadius'], shape['height']], [0., shape['height']]],
        }
        colors = {'gray-housing': [110, 117, 123, 255], 'yellow-body': [238, 194, 43, 255],
                  'red-actuator': [168, 43, 42, 255]}
        for name, profile in profiles.items():
            # ponytail: 64 radial segments preserve exact cardinal diameters;
            # this is a diagnostic surface, not manufactured CAD tessellation.
            mesh = trimesh.creation.revolve(profile, sections=64)
            if name == 'gray-housing':
                mesh.apply_scale([shape['grayWidth'] / 2, shape['grayDepth'] / 2, 1.])
            mesh.apply_transform(transform)
            mesh.visual.face_colors = colors[name]
            scene.add_geometry(mesh, node_name=name, geom_name=name)
            manifest['parts'].append({'id': name, 'geometryStatus': 'shape_hypothesis'})
        model = scene.export(file_type='glb')
        manifest.update(modelFile='reference-candidate.glb', candidateScaleForPreviewOnly=scale,
                        baseNative=vectors['base'].tolist(), axisNative=vectors['axis'].tolist(),
                        boundsNative=scene.bounds.tolist(), modelSha256=hashlib.sha256(model).hexdigest(),
                        modelSizeBytes=len(model), knownDimensionScope='Supplied constraints enforced in candidate, not new measurements.',
                        referenceGeometryIdentifiable=result.get('referenceGeometryIdentifiable'))
    # Validate before creating output; preserve previous runs rather than overwrite.
    manifest_text = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    out.mkdir(parents=True, exist_ok=False)
    if model is not None:
        (out / 'reference-candidate.glb').write_bytes(model)
    (out / 'reference-candidate.json').write_text(manifest_text)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('references', nargs='+', type=Path, help='Saved joint-reference.json files')
    parser.add_argument('--subdir', default='reference-model', help='New child directory beside each reference file')
    args = parser.parse_args()
    if Path(args.subdir).name != args.subdir or args.subdir in ('.', '..'):
        parser.error('--subdir must be a single new child directory name')
    for source in args.references:
        target = source.parent / args.subdir
        manifest = export_reference_candidate(json.loads(source.read_text()), target)
        print(json.dumps({'source': str(source), 'out': str(target), 'status': manifest['status'], 'modelFile': manifest['modelFile']}))


if __name__ == '__main__':
    main()
