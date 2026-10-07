"""Per-object model-to-photo correspondence audit of one report revision.

For every entity with an active model and a source observation, the model's
triangles are rasterized through that photo's saved camera onto the canonical
image grid and compared with the observation's saved segmentation polygons.
Intersection over union measures model/photo agreement in that view only:
occlusion, mask errors and unseen geometry all lower it, so it is a triage
signal, not a physical accuracy score. Per-photo robot pose variants are
audited against their own photo.

python scripts/workcell_object_audit.py --root RUN [--out audit.json]
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh


def _mask(polygons, shape):
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.rint(np.asarray(polygon, float)).astype(np.int32)], 1)
    return mask.astype(bool)


def _silhouette(vertices, faces, camera, shape):
    pose, K = np.asarray(camera['cameraToWorld'], float), np.asarray(camera['K'], float)
    local = (vertices - pose[:3, 3]) @ pose[:3, :3]
    mask = np.zeros(shape, np.uint8)
    front = local[:, 2] > 1e-6
    keep = front[faces].all(1)
    if not keep.any():
        return mask.astype(bool), int((~front).any())
    uv = local[:, :2] / np.maximum(local[:, 2:3], 1e-9) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    cv2.fillPoly(mask, list(np.rint(uv[faces[keep]]).astype(np.int32)), 1)
    return mask.astype(bool), int((~keep).sum())


def audit(root, report=None):
    root = Path(root)
    report = report or json.loads((root / 'scene-report.json').read_text())
    document = report['revision']['document']
    cameras = {camera['imageId']: camera for camera in document['cameras']}
    observations = {row['id']: row for row in document['observations']}
    rows = []
    for entity in document['entities']:
        if entity.get('sourceContext'):
            continue
        variants = entity.get('modelVariants') or {}
        for oid in entity.get('observationRefs', []):
            observation = observations[oid]
            photo = observation['imageId'].removeprefix('photo-')
            rep = variants.get(photo) or next((r for r in entity['representations'] if r['id'] == entity.get('activeModelRepresentationId')), None)
            polygons = observation.get('originalPixelPolygons') or []
            if rep is None or not polygons:
                rows.append({'entityId': entity['id'], 'photo': int(photo), 'observationId': oid, 'status': 'no_model' if rep is None else 'no_polygon'})
                continue
            camera = cameras[observation['imageId']]
            shape = (camera['height'], camera['width'])
            mesh = trimesh.load(root / report['assetURLs'][rep['assetId']], force='mesh', process=False)
            vertices = np.asarray(mesh.vertices) + rep['transform']['position']
            model, clipped = _silhouette(vertices, np.asarray(mesh.faces), camera, shape)
            observed = _mask(polygons, shape)
            union = int((model | observed).sum())
            rows.append({'entityId': entity['id'], 'label': entity.get('label'), 'photo': int(photo), 'observationId': oid,
                         'representationId': rep['id'], 'status': 'measured', 'iou': float((model & observed).sum() / union) if union else None,
                         'modelPixels': int(model.sum()), 'maskPixels': int(observed.sum()), 'facesBehindCamera': clipped})
    by_entity = {}
    for row in rows:
        if row['status'] == 'measured' and row['iou'] is not None:
            by_entity.setdefault(row['entityId'], []).append(row['iou'])
    summary = {entity: {'views': len(values), 'minIoU': min(values), 'medianIoU': float(np.median(values))} for entity, values in by_entity.items()}
    return {'schemaVersion': 1, 'revisionId': report['revision']['id'], 'documentSha256': report['revision']['documentSha256'],
            'scope': 'Model silhouette through the saved camera versus the saved segmentation polygon, per source view. Occlusion and mask errors lower IoU; not physical accuracy.',
            'objects': len({row['entityId'] for row in rows}), 'views': rows, 'summaryByEntity': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.out:
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    low = sorted(result['summaryByEntity'].items(), key=lambda item: item[1]['minIoU'])
    print(json.dumps({'objects': result['objects'], 'lowest': low[:8]}, ensure_ascii=False, indent=1))
