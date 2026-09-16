"""Materialize an evidence-reviewed, exhaustive source-triangle ownership manifest.

Run with python -m scripts.research.partition_model_parts --document document.json
--manifest ownership.json --asset source.bin --output-dir new-partition. This
creates local derived assets and a new document; it never edits the source files.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from ehs_spatial.platform.contracts import PlatformError, canonical, digest, validate_document
from ehs_spatial.platform.blender_export import mesh_from_asset
from ehs_spatial.platform.identity import set_part_relation
from ehs_spatial.platform.reconstruction import _plan_projection
from ehs_spatial.platform.spatial import partition_packed_mesh


def partition_document(source, manifest, payload):
    """Return the new document and asset byte map only after all checks pass."""
    validate_document(source)
    if (source.get('schemaVersion') != 2 or manifest.get('schemaVersion') != 1 or manifest.get('confirmed') is not True
            or not isinstance(manifest.get('sourceRevisionId'), str) or not manifest['sourceRevisionId']
            or manifest.get('documentSha256') != digest(source)):
        raise PlatformError('confirmed_partition_source_required', 422)
    document = deepcopy(source)
    entities = {e['id']: e for e in document['entities']}
    parent_id = manifest.get('parentEntityId')
    parent = entities.get(parent_id)
    if parent is None:
        raise PlatformError('part_parent_not_found', 422)
    rep = next((r for r in parent.get('representations', []) if r['id'] == manifest.get('sourceRepresentationId')), None)
    asset = next((a for a in document['assets'] if a['id'] == manifest.get('sourceAssetId')), None)
    if (rep is None or rep['kind'] != 'generated_mesh' or parent.get('activeModelRepresentationId') != rep['id']
            or rep.get('sourceValidity') == 'stale' or asset is None or rep.get('assetId') != asset['id']
            or asset.get('sha256') != manifest.get('sourceAssetSha256')
            or hashlib.sha256(payload).hexdigest() != asset.get('sha256')):
        raise PlatformError('part_source_model_mismatch', 422)
    parts = manifest.get('parts')
    if (not isinstance(parts, list) or not parts or any(not isinstance(p, dict) or p.get('entityId') not in entities for p in parts)
            or len({p['entityId'] for p in parts}) != len(parts) or parent_id in {p['entityId'] for p in parts}):
        raise PlatformError('invalid_part_entities', 422)
    if any(e.get('parentEntityId') == parent_id for e in entities.values()):
        raise PlatformError('part_source_already_partitioned', 422)
    for part in parts:
        child = entities[part['entityId']]
        if child.get('activeModelRepresentationId') is not None or child.get('parentEntityId') is not None:
            raise PlatformError('part_target_already_modeled', 422)
    ownership = {parent_id: manifest.get('residualFaceIndices')}
    ownership.update({p['entityId']: p.get('faceIndices') for p in parts})
    slices = partition_packed_mesh(payload, asset, ownership)
    if any(slices[p['entityId']] is None for p in parts):
        raise PlatformError('part_geometry_empty', 422)
    source_pose = deepcopy(parent.get('currentModelTransform') or rep['transform'])
    partition_id = digest(manifest)
    outputs = {}
    provenance = {'operation': 'partition_model_parts', 'manifestSha256': partition_id,
        'sourceRevisionId': manifest['sourceRevisionId'], 'sourceRepresentationId': rep['id'],
        'sourceAssetId': asset['id'], 'sourceAssetSha256': asset['sha256'],
        'reason': manifest.get('reason'), 'evidenceRefs': deepcopy(manifest.get('evidenceRefs'))}
    for identity, sliced in slices.items():
        item = entities[identity]
        item['currentModelTransform'] = deepcopy(source_pose)
        lineage = {**deepcopy(provenance), 'sourceFaceIndices': ownership[identity]}
        if sliced is None:
            item['activeModelRepresentationId'] = None
            lineage['representationId'] = None
        else:
            data, metadata = sliced
            asset_id = str(uuid5(NAMESPACE_URL, f'part-asset:{partition_id}:{identity}'))
            representation_id = str(uuid5(NAMESPACE_URL, f'part-model:{partition_id}:{identity}'))
            derived_asset = {'id': asset_id, 'kind': 'generated_mesh', 'sha256': hashlib.sha256(data).hexdigest(),
                'sizeBytes': len(data), 'mediaType': 'application/octet-stream', **metadata,
                'sourcePartition': deepcopy(provenance)}
            derived_rep = {**deepcopy(rep), 'id': representation_id, 'assetId': asset_id,
                'coordinateFrameId': source_pose['coordinateFrameId'], 'transform': deepcopy(source_pose),
                'bounds': deepcopy(metadata['bounds']), 'sourceRefs': deepcopy(rep.get('sourceRefs') or []) + [deepcopy(provenance)]}
            derived_rep['placementSource'] = {'type': 'derived_source_partition', 'representationId': representation_id,
                'sourceRepresentationId': rep['id'], 'sourceRevisionId': manifest['sourceRevisionId'],
                'manifestSha256': partition_id, 'transformSha256': digest(source_pose),
                'sourcePlacementSource': deepcopy(rep.get('placementSource'))}
            # The reader only accepts a projection bound to these exact bytes
            # and pose; the old full-mesh cache cannot describe a partition.
            for field in ('planProjection', 'modelProjection', 'coverage'):
                derived_rep.pop(field, None)
            projection = _plan_projection(document, derived_rep, mesh_from_asset(data, derived_asset),
                                          derived_asset['sha256'], source_pose)
            if projection is not None:
                derived_rep['planProjection'] = projection
            item.setdefault('representations', []).append(derived_rep)
            item['activeModelRepresentationId'] = representation_id
            document['assets'].append(derived_asset)
            outputs[asset_id] = data
            lineage['representationId'] = representation_id
        item.setdefault('lineage', []).append(lineage)
    for part in parts:
        set_part_relation(document, {'type': 'setPartRelation', 'entityId': part['entityId'], 'parentEntityId': parent_id,
            'evidenceRefs': manifest.get('evidenceRefs'), 'reason': manifest.get('reason')}, base_revision_id=manifest['sourceRevisionId'])
    validate_document(document)
    return document, outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('document', 'manifest', 'asset', 'output-dir'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    source, manifest = json.loads(args.document.read_bytes()), json.loads(args.manifest.read_bytes())
    document, assets = partition_document(source, manifest, args.asset.read_bytes())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for identity, data in assets.items():
        (args.output_dir / f'{identity}.bin').write_bytes(data)
    (args.output_dir / 'document.json').write_bytes(canonical(document) + b'\n')
    (args.output_dir / 'ownership.json').write_bytes(canonical(manifest) + b'\n')
    print(json.dumps({'document': str(args.output_dir / 'document.json'), 'documentSha256': digest(document),
        'derivedAssetCount': len(assets), 'sourceUnchanged': digest(source) == manifest['documentSha256']}))


if __name__ == '__main__':
    main()
