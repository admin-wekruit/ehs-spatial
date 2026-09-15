"""Read-only, byte-verified model coverage for a revision and publication catalog."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ehs_spatial.platform.blender_export import mesh_from_asset
from ehs_spatial.platform.contracts import PlatformError, digest
from ehs_spatial.platform.spatial import primitive_mesh
from ehs_spatial.platform.storage import LocalBlobStore


def audit_document(document, blob_root, *, cache=None):
    cache = {} if cache is None else cache
    store = LocalBlobStore(blob_root)
    assets = {a['id']: a for a in document['assets']}
    checks, raw_assets = [], {}
    for asset in assets.values():
        key = asset.get('storageKey') or 'sha256/' + str(asset.get('sha256', ''))
        record = {'assetId': asset['id'], 'expectedSha256': asset.get('sha256'),
                  'expectedSizeBytes': asset.get('sizeBytes')}
        try:
            path = store.root / key
            if not path.resolve().is_relative_to(store.root) or key != 'sha256/' + asset.get('sha256', ''):
                raise PlatformError('invalid_storage_key')
            cached = cache.get(str(path))
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if not cached or cached[0] != signature:
                with path.open('rb') as stream:
                    cache[str(path)] = (signature, hashlib.file_digest(stream, 'sha256').hexdigest())
            actual = cache[str(path)][1]
            record.update(actualSha256=actual, actualSizeBytes=stat.st_size,
                          verified=actual == asset['sha256'] and stat.st_size == asset['sizeBytes'])
            if record['verified']:
                raw_assets[asset['id']] = path
            else:
                record['errorCode'] = 'blob_integrity_error'
        except (OSError, PlatformError) as error:
            record.update(actualSha256=None, verified=False,
                          errorCode=error.code if isinstance(error, PlatformError) else type(error).__name__)
        checks.append(record)

    observations = {o['id']: o for o in document['observations']}
    rows = []
    for entity in document['entities']:
        if entity.get('sourceContext'):
            continue
        active = next((r for r in entity.get('representations', [])
                       if r['id'] == entity.get('activeModelRepresentationId')), None)
        refs = [observations[o] for o in entity.get('observationRefs', []) if o in observations]
        row = {'sourceEntityId': entity['id'], 'label': entity.get('label'),
               'observationRefs': [{'id': o['id'], 'revision': o.get('revision'), 'imageId': o['imageId'],
                                    'maskAssetId': o.get('maskAssetId'), 'sha256': digest(o)} for o in refs],
               'parentEntityId': entity.get('parentEntityId'), 'geometryRole': entity.get('geometryRole'),
               'activeRepresentationId': (active or {}).get('id'), 'modelDeclared': False,
               'modelDecoded': False, 'vertexCount': 0, 'triangleCount': 0,
               'placementState': (active or {}).get('placementState'), 'errorCode': None,
               'inputSha256': digest({'entity': entity, 'observations': refs}),
               'missingObservationIds': sorted(set(entity.get('observationRefs', [])) - observations.keys())}
        if active and active.get('kind') in ('generated_mesh', 'primitive'):
            row['modelDeclared'] = True
            try:
                if active.get('sourceValidity') == 'stale':
                    raise PlatformError('stale_model')
                if active['kind'] == 'primitive':
                    mesh = primitive_mesh(active['primitive'])
                else:
                    asset_id = active.get('assetId')
                    if asset_id not in raw_assets:
                        raise PlatformError('model_asset_unverified')
                    mesh = mesh_from_asset(raw_assets[asset_id].read_bytes(), assets[asset_id])
                if len(mesh.faces) == 0:
                    raise PlatformError('empty_model_faces')
                row.update(modelDecoded=True, vertexCount=len(mesh.vertices), triangleCount=len(mesh.faces))
            except (PlatformError, ValueError, KeyError, TypeError) as error:
                row['errorCode'] = error.code if isinstance(error, PlatformError) else type(error).__name__
        rows.append(row)
    return {'schemaVersion': 1, 'documentSha256': digest(document), 'newModelCalls': 0,
            'entityNodes': len(document['entities']), 'objectRecords': len(rows),
            'sourceContextIds': [e['id'] for e in document['entities'] if e.get('sourceContext')],
            'observations': len(observations), 'assetChecks': checks, 'objects': rows,
            'modelDeclaredCount': sum(r['modelDeclared'] for r in rows),
            'modelDecodedCount': sum(r['modelDecoded'] for r in rows),
            'allAssetsVerified': all(a['verified'] for a in checks),
            'scope': 'Stored bytes and mesh decoding; browser visibility and placement are not certified.'}


def audit_catalog(catalog):
    reports, cache = [], {}
    for path in sorted(Path(catalog).glob('*/bundle.json')):
        bundle = json.loads(path.read_bytes())
        publication = bundle['responses']['/api/publications/' + bundle['publicationId']]
        revision = publication['snapshot']['revision']
        document = revision['document']
        # Publication exports use hash-named blobs, unlike LocalBlobStore keys.
        verified, failures = [], []
        manifest = {item['assetId']: item for item in publication['snapshot']['assetManifest']}
        for asset in document['assets']:
            item = manifest.get(asset['id'])
            if item is None:
                failures.append({'assetId': asset['id'], 'errorCode': 'scene_asset_missing_from_manifest'})
            elif any(asset.get(key) is not None and asset[key] != item.get(key) for key in ('sha256', 'sizeBytes')):
                failures.append({'assetId': asset['id'], 'errorCode': 'scene_asset_manifest_mismatch'})
        for item in publication['snapshot']['assetManifest']:
            sha = item['sha256']
            if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
                failures.append({'assetId': item['assetId'], 'errorCode': 'invalid_sha256'})
                continue
            blob = path.parent / 'blobs' / sha
            try:
                stat = blob.stat()
                signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                previous = cache.get(str(blob))
                if not previous or previous[0] != signature:
                    with blob.open('rb') as stream:
                        cache[str(blob)] = (signature, hashlib.file_digest(stream, 'sha256').hexdigest())
                if cache[str(blob)][1] != sha or stat.st_size != item['sizeBytes']:
                    raise ValueError('blob_integrity_error')
                verified.append(item['assetId'])
            except (OSError, ValueError) as error:
                failures.append({'assetId': item['assetId'], 'errorCode': type(error).__name__})
        reports.append({'publicationId': publication['id'], 'projectId': publication['projectId'],
                        'revisionId': revision['id'], 'documentSha256': digest(document),
                        'objectRecords': sum(not e.get('sourceContext') for e in document['entities']),
                        'verifiedAssetCount': len(verified), 'failures': failures})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', required=True, type=Path)
    parser.add_argument('--blob-root', default='.platform/blobs', type=Path)
    parser.add_argument('--catalog', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit_document(json.loads(args.document.read_bytes()), args.blob_root)
    if args.catalog:
        result['catalog'] = audit_catalog(args.catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('objectRecords', 'observations', 'modelDeclaredCount',
                                             'modelDecodedCount', 'allAssetsVerified', 'newModelCalls')}))
    if (not result['allAssetsVerified'] or any(x['failures'] for x in result.get('catalog', []))
            or any(row['modelDeclared'] and not row['modelDecoded'] for row in result['objects'])):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
