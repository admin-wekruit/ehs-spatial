#!/usr/bin/env python3
"""Operator CLI: publish one capture run on the on-prem platform (containers/onprem/README.md, step 4).

import the public scene (with its geometry root) -> migrateScene v2 + setCalibration from a reference-object scale JSON ->
publish -> export the publication into the catalog -> optionally prepare the publication service's HTTP bodies.

  python scripts/onprem_publish.py RUN/public/scene.json --geometry-root RUN --scale SCALE.json --photo 2 \
      --object 'emergency stop BOR1 +00030.RAT01-ES01 (photo 2)' --title 'TITLE' \
      [--specification '{"redHeadDiameterM": 0.04, ...}'] [--imports-dir /data/imports] [--catalog /catalog] \
      [--prepare /publication-http] [--dry-run]

SCALE.json: the output of scripts/workcell_estop_scale.py (ehs-spatial worktree) or any file with the same core, i.e. what
reference_object_scale.joint_scale returns: nativeToMeters, limit, maxDeviation, passed, features[{specM, measuredNative, photo}];
photo = an integer, "... photo N" or the scene camera's image id (workcell_estop_scale.py).
The scale is accepted only if it passed and its nativeToMeters and maxDeviation recompute from its features. Every other key of
the file is kept as calibration provenance in the revision (scale.sourceRefs[0]). --photo N = the Nth photo of the scene
(camera order), the photo the scale was measured on (comma list for several); it must be exactly the set of photos the
features name ("... photo N", e.g. "030 photo 2").

Settings come from the environment, as for the platform API: PANOPTES_DATABASE_URL, PANOPTES_BLOB_BACKEND, PANOPTES_BLOB_ROOT
(ehs_spatial.platform.config.PlatformConfig.from_env) and PANOPTES_PLATFORM_API (default http://127.0.0.1:8792; the export
reads the publication back through the API's public GETs). The project's management capability is created by the importer in
--imports-dir (0600) and read from there internally; it is never printed or written anywhere else.
Re-running is safe: the import, the edit (skipped when the scale is already set) and the publication are idempotent, and an
existing export is kept.

--dry-run needs no database, no blob store and no API: it converts the scene in memory, validates the scale (and checks that a
tampered scale is refused), applies the exact edit operations with the platform's own apply_operations and validates the result.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from uuid import NAMESPACE_URL, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def checked_scale(scale):
    """The scale JSON at the trust boundary: it must have passed and be internally consistent (joint_scale recomputed)."""
    features = scale.get('features')
    if not isinstance(features, list) or not features:
        raise ValueError('scale: features missing')
    if any(not isinstance(f, dict) or not (isinstance(f.get('specM'), (int, float)) and f['specM'] > 0 and isinstance(f.get('measuredNative'), (int, float)) and f['measuredNative'] > 0) for f in features):
        raise ValueError('scale: every feature needs positive specM and measuredNative')
    value = scale.get('nativeToMeters')
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('scale: nativeToMeters must be a positive number')
    joint = math.exp(sum(math.log(f['specM'] / f['measuredNative']) for f in features) / len(features))
    worst = max(abs(f['measuredNative'] * joint / f['specM'] - 1) for f in features)
    if abs(joint / value - 1) > 1e-9 or abs(worst - scale.get('maxDeviation', -1)) > 1e-9:
        raise ValueError(f'scale: nativeToMeters / maxDeviation do not recompute from the features ({joint!r}, {worst!r})')
    if scale.get('passed') is not True or not worst < scale.get('limit', 0):
        raise ValueError(f'scale: not passed (max deviation {worst:.4f}, limit {scale.get("limit")})')
    return value


def scale_photos(scale, cameras):
    """The scene photo numbers (camera order, 1-based) the scale's features were measured on. A feature names its photo as an
    integer, as "... photo N" ('030 photo 2'), or by the image id of the scene camera it was measured in (what
    workcell_estop_scale.py writes); anything else is refused."""
    by_image = {c['imageId']: k + 1 for k, c in enumerate(cameras)}
    out = set()
    for f in scale['features']:
        photo = f.get('photo')
        if isinstance(photo, int) and not isinstance(photo, bool):
            out.add(photo); continue
        text = photo.strip() if isinstance(photo, str) else ''
        hit = next((n for image, n in by_image.items() if image and image in text), None)
        m = re.fullmatch(r'.*\bphoto\s*(\d+)', text, re.I)
        if hit is None and not m:
            raise ValueError('scale: every feature must name its photo: an integer, "... photo N" (scene camera order) or the camera image id')
        out.add(hit if hit is not None else int(m.group(1)))
    return out


def edit_operations(document, scale, photos, reference_object, specification=None):
    """migrateScene (v1 only) + setCalibration of the scene's frame; [] when that scale is already set."""
    value = checked_scale(scale)
    cameras = document['cameras']
    if not photos or any(not 1 <= n <= len(cameras) for n in photos):
        raise ValueError(f'--photo must be 1..{len(cameras)}')
    measured = scale_photos(scale, cameras)
    if set(photos) != measured:
        raise ValueError(f'--photo {sorted(set(photos))} is not the photo set the scale was measured on {sorted(measured)}')
    images = [cameras[n - 1]['imageId'] for n in photos]
    ref = {**{k: deepcopy(v) for k, v in scale.items() if k != 'nativeToMeters'}, 'kind': 'reference_object', 'object': reference_object,
           **({'imageId': images[0]} if len(images) == 1 else {'imageIds': images})}
    if specification is not None:
        ref['specification'] = specification
    frame = document['coordinateFrames'][0]  # ponytail: imported scenes have one native frame; a --frame flag when that changes
    calibrated = {'status': 'operator_anchored', 'nativeToMeters': value, 'sourceRefs': [ref]}
    if document.get('schemaVersion') == 2 and frame.get('scale') == calibrated:
        return []
    return ([{'type': 'migrateScene', 'schemaVersion': 2}] if document.get('schemaVersion') == 1 else []) + \
           [{'type': 'setCalibration', 'coordinateFrameId': frame['id'], 'scale': calibrated}]


def dry_run(args, scale, specification):
    from ehs_spatial.platform.contracts import validate_document
    from ehs_spatial.platform.repository import apply_operations
    from scripts.import_public_scene import import_document

    for bad in ({**scale, 'nativeToMeters': scale['nativeToMeters'] * 1.01}, {**scale, 'passed': False}, {**scale, 'features': []}):
        try:
            checked_scale(bad)
        except ValueError:
            continue
        raise AssertionError('a tampered scale was accepted')

    def put_asset(data, media_type, metadata):  # stand-in for blobs + register_asset: content-addressed ids, nothing stored
        sha = hashlib.sha256(data).hexdigest()
        return {'id': str(uuid5(NAMESPACE_URL, 'dry-run-asset:' + sha)), 'sha256': sha, 'sizeBytes': len(data), 'mediaType': media_type, 'storageKey': 'sha256/' + sha}
    document, manifest = import_document(args.scene, put_asset, geometry_root=args.geometry_root)
    document['captureId'] = str(uuid5(NAMESPACE_URL, 'dry-run-capture'))  # run_import sets the import job's capture id
    operations = edit_operations(document, scale, args.photo, args.object, specification)
    if len(document['cameras']) > 1:  # a photo set the scale was not measured on is refused
        all_photos = list(range(1, len(document['cameras']) + 1))
        wrong = [n for n in all_photos if n not in args.photo] or args.photo[:-1]  # the complement, or a strict subset when all are used
        try:
            edit_operations(document, scale, wrong, args.object, specification)
        except ValueError:
            pass
        else:
            raise AssertionError('a --photo the scale was not measured on was accepted')
    result, _ = apply_operations(document, operations, base_revision_id=str(uuid5(NAMESPACE_URL, 'dry-run-revision')))
    validate_document(result)
    frame = result['coordinateFrames'][0]
    assert result['schemaVersion'] == 2 and frame['scale']['status'] == 'operator_anchored' and frame['scale']['nativeToMeters'] == scale['nativeToMeters']
    assert edit_operations(result, scale, args.photo, args.object, specification) == [], 'a re-run must not edit again'
    return {'dryRun': True, 'scene': str(args.scene), 'sourceSha256': manifest['sourceSha256'], 'entities': len(result['entities']),
            'cameras': len(result['cameras']), 'assets': len(result['assets']), 'operations': [o['type'] for o in operations],
            'coordinateFrameId': frame['id'], 'nativeToMeters': frame['scale']['nativeToMeters'], 'maxDeviation': scale['maxDeviation'],
            'photo': args.photo, 'title': args.title,
            'checks': 'scale recomputed; tampered scales and a wrong --photo refused; edit applied and validated; re-run is a no-op'}


def publish(args, scale, specification, secret):
    from ehs_spatial.platform.config import PlatformConfig
    from ehs_spatial.platform.contracts import digest
    from ehs_spatial.platform.publication_site import compile_catalog
    from ehs_spatial.platform.runtime import services
    from scripts.export_platform_publication import export_publication
    from scripts.import_public_scene import run_import

    repo, blobs = services(PlatformConfig.from_env())
    repo.migrate()
    imports = Path(args.imports_dir)
    imported = run_import(args.scene, repo, blobs, imports, geometry_root=args.geometry_root)
    project, branch = imported['projectId'], imported['branchId']
    management = imports / (hashlib.sha256(Path(args.scene).read_bytes()).hexdigest() + '.management.json')
    secret.append(json.loads(management.read_text())['capability'])
    head_id = next(b for b in repo.get_project(project)['branches'] if b['id'] == branch)['headRevisionId']
    head = repo.get_revision(head_id)
    operations = edit_operations(head['document'], scale, args.photo, args.object, specification)
    if operations:
        body = {'requestId': str(uuid5(NAMESPACE_URL, 'onprem-publish:edit:' + digest([project, head_id, operations]))), 'branchId': branch,
                'baseRevisionId': head_id, 'operations': operations, 'label': 'v2 scene; reference-object scale'}
        head = repo.commit_edits(project, secret[0], body)['revision']
    revision = str(head['id'])
    scale_set = head['document']['coordinateFrames'][0]['scale']
    if head['document'].get('schemaVersion') != 2 or scale_set.get('nativeToMeters') != scale['nativeToMeters']:
        raise RuntimeError('the calibrated revision does not carry the requested scale')
    publication = repo.create_publication(project, secret[0], {'requestId': str(uuid5(NAMESPACE_URL, 'onprem-publish:publication:' + revision)),
                                                               'sceneRevisionId': revision, 'title': args.title})
    pid = str(publication['id'])
    output = Path(args.catalog) / pid
    exported = not output.exists()
    if exported:
        export_publication(os.environ.get('PANOPTES_PLATFORM_API', 'http://127.0.0.1:8792'), pid, output)
    result = {'projectId': project, 'importRevisionId': str(imported['sceneRevisionId']), 'revisionId': revision, 'publicationId': pid,
              'title': publication['title'], 'operations': [o['type'] for o in operations], 'nativeToMeters': scale_set['nativeToMeters'],
              'export': str(output), 'exported': exported}
    if args.prepare:
        result['prepare'] = compile_catalog(args.catalog, args.prepare)  # {routes, assets}; then restart the publication service
    return result


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('scene', type=Path)
    a.add_argument('--geometry-root', type=Path, required=True)
    a.add_argument('--scale', type=Path, required=True)
    a.add_argument('--photo', required=True, type=lambda s: [int(x) for x in s.split(',')])
    a.add_argument('--object', required=True, help='the reference object, as it should read in the calibration provenance')
    a.add_argument('--specification', help='JSON: the reference object specification and its source (provenance only)')
    a.add_argument('--title', required=True)
    a.add_argument('--imports-dir', default='/data/imports')
    a.add_argument('--catalog', default='/catalog')
    a.add_argument('--prepare', help='publication service HTTP bodies directory (scripts/prepare_publication_site.py)')
    a.add_argument('--dry-run', action='store_true')
    args = a.parse_args(argv)
    scale = json.loads(args.scale.read_text())
    specification = json.loads(args.specification) if args.specification else None
    secret = []
    try:
        result = dry_run(args, scale, specification) if args.dry_run else publish(args, scale, specification, secret)
        text = json.dumps(result, ensure_ascii=False, default=str)
        if secret and secret[0] in text:
            raise RuntimeError('refusing to print a result that contains the capability')
        print(text)
    except Exception as error:  # noqa: BLE001  report without the capability, whatever failed
        import traceback
        text = traceback.format_exc()
        for value in secret:
            text = text.replace(value, '[redacted]')
        print(text, file=sys.stderr)
        raise SystemExit(f'onprem_publish: failed: {type(error).__name__}')
    finally:
        secret.clear()


if __name__ == '__main__':
    main()
