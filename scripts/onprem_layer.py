#!/usr/bin/env python3
"""Operator CLI: install a published measurement layer for an on-prem publication (containers/onprem/README.md, step 5).

A measurement layer (measurement-layer/<publicationId>.json + its mesh files, from the report website it was published on)
applies only to the revision it names. Publishing the same capture on-prem (scripts/onprem_publish.py) creates new publication,
project, branch, revision and asset ids and new revision-derived ids, so the layer is rebased onto the new publication, and only
if the new revision's document equals the layer's revision document up to:
  - asset ids, matched by sha256; project, branch, revision and job ids, matched by role;
  - ids derived from the revision id (measurementEvidence, identityDecisions): the renaming must be a bijection, every id must
    recompute from its record with the platform's digest in both documents, and the published record with the new ids put in
    must recompute to exactly its new partner. A derived id also hashes its measurement value; where that value differs only
    by float rounding (<= 1e-12, e.g. Mac arm64 vs Linux x86_64) the new value is put in too and the id must then reproduce;
  - the calibration provenance record (coordinateFrames[].scale.sourceRefs) and numbers <= 1e-9 apart.
Anything else differing: refused, nothing written.

  python scripts/onprem_layer.py --published PUBLISHED.json --layer LAYER_DIR/measurement-layer/SOURCE_ID.json \
      --publication NEW_ID [--catalog /catalog] [--www /www] [--dry-run]

PUBLISHED.json = GET /api/publications/SOURCE_ID of the publication service the layer was made for (its snapshot holds the
document). The new publication is read from the catalog export (CATALOG/NEW_ID/bundle.json). Writes WWW/measurement-layer/NEW_ID.json
and copies the layer's files (sha256-checked) to WWW/<their url>; a different file already installed under the same name is
refused. Re-running is safe. --dry-run writes nothing and also checks that a tampered document and a layer made for another
revision are refused. Prints one JSON line (ids and the comparison; no capability is involved).
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys
from uuid import NAMESPACE_URL, uuid4, uuid5

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
CALIBRATION = re.compile(r'^/coordinateFrames\[\d+\]/scale/sourceRefs')
LAYER_FILE = re.compile(r'measurement-layer/[A-Za-z0-9][A-Za-z0-9._-]*\.bin')


def deep_diff(a, b, path, out, numeric):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                out.append((f'{path}/{key}', a.get(key), b.get(key)))
            else:
                deep_diff(a[key], b[key], f'{path}/{key}', out, numeric)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for index, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f'{path}[{index}]', out, numeric)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
        if a != b:
            numeric['count'] += 1
            numeric['max'] = max(numeric['max'], abs(a - b))
            if CALIBRATION.match(path) or abs(a - b) > 1e-9:
                out.append((path, a, b))
    elif a != b:
        out.append((path, a, b))


def substitute(value, names):
    return json.loads(UUID.sub(lambda m: names.get(m.group(0), m.group(0)), json.dumps(value, ensure_ascii=False)))


def chain(publication):
    """Import revision first, published revision last (the publication snapshot's edit batches)."""
    batches = publication['snapshot']['editBatches']
    return [batches[0]['baseRevisionId']] + [b['revisionId'] for b in batches] if batches else [publication['sceneRevisionId']]


def roles(publication):
    revisions = chain(publication)
    return {publication['projectId']: 'project', publication['snapshot']['revision']['branchId']: 'branch',
            **{r: f'revision:intermediate{i}' for i, r in enumerate(revisions[1:-1], 1)},
            revisions[0]: 'revision:import', revisions[-1]: 'revision:published'}


def asset_sha(publication):
    snapshot = publication['snapshot']
    return {a['id']: a['sha256'] for a in snapshot['revision']['document']['assets']} | {e['assetId']: e['sha256'] for e in snapshot['assetManifest']}


def normalized(publication):
    """Ids a fresh database assigns anew (asset/project/branch/revision/job uuid4, timestamps) replaced by content or role."""
    document = publication['snapshot']['revision']['document']
    jobs = sorted({asset['jobId'] for asset in document.get('assets', []) if asset.get('jobId')})
    value = substitute(document, {**{aid: 'asset:' + sha for aid, sha in asset_sha(publication).items()}, **roles(publication),
                                  **{job: f'job{i}' for i, job in enumerate(jobs)}})

    def drop(node):
        if isinstance(node, dict):
            return {k: drop(v) for k, v in node.items() if k != 'createdAt'}
        return [drop(v) for v in node] if isinstance(node, list) else node
    return drop(value)


def derived_ids(document):
    """Every revision-derived id recomputed from its own record (ehs_spatial/platform/identity.py: snapshot_measurements,
    repair_measurement_sources, migrate_document)."""
    from ehs_spatial.platform.contracts import digest
    out = {}
    for e in document['entities']:
        for r in e.get('measurementEvidence', []):
            key = ('measurement-source:' + digest([r['sourceRevisionId'], r['sourceEvidenceId'], r['sourceRefs'], r['coordinateFrameId'], r['observationRefs']])
                   if r.get('sourceEvidenceId') else 'measurement:' + digest([r['sourceRevisionId'], r['sourceEntityId'], r['measurementKey'], r['originalMeasurement']]))
            out[r['id']] = str(uuid5(NAMESPACE_URL, key))
    for d in document.get('identityDecisions', []):
        out[d['id']] = str(uuid5(NAMESPACE_URL, 'source-binding:' + digest([d['baseRevisionId'], d['entityIds'][0]]))) if d.get('source') == 'source_binding' else None
    return out


def rebase(published, target, layer):
    """The layer moved onto `target`, and the comparison; ValueError (nothing to install) unless the documents match."""
    pub_doc, our_doc = published['snapshot']['revision']['document'], target['snapshot']['revision']['document']
    if layer.get('schemaVersion') != 1 or layer['publicationId'] != published['id'] or layer['revisionId'] != published['sceneRevisionId']:
        raise ValueError('the layer was not made for --published (publicationId / revisionId differ)')
    new_by_sha = {asset['sha256']: asset['id'] for asset in our_doc['assets']}
    assets = {aid: new_by_sha[sha] for aid, sha in asset_sha(published).items() if sha in new_by_sha}
    diffs, numeric = [], {'count': 0, 'max': 0.0}
    deep_diff(normalized(published), normalized(target), '', diffs, numeric)
    pairs, calibration, other = {}, [], []
    for path, a, b in diffs:
        if CALIBRATION.match(path):
            calibration.append(path)
        elif isinstance(a, str) and isinstance(b, str) and UUID.fullmatch(a) and UUID.fullmatch(b):
            pairs.setdefault(a, set()).add(b)
        else:
            other.append(path)
    renaming = {a: next(iter(b)) for a, b in pairs.items() if len(b) == 1}
    bijection = len(renaming) == len(pairs) and len(set(renaming.values())) == len(renaming)
    # Derived from the revision id? Put the new database's ids (import revision, project, branch, assets, the renaming itself)
    # into the published document: each renamed record must then recompute to exactly its new partner.
    src_chain, new_chain = chain(published), chain(target)
    swapped = substitute(pub_doc, {**assets, src_chain[0]: new_chain[0], published['projectId']: target['projectId'],
                                   published['snapshot']['revision']['branchId']: target['snapshot']['revision']['branchId'], **renaming})
    ours_records = {r['id']: r for e in our_doc['entities'] for r in e.get('measurementEvidence', [])}
    rounded, rounding = deepcopy(swapped), {}
    for record in (r for e in rounded['entities'] for r in e.get('measurementEvidence', [])):
        mine = ours_records.get(record['id'])
        if mine and record['originalMeasurement'] != mine['originalMeasurement']:
            out, num = [], {'count': 0, 'max': 0.0}
            deep_diff(record['originalMeasurement'], mine['originalMeasurement'], '', out, num)
            if not out and num['max'] <= 1e-12:
                record['originalMeasurement'], rounding[record['id']] = mine['originalMeasurement'], num
    derived = {name: derived_ids(doc) for name, doc in (('published', pub_doc), ('ours', our_doc), ('swapped', swapped), ('rounded', rounded))}
    from_revision = {a: derived['swapped'].get(b) == b for a, b in renaming.items()}
    with_rounding = {a: derived['rounded'].get(b) == b for a, b in renaming.items()}
    all_derived = all(a in derived['published'] and b in derived['ours'] for a, b in renaming.items())
    self_pub = all(k == v for k, v in derived['published'].items())
    self_ours = all(k == v for k, v in derived['ours'].items())
    pub_scale, our_scale = pub_doc['coordinateFrames'][0]['scale'], our_doc['coordinateFrames'][0]['scale']
    entity_of = {r['id']: e['id'] for e in our_doc['entities'] for r in e.get('measurementEvidence', [])}
    report = {
        'renamedIds': len(pairs), 'renamingBijection': bijection, 'renamedIdsAreRevisionDerived': all_derived,
        'derivedIdsRecomputePublished': f'{sum(k == v for k, v in derived["published"].items())}/{len(derived["published"])}',
        'derivedIdsRecomputeOnPrem': f'{sum(k == v for k, v in derived["ours"].items())}/{len(derived["ours"])}',
        'renamingReproducedFromRevisionIdAlone': f'{sum(from_revision.values())}/{len(from_revision)}',
        'renamingReproducedWithFloatRounding': f'{sum(with_rounding.values())}/{len(with_rounding)}',
        'floatRoundedRecords': [{'entityId': entity_of.get(b), 'id': b, 'numericLeaves': n['count'], 'maxAbs': n['max']} for b, n in rounding.items()],
        'assetIdsMatchedBySha': len(assets), 'calibrationProvenanceDifferences': len(calibration),
        'contentDifferences': len(other), 'contentDifferenceExamples': other[:25],
        'numericLeavesDiffering': numeric['count'], 'maxAbsNumericDifference': numeric['max'],
        'scaleEqual': (pub_scale.get('status'), pub_scale.get('nativeToMeters')) == (our_scale.get('status'), our_scale.get('nativeToMeters'))}
    if not (not other and bijection and all_derived and self_pub and self_ours and all(with_rounding.values()) and report['scaleEqual']):
        raise ValueError('the documents differ beyond verified id renaming: ' + json.dumps(report, ensure_ascii=False)[:3000])
    rebased = substitute(layer, {**assets, **renaming})
    rebased.update(publicationId=target['id'], revisionId=target['sceneRevisionId'])
    unresolved = sorted(set(UUID.findall(json.dumps(rebased))) - set(UUID.findall(json.dumps(our_doc))) - {target['id'], target['sceneRevisionId']})
    if unresolved:
        raise ValueError(f'rebased layer names ids the new revision does not have: {unresolved[:10]}')
    report['idsChanged'] = sorted({u for u in UUID.findall(json.dumps(layer)) if u in assets or u in renaming})
    return rebased, report


def layer_files(layer, layer_path, www):
    """(destination, bytes) for every mesh file of the layer, sha256-checked; refuses another file under the same name."""
    files = []
    for asset in layer.get('assets', []):
        if not LAYER_FILE.fullmatch(asset['url']):
            raise ValueError(f'layer file url must be measurement-layer/NAME.bin: {asset["url"]!r}')
        data = (layer_path.parent.parent / asset['url']).read_bytes()
        if hashlib.sha256(data).hexdigest() != asset['sha256'] or len(data) != asset['sizeBytes']:
            raise ValueError(f'{asset["url"]}: the file does not match the layer (sha256 / size)')
        destination = www / asset['url']
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != asset['sha256']:
            raise ValueError(f'{destination}: a different file is installed under this name')
        files.append((destination, data))
    return files


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_bytes(data)
    temporary.replace(path)


def self_test(published, target, layer):
    """--dry-run: a content change and a layer for another revision must both be refused."""
    tampered = deepcopy(target)
    entity = tampered['snapshot']['revision']['document']['entities'][0]
    entity['label'] = str(entity.get('label')) + ' (tampered)'
    for case, args in (('tampered document', (published, tampered, layer)), ('layer for another revision', (published, target, {**layer, 'revisionId': str(uuid4())}))):
        try:
            rebase(*args)
        except ValueError:
            continue
        raise AssertionError(f'{case} was accepted')
    return 'a tampered document and a layer for another revision are refused'


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    a.add_argument('--published', type=Path, required=True, help='GET /api/publications/SOURCE_ID of the layer\'s publication')
    a.add_argument('--layer', type=Path, required=True, help='LAYER_DIR/measurement-layer/SOURCE_ID.json (its .bin files next to it)')
    a.add_argument('--publication', required=True, help='the on-prem publication id (onprem_publish.py output)')
    a.add_argument('--catalog', type=Path, default=Path('/catalog'))
    a.add_argument('--www', type=Path, default=Path('/www'), help='report website root (its measurement-layer/ is written)')
    a.add_argument('--dry-run', action='store_true')
    args = a.parse_args(argv)
    published, layer = json.loads(args.published.read_text()), json.loads(args.layer.read_text())
    bundle = json.loads((args.catalog / args.publication / 'bundle.json').read_text())
    target = bundle['responses']['/api/publications/' + args.publication]
    rebased, report = rebase(published, target, layer)
    files = layer_files(layer, args.layer, args.www)
    destination = args.www / 'measurement-layer' / f'{target["id"]}.json'
    result = {'dryRun': args.dry_run, 'publicationId': target['id'], 'revisionId': target['sceneRevisionId'],
              'sourcePublicationId': published['id'], 'sourceRevisionId': layer['revisionId'], 'layer': str(destination),
              'files': [str(path) for path, _ in files], 'comparison': report}
    if args.dry_run:
        result['selfTest'] = self_test(published, target, layer)
    else:
        for path, data in files:
            write(path, data)
        write(destination, json.dumps(rebased, ensure_ascii=False).encode())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
