"""Policy evidence chain for one report revision; never a safety verdict.

object identity -> this revision's spatial facts -> compiled policy (source and
hash) -> the platform policy engine. Applicability is only ever asserted by a
reviewer with evidence, so the engine abstains here (applicability unknown,
no machine result). The ledger also lists what each rule would still need
after applicability is confirmed, read from this revision's actual facts.
"""
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
POLICIES = REPO / 'docs/policies'
# Catalog detection kinds -> perception labels the compiled rules name.
PERCEPTION = {'safety fence': 'safety fence', 'emergency stop button': 'emergency stop button', 'robot': 'industrial robot arm'}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_evidence(report, catalog):
    from ehs_spatial.platform import policy_engine
    from ehs_spatial.platform.contracts import validate_document
    document = json.loads(json.dumps(report['revision']['document']))
    kinds = {item['id']: item.get('kind') for item in catalog}
    for entity in document['entities']:
        # Engine targets by perception label; display labels stay in the report.
        entity['label'] = PERCEPTION.get(kinds.get(entity['id']), entity['label'])
    validate_document(document)
    scale = report['modelMeasurementScale']
    endpoints = report.get('endpointEstimation', {}).get('endpoints', [])
    rule_set = POLICIES / 'workcell_v2.md'
    items = []
    for path in sorted((POLICIES / 'compiled_v2').glob('*.json')):
        spec = json.loads(path.read_text())
        subjects = [{'entityId': entity_id, 'kind': kind, 'label': next(i['label'] for i in catalog if i['id'] == entity_id),
                     'identitySource': 'catalog detection kind from source-photo segmentation; not reviewer-confirmed'}
                    for entity_id, kind in kinds.items() if PERCEPTION.get(kind) in spec['subject_labels']]
        objects = [{'entityId': entity_id, 'kind': kind} for entity_id, kind in kinds.items() if PERCEPTION.get(kind) in spec['object_labels']]
        row = {'policyId': spec['policy_id'], 'sourceText': spec['source_text'], 'predicate': spec['predicate'],
               'threshold': spec['threshold'], 'unit': spec['unit'], 'severity': spec['severity'],
               'spec': {'file': path.relative_to(REPO).as_posix(), 'sha256': _sha(path)},
               'subjects': subjects, 'objects': objects, 'machineResult': None}
        if spec['unsupported_reason']:
            row.update(compileStatus='refused', applicability='unknown', refusal=spec['unsupported_reason'],
                       missingEvidence=['rule_cannot_be_executed_as_compiled'])
            items.append(row)
            continue
        check = {'kind': 'geometry', 'spec': {key: spec[key] for key in ('predicate', 'subject_labels', 'object_labels', 'threshold', 'unit')}}
        findings = policy_engine.evaluate_document(document, spec['policy_id'], {'jdm': policy_engine._jdm(check)}, 'reconstruction')
        engine = [{key: finding[key] for key in ('entityId', 'applicability', 'machineResult', 'missingEvidence')} for finding in findings]
        needed = [{'id': 'reviewer_applicability_confirmation', 'status': 'missing', 'detail': 'No reviewer asserted that this rule applies to this workcell.'},
                  {'id': 'operator_anchored_metric_scale', 'status': 'available' if scale['status'] == 'accepted_3d_reference' else 'missing',
                   'detail': f"Model scale status {scale['status']}; the engine requires operator-anchored metres."}]
        if spec['predicate'] in ('min_height', 'max_height'):
            measured = {row['objectId'] for row in endpoints if row.get('railPart', 'lower_edge') == 'lower_edge'}
            hypotheses = {row['objectId'] for row in endpoints if row.get('railPart', 'lower_edge') != 'lower_edge'}
            needed.append({'id': 'subject_full_height', 'status': 'missing',
                           'detail': 'This revision measures lower edges only' + (f" ({', '.join(sorted(measured))})" if measured else '')
                                     + (f"; {', '.join(sorted(hypotheses))}: lower-envelope hypothesis, not an edge" if hypotheses else '')
                                     + '; no top-of-object height fact exists.'})
        if spec['predicate'] in ('max_separation', 'min_separation'):
            needed.append({'id': 'reference_region', 'status': 'missing' if 'industrial robot arm' in spec['object_labels'] else 'unknown',
                           'detail': 'A robot work-area envelope is not modelled; the one static robot model of this scene is not a hazard zone.'
                                     if 'industrial robot arm' in spec['object_labels'] else 'Reference region identity is not a perception class.'})
        row.update(compileStatus='compiled', applicability=engine[0]['applicability'] if engine else 'unknown', engine=engine,
                   evidenceStillNeeded=needed, missingEvidence=sorted({item for finding in engine for item in finding['missingEvidence']}))
        if any(finding['machineResult'] is not None for finding in engine):
            raise ValueError('Policy engine produced a result without reviewer applicability')
        items.append(row)
    return {'schemaVersion': 1, 'engine': policy_engine.ENGINE_VERSION, 'revisionId': report['revision']['id'],
            'documentSha256': report['revision']['documentSha256'],
            'ruleSet': {'file': rule_set.relative_to(REPO).as_posix(), 'sha256': _sha(rule_set),
                        'status': 'illustrative rule set, not certified; no compliance authority'},
            'items': items, 'conclusion': 'No safety conclusion. Applicability is unconfirmed and the metric scale is unvalidated; '
                                          'photos cannot establish stopping performance, interlocks or detection-zone validity.'}
