"""Identity changes preserve evidence and use the ordinary revision writer."""
from copy import deepcopy
from uuid import uuid4

import pytest

from ehs_spatial.platform.contracts import PlatformError, SceneDocument, digest, empty_document, validate_document
from ehs_spatial.platform.identity import migrate_document, resolve_entity_id
from ehs_spatial.platform.repository import apply_operations


BASE = str(uuid4())


def source_scene():
    doc = empty_document()
    doc["captureId"] = "capture"
    doc["coordinateFrames"] = [{"id": "frame", "convention": "opencv", "scale": {"status": "uncalibrated", "nativeToMeters": None}, "ground": None}]
    for i in (1, 2):
        image, obs, eid, rid = f"image-{i}", f"observation-{i}", f"entity-{i}", f"model-{i}"
        doc["assets"].append({"id": image, "kind": "source_image", "sha256": str(i) * 64})
        doc["cameras"].append({"id": f"camera-{i}", "imageId": image, "coordinateFrameId": "frame", "width": 20, "height": 20,
                               "K": [[10, 0, 10], [0, 10, 10], [0, 0, 1]], "cameraToWorld": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]})
        doc["observations"].append({"id": obs, "revision": 1, "imageId": image, "originalPixelBox": [1, 2, 4, 6], "maskAssetId": None})
        transform = {"coordinateFrameId": "frame", "position": [i, 0, 1], "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]}
        rep = {"id": rid, "kind": "primitive", "assetId": None, "coordinateFrameId": "frame", "transform": transform,
               "primitive": {"kind": "box", "dimensions": [1, 2, 3]}, "placementState": "confirmed", "sourceRefs": [{"observationId": obs}]}
        doc["entities"].append({"id": eid, "label": "Same appearance", "observationRefs": [obs], "associationState": "association_pending",
                                "representations": [rep], "currentModelTransform": deepcopy(transform), "measurements": {"height": {"value": i, "unit": "native", "sourceRefs": [{"observationId": obs, "revision": 1}]}},
                                "sourceRefs": [{"recordId": f"source-{i}"}], "lineage": [], "material": {"color": f"color-{i}"}})
    doc["annotations"] = [{"id": "note", "kind": "note", "entityId": "entity-2", "sourceRefs": [{"observationId": "observation-2"}]}]
    doc["reportEvidence"] = {"objects": [{"entityId": "entity-2", "sourceRecordId": "source-2", "views": [{"observationId": "observation-2"}]}],
                             "historical": {"inventory": [{"inventoryIndex": 7, "entityIds": ["entity-2"]}],
                                            "cad": {"regions": [{"inventoryIndex": 7, "entityIds": ["entity-2"], "polygon": [[1, 2], [3, 4]]}]},
                                            "findings": [{"entityId": "entity-2", "facts": [{"entityId": "native-source-id", "value": 9}]}]}}
    return doc


def decision(doc, value="same", ids=None, *, source="manual", supersedes=None, base=BASE):
    ids = ids or [e["id"] for e in doc["entities"]]
    groups = [next(e["observationRefs"] for e in doc["entities"] if e["id"] == identity) for identity in ids]
    return {"id": str(uuid4()), "decision": value, "source": source, "baseRevisionId": base, "entityIds": ids,
            "observationGroups": deepcopy(groups), "survivorId": ids[0] if value == "same" else None,
            "evidenceRefs": [{"kind": "observation", "observationId": oid, "observationRevision": 1} for group in groups for oid in group],
            "reason": "Reviewed both observations", "supersedesDecisionId": supersedes}


def merge(doc, chosen=None, *, source="manual", supersedes=None):
    d = decision(doc, source=source, supersedes=supersedes)
    op = {"type": "mergeEntities", "entityIds": d["entityIds"], "survivorId": d["survivorId"], "decisionId": d["id"]}
    if chosen is not None:
        op["activeModelRepresentationId"] = chosen
    return apply_operations(doc, [{"type": "recordIdentityDecision", "decision": d}, op], base_revision_id=BASE)


def test_explicit_migration_preserves_frozen_source_and_builds_evidence_and_bindings():
    source = source_scene()
    before = digest(source)
    doc = migrate_document(source, base_revision_id=BASE)
    assert digest(source) == before and source["schemaVersion"] == 1
    assert doc["schemaVersion"] == 2 and doc["captureIds"] == ["capture"]
    assert doc["geometryBindings"]["image-1"] == {"geometrySolutionId": "frame", "cameraId": "camera-1"}
    assert {o["captureId"] for o in doc["observations"]} == {"capture"}
    item = doc["entities"][0]
    assert item["activeModelRepresentationId"] == "model-1"
    record = item["measurementEvidence"][0]
    assert record["originalMeasurement"] == source["entities"][0]["measurements"]["height"]
    assert record["sourceRevisionId"] == BASE and record["sourceEntityId"] == item["id"]
    assert item["measurementSelections"]["height"] == record["id"]
    assert SceneDocument.model_validate(doc).model_dump(exclude_unset=True)["schemaVersion"] == 2
    assert migrate_document(source, base_revision_id=BASE) == doc


def test_source_equivalence_preserves_each_observation_and_prefers_existing_source_group():
    from ehs_spatial.platform.identity import apply_source_equivalences
    source = source_scene()
    extra = deepcopy(source['observations'][1])
    extra['id'] = 'observation-3'
    source['observations'].append(extra)
    source['observations'][1]['imageId'] = 'image-1'
    source['entities'][1]['observationRefs'].append(extra['id'])
    source['entities'][1]['associationEvidence'] = {'method': 'explicit_source_id'}
    source['assets'] += [{'id': 'sam', 'sha256': 'a'*64}, {'id': 'proof', 'sha256': 'b'*64}]
    doc = migrate_document(source, base_revision_id=BASE)
    doc['sourceIdentityEvidence'] = [{'assetId': 'proof', 'sha256': 'b'*64}]
    pair = {'observationRefs': [{'observationId': 'observation-1', 'revision': 1}, {'observationId': 'observation-2', 'revision': 1}],
            'imageId': 'image-1', 'imageSha256': '1'*64,
            'sourceRef': {'assetId': 'sam', 'sha256': 'a'*64, 'jsonPointer': '/rle/0'},
            'canonicalMaskSha256': 'c'*64, 'canonicalShape': [20, 20],
            'evidenceRefs': [{'assetId': 'proof', 'sha256': 'b'*64, 'jsonPointer': '/pairs/0'}]}
    before = deepcopy(doc)
    result = apply_source_equivalences(doc, [pair], base_revision_id=BASE)
    kept = doc['entities'][0]
    assert kept['id'] == 'entity-2' and len(result) == 1
    assert set(kept['observationRefs']) == {'observation-1', 'observation-2', 'observation-3'}
    assert doc['observations'] == before['observations'] and doc['assets'] == before['assets']
    assert sorted(kept['representations'], key=lambda r:r['id']) == sorted([r for e in before['entities'] for r in e['representations']], key=lambda r:r['id'])
    assert kept['activeModelRepresentationId'] == 'model-2'
    assert kept['measurements']['height'] is None and len(kept['measurementEvidence']) == 2
    assert doc['reportEvidence']['historical']['findings'] == before['reportEvidence']['historical']['findings']
    assert apply_source_equivalences(doc, [pair], base_revision_id=BASE) == []
    validate_document(doc)
    repeated = deepcopy(before)
    apply_source_equivalences(repeated, [{**pair, 'observationRefs': pair['observationRefs'][::-1]}], base_revision_id=BASE)
    assert repeated == doc
    unselected = deepcopy(before)
    unselected['entities'][1]['activeModelRepresentationId'] = None
    unselected['entities'][1]['currentModelTransform'] = None
    apply_source_equivalences(unselected, [pair], base_revision_id=BASE)
    assert unselected['entities'][0]['activeModelRepresentationId'] is None
    forged = doc['identityDecisions'][-1]
    with pytest.raises(PlatformError, match='identity_source_binding_migration_only'):
        apply_operations(before, [{'type': 'recordIdentityDecision', 'decision': forged}], base_revision_id=BASE)
    for bad in ({**pair, 'imageSha256':'f'*64}, {**pair, 'observationRefs':[pair['observationRefs'][0], {'observationId':'observation-2','revision':2}]}):
        unchanged = deepcopy(before)
        with pytest.raises(PlatformError):
            apply_source_equivalences(unchanged, [bad], base_revision_id=BASE)
        assert unchanged == before
    different = decision(before, 'different')
    separated, _ = apply_operations(before, [{'type':'recordIdentityDecision', 'decision':different}], base_revision_id=BASE)
    unchanged = deepcopy(separated)
    with pytest.raises(PlatformError, match='identity_decision_conflict'):
        apply_source_equivalences(separated, [pair], base_revision_id=BASE)
    assert separated == unchanged


def test_measurement_source_binding_is_exact_and_repairs_only_selected_records():
    from ehs_spatial.platform.identity import repair_measurement_sources, source_observation_ids
    source = source_scene()
    source['assets'].append({'id':'source-json', 'sha256':'a'*64})
    for observation in source['observations']:
        observation['sourceRefs'] = [{'assetId':'source-json', 'sourceRecordId':observation['id']}]
    entity = source['entities'][0]
    refs = [{'assetId':'source-json', 'sourceRecordId':'observation-1'}]
    entity['measurements'] = {'widthNative': 4, 'basis': {'kind':'native'}, 'unknown':None, 'coordinateFrameId':'frame', 'sourceRefs':refs}
    doc = migrate_document(source, base_revision_id=BASE)
    entity = doc['entities'][0]
    assert all(r['observationRefs'] == ['observation-1'] for r in entity['measurementEvidence'])
    assert source_observation_ids(doc, entity, {'sourceRefs':refs}, coordinate_frame_id='other-frame') == []
    assert source_observation_ids(doc, entity, {'sourceRefs':[dict(refs[0], assetId='other-asset')]}, coordinate_frame_id='frame') == []
    assert source_observation_ids(doc, entity, {'sourceRefs':[dict(refs[0], imageId='image-2')]}, coordinate_frame_id='frame') == []
    assert source_observation_ids(doc, entity, {'sourceRefs':[dict(refs[0], cameraId='camera-2')]}, coordinate_frame_id='frame') == []
    assert source_observation_ids(doc, entity, {'sourceRefs':[dict(refs[0], sha256='f'*64)]}, coordinate_frame_id='frame') == []
    # Simulate an existing v2 revision before source binding was recorded.
    for record in entity['measurementEvidence']:
        record['observationRefs'] = []
        record.pop('sourceRefs', None)
        record.pop('coordinateFrameId', None)
    # JSONB/canonical order must not let repairing the sourceRefs selection hide
    # that original group from later scalar records in the same pass.
    entity['measurementSelections'] = dict(sorted(entity['measurementSelections'].items()))
    old = deepcopy(entity['measurementEvidence'])
    entity['measurementSelections']['basis'] = None
    entity['measurements']['basis'] = None
    changed = repair_measurement_sources(doc, base_revision_id='source-repair')
    assert changed and entity['measurementEvidence'][:len(old)] == old
    selected = next(r for r in entity['measurementEvidence'] if r['id'] == entity['measurementSelections']['widthNative'])
    assert selected['observationRefs'] == ['observation-1'] and selected['originalMeasurement'] == 4
    assert selected['sourceEvidenceId'] == next(r['id'] for r in old if r['measurementKey'] == 'widthNative')
    assert entity['measurements']['basis'] is None and entity['measurementSelections']['basis'] is None
    assert entity['measurements']['unknown'] is None
    assert repair_measurement_sources(doc, base_revision_id='source-repair') == []
    validate_document(doc)


def test_merge_preserves_observations_models_measurements_and_only_maps_attachments():
    doc = migrate_document(source_scene(), base_revision_id=BASE)
    before = deepcopy(doc)
    result, inverse = merge(doc)
    kept = result["entities"][0]
    assert len(result["entities"]) == 1 and kept["id"] == "entity-1"
    assert result["observations"] == before["observations"] and result["assets"] == before["assets"]
    assert kept["representations"] == [r for e in before["entities"] for r in e["representations"]]
    assert len(kept["measurementEvidence"]) == 2 and kept["measurements"]["height"] is None
    assert kept["measurementSelections"]["height"] is None
    assert kept["activeModelRepresentationId"] == "model-1"
    assert result["annotations"][0]["entityId"] == "entity-1"
    assert result["reportEvidence"]["objects"][0]["entityId"] == "entity-1"
    assert result["reportEvidence"]["historical"]["inventory"][0]["entityIds"] == ["entity-1"]
    assert result["reportEvidence"]["historical"]["cad"]["regions"][0]["entityIds"] == ["entity-1"]
    assert result["reportEvidence"]["historical"]["findings"] == before["reportEvidence"]["historical"]["findings"]
    assert inverse == [{"type": "restoreDocument", "document": before}] and doc == before
    assert resolve_entity_id(result, "entity-2") == "entity-1"


def test_active_model_edits_never_change_alternative_models():
    merged, _ = merge(migrate_document(source_scene(), base_revision_id=BASE))
    alternative = deepcopy(merged["entities"][0]["representations"][1])
    transform = {**merged["entities"][0]["currentModelTransform"], "position": [9, 8, 7]}
    edited, _ = apply_operations(merged, [{"type": "setTransform", "entityId": "entity-1", "transform": transform},
                                        {"type": "setMaterial", "entityId": "entity-1", "material": {"color": "changed"}},
                                        {"type": "setPrimitive", "entityId": "entity-1", "primitive": {"kind": "box", "dimensions": [2, 2, 2]}}], base_revision_id=BASE)
    assert edited["entities"][0]["representations"][1] == alternative
    assert edited["entities"][0]["representations"][0]["transform"] == transform
    assert edited["entities"][0]["representations"][0]["material"] == {"color": "changed"}


def test_same_decision_cannot_be_recorded_without_merge_and_stale_evidence_is_rejected():
    doc = migrate_document(source_scene(), base_revision_id=BASE)
    d = decision(doc)
    with pytest.raises(PlatformError, match="identity_same_requires_merge"):
        apply_operations(doc, [{"type": "recordIdentityDecision", "decision": d}], base_revision_id=BASE)
    d["evidenceRefs"][0]["observationRevision"] = 2
    with pytest.raises(PlatformError, match="identity_observation_revision_mismatch"):
        apply_operations(doc, [{"type": "recordIdentityDecision", "decision": d}], base_revision_id=BASE)
    with pytest.raises(PlatformError, match="identity_base_revision_mismatch"):
        apply_operations(doc, [{"type": "recordIdentityDecision", "decision": decision(doc, "different")}], base_revision_id=str(uuid4()))


def test_manual_different_survives_mask_revision_and_blocks_automatic_and_manual_merge():
    doc = migrate_document(source_scene(), base_revision_id=BASE)
    d = decision(doc, "different")
    recorded, _ = apply_operations(doc, [{"type": "recordIdentityDecision", "decision": d}], base_revision_id=BASE)
    recorded["observations"][0]["revision"] = 2
    validate_document(recorded)
    for source in ("geometry", "manual"):
        candidate = decision(recorded, source=source)
        candidate["evidenceRefs"][0]["observationRevision"] = 2
        with pytest.raises(PlatformError, match="identity_decision_conflict"):
            apply_operations(recorded, [{"type": "recordIdentityDecision", "decision": candidate}, {"type": "mergeEntities", "decisionId": candidate["id"], "entityIds": candidate["entityIds"], "survivorId": "entity-1"}], base_revision_id=BASE)
    candidate["source"] = "manual"
    candidate["supersedesDecisionId"] = d["id"]
    accepted, _ = apply_operations(recorded, [{"type": "recordIdentityDecision", "decision": candidate}, {"type": "mergeEntities", "decisionId": candidate["id"], "entityIds": candidate["entityIds"], "survivorId": "entity-1"}], base_revision_id=BASE)
    assert len(accepted["entities"]) == 1


def test_split_partitions_models_and_evidence_and_does_not_broadcast_record_metrics():
    doc, _ = merge(migrate_document(source_scene(), base_revision_id=BASE))
    item = doc["entities"][0]
    doc["reportEvidence"]["objects"] = [{"entityId": item["id"], "sourceRecordId": "whole-record", "views": [{"observationId": f"observation-{i}"} for i in (1, 2)], "metrics": {"score": 0.5}}]
    d = decision(doc, "different", ids=[item["id"]])
    d["observationGroups"] = [[f"observation-{i}"] for i in (1, 2)]
    d["supersedesDecisionId"] = doc["identityDecisions"][-1]["id"]
    groups = [{"id": f"child-{i}", "observationRefs": [f"observation-{i}"], "representationIds": [f"model-{i}"],
               "measurementEvidenceIds": [r["id"] for r in item["measurementEvidence"] if r["sourceEntityId"] == f"entity-{i}"]} for i in (1, 2)]
    split, _ = apply_operations(doc, [{"type": "recordIdentityDecision", "decision": d}, {"type": "splitEntity", "entityId": item["id"], "decisionId": d["id"], "groups": groups}], base_revision_id=BASE)
    assert split["observations"] == doc["observations"] and split["assets"] == doc["assets"]
    assert sum(len(e["representations"]) for e in split["entities"]) == 2
    assert sum(len(e["measurementEvidence"]) for e in split["entities"]) == 2
    record = split["reportEvidence"]["objects"][0]
    assert record["entityId"] is None and record["metrics"] == {"score": 0.5}
    assert [v["entityId"] for v in record["views"]] == ["child-1", "child-2"]
    assert resolve_entity_id(split, "entity-2", observation_id="observation-2") == "child-2"
    with pytest.raises(PlatformError, match="identity_resolution_ambiguous"):
        resolve_entity_id(split, "entity-1")
    with pytest.raises(PlatformError, match="identity_decision_conflict"):
        merge(split, source="geometry")


def test_cad_reference_follows_merge_and_split_observation_ownership():
    doc = migrate_document(source_scene(), base_revision_id=BASE)
    extra = {**deepcopy(doc['observations'][0]), 'id': 'observation-3'}
    doc['observations'].append(extra)
    doc['entities'][1]['observationRefs'].append(extra['id'])
    reference = {'referenceImageId': 'image-1', 'status': 'resolved', 'source': 'explicit_reference_image',
                 'sourceRefs': [{'observationId': 'observation-1', 'revision': 1}], 'evidenceRefs': [{'baseRevisionId': BASE}]}
    doc['entities'][0]['cadReference'] = deepcopy(reference)
    merged, inverse = merge(doc)
    parent = merged['entities'][0]
    assert parent['cadReference'] == {**reference, 'sourceRefs': [
        {'observationId': 'observation-1', 'revision': 1}, {'observationId': 'observation-3', 'revision': 1}]}
    assert inverse == [{'type': 'restoreDocument', 'document': doc}]
    d = decision(merged, 'different', ids=[parent['id']], supersedes=merged['identityDecisions'][-1]['id'])
    d['observationGroups'] = [[f'observation-{i}'] for i in (1, 2, 3)]
    groups = [{'id': f'child-{i}', 'observationRefs': [f'observation-{i}'],
               'representationIds': [f'model-{i}'] if i < 3 else [],
               'measurementEvidenceIds': [r['id'] for r in parent['measurementEvidence'] if r['sourceEntityId'] == f'entity-{i}']}
              for i in (1, 2, 3)]
    split, _ = apply_operations(merged, [{'type': 'recordIdentityDecision', 'decision': d},
        {'type': 'splitEntity', 'entityId': parent['id'], 'decisionId': d['id'], 'groups': groups}], base_revision_id=BASE)
    for child, image_id in zip(split['entities'], ('image-1', 'image-2', 'image-1')):
        ref = child['cadReference']
        assert ref['referenceImageId'] == image_id and ref['status'] == 'resolved'
        assert ref['sourceRefs'] == [{'observationId': child['observationRefs'][0], 'revision': 1}]
        assert ref['source'] == ('single_source_image' if image_id == 'image-2' else reference['source'])
    assert split['observations'] == doc['observations'] and split['assets'] == doc['assets']
    from ehs_spatial.platform.identity import refresh_cad_reference
    ambiguous = {'observationRefs': ['observation-1', 'observation-2']}
    refresh_cad_reference(doc, ambiguous)
    assert ambiguous['cadReference']['status'] == 'unresolved' and ambiguous['cadReference']['source'] == 'ambiguous_sources'


def test_invalid_split_is_atomic_and_no_source_representation_can_disappear():
    doc, _ = merge(migrate_document(source_scene(), base_revision_id=BASE))
    before = deepcopy(doc)
    d = decision(doc, "different")
    d["observationGroups"] = [["observation-1"], ["observation-2"]]
    d["supersedesDecisionId"] = doc["identityDecisions"][-1]["id"]
    groups = [{"id": f"child-{i}", "observationRefs": [f"observation-{i}"], "representationIds": [], "measurementEvidenceIds": []} for i in (1, 2)]
    with pytest.raises(PlatformError, match="identity_split_partition_invalid"):
        apply_operations(doc, [{"type": "recordIdentityDecision", "decision": d}, {"type": "splitEntity", "entityId": "entity-1", "decisionId": d["id"], "groups": groups}], base_revision_id=BASE)
    assert doc == before


def test_v2_rejects_dangling_attachments_and_multiple_observation_owners():
    doc = migrate_document(source_scene(), base_revision_id=BASE)
    bad = deepcopy(doc)
    bad["reportEvidence"]["historical"]["cad"]["regions"][0]["entityIds"] = ["not-in-scene"]
    with pytest.raises(PlatformError, match="identity_attachment_not_found"):
        validate_document(bad)
    bad = deepcopy(doc)
    bad["entities"][1]["observationRefs"].append("observation-1")
    with pytest.raises(PlatformError, match="observation_multiple_owners"):
        validate_document(bad)


def test_manual_identity_without_observations_still_has_entity_scope_and_exclusions():
    source = source_scene()
    for entity in source["entities"]:
        entity.update(observationRefs=[], measurements={})
    source["annotations"], source["reportEvidence"] = [], {}
    doc = migrate_document(source, base_revision_id=BASE)
    d = decision(doc, "different")
    d["evidenceRefs"] = [{"kind": "asset", "assetId": "image-1", "sha256": "1" * 64}]
    recorded, _ = apply_operations(doc, [{"type": "recordIdentityDecision", "decision": d}], base_revision_id=BASE)
    same = {**d, "id": str(uuid4()), "decision": "same", "survivorId": "entity-1"}
    operations = [{"type": "recordIdentityDecision", "decision": same}, {"type": "mergeEntities", "entityIds": same["entityIds"], "survivorId": same["survivorId"], "decisionId": same["id"]}]
    with pytest.raises(PlatformError, match="identity_decision_conflict"):
        apply_operations(recorded, operations, base_revision_id=BASE)
    same["supersedesDecisionId"] = d["id"]
    merged, _ = apply_operations(recorded, operations, base_revision_id=BASE)
    assert resolve_entity_id(merged, "entity-2") == "entity-1"
    assert len(merged["entities"][0]["representations"]) == 2


def test_export_after_merge_and_transform_contains_only_the_active_model(tmp_path):
    from ehs_spatial.platform.blender_export import prepare_export, write_glb
    merged, _ = merge(migrate_document(source_scene(), base_revision_id=BASE))
    entity = merged['entities'][0]
    entity.pop('material',None)
    for rep in entity['representations']:
        rep.pop('material',None)
        rep['primitive']={'type':'box','dimensions':[1,2,3]}
    source_pose=deepcopy(entity['representations'][1]['transform'])
    pose={**entity['currentModelTransform'],'position':[4.,5.,6.]}
    updated,_=apply_operations(merged,[{'type':'setTransform','entityId':entity['id'],'transform':pose}],base_revision_id=BASE)
    prepared=prepare_export(BASE,updated,lambda _:pytest.fail('Primitive export must not request an asset'))
    assert len(prepared['objects'])==1
    assert prepared['objects'][0]['id']==entity['activeModelRepresentationId']
    assert prepared['objects'][0]['transform']==pose
    assert prepared['document']['entities'][0]['representations'][1]['transform']==source_pose
    assert prepared['excludedRepresentations']==[{'entityId':entity['id'],'representationId':'model-2','reason':'source_model_candidate'}]
    result=write_glb(prepared,tmp_path/'scene.glb')
    assert (tmp_path/'scene.glb').stat().st_size>0


def test_split_retains_cross_group_measurement_as_source_without_copying_to_children():
    from ehs_spatial.platform.identity import snapshot_measurements
    doc, _ = merge(migrate_document(source_scene(), base_revision_id=BASE))
    parent = doc['entities'][0]
    parent['measurements']['span'] = {'value': 7, 'sourceRefs': [{'observationId': oid} for oid in parent['observationRefs']]}
    snapshot_measurements(parent, source_revision_id=BASE)
    record = next(r for r in parent['measurementEvidence'] if r['measurementKey'] == 'span')
    d = decision(doc, 'different')
    d.update(observationGroups=[['observation-1'], ['observation-2']], supersedesDecisionId=doc['identityDecisions'][-1]['id'])
    groups = [{'id': f'child-{i}', 'observationRefs': [f'observation-{i}'], 'representationIds': [f'model-{i}'],
               'measurementEvidenceIds': [r['id'] for r in parent['measurementEvidence'] if r['measurementKey'] != 'span' and r['sourceEntityId'] == f'entity-{i}']} for i in (1, 2)]
    split, _ = apply_operations(doc, [{'type': 'recordIdentityDecision', 'decision': d}, {'type': 'splitEntity', 'entityId': parent['id'], 'decisionId': d['id'], 'groups': groups, 'retainedMeasurementEvidenceIds': [record['id']]}], base_revision_id=BASE)
    assert all('span' not in e['measurements'] for e in split['entities'])
    assert split['entities'][0]['lineage'][0]['sourceFields']['retainedMeasurementEvidence'] == [record]
    assert 'sourceFields' not in split['entities'][1]['lineage'][0]


def test_evidence_fulfillment_requires_exact_source_observations_after_identity_change():
    from ehs_spatial.platform.policy_repository import _identity_evidence_binding
    source = migrate_document(source_scene(), base_revision_id=BASE)
    target, _ = merge(source)
    original = {'id': 'old-finding', 'entityId': 'entity-2', 'facts': []}
    finding = {'id': 'new-finding', 'entityId': 'entity-1', 'facts': [{'sourceRefs': [{'observationId': 'observation-2'}]}]}
    refs = [{'observationId': 'observation-2'}]
    binding = _identity_evidence_binding(source, target, original, finding, refs, source_revision_id=BASE, ancestry_ids={BASE})
    assert binding['observationIds'] == ['observation-2'] and binding['sourceEntityId'] == 'entity-2'
    for invalid in ([{'observationId': 'observation-1'}], [{'imageId': 'image-2'}], refs + [{'observationId': 'observation-1'}]):
        with pytest.raises(PlatformError, match='evidence_identity_scope_mismatch'):
            _identity_evidence_binding(source, target, original, finding, invalid, source_revision_id=BASE, ancestry_ids={BASE})
    with pytest.raises(PlatformError, match='evidence_identity_scope_mismatch'):
        _identity_evidence_binding(source, target, original, finding, refs, source_revision_id=BASE, ancestry_ids=set())
    broad = {**finding, 'facts': [{'sourceRefs': refs + [{'observationId': 'observation-1'}]}]}
    with pytest.raises(PlatformError, match='evidence_identity_scope_mismatch'):
        _identity_evidence_binding(source, target, original, broad, refs, source_revision_id=BASE, ancestry_ids={BASE})


def test_migration_binds_exact_frozen_geometry_and_ignores_unrelated_source_json():
    source = source_scene()
    source['assets'].extend([{'id': 'source-scene-json', 'kind': 'source_document'}, {'id': 'verified-geometry', 'kind': 'native_geometry_manifest'}])
    for camera in source['cameras']:
        camera['sourceRefs'] = [{'assetId': 'source-scene-json'}]
    source['geometryEvidence'] = {'coordinateFrameId': 'frame', 'manifestAssetId': 'verified-geometry', 'frames': [
        {'cameraId': camera['id'], 'assets': {'input': camera['imageId']}} for camera in source['cameras']]}
    doc = migrate_document(source, base_revision_id=BASE)
    assert {value['geometrySolutionId'] for value in doc['geometryBindings'].values()} == {'verified-geometry'}
    source.pop('geometryEvidence')
    assert {value['geometrySolutionId'] for value in migrate_document(source, base_revision_id=BASE)['geometryBindings'].values()} == {'frame'}


def test_unselected_historical_measurement_never_becomes_current_during_merge():
    doc = migrate_document(source_scene(), base_revision_id=BASE)
    old_evidence = deepcopy(doc['entities'][0]['measurementEvidence'])
    doc['entities'][0]['measurements']['height'] = None
    doc['entities'][0]['measurementSelections']['height'] = None
    merged, _ = merge(doc)
    kept = merged['entities'][0]
    assert kept['measurements']['height'] == doc['entities'][1]['measurements']['height']
    assert all(record in kept['measurementEvidence'] for record in old_evidence)


def test_explicit_null_measurement_keys_survive_merge_split_and_selection():
    source = source_scene()
    for i, entity in enumerate(source['entities'], 1):
        entity['measurements'][f'unknown-{i}'] = None
    doc = migrate_document(source, base_revision_id=BASE)
    merged, inverse = merge(doc)
    assert inverse[0]['document'] == doc
    item = merged['entities'][0]
    for key in ('unknown-1', 'unknown-2'):
        assert key in item['measurements'] and item['measurements'][key] is None
        assert key in item['measurementSelections'] and item['measurementSelections'][key] is None
    d = decision(merged, 'different', ids=[item['id']], supersedes=merged['identityDecisions'][-1]['id'])
    d['observationGroups'] = [[f'observation-{i}'] for i in (1, 2)]
    groups = [{'id': f'child-{i}', 'observationRefs': [f'observation-{i}'], 'representationIds': [f'model-{i}'],
               'measurementEvidenceIds': [r['id'] for r in item['measurementEvidence'] if r['sourceEntityId'] == f'entity-{i}']} for i in (1, 2)]
    split, _ = apply_operations(merged, [{'type': 'recordIdentityDecision', 'decision': d},
                                       {'type': 'splitEntity', 'entityId': item['id'], 'decisionId': d['id'], 'groups': groups}], base_revision_id=BASE)
    for current in (merged, split):
        selections = [{'type': 'selectMeasurementEvidence', 'entityId': e['id'], 'measurementKey': key, 'measurementEvidenceId': None}
                      for e in current['entities'] for key in ('unknown-1', 'unknown-2')]
        selected, _ = apply_operations(current, selections, base_revision_id=BASE)
        assert selected == current
        assert selected['observations'] == doc['observations'] and selected['assets'] == doc['assets']
        assert {r['id']: r for e in selected['entities'] for r in e['measurementEvidence']} == {r['id']: r for e in doc['entities'] for r in e['measurementEvidence']}
