"""Versioned object identity decisions, preserving source evidence and poses.

Callers mutate a private document copy and commit it through the existing CAS
writer. This module has no database, blob, provider, or publication side effects.
"""
from __future__ import annotations

from copy import deepcopy
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from .contracts import GeometryBinding, IdentityDecision, MeasurementEvidence, PartRelation, PlatformError, SetPartRelationOperation, SourceIdentityEvidence, SourceObservationEquivalence, canonical, digest


MODEL_KINDS = frozenset(("generated_mesh", "primitive"))


def _require(condition, code, **params):
    if not condition:
        raise PlatformError(code, 422, **params)


def _typed(model, value, code):
    try:
        return model.model_validate(value).model_dump(mode="json", exclude_unset=True)
    except (ValidationError, TypeError, ValueError):
        raise PlatformError(code, 422) from None


def _unique(values):
    return list(dict.fromkeys(values))


def model_family(document, entity_id):
    """The target and its explicit descendants, in stable parent-first order."""
    entities = {e['id']: e for e in document['entities']}
    _require(entity_id in entities, 'entity_not_found', entityId=entity_id)
    children = {}
    for item in entities.values():
        children.setdefault(item.get('parentEntityId'), []).append(item)
    result, pending, seen = [], [entities[entity_id]], set()
    while pending:
        item = pending.pop()
        _require(item['id'] not in seen, 'part_relation_cycle')
        seen.add(item['id'])
        result.append(item)
        pending.extend(reversed(children.get(item['id'], [])))
    return result


def validate_part_relations(document):
    entities = {e['id']: e for e in document['entities']}
    observations = {o['id']: o for o in document['observations']}
    for item in entities.values():
        parent_id = item.get('parentEntityId')
        relation = item.get('partRelation')
        if parent_id is None and relation is None:
            continue
        _require(document['schemaVersion'] == 2, 'parts_require_scene_v2')
        relation = _typed(PartRelation, relation, 'part_relation_evidence_required')
        _require(bool(relation['reason'].strip()), 'part_relation_evidence_required')
        refs = relation['evidenceRefs']
        _require(len({r['observationId'] for r in refs}) == len(refs), 'part_relation_evidence_invalid')
        for ref in refs:
            _require(ref['observationId'] in observations and ref['revision'] <= observations[ref['observationId']].get('revision', 1), 'part_observation_revision_mismatch')
        if parent_id is None:
            continue
        _require(parent_id in entities, 'part_parent_not_found')
        _require(parent_id != item['id'], 'part_relation_cycle')
        parent = entities[parent_id]
        evidence_ids = {ref['observationId'] for ref in refs}
        _require(bool(evidence_ids.intersection(item.get('observationRefs', []))) and bool(evidence_ids.intersection(parent.get('observationRefs', []))), 'part_relation_evidence_scope')
        own_pose, parent_pose = item.get('currentModelTransform'), parent.get('currentModelTransform')
        _require(own_pose is not None and parent_pose is not None and own_pose['coordinateFrameId'] == parent_pose['coordinateFrameId'], 'part_coordinate_frame_mismatch')
        ancestors = {item['id']}
        while parent_id is not None:
            _require(parent_id not in ancestors, 'part_relation_cycle')
            _require(parent_id in entities, 'part_parent_not_found')
            ancestors.add(parent_id)
            parent_id = entities[parent_id].get('parentEntityId')


def set_part_relation(document, operation, *, base_revision_id):
    _require(document.get('schemaVersion') == 2, 'parts_require_scene_v2')
    op = _typed(SetPartRelationOperation, operation, 'part_relation_evidence_required')
    _require(isinstance(base_revision_id, str) and bool(base_revision_id), 'part_base_revision_required')
    entities = {e['id']: e for e in document['entities']}
    _require(op['entityId'] in entities, 'entity_not_found')
    item = entities[op['entityId']]
    old_parent_id = item.get('parentEntityId')
    parent_id = op['parentEntityId']
    _require(parent_id is None or parent_id in entities, 'part_parent_not_found')
    observations = {o['id']: o for o in document['observations']}
    for ref in op['evidenceRefs']:
        _require(observations.get(ref['observationId'], {}).get('revision', 1) == ref['revision'] and ref['observationId'] in observations, 'part_observation_revision_mismatch')
    evidence_ids = {ref['observationId'] for ref in op['evidenceRefs']}
    _require(bool(evidence_ids.intersection(item.get('observationRefs', []))), 'part_relation_evidence_scope')
    item.setdefault('lineage', []).append({'operation': 'setPartRelation', 'sourceRevisionId': base_revision_id,
        'parentEntityId': item.get('parentEntityId'), 'partRelation': deepcopy(item.get('partRelation'))})
    item.update(parentEntityId=parent_id, partRelation={'source': 'manual', 'baseRevisionId': base_revision_id,
        'evidenceRefs': deepcopy(op['evidenceRefs']), 'reason': op['reason']})
    old_parent = entities.get(old_parent_id)
    if old_parent is not None and old_parent.get('activeModelRepresentationId') is None and not any(e.get('parentEntityId') == old_parent_id for e in entities.values()):
        old_parent['currentModelTransform'] = None
    validate_part_relations(document)


def refresh_cad_reference(document, entity, *, reference=None):
    """Retain a declared exposure while refreshing its exact current owners."""
    observations = {o['id']: o for o in document['observations']}
    cameras = {c['id']: c for c in document['cameras']}
    own = [observations[oid] for oid in entity['observationRefs']]
    bound = {image_id for image_id, binding in document.get('geometryBindings', {}).items()
             if binding and cameras.get(binding['cameraId'], {}).get('imageId') == image_id}
    available = {o['imageId'] for o in own if o['imageId'] in bound}
    previous = reference if reference is not None else entity.get('cadReference') or {}
    selected = previous.get('referenceImageId') if previous.get('status') == 'resolved' and previous.get('referenceImageId') in available else None
    source, evidence = (previous.get('source'), deepcopy(previous.get('evidenceRefs', []))) if selected else (None, [])
    if selected is None and len(available) == 1:
        selected, source = next(iter(available)), 'single_source_image'
    if selected is None:
        source = 'no_observations' if not own else 'unbound_geometry' if not available else 'ambiguous_sources'
    entity['cadReference'] = {'referenceImageId': selected, 'status': 'resolved' if selected else 'unresolved',
        'source': source, 'sourceRefs': [{'observationId': o['id'], 'revision': o['revision']} for o in own if o['imageId'] == selected],
        'evidenceRefs': evidence}


def _observation_refs(value):
    """Only named observation references are identity evidence; labels are not."""
    if isinstance(value, dict):
        found = [value["observationId"]] if isinstance(value.get("observationId"), str) else []
        return _unique(found + [ref for v in value.values() for ref in _observation_refs(v)])
    if isinstance(value, list):
        return _unique([ref for v in value for ref in _observation_refs(v)])
    return []


def observation_owners(document):
    owners = {}
    for entity in document["entities"]:
        for oid in entity.get("observationRefs", []):
            _require(oid not in owners, "observation_multiple_owners", observationId=oid)
            owners[oid] = entity["id"]
    return owners


def effective_decisions(document):
    decisions = document.get("identityDecisions", [])
    superseded = {d["supersedesDecisionId"] for d in decisions if d.get("supersedesDecisionId")}
    return [d for d in decisions if d["id"] not in superseded]


def _source_refs(value):
    if isinstance(value, dict):
        own = [value] if value.get('observationId') or (value.get('assetId') and value.get('sourceRecordId')) else []
        return own + [ref for child in value.values() for ref in _source_refs(child)]
    return [ref for child in value for ref in _source_refs(child)] if isinstance(value, list) else []


def source_observation_ids(document, entity, value, *, coordinate_frame_id=None):
    """Resolve source records within their owner; names/overlap supply no identity."""
    refs = _source_refs(value)
    assets = {a['id']: a for a in document['assets']}
    cameras = {c['id']: c for c in document['cameras']}
    found = []
    for observation in document['observations']:
        if observation['id'] not in entity.get('observationRefs', []):
            continue
        image_id = observation['imageId']
        if document['schemaVersion'] == 2:
            binding = (document.get('geometryBindings') or {}).get(image_id)
            camera = cameras.get((binding or {}).get('cameraId'))
        else:
            choices = [c for c in cameras.values() if c['imageId'] == image_id]
            camera = choices[0] if len(choices) == 1 else None
        frame = (observation.get('geometrySupport') or {}).get('coordinateFrameId') or (camera or {}).get('coordinateFrameId')
        if coordinate_frame_id is not None and frame != coordinate_frame_id:
            continue
        def scope(ref):
            return ((not ref.get('assetId') or ref['assetId'] in assets)
                    and (not ref.get('sha256') or ref['sha256'] == assets.get(ref.get('assetId'), {}).get('sha256'))
                    and (not ref.get('imageId') or ref['imageId'] == image_id)
                    and (not ref.get('imageSha256') or ref['imageSha256'] == assets.get(image_id, {}).get('sha256'))
                    and (not ref.get('cameraId') or ref['cameraId'] == (camera or {}).get('id')))
        observation_refs = _source_refs(observation.get('sourceRefs', []))
        for ref in refs:
            if not scope(ref):
                continue
            if ref.get('observationId'):
                matched = ref['observationId'] == observation['id']
            else:
                matched = any(scope(other) and ref['assetId'] == other.get('assetId') and ref['sourceRecordId'] == other.get('sourceRecordId') and
                              (not ref.get('sourceFrameId') or ref['sourceFrameId'] == other.get('sourceFrameId')) for other in observation_refs)
            if matched:
                found.append(observation['id'])
                break
    return sorted(found)


def _measurement_source(entity, key, value):
    measurements = entity.get('measurements') or {}
    refs = _source_refs(value)
    if not refs and key in {'dimensionsNative', 'coordinateFrameId', 'dimensionBasis', 'groundHeightNative'}:
        refs = _source_refs(measurements.get('observedBounds'))
    if not refs:
        refs = _source_refs(measurements.get('sourceRefs'))
    frame = value.get('coordinateFrameId') if isinstance(value, dict) else None
    return refs, frame or measurements.get('coordinateFrameId')


def snapshot_measurements(entity, *, source_revision_id, document=None):
    """Record newly produced current values without rewriting earlier evidence."""
    _require(isinstance(source_revision_id, str) and bool(source_revision_id), "identity_base_revision_required")
    records = entity.setdefault("measurementEvidence", [])
    selections = entity.setdefault("measurementSelections", {})
    lookup = {r["id"]: r for r in records}
    for key, value in (entity.get("measurements") or {}).items():
        selected = lookup.get(selections.get(key))
        if selected and selected["originalMeasurement"] == value:
            continue
        if value is None:
            selections[key] = None
            continue
        identity = str(uuid5(NAMESPACE_URL, "measurement:" + digest([source_revision_id, entity["id"], key, value])))
        if identity not in lookup:
            refs, frame = _measurement_source(entity, key, value)
            source_refs = source_observation_ids(document, entity, refs, coordinate_frame_id=frame) if document is not None else _observation_refs(refs)
            record = {"id": identity, "measurementKey": key, "originalMeasurement": deepcopy(value),
                      "sourceRevisionId": source_revision_id, "sourceEntityId": entity["id"],
                      "observationRefs": source_refs,
                      "representationId": value.get("representationId") if isinstance(value, dict) else None}
            if refs:
                record['sourceRefs'] = deepcopy(refs)
            if frame:
                record['coordinateFrameId'] = frame
            records.append(record)
            lookup[identity] = record
        selections[key] = identity
    for key in set(selections) - set(entity.get("measurements") or {}):
        # A producer can stop selecting an old quantity; its source record remains.
        del selections[key]


def repair_measurement_sources(document, *, base_revision_id):
    """Append source bindings for selected legacy values; never select unknowns."""
    _require(document.get('schemaVersion') == 2, 'identity_requires_scene_v2')
    _require(isinstance(base_revision_id, str) and bool(base_revision_id), 'identity_base_revision_required')
    added = []
    for entity in document['entities']:
        by_id = {r['id']: r for r in entity['measurementEvidence']}
        selections = dict(entity['measurementSelections'])
        for key, selected in selections.items():
            record = by_id.get(selected)
            if not record or record['observationRefs'] or record['originalMeasurement'] is None:
                continue
            refs, frame = _measurement_source(entity, key, record['originalMeasurement'])
            if record.get('sourceRefs'):
                refs = record['sourceRefs']
            elif not _source_refs(record['originalMeasurement']):
                group = by_id.get(selections.get('sourceRefs'))
                if not group or (group['sourceEntityId'], group['sourceRevisionId']) != (record['sourceEntityId'], record['sourceRevisionId']):
                    continue
            frame = record.get('coordinateFrameId') or frame
            observation_ids = source_observation_ids(document, entity, refs, coordinate_frame_id=frame)
            if not observation_ids:
                continue
            identity = str(uuid5(NAMESPACE_URL, 'measurement-source:' + digest([base_revision_id, record['id'], refs, frame, observation_ids])))
            bound = {**deepcopy(record), 'id': identity, 'sourceEvidenceId': record['id'], 'sourceRevisionId': base_revision_id,
                     'observationRefs': observation_ids, 'sourceRefs': deepcopy(refs), 'coordinateFrameId': frame}
            entity['measurementEvidence'].append(bound)
            entity['measurementSelections'][key] = identity
            added.append(identity)
    return added


def _select_measurements(entity, previous=None, *, eligible_ids=None):
    previous = previous or {}
    by_key = {key: [] for key, selected in previous.items() if selected is None}
    for record in entity["measurementEvidence"]:
        by_key.setdefault(record["measurementKey"], []).append(record)
    values, selections = {}, {}
    for key, records in by_key.items():
        records = [record for record in records if eligible_ids is None or record["id"] in eligible_ids]
        unique_values = {canonical(r["originalMeasurement"]) for r in records}
        chosen = None
        if len(unique_values) == 1:
            chosen = next((r for r in records if r["id"] == previous.get(key)), min(records, key=lambda r: r["id"]))
        values[key] = deepcopy(chosen["originalMeasurement"]) if chosen else None
        selections[key] = chosen["id"] if chosen else None
    entity["measurements"], entity["measurementSelections"] = values, selections


def _active(entity, requested=...):
    models = {r["id"]: r for r in entity.get("representations", []) if r["kind"] in MODEL_KINDS}
    if requested is ...:
        requested = entity.get("activeModelRepresentationId")
        if requested is None and len(models) == 1:
            requested = next(iter(models))
    _require(requested is None or requested in models, "active_model_representation_invalid")
    entity["activeModelRepresentationId"] = requested
    entity["currentModelTransform"] = deepcopy(models[requested]["transform"]) if requested else None


def migrate_document(source, *, base_revision_id, active_model_selections=None):
    """Explicit v1 -> v2 copy. The caller creates a new immutable revision."""
    from .contracts import validate_document
    validate_document(source)
    _require(source["schemaVersion"] == 1, "scene_migration_requires_v1")
    _require(isinstance(base_revision_id, str) and bool(base_revision_id), "identity_base_revision_required")
    document = deepcopy(source)
    capture_id = document.get("captureId")
    _require(capture_id or not document["observations"], "observation_capture_required")
    document.update(schemaVersion=2, captureIds=[capture_id] if capture_id else [], identityDecisions=[], geometryBindings={})
    for observation in document["observations"]:
        observation["captureId"] = capture_id
    selections = active_model_selections or {}
    _require(set(selections) <= {e["id"] for e in document["entities"]}, "active_model_entity_not_found")
    for entity in document["entities"]:
        original_transform = deepcopy(entity.get("currentModelTransform"))
        entity.update(measurementEvidence=[], measurementSelections={})
        models = [r for r in entity.get("representations", []) if r["kind"] in MODEL_KINDS]
        matches = [r for r in models if r["transform"] == original_transform]
        choice = selections.get(entity["id"], matches[0]["id"] if len(matches) == 1 else models[0]["id"] if len(models) == 1 else None)
        _active(entity, choice)
        if choice is not None:
            active = next(r for r in models if r["id"] == choice)
            if entity.get("material") is not None and "material" not in active:
                active["material"] = deepcopy(entity["material"])
            if original_transform is not None and (len(models) == 1 or choice in {r["id"] for r in matches}):
                entity["currentModelTransform"] = original_transform
        entity.setdefault("lineage", []).append({"operation": "migrate_scene", "sourceRevisionId": base_revision_id,
                                                 "entityId": entity["id"], "sourceModelTransform": original_transform})
        evidence = entity.get("associationEvidence") or {}
        if evidence.get("method") == "explicit_source_id" and len(entity.get("observationRefs", [])) > 1:
            observations = {o["id"]: o for o in document["observations"]}
            document["identityDecisions"].append({"id": str(uuid5(NAMESPACE_URL, "source-binding:" + digest([base_revision_id, entity["id"]]))),
                "decision": "same", "source": "source_binding", "baseRevisionId": base_revision_id, "entityIds": [entity["id"]],
                "observationGroups": [[oid] for oid in entity["observationRefs"]], "survivorId": entity["id"],
                "evidenceRefs": [{"kind": "observation", "observationId": oid, "observationRevision": observations[oid].get("revision", 1)} for oid in entity["observationRefs"]],
                "reason": "Migrated explicit source identity; not a new geometric verification", "supersedesDecisionId": None})
    references = (document.get("reportEvidence") or {}).get("frames", [])
    for image_id in _unique([a["id"] for a in document["assets"] if a.get("kind") == "source_image"] + [c["imageId"] for c in document["cameras"]]):
        cameras = [c for c in document["cameras"] if c["imageId"] == image_id]
        pinned = {r.get("cameraId") for r in references if r.get("imageId") == image_id}
        choices = cameras if len(cameras) == 1 else [c for c in cameras if c["id"] in pinned]
        camera = choices[0] if len(choices) == 1 else None
        frozen = document.get("geometryEvidence") or {}
        frozen_frames = [saved for saved in frozen.get("frames", []) if saved.get("cameraId") == (camera or {}).get("id") and (saved.get("assets") or {}).get("input") == image_id]
        assets = {a["id"]:a for a in document["assets"]}
        verified_frozen = camera is not None and len(frozen_frames) == 1 and frozen.get("coordinateFrameId") == camera["coordinateFrameId"] and frozen.get("manifestAssetId") in assets
        refs = _unique([r["assetId"] for r in (camera or {}).get("sourceRefs", []) if isinstance(r, dict) and r.get("assetId") in assets and ((assets[r["assetId"]].get("metadata") or {}).get("stage") in {"geometry", "registered_geometry"} or assets[r["assetId"]].get("kind") in {"geometry", "registered_geometry"})])
        solution_id = frozen["manifestAssetId"] if verified_frozen else refs[0] if len(refs) == 1 else (camera or {}).get("coordinateFrameId")
        document["geometryBindings"][image_id] = {"geometrySolutionId": solution_id, "cameraId": camera["id"]} if camera and (verified_frozen or len(refs) <= 1) else None
    for entity in document['entities']:
        snapshot_measurements(entity, source_revision_id=base_revision_id, document=document)
    validate_document(document)
    return document


def resolve_entity_id(document, entity_id, *, observation_id=None, _visited=None):
    active = {e["id"]: e for e in document["entities"]}
    owners = observation_owners(document)
    if entity_id in active:
        _require(observation_id is None or owners.get(observation_id) == entity_id, "identity_observation_scope_invalid")
        return entity_id
    refs = set()
    for decision in document.get("identityDecisions", []):
        if entity_id not in decision["entityIds"]:
            continue
        groups = decision["observationGroups"]
        if len(decision["entityIds"]) == len(groups):
            refs.update(groups[decision["entityIds"].index(entity_id)])
        else:
            refs.update(oid for group in groups for oid in group)
    if observation_id is not None:
        _require(observation_id in refs, "identity_observation_scope_invalid")
        refs = {observation_id}
    targets = sorted({owners[oid] for oid in refs if oid in owners})
    if not targets and observation_id is None:
        visited = set(_visited or ()) | {entity_id}
        survivors = {d["survivorId"] for d in document.get("identityDecisions", []) if d["decision"] == "same" and entity_id in d["entityIds"] and d["survivorId"] not in visited}
        targets = sorted({resolve_entity_id(document, survivor, _visited=visited) for survivor in survivors})
    _require(bool(targets), "entity_not_found", entityId=entity_id)
    _require(len(targets) == 1, "identity_resolution_ambiguous", entityId=entity_id, entityIds=targets)
    return targets[0]


def _targets(document, entity_id, refs=()):
    owners = observation_owners(document)
    if refs:
        return sorted({owners[oid] for oid in refs if oid in owners})
    try:
        return [resolve_entity_id(document, entity_id)]
    except PlatformError as exc:
        if exc.code == "identity_resolution_ambiguous":
            return exc.params["entityIds"]
        if exc.code == "entity_not_found":
            return []
        raise


def remap_attachments(document, *, base_revision_id, affected_entity_ids):
    """Map named current attachments. Never rewrite historical finding bodies."""
    affected = set(affected_entity_ids)
    owners = observation_owners(document)
    def bind(row, source_ids, refs=(), *, manual=False):
        targets = sorted({target for eid in source_ids for target in _targets(document, eid, refs)})
        unique = len(targets) == 1 and not (manual and not refs)
        row["identityBinding"] = {"sourceRevisionId": base_revision_id, "sourceEntityIds": source_ids,
                                  "entityIds": targets, "status": "resolved" if unique else "requires_review"}
        return targets if unique else []
    for annotation in document["annotations"]:
        source_id = annotation.get("entityId")
        if source_id in affected:
            targets = bind(annotation, [source_id], _observation_refs(annotation.get("sourceRefs", [])), manual=annotation.get("kind") == "manual_evidence")
            annotation["entityId"] = targets[0] if targets else None
    report = document.get("reportEvidence") or {}
    for record in report.get("objects", []):
        old = record.get("entityId")
        views = record.get("views", [])
        view_owners = []
        for view in views:
            if view.get("observationId"):
                view["entityId"] = owners.get(view["observationId"])
                view_owners.append(view["entityId"])
        if view_owners:
            record["entityId"] = view_owners[0] if view_owners[0] and all(x == view_owners[0] for x in view_owners) else None
            if old in affected:
                bind(record, [old], [v["observationId"] for v in views if v.get("observationId")])
        elif old in affected:
            targets = bind(record, [old])
            record["entityId"] = targets[0] if targets else None
    historical = report.get("historical") or {}
    for record in historical.get("inventory", []) + (historical.get("cad") or {}).get("regions", []):
        old = record.get("entityIds", [])
        if affected.intersection(old):
            # A source region without observation scope cannot be assigned to all split children.
            targets = []
            for eid in old:
                resolved = _targets(document, eid)
                if len(resolved) == 1:
                    targets.extend(resolved)
            bind(record, old)
            record["entityIds"] = _unique(targets)


def apply_source_equivalences(document, verified_pairs, *, base_revision_id):
    """Consume producer-verified immutable source/mask pairs before geometry.

    The producer must verify JSON pointers, source RLE and actual canonical mask
    equality. This private boundary additionally checks current revisions and
    asset identities. It is deliberately unavailable as a client edit operation.
    """
    from .contracts import validate_document
    _require(document.get('schemaVersion') == 2, 'identity_requires_scene_v2')
    _require(isinstance(base_revision_id, str) and bool(base_revision_id), 'identity_base_revision_required')
    observations = {o['id']: o for o in document['observations']}
    assets = {a['id']: a for a in document['assets']}
    owners = observation_owners(document)
    proof_assets = {(r['assetId'], r['sha256']) for r in document.get('sourceIdentityEvidence', [])}
    pairs, graph = [], {}
    for value in verified_pairs:
        pair = _typed(SourceObservationEquivalence, value, 'invalid_source_observation_equivalence')
        pair['observationRefs'].sort(key=lambda ref:ref['observationId'])
        pair['evidenceRefs'].sort(key=canonical)
        refs = pair['observationRefs']
        _require(refs[0]['observationId'] != refs[1]['observationId'], 'source_equivalence_requires_distinct_observations')
        image = assets.get(pair['imageId'])
        _require(image is not None and image.get('sha256') == pair['imageSha256'], 'source_equivalence_image_mismatch')
        for ref in refs:
            observation = observations.get(ref['observationId'])
            _require(observation is not None and ref['observationId'] in owners, 'identity_observation_scope_invalid')
            _require(observation.get('revision', 1) == ref['revision'], 'identity_observation_revision_mismatch')
            _require(observation['imageId'] == pair['imageId'], 'source_equivalence_image_mismatch')
        for ref in [pair['sourceRef'], *pair['evidenceRefs']]:
            _require(assets.get(ref['assetId'], {}).get('sha256') == ref['sha256'], 'identity_asset_evidence_mismatch')
        _require(any((r['assetId'], r['sha256']) in proof_assets for r in pair['evidenceRefs']), 'source_equivalence_proof_required')
        first, second = [owners[r['observationId']] for r in refs]
        if first != second:
            graph.setdefault(first, set()).add(second)
            graph.setdefault(second, set()).add(first)
            pairs.append(pair)
    # Work on a private copy so a conflict in any connected group is atomic.
    working, results = deepcopy(document), []
    remaining = set(graph)
    while remaining:
        component, pending = set(), {min(remaining)}
        while pending:
            entity_id = pending.pop()
            if entity_id not in component:
                component.add(entity_id)
                pending.update(graph[entity_id] - component)
        remaining -= component
        entities = {e['id']: e for e in working['entities']}
        ids = sorted(component)
        group_pairs = sorted([p for p in pairs if owners[p['observationRefs'][0]['observationId']] in component], key=canonical)
        source_groups = [d for d in effective_decisions(document) if d['source'] == 'source_binding' and d['decision'] == 'same']
        def source_group_size(entity_id):
            return max([len({observations[oid]['imageId'] for group in d['observationGroups'] for oid in group})
                        for d in source_groups if {owners.get(oid) for group in d['observationGroups'] for oid in group} == {entity_id}] or [0])
        survivor = min(ids, key=lambda entity_id:(-source_group_size(entity_id), entity_id))
        groups = [sorted(entities[eid]['observationRefs']) for eid in ids]
        evidence = [{'kind':'observation', 'observationId':oid, 'observationRevision':observations[oid].get('revision', 1)} for group in groups for oid in group]
        asset_refs = {(r['assetId'], r['sha256']) for p in group_pairs for r in [p['sourceRef'], *p['evidenceRefs']]}
        evidence += [{'kind':'asset', 'assetId':asset_id, 'sha256':sha} for asset_id, sha in sorted(asset_refs)]
        evidence.append({'kind':'method', 'name':'same_source_observation_equivalence', 'version':'1', 'configSha256':digest(group_pairs)})
        decision_id = str(uuid5(NAMESPACE_URL, 'source-equivalence:' + digest([base_revision_id, groups, evidence])))
        decision = {'id':decision_id, 'decision':'same', 'source':'source_binding', 'baseRevisionId':base_revision_id,
                    'entityIds':ids, 'observationGroups':groups, 'survivorId':survivor, 'evidenceRefs':evidence,
                    'reason':'Same immutable source artifact and RLE instance, mapped photo, and verified equal canonical masks', 'supersedesDecisionId':None}
        _record_decision(working, decision, base_revision_id, source_binding=True)
        operation = {'type':'mergeEntities', 'entityIds':ids, 'survivorId':survivor, 'decisionId':decision_id,
                     'activeModelRepresentationId':entities[survivor]['activeModelRepresentationId']}
        apply_identity_operation(working, operation, base_revision_id=base_revision_id)
        results.append(operation)
    if results:
        validate_document(working)
        document.clear()
        document.update(working)
    return results


def _record_decision(document, value, base_revision_id, *, source_binding=False):
    decision = _typed(IdentityDecision, value, "invalid_identity_decision")
    _require(base_revision_id and decision["baseRevisionId"] == base_revision_id, "identity_base_revision_mismatch")
    _require(source_binding or decision["source"] != "source_binding", "identity_source_binding_migration_only")
    prior = next((d for d in document["identityDecisions"] if d["id"] == decision["id"]), None)
    _require(prior is None, "identity_decision_already_exists")
    entities = {e["id"]: e for e in document["entities"]}
    ids, groups = decision["entityIds"], decision["observationGroups"]
    _require(len(ids) == len(set(ids)) and set(ids) <= set(entities), "identity_entity_scope_invalid")
    refs = [oid for group in groups for oid in group]
    _require(len(refs) == len(set(refs)) and (decision["source"] == "manual" or all(groups)), "identity_observation_groups_invalid")
    expected = [entities[eid].get("observationRefs", []) for eid in ids]
    split = decision["decision"] == "different" and len(ids) == 1
    _require(not split or all(groups), "identity_observation_groups_invalid")
    _require(set(refs) == {oid for group in expected for oid in group} if split else len(groups) == len(expected) and all(set(a) == set(b) for a, b in zip(groups, expected)), "identity_observation_groups_invalid")
    _require((decision["survivorId"] in ids) if decision["decision"] == "same" else decision["survivorId"] is None, "identity_survivor_invalid")
    observations = {o["id"]: o for o in document["observations"]}
    assets = {a["id"]: a for a in document["assets"]}
    for ref in decision["evidenceRefs"]:
        if ref["kind"] == "observation":
            _require(ref["observationId"] in refs, "identity_observation_scope_invalid")
            _require(observations[ref["observationId"]].get("revision", 1) == ref["observationRevision"], "identity_observation_revision_mismatch")
        elif ref["kind"] == "asset":
            asset = assets.get(ref["assetId"])
            _require(asset is not None and asset.get("sha256") == ref["sha256"], "identity_asset_evidence_mismatch")
    _require({r["observationId"] for r in decision["evidenceRefs"] if r["kind"] == "observation"} == set(refs), "identity_observation_evidence_required")
    supersedes = decision.get("supersedesDecisionId")
    if supersedes:
        previous = next((d for d in effective_decisions(document) if d["id"] == supersedes), None)
        previous_targets = {target for eid in (previous or {}).get("entityIds", []) for target in _targets(document, eid)}
        _require(previous is not None and previous_targets and previous_targets <= set(ids), "identity_supersedes_invalid")
        _require(decision["source"] == "manual" or previous["source"] == "geometry", "identity_manual_decision_required")
    proposed = set(refs)
    for previous in effective_decisions(document):
        if previous["id"] == supersedes or decision["decision"] == "undecided":
            continue
        conflicts = previous["decision"] == "different" and decision["decision"] == "same" and sum(bool(proposed.intersection(g)) for g in previous["observationGroups"]) > 1
        if previous["decision"] == "different" and decision["decision"] == "same" and len(previous["entityIds"]) == len(previous["observationGroups"]):
            previous_targets = [_targets(document, eid, group) for eid, group in zip(previous["entityIds"], previous["observationGroups"])]
            conflicts = conflicts or sum(bool(set(ids).intersection(targets)) for targets in previous_targets) > 1
        if decision["decision"] == "different" and previous["decision"] == "same" and previous["source"] == "manual":
            old_refs = {oid for g in previous["observationGroups"] for oid in g}
            conflicts = sum(bool(old_refs.intersection(g)) for g in groups) > 1
        _require(not conflicts, "identity_decision_conflict", decisionId=previous["id"])
    document["identityDecisions"].append(decision)


def apply_identity_operation(document, operation, *, base_revision_id):
    _require(document.get("schemaVersion") == 2, "identity_requires_scene_v2")
    kind = operation.get("type")
    if kind == "recordIdentityDecision":
        _record_decision(document, operation.get("decision"), base_revision_id)
        return
    identity = operation.get("entityId")
    entity = next((e for e in document["entities"] if e["id"] == identity), None)
    if kind in ("setActiveModelRepresentation", "selectMeasurementEvidence"):
        if entity is None:
            resolved = resolve_entity_id(document, identity)
            entity = next(e for e in document["entities"] if e["id"] == resolved)
        _require(entity is not None, "entity_not_found")
        if kind == "setActiveModelRepresentation":
            _require("representationId" in operation, "active_model_representation_required")
            assembly_pose = entity.get('currentModelTransform') if any(e.get('parentEntityId') == entity['id'] for e in document['entities']) else None
            _active(entity, operation["representationId"])
            if operation['representationId'] is None and assembly_pose is not None:
                entity['currentModelTransform'] = assembly_pose
        else:
            key, evidence_id = operation.get("measurementKey"), operation.get("measurementEvidenceId")
            evidence = next((r for r in entity["measurementEvidence"] if r["id"] == evidence_id and r["measurementKey"] == key), None)
            _require(key in entity["measurementSelections"] and (evidence is not None or evidence_id is None), "measurement_selection_invalid")
            entity["measurements"][key] = deepcopy(evidence["originalMeasurement"]) if evidence else None
            entity["measurementSelections"][key] = evidence_id
        return
    decision = next((d for d in effective_decisions(document) if d["id"] == operation.get("decisionId")), None)
    _require(decision is not None and decision["baseRevisionId"] == base_revision_id, "identity_operation_decision_mismatch")
    if kind == "mergeEntities":
        ids, survivor = operation.get("entityIds", []), operation.get("survivorId")
        _require(decision["decision"] == "same" and decision["entityIds"] == ids and decision["survivorId"] == survivor, "identity_operation_decision_mismatch")
        entities = {e["id"]: e for e in document["entities"]}
        _require(len(ids) >= 2 and len(ids) == len(set(ids)) and set(ids) <= set(entities), "invalid_merge")
        _require(len({entities[eid].get('parentEntityId') for eid in ids}) == 1, 'identity_merge_part_parent_conflict')
        _require(not any(e['id'] in ids for eid in ids for e in model_family(document, eid)[1:]), 'identity_merge_part_cycle')
        kept = entities[survivor]
        old_selection = {key: None for eid in ids for key in entities[eid]["measurementSelections"]} | kept["measurementSelections"]
        eligible_measurements = {selected for eid in ids for selected in entities[eid]["measurementSelections"].values() if selected is not None}
        source_fields = [{"operation": "merge", "entityId": eid, "sourceRevisionId": base_revision_id, "decisionId": decision["id"],
                          "sourceFields": {k: deepcopy(v) for k, v in entities[eid].items() if k not in {"observationRefs", "representations", "measurementEvidence", "measurementSelections", "measurements", "lineage"}}} for eid in ids]
        refs_by_value = {canonical(ref): deepcopy(ref) for eid in ids for ref in (entities[eid].get("sourceRefs") or [])}
        for eid in ids:
            item = entities[eid]
            if eid == survivor:
                continue
            kept["observationRefs"] = _unique(kept["observationRefs"] + item["observationRefs"])
            kept["representations"].extend(deepcopy(item["representations"]))
            kept["measurementEvidence"].extend(deepcopy(item["measurementEvidence"]))
            kept.setdefault("lineage", []).extend(deepcopy(item.get("lineage", [])))
        kept.setdefault("lineage", []).extend(source_fields)
        if refs_by_value:
            kept["sourceRefs"] = list(refs_by_value.values())
        document["entities"] = [e for e in document["entities"] if e["id"] not in set(ids) - {survivor}]
        for child in document['entities']:
            if child.get('parentEntityId') in ids:
                child['parentEntityId'] = survivor
        kept["associationState"] = "confirmed"
        kept["associationEvidence"] = {"method": "identity_decision", "status": "confirmed", "decisionId": decision["id"], "observationIds": list(kept["observationRefs"]), "sourceRefs": deepcopy(decision["evidenceRefs"])}
        old_transform = deepcopy(kept.get("currentModelTransform"))
        old_active = kept.get("activeModelRepresentationId")
        _active(kept, operation.get("activeModelRepresentationId", ...))
        if old_transform is not None and old_active == kept["activeModelRepresentationId"]:
            kept["currentModelTransform"] = old_transform
        _select_measurements(kept, old_selection, eligible_ids=eligible_measurements)
        refresh_cad_reference(document, kept)
        remap_attachments(document, base_revision_id=base_revision_id, affected_entity_ids=ids)
        return
    _require(kind == "splitEntity" and entity is not None, "unknown_identity_operation")
    _require(not any(e.get('parentEntityId') == identity for e in document['entities']), 'identity_split_part_parent_requires_detach')
    _require(decision["decision"] == "different" and decision["source"] == "manual" and decision["entityIds"] == [identity], "identity_operation_decision_mismatch")
    groups = operation.get("groups", [])
    _require(isinstance(groups, list) and len(groups) >= 2 and all(isinstance(g, dict) and isinstance(g.get("id"), str) and g["id"] and
             all(isinstance(g.get(key), list) and all(isinstance(v, str) for v in g[key]) for key in ("observationRefs", "representationIds", "measurementEvidenceIds")) for g in groups), "invalid_split")
    child_ids = [g["id"] for g in groups]
    historic_ids = {eid for d in document["identityDecisions"] for eid in d["entityIds"]}
    _require(len(child_ids) == len(set(child_ids)) and not set(child_ids).intersection(historic_ids | {e["id"] for e in document["entities"]}), "identity_reused_entity_id")
    _require(sorted(sorted(g["observationRefs"]) for g in groups) == sorted(sorted(g) for g in decision["observationGroups"]), "identity_observation_groups_invalid")
    retained_ids = operation.get("retainedMeasurementEvidenceIds", [])
    _require(isinstance(retained_ids, list) and all(isinstance(value, str) for value in retained_ids), "identity_split_partition_invalid")
    for key, expected in (("observationRefs", entity["observationRefs"]), ("representationIds", [r["id"] for r in entity["representations"]]), ("measurementEvidenceIds", [r["id"] for r in entity["measurementEvidence"]])):
        values = [v for group in groups for v in group.get(key, [])] + (retained_ids if key == "measurementEvidenceIds" else [])
        _require(len(values) == len(set(values)) and set(values) == set(expected), "identity_split_partition_invalid", field=key)
    retained = [deepcopy(r) for r in entity["measurementEvidence"] if r["id"] in retained_ids]
    for record in retained:
        matches = [g for g in groups if (record["observationRefs"] or record["representationId"]) and set(record["observationRefs"]) <= set(g["observationRefs"]) and (record["representationId"] is None or record["representationId"] in g["representationIds"])]
        _require(len(matches) != 1, "measurement_split_scope_is_unique")
    document["entities"].remove(entity)
    for index, group in enumerate(groups):
        child = {"id": group["id"], "label": group.get("label", entity.get("label")), "observationRefs": deepcopy(group["observationRefs"]),
                 "associationState": "confirmed", "representations": [deepcopy(r) for r in entity["representations"] if r["id"] in group["representationIds"]],
                 "measurementEvidence": [deepcopy(r) for r in entity["measurementEvidence"] if r["id"] in group["measurementEvidenceIds"]],
                 "measurements": {}, "measurementSelections": {}, "groupId": entity.get("groupId"),
                 "lineage": [{"operation": "split", "entityId": identity, "sourceRevisionId": base_revision_id, "decisionId": decision["id"]}]}
        if entity.get('parentEntityId') is not None:
            child.update(parentEntityId=entity['parentEntityId'], partRelation=deepcopy(entity['partRelation']))
        if index == 0 and retained:
            child["lineage"][0]["sourceFields"] = {"retainedMeasurementEvidence": retained}
        for record in child["measurementEvidence"]:
            _require(set(record["observationRefs"]) <= set(child["observationRefs"]), "measurement_split_scope_invalid")
            _require(record["representationId"] is None or record["representationId"] in group["representationIds"], "measurement_split_scope_invalid")
        _active(child, group.get("activeModelRepresentationId", ...))
        _select_measurements(child, entity["measurementSelections"], eligible_ids=set(entity["measurementSelections"].values()) - {None})
        refresh_cad_reference(document, child, reference=entity.get('cadReference'))
        document["entities"].append(child)
    remap_attachments(document, base_revision_id=base_revision_id, affected_entity_ids=[identity])


def validate_identity_document(document):
    _require(isinstance(document.get("captureIds"), list) and all(isinstance(x, str) and x for x in document["captureIds"]) and len(document["captureIds"]) == len(set(document["captureIds"])), "invalid_capture_identities")
    owners = observation_owners(document)
    assets = {a['id']:a for a in document['assets']}
    source_evidence = document.get('sourceIdentityEvidence', [])
    _require(isinstance(source_evidence, list), 'invalid_source_identity_evidence')
    for value in source_evidence:
        ref = _typed(SourceIdentityEvidence, value, 'invalid_source_identity_evidence')
        _require(assets.get(ref['assetId'], {}).get('sha256') == ref['sha256'], 'identity_asset_evidence_mismatch')
    observations = {o["id"]: o for o in document["observations"]}
    for observation in observations.values():
        _require(observation.get("captureId") in document["captureIds"], "observation_capture_required")
    entities = {e["id"]: e for e in document["entities"]}
    representation_ids = {r["id"] for e in document["entities"] for r in e.get("representations", [])}
    evidence_ids = set()
    for entity in entities.values():
        _require("activeModelRepresentationId" in entity, "active_model_selection_required")
        model_ids = {r["id"] for r in entity.get("representations", []) if r["kind"] in MODEL_KINDS}
        _require(entity["activeModelRepresentationId"] is None or entity["activeModelRepresentationId"] in model_ids, "active_model_representation_invalid")
        partition_sources = {event.get('sourceRepresentationId') for event in entity.get('lineage', []) if isinstance(event, dict) and event.get('operation') == 'partition_model_parts'}
        _require(entity['activeModelRepresentationId'] is None or entity['activeModelRepresentationId'] not in partition_sources, 'part_source_model_inactive')
        assembly = any(e.get('parentEntityId') == entity['id'] for e in entities.values())
        _require(entity["activeModelRepresentationId"] is not None or entity.get("currentModelTransform") is None or assembly, "inactive_model_transform")
        records, selections = entity.get("measurementEvidence"), entity.get("measurementSelections")
        _require(isinstance(records, list) and isinstance(selections, dict), "measurement_evidence_required")
        by_id = {}
        for value in records:
            record = _typed(MeasurementEvidence, value, "invalid_measurement_evidence")
            _require(record["id"] not in evidence_ids, "duplicate_measurement_evidence_identity")
            evidence_ids.add(record["id"])
            _require(set(record["observationRefs"]) <= set(observations), "measurement_observation_not_found")
            _require(record["representationId"] is None or record["representationId"] in representation_ids, "measurement_representation_not_found")
            by_id[record["id"]] = record
        _require(set(selections) == set(entity.get("measurements") or {}), "measurement_selection_invalid")
        for key, selected in selections.items():
            record = by_id.get(selected)
            value = entity["measurements"][key]
            _require(value is None if selected is None else record is not None and record["measurementKey"] == key and record["originalMeasurement"] == value, "measurement_selection_invalid")
    decisions = document.get("identityDecisions")
    _require(isinstance(decisions, list), "identity_decisions_required")
    seen = set()
    for value in decisions:
        decision = _typed(IdentityDecision, value, "invalid_identity_decision")
        _require(decision["id"] not in seen and (decision["supersedesDecisionId"] is None or decision["supersedesDecisionId"] in seen), "identity_decision_history_invalid")
        seen.add(decision["id"])
        groups = decision["observationGroups"]
        refs = [oid for group in groups for oid in group]
        _require((decision["source"] == "manual" or all(groups)) and len(refs) == len(set(refs)) and set(refs) <= set(observations), "identity_observation_groups_invalid")
        for ref in decision["evidenceRefs"]:
            if ref["kind"] == "observation":
                _require(ref["observationId"] in refs and ref["observationRevision"] <= observations[ref["observationId"]].get("revision", 1), "identity_observation_revision_mismatch")
    active_decisions = effective_decisions(document)
    manual_different = [d for d in active_decisions if d["source"] == "manual" and d["decision"] == "different"]
    for decision in active_decisions:
        groups = decision["observationGroups"]
        owner_groups = [{owners[oid] for oid in group if oid in owners} for group in groups]
        if len(decision["entityIds"]) == len(groups):
            owner_groups = [owners_for_group or set(_targets(document, eid)) for eid, owners_for_group in zip(decision["entityIds"], owner_groups)]
        if decision["decision"] == "different":
            _require(not any(a & b for i, a in enumerate(owner_groups) for b in owner_groups[i + 1:]), "identity_different_requires_split")
        elif decision["decision"] == "same":
            # Explicit manual separation can disprove old geometric/source bindings.
            refs = {oid for g in groups for oid in g}
            separated = decision["source"] != "manual" and any(sum(bool(refs.intersection(g)) for g in d["observationGroups"]) > 1 for d in manual_different)
            _require(separated or len(set().union(*owner_groups)) == 1, "identity_same_requires_merge")
    cameras = {c["id"]: c for c in document["cameras"]}
    bindings = document.get("geometryBindings")
    image_ids = {a["id"] for a in document["assets"] if a.get("kind") == "source_image"} | {c["imageId"] for c in document["cameras"]}
    _require(isinstance(bindings, dict) and set(bindings) == image_ids, "geometry_bindings_invalid")
    for image_id, value in bindings.items():
        if value is not None:
            binding = _typed(GeometryBinding, value, "geometry_binding_invalid")
            _require(binding["cameraId"] in cameras and cameras[binding["cameraId"]]["imageId"] == image_id, "geometry_binding_camera_mismatch")
    def attachment(value):
        _require(value is None or value in entities, "identity_attachment_not_found", entityId=value)
    for annotation in document["annotations"]:
        attachment(annotation.get("entityId"))
    report = document.get("reportEvidence") or {}
    for record in report.get("objects", []):
        attachment(record.get("entityId"))
        for view in record.get("views", []):
            if view.get("observationId"):
                _require(view["observationId"] in observations, "identity_observation_attachment_not_found")
                if view.get("entityId"):
                    _require(owners.get(view["observationId"]) == view["entityId"], "identity_view_owner_mismatch")
    historical = report.get("historical") or {}
    for row in historical.get("inventory", []) + (historical.get("cad") or {}).get("regions", []):
        for identity in row.get("entityIds", []):
            attachment(identity)
