"""Shared database authority and budget boundary for frozen research dispatch."""
from decimal import Decimal

from .contracts import PlatformError, digest


def admin_context(repository, connection, project_id, branch_id, base_id):
    authority = connection.execute("""SELECT current_user AS role,
        has_table_privilege(current_user,'jobs','INSERT') AND
        has_table_privilege(current_user,'assets','INSERT') AS permitted""").fetchone()
    if not authority['permitted']:
        raise PlatformError('admin_job_required', 403)
    repository._branch(connection, project_id, branch_id)
    revision = repository._revision(connection, project_id, base_id)
    if str(revision['branch_id']) != branch_id:
        raise PlatformError('research_baseline_branch_mismatch', 409)
    return {'source':'database_admin', 'databaseRole':authority['role']}, revision


def check_budget(repository, connection, protocol):
    budget = repository.paid_budget
    if budget is None or budget <= 0:
        raise PlatformError('paid_budget_not_configured', 409)
    total = Decimal(str(protocol['callLimits']['maxTotalCostUsd']))
    spent = connection.execute('SELECT COALESCE(sum(COALESCE(actual_cost,estimated_cost)),0) AS cost FROM model_calls').fetchone()['cost']
    if total > budget or spent + total > budget:
        raise PlatformError('paid_budget_exceeded', 409)
    return {'configuredBudgetUsd':str(budget), 'spentOrReservedUsd':str(spent)}


def validate_prepared(repository, connection, frozen, sha256):
    """Validate before a transaction enqueues; reserve_model_call remains the spending lock."""
    from .reconstruction import _unpacked, _validate_research_inputs, _research_stage, providers_from_manifest
    if (digest(frozen) != sha256 or frozen.get('schemaVersion') != 1 or
            frozen.get('authority', {}).get('source') != 'database_admin'):
        raise PlatformError('research_input_hash_mismatch', 409)
    authority, source = admin_context(repository, connection, *(frozen[k] for k in ('projectId','branchId','baseRevisionId')))
    if digest(source['document']) != frozen.get('baseDocumentSha256'):
        raise PlatformError('research_input_hash_mismatch', 409)
    protocol = frozen['protocol']
    stage = _research_stage(protocol, frozen['providerManifest'])
    job = {**{k:frozen[k] for k in ('projectId','branchId','baseRevisionId')},
           'kind':'validate_model', 'inputs': {}, 'config':{'researchProtocolSha256':digest(protocol)}}
    _validate_research_inputs(job,stage,_unpacked(frozen['payload']),frozen['images'],frozen['providerManifest'],protocol,source['document'], repository=repository, blobs=repository.blobs)
    provider = providers_from_manifest(frozen['providerManifest'], _research=True)[stage]
    provider.validate(stage, research_protocol=protocol)
    if provider.paid is not True or not 0 < provider.estimated_cost_usd <= protocol['callLimits']['maxCostPerCallUsd']:
        raise PlatformError('research_call_budget_invalid', 409)
    check_budget(repository, connection, protocol)
    scene_assets = {a['id']:a for a in source['document']['assets']}
    for ref in protocol['inputAssetHashes']:
        asset = repository._one(connection,'SELECT * FROM assets WHERE id=%s',(ref['assetId'],),code='asset_not_found')
        if (ref['assetId'] not in scene_assets or asset['sha256'].strip() != ref['sha256'] or
                scene_assets[ref['assetId']]['sha256'] != ref['sha256']):
            raise PlatformError('research_input_hash_mismatch', 409)
        if repository.blobs is None:
            raise PlatformError('asset_storage_unavailable', 409)
        repository.blobs.get(asset['storage_key'],ref['sha256'],asset['size_bytes'])
    return authority
