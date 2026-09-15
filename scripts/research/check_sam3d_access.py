"""Diagnose the configured HF identity without downloading model weights."""
import json
import os
from pathlib import Path, PurePosixPath

import modal

MODEL = 'facebook/sam-3d-objects'
REVISION = '2e73555018d2741ccd486e56c24fac41155a1dc6'
app = modal.App('panoptes-sam3d-access-check')


def inspect_access():
    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url, hf_hub_download
    import yaml
    token = next((os.environ[k] for k in ('HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN',
                  'HUGGINGFACE_TOKEN', 'HUGGINGFACE_HUB_TOKEN') if os.environ.get(k)), None)
    report = {'model': MODEL, 'modelRevision': REVISION, 'credentialPresent': bool(token),
              'newModelCalls': 0, 'gpuUsed': False, 'checks': []}
    if not token:
        return {**report, 'status': 'credential_unavailable'}

    def attempt(name, action):
        try:
            result = action()
            report['checks'].append({'check': name, 'status': 'accessible'})
            return result
        except Exception as error:
            response = getattr(error, 'response', None)
            headers = response.headers if response is not None else {}
            # Error text and redirect URLs may contain credentials; never return them.
            message = str(getattr(error, 'server_message', '') or '').lower()
            reason = ('token_gated_permission_missing' if 'public gated' in message and 'token' in message
                      else 'account_access_not_approved' if 'authorized list' in message or 'not been granted' in message
                      else 'access_request_pending' if 'awaiting' in message or 'pending' in message
                      else 'gated_access_reason_unclassified')
            report['checks'].append({'check': name, 'status': 'failed', 'type': type(error).__name__,
                'httpStatus': response.status_code if response is not None else None,
                'errorCode': headers.get('x-error-code'), 'requestId': headers.get('x-request-id'), 'reason': reason})
            return None

    auth = attempt('identity', lambda: HfApi().whoami(token=token))
    if auth is None:
        return {**report, 'status': 'identity_not_established'}
    report['authenticatedAccount'] = auth.get('name')
    info = attempt('fixed_revision', lambda: HfApi().model_info(MODEL, revision=REVISION, token=token))
    if info is None:
        return {**report, 'status': 'repository_access_not_established'}
    report['resolvedRevision'] = info.sha
    names = {s.rfilename for s in info.siblings}
    filename = 'checkpoints/pipeline.yaml'
    report['configListed'] = filename in names
    metadata = attempt('config_metadata', lambda: get_hf_file_metadata(hf_hub_url(MODEL, filename, revision=REVISION), token=token))
    if metadata is None:
        return {**report, 'status': 'file_access_not_established'}
    report['config'] = {'commit': metadata.commit_hash, 'etag': metadata.etag, 'sizeBytes': metadata.size}
    if metadata.commit_hash != REVISION or metadata.size is None or metadata.size > 1024 * 1024:
        return {**report, 'status': 'unexpected_config_metadata'}
    path = attempt('config_read', lambda: hf_hub_download(MODEL, filename, revision=REVISION, token=token))
    if path is None:
        return {**report, 'status': 'config_read_failed'}
    config = yaml.safe_load(Path(path).read_bytes())
    weight = config.get('slat_decoder_mesh_ckpt_path')
    if not isinstance(weight, str) or PurePosixPath(weight).is_absolute() or '..' in PurePosixPath(weight).parts:
        return {**report, 'status': 'mesh_checkpoint_path_unresolved'}
    candidates = [p for p in (weight, str(PurePosixPath(filename).parent / weight)) if p in names]
    if len(set(candidates)) != 1:
        return {**report, 'status': 'mesh_checkpoint_path_unresolved'}
    weight = candidates[0]
    metadata = attempt('mesh_weight_metadata', lambda: get_hf_file_metadata(hf_hub_url(MODEL, weight, revision=REVISION), token=token))
    if metadata is None:
        return {**report, 'status': 'weight_access_not_established'}
    report['weight'] = {'file': weight, 'commit': metadata.commit_hash, 'etag': metadata.etag, 'sizeBytes': metadata.size}
    return {**report, 'status': 'accessible' if metadata.commit_hash == REVISION else 'revision_mismatch'}


@app.function(image=modal.Image.debian_slim(python_version='3.12').pip_install('huggingface_hub==1.23.0', 'PyYAML==6.0.3'),
              secrets=[modal.Secret.from_name('huggingface')], timeout=90, retries=0, max_containers=1)
def check():
    return inspect_access()


@app.local_entrypoint()
def main(output: str = '.platform/model-correspondence/hf-access.json'):
    result = check.remote()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
