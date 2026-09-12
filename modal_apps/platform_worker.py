"""Durable job runner; configure image digest and secrets before deployment.

The image must contain this exact source revision plus its locked dependencies.
Separate platform_models workers carry model-specific CUDA environments.
"""
import os
import modal

image_ref=os.environ.get('PANOPTES_WORKER_IMAGE')
if not image_ref or '@sha256:' not in image_ref:
    raise ValueError('PANOPTES_WORKER_IMAGE must be an audited immutable image digest')
secret_name=os.environ.get('PANOPTES_WORKER_SECRET')
if not secret_name:
    raise ValueError('PANOPTES_WORKER_SECRET must name the configured DB/storage service secret')
app=modal.App('panoptes-platform-worker')
image=modal.Image.from_registry(image_ref)


@app.function(image=image,secrets=[modal.Secret.from_name(secret_name)],timeout=3600)
def execute(job_id: str):
    from uuid import UUID
    from ehs_spatial.platform.runtime import services
    from panoptes_worker.__main__ import run_job
    repository,blobs=services()
    result=run_job(repository,blobs,str(UUID(job_id)))
    return {'jobId':result['id'],'status':result['status'],'headAdvanced':result.get('headAdvanced',False)}
