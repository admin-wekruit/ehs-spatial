"""One hosted SAM3D research call; service coordinates are not release evidence."""
import base64
import hashlib
from importlib.metadata import version
import io
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

from argus.platform.contracts import PlatformError, canonical, digest
from argus.platform.spatial import FrameGeometry, transform_points


ENDPOINT = 'fal-ai/sam-3/3d-objects'
SCHEMA_SHA256 = '06c7cc7c20594c86c1b1fbf5e9e616f8e5afb2d4271a5306c2fdcce81c5a217e'
PINS = {'model':'facebook/sam-3d-objects', 'adapter':'fal-sam3d-research-v1',
        'endpoint':ENDPOINT, 'schemaSha256':SCHEMA_SHA256}
POINTMAP_CONVENTION = 'owned_support_for_native_placement_shape_only_generation'
UPSTREAM_SOURCE = ('https://github.com/facebookresearch/sam-3d-objects/blob/'
    'f91db411c50efee93d8db7aeb323885650f6f722/sam3d_objects/pipeline/inference_pipeline_pointmap.py')
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def native_pose_hint(metadata, view):
    """Initialize shape placement from owned depth, never from provider camera scale.

    SAM's upstream row-vector transform uses wxyz and its GLB needs the export
    basis. This is only a seed: all owned views must validate the fitted result.
    """
    from scipy.spatial.transform import Rotation
    from argus.platform.spatial import affine
    try:
        quaternion = np.asarray(metadata['rotation'], dtype=float).reshape(4)
        scale = np.asarray(metadata['scale'], dtype=float).reshape(3)
        translation = np.asarray(metadata['translation'], dtype=float).reshape(3)
        depth, mask, valid = (np.asarray(view[k]) for k in ('depth','mask','valid'))
        if (not np.isfinite(np.r_[quaternion,scale,translation]).all() or
                np.linalg.norm(quaternion) < 1e-8 or (scale <= 0).any() or translation[2] <= 0 or
                mask.dtype != bool or valid.dtype != bool or depth.ndim != 2 or
                mask.shape != depth.shape or valid.shape != depth.shape):
            raise ValueError('invalid seed')
        support = mask & valid & np.isfinite(depth) & (depth > 0)
        if support.sum() < 8:
            raise ValueError('insufficient depth')
        ratio = float(np.median(depth[support]) / translation[2])
        basis = np.array([[1,0,0],[0,0,-1],[0,1,0.]])
        convention = np.diag([-1.,-1.,1.])
        rotation = Rotation.from_quat(quaternion[[1,2,3,0]]).as_matrix().T
        camera_pose = np.eye(4)
        camera_pose[:3,:3] = convention @ rotation @ np.diag(scale) @ basis * ratio
        camera_pose[:3,3] = convention @ translation * ratio
        return affine(view['cameraToWorld'], rigid=True) @ camera_pose
    except (ValueError,KeyError,TypeError) as exc:
        raise PlatformError('hosted_sam3d_pose_seed_invalid',422) from exc


def validate_runtime(protocol, pins):
    runtime = (protocol.get('runtimeManifest') or {}).get('generation') or {}
    if (protocol.get('purpose') != 'runtime_validation' or pins != PINS or
            set(protocol.get('runtimeManifest') or {}) != {'generation'} or
            set(runtime) != {'pins','provider','endpoint','schemaSha256','adapterSourceSha256',
                             'falClientVersion','pointmapConvention'} or
            runtime.get('pins') != PINS or runtime.get('provider') != 'fal' or
            runtime.get('endpoint') != ENDPOINT or runtime.get('schemaSha256') != SCHEMA_SHA256 or
            runtime.get('falClientVersion') != '1.0.0' or version('fal-client') != '1.0.0' or
            runtime.get('adapterSourceSha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest() or
            runtime.get('pointmapConvention') != POINTMAP_CONVENTION):
        raise PlatformError('research_runtime_unpinned', 409)


def _file(data, media_type):
    if not data or len(data) > MAX_FILE_BYTES:
        raise PlatformError('hosted_sam3d_file_size_invalid', 422)
    return {'bytes':data, 'sha256':hashlib.sha256(data).hexdigest(),
            'sizeBytes':len(data), 'mediaType':media_type}


def arguments(payload):
    """Freeze a test hypothesis from owned canonical pixels, without resizing."""
    image, mask = np.asarray(payload['image']), np.asarray(payload['mask'])
    frame = FrameGeometry(payload['imageId'],payload['coordinateFrameId'],payload['imageSha256'],
        np.asarray(payload['points']),np.asarray(payload['valid']),np.asarray(payload['K']),np.asarray(payload['cameraToWorld']))
    if (image.dtype != np.uint8 or image.shape != frame.points.shape or mask.dtype != np.bool_ or
            mask.shape != frame.valid.shape or not mask.any() or not np.isfinite(frame.points).all() or
            type(payload['seed']) is not int or frame.points.nbytes + image.nbytes + mask.nbytes > MAX_FILE_BYTES):
        raise PlatformError('generation_grid_mismatch', 422)
    # Upstream compute_pointmap consumes HWC PyTorch3D camera points. The hosted
    # wrapper's interpretation is a hypothesis to measure, not an asserted pin.
    pointmap = transform_points(frame.points,np.linalg.inv(frame.camera_to_world))
    pointmap = np.asarray(pointmap * [-1,-1,1],dtype='<f4')
    if not np.isfinite(pointmap).all():
        raise PlatformError('sam3d_pointmap_nonfinite', 422)
    files = {}
    for name, array in [('image',image),('mask',mask.astype(np.uint8)*255)]:
        stream = io.BytesIO()
        Image.fromarray(array).save(stream,format='PNG')
        files[name] = _file(stream.getvalue(),'image/png')
    stream = io.BytesIO()
    np.save(stream,pointmap,allow_pickle=False)
    files['pointmap'] = _file(stream.getvalue(),'application/octet-stream')
    def uri(name):
        f = files[name]
        return 'data:'+f['mediaType']+';base64,'+base64.b64encode(f['bytes']).decode()
    request = {'image_url':uri('image'),'mask_urls':[uri('mask')],
        'prompt':None,'point_prompts':[],'box_prompts':[],'seed':payload['seed'],'export_textured_glb':False}
    evidence = {'files':files,'requestSha256':digest(request),'pointmapConvention':POINTMAP_CONVENTION,
        'conventionStatus':'native_support_retained_for_placement','pointmapSentToProvider':False,'upstreamReference':UPSTREAM_SOURCE,
        'shapeHWC':list(pointmap.shape),'dtype':str(pointmap.dtype),
        'nativeToOpenCVCamera':np.linalg.inv(frame.camera_to_world).tolist(),
        'opencvCameraToSubmitted':[[-1,0,0],[0,-1,0],[0,0,1]],
        'invalidDepthPolicy':'preserve_finite_source_values_without_imputation',
        'sourceValidPixels':int(frame.valid.astype(bool).sum()),'maskPixels':int(mask.sum()),
        'sourceImageSha256':payload['imageSha256'],'coordinateFrameId':payload['coordinateFrameId']}
    return request,evidence


def _url(url, *, queue=False):
    parsed = urlsplit(url)
    host = parsed.hostname or ''
    allowed = host == 'queue.fal.run' if queue else host == 'fal.media' or host.endswith('.fal.media')
    if parsed.scheme != 'https' or not allowed or parsed.username or parsed.password or parsed.port not in (None,443):
        raise PlatformError('hosted_sam3d_response_url_invalid', 422)
    return url


def _read(client, url, maximum):
    with client.stream('GET',url,follow_redirects=False,timeout=30) as response:
        if int(response.headers.get('content-length','0')) > maximum:
            raise PlatformError('hosted_sam3d_file_size_invalid', 422)
        data = bytearray()
        for chunk in response.iter_bytes():
            data.extend(chunk)
            if len(data) > maximum:
                raise PlatformError('hosted_sam3d_file_size_invalid', 422)
        return response.status_code,bytes(data)


def _mesh(data):
    from argus.platform.blender_export import _read_glb, mesh_from_asset
    spec,_ = _read_glb(data)
    nodes = spec.get('nodes',[])
    children = [child for node in nodes for child in node.get('children',[])]
    counts = spec.get('accessors',[])
    # Bound decoded geometry before the shared parser. A tree excludes exponential
    # repeated-node expansion; textured decoding is outside this vertex-color trial.
    if (len(nodes) > 128 or len(children) != len(set(children)) or spec.get('images') or
            any(type(a.get('count')) is not int or a['count'] < 0 for a in counts) or
            sum(a['count'] for a in counts)*max(1,len(nodes)) > 12_000_000):
        raise PlatformError('hosted_sam3d_mesh_limit', 422)
    mesh = mesh_from_asset(data,{'sha256':hashlib.sha256(data).hexdigest()})
    if len(mesh.vertices) > 1_000_000 or len(mesh.faces) > 2_000_000:
        raise PlatformError('hosted_sam3d_mesh_limit', 422)
    triangle = mesh.vertices[mesh.faces]
    if not np.any(np.any(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]) != 0,axis=1)):
        raise PlatformError('hosted_sam3d_mesh_degenerate', 422)
    return mesh


def invoke(payload, *, on_dispatched):
    import fal_client
    import httpx
    from fal_client.client import QUEUE_URL_FORMAT
    from argus.platform.reconstruction import ProviderResponseError, _telemetry

    validate_runtime(payload.get('_researchProtocol',{}),PINS)
    request,submitted = arguments(payload)
    evidence = {'submitted':submitted,'endpoint':ENDPOINT,'schemaSha256':SCHEMA_SHA256,
        'runtimeManifestSha256':digest(payload['_researchProtocol']['runtimeManifest']['generation']),
        'checkpointRevision':None,'internalDepthCalls':None,'nativePoseStatus':'unverified'}
    receipt = {}
    service = fal_client.SyncClient()
    # The hosted external-pointmap wrapper rejects NPY arrays inside its model.
    # Use SAM3D for shape only. Owned native points remain the placement source;
    # returned service pose/depth is never silently accepted as the source camera.
    evidence['submittedRequestSha256'] = digest(request)
    phase, queued_data = 'dispatch', {}
    try:
        # Exactly one charged POST. fal-client.submit retries POST, so reuse only
        # its authenticated HTTP client; all subsequent requests are read-only.
        with service._client as client:
            queued = client.post(QUEUE_URL_FORMAT+ENDPOINT,json=request,follow_redirects=False,timeout=60)
            if 400 <= queued.status_code < 500:
                evidence['queueResponse'] = _file(queued.content[:MAX_RESPONSE_BYTES],'application/json')
                return {'providerError':{'code':'hosted_sam3d_request_rejected','httpStatus':queued.status_code},'runtimeEvidence':evidence}
            queued.raise_for_status()
            queued_data = queued.json()
            request_id = queued_data['request_id']
            if not isinstance(request_id,str) or not 1 <= len(request_id) <= 128:
                raise ValueError('invalid_provider_request_id')
            receipt = {'providerRequestId':request_id}
            on_dispatched(request_id)
            phase = 'queue_urls'
            evidence['queueResponse'] = _file(queued.content,'application/json')
            status_url = _url(queued_data['status_url'],queue=True)
            response_url = _url(queued_data['response_url'],queue=True)
            deadline = time.monotonic()+600
            phase = 'poll'
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('provider_poll_timeout')
                status,raw = _read(client,status_url,MAX_RESPONSE_BYTES)
                if status == 429 or status >= 500:
                    time.sleep(1)
                    continue
                if status not in (200,202):
                    raise ValueError('provider_status_unavailable')
                state = json.loads(raw)['status']
                if state == 'COMPLETED':
                    break
                if state not in ('IN_QUEUE','IN_PROGRESS'):
                    raise ValueError('provider_status_unknown')
                time.sleep(1)
            phase = 'result'
            status,raw = _read(client,response_url,MAX_RESPONSE_BYTES)
    except Exception as exc:
        metadata = {**_telemetry(receipt),'failurePhase':phase,'failureType':type(exc).__name__,
                    'queueUrls':{k:queued_data.get(k) for k in ('status_url','response_url')}}
        raise ProviderResponseError(metadata,outcome='outcome_unknown') from None
    evidence['rawResponse'] = _file(raw,'application/json')
    result = {**receipt,'runtimeEvidence':evidence,'telemetry':{'actualCostUsd':None},
        'proposedObjectToNative':None,'provenance':{'pins':PINS,'shapeStatus':'research_only',
            'geometryCoordinates':'returned_glb_scene','nativePoseStatus':'unverified',
            'pointmapConventionStatus':'native_support_retained_for_placement'}}
    try:
        response = json.loads(raw)
        evidence['response'] = response
        if status != 200:
            raise PlatformError('hosted_sam3d_provider_rejected', 422)
        metadata = response['metadata']
        if not isinstance(metadata,list) or len(metadata) != 1 or metadata[0].get('object_index') != 0:
            raise PlatformError('hosted_sam3d_object_count_mismatch', 422)
        evidence['poseMetadata'] = _file(canonical(metadata),'application/json')
        # A fresh unauthenticated client never forwards FAL credentials to media.
        with httpx.Client() as client:
            code,data = _read(client,_url(response['model_glb']['url']),MAX_FILE_BYTES)
        evidence['modelGlb'] = _file(data,'model/gltf-binary')
        if code != 200:
            raise PlatformError('hosted_sam3d_mesh_unavailable', 422)
        mesh = _mesh(data)
        result.update(vertices=mesh.vertices,faces=mesh.faces,colors=mesh.colors)
    except Exception as exc:
        result['providerError'] = {'code':exc.code if isinstance(exc,PlatformError) else 'hosted_sam3d_response_invalid'}
    return result
