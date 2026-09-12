"""S3 supported-subset contract; no live provider or network calls."""
import hashlib
import io

import boto3
from botocore.response import StreamingBody
from botocore.stub import Stubber
import pytest

from ehs_spatial.platform.contracts import PlatformError
from ehs_spatial.platform.storage import S3BlobStore


def test_content_addressed_s3_subset_reuse_integrity_and_access_errors(monkeypatch):
    client = boto3.client('s3', region_name='us-east-1', aws_access_key_id='test', aws_secret_access_key='test')
    configured = {}
    def factory(*args, **kwargs):
        configured.update(kwargs)
        return client
    monkeypatch.setattr(boto3, 'client', factory)
    store = S3BlobStore('fixture')
    assert configured['config'].signature_version == 's3v4'
    assert configured['config'].request_checksum_calculation == 'when_required'
    assert configured['config'].response_checksum_validation == 'when_required'
    data = b'immutable spatial asset'
    sha = hashlib.sha256(data).hexdigest()
    key = 'sha256/' + sha
    get_params = {'Bucket':'fixture', 'Key':'panoptes/' + key}
    put_params = {**get_params, 'Body':data, 'ContentType':'application/octet-stream', 'Metadata':{'sha256':sha}}
    def response(body=data):
        return {'Body':StreamingBody(io.BytesIO(body),len(body))}
    with Stubber(client) as stub:
        # Missing object -> ordinary PUT -> byte verification; no unsupported
        # IfNoneMatch / ChecksumSHA256 / ChecksumMode parameters are accepted.
        stub.add_client_error('get_object', 'NoSuchKey', http_status_code=404, expected_params=get_params)
        stub.add_response('put_object', {}, put_params)
        stub.add_response('get_object', response(), get_params)
        assert store.put(data)['sha256'] == sha
        # Existing valid object -> no write.
        stub.add_response('get_object', response(), get_params)
        assert store.put(data)['sizeBytes'] == len(data)
        # A permission failure is not a missing object and must never PUT.
        stub.add_client_error('get_object', 'AccessDenied', http_status_code=403, expected_params=get_params)
        with pytest.raises(PlatformError, match='blob_read_failed'):
            store.put(data)
        # Corrupt bytes cannot be overwritten by a supposedly repairing upload.
        stub.add_response('get_object', response(b'corrupt'), get_params)
        with pytest.raises(PlatformError, match='blob_integrity_error'):
            store.put(data)
        # HEAD reports metadata; open/get still check the content-derived hash.
        stub.add_response('head_object', {'ContentLength':len(data), 'ContentType':'application/octet-stream'}, get_params)
        assert store.head(key)['sha256'] == sha
        stub.add_client_error('head_object', '404', http_status_code=404, expected_params=get_params)
        assert store.head(key) is None
        stub.assert_no_pending_responses()
    with pytest.raises(PlatformError, match='blob_integrity_error'):
        store.put_immutable(key,b'other payload',sha,len(data),'application/octet-stream')
