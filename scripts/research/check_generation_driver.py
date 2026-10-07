"""Offline check that a deployed CPU worker uses the named GPU function once."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import generate_lucida_assets as driver


def check():
    payload=b'exact-payload'
    calls=[]
    class Upload:
        def __enter__(self): return self
        def __exit__(self,*args): return False
        def put_file(self,path,destination):
            assert Path(path).read_bytes()==payload and destination.endswith('/input.npz')
    def remote(job_id,*,object_id,seed):
        calls.append((object_id,seed))
        return {'job_id':job_id}
    def named(app,name):
        assert (app,name)==('deployed-test','generate_object')
        return SimpleNamespace(remote=remote)
    def cannot_start_app():
        raise AssertionError('A deployed CPU worker must not nest app.run()')
    native={'status':'complete','source_payload_sha256':hashlib.sha256(payload).hexdigest(),
            'gpu_function_seconds':1}
    fake=SimpleNamespace(app=SimpleNamespace(run=cannot_start_app),generate_object=None,
        volume=SimpleNamespace(batch_upload=Upload),read_outputs=lambda _:{'output.json':json.dumps(native).encode()},
        CODE_REV='code',MODEL_REV='model')
    with tempfile.TemporaryDirectory(prefix='deployed-driver-check-') as temporary:
        root=Path(temporary)
        (root/'evidence').mkdir();(root/'evidence/objects.json').write_text('{"objects":[{"object_id":"object_exact"}]}')
        generation=root/'generation';(generation/'environment-recgen').mkdir(parents=True)
        (generation/'environment-recgen/output.json').write_text('{"status":"complete","gpu_function_seconds":1}')
        (generation/'gpu-budget.json').write_text('{"limit_seconds":630,"calls":[]}')
        with patch.dict(sys.modules,{'modal_apps.lucida_assets':fake,'modal':SimpleNamespace(Function=SimpleNamespace(from_name=named))}), \
             patch.dict(driver.os.environ,{'LUCIDA_DEPLOYED_APP':'deployed-test'}), \
             patch.object(sys,'argv',['generate_lucida_assets.py','--root',str(root),'--object-ids','object_exact']), \
             patch.object(driver,'payload_for_object',return_value=(payload,{'generation_input':'canonical'})):
            driver.main()
        ledger=json.loads((generation/'gpu-budget.json').read_text())
        assert calls==[('object_exact',42)] and len(ledger['calls'])==1
        assert ledger['calls'][0]['status']=='complete' and ledger['calls'][0]['charged_seconds']==31
        assert (generation/'object_exact/input.npz').read_bytes()==payload
    print('PASS: deployed named function, one exact candidate, same payload, no app.run, budget preserved; no network/GPU calls')


if __name__=='__main__':check()
