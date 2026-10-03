"""CPU-only check: python scripts/check_workcell_recgen_worker.py."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

from workcell_recgen_worker import main, validate_plan


def check():
    calls, pipeline = [], object()
    def load():
        calls.append('load')
        return pipeline
    def run(loaded, views, seed, settings):
        assert loaded is pipeline and seed == 42
        calls.append([int(v['rgb'][0,0,0]) for v in views])
        return {'vertices': np.zeros((3,3)), 'faces': np.array([[0,1,2]]), 'colors': np.ones((3,3)),
                'seconds': .1, 'faces_n': 1, 'stages': {'generation': .1}}
    previous = sys.modules.get('fast_report')
    sys.modules['fast_report'] = SimpleNamespace(x7=SimpleNamespace(load_recgen=load),
        recgen_fast=SimpleNamespace(run=run, FAST={}, setting=lambda **kw: kw))
    try:
        with tempfile.TemporaryDirectory(prefix='recgen-plan-check-') as directory:
            root=Path(directory)
            for kind,value in [('cart',20),('guard',70)]:
                arrays={}
                for i in (1,2):
                    arrays.update({f'v{i}_rgb': np.full((2,2,3),value+i,np.uint8),
                                   f'v{i}_depth': np.ones((2,2)), f'v{i}_mask': np.ones((2,2),np.uint8), f'v{i}_K': np.eye(3)})
                np.savez(root/f'{kind}.npz',**arrays)
            jobs=[{'kind':'cart','source':'cart.npz','target':'cart-result','groups':[['single',[1]]]},
                  {'kind':'guard','source':'guard.npz','target':'guard-result','groups':[['multi',[1,2]],['v2',[2]]]}]
            path=root/'plan.json'; path.write_text(json.dumps(jobs))
            with contextlib.redirect_stdout(io.StringIO()): report=main(path)
            assert calls == ['load',[21],[71,72],[72]], calls
            assert set(report['jobs'])=={'cart','guard'} and report['modelLoadSeconds']>=0
            assert report['jobs']['guard']['models']['multi']['views']==[1,2]
            assert all((root/f'{job["target"]}-{name}.npz').is_file() for job in jobs for name,_ in job['groups'])
            # A bad later job must fail before another expensive model load.
            for change in ({'source':'missing.npz'}, {'target':'cart-result'}, {'groups':[['bad',[3]]]},
                           {'groups':[['bad',[True]]]}):
                broken=[jobs[0], jobs[1] | change]; path.write_text(json.dumps(broken))
                before=list(calls)
                try: main(path)
                except ValueError: pass
                else: raise AssertionError(f'Invalid plan accepted: {change}')
                assert calls == before
            bad=np.load(root/'guard.npz'); arrays={k:bad[k] for k in bad.files}; bad.close()
            arrays['v1_depth'][0,0]=np.nan; np.savez(root/'guard.npz',**arrays)
            path.write_text(json.dumps(jobs))
            try: validate_plan(path)
            except ValueError: pass
            else: raise AssertionError('Nonfinite depth accepted')
    finally:
        if previous is None: sys.modules.pop('fast_report',None)
        else: sys.modules['fast_report']=previous
    print('workcell_recgen_worker: shared pipeline and complete preflight checks passed')


if __name__=='__main__':
    check()
