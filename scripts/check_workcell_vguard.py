"""Known capture regression: separate guard support, geometry and saved ground angles."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from workcell_photo_oneshot import _guard_views

assert _guard_views([{'photo':1,'supportedPixels':400}, {'photo':2,'supportedPixels':30},
                     {'photo':3,'supportedPixels':800}, {'photo':4,'supportedPixels':700}]) == [3,4]

root=Path(sys.argv[1]);report=json.loads((root/'scene-report.json').read_text())
objects={o['id']:o for o in report['objects']}
assert 'v-guard' in objects, 'V-guard must be an independent selectable object'
guard=objects['v-guard'];assert guard['model']['file']=='guard-multi.glb'
z=np.load(root/'guard-input.npz');cart=np.load(root/'cart-input.npz')
for i in range(1,5):
    m=z[f'v{i}_mask']>0;c=cart[f'v{i}_mask']>0
    assert m.sum()>=100 and c.sum()>=2500
    assert not (m&c).any(), 'Guard pixels must not enter cart generation'
mesh=trimesh.load(root/'guard-multi.glb',force='mesh')
assert len(mesh.faces)>10 and np.isfinite(mesh.vertices).all()
placement=json.loads((root/'guard-placement.json').read_text())
assert placement['generationViews'] == _guard_views(json.loads((root/'guard-mask-selection.json').read_text()))
assert placement['loss_after'] <= placement['loss_before']
assert len(placement['sourceChecks']) == 4
rows=report['inclinationAnalysis']['items'];row=next(r for r in rows if r['entityId']=='v-guard')
assert row['surfaces'], 'No supported V-guard plane available for its ground-angle card'
for s in row['surfaces']:
    assert 0<=s['inclinationDeg']<=90
    assert np.isclose(s['inclinationDeg']+s['deviationFromVerticalDeg'],90)
    assert s['result']['coordinateFrameId']=='workcell-floor'
    assert s['result']['source']=='model_inference'
    assert s['classification']=='direction_unverified', 'No independently measured ground angular error'
    assert s['result']['lines'] and s['result']['references'][0]['entityId']=='v-guard'
print('PASS: independent V-guard masks/model, cart exclusion, saved ground angles;',len(row['surfaces']),'supported planes')
