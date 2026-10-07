"""Capture regression (run: python scripts/check_workcell_vguard.py RUN): three guard boards and intrinsic two-face bend angles."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from workcell_photo_oneshot import GUARD_MIN_PIXELS, CART_MIN_PIXELS, _visible_views, scene_photos

# The guard is generated from its two best-supported views.
guard_views = lambda rows: _visible_views(rows, GUARD_MIN_PIXELS)[:2]
assert guard_views([{'photo':1,'supportedPixels':400}, {'photo':2,'supportedPixels':30},
                    {'photo':3,'supportedPixels':800}, {'photo':4,'supportedPixels':700}]) == [3,4]

root=Path(sys.argv[1]);report=json.loads((root/'scene-report.json').read_text())
objects={o['id']:o for o in report['objects']}
ids={'v-guard-'+side for side in ('left','center','right')}
assert {key for key in objects if key.startswith('v-guard')} == ids, 'Three physical boards, not local planar patches'
z=np.load(root/'guard-input.npz');cart=np.load(root/'cart-input.npz')
count=scene_photos(root)[0]
generation=guard_views(json.loads((root/'guard-mask-selection.json').read_text()))
assert len(generation)==2 and all((z[f'v{i}_mask']>0).sum()>=GUARD_MIN_PIXELS for i in generation)
for i in range(1,count+1):
    m=z[f'v{i}_mask']>0;c=cart[f'v{i}_mask']>0
    assert not c.any() or c.sum()>CART_MIN_PIXELS, 'a view either shows a cart mask or none'
    assert not (m&c).any(), 'Guard pixels must not enter cart generation'
mesh=trimesh.load(root/'guard-multi.glb',force='mesh')
assert len(mesh.faces)>10 and np.isfinite(mesh.vertices).all()
# A promoted structural fit replaces guard-placement.json; the generated model's placement is kept as the initializer.
placement=json.loads((root/('guard-placement-initializer.json' if (root/'guard-placement-initializer.json').is_file() else 'guard-placement.json')).read_text())
assert placement['generationViews'] == generation
assert placement['loss_after'] <= placement['loss_before']
assert len(placement['sourceChecks']) == count, 'the placed guard is scored in every photo'

partition=json.loads((root/'guard-partition.json').read_text())
ownership=[index for part in partition['parts'] for index in part['sourceFaceIndices']]
assert sorted(ownership)==list(range(len(mesh.faces))), 'Every source face must occur in exactly one board'
for part in partition['parts']:
    child=trimesh.load(root/part['file'],force='mesh')
    source=mesh.triangles[part['sourceFaceIndices']]
    assert len(child.faces)==len(source) and np.allclose(child.triangles,source,atol=1e-6), 'Splitting must preserve source geometry'
rows={r['entityId']:r for r in report['bendAnalysis']['items']}
assert set(rows)==ids
for ident,row in rows.items():
    if row['status']=='measured':
        result=row['result']
        assert result['kind']=='bend' and result['method']=='same-mesh-two-surface-interior-bend-v2'
        assert 0<result['value']<180 and len(result['lines'])==5
        assert result['references'][0]['entityId']==ident
    else:
        assert row['status']=='unsupported' and row['reason']=='measurement_no_stable_bend' and not row.get('result')
structural=json.loads((root/'structural-result.json').read_text()) if (root/'structural-result.json').is_file() else {}
if structural.get('promotionAllowed'):
    assert rows['v-guard-left']['status']==rows['v-guard-right']['status']=='measured', 'promoted two-sheet boards always carry their bend'
else:
    assert not structural or structural['fitGate']['reasons'], 'an unpromoted structural fit states why'
print('PASS: three exhaustive unchanged board meshes, intrinsic per-board bend annotations (measured on promoted two-sheet boards), unsupported geometry has no invented angle;', count, 'photos')
