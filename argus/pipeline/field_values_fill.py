"""Field values from the unchanged fair evaluator for both MVS+fill workcells."""
from argus import ROOT
import json
import os
from pathlib import Path
import sys

HERE = Path(os.environ['PANOPTES_DATA_ROOT']) / 'pipeline'
HERE.mkdir(parents=True, exist_ok=True)
RN = HERE.parent
SP = Path(os.environ['PANOPTES_DATA_ROOT'])   # env.sh (env.template section 6)
AN = SP / 'checks/bbab-analyse'
import argus.pipeline.field_summary_safe as clean
fc = clean.fc


def main():
    runs = {}
    fill030 = '030-mvs-fill-padded'
    for name, key in (('090-mvs-fill-padded', 'mvs-fill'), (fill030, 'mvs-fill'),
                      ('090-mvs-da3-base-padded', 'mvs-da3-base'), ('030-mvs-da3-base-padded', 'mvs-da3-base')):
        r = json.loads((AN / f'{name}.json').read_text()); r = dict(r, backbone=key)
        runs[(r['cell'], key, r['variant'])] = r
    fc.LICENCE.setdefault('mvs-fill', 'MIT / BSD-3 / Apache-2.0 + MoGe-3 (MIT) in-mask fill')
    fc.LICENCE.setdefault('mvs-da3-base', 'MIT / BSD-3 / Apache-2.0')
    out = {}
    for key in ('mvs-da3-base', 'mvs-fill'):
        r = clean.safe_config(runs, key, 'padded', 'maxInlier', 'heightCm')
        v = r['values']
        out[key] = {'housing090R_cm': round(v['090R housing']['cm'], 1), 'housing090R_err': round(v['090R housing']['errCm'], 1),
                    'fence090R_cm': round(v['090R fence']['cm'], 1), 'fence090R_err': round(v['090R fence']['errCm'], 1),
                    'housing090L_cm': round(r['listed090L'], 1), 'estopMaxDevPct090': round(r['gate']['090']['maxDeviationPct'], 2),
                    'maeCm4values': round(r['maeCm'], 2), 'maxAbsErrCm': round(r['maxAbsErrCm'], 2),
                    'floorP95Cm090': r['floorP95Cm']['090'], 'cameraHeightRangeCm': round(r['cameraHeightRangeCm'], 1),
                    'housing030L_cm': round(v['030L housing']['cm'], 1), 'housing030L_err': round(v['030L housing']['errCm'], 1),
                    'housing030R_cm': round(v['030R housing']['cm'], 1), 'housing030R_err': round(v['030R housing']['errCm'], 1),
                    'estopMaxDevPct030': round(r['gate']['030']['maxDeviationPct'], 2), 'floorP95Cm030': r['floorP95Cm']['030'],
                    'label': {'mvs-da3-base': 'MVS, CERT 0.05 (DA3-BASE start)', 'mvs-fill': 'MVS + MoGe-3 in-mask fill'}[key],
                    'estopNativeToMeters090': runs[('090', key, 'padded')]['floors']['maxInlier']['estop']['nativeToMeters'],
                    'estopNativeToMeters030': runs[('030', key, 'padded')]['floors']['maxInlier']['estop']['nativeToMeters'],
                    'note': ('030 side = ' + fill030) if key == 'mvs-fill' else 'the geometry-backbone-ab row, recomputed here'}
        print(key, json.dumps(out[key]))
    (HERE / 'field-values-mvs-fill.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
