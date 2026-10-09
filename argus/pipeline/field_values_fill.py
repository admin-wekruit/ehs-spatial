"""Field values from the unchanged fair evaluator for one MVS+fill workcell."""
import argparse
import json
import os
from pathlib import Path
import statistics as st

from argus.checks.field_geometry import FIELD_CM
from argus.pipeline.field_summary import SCORED, NAMES

SP = Path(os.environ['PANOPTES_DATA_ROOT'])
HERE = SP / 'pipeline'
AN = SP / 'checks/bbab-analyse'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell', required=True, choices=sorted({c for c, _, _ in SCORED}))
    cell = parser.parse_args().cell
    out = {}
    for key in ('mvs-da3-base', 'mvs-fill'):
        name = f'{cell}-{key}-padded'
        floor = json.loads((AN / f'{name}.json').read_text())['floors']['maxInlier']
        if not floor['gatePassed']:
            raise RuntimeError(f'{name}: e-stop reference gate failed')
        values, errors = {}, []
        for c, eid, kind in SCORED:
            if c != cell:
                continue
            value = floor[kind][eid]['heightCm']
            if value is None:
                raise RuntimeError(f'{name}: missing scored field value: {eid}')
            error = value - FIELD_CM[kind]
            field = kind + NAMES[eid].split()[0]
            values.update({field + '_cm': round(value, 1), field + '_err': round(error, 1)})
            errors.append(abs(error))
        out[key] = dict(values, maeCm=round(st.mean(errors), 2), maxAbsErrCm=round(max(errors), 2),
                        cameraHeightRangeCm=round(max(floor['cameraHeightsCm']) - min(floor['cameraHeightsCm']), 1),
                        label={'mvs-da3-base': 'MVS, CERT 0.05 (DA3-BASE start)', 'mvs-fill': 'MVS + MoGe-3 in-mask fill'}[key],
                        note=name)
        out[key].update({f'estopMaxDevPct{cell}': round(100 * floor['estop']['maxDeviation'], 2),
                         f'floorP95Cm{cell}': floor['residualP95Cm'],
                         f'estopNativeToMeters{cell}': floor['estop']['nativeToMeters']})
        if cell == '090':
            out[key]['housing090L_cm'] = round(floor['housing']['5163a9b0']['heightCm'], 1)
        print(key, json.dumps(out[key]))
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / f'field-values-{cell}-mvs-fill.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
