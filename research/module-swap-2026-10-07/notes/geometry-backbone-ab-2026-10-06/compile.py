"""Aggregate every geometry backbone on the one contract and the one evaluator -> results.json + the README tables.

Per-run results come from three folders, all written by the same unchanged evaluator (fair_ab_modal.analyse_one):
  da3fair-analyse  geometry-licence-ab-fair-2026-10-05       Pi3X published, DA3-LARGE-1.1, DA3-BASE (carried over)
  clean-analyse    licence-clean-stack-2026-10-06/geometry  RoMa + pycolmap BA rows, MVS at CERT 0.5, MVS + fill (carried over)
  bbab-analyse     this note                                 MVS completed (CERT = RoMa sample_thresh), sensitivity, diagnostics
Rows are scored by the fair compile.config(), unchanged. Pass rule (the coordinator's, fixed before the new MVS numbers):
  both e-stop gates passed, MAE (4 field values) <= 1.56 cm (Pi3X 1.44 + 0.12 noise), every field error <= 3 cm,
  floor residual p95 <= 2.5 cm in both cells (the loosest backbone already scored, DA3-BASE, is 2.3)
"""
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

NOTE = Path(__file__).resolve().parent
FAIR = NOTE.parent / 'geometry-licence-ab-fair-2026-10-05'
sys.path.insert(0, str(NOTE))
import backbones as bb  # noqa: E402
_spec = importlib.util.spec_from_file_location('clean_compile', NOTE.parent / 'licence-clean-stack-2026-10-06/geometry/compile.py')
clean = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(clean)
fc = clean.fc  # the fair compile module, unchanged (config, RULES, LICENCE); clean.safe_config = fc.config that keeps incomplete rows

SCR = fc.SCR
ANALYSE = [SCR / 'checks/da3fair-analyse', SCR / 'checks/clean-analyse', SCR / 'checks/bbab-analyse']
RUNS = Path(os.environ.get('PANOPTES_RUNS', '/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation'))
CELL_RUN = {'090': RUNS / 'lucida-replica-01', '030': RUNS / 'bor1-030-01'}
CLEAN_MVS = 'MIT / BSD-3 / Apache-2.0 (RoMa outdoor, pycolmap, OpenCV; the start backbone only initialises the BA)'
# (backbone key, label, licence, licence clean, group)
ROWS = [('pi3x-published', 'Pi3X (published)', 'CC BY-NC 4.0', False, 'carried: fair A/B'),
        ('da3-large-1.1', 'DA3-LARGE-1.1', 'disputed (card Apache-2.0, README CC BY-NC 4.0)', False, 'carried: fair A/B'),
        ('da3-base', 'DA3-BASE', 'Apache-2.0', True, 'carried: fair A/B'),
        ('map-anything-apache', 'map-anything-apache (image only)', 'Apache-2.0', True, 're-scored here'),
        ('mapanything-ba-f', 'map-anything-apache + RoMa BA-f', 'Apache-2.0 + MIT/BSD', True, 'carried: clean A/B'),
        ('moge-ba-f', 'MoGe-3 + RoMa BA-f', 'MIT + MIT/BSD', True, 'carried: clean A/B'),
        ('da3-base-ba-f', 'DA3-BASE + RoMa BA-f', 'Apache-2.0 + MIT/BSD', True, 'carried: clean A/B'),
        ('da3-base-ba-f-fused', 'MVS + DA3-BASE fill (CERT 0.5)', 'Apache-2.0 + MIT/BSD', True, 'carried: clean A/B'),
        ('da3-base-ba-f-mvs', 'MVS, CERT 0.5 (DA3-BASE start)', CLEAN_MVS, True, 'carried: clean A/B'),
        ('mvs-da3-base', 'MVS, CERT 0.05 (DA3-BASE start)', CLEAN_MVS, True, 'new'),
        ('mvs-moge', 'MVS, CERT 0.05 (MoGe-3 start)', CLEAN_MVS, True, 'new'),
        ('mvs-da3-base-c0.1', 'MVS, CERT 0.1 (sensitivity)', CLEAN_MVS, True, 'new: sensitivity'),
        ('mvs-da3-base-c0.25', 'MVS, CERT 0.25 (sensitivity)', CLEAN_MVS, True, 'new: sensitivity'),
        ('mvs-da3-base-c0.5', 'MVS, CERT 0.5 re-run (= clean A/B MVS)', CLEAN_MVS, True, 'new: reproduction'),
        ('classic', 'ALIKED + LightGlue poses, focal-sweep K, MoGe-3 depth', 'MIT / Apache-2.0 / BSD-3', True, 'new: superseded diagnostic'),
        ('classic-mogeK', 'same, MoGe-3 K, MoGe-3 own fov (earlier code version)', 'MIT / Apache-2.0 / BSD-3', True, 'new: superseded diagnostic')]
for key, _, lic, _, _ in ROWS:
    fc.LICENCE.setdefault(key, lic)
GATE = dict(maeCm=1.56, maxAbsErrCm=3., floorP95Cm=2.5)
NAMES = ('090R housing', '030L housing', '030R housing', '090R fence')


def load():
    runs = {}
    for d in ANALYSE:
        for f in d.glob('*.json'):
            if not f.name.startswith('spend'):
                r = json.loads(f.read_text()); runs.setdefault((r['cell'], r['backbone'], r['variant']), r)
    return runs


def geometry_dir(b, cell):
    for p in (SCR / f'checks/bbab-geom/{cell}-{b}-padded/geometry', SCR / f'checks/clean-geom/{cell}-{b}/geometry',
              SCR / f'checks/da3fair-geom/{cell}-{b}-padded/geometry'):
        if p.exists():
            return p
    assert b == 'pi3x-published', b
    return CELL_RUN[cell] / 'geometry'


def verdict(r):
    if not r['gatePassedBothCells']:
        return 'no (e-stop gate)'
    if r.get('maeCm') is None:
        return 'no (incomplete: ' + ', '.join(n for n in NAMES if r['values'][n]['cm'] is None) + ')'
    fails = [k for k, ok in (('MAE', r['maeCm'] <= GATE['maeCm']), ('max error', r['maxAbsErrCm'] <= GATE['maxAbsErrCm']),
                             ('floor p95', max(r['floorP95Cm'].values()) <= GATE['floorP95Cm'])) if not ok]
    return 'yes' if not fails else 'no (' + ', '.join(fails) + ')'


def main():
    runs = load(); present = {b for _, b, _ in runs}
    rows = []
    for key, label, lic, is_clean, group in ROWS:
        if key not in present:
            continue
        for rule in fc.RULES:
            for s in ('heightCm', 'verticalPlaneCm'):
                r = clean.safe_config(runs, key, 'padded', rule, s)
                r.update(label=label, licenceClean=is_clean, group=group, passes=verdict(r) if key != 'pi3x-published' else 'reference')
                rows.append(r)
    contract = {key: {c: bb.check_geometry(geometry_dir(key, c), CELL_RUN[c]) for c in ('090', '030')} for key, *_ in ROWS if key in present}
    GE = SCR / 'checks/bbab-geom'
    spend = [json.loads(p.read_text()) for p in sorted(list(GE.glob('spend-*.json')) + list((SCR / 'checks/bbab-analyse').glob('spend-ledger-*.json')))]
    mvs_manifest = {c: json.loads((GE / f'{c}-mvs-da3-base-padded/geometry/candidate_manifest.json').read_text()) for c in ('090', '030')}
    extra = dict(vggtAccess=json.loads((GE / 'vggt-access.json').read_text()),
                 mvsReport={c: m['report'] for c, m in mvs_manifest.items()}, mvsThresholds=mvs_manifest['090']['thresholds'],
                 intrinsicsFx={key: {c: [round(float(np.load(f / 'intrinsics.npy')[0, 0]), 1) for f in sorted((geometry_dir(key, c) / 'frames').iterdir())]
                                     for c in ('090', '030')} for key, *_ in ROWS if key in present},
                 classicDiagnostic=dict(ourKMoGe=json.loads((GE / 'our-k-moge.json').read_text()),
                                        note='ALIKED + LightGlue at 518: 217-572 matches per pair (RoMa: 10 000); superseded by the clean A/B RoMa BA'))
    out = dict(field=fc.fa.FIELD_CM, gate=fc.fa.GATE, passRule=GATE, contract=contract, rows=rows, extra=extra,
               spend=dict(ledgers=spend, totalEstimateUsd=sum(s['estimateUsd'] for s in spend)),
               sources=dict(analyse=[str(p) for p in ANALYSE], fair=str(FAIR / 'results.json'),
                            clean=str(NOTE.parent / 'licence-clean-stack-2026-10-06/geometry/results.json')))
    (NOTE / 'results.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    f1 = lambda x: '—' if x is None else f'{x:.1f}'
    sg = lambda x: '—' if x is None else f'{x:+.1f}'
    print('| backbone | licence | e-stop dev 090 / 030 | ' + ' | '.join(NAMES) + ' | MAE | max | housing range | cam range | floor p95 090 / 030 | pass |')
    for r in rows:
        if r['surface'] == 'heightCm' and r['floor'] == 'maxInlier':
            v = r['values']
            print(f"| {r['label']} | {fc.LICENCE[r['backbone']]} | {r['gate']['090']['maxDeviationPct']:.2f} / {r['gate']['030']['maxDeviationPct']:.2f} % | "
                  + ' | '.join(f"{f1(v[n]['cm'])} ({sg(v[n]['errCm'])})" for n in NAMES)
                  + f" | {f1(r.get('maeCm'))} | {f1(r.get('maxAbsErrCm'))} | {f1(r.get('housingRangeCm'))} | {f1(r['cameraHeightRangeCm'])} | "
                  f"{r['floorP95Cm']['090']:.2f} / {r['floorP95Cm']['030']:.2f} | {r['passes']} |")
    for r in rows:
        if r['backbone'].startswith('mvs') or r['backbone'] in ('pi3x-published', 'da3-large-1.1'):
            print(r['backbone'], r['floor'], r['surface'], 'MAE', f1(r.get('maeCm')), 'max', f1(r.get('maxAbsErrCm')), r['passes'],
                  'cams', [round(x, 1) for x in r['cameraHeightsCm']], 'y/r %+.1f%%' % r['yellowOverRedMeanErrPct'],
                  'vis', {c: round(x, 2) for c, x in r['visibleHeightMedianCm'].items()}, '090L', f1(r['listed090L']))
    for key, cs in contract.items():
        print('contract', key, {c: [(round(x['validFraction'], 2), round(x['reprojMedianPx'], 2)) for x in fr.values()] for c, fr in cs.items()})
    print('spend (this note)', round(out['spend']['totalEstimateUsd'], 3))


if __name__ == '__main__':
    main()
