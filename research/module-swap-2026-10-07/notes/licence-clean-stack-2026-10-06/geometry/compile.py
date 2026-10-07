"""Aggregate the licence-clean geometry A/B -> results.json + README tables.

Runs: SCR/checks/clean-analyse/*.json (new candidates, modal_apps/geometry_clean_ab.py) next to the fair A/B's own runs
(SCR/checks/da3fair-analyse: published Pi3X, its CUDA re-run, DA3-BASE, DA3-LARGE-1.1). Scoring is the fair A/B's compile.config,
unchanged: both cells together, any run outside the 4 % e-stop gate is dropped from every aggregate.
"""
import json
import os
from pathlib import Path
import sys

NOTE = Path(__file__).resolve().parent
FAIR = Path(os.environ.get('SWAP_NOTES', '/Users/adam/Desktop/panoptes-public/research-notes')) / 'geometry-licence-ab-fair-2026-10-05'
sys.path.insert(0, str(FAIR))
import compile as fc  # noqa: E402  fair A/B scoring (config), unchanged

SCR = fc.SCR
NEW, GEOM, GPUD = SCR / 'checks/clean-analyse', SCR / 'checks/clean-geom', SCR / 'checks/clean-gpu'
NOISE_CM = None  # set from the data: |MAE(Pi3X CUDA re-run) - MAE(Pi3X published, MPS)|
REFINE = 'RoMa outdoor (MIT) matches + pycolmap 4.2.1 BA (BSD-3)'
MVS = REFINE + ' + RoMa dense warp, two-view triangulation'
FUSED = MVS + ' where triangulated, else backbone depth x smoothed MVS ratio'
CANDIDATES = [  # (backbone key, variant, licence, role)
    ('pi3x-published', 'padded', 'CC-BY-NC-4.0', 'reference (published)'),
    ('pi3x', 'padded', 'CC-BY-NC-4.0', 'noise: CUDA re-run of the same weights'),
    ('pi3x-ba', 'padded', f'CC-BY-NC-4.0; {REFINE}', 'control: what BA does to Pi3X'),
    ('pi3x-ba-f', 'padded', f'CC-BY-NC-4.0; {REFINE}', 'control'),
    ('da3-large-1.1', 'padded', 'disputed (HF card Apache-2.0, README CC BY-NC 4.0)', 'reference (disputed)'),
    ('da3-base', 'padded', 'Apache-2.0', 'clean, feed-forward'),
    ('da3-base-ba', 'padded', f'Apache-2.0; {REFINE}', 'clean'),
    ('da3-base-ba-f', 'padded', f'Apache-2.0; {REFINE}', 'clean'),
    ('mapanything-raw', 'padded', 'Apache-2.0 (map-anything-apache)', 'clean, feed-forward'),
    ('mapanything-ba', 'padded', f'Apache-2.0; {REFINE}', 'clean'),
    ('mapanything-ba-f', 'padded', f'Apache-2.0; {REFINE}', 'clean'),
    ('moge', 'padded', f'MIT (MoGe-3); RoMa MIT; sim(3) only', 'clean, per photo, no BA'),
    ('moge-ba', 'padded', f'MIT (MoGe-3); {REFINE}', 'clean'),
    ('moge-ba-f', 'padded', f'MIT (MoGe-3); {REFINE}', 'clean'),
    ('da3-base-ba-f-mvs', 'padded', f'Apache-2.0 cameras only; {MVS}', 'clean, triangulated surface'),
    ('mapanything-ba-f-mvs', 'padded', f'Apache-2.0 cameras only; {MVS}', 'clean, triangulated surface'),
    ('moge-ba-f-mvs', 'padded', f'MIT cameras only; {MVS}', 'clean, triangulated surface'),
    ('pi3x-ba-f-mvs', 'padded', f'CC-BY-NC-4.0 cameras; {MVS}', 'control'),
    ('da3-base-ba-f-fused', 'padded', f'Apache-2.0; {FUSED}', 'clean, MVS + backbone fill'),
    ('mapanything-ba-f-fused', 'padded', f'Apache-2.0; {FUSED}', 'clean, MVS + backbone fill'),
    ('moge-ba-f-fused', 'padded', f'MIT (MoGe-3); {FUSED}', 'clean, MVS + backbone fill'),
    ('pi3x-ba-f-fused', 'padded', f'CC-BY-NC-4.0; {FUSED}', 'control'),
]


def load():
    runs = fc.load()  # the fair A/B's runs (fc.A = da3fair-analyse)
    for f in NEW.glob('*.json'):
        if not f.name.startswith('spend'):
            r = json.loads(f.read_text()); runs[(r['cell'], r['backbone'], 'padded')] = r
    return runs


def refine_summary(bb):
    """BA report per cell from the refined geometry's manifest (focal, depth rescale, pose change, reprojection)."""
    out = {}
    for c in ('090', '030'):
        m = GEOM / f'{c}-{bb}' / 'geometry' / 'candidate_manifest.json'
        if not m.exists():
            continue
        rep = json.loads(m.read_text()).get('report') or {}
        if 'photos' not in rep:
            out[c] = rep if bb != 'moge' else json.loads(m.read_text()).get('report'); continue
        out[c] = dict(pairs=rep['pairs'], reprojPxFinal=rep['ba'][-1]['reprojPx'], pointsFinal=rep['ba'][-1]['points'], droppedFirstPass=rep['ba'][0]['dropped'],
                      focal=[[round(x, 1) for x in p['focalBefore']] + [round(p['focalAfter'][0], 1)] for p in rep['photos']],
                      depthScale=[round(p['depthScale'], 4) for p in rep['photos']], depthRatioIqr=[p['depthRatioIqr'] for p in rep['photos']],
                      depthRatioLogSlope=[round(p['depthRatioLogSlope'], 3) for p in rep['photos']],
                      poseChange={k: {q: round(v, 3) for q, v in d.items()} for k, d in rep['poseChange'].items()})
    return out


def safe_config(runs, bb, var, rule, s):
    """fc.config; when a scored value is missing (too few surface points) the row keeps its values but no aggregates."""
    try:
        return fc.config(runs, bb, var, rule, s)
    except TypeError:
        import copy
        tmp = {k: copy.deepcopy(v) if k[1] == bb else v for k, v in runs.items()}
        for c in ('090', '030'):
            tmp[(c, bb, var)]['floors'][rule]['gatePassed'] = False
        row = fc.config(tmp, bb, var, rule, s)
        for c in ('090', '030'):
            row['gate'][c]['passed'] = runs[(c, bb, var)]['floors'][rule]['gatePassed']
        row['gatePassedBothCells'] = all(g['passed'] for g in row['gate'].values())
        errs = [abs(x['errCm']) for x in row['values'].values() if x['errCm'] is not None]
        row.update(incomplete=True, maeAvailableCm=sum(errs) / len(errs), nAvailable=len(errs))
        return row


def main():
    runs = load()
    for bb, _, lic, _ in CANDIDATES:
        fc.LICENCE.setdefault(bb, lic)
    have = [c for c in CANDIDATES if all((cell, c[0], c[1]) in runs for cell in ('090', '030'))]
    rows = [dict(safe_config(runs, bb, var, rule, s), role=role, licence=lic) for bb, var, lic, role in have
            for rule in fc.RULES for s in ('heightCm', 'verticalPlaneCm')]
    prim = {r['backbone']: r for r in rows if r['floor'] == 'maxInlier' and r['surface'] == 'heightCm'}
    noise = abs(prim['pi3x']['maeCm'] - prim['pi3x-published']['maeCm'])
    ref = prim['pi3x-published']['maeCm']
    for r in rows:
        if r.get('maeCm') is not None:
            r['maeMinusPi3xCm'] = r['maeCm'] - ref
            r['atLeastAsAccurateAsPi3x'] = bool(r['maeCm'] <= ref + noise)
    access = json.loads((NOTE / 'vggt-access.json').read_text()) if (NOTE / 'vggt-access.json').exists() else None
    led = [json.loads(p.read_text()) for p in [*GPUD.glob('spend-ledger*.json'), *GEOM.glob('spend-ledger-*.json'), *NEW.glob('spend-ledger-*.json')]]
    out = dict(field=fc.fa.FIELD_CM, gate=fc.fa.GATE, noiseCm=noise, pi3xPublishedMaeCm=ref, rule='at least as accurate = MAE <= Pi3X published MAE + noise',
               vggtAccess=access, rows=rows, refinement={bb: refine_summary(bb) for bb, *_ in have if bb.endswith(('-ba', '-ba-f', '-mvs')) or bb == 'moge'},
               spend=dict(ledgers=led, totalEstimateUsd=sum(x['estimateUsd'] for x in led),
                          note='list-rate estimate of function seconds; the fair A/B runs reused here were paid for in geometry-licence-ab-fair-2026-10-05'))
    (NOTE / 'results.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    f1 = lambda x: '—' if x is None else f'{x:.1f}'
    sg = lambda x: '—' if x is None else f'{x:+.1f}'
    print(f'noise {noise:.3f} cm, Pi3X published MAE {ref:.3f}')
    print('| backbone | licence | e-stop dev 090 / 030 (gate) | 090R housing | 030L housing | 030R housing | 090R fence | MAE (4) | worst | housing range | cam range (5) | floor p95 090 / 030 | >= Pi3X? |')
    for r in rows:
        if r['floor'] != 'maxInlier' or r['surface'] != 'heightCm':
            continue
        v = r['values']; g = r['gate']
        print(f"| {r['backbone']} | {r['licence'].split(';')[0]} | {g['090']['maxDeviationPct']:.2f} / {g['030']['maxDeviationPct']:.2f} % ({'pass' if r['gatePassedBothCells'] else 'FAIL'}) | "
              + ' | '.join(f"{f1(v[n]['cm'])} ({sg(v[n]['errCm'])})" for n in ('090R housing', '030L housing', '030R housing', '090R fence'))
              + f" | {f1(r.get('maeCm'))} | {f1(r.get('maxAbsErrCm'))} | {f1(r.get('housingRangeCm'))} | {f1(r['cameraHeightRangeCm'])} | "
              f"{r['floorP95Cm']['090']:.2f} / {r['floorP95Cm']['030']:.2f} | {'yes' if r.get('atLeastAsAccurateAsPi3x') else ('incomplete' if r.get('incomplete') else 'no')} |")
    print()
    for rule in ('lsq', 'lowest'):
        print(rule, {r['backbone']: (f1(r.get('maeCm')), f1(r.get('maxAbsErrCm'))) for r in rows if r['floor'] == rule and r['surface'] == 'heightCm'})
    print('vertical', {r['backbone']: f1(r.get('maeCm')) for r in rows if r['floor'] == 'maxInlier' and r['surface'] == 'verticalPlaneCm'})
    for r in rows:
        if r['floor'] == 'maxInlier' and r['surface'] == 'heightCm':
            print(r['backbone'], 'cams', [round(x, 1) for x in r['cameraHeightsCm']], 'y/r err %', round(r['yellowOverRedMeanErrPct'], 2),
                  'vis', {c: round(x, 2) for c, x in r['visibleHeightMedianCm'].items()}, '090L', f1(r['listed090L']),
                  'scale', {c: round(g['scale'], 5) for c, g in r['gate'].items()})
    for bb, s in out['refinement'].items():
        print(bb, json.dumps({c: {k: x[k] for k in ('focal', 'depthScale', 'depthRatioLogSlope', 'reprojPxFinal', 'poseChange') if k in x} for c, x in s.items()}))
    print('spend', round(out['spend']['totalEstimateUsd'], 3))


if __name__ == '__main__':
    main()
