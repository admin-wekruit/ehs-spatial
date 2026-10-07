"""Aggregate the fair A/B: SCR/checks/da3fair-analyse/*.json + pi3x-on-model.json -> results.json, and print the README tables.

Per run (cell, backbone, input variant, floor rule): e-stop gate, housing / fence heights against the field values, camera
heights, floor flatness, e-stop ratio and visible height. A configuration (backbone, variant, floor) is scored on both cells
together; any run failing the 4 % e-stop gate is dropped from every aggregate (none failed).
"""
import json
import os
from pathlib import Path
import statistics as st
import sys

NOTE = Path(__file__).resolve().parent
SCR = Path(os.environ.get('SWAP_SCRATCH', '/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad'))
sys.path.insert(0, str(NOTE))
import fair_ab as fa  # noqa: E402

A = SCR / 'checks/da3fair-analyse'
BACKBONES = [('pi3x-published', 'padded'), ('pi3x', 'padded'), ('pi3x', 'unpadded'), ('da3-large-1.1', 'padded'), ('da3-large-1.1', 'unpadded'),
             ('da3-base', 'padded'), ('da3-base', 'unpadded')]
LICENCE = {'pi3x-published': 'CC-BY-NC-4.0', 'pi3x': 'CC-BY-NC-4.0',
           'da3-large-1.1': 'disputed: HF card apache-2.0, official README CC BY-NC 4.0', 'da3-base': 'Apache-2.0 (card and README)'}
RULES = ('maxInlier', 'lsq', 'lowest')
SCORED = [('090', '169518d8', 'housing'), ('030', '606109af', 'housing'), ('030', 'd72e25ef', 'housing'), ('090', 'ce9516a5', 'fence')]
NAMES = {'169518d8': '090R housing', '606109af': '030L housing', 'd72e25ef': '030R housing', 'ce9516a5': '090R fence', '5163a9b0': '090L housing (listed)'}


def load():
    runs = {}
    for f in A.glob('*.json'):
        if not f.name.startswith('spend'):
            r = json.loads(f.read_text()); runs[(r['cell'], r['backbone'], r['variant'])] = r
    return runs


def config(runs, bb, var, rule, surface='heightCm'):
    cells = {c: runs[(c, bb, var)]['floors'][rule] for c in ('090', '030')}
    gate = {c: dict(passed=f['gatePassed'], maxDeviationPct=100 * f['estop']['maxDeviation'], scale=f['estop']['nativeToMeters']) for c, f in cells.items()}
    ok = all(g['passed'] for g in gate.values())
    vals = {}
    for c, eid, kind in SCORED:
        f = cells[c]
        v = f['housing'][eid][surface] if kind == 'housing' else f['fence'][eid]['heightCm']
        vals[NAMES[eid]] = dict(cm=v, errCm=None if v is None else v - fa.FIELD_CM[kind])
    listed = cells['090']['housing']['5163a9b0'][surface]
    errs = [abs(x['errCm']) for x in vals.values() if x['errCm'] is not None]
    hous = [vals[n]['cm'] for n in ('090R housing', '030L housing', '030R housing') if vals[n]['cm'] is not None]
    cams = cells['090']['cameraHeightsCm'] + cells['030']['cameraHeightsCm']
    yr = [p['yellowOverRed'] for c in cells.values() for p in c['estop']['perPhoto'].values()]
    vis = {c: st.median([p['visibleHeightCm']['mid'] for p in f['estop']['perPhoto'].values() if 'mid' in p['visibleHeightCm']]) for c, f in cells.items()}
    row = dict(backbone=bb, variant=var, floor=rule, surface=surface, licence=LICENCE[bb], gate=gate, gatePassedBothCells=ok, values=vals,
               listed090L=listed, floorP95Cm={c: f['residualP95Cm'] for c, f in cells.items()},
               floorAngleToLowestDeg={c: f['angleToLowestDeg'] for c, f in cells.items()},
               cameraHeightsCm=cams, cameraHeightRangeCm=max(cams) - min(cams), yellowOverRed=yr,
               yellowOverRedMeanErrPct=100 * (st.mean(yr) / fa.FIELD_CM['yellowOverRed'] - 1), visibleHeightMedianCm=vis)
    if ok:  # aggregates only for runs inside the e-stop gate
        row.update(maeCm=st.mean(errs) if len(errs) == 4 else None, maxAbsErrCm=max(errs) if errs else None,
                   housingMaeCm=st.mean(abs(vals[n]['errCm']) for n in ('090R housing', '030L housing', '030R housing')),
                   housingRangeCm=max(hous) - min(hous) if len(hous) == 3 else None)
    return row


def published_reference(on_model):
    """As published (not under the fair rules): Pi3X, report floors (090 = measurement-layer ground), published scales,
    housing rays on the displayed models (workcell-housing-edge), fence = clearance-B two-view 20.47 cm."""
    v = {'090R housing': on_model['090 169518d8']['heightCm']['report'], '030L housing': on_model['030 606109af']['heightCm']['report'],
         '030R housing': on_model['030 d72e25ef']['heightCm']['report'], '090R fence': 20.47}
    e = {k: x - (20. if 'fence' in k else 24.) for k, x in v.items()}
    return dict(values={k: dict(cm=x, errCm=e[k]) for k, x in v.items()}, maeCm=st.mean(map(abs, e.values())), maxAbsErrCm=max(map(abs, e.values())))


def main():
    runs = load(); on_model = json.loads((NOTE / 'pi3x-on-model.json').read_text())
    rows = [config(runs, bb, var, rule, s) for bb, var in BACKBONES for rule in RULES for s in ('heightCm', 'verticalPlaneCm')]
    # Pi3X on-model (its displayed models) on the same neutral floors and scales, for reference
    onm = {}
    for rule in RULES:
        v = {NAMES[k.split()[1]]: d['heightCm'][rule] for k, d in on_model.items() if d['group'] == 'housing'}
        fen = runs[('090', 'pi3x-published', 'padded')]['floors'][rule]['fence']['ce9516a5']['heightCm']
        e = [abs(x - 24.) for x in v.values()] + [abs(fen - 20.)]
        onm[rule] = dict(housingCm=v, fenceCm=fen, maeCm=st.mean(e), maxAbsErrCm=max(e))
    gpu = json.loads((SCR / 'checks/da3fair-geom/spend-ledger.json').read_text())
    cpu = [json.loads(p.read_text()) for p in A.glob('spend-ledger-*.json')]
    reproduction = dict(
        pi3xPublishedLowestRule={c: dict(scale=runs[(c, 'pi3x-published', 'padded')]['floors']['lowest']['estop']['nativeToMeters'],
                                         cameraHeightsCm=runs[(c, 'pi3x-published', 'padded')]['floors']['lowest']['cameraHeightsCm'],
                                         reportCameraMaxAbs=runs[(c, 'pi3x-published', 'padded')]['vsReportCameras']) for c in ('090', '030')},
        expected=dict(scale={'090': 1.28655, '030': 1.1166239729}, cameraHeightsCm={'090': [155.7, 154.8, 156.7], '030': [152.2, 154.3]},
                      source='geometry-licence-ab-2026-10-05 pipeline column; published 030 scale; estop-dimensions-2026-10-05 ratios / heights'),
        pi3xOnModelReportFloor={k: d['heightCm']['report'] for k, d in on_model.items()},
        expectedOnModel={'090 169518d8': 24.0, '030 606109af': 26.2, '030 d72e25ef': 22.5, 'source': 'workcell-housing-edge-2026-10-05'})
    out = dict(field=fa.FIELD_CM, gate=fa.GATE, params={k: v for k, v in fa.P.items()}, pinned=fa.PIN, rows=rows, pi3xOnModel=onm,
               publishedReference=published_reference(on_model), reproduction=reproduction,
               spend=dict(gpu=gpu, cpu=cpu, totalEstimateUsd=gpu['estimateUsd'] + sum(c['estimateUsd'] for c in cpu)))
    (NOTE / 'results.json').write_text(json.dumps(out, indent=1, default=float) + '\n')
    # ---- tables
    f1 = lambda x: '—' if x is None else f'{x:.1f}'
    sg = lambda x: '—' if x is None else f'{x:+.1f}'
    print('| backbone | input | floor | licence | e-stop dev 090 / 030 | 090R housing | 030L housing | 030R housing | 090R fence | MAE (4) | max abs | housing range | cam range (5) | floor p95 090 / 030 |')
    for r in rows:
        if r['surface'] != 'heightCm':
            continue
        v = r['values']
        print(f"| {r['backbone']} | {r['variant']} | {r['floor']} | {r['licence'].split(':')[0]} | {r['gate']['090']['maxDeviationPct']:.2f} / {r['gate']['030']['maxDeviationPct']:.2f} % | "
              + ' | '.join(f"{f1(v[n]['cm'])} ({sg(v[n]['errCm'])})" for n in ('090R housing', '030L housing', '030R housing', '090R fence'))
              + f" | {f1(r.get('maeCm'))} | {f1(r.get('maxAbsErrCm'))} | {f1(r.get('housingRangeCm'))} | {f1(r['cameraHeightRangeCm'])} | {r['floorP95Cm']['090']:.2f} / {r['floorP95Cm']['030']:.2f} |")
    print()
    for r in rows:
        if r['surface'] == 'verticalPlaneCm' and r['floor'] == 'maxInlier':
            print(r['backbone'], r['variant'], 'vertical-plane MAE', f1(r.get('maeCm')), 'housing', {n: f1(r['values'][n]['cm']) for n in ('090R housing', '030L housing', '030R housing')})
    print('on-model', {k: (round(v['maeCm'], 2), {n: round(x, 1) for n, x in v['housingCm'].items()}) for k, v in onm.items()})
    print('published', out['publishedReference'])
    for r in rows:
        if r['surface'] == 'heightCm' and r['floor'] == 'maxInlier':
            print(r['backbone'], r['variant'], 'cams', [round(x, 1) for x in r['cameraHeightsCm']], 'y/r', [round(x, 3) for x in r['yellowOverRed']],
                  'vis', {c: round(x, 2) for c, x in r['visibleHeightMedianCm'].items()}, '090L', f1(r['listed090L']))
    print('spend', round(out['spend']['totalEstimateUsd'], 3))


if __name__ == '__main__':
    main()
