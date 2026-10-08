"""Unchanged field scoring for the two capture configurations."""
from argus import ROOT
import json
import os
from pathlib import Path
import statistics as st
import sys
import argus.checks.field_geometry as fa
SCORED = [('090', '169518d8', 'housing'), ('030', '606109af', 'housing'), ('030', 'd72e25ef', 'housing'), ('090', 'ce9516a5', 'fence')]
NAMES = {'169518d8': '090R housing', '606109af': '030L housing', 'd72e25ef': '030R housing', 'ce9516a5': '090R fence', '5163a9b0': '090L housing (listed)'}

LICENCE = {'da3-base': 'Apache-2.0'}

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
