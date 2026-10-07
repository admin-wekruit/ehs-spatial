"""Verdict engine trial: Scene (scene.py) + derived relations (relations.py) -> clingo facts -> rules.lp -> Verdict records.

  python engine.py <scene.json> <out dir>
Writes verdicts.json (records), verdicts.md (table), facts.lp (what the rules saw). No reconstruction code is imported here.
"""
import json
import math
from pathlib import Path
import sys

import clingo

import relations
from scene import Scene

HERE = Path(__file__).resolve().parent
K = 2                       # expanded uncertainty U = k * sigma_combined
DEFAULT_SIGMA_M = 0.05      # when the layer has no sigma for a dimension (flagged in the record)
DEFAULT_SCALE_REL = 0.02    # when the layer gives no scale uncertainty (flagged)
BLOCKERS, HAZARDS = {'fence', 'guard', 'bollard', 'light_curtain'}, {'robot'}
RULE_TEXT = {
    'fence_height': ('围栏高度 ≥ 1400 mm', 'ISO 13857:2019 表 2 注'),
    'floor_gap': ('防护离地缝 ≤ 180 mm', 'ISO 13857:2019 4.4'),
    'lc_lowest_beam': ('光幕最低光束 ≤ 300 mm', 'ISO 13855（2010 数字）'),
    'crush_gap': ('机器人 ↔ 固定物 间隙 ≥ 500 mm', 'ISO 13854 表 1 / 10218-2（条款待核）'),
    'enclosure': ('危险区被防护物闭合包围', '拓扑规则（占用格可达）'),
}
STATUS = {'pass': 'PASS', 'fail': 'FAIL', 'needs_meas': 'NEEDS_MEASUREMENT', 'cannot_determine': 'CANNOT_DETERMINE', 'open': 'CANNOT_DETERMINE'}


def expanded_u(value_m, sigmas_m, rel):
    """U (metres, k=K) from the 1-sigma terms that enter the measurand plus the relative scale term; flags defaults used."""
    flags = []
    terms = []
    for s in sigmas_m:
        if s is None:
            terms.append(DEFAULT_SIGMA_M); flags.append('default_sigma')
        else:
            terms.append(s)
    if rel is None:
        rel = DEFAULT_SCALE_REL; flags.append('default_scale_unc')
    return K * math.sqrt(sum(t * t for t in terms) + (rel * abs(value_m)) ** 2), sorted(set(flags))


def mm(x):
    return int(round(x * 1000))


def facts_for(scene):
    rel = scene.scale_rel_unc
    lines, meta = [], {}
    q = lambda s: '"' + s + '"'
    for o in scene.objects:
        lines.append(f'obj({q(o.id)},{o.cls}).')
        ub, fb = expanded_u(o.bottom_m, [o.sigma_m.get('bottom')], rel)
        ut, ft = expanded_u(o.top_m, [o.sigma_m.get('bottom'), o.sigma_m.get('H')], rel)
        lines += [f'bottom({q(o.id)},{mm(o.bottom_m)},{mm(ub)}).', f'top({q(o.id)},{mm(o.top_m)},{mm(ut)}).']
        meta[o.id] = {'bottom': (o.bottom_m, ub, fb), 'top': (o.top_m, ut, ft)}
    pairs = relations.all_pairs(scene, 'robot', BLOCKERS)
    for (a, b), d in relations.pair_distances(scene, pairs).items():
        oa, ob = relations.scene_obj(scene, a), relations.scene_obj(scene, b)
        ud, fd = expanded_u(d, [max((s for s in (oa.sigma_m.get('L'), oa.sigma_m.get('W')) if s is not None), default=None),
                                max((s for s in (ob.sigma_m.get('L'), ob.sigma_m.get('W')) if s is not None), default=None)], rel)
        lines.append(f'dist({q(a)},{q(b)},{mm(d)},{mm(ud)}).')
        meta[(a, b)] = {'dist': (d, ud, fd)}
    grid = relations.plan_grid(scene, BLOCKERS, HAZARDS)
    c = lambda xy: f'c({xy[0]},{xy[1]})'
    lines += [f'cell({c(xy)}).' for xy in grid['cells']]
    lines += [f'adj({c(a)},{c(b)}).' for a, b in grid['adj']]
    lines += [f'blocked({c(xy)}).' for xy in sorted(grid['blocked'])]
    lines += [f'hazard({c(xy)}).' for xy in sorted(grid['hazard'])]
    lines += [f'outside({c(xy)}).' for xy in sorted(grid['outside'])]
    return '\n'.join(lines) + '\n', meta, grid


def solve(facts):
    ctl = clingo.Control(['--warn=none'])
    ctl.add('base', [], (HERE / 'rules.lp').read_text() + facts)
    ctl.ground([('base', [])])
    atoms = []
    ctl.solve(on_model=lambda m: atoms.extend(m.symbols(shown=True)))
    return atoms


def subject_ids(sym):
    return [str(a).strip('"') for a in (sym.arguments if sym.type == clingo.SymbolType.Function and sym.name == '' else [sym])]


def records(scene, atoms, meta, grid):
    byid = {o.id: o for o in scene.objects}
    margins = {(str(a.arguments[0]), str(a.arguments[1])): a.arguments[2].number for a in atoms if a.name == 'margin'}
    openings = [(a.arguments[0].arguments[0].number, a.arguments[0].arguments[1].number) for a in atoms if a.name == 'opening']
    out = []
    for a in sorted((a for a in atoms if a.name == 'status'), key=str):
        rule, subj, st = str(a.arguments[0]), a.arguments[1], str(a.arguments[2])
        ids = subject_ids(subj)
        rec = {'rule': rule, 'rule_text': RULE_TEXT[rule][0], 'source': RULE_TEXT[rule][1], 'status': STATUS[st], 'subjects': ids,
               'labels': [byid[i].label for i in ids if i in byid], 'notes': []}
        if rule == 'enclosure':
            if st == 'open':
                e1, e2 = grid['basis']; lo, cm = grid['origin'], grid['cell_m']
                hz = [(lo[0] + (x + .5) * cm, lo[1] + (y + .5) * cm) for x, y in grid['hazard']]
                cx, cy = sum(p[0] for p in hz) / len(hz), sum(p[1] for p in hz) / len(hz)
                dirs = sorted({('+e2' if (lo[1] + (y + .5) * cm) - cy > abs((lo[0] + (x + .5) * cm) - cx) else '-e2' if cy - (lo[1] + (y + .5) * cm) > abs((lo[0] + (x + .5) * cm) - cx)
                                else '+e1' if (lo[0] + (x + .5) * cm) > cx else '-e1') for x, y in openings})
                rec['notes'].append(f'外部可达危险区：{len(openings)} 个开口格，方向 {dirs}（平面基 e1/e2）；照片未覆盖的区域和真实开口在这一层分不开 → 不下结论')
                rec['openings'] = len(openings); rec['open_directions'] = dirs
            rec['notes'].append('危险区 = 机器人姿态盒的平面占用，不是受限空间（需控制器配置）')
            out.append(rec); continue
        key = ids[0] if len(ids) == 1 else (ids[0], ids[1])
        m = meta.get(key, {})
        which = {'fence_height': 'top', 'floor_gap': 'bottom', 'lc_lowest_beam': 'bottom', 'crush_gap': 'dist'}[rule]
        if which in m:
            v, u, flags = m[which]
            rec.update(measured_mm=mm(v), U_mm=mm(u), threshold_mm={'fence_height': 1400, 'floor_gap': 180, 'lc_lowest_beam': 300, 'crush_gap': 500}[rule],
                       margin_mm=margins.get((rule, str(subj))), uncertainty_flags=flags,
                       confidence=[byid[i].confidence for i in ids if i in byid], evidence_views=sorted({p for i in ids if i in byid for p in byid[i].views}))
            if 'default_sigma' in flags:
                rec['notes'].append(f'该维度无 σ，用默认 {DEFAULT_SIGMA_M * 100:.0f} cm')
            if 'default_scale_unc' in flags:
                rec['notes'].append(f'尺度无 ±%，用默认 {DEFAULT_SCALE_REL * 100:.0f} %')
        if rule == 'crush_gap':
            rec['notes'].append('机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间')
        out.append(rec)
    return out


def table(recs):
    rows = ['| 规则 | 对象 | 实测 ± U (mm) | 阈值 | 判定 | 裕量 (mm) | 置信度 | 备注 |', '|---|---|---|---|---|---|---|---|']
    for r in recs:
        meas = f"{r['measured_mm']} ± {r['U_mm']}" if 'measured_mm' in r else '—'
        thr = str(r.get('threshold_mm', '—'))
        rows.append(f"| {r['rule_text']} | {' ↔ '.join(r['labels']) or r['subjects'][0]} | {meas} | {thr} | **{r['status']}** | {r.get('margin_mm', '—')} | {','.join(r.get('confidence', [])) or '—'} | {'；'.join(r['notes'])} |")
    return '\n'.join(rows)


def main(scene_path, out_dir):
    scene = Scene.load(scene_path)
    facts, meta, grid = facts_for(scene)
    atoms = solve(facts)
    recs = records(scene, atoms, meta, grid)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'facts.lp').write_text(facts)
    (out / 'verdicts.json').write_text(json.dumps({'schema': 'panoptes.verdict/0-trial', 'scene_id': scene.scene_id, 'rule_pack': 'rules.lp v0 2026-10-07',
                                                   'decision_rule': f'guard_band_U_k{K}', 'verdicts': recs}, indent=1, ensure_ascii=False))
    md = f"# {scene.scene_id}: {len(recs)} verdicts, grid {grid['shape']} cells of {grid['cell_m']} m, blocked {len(grid['blocked'])}, hazard {len(grid['hazard'])}\n\n" + table(recs) + '\n'
    (out / 'verdicts.md').write_text(md)
    counts = {}
    for r in recs:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print(md); print('counts:', counts)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
