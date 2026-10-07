"""Verdict engine trial: scene graph (scene_graph.py JSON) -> clingo facts -> rules.lp -> Verdict records.

  python engine.py <scene-graph.json> <out dir>
Writes verdicts.json (records), verdicts.md (table), facts.lp (what the rules saw). The engine reads the graph only: no geometry
is computed here and no reconstruction code is imported.
"""
import json
from pathlib import Path
import sys

import clingo

from scene_graph import FIXED, HAZARD, u_mm

HERE = Path(__file__).resolve().parent
RULE_TEXT = {
    'fence_height': ('围栏高度 ≥ 1400 mm', 'ISO 13857:2019 表 2 注'),
    'floor_gap': ('防护离地缝 ≤ 180 mm', 'ISO 13857:2019 4.4'),
    'lc_lowest_beam': ('光幕最低光束 ≤ 300 mm', 'ISO 13855（2010 数字）'),
    'crush_gap': ('机器人 ↔ 固定物 间隙 ≥ 500 mm', 'ISO 13854 表 1 / 10218-2（条款待核）'),
    'reach_over': ('越过防护够到：a 危险高 / b 防护高 / c 水平距离', 'ISO 13857:2019 表 2（查表值待对正文核）'),
    'enclosure': ('危险区被防护物闭合包围', '拓扑规则（占用格可达）'),
}
THRESHOLD = {'fence_height': 1400, 'floor_gap': 180, 'lc_lowest_beam': 300, 'crush_gap': 500}
STATUS = {'pass': 'PASS', 'fail': 'FAIL', 'needs_meas': 'NEEDS_MEASUREMENT', 'cannot_determine': 'CANNOT_DETERMINE', 'open': 'CANNOT_DETERMINE',
          'needs_input': 'NEEDS_INPUT'}


def facts_for(graph):
    q = lambda s: '"' + s + '"'
    nodes = {n['id']: n for n in graph['nodes'] if n['cls'] != 'floor'}
    lines, meta = [], {}
    for nid, n in nodes.items():
        lines.append(f'obj({q(nid)},{n["cls"]}).')
        ub, fb = u_mm(n['bottom_m'], [n['sigma_m'].get('bottom')], None if 'scale_rel_unc' not in graph else graph['scale_rel_unc'])
        ut, ft = u_mm(n['top_m'], [n['sigma_m'].get('bottom'), n['sigma_m'].get('H')], None if 'scale_rel_unc' not in graph else graph['scale_rel_unc'])
        lines += [f'bottom({q(nid)},{round(n["bottom_m"] * 1000)},{ub}).', f'top({q(nid)},{round(n["top_m"] * 1000)},{ut}).']
        meta[nid] = {'bottom': (round(n['bottom_m'] * 1000), ub, fb, n['views']), 'top': (round(n['top_m'] * 1000), ut, ft, n['views'])}
    for e in graph['edges']:
        if e['type'] == 'min_distance_3d' and {nodes[e['a']]['cls'] in HAZARD, nodes[e['b']]['cls'] in HAZARD} == {True, False}:
            r, x = (e['a'], e['b']) if nodes[e['a']]['cls'] in HAZARD else (e['b'], e['a'])
            if nodes[x]['cls'] in FIXED:
                lines.append(f'dist({q(r)},{q(x)},{e["value_mm"]},{e["U_mm"]}).')
                meta[(r, x)] = {'dist': (e['value_mm'], e['U_mm'], e['flags'], e['views'])}
        if e['type'] == 'reach_over':
            lines.append(f'reach_over({q(e["a"])},{q(e["b"])},{e["a_mm"]},{e["b_mm"]},{e["c_mm"]},{max(e["a_U_mm"], e["b_U_mm"], e["c_U_mm"])}).')
            meta[(e['a'], e['b'])] = {**meta.get((e['a'], e['b']), {}), 'reach_over': (e['a_mm'], e['a_U_mm'], e['b_mm'], e['b_U_mm'], e['c_mm'], e['c_U_mm'], e['views'])}
    occ = graph['plan_occupancy']
    nx, ny = occ['shape']
    c = lambda xy: f'c({xy[0]},{xy[1]})'
    lines += [f'cell({c((x, y))}).' for x in range(nx) for y in range(ny)]
    lines += [f'adj({c((x, y))},{c((x + dx, y + dy))}).' for x in range(nx) for y in range(ny) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 0 <= x + dx < nx and 0 <= y + dy < ny]
    lines += [f'blocked({c(xy)}).' for xy in occ['blocked']]
    lines += [f'hazard({c(xy)}).' for xy in occ['hazard']]
    lines += [f'outside({c(xy)}).' for xy in occ['outside']]
    return '\n'.join(lines) + '\n', meta, nodes


def solve(facts):
    ctl = clingo.Control(['--warn=none'])
    ctl.add('base', [], (HERE / 'rules.lp').read_text() + facts)
    ctl.ground([('base', [])])
    atoms = []
    ctl.solve(on_model=lambda m: atoms.extend(m.symbols(shown=True)))
    return atoms


def subject_ids(sym):
    return [str(a).strip('"') for a in (sym.arguments if sym.type == clingo.SymbolType.Function and sym.name == '' else [sym])]


def records(graph, atoms, meta, nodes):
    margins = {(str(a.arguments[0]), str(a.arguments[1])): a.arguments[2].number for a in atoms if a.name == 'margin'}
    openings = [(a.arguments[0].arguments[0].number, a.arguments[0].arguments[1].number) for a in atoms if a.name == 'opening']
    occ = graph['plan_occupancy']
    out = []
    for a in sorted((a for a in atoms if a.name == 'status'), key=str):
        rule, subj, st = str(a.arguments[0]), a.arguments[1], str(a.arguments[2])
        ids = subject_ids(subj)
        rec = {'rule': rule, 'rule_text': RULE_TEXT[rule][0], 'source': RULE_TEXT[rule][1], 'status': STATUS[st], 'subjects': ids,
               'labels': [nodes[i]['label'] for i in ids if i in nodes], 'notes': []}
        if rule == 'enclosure':
            if st == 'open':
                lo, cm = occ['origin'], occ['cell_m']
                hz = [(lo[0] + (x + .5) * cm, lo[1] + (y + .5) * cm) for x, y in occ['hazard']]
                cx, cy = sum(p[0] for p in hz) / len(hz), sum(p[1] for p in hz) / len(hz)
                dirs = sorted({('+e2' if (lo[1] + (y + .5) * cm) - cy > abs((lo[0] + (x + .5) * cm) - cx) else '-e2' if cy - (lo[1] + (y + .5) * cm) > abs((lo[0] + (x + .5) * cm) - cx)
                                else '+e1' if (lo[0] + (x + .5) * cm) > cx else '-e1') for x, y in openings})
                rec['notes'].append(f'外部可达危险区：{len(openings)} 个开口格，方向 {dirs}；观察范围未提供（{graph["coverage"]["observed_floor"]}）→ 不下结论')
                rec['openings'] = len(openings); rec['open_directions'] = dirs
            rec['notes'].append('危险区 = 机器人姿态盒的平面占用，不是受限空间（需控制器配置）')
            out.append(rec); continue
        key = ids[0] if len(ids) == 1 else (ids[0], ids[1])
        m = meta.get(key, {})
        if rule == 'reach_over':
            a_, au, b_, bu, c_, cu, views = m['reach_over']
            rec.update(inputs={'a_hazard_top_mm': [a_, au], 'b_structure_top_mm': [b_, bu], 'c_horizontal_mm': [c_, cu]}, evidence_views=views,
                       confidence=[nodes[i]['confidence'] for i in ids])
            rec['notes'].append('三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定')
            out.append(rec); continue
        which = {'fence_height': 'top', 'floor_gap': 'bottom', 'lc_lowest_beam': 'bottom', 'crush_gap': 'dist'}[rule]
        if which in m:
            v, u, flags, views = m[which]
            rec.update(measured_mm=v, U_mm=u, threshold_mm=THRESHOLD[rule], margin_mm=margins.get((rule, str(subj))), uncertainty_flags=flags,
                       confidence=[nodes[i]['confidence'] for i in ids if i in nodes], evidence_views=views)
            if 'default_sigma' in flags:
                rec['notes'].append('该维度无 σ，用默认 5 cm')
            if 'default_scale_unc' in flags:
                rec['notes'].append('尺度无 ±%，用默认 2 %')
        if rule == 'crush_gap':
            rec['notes'].append('机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间')
        out.append(rec)
    return out


def table(recs):
    rows = ['| 规则 | 对象 | 实测 ± U (mm) | 阈值 | 判定 | 裕量 (mm) | 置信度 | 备注 |', '|---|---|---|---|---|---|---|---|']
    for r in recs:
        if 'inputs' in r:
            meas = '；'.join(f"{k.split('_')[0]}={v[0]}±{v[1]}" for k, v in r['inputs'].items())
        else:
            meas = f"{r['measured_mm']} ± {r['U_mm']}" if 'measured_mm' in r else '—'
        rows.append(f"| {r['rule_text']} | {' ↔ '.join(r['labels']) or r['subjects'][0]} | {meas} | {r.get('threshold_mm', '—')} | **{r['status']}** | {r.get('margin_mm', '—')} | {','.join(r.get('confidence', [])) or '—'} | {'；'.join(r['notes'])} |")
    return '\n'.join(rows)


def main(graph_path, out_dir):
    graph = json.loads(Path(graph_path).read_text())
    facts, meta, nodes = facts_for(graph)
    atoms = solve(facts)
    recs = records(graph, atoms, meta, nodes)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'facts.lp').write_text(facts)
    (out / 'verdicts.json').write_text(json.dumps({'schema': 'panoptes.verdict/0-trial', 'scene_id': graph['scene_id'], 'scene_graph': str(graph_path),
                                                   'rule_pack': 'rules.lp v0 2026-10-07', 'decision_rule': 'guard_band_U_k2', 'verdicts': recs}, indent=1, ensure_ascii=False))
    occ = graph['plan_occupancy']
    md = (f"# {graph['scene_id']}: {len(recs)} verdicts from scene graph ({len(graph['nodes'])} nodes, {len(graph['edges'])} edges); grid {occ['shape']} × {occ['cell_m']} m, "
          f"blocked {len(occ['blocked'])}, hazard {len(occ['hazard'])}\n\n" + table(recs) + '\n')
    (out / 'verdicts.md').write_text(md)
    counts = {}
    for r in recs:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print(md); print('counts:', counts)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
