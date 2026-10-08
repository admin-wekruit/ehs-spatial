"""How a rule (spec side) binds to the scene graph (instance side): left = rule templates with their RASE parts and variables,
right = scene nodes/edges of one cell, arrows = the bindings the engine found, labelled with the measured edge and the verdict.
  python viz_matching.py out/090/scene-graph.json out/090/verdicts.json out/090/matching.png
"""
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Sans GB', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import FancyBboxPatch

COLOR = {'PASS': '#2ca02c', 'FAIL': '#d62728', 'NEEDS_MEASUREMENT': '#ff7f0e', 'NEEDS_INPUT': '#9467bd', 'CANNOT_DETERMINE': '#7f7f7f'}
RULES = [  # rule id, title, RASE parts shown on the left
    ('floor_gap', '规则 A  防护离地缝 ≤ 180 mm（ISO 13857 4.4）',
     ['Selection  F : class ∈ {fence, guard}', 'Applicability  F 是危险区 Z 周界的一部分（zone 节点待加）',
      'Requirement  边 floor_gap(F → floor) ≤ 180，护带 U', 'Exception  —']),
    ('crush_gap', '规则 B  机器人 ↔ 固定物 ≥ 500 mm（ISO 13854 / 10218-2）',
     ['Selection  R : robot；X : class ∈ {fence, guard, bollard, light_curtain}', 'Applicability  X 在 R 的运动包络附近（包络 = 受限空间，待输入）',
      'Requirement  边 min_distance_3d(R, X) ≥ 500，护带 U', 'Exception  —']),
    ('enclosure', '规则 C  危险区被防护闭合包围（拓扑）',
     ['Selection  Z : hazard zone（今天 = 机器人占用格）', 'Applicability  占用格层存在',
      'Requirement  不存在 outside → … → Z 的未被挡路径', 'Exception  联锁门 / ESPE 覆盖的开口不算（属性待加）']),
]


def main(graph_path, verdicts_path, out_png):
    g = json.load(open(graph_path)); v = json.load(open(verdicts_path))['verdicts']
    nodes = {n['id']: n for n in g['nodes'] if n['cls'] != 'floor'}
    fig, ax = plt.subplots(figsize=(17, 10)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.text(14, 97, '规范侧：规则模板（变量 + RASE 四部分）', fontsize=12, weight='bold', ha='center')
    ax.text(84, 97, f"场景侧：{g['scene_id']} 的场景图（实例）", fontsize=12, weight='bold', ha='center')
    ax.text(50, 97, '匹配 = 变量绑定（引擎自动做）', fontsize=12, weight='bold', ha='center')
    # right: scene nodes
    order = ['fence', 'guard', 'bollard', 'light_curtain', 'robot', 'cart']
    ids = sorted(nodes, key=lambda i: (order.index(nodes[i]['cls']) if nodes[i]['cls'] in order else 9, nodes[i]['label']))
    ypos = {}
    for k, i in enumerate(ids):
        y = 90 - k * 9; ypos[i] = y; n = nodes[i]
        ax.add_patch(FancyBboxPatch((72, y - 2.6), 26, 5.2, boxstyle='round,pad=0.3', fc='#f4f4f4', ec='#333333', lw=1))
        ax.text(85, y + 0.6, f"{n['label']}  [{n['cls']}]", ha='center', fontsize=9)
        ax.text(85, y - 1.5, f"bottom {n['bottom_m'] * 1000:+.0f} mm · top {n['top_m'] * 1000:.0f} mm · {n['confidence']}", ha='center', fontsize=7, color='#555555')
    ax.add_patch(FancyBboxPatch((72, 2), 26, 5, boxstyle='round,pad=0.3', fc='#fff3e0', ec='#333333', lw=1))
    ax.text(85, 4.5, 'floor / 占用格层（blocked / hazard / outside，observed = unknown）', ha='center', fontsize=7)
    # left: rules
    ytop = 90
    for rid, title, parts in RULES:
        h = 3.2 + 2.6 * len(parts)
        ax.add_patch(FancyBboxPatch((1, ytop - h), 38, h, boxstyle='round,pad=0.3', fc='#eef3ff', ec='#333333', lw=1))
        ax.text(2, ytop - 2, title, fontsize=9, weight='bold')
        for j, ptxt in enumerate(parts):
            ax.text(3, ytop - 4.6 - 2.6 * j, ptxt, fontsize=7.5)
        # bindings = verdict records of this rule
        recs = [r for r in v if r['rule'] == rid]
        for m, r in enumerate(recs):
            col = COLOR[r['status']]
            if rid == 'enclosure':
                ax.annotate('', xy=(72, 4.5), xytext=(39, ytop - h / 2), arrowprops=dict(arrowstyle='->', color=col, lw=1.4))
                ax.text(55, 8, f"Z ← 机器人占用格 · 48 个开口格 → {r['status']}（观察范围缺失）", fontsize=7.5, color=col, ha='center')
                continue
            subj = r['subjects']
            target = subj[-1] if rid == 'crush_gap' else subj[0]
            if target not in ypos:
                continue
            label = f"{r.get('measured_mm', '?')} ± {r.get('U_mm', '?')} mm → {r['status']}"
            ax.annotate('', xy=(72, ypos[target]), xytext=(39, ytop - h / 2), arrowprops=dict(arrowstyle='->', color=col, lw=1.3, alpha=0.9, connectionstyle=f'arc3,rad={0.08 * (m - len(recs) / 2)}'))
            ax.text(55, ypos[target] + (1.4 if rid == 'floor_gap' else -1.4), label, fontsize=7, color=col, ha='center',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec=col, lw=0.6, alpha=0.9))
        ytop -= h + 3
    ax.text(50, 1, '箭头 = 一个绑定（规则变量 ← 场景节点），标签 = 被测的边和判定；一条规则有几个绑定就出几条判定。'
                   'Applicability 里标"待加 / 待输入"的条件今天没有，所以判定都是"若适用"的条件判定。', fontsize=8, ha='center', color='#333333')
    fig.savefig(out_png, dpi=130, bbox_inches='tight'); print(out_png, 'bindings drawn:', sum(1 for r in v if r['rule'] in {'floor_gap', 'crush_gap', 'enclosure'}))


if __name__ == '__main__':
    main(*sys.argv[1:4])
