"""Layer / contract diagram of the verdict layer (docs/research/verdict-layer-architecture-2026-10-08.md).
  python viz_layers.py <out.png>
"""
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Hiragino Sans GB', 'DejaVu Sans']; plt.rcParams['axes.unicode_minus'] = False
from matplotlib.patches import FancyBboxPatch

LAYERS = [  # (label, today, later, colour)
    ('L7 证据与报告', '表 / 平面图 / 立体图 / 匹配图', '审核员屏；coverage / gaps 反馈 L1、L4', '#e8f0fe'),
    ('L6 判定引擎', 'clingo + 护带 + 五态', '逐帧 → 裕量信号 → STL / STREL（RTAMT, MoonLight）', '#e6f4ea'),
    ('L5 检查合成与验证', '—', '对 C2 API 合成检查 → 冗余差分 / 性质测试 / 蜕变 → 字面化审阅', '#fff4e5'),
    ('L4 规范侧（独立于场景）', '8 月 7 谓词编译器（已取代）', '条款图（AEC3PO）+ 对齐表（GinSign 式）+ 检索（DriveReg 式）', '#fff4e5'),
    ('L3 感知规约监控', 'shape / floor / lines / obvious_errors 检查', 'STPL 式：持久性、尺寸一致、地面接触、coverage 阈值', '#f3e8fd'),
    ('L2 薄场景图', 'scene_graph.py：3D 距离 / 离地缝 / z 重叠 / 越过 / 视线 / 占用 + Spark-DSG', '关系库扩展 + 查询接口', '#f3e8fd'),
    ('L1 感知 / 重建', '照片：SAM 3 → MVS + MoGe-3 → SAM 3D → 组装 v2', '视频：Hydra / DAAAM / WorldSGG / Khronos（各一个 adapter）', '#fdecea'),
]
CONTRACTS = {  # between layer index i (upper) and i+1 (lower) in LAYERS order
    (5, 6): 'C1 场景契约：对象 id、类别+置信度、有向盒、σ、视角、t、coverage、区域',
    (1, 2): 'C4 规则包：rule@version + 条款 id + ASP + 测试 + 审阅记录',
    (0, 1): 'C5 判定对象：status、measured ± U、threshold、evidence、provenance',
    (4, 5): 'C2 事实 schema：类型化谓词（单位、值、U、来源、t）',
}


def main(out):
    fig, ax = plt.subplots(figsize=(15, 10.5)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')
    ax.text(50, 97.5, '判定层：七层、四个契约（每层只认上一层的契约，不认实现）', ha='center', fontsize=13, weight='bold')
    n = len(LAYERS); top, h, gap = 92, 9.2, 3.2
    ys = []
    for i, (label, today, later, col) in enumerate(LAYERS):
        y = top - i * (h + gap) - h; ys.append(y)
        ax.add_patch(FancyBboxPatch((4, y), 92, h, boxstyle='round,pad=0.3', fc=col, ec='#333333', lw=1))
        ax.text(6, y + h - 2.2, label, fontsize=10.5, weight='bold')
        ax.text(6, y + h - 5.0, '今天：' + today, fontsize=8.2, color='#333333')
        ax.text(6, y + h - 7.6, '以后：' + later, fontsize=8.2, color='#555555')
    for (a, b), text in CONTRACTS.items():
        ya, yb = ys[a], ys[b] + h
        ax.annotate('', xy=(50, ya), xytext=(50, yb), arrowprops=dict(arrowstyle='-|>', color='#1f77b4', lw=1.6))
        ax.text(52, (ya + yb) / 2 - 0.4, text, fontsize=8, color='#1f77b4', va='center')
    # C3 signature: shared registry between L2 and L4 drawn at the right margin
    y2, y4 = ys[5] + h / 2, ys[3] + h / 2
    ax.add_patch(FancyBboxPatch((82, y2 - 1.2), 13, (y4 - y2) + 2.4, boxstyle='round,pad=0.2', fc='#ffffff', ec='#d62728', lw=1.2, ls='--'))
    ax.text(88.5, (y2 + y4) / 2, 'C3 签名注册表\n（类别 / 区域 / 边 /\n属性 / 声明输入）\nL2 与 L4 共用：\n对齐目标，\n不是拒绝的门', ha='center', va='center', fontsize=7.8, color='#d62728')
    ax.text(50, 1.2, '横切：版本三元组（场景 schema / 规则包 / 引擎）随每条判定；评测 harness 冻结基准 + 固定输出。'
                     '4D 工作（DAAAM、WorldSGG、Hydra、Khronos）全是 C1 的产者，不进 L2–L7；ChronoGraph 是规划表示，暂不接。', ha='center', fontsize=8.2, color='#333333')
    fig.savefig(out, dpi=130, bbox_inches='tight'); print(out)


if __name__ == '__main__':
    main(sys.argv[1])
