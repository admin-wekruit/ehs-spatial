# 判定层试跑（2026-10-07）：clingo 规则包 v0 对 090 / 030 已发布的 measurement layer

**目的**：看"场景契约 → 关系 → 规则 → 护带判定"这条路在今天的数据上出什么效果。不是产品代码；数字来自 `docs/research/verdict-layer-rules-2026-10-07.md` C 节，
进生产前要对购买的标准正文核。

**边界（用户要求：判定层必须和重建解耦）**：这个目录只读 `scene.py` 定义的米制场景契约（对象 + 有向盒 + 离地高度 + σ + 置信度 + 照片 id）。
不读网格、run 目录、照片、流程清单，不 import 任何重建代码。`adapter_measurement_layer.py` 是唯一知道 measurement layer 编码的文件；
换一种重建输出 = 换一个 adapter，引擎和规则不动；改一条规则 = 改 `rules.lp`，重建不动。

```
measurement-layer/<id>.json ──adapter_measurement_layer.py──▶ out/scene-<cell>.json   (契约 panoptes.verdict.scene/0)
out/scene-<cell>.json ──relations.py（两两盒距离、平面占用格）──▶ 事实 ──rules.lp (clingo)──▶ out/<cell>/verdicts.{json,md}
```

**跑法**（隔离环境：clingo 5.8.2、numpy、scipy、shapely；不装进项目 venv）
```
uv venv --python 3.12 /tmp/verdict-venv && uv pip install --python /tmp/verdict-venv/bin/python "clingo>=5.7,<6" numpy scipy shapely
PY=/tmp/verdict-venv/bin/python; cd research/verdict-layer-trial-2026-10-07
$PY relations.py                                      # 自检：0.3 m 盒距、围栏环闭合
$PY adapter_measurement_layer.py <layer.json> 090-a9a6e0a0 out/scene-090.json
$PY engine.py out/scene-090.json out/090
```

**规则包 v0（`rules.lp`）**：围栏高 ≥ 1400（13857 表 2 注）、防护离地缝 ≤ 180（13857 4.4）、光幕最低光束 ≤ 300（13855 2010 数字）、
机器人 ↔ 固定物 ≥ 500（13854 表 1 / 10218-2，条款待核）、危险区闭合包围（拓扑：占用格可达）。
判定 = 护带：区间 V ± U 整体在一侧才选边，否则 NEEDS_MEASUREMENT；U = 2·√(Σσ² + (rel·V)²)。
layer 里没有 σ 的维度用默认 5 cm、没有尺度 ±% 用默认 2 %，记录里都标了 flag。

## 结果

# 090-a9a6e0a0: 15 verdicts, grid (75, 55) cells of 0.1 m, blocked 720, hazard 149

| 规则 | 对象 | 实测 ± U (mm) | 阈值 | 判定 | 裕量 (mm) | 置信度 | 备注 |
|---|---|---|---|---|---|---|---|
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left fence | 533 ± 143 | 500 | **NEEDS_MEASUREMENT** | 33 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right bollard | 3929 ± 187 | 500 | **PASS** | 3429 | low,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left bollard | 3819 ± 183 | 500 | **PASS** | 3319 | low,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left light curtain | 3698 ± 179 | 500 | **PASS** | 3198 | low,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right light curtain | 3718 ± 179 | 500 | **PASS** | 3218 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ safety guard | 184 ± 142 | 500 | **FAIL** | -316 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right fence | 809 ± 145 | 500 | **PASS** | 309 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 危险区被防护物闭合包围 | hazard_zone | — | — | **CANNOT_DETERMINE** | — | — | 外部可达危险区：48 个开口格，方向 ['+e1', '+e2', '-e1', '-e2']（平面基 e1/e2）；照片未覆盖的区域和真实开口在这一层分不开 → 不下结论；危险区 = 机器人姿态盒的平面占用，不是受限空间（需控制器配置） |
| 围栏高度 ≥ 1400 mm | left fence | 2664 ± 177 | 1400 | **PASS** | 1264 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 围栏高度 ≥ 1400 mm | right fence | 2682 ± 178 | 1400 | **PASS** | 1282 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | left fence | -2 ± 100 | 180 | **PASS** | 182 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | safety guard | 422 ± 101 | 180 | **FAIL** | -242 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | right fence | 84 ± 100 | 180 | **NEEDS_MEASUREMENT** | 96 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 光幕最低光束 ≤ 300 mm | left light curtain | 249 ± 12 | 300 | **PASS** | 51 | medium | 尺度无 ±%，用默认 2 % |
| 光幕最低光束 ≤ 300 mm | right light curtain | 236 ± 13 | 300 | **PASS** | 64 | low | 尺度无 ±%，用默认 2 % |

# 030-fafdeb6b: 12 verdicts, grid (83, 49) cells of 0.1 m, blocked 204, hazard 156

| 规则 | 对象 | 实测 ± U (mm) | 阈值 | 判定 | 裕量 (mm) | 置信度 | 备注 |
|---|---|---|---|---|---|---|---|
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right fence | 1802 ± 159 | 500 | **PASS** | 1302 | unverified,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ safety guard | 2215 ± 167 | 500 | **PASS** | 1715 | unverified,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right light curtain | 4546 ± 208 | 500 | **PASS** | 4046 | unverified,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right bollard | 5184 ± 230 | 500 | **PASS** | 4684 | unverified,unverified | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left bollard | 4929 ± 221 | 500 | **PASS** | 4429 | unverified,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left light curtain | 4728 ± 214 | 500 | **PASS** | 4228 | unverified,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 危险区被防护物闭合包围 | hazard_zone | — | — | **CANNOT_DETERMINE** | — | — | 外部可达危险区：50 个开口格，方向 ['+e1', '+e2', '-e1', '-e2']（平面基 e1/e2）；照片未覆盖的区域和真实开口在这一层分不开 → 不下结论；危险区 = 机器人姿态盒的平面占用，不是受限空间（需控制器配置） |
| 围栏高度 ≥ 1400 mm | right fence | 2697 ± 178 | 1400 | **PASS** | 1297 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | right fence | 174 ± 100 | 180 | **NEEDS_MEASUREMENT** | 6 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | safety guard | 316 ± 101 | 180 | **FAIL** | -136 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 光幕最低光束 ≤ 300 mm | right light curtain | 260 ± 12 | 300 | **PASS** | 40 | medium | 尺度无 ±%，用默认 2 % |
| 光幕最低光束 ≤ 300 mm | left light curtain | 247 ± 13 | 300 | **PASS** | 53 | medium | 尺度无 ±%，用默认 2 % |

## 读结果

- **能判的已经判了**：围栏高度（2.66–2.70 m，PASS，裕量 > 1.2 m）；光幕最低光束（0.24–0.26 m，PASS，U 只有 12 mm，因为这两个物体 layer 里有 σ）；
  机器人到立柱 / 光幕的间隙（3.7–5.2 m，PASS）。
- **FAIL 的两类要看适用性**：(1) "safety guard" 离地缝 422 / 316 mm > 180：这条规则针对周界防护，低矮护板是否适用由审核员定（平台引擎的 applicability 机制保留）；
  (2) 090 机器人 ↔ safety guard 184 mm < 500：机器人盒是照片里的姿态，不是运动包络，这条判定在拿到受限空间前只是"条件判定"。
- **NEEDS_MEASUREMENT 全部由 U 太大造成**：右围栏离地缝 84 ± 100（090）和 174 ± 100（030）——阈值 180 落在区间里；
  U 大是因为 layer 对 low 置信度的盒没有 σ，引擎按默认 5 cm 算。σ 是重建层的事：它给出 ±2 cm，这两条就能判（090 变 PASS，030 仍是 NEEDS_MEASUREMENT，174 太贴近 180）。
- **CANNOT_DETERMINE（包围）是诚实的**：外部从四个方向都能走到机器人格——照片只拍了工位的一部分，没拍到的区域和真实开口在这一层分不开。
  要判包围需要重建层给"观察到的地面范围"（INFERENCE-POLICY.md 的 observed 区）或审核员画危险区边界。
- 对象分类来自 layer 的英文标签（fence / bollard / light curtain / guard / robot / cart），是语义输入的占位；真正的语义规则（光幕分辨率 d、急停颜色等）等用户的 safety specs。

## 下一步（等用户提供 safety specs 后）
1. 把 specs 按 RASE（Requirement / Applicability / Selection / Exception）标注，LLM 编译成 `rules.lp` 的规则 + 阈值事实，封闭词表 + 拒绝机制沿用 `scripts/policy_compile.py` 的做法；
2. 每条规则一正一负合成场景（`eval_pack.py` 的扩展）；
3. 重建层：给 low 置信度盒也产 σ、给观察到的地面范围、把尺度 ±% 写进 layer（契约里已留字段）。
