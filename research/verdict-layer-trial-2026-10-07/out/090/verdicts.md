# 090-a9a6e0a0: 22 verdicts from scene graph (10 nodes, 132 edges); grid [75, 55] × 0.1 m, blocked 720, hazard 149

| 规则 | 对象 | 实测 ± U (mm) | 阈值 | 判定 | 裕量 (mm) | 置信度 | 备注 |
|---|---|---|---|---|---|---|---|
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left fence | 533 ± 143 | 500 | **NEEDS_MEASUREMENT** | 33 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right bollard | 3929 ± 187 | 500 | **PASS** | 3429 | low,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left bollard | 3819 ± 183 | 500 | **PASS** | 3319 | low,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ left light curtain | 3698 ± 179 | 500 | **PASS** | 3198 | low,medium | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right light curtain | 3718 ± 179 | 500 | **PASS** | 3218 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ safety guard | 184 ± 142 | 500 | **FAIL** | -316 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 机器人 ↔ 固定物 间隙 ≥ 500 mm | robot ↔ right fence | 809 ± 145 | 500 | **PASS** | 309 | low,low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 %；机器人盒 = 照片里的姿态，不是运动包络；真实判定要受限空间 |
| 危险区被防护物闭合包围 | hazard_zone | — | — | **CANNOT_DETERMINE** | — | — | 外部可达危险区：48 个开口格，方向 ['+e1', '+e2', '-e1', '-e2']；观察范围未提供（not provided by the reconstruction layer; cells outside footprints are unknown, not empty）→ 不下结论；危险区 = 机器人姿态盒的平面占用，不是受限空间（需控制器配置） |
| 围栏高度 ≥ 1400 mm | left fence | 2664 ± 177 | 1400 | **PASS** | 1264 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 围栏高度 ≥ 1400 mm | right fence | 2682 ± 178 | 1400 | **PASS** | 1282 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | left fence | -2 ± 100 | 180 | **PASS** | 182 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | safety guard | 422 ± 101 | 180 | **FAIL** | -242 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 防护离地缝 ≤ 180 mm | right fence | 84 ± 100 | 180 | **NEEDS_MEASUREMENT** | 96 | low | 该维度无 σ，用默认 5 cm；尺度无 ±%，用默认 2 % |
| 光幕最低光束 ≤ 300 mm | left light curtain | 249 ± 12 | 300 | **PASS** | 51 | medium | 尺度无 ±%，用默认 2 % |
| 光幕最低光束 ≤ 300 mm | right light curtain | 236 ± 13 | 300 | **PASS** | 64 | low | 尺度无 ±%，用默认 2 % |
| 越过防护够到：a 危险高 / b 防护高 / c 水平距离 | robot ↔ left fence | a=2699±178；b=2664±177；c=532±143 | — | **NEEDS_INPUT** | — | low,low | 三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定 |
| 越过防护够到：a 危险高 / b 防护高 / c 水平距离 | robot ↔ right bollard | a=2699±178；b=880±37；c=3928±187 | — | **NEEDS_INPUT** | — | low,medium | 三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定 |
| 越过防护够到：a 危险高 / b 防护高 / c 水平距离 | robot ↔ left bollard | a=2699±178；b=895±37；c=3819±183 | — | **NEEDS_INPUT** | — | low,medium | 三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定 |
| 越过防护够到：a 危险高 / b 防护高 / c 水平距离 | robot ↔ left light curtain | a=2699±178；b=1828±74；c=3698±179 | — | **NEEDS_INPUT** | — | low,medium | 三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定 |
| 越过防护够到：a 危险高 / b 防护高 / c 水平距离 | robot ↔ right light curtain | a=2699±178；b=1821±74；c=3718±179 | — | **NEEDS_INPUT** | — | low,low | 三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定 |
| 越过防护够到：a 危险高 / b 防护高 / c 水平距离 | robot ↔ safety guard | a=2699±178；b=1058±148；c=173±142 | — | **NEEDS_INPUT** | — | low,low | 三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定 |
| 越过防护够到：a 危险高 / b 防护高 / c 水平距离 | robot ↔ right fence | a=2699±178；b=2682±178；c=809±145 | — | **NEEDS_INPUT** | — | low,low | 三个输入齐了；表 2 的查表值和低 / 高风险的选择待对购买正文核 → 不发判定 |
