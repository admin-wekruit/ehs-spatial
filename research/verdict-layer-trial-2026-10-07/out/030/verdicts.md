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
