# 待检查地址（2026-10-06）

三份 090 报告放在一起对照。原报告一律不改，新东西都是另开的。

## 090 工位（9 月照片 3 张）

| 版本 | 几何 | 补全模型 | 地址 |
|---|---|---|---|
| ① 原报告 | Pi3X（非商用） | RecGen（非商用） | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=4b58dbd2-3846-47f2-af97-57eaa108753c#/reports/4b58dbd2-3846-47f2-af97-57eaa108753c |
| ② 对照版 | Pi3X（非商用） | SAM 3D（可商用），出明显错误时用测量盒 | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=4b58dbd2-3846-47f2-af97-57eaa108753c&layer=sam3d#/reports/4b58dbd2-3846-47f2-af97-57eaa108753c |
| ③ 全新可商用版（审查修复中，先别细看） | MVS（RoMa + numpy LM 光束法平差，不含 GPL） | SAM 3D | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html#/reports/25686138-ba21-475f-81f0-c2818bc4ab6a |

③ 的已知问题，正在修：
- 3D 场景加载很慢，有空场景的截图，正在查原因；
- 防护板两处折角读成 134°，旧报告实测是 85–89°；
- 站牌、线缆托架没有模型；
- 左防撞柱的盒子太薄；
- 右围栏深度残差 6.7%。

③ 的明显错误门 G1–G8 是 0 个（旧报告 8 个，对照版 1 个），说明这组检查覆盖不到上面这些问题。逐物体对照表：`workcell-clean-report-090-2026-10-06/table.md`。

### 逐物体直达（② 对照版；点开右侧有"完整流程"一栏）

在 ② 的地址后面加 `?object=<id>`。

| 物体 | id |
|---|---|
| 左光幕 | `5163a9b0-0bb6-5bb5-a9bc-927cb94f8d08` |
| 右光幕 | `169518d8-4f3a-5f67-a565-3180539acad3` |
| 左围栏 | `929b5b7e-17cb-5e05-bbf9-7767114ddd1e` |
| 右围栏 | `ce9516a5-a385-5ecb-b6f5-3d543f3ec941` |
| 左防撞柱 | `27d00998-2702-547e-97b2-d200cd92264a` |
| 右防撞柱 | `da0a50da-6459-5471-b25b-d299129c354a` |
| 机器人 | `d5780205-5e83-5f09-8e90-fe5450d96315` |
| 料车 | `a719e41c-bc9f-5963-b9f8-2caa031a8c09` |
| 防护板（测量盒代替） | `0bc9608d-041f-516b-892d-cee0eb174aeb` |

例：左光幕
https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=4b58dbd2-3846-47f2-af97-57eaa108753c&layer=sam3d#/reports/4b58dbd2-3846-47f2-af97-57eaa108753c?object=5163a9b0-0bb6-5bb5-a9bc-927cb94f8d08

## 030 工位（参考，暂不改）

- 原报告：https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=cd84d3fb-7d1f-4736-8ffa-d44855e59fab#/reports/cd84d3fb-7d1f-4736-8ffa-d44855e59fab

## 已有的数字（详见各说明）

- 补全对照（② 对 ①）：`completion-ab-090-2026-10-06/README.md`，逐物体对比图在同一目录 `*.jpg`。
  - 明显错误（G1–G8 门）：① 8 个物体，② 1 个。剩下的是线缆托架，它原来的模型本来就不对，也不是 RecGen 生成的。
- Pi3X 的替代：`geometry-backbone-ab-2026-10-06/README.md`。
  - MVS 路线和 Pi3X 用同一输入、同一评判器。
  - 4 个现场值的平均误差：MVS 0.6 cm，Pi3X 1.4 cm；最大误差 1.0 对 2.7 cm。
- Pi3X 替代的独立复核：`geometry-backbone-ab-2026-10-06/VERIFY.md`。
  - 结论：可以用。数字能复算，评判器没改，现场值没有进入任何输入。
  - 两处要改正：
    - 实际是两处改动叠加（阈值 0.05，外加保留像素的置信度设为 1）。置信度按原值算时，平均误差 0.77、最大 1.61 cm，仍然通过。
    - 公平地比，Pi3X 走同一条 MVS 路线是 0.59 cm，两者打平。比已发布的 Pi3X 流程好，但"比 Pi3X 更准"的说法不成立。
- 许可：`licence-clean-stack-2026-10-06/licences.md`。
  - pycolmap 的 PyPI 包里带 GPL 代码，已换成纯 numpy 的光束法平差（LM 算法）。090 和 030 都重新打分，现场值和换之前相差 ≤ 0.0004 cm，平均误差仍是 0.60 cm。
  - 4 个权重已镜像，并记录了 sha256：RoMa outdoor、DINOv2、DA3-BASE、MoGe-3。
  - 详见 `geometry-backbone-ab-2026-10-06/README.md` 开头的"Update 2026-10-06"一节。
  - 正在做：离线部署镜像，以及在断网环境下复现 090 和 030。

## 需要你本人做的（不影响上面的进度）

- 在 HF 上为 Modal 用的那个账号申请 `facebook/VGGT-1B-Commercial` 的访问权限，现在是 403。申请下来后补跑这一项 A/B。
- Meta 系模型（SAM 3 / SAM 3D）的许可禁止核、军工、ITAR 用途：要问客户站点是否涉及。

## 已发布的模块替换报告（2026-10-07，090，组装 v2 = 地面接触罚项；五个版本并排看）

地址格式：`https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=<id>#/reports/<id>`（点物体 → 右侧"完整流程"）

| 版本 | 几何 | 补全 | 可商用 | G1–G8 明显错误 | 轮廓 IoU 均值 | 现场值误差（4 个均值） | 地址 |
|---|---|---|---|---|---|---|---|
| 原版（已发布，组装 v1） | Pi3X | RecGen | 否 | 8 | 0.793 | 1.44 cm | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=4b58dbd2-3846-47f2-af97-57eaa108753c#/reports/4b58dbd2-3846-47f2-af97-57eaa108753c |
| 原版 + 组装 v2 | Pi3X | RecGen | 否 | 2（右光幕 ∩ 右围栏穿插） | 0.798 | 1.44 cm | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=d2b4fb83-180d-40a4-82b9-1e2246f0ced5#/reports/d2b4fb83-180d-40a4-82b9-1e2246f0ced5 |
| 只换补全 + 组装 v2 | Pi3X | SAM 3D | 否（几何非商用） | 0 | 0.846 | 1.44 cm | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=c616f13e-9430-4809-9119-05633fedca73#/reports/c616f13e-9430-4809-9119-05633fedca73 |
| 只换几何 + 组装 v2（内部对照） | MVS | RecGen | 否 | 3（RecGen 模型错） | 0.669 | 0.60 cm | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=32265011-026b-45df-88c8-1f1984a6fbb9#/reports/32265011-026b-45df-88c8-1f1984a6fbb9 |
| 两个都换 + 组装 v2 | MVS | SAM 3D | **是** | 0 | 0.794 | 0.60 cm | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=02bf15f9-682f-46e2-9797-40e5ea703227#/reports/02bf15f9-682f-46e2-9797-40e5ea703227 |
| **两个都换 + 补洞 + 组装 v2（推荐）** | MVS + MoGe-3 补洞 | SAM 3D | **是** | **0** | **0.823** | **0.50 cm**（含 030 补洞后） | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=a9a6e0a0-a77e-4ca0-b460-aa6f18d72698#/reports/a9a6e0a0-a77e-4ca0-b460-aa6f18d72698 |

判断：补全换 SAM 3D、几何换 MVS + 补洞、组装加地面接触罚项，三项都换。防护板两种补全都做不好（测量盒代替）；报告级 G1 地面覆盖（平台侧地面面片）新版本都 75 %，与模块无关。数字全部来自 `module-swap-090-2026-10-07/{table.md, layers-table.md}`。

### 030（同一套脚本；三个版本，都是组装 v2）

| 版本 | 几何 | 补全 | 可商用 | G1–G8 明显错误 | 轮廓 IoU 均值 | 罩壳下沿（现场 24 / 24） | 地址 |
|---|---|---|---|---|---|---|---|
| 原版（已发布，组装 v1） | Pi3X | RecGen | 否 | 见报告 | — | — | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=cd84d3fb-7d1f-4736-8ffa-d44855e59fab#/reports/cd84d3fb-7d1f-4736-8ffa-d44855e59fab |
| 原版 + 组装 v2 | Pi3X | RecGen | 否 | 0 | 0.737 | Pi3X 行（公平评判器）| https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=e7bfb200-34ac-4ea3-b942-d4ca2f6c1358#/reports/e7bfb200-34ac-4ea3-b942-d4ca2f6c1358 |
| 只换补全 + 组装 v2 | Pi3X | SAM 3D | 否（几何非商用） | 0 | **0.819** | 同上 | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=97a6d27c-782f-4417-a874-873847e0c305#/reports/97a6d27c-782f-4417-a874-873847e0c305 |
| 两个都换 + 补洞 + 组装 v2 | MVS + MoGe-3 补洞 | SAM 3D | **是** | 1（防护板轮廓不符，照片 1 IoU 0.15） | 0.762 | 23.6 / 24.6（误差 −0.4 / +0.6） | https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/app.html?report=fafdeb6b-7122-4434-a62d-676b3ff9e50a#/reports/fafdeb6b-7122-4434-a62d-676b3ff9e50a |

030 判断：可商用组合插地 0，现场值误差 ≤ 0.6 cm，轮廓比原版好（0.76 对 0.74）但比 Pi3X + SAM 3D 差（0.82）；唯一明显错误是防护板（和 090 一样，SAM 3D 对折板不行，照片 1 的候选被门拦住后只剩照片 2 的）。030 只有 2 张照片，右围栏、右防撞柱各只有 1 张有掩码——这两件的深度项只有一张照片，悬空 8–18 cm 没被拉下来（罚项只罚下沉）。030 没做 MVS + RecGen 的内部对照。表：`module-swap-090-2026-10-07/table-030.md`。

报告标题里的"(local, unpublished)"是发布前起的名字，内容就是现在线上的版本。

## 模块替换对比（2026-10-07，同一流程只换模块；表与说明）

- 说明与结论：`module-swap-090-2026-10-07/README.md`
- 第一层（组装 / 生成 / 几何 JSON）：`module-swap-090-2026-10-07/table.md`
- 第二层（原流程后段各项检查）：`module-swap-090-2026-10-07/layers-table.md`
- 逐物体对比图：`module-swap-090-2026-10-07/cmp-pi3x/*.jpg`（Pi3X 几何）、`cmp-mvs/*.jpg`（MVS 几何）、`cmp-mvs-fill/*.jpg`（MVS + 掩码内补洞）
- 组装第 2 版（S5 加地面接触罚项，`panoptes-serving/scripts/research/assemble_lucida_scene.py` 的 `floor_penalty`；五个版本副本重跑）：两张表里的"组装v2"列；说明在 README "组装第 2 版" 一节；运行目录 `scratchpad/swap-runs/v2/<版本>`，本地发布 `.platform/swap-20261007/v2-<版本>/`
- 第 5 列（几何模块第 2 版：MVS 点图在物体掩码内用 MoGe-3 补洞，流程其余不变）：`module-swap-090-2026-10-07/README.md` 的"几何模块第 2 版"一节；补洞脚本 `fill_geometry.py`，补了多少 `checks/bbab-geom/090-mvs-fill-padded/fill-record.json`；现场值验收 `field-values-mvs-fill.json`
