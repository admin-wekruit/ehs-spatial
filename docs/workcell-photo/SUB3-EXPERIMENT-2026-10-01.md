# 现有四照片、按钮标尺与 3 cm 目标

本次输入仍是原有四张 BOR1 原 JPEG 和用户给定的按钮尺寸：红帽直径 4 cm、主体最大直径 8.5 cm、整体高度 10 cm。没有要求补充实测相机参数或拍摄高度。照片没有焦距/镜头 EXIF；相机由既有 MapAnything 估计，并另做 COLMAP 多视图优化对照。

目标是围栏下横杆、右光幕外壳、左光幕外壳的离地绝对误差分别小于 3 cm。现场 20/24/24 cm 仅用于保存预测后的评估，未上传到求解函数；本地测量配置先通过校验，再仅取 `reference` 字段上传。三点已向开发者公开，因此这是开发检查集。

## 结论与实际结果

**三轮实验已完成，尚未达到三处误差都小于 3 cm。** 最后一轮 `workcell-sub3-metrology-2026-10-01-c` 包含下面三处实现修复，两个相机路线均正常结束，没有阶段异常。

| 对象 | 已发布估计 | 本次原相机 + 新边线/地面（C） | 现场检查值 | 本次误差 |
|---|---:|---:|---:|---:|
| 围栏下横杆 | 17.13 cm | 16.22 cm，条件估计 | 20 cm | −3.78 cm |
| 右光幕，按钮侧 | 14.36 cm | 未通过跨图下沿验证 | 24 cm | 未取得新估计 |
| 左光幕 | 15.71 cm | 未通过跨图下沿验证 | 24 cm | 未取得新估计 |

原相机与方形像素优化相机下，红色/黄色圆直径候选均未通过留出照片验证，B/D 没有受支持的统一尺度；没有把未通过的拟合数值套给物体。原相机 C 仍使用原整体 10 cm 高度尺度，因此它也不是“两个直径联合标定已经成功”的结果。优化相机得到 24 个地面平面入选点，但三处附近的支持不足，未产生 C 的受支持间隙。

修复排除了旧三维锚点错误干扰圆/线求解等确定性问题；实图中标尺投影、光幕下沿匹配和真实地面仍没有共同建立可靠约束。**这些结果说明当前实现未达标，并不证明这四张照片在原理上无法达到 3 cm。** 也不能只调整一个全局倍率，或把 20/24 cm 写回模型后称为预测准确。

主场景的 52 对象可选、可旋转模型没有被这些未验证候选替换。下一步的几何工作是把已知按钮尺寸放入相机/结构联合拟合，并核实光幕实体端点与混凝土地面；本轮没有完成该算法，因此尚不能声称原图 oneshot 已满足厘米目标。

### 实际用时与资源估算

| 调用 | 函数执行窗口 | 总远程调用窗口 | 函数窗口资源费率估算 |
|---|---:|---:|---:|
| a，修复前 | 13.34 s | 79.06 s | $0.0237 |
| b，方形像素对照，修复前 | 43.47 s | 55.20 s | $0.0772 |
| c，三处修复后 | 34.09 s | 613.84 s | $0.0605 |
| 合计 | 90.90 s | 748.11 s | $0.1614 |

c 中原相机子进程为 27.78 s，优化相机为 30.88 s，两路并行。c 的约 580 s 差额在函数执行窗口之外；现有记录没有进一步区分调度、启动与传输，不能将该差额归因于几何计算。期间曾怀疑重复拟合导致长等待；收到实际函数记录后已更正，没有为此引入未实跑的缓存版本。当前 solver/check 源码与 c manifest 逐字匹配。

费率按 2×A100-80GB、16 CPU、80 GiB 预留资源计算，`actualBilledUsd=null`，不是账单。ledger 另保存“总调用窗口乘同一费率”的参考值；含函数外等待，不能当成已经花费的 GPU 费用。没有重跑完整神经网络/建模流程，历史 334.60 s 原图完整报告成绩仍保留原定义。

最终预测文件 SHA256：

```text
ac2db1b18eed18050b84c6b61306a158de93ee70dd385ef0238da67794534b81  c/original-cameras/results.json
e0badf5429736354dba4a9ec207b47b808158e1572378bbb9ad6d7edfe12bb67  c/refined-cameras/results.json
```

运行与源码 manifest、完整源像素、入选/剔除几何及 ledger 均随公开实验页发布。a/b 附 `review.json` 标为含实现问题的历史记录；不得把其早期审核通过摘要当成当前结论。

## 实际实现范围

- A：已发布 native 几何，按按钮整体 10 cm 换算。
- B：原图两个已知圆部件分别拟合直径尺度，保留旧几何。
- C：原图实体下沿、附近地面重新求解，保留原高度尺度。
- D：新的下沿/地面和通过源观测验证的直径尺度组合。

D 不是“按钮、相机、地面全部变量一起优化”。相机路线之间分别运行几何求解，尚未加入按钮尺寸约束的相机联合 BA，也没有把这些对照作为一条通过验收的全流程算法。主场景保留原有 52 对象模型；实验失败不能通过仅改卡片上的数字消除。

运行使用 `modal_apps/workcell_guard_experiments.py`，始终为临时 `modal run`、2×A100-80GB、16 CPU、80 GiB、无重试或常驻服务。原相机和优化相机两路并行。缓存了原始神经网络输出；本轮计时是增量几何实验，不是新的原图完整 oneshot 成绩。此前完整报告历史成绩为 334.60 秒。

## 前两轮复核发现的问题

原始保存目录为 `workcell-sub3-metrology-2026-10-01-a` 和 `workcell-sub3-metrology-2026-10-01-b`，均保留完整失败结果、输入/源码哈希与支出估算。不能把这两轮的失败解释成“四照片本身无法测量”：

1. 圆轮廓切平面的朝向错误依赖旧的三维按钮中心。相机改变后，旧中心可能投影到两个切线同一侧，导致法线同向、半径为负。法线应由源图观察到的角度左右边界决定。
2. 水平三维边线求解把旧中心沿轴方向的位置当固定坐标。带像素噪声时，这个坐标会影响实际直线高度；同一观测因此得到不同量测。沿轴坐标应仅是参数规范，真实残差应来自投影直线与源图端点。
3. 光幕的两片壳体合并宽度，被用于筛选每一条短下沿。地线/端子把实体边缘分成片段后，正确边也被删掉。宽度应来自该边所属壳体面的 RGB 侧边；不同高度后片、线缆不能拼成一个实体底端。

以上问题由合成几何检查和原图审阅定位；没有降低 3 cm 目标或按现场距离选择照片。

地面问题仍需单独考虑：原推断点图的入选平面中包含安装板/导轨附近样本；新相机下的地面 RGB 匹配较少。低平面残差不等于已确认真实混凝土地面。所有相关源像素、入选/剔除状态和局部支持保存在结果与叠线图中。

## 文件与复现

- `scripts/workcell_photo_metrology.py`：仅接收标尺规格的源观测求解；结果包含失败状态、原图边界、点到地面垂线、尺度候选和留出照片误差。
- `scripts/workcell_guard_controls.py`：COLMAP 相机对照；新增 `square_pixels=True` 在原 JPEG 栅格中约束横纵焦距一致，仍允许每图焦距不同。
- `scripts/workcell_metrology_report.py`：先固定预测字节/hash，再加载现场检查值进行对照；结果附加到现有完整 3D 报告。
- `scripts/check_workcell_photo_metrology.py`：投影、标尺、实体边界、遮挡、测量与地面的一致性检查。
- `scripts/check_workcell_camera_pixels.py`：逐图原图/推断图/COLMAP 栅格与像素中心往返检查。
- `scripts/check_workcell_metrology_report.py`：改变检查真值不会改变估计，拒绝不一致尺度、重复/漏对象和伪造 unsupported 数值。

先依照 [HANDOFF.md](HANDOFF.md) 获取 `workcell-photo-handoff-2026-09-30` 固定复现包、安装环境，使用包内四张原 JPEG 和冻结 baseline。不要用 Pages 的缩小照片代替原图。

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_photo_metrology.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_camera_pixels.py
PYTHONPATH=.:scripts .venv/bin/python scripts/check_workcell_metrology_report.py

# 一次临时双 A100 调用：原推断相机，与重新运行的方形像素相机并行对照。
PYTHONPATH=.:scripts .venv/bin/python -m modal run modal_apps/workcell_guard_experiments.py \
  --baseline "$BASELINE" --out "$ARTIFACTS/metrology-new" \
  --sources "$INPUTS/image_01.jpg,$INPUTS/image_02.jpg,$INPUTS/image_03.jpg,$INPUTS/image_04.jpg" \
  --mode metrology-square-pixels \
  --measurements docs/workcell-photo/measurements-2026-10-01.json

# 已有优化相机可直接复用，不必重复做相机对照。
PYTHONPATH=.:scripts .venv/bin/python -m modal run modal_apps/workcell_guard_experiments.py \
  --baseline "$BASELINE" --out "$ARTIFACTS/metrology-replay-new" \
  --sources "$INPUTS/image_01.jpg,$INPUTS/image_02.jpg,$INPUTS/image_03.jpg,$INPUTS/image_04.jpg" \
  --mode metrology --control "$ARTIFACTS/metrology-new/square-pixel-control" \
  --measurements docs/workcell-photo/measurements-2026-10-01.json
```

发布报告时先构建 `web/dist-photo`。传入此前完整测量报告（含 `page/`）以及各次实验目录；输出目录必须不存在。`--run` 按实际执行顺序提供，可以重复多个。所有历史对照折叠展示，保留原始数据链接。

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/workcell_metrology_report.py \
  --baseline "$ARTIFACTS/workcell-measured-reference-2026-10-01-final" \
  --run "$ARTIFACTS/metrology-new" --run "$ARTIFACTS/metrology-replay-new" \
  --measurements docs/workcell-photo/measurements-2026-10-01.json \
  --out "$ARTIFACTS/metrology-report-new"
```

公开站点只更新 `workcell-photo-direct/`，其他目录保持各自会话的结果。当前源码位于公开 `admin-wekruit/ehs-spatial` 的 `codex/workcell-photo-speed` 分支。
