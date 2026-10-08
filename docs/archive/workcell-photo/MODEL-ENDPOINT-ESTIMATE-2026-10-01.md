# 当前模型的高低估计

2026-10-01，复用 `workcell-pointcloud-inspection-2026-10-01` 的当前显示模型。报告运行目录：`/Users/adam/Desktop/panoptes-public/research-notes/workcell-endpoint-estimate-2026-10-01`。

## 结果与对象

- 光幕 `posts.glb / box-1` 的实际底面中心：0.361018705754 native。
- 紧邻围栏 `fence-fitted.glb / section-0-continued-3` 的底面中心线最近点：0.271016399164 native。此段为已观测下横杆的推断延伸。
- 同一地面法向的差：+0.090002306590 native，光幕更高。测点和垂足经同一个刚体变换一次进入报告 Z-up 坐标。
- 照片 4 主体直径 8.5 cm 条件尺度为 0.6844702474894 m/native，得到 **24.71 / 18.55 cm，高差 +6.16 cm**。
- 红帽直径 4 cm 独立尺度为 0.6737564970412 m/native，高差 +6.06 cm。两个尺度相差 1.58%。
- 历史 10 cm 平面包络比例为 0.5964517394154 m/native，同一模型点得到 21.53 / 16.16 cm，高差 +5.37 cm。该包络不是可靠轴向高度，不能把此前数字改变解释为模型精度改善。

8.5 cm 主体是同图较大的命名参考特征，预先选择；20/24 cm 检查值不进入估计。模型估计与这些实测对照分别相差约 +0.71 / −1.45 cm，高差相差 +2.16 cm；这不证明全流程 sub-3 cm。按钮轴向、用表面深度代替轴心深度以及模型端点仍有系统误差。百分位范围仅表示参考表面深度敏感性，不是精度或置信区间。接受的场景尺度仍为 null，米制导出仍禁用。

## 为什么此前方向反了

模型底端来自点云分位数包络；源图端边测量来自另一套深度/局部平面。光幕模型底面中心投回照片 4 比可见黄壳下缘高约 63 原始像素。白栏边缘直接采样又可能落到地面；局部平面支持不够平面。两条原图诊断链能给出相反关系，不能混用，也不能通过翻转符号或统一尺度修复。

此次只把用户指的实际显示模型测点测出来，并在同一视图绘线；没有宣称修好了原始深度或物理端点识别。

## 复现与检查

设置项目根、Python 环境和 `PYTHONPATH=.:scripts:modal_apps`。模型估计限定这一捕获的节点，显式保存后报告检查 GLB/地面 SHA；没有该诊断文件的其他运行不启用。

```python
import json
from pathlib import Path
from scripts.workcell_endpoint_estimate import estimate
from scripts.workcell_photo_report import build

root = Path('/Users/adam/Desktop/panoptes-public/research-notes/workcell-endpoint-estimate-2026-10-01')
(root/'model-endpoint-estimate.json').write_text(json.dumps(estimate(root), indent=2))
report = build(root)
assert report['nativeToMetersDefault'] is None
e = report['endpointEstimation']
assert e['groundTruthUsedForEstimation'] is False
assert e['difference']['valueCm'] > 0
assert abs(e['difference']['valueCm'] - sum((1 if r['objectId']=='post-box-1' else -1)*r['estimateCm'] for r in e['endpoints'])) < 1e-8
```

- `python scripts/check_workcell_endpoint_estimate.py`：真实模型底面、地面旋转、非单位法向、节点变换和邻近测点检查。
- `python scripts/workcell_conditional_scale.py`：平移/旋转相机下解析圆检查。
- `cd web && node checks/photo-scale-check.mjs`：条件估计不提升全局尺度、端点顺序、沿 Z 的高差及跨照片注释边界。
- 现成报告 UI 重新打包为 `page/`；入口参数 `?photo=4&object=post-box-1&view=model&measurement=endpoints#scene` 自动开启测量线。选中光幕或围栏可在全屏右栏看到数值。

缓存报告构建/打包 3.22 秒，不是新推理延迟。原始运行 366.93 秒保留原定义；新增 GPU 调用 0，新增 GPU 花费 0。支出见 `endpoint-update.json` / `spend-ledger.json`。
