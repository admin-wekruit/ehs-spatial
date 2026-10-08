# 统一地面与 3D 模型卡尺

日期：2026-10-02。源码分支 `codex/workcell-photo-speed`；发布目录仅 `panoptes-workcell-report/workcell-photo-direct/`。

[打开完整报告](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/?photo=4&object=post-box-1&view=model&measurement=endpoints#scene)

## 本轮完成

- 最终地面确定后，统一刷新 `geometry.clearances`、`fence.beams[].clearance` 中的高度、垂足和有原始样本支持的范围。观测点和模型形状保留；没有原始范围样本的旧区间清空，不能沿用旧地面区间。
- 报告坐标为地面 Z-up、零高度；法向与平面偏移一起归一化。所有卡尺沿这个法向测量，和屏幕方向无关。
- 复用现有网格射线拾取：一点离地；两点直线距离和 B−A 有符号高差；三个不共线表面点形成的采样三角形高度范围和倾角。三点结果只代表该采样区域。
- 卡尺、底端对比卡、50 cm 模型网格、网页 GLB 下载和离线 GLB 使用同一显式模型比例。导出统一转为 Y-up，地面 Y=0。保留护板材质及顶点颜色。
- 可下载测量 JSON：保留原生测点、实际命中对象/模型/哈希、地面、垂足、换算比例和来源。标尺试算单独记录当前尺寸、当前范围、同比系数与基准证据。
- 空白点击不产生点；换照片/模型/地面清除旧点；拖动旋转不误取点；清除测量恢复已有标注。保留原有角度功能。

## 如何使用

1. 打开报告，选择任意对象，再进入“可旋转 3D 模型”。右侧顶部是“空间测量”。
2. 选择“一点”“两点”或“三点”工具，按“在 3D 中选择…个点”。可以跨物体点击真实模型表面；取点列表显示实际命中对象。
3. 按“计算并标注”。红线连接测点和同一地面；两点的 Δh 为第二点减第一点，正值表示第二点更高。
4. “下载测量 JSON”保存证据；“清除测量并恢复原标注”回到既有底边对比。
5. 在上方“统一模型标尺 · 急停按钮”修改整体高度，可同比试算三尺寸。更改三者比例仍需要重新拟合。基准为 10 / 8.5 / 4 cm。

## 坐标与比例依据

平面 `n·x+d=0` 先使 `||n||=1`。点高度 `h=n·p+d`，垂足 `p−hn`。两点距离 `||B−A||`，高差 `n·(B−A)`。因此不同深度的等高点仍为零高差。

本次模型比例沿用照片 4 的 8.5 cm 主体条件比例：**0.6844702474893979 m/native**；4 cm 红帽用于独立交叉检查。10/8.5/4 cm 联合标定仍未通过，accepted physical scale 仍为 null。条件模型厘米值没有被写成已验证物理尺寸。

当前原模型底面估计仍为右围栏 **18.55 cm**、右光幕 **24.71 cm**，光幕高 **6.16 cm**。左右模型差异仍待真实底边重建；本轮没有利用 20/24 cm 对照值或同高先验调整模型，也没有宣称 sub-3 cm。

## 验证记录

- Python：最终地面缓存、非单位/倾斜/平移地面、物理边缘契约、标尺与评估值隔离、GLB 坐标/尺度/颜色检查通过。
- TypeScript：一点高度、反向高差、不同深度等高点、三点倾角/范围、非有限/重合/共线拒绝通过；编译和照片报告构建通过。
- 真浏览器：52/52 模型加载；一点、两点、三点工具可用；旋转、空白点击、清除、照片切换已检查。下载的测量 JSON 垂足平面残差为 0。
- 真浏览器标尺 10→20 cm：同一测点由约 103.5→207 cm，底端高差 6.16→12.32 cm；50 cm 网格原生间距减半。导出保留 2× 系数和原始证据，现场实测对照不变。
- 浏览器下载 2× GLB 与离线 1× GLB 的世界包围盒缩放一致，最大差 `4.31e-16 m`。实际离线 GLB 的 **52/52 primitive 保留 COLOR_0**，左右护板颜色与源模型逐值相同。
- 对抗审查的两个 P2（试算证据混用、纹理视觉复制丢颜色）已修复并复查，无剩余已确认 P1/P2。

最小检查命令（在源码根目录，使用项目 Python 环境）：

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:scripts:modal_apps OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/check_workcell_photo_metrology.py
python scripts/check_workcell_ground_distance.py
python scripts/check_workcell_photo_calibration.py
python scripts/check_workcell_metric_export.py
cd web
node --experimental-strip-types tests/ground-caliper-check.ts
node checks/photo-scale-check.mjs
npx tsc --noEmit
node node_modules/vite/bin/vite.js build --config vite.photo.config.ts
```

## 产物、延迟与余项

本地输出：`research-notes/workcell-ground-caliper-2026-10-02/`，包含 `implementation-manifest.json`、`browser-validation.json`、实际下载的卡尺 JSON 和 `workcell-conditional.glb`。来源为 `workcell-endpoint-estimate-2026-10-01/` 的冻结模型/相机/原始深度；不是新推理。

一次缓存复用封装测得 **3.05 s**，新增 GPU 调用 **0**、新增 GPU 推理费用 **$0**。原始完整 oneshot **366.93 s**，本轮没有重新测试完整推理速度。场景再导出产生最多 `1.17e-10 native` 的数值舍入差，没有改变模型形状。

剩余三项仍见 [HANDOFF](HANDOFF.md)：多视角真实底边与左右一致性；三尺寸联合尺度/地面标定；修正后的完整 oneshot 及独立精度验证。该卡尺层已接入现有流程，后续重建输出可直接复用。
