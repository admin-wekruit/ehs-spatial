# 回到 9 月四视图报告：急停标定尺度、度量层、护板接缝

更新：2026-10-05。**本文件是当前接手首读。** 用户决定：从 9 月的四视图报告（`web/src/WorkcellReport.tsx`）和它的模型出发，
不再扩展 `workcell-photo-direct/` 的按工位页面（[CELLS-2026-10-04.md](CELLS-2026-10-04.md) 仅作历史）。

- 报告：<https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/report/>（打开即发布 `4b58dbd2`）
- 发布服务：`panoptes-publications-estop`（<https://wekruit-livekit-agents--panoptes-publications-estop-web.modal.run>），只含这一份
  发布；主服务 `panoptes-publications` 与其他 35 份报告未改动。
- 证据与脚本：`research-notes/workcell-measurement-layer-2026-10-04/`、`research-notes/sept-guard-sheet-2026-10-04-d/`。

## 1. 正式版本（写入平台，发布 4b58dbd2 / 版本 997bf52b）

- 基于 9 月发布 `d6c2d4d3`（版本 `f5f4b1d4`，090 工位，3 张照片：新照片 3、4 和一张 9 月侧面照）。
- `setCalibration`：尺度 `operator_anchored`，1 原生单位 = 1.28598 m。来源：急停红钮 4 cm 与黄色本体 8 cm，在报告自己的三台相机里
  用台阶模型亚像素测边、三角化急停后取六个估计的几何平均。约束（`scripts/reference_object_scale.py`）：每张照片每个特征都须在
  规格 ±4% 内，否则不给米制尺度。本版最大偏差 2.33%。照片里黄 ÷ 红 = 2.07（规格 2.00），用户接受 4%。
- `setPrimitive`：急停换成规格圆柱（直径 8 cm、高 10 cm），位置为三台相机三角化的红钮；原生成网格连同支架和标牌一起生成，
  按新尺度是 10.7 × 12.6 × 17.3 cm。
- 操作脚本（需要项目管理凭证，凭证在 `.platform/imports/`，脚本内读取、不打印）：
  `panoptes-platform/.platform/publish-estop-scale-20261004.py`，计划 `.platform/estop-scale-20261004/plan.json`。
  9 月平台库是 Homebrew Postgres 16（`/opt/homebrew/var/postgresql@16`，端口 55432）；启动需
  `LC_ALL=en_US.UTF-8 pg_ctl -D ... -o "-p 55432 -c listen_addresses=127.0.0.1" start`，用完即停。54329 是第二阶段库，不要动。
- 单独部署：`cp -al` 建只含该发布的 catalog → `scripts/prepare_publication_site.py` → 设
  `PANOPTES_PUBLICATION_APP / _FEEDBACK_VOLUME / _CATALOG / _HTTP` 后 `modal deploy modal_apps/publication_site.py`
  （平台提交 `c0e333e`）。整体重部署需重新预处理 20 GB，本机放不下（磁盘须保持 ≥ 8 GB 空闲）。

## 2. 度量层（静态，不改发布）

`workcell-photo-direct/report/measurement-layer/<发布号>.json`，由 `web/src/measurement-layer.ts` 读入，只作用于它指名的版本：
地面修正（两视角平面扫描：存储地面低 1.9 cm）、照片测量结果（光幕外壳下端离地：右 19.2 cm、左 18.1 cm；急停实测红 3.93 cm、
黄 8.13 cm）、以及附加网格（`assets`，按 ID 由页面旁的静态文件提供）。

护板"断开"（2026-10-05 更正）：此前归因于父模型接缝未画、把接缝并回中央板——诊断错误（主场景一直画着父模型余量），已撤回。
真正原因是 9 月生成模型的深度：中央板前移约 0.35 m 且后仰，两翼在中段与它 T 形相交，俯视呈"工"字形。见第 2.1 节。

### 2.1 护板形状校验（三照片立体）

方法与证据：`research-notes/sept-guard-shape-2026-10-05/`。以每张照片为参考做平面扫描立体（另两张须一致），三个参考互证 ≤1.5 cm
的 76 176 个点；RANSAC 得三个真实平面（RMS ≤0.53 cm）：背板与水平 85.5°（竖直），左翼 39.8°、右翼 41.9°（向中间下倾），
折角 84.6° / 89.1°。实物是 V 形导向槽：背板宽 1.48 m，两翼前后长 1.46 m，低边在料车两侧外 7–8 cm。

校正：9 月网格沿照片 3 视线移到实测平面（照片 3 轮廓不变，折线处天然相接），只留照片看到的一面，颜色按可见性取自照片（79%，
其余为料车遮挡处最近邻补色）。三照片光度一致性 1-2 / 1-3 / 2-3：−0.09 / 0.51 / 0.05 → 0.73 / 0.78 / 0.70；实测点距离中位
11.6 → 0.7 cm；轮廓 IoU 不变（0.72 / 0.66 / 0.76）——轮廓检查看不出深度错误，这也是 9 月模型当时"通过"的原因。
经度量层发布（4 个网格 + 事实 + 两处实测折角）。报告规则同时改为：已存的折弯/倾角结果只对它测量的那个模型有效
（`ReportScene.tsx`），模型被替换后旧值（145.6° 等）自动不再显示。

## 3. 验证

全部在 Modal CPU 云端浏览器里打开线上页面（`modal_apps/workcell_browser_check.py --url ... --object-id ... --expected ...`，
本机不渲染）：急停读数 0.080 × 0.080 × 0.100 m；度量层与网格均 200；无控制台错误（检查 q–u）。护板：检查 2026-10-05-a/b（折弯面板只剩两处实测折角，所选对象俯视为导向槽形）。

## 4. 下一步

1. 其余模型已逐个校验（`scripts/workcell_shape_check.py` + `modal_apps/workcell_shape_check.py`，Modal CPU；证据
   `research-notes/workcell-shape-check-2026-10-05-b/`，结论已作为每个物体的"形状校验"事实写进报告）：纹理充足的模型深度一致
   在 1–3% 内；只有 9 月护板确有错误（已校正）；警示牌 +36 cm 但另一深度也较一致，待确认；线网、透明板、周期标线照片判断不了，
   需要按直线/边缘的检查。料车被挡住的后部深度（伸到实测背板后 0.36 m）照片看不到。
2. 围栏横梁（水平直线）检测：找边须经过围栏方向的消失点，再多视角三角化。
3. 逐个核对其余生成模型的尺寸（急停的生成模型偏大 40–80%）。
4. 030 工位：只有 2 张照片（急停只在第 2 张完整），需要 9 月同类报告与补拍侧面。
5. 遮挡（防撞柱挡住光幕下端）目前人工剔除，需自动判断。

规则不变：只用 Modal 临时运行做重计算，不在本机渲染；不打印凭证；不打开 `.platform/imports/`（发布脚本由用户授权运行）；
磁盘 ≥ 8 GB；不改历史证据。
