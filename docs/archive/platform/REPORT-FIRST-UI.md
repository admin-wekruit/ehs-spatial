# 完整工位报告与同页复核

日期：2026-09-12。此轮交付为本地平台 UI、来源导入和固定报告；未切换正式 Pages 入口，未调用新的 VLM/重建模型。

## 入口和使用路径

最终固定报告：`http://127.0.0.1:8792/app.html#/reports/2d77667b-5369-4d66-b5c4-d426e9833e18`。

项目 `a2b7c04d-0162-488b-b7db-37711a37ea62`，场景版本 `518e7869-3ce2-4c2e-be18-64750483846d`。

1. 上传、项目库和报告库首先进入工位报告。1–4 张视图属于同一个工位，不把同一工位按渲染技术拆成多个页面。只读阶段可以选择对象、切照片、旋转、切模型/表面/点云和查看依据。
2. 报告第二步为“补充与复核”：圈选漏检区域、携带图片与选框打开 Agent、查看政策与理由、提交人工证据或精确评估复核。访客保存自己的副本后写入，不改原发布快照。建模/布局的完整工作台仍有单独入口。

原图、3D、CAD 投影和交互平面共享实体选择。只有一个主 WebGL；手机以视图标签显示同一组内容。选择与图层是浏览状态，不创建推理任务。原图坐标轴通过保存的相机投影；未知位置不捏造定位。

## 找回并接入的真实内容

| 内容 | 当前报告中的身份及展示 |
| --- | --- |
| 输入与对象 | 3 张原图、67 条照片观察、68 条对象记录，保留尚未确认的跨图身份；不把记录数当作确认物体数 |
| 3D | 55 条对象有空间表示，9 个对象模型；生成模型、观测表面和背景点云可切换，候选放置明确标识 |
| 分割 | 使用保存的原图多边形及像素映射，处理孔洞、小对象和留白；没有多边形时保留真实原图框 |
| 点云 | 原始 783,473 点逐点/颜色对照冻结 pointmap；去掉原有 content mask 外的 padding 后显示 581,301 点；没有新推理和配准 |
| 空间位置与 CAD | 68 个已有投影轮廓，圆柱保持圆形轮廓；同 frame 与投影矩阵；变换、几何或地面变化后不复用旧轮廓，使用当前几何投影 |
| 图片理解 | 当前来源 62 条，31 条可明确关联对象，另 31 条保留在来源详情；缺失/筛除候选、原始理由可展开 |
| 历史 EHS | 原 `user-bor1-02` 的 9 项结果、政策文本、理由、37 项清单、CAD 和 4 张原图；与当前版本评估分开 |
| 实验质量 | 每个原始候选的同 run 指标及文件；后来的 Blender 圆柱不继承 RecGen 的分数 |
| Blender | 历史原 `.blend` 保留；当前版本另经真实 CPU worker 导出、重新打开，结果与 manifest 一并冻结 |
| 历史和任务 | 发布快照冻结输入或产物对应该 revision 的任务与资产；步骤、状态、实录耗时、产物在同一报告，不读取后续活任务伪装旧状态 |

当前版本的 Blender 任务 `69c5b5e1` 导出 48 个对象，22 个缺少可导出网格或未确认放置的对象仍为未完成。背景点云作为 context 单独记录 `excludedRepresentations`，不会把未完成物体计数加一。点云 GLB 可单独下载，不声称是 Blender mesh。

最终 Publication 冻结 180 个资产和 2 个任务。新 `.blend` 资产 `c3c0b925-9ea3-4904-8aa2-6278105fcaa5`、GLB `11e8f097-cf46-4cc8-a1f1-0fa23b5b6a10` 及校验文件均经实际 HTTP 下载校验 SHA/大小。浏览器下载通过本站内容接口和本地 blob，不跳到供应商域名。

当前版本尚无新的 EHS evaluation；界面如实显示“尚未评估”。历史示例规则未冒充当前法规结论或默认安全距离。新图模型及真实付费 Agent 的发布门槛保持未通过。

## 实现和检查

共享报告组件 `WorkcellReport` 替换重复的旧报告页面实现，复用 `PhotoView`、`SpatialView`、`PlanView`、Deep Chat 和同一编辑接口。渲染器不保存项目；publication immutable，复核写新副本/版本。

可运行检查（在产品工作树根目录）：

```sh
PANOPTES_TEST_DATABASE_URL=postgresql://adam@127.0.0.1:55432/panoptes_platform .venv/bin/pytest -q
npm --prefix web run check
npm --prefix web run build
node --experimental-strip-types web/checks/renderer.mjs
node --experimental-strip-types web/checks/report-interactions.mjs
node --experimental-strip-types web/checks/report-scene.mjs
node --experimental-strip-types web/checks/report-download.mjs
node web/tests/photo-draw-check.mjs
node web/tests/report-review-check.mjs
node web/tests/report-context-check.mjs
PANOPTES_TEST_PUBLICATION_URL=http://127.0.0.1:8792/api/publications/2d77667b-5369-4d66-b5c4-d426e9833e18 node web/tests/report-evidence-check.mjs
```

新增检查覆盖来源 SHA/像素坐标、点云过滤、导入幂等、发布时任务资产完整性、错误下载/截断文件、测量来源、轮廓失效条件、选择/框选事件顺序及复核快照。全量后端562通过/28跳过，9项前端检查及构建通过。浏览器实际补充检查记录在 `BROWSER-QA.md`，汇总为 `report-first-verification.json`。
