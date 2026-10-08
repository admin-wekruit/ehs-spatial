# 对象关联发布验收记录

日期：2026-09-13。本文记录本次数据恢复、原始证据复算与已部署公网版本的实际验收。

| 发布标识 | 值 |
| --- | --- |
| Scene revision | `a97267c4-d435-43ee-915f-a59f2bb44ed6` |
| 最终 publication | `f4e5ca43-543d-4624-8ea0-27aa2843e6e4` |
| 本地数据与来源验证 | 已完成，结果如下 |
| 公网部署及浏览器验证 | 已完成，详见文末实际检查；没有宣称全部物理身份去重完成 |

## 恢复结果及其含义

原导入器给每条来源记录默认设置 `association_pending`，即使该记录已包含明确来源绑定也不更新。它还只把主对象的一个 reference/anchor view 写入核心 observations，报告中的其他已绑定视角没有进入同一 entity 的 `observationRefs`。因此“全部待关联”不等于全部对象经过匹配后失败。

| 项目 | 恢复前 | 恢复后 |
| --- | ---: | ---: |
| 业务对象 | 68 | 68 |
| 核心 observations | 67 | 84 |
| 总 entities，含 3 个 context | 71 | 71 |
| 已确认多视角的业务对象 | 0 | 9 |
| 业务对象观察数 | 67 个单视图、1 个无观测 | 7 个三视图、2 个双视图、59 个单视图 |
| Representations | 58 | 58 |
| Assets | 179 | 179 |
| 已打包 raster masks | 58 | 58 |
| 报告 views 与核心 observation 的反向绑定 | 未提供 | 84/84 个 view 有 `observationId` |

恢复保留了全部 71 个 entity ID，以及原 67 个 anchor observation 的 ID、像素框和 maskAssetId。增加的 17 条由 9 个主对象的 16 个额外照片视角与 `observed_floor` 的 1 个已有照片视角构成。没有应用几何候选合并，没有新增模型调用。

9 个对象为 `left_post`、`right_post`、`right_fence`、`robot`、`left_fence`、`cart`、`guard`、`left_light_curtain`、`right_light_curtain`。它们的 `confirmed` **继承冻结来源中已有的对象身份确认及明确 source ID 归属，不是本次新增 VLM 视觉审核，也不是本次重新证明了物理身份**。原始 `evidence/objects.json` 保存了相应 `physical_identity` 判定及依据。

单视图业务对象仍保留原 `association_pending` 枚举；这表示尚未完成跨照片核对，不能显示成检测失败、照片未定位或对象不存在。总状态为 10 confirmed（9 个业务对象加 1 个点云 context）与 61 pending（59 个单视图业务对象加 2 个原 context）。缺少 mesh 的 13 个业务对象继续保留照片观测。

## 新增 17 个视图的来源复算

结论：**17/17 的 bbox 与 polygons 均与冻结 canonical segmentation mask 经实际报告构建函数计算的结果完全相同。它们是照片分割轮廓，不是生成 mesh 的照片投影。**

逐项检查包括：

1. `manifest.json`、`evidence/objects.json`、`evidence/floor.json` 的 SHA-256 与来源场景的冻结引用匹配。
2. 每个 `canonical_mask.npy` 的 SHA-256 与对应 view 的 `sha256["canonical_mask.npy"]` 匹配。
3. 用实际 `build-unified-data.py` 的 `contours(mask)` 重新生成 bbox/polygons，与当前 `workcell-data.json` 对应 object/view 逐项相等。
4. SAM2 视图的结果 JSON 哈希、candidate mask 哈希、结果记录中的原图哈希、manifest 原图哈希与实际原图内容匹配；选中 candidate 确实存在于相应结果记录中。
5. 旧 SAM 来源文件的内容哈希与 view provenance 中的来源哈希匹配。

下表像素数为 canonical mask 中的前景像素数，全部行均通过上述适用检查。

| Source object ID | Frame | 分割来源 | Mask 像素数 |
| --- | --- | --- | ---: |
| `left_post` | `frame_0001` | 旧 SAM，同次拍摄 | 4523 |
| `left_post` | `frame_0002` | SAM2 原照片分割 | 3379 |
| `right_post` | `frame_0001` | 旧 SAM，同次拍摄 | 3132 |
| `right_post` | `frame_0002` | SAM2 原照片分割 | 4430 |
| `right_fence` | `frame_0003` | 旧 SAM，同次拍摄 | 21022 |
| `robot` | `frame_0001` | SAM2 原照片分割 | 3515 |
| `robot` | `frame_0002` | SAM2 原照片分割 | 3431 |
| `left_fence` | `frame_0003` | SAM2 原照片分割 | 10164 |
| `cart` | `frame_0001` | SAM2 原照片分割 | 7491 |
| `cart` | `frame_0002` | SAM2 原照片分割 | 8497 |
| `guard` | `frame_0001` | SAM2 原照片分割 | 7153 |
| `guard` | `frame_0002` | SAM2 原照片分割 | 6978 |
| `left_light_curtain` | `frame_0001` | SAM2 原照片分割 | 4385 |
| `left_light_curtain` | `frame_0002` | SAM2 原照片分割 | 3303 |
| `right_light_curtain` | `frame_0001` | SAM2 原照片分割 | 3625 |
| `right_light_curtain` | `frame_0002` | SAM2 原照片分割 | 3006 |
| `observed_floor` | `frame_0003` | 旧 SAM floor 与可见地面 ROI 的交集 | 17092 |

合计为 3 条旧 SAM、13 条 SAM2、1 条 SAM floor/ROI。这里的“相同”是准确保留已存 canonical 轮廓及其像素映射：轮廓简化容差为 0.5 canonical 像素，旧 SAM mask 本身为 518 网格，不能宣称恢复了原图分辨率的真实精确边缘。

九个主对象及地面的全部 26 个原始 view 均标记 `observed_only: true`；26 份 `canonical_mask.npy` 和 26 份 `mask.png` 在冻结 geometry root 内实际存在。本次仅补齐已有框/多边形观测，没有将这些源 mask 额外打包，也没有合成 raster mask 或推导 geometrySupport。恢复后的 84 条 observation 因而仍只有 58 个 maskAssetId，其余 26 条保留 `source_mask_not_packaged`。

## 实际源脚本与数据链

- [build-unified-data.py:203](/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/build-unified-data.py:203)：203–213 行读取并验证 canonical mask，然后生成照片 view。195–201 行的 mesh 顶点处理用于 `plan`；两条路径不同。
- [build-unified-data.py:62](/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/build-unified-data.py:62)：62–73 行定义真实复算使用的 `contours`，保留内外环，采用 evenodd 填充和右/下边界不包含的 bbox。
- [build-source-bridge.py:166](/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/build-source-bridge.py:166)：166–183 行修改主对象的 plan 与 metrics_source，保留原 views；275–277 行补充 `scene_object_id`。
- [prepare_lucida_evidence.py:26](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/prepare_lucida_evidence.py:26)：26–61 行保存原图/canonical mask、映射、观测支持、SHA 与 `observed_only`。
- [prepare_lucida_evidence.py:64](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/prepare_lucida_evidence.py:64)：64–88 行读取旧 SAM RLE，并保存当时的明确物体对应关系。
- [prepare_lucida_evidence.py:99](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/prepare_lucida_evidence.py:99)：99–105 行产生地面 SAM/ROI 观测。
- [prepare_lucida_evidence.py:155](/Users/adam/Desktop/panoptes-public/panoptes-serving/scripts/research/prepare_lucida_evidence.py:155)：155–175 行在实际原照片上执行历史 SAM2 分割，保存候选 mask、prompt 与 input SHA。本次复算没有执行此模型调用路径。

审计时源脚本 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `build-unified-data.py` | `4dcce2c93785de9bb5e2d64294791134652a07ab036b395d4ff39dd629c55694` |
| `build-source-bridge.py` | `de6eb5118bde83749f92a7aeba62877074c52f5650978e9232b42e0f32da12a8` |
| `prepare_lucida_evidence.py` | `b130b72262d0bebd4d437919ea227a6ef1d99d0a4d238c638ee676ab0ea8d12a` |

真实离线导入和来源复算使用以下输入：

```text
source_scene=/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-scene.json
report=/Users/adam/Desktop/panoptes-public/panoptes-workcell-pages/workcell-data.json
legacy_root=/Users/adam/Desktop/Tesla/ehs-spatial/runs/user-bor1-02
observation_root=/Users/adam/Desktop/Tesla/ehs-spatial/runs/bor1-components-20260909
geometry_root=/Users/adam/Desktop/panoptes-public/panoptes-serving/outputs/candidate-evaluation/lucida-replica-01
```

冻结数据 SHA-256：

| 文件 | SHA-256 |
| --- | --- |
| `workcell-scene.json` | `043ae2f0ad22c98d337d9f742acabe7c2144443dfc37500ba538dcd838d0e893` |
| geometry `manifest.json` | `c5d3d3c75dc1101e6c4c53338c18deb94ad299bcb0110a1c50e924712f1567d4` |
| geometry `evidence/objects.json` | `a2b58084e48adf758cf1ac3609d1a2e564bac139768688e6308da5bb25432760` |
| geometry `evidence/floor.json` | `83809689ab54b04df4b40737bead374aa50b6eeb8cae94180ba77bf7443d8ced` |

## 仍存在的跨照片身份歧义

这次恢复没有宣称 68 个业务对象已经全部完成物理实体去重。58 个补充 regions 的源记录仍是独立的单帧观测；不同生成途径可能产生相互重叠的实例候选，标签相同不能直接建立身份。

只读使用现有 `associate_observations`、发布包 58 张 canonical masks 与冻结 geometry 进行核查：816 个跨帧候选对中，35 对达到既有几何门槛，7 对被算法接受，28 对仍有竞争歧义；形成 44 个 singleton 与 7 个双观测 group。7 对涉及的 14 条 observation 互不重叠，但算法接受并不等于物理身份已经验证。其中 `object_6171ea7083775e2886f9b18e`（safety fence）与 `object_03cdc58071cc80458437601e`（sloped surface）可能涉及不同命名或部分/整体，不能仅凭几何重叠确认。**这些 7 对没有用于本次 entity 合并。**

本次实现没有按名称推断身份，没有把模型投影当作观测，没有将缺少 mesh 当作缺少对象。任意新输入也不能仅因存在一个 `views` 字段就证明它是照片观测；本记录对当前冻结数据的来源作了具体追溯。当前 imported association sourceRefs 保存了精确报告指针与照片/canonical 哈希；底层完整分割 provenance 和原身份判定仍在上述冻结源文件中。

## 本地检查与公网验收

导入修复的独立回归检查：

```sh
.venv/bin/python -m pytest -q tests/test_platform_import.py tests/test_platform_report_evidence.py
```

当次结果为 10 passed、1 skipped（PostgreSQL 依赖项）；`git diff --check` 通过。覆盖双视图同实体、已有 anchor 复用、重复同帧不增 observation、地面补观测、无 source ID 映射不编造、非单位像素变换、错误照片/变换拒绝、ID 与转换结果稳定性。本文不把这些结果扩展为所有模块的完整回归通过。

实际公网验收于 2026-09-13（America/Chicago）完成：

- Pages commit `f59652d133b0d3f05ee71b31fbb3782072745d42` 的 GitHub Pages 状态为 `built`；公网 `app.html` 与生产构建逐字节一致，加载入口 `app-DuE3mtxk.js`。
- 公网新报告为 [完整工位 · 对象与多视角证据](https://admin-wekruit.github.io/panoptes-workcell-report/app.html#/reports/f4e5ca43-543d-4624-8ea0-27aa2843e6e4)。旧报告 `f55f9704-0999-460e-863a-4c58cca86fb9` 及其旧 revision 均保持原 JSON，不被覆盖。
- 通过 Codex computer use 在公网页面点击料车：照片 1 的 observation `f98dd73a-5a57-5a77-a83a-4cf66587c5d7` 切换到照片 2 的新增 observation `d61affcf-97bd-5c70-bc56-c72d77bbf097`，entity ID 始终为 `a719e41c-bc9f-5963-b9f8-2caa031a8c09`。
- 实际展开全屏四视图，截图核对原图轮廓、完整场景 3D 高亮、CAD 编号 06 与交互平面的同一料车高亮，右侧显示三张原图及已关联的多视角身份。显示保留原候选模型放置状态、模型单位与未评估状态。
- 来源理解明确显示 31 条当前对象已定位、18 条原图可查但当前场景对应未确认、13 条缺照片绑定。历史 13 条仍作为历史归档，没有计为新的物体检测失败。
- 在公网展开上述 18 条中的按钮记录，真实裁图显示红色按钮、黄色底座及橙色检测框；再切换完整原图，保留料车选择 URL。切换 EN 后图片、展开状态与选择保留，最后恢复中文。首次控件点击超时后重新检查发现来源组处于折叠状态，展开后正常完成；没有依靠强制点击或 DOM 状态写入。
- 公网浏览器此次检查的 error/warn 日志为空；手机宽度视图标签和桌面全屏四视图均实际查看。未宣称真实 iPhone Safari 或所有对象逐项物理对应关系验收。
- 公网两份 publication、新旧 revision、报告列表与本地冻结内容精确一致。187/187 独立资产下载校验 SHA、大小、ETag、immutable 和 CORS 全通过，共 327,118,893 字节，包括旧报告独有的四个导出资产；Range/HEAD/预检通过。详见 [公网 API 验证摘要](association-public-api-verification.json)。

另执行 `tests/test_platform_reconstruction.py` 后，本次三个 Python 文件合计 27 passed、1 skipped；前端生产构建、report-context-check、report-evidence-check、entity-evidence-check 均通过。修复场景版本重新执行 Blender 导出并重开验证，当前 `.blend`、GLB、manifest、validation 四份资产已打包至最终 publication。导出任务为 `incomplete`，沿用 22 个对象缺三维资产或候选放置未确认的真实状态，未标记成完整重建。

