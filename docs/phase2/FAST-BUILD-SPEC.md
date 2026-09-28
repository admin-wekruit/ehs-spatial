# 快速版搭建规格：上传 30 s 视频，报告各层陆续出现（2026-09-28）

**依据：** `FAST-PATH-PLAN.md`，E1–E8 的结果（`m3/results` 的 `FAST-PATH-RESULTS.md`），以及后续实验 E2b、E5b、E6b、E9 和它们的复核。
**标注：** [M] 实测；[M×n] 用实测单价乘以数量；[E] 估计；**未测** 表示没有任何测量，不拿估计值去补。
**这份规格只管快速版：** 当前目标只有速度。像 HomeBody 那样的演示效果以后再做。

## 0. 目标和计时口径

- **计时起点：** MP4 字节已经在容器里（`run()` 收到参数的那一刻，记为 t0）。
- **计时终点：** 每一层各自**写完**，也就是 Volume commit 返回。另外单独记两个时间：
  - 发给本机（`sent_s`）；
  - 查看器第一次取到（`served_s`）。
- **冷启动不算在分析时间里，但要单独记录。** 以后会部署到本地机房。
- **不常驻：** `min_containers=0`。一次开机里连跑几段视频，第一次调用单独标出来。
- **硬件：** 一个 Modal 容器，`gpu="A100-80GB:2"`，开 MPS。GPU 型号每次都要记下来：PCIe 和 SXM4 都可能分到。
- **批处理：** 能合成一批的一律合成一批。
- **显存：** 每个阶段、每张卡都记峰值；超过 90%（72 GB）就打标记。
- **质量优先：** 任何降低质量的捷径，要先证明质量不变才能用。
- **结果标注的四类：**
  - **observed：** 直接来自像素，例如视频、掩码。
  - **estimated：** 模型从观测估出来的几何，例如位姿、深度、网格、尺度。
  - **inferred：** VLM 给出的名字和事件。
  - **generated：** SAM 3D 模型和泼溅。它们只用于显示，不参与任何测量。
- **尺度：** 快速版的尺度来自地面规则，外加假设的 1.6 m 相机高度，一律标"估计"。
  - 不写任何米制结论。
  - 规则判定碰到尺度，一律给 NEEDS_REVIEW。`PeopleLoop` 已经这样处理了。
- **许可：** DA3-GIANT 是 CC BY-NC，界面上要标"研究许可"。

## 1. 已验证的设置（直接照搬）

| 部件 | 设置 | 依据 | 代码来源分支 |
|---|---|---|---|
| 常驻核心 | 一个类装下全部模型。SAM 3 的活放进一个优先级队列，两张卡都去领。解码时就开始喂 SAM 3。切点检测放在 12 个进程里 | E9：双卡 20.4 s，单卡 37.5 s，峰值 35.5 / 48.0 GB [M] | `m3/fu-e9-onegpu`（**从这里起步**） |
| 相机和深度 | DA3-GIANT-1.1 any-view，504×280，cam 头，每个镜头一次前向 | E1：ATE 0.043 m；E9：1.2 + 6.4 s [M] | `m3/exp-e1-e7-geometry` |
| 融合 | Open3D CUDA VoxelBlockGrid，3 cm TSDF 加点云；掩码抬升放在 GPU 上 | E7：0.5 s 和 0.1 s [M] | 同上（`m3_exp_geometry.py`） |
| SAM 3 | bf16。每帧只编码一次图像；每套词表只编码一次文本；每次前向 80 个（帧，词）对；分数下限 0.3；不设每词 12 个的上限 | E2/E2b：约 0.05 + 0.0075 × 词数 s/帧 [M] | `m3/exp-e2-e6-segment`、`m3/fu-e2b-vocab` |
| 每段视频的词表 | 默认：Qwen3-VL-8B 调用一次（v1 提示），取前 50 条，再加 8 个 EHS 核心词。可选：Gemini 两次并行调用（3 帧和 5 帧），合并后加核心词 | E2b：89% / 80–85%，单次运行，调用之间相差约 ±10 个百分点 [M] | `m3/fu-e2b-vocab` |
| 人 | SAM 3 'person'，5 fps；包含去重（`fast5_dedupe`）；`ehs_spatial.live_people.PeopleLoop` 关联，带限速 | E3：路径差约 1 cm [M] | `m3/exp-e3-people` |
| 视频轮廓 | 物体关键帧（约 1.7 fps）上用 SAM 3 掩码，标 `segmented`；其余 5 fps 关键帧用 'pair' 投影，标 `projected` | E6b：3.6–4.1 ms/帧 [M] | `m3/fu-e6b-outlines` |
| 事件 | Qwen3-VL-8B 走 vLLM，eager 模式，`gpu_memory_utilization` 0.35，两个窗口一起发 | E9：10.1 s，图像是新请求 [M] | `m3/fu-e9-onegpu` |
| SAM 3D | s1cfg12：stage 1 保留 CFG 跑 12 步，stage 2 走 shortcut 4 步。每次 3.49 s。GPU0 上 2 个进程，开 MPS（1.63×，53.5 GB） | E4 [M] | `m3/exp-e4-sam3d` |
| 泼溅 | 单卡快速 step（A100 上 36.5 → 11.5 ms），种子用 DA3 点、2 cm 体素，上限 500k | E5b：A100 上 120 s 为 27.48 dB，180 s 为 27.99 dB（DROID 相机）[M] | `m3/fu-e5b-splat` |
| 写层 | 按内容寻址的 blob 加补丁 JSON，每批做一次 Volume commit | E8b：写一层 1.5–2.8 s，从开始写到可读 2.7–5.6 s [M] | `m3/exp-e8-import` |

**复核提出的三个问题，这份规格已经处理：**
1. **20.4 s 只是核心层的下限。** 那次词表在 t≈0 就已知，没有 SAM 2，也没有泼溅。第 4 节按实际要做的活重新排班。
2. **E2b 的召回主要靠泛词撑起来。** 对照组（固定的泛词表）没跑，也没有要求 SAM 3 的词和名字对上。这两件由 D 在三段视频上补测（第 10 节），测完才冻结默认词表。
3. **E6b 的"通过"是换了指标才算过的。** 轮廓层只标 `projected`，不宣称精度。D 在 SAM 3 掩码上补测，并补上"用人的掩码切掉投影轮廓"这一步（第 10 节）。

## 2. 模块布局

```
modal_apps/fast_report.py      A  一个 Modal app：镜像、FastReport 类（boot / run）、CLI 入口
fast_report/__init__.py
fast_report/core.py            A  解码、切点、关键帧、DA3 按镜头、地面尺度、TSDF+点云、人（PeopleLoop）
fast_report/segment.py         A  SAM 3 队列（两卡动态分）、分两波的词表、洪水式掩码处理、抬升+合并+命名、轮廓
fast_report/vlm.py             A  vLLM 边车、每段视频的词表（Qwen / Gemini 中转 / 站点缓存）、事件
fast_report/sam3d.py           B  SAM 3D 进程池（GPU0，MPS）、并行闸门、显示网格；它自己的 venv 配方
fast_report/splat.py           B  泼溅进程（GPU1）：预览加后台续训；它自己的 venv 配方
fast_report/layers.py          C  补丁存储写入器、本机镜像、本机 HTTP 端点
fast_report/instrument.py      D  时钟、阶段、每卡显存采样、run.json
scripts/fast_report_eval.py    D  和交付的完整报告对照质量
scripts/fast_report_bench.py   D  三段视频的运行框架
web/src/LiveReport.tsx         C  新路由 #/live/<reportId>
web/src/live-report.ts         C  补丁 → SceneDocument 适配器（加一个 node 自检）
```

- 每个文件只有一个负责人。
- `fast_report.py` 归 A，其他人只在第 12 节列出的挂钩处被它调用。
- 没有别的新文件。**先复用已有代码：**
  - `m3_exp_geometry.fuse / edge_filter / align_sim3`
  - `detect_shot_cuts`
  - `video_events`
  - `live_people.PeopleLoop`
  - `complete_video_objects` 的闸门函数
  - `splat_train` 加 E5b 的快速 step
  - `native-viewer.ts`、`VideoView.tsx`、`splat-layer.ts`

## 3. Modal app 和 CLI

```python
app = modal.App("panoptes-fast-report")            # 只用临时 `modal run`，不部署
@app.cls(image=image, gpu="A100-80GB:2", cpu=32, memory=160 * 1024, volumes=VOLUMES,
         timeout=3600, retries=0, max_containers=1, scaledown_window=60)
class FastReport:
    @modal.enter()
    def boot(self): ...
    @modal.method()
    def boot_info(self) -> dict: ...                  # 冷启动各段时长，只记录
    @modal.method()
    def run(self, mp4: bytes, site: str, report_id: str, options: dict) -> Iterator[dict]: ...
```

**`VOLUMES`：**
- E9 的三个：`moge3-hf-cache`、`sam3-hf-cache`、`panoptes-vlm-cache`；
- SAM 3D 权重：`panoptes-sam3d-weights`；
- 补丁存储：`panoptes-fb-layers`（第 7 节）。

**boot（冷启动，不计入分析时间）按这个顺序：**
1. 先起 MPS 守护进程，必须在任何 CUDA 上下文之前（E9 的代码）。
2. 在 GPU1 上起 vLLM，占 0.35。
3. 起切点用的进程池（12 个进程；CPU 多了就再加）。
4. 起闸门用的进程池。
5. DA3 装到 GPU0。
6. SAM 3 在 GPU0、GPU1 上各放一份。
7. 在 GPU0 上起 SAM 3D 的 2 个工作进程：各自载入权重，热跑一次。E4 实测：载入 47–58 s，首次调用 14 s。
8. 在 GPU1 上起泼溅工作进程：import gsplat，跑几步真实 step 让 JIT 编译完。
9. 所有模型都按真实形状热跑一次，然后 `empty_cache`。
10. 冷启动预计 100–130 s [E]。已知的最长一段是 vLLM，83–107 s [M]。

**`run()` 是生成器：**
- 写入器每写好一批，就把它的事件原样 yield 出去：先发补丁和 blob（记 `sent_s`），commit 返回后再发一条 `written`（第 7 节）。
- 最后一个事件是 `run.json`。
- `options` 的字段：
  - `vocab`：`qwen` 或 `gemini`；
  - `vocab_queue`：选 Gemini 时，本机用 `modal.Queue.ephemeral()` 把词表送回来；
  - `splat_preview_s`：默认 120；
  - `background_s`：后台层最多再跑多久，默认 0；
  - `eval_holdout`：D 用；
  - `client_has`：本机已有的 blob 的 sha256，不再回传。

**CLI：**

```
modal run modal_apps/fast_report.py --video PATH --start S --end E --site NAME \
    [--vocab qwen|gemini] [--serve] [--background-s 0] [--out RUNS/fb-<key>-NNN]
```

- **裁剪：** 本机按 `prepare_video_clip.full_video` 的同一规则裁出 [S, E)：`first_frame = round(S × fps)`，用 cv2 的 avc1 编码，1280 宽。这样第 i 帧就是片段的第 i 帧，和交付报告逐帧对齐。裁剪不计时。
- **等开机：** 调 `boot_info()` 等容器就绪，把冷启动记入 run.json 的 `boot`。
- **跑分析：** 调 `run.remote_gen(...)`。每个补丁先核对 sha256，再镜像到 `--out`。
- **可选端点：** 加 `--serve` 时，在 127.0.0.1:8793 起本机端点（第 7 节），并打印查看器地址 `http://127.0.0.1:5173/app.html#/live/<reportId>`。
- **报告号：** `fb-<site>-<video sha256 前 8 位>-<unix 秒>`。
- **从浏览器上传以后再做。** 做法是端点收 POST，再走同一条路径。

## 4. GPU 排班

t 从 MP4 进容器算起，热启动。

| t（s） | GPU0（geo） | GPU1（seg） | CPU / 其他 |
|---|---|---|---|
| 0–1.7 | SAM 3 {人, 地面}，从队列领 | SAM 3 {人, 地面}，从队列领 | 解码；每 6 帧取最清晰的一帧作关键帧，8 帧一组送上两张卡 [M 1.7 s]。写 `video` 层 |
| 1.7–~11.5 | 同上；另外第 1 波词表（核心 8 词加站点缓存）按物体关键帧跑，每 3 个关键帧取 1 个 | 同上；vLLM 同时生成词表：均匀取 5 个关键帧，Qwen 约 9.4–9.8 s [M] | 切点在 12 个进程里算，约 7 s 出结果 [M] |
| ~8.6–15.7 | DA3：每个镜头一次前向，1.2 + 6.4 s [M] | SAM 3 | — |
| 15.7–16.2 | 地面平面、尺度、TSDF+点云、人 [M 0.3 s] | SAM 3 | `PeopleLoop` 关联 |
| ~16–19 | 从队列领第 2 波词表 | 第 2 波（VLM 新给的词），复用已缓存的视觉特征；vLLM 跑事件，约 10 s [M] | 写 `cameras`、`room`、`people`，一次 commit，约 **18–19 s** [M×n] |
| ~19–25 | 第 2 波 | 第 2 波 | 50 帧 × 0.0075 × 约 50 词 ≈ 19 GPU·s [M×n]，两卡分 → 约 23–26 s 清空 [E] |
| ~25–28 | 抬升+合并+命名、轮廓（约 1 s [E]） | 空出来，交给泼溅 | 写 `objects`、`outlines`、`events`，一次 commit，约 **27–30 s** [E] |
| 26 → | SAM 3D，2 个进程开 MPS：前 30 个物体各试 1 次，每次有效 2.09 s → 约 63 s [M×n] | 泼溅预览：120 s 训练预算 → 约 **150–155 s** 写出 [E] | 闸门：prepare 用内存里的视角，assess 放进进程池；接受一个，写一次 `models` 补丁。第一个模型约 **60 s** [E]，30 次尝试全部判完约 **120 s** [E] |
| 155 → | SAM 3D 在后台补：其余视角、种子 43、更多物体 | 泼溅续训：300 / 600 / 1200 s 各出一个快照 → `splat` 层升级为 full；A100 上到约 31 dB 要约 30 min [E] | 只在 `background_s > 0` 时跑 |

- **动态分 SAM 3：** 照搬 E9。队列里 {人, 地面} 在前，词表帧在后。
  - GPU0 在切点出来之前、DA3 和融合做完之后都去领活。
  - 第 2 波的活直接加进同一个队列。
  - 泼溅要等队列清空才上 GPU1。复核说过，一开始就把泼溅放上 GPU1，会让核心退回 29–37 s；泼溅晚开约 9 s 换来物体层早出。
- **词表分两波（缓存）：**
  - 第 1 波在 t=0 就已知：核心词加站点缓存。
  - 第 2 波只跑 VLM 新给的词。
  - 物体关键帧的 FPN 特征在第 1 波时已经缓存（E9 已经这样做，每帧约 56 MB [E]，50 帧约 2.8 GB）。
  - 最终的词集合等于三者的并集，所以结果和一次跑全表相同，只是顺序变了。
- **MPS：** 整个容器都开。
  - 核心阶段没有收益（E9：39.4 s 对 40.3 s）；SAM 3D 两个进程有 1.63× 的收益（E4）。
  - E9 在开 MPS 时跑过 vLLM，没有问题。
- **GIL：** 两个 SAM 3 线程抢 GIL，E9 里解码从 1.7 s 拖到 4.5 s。所以 SAM 3D 和泼溅都放在子进程里跑。
- **切点是剩下的关键路径（约 7 s，挡在 DA3 前面）。** A 可以试：cpu=32 时开 24 个进程。目标约 2 s；做到的话核心能到 16–17 s [E]。

## 5. 显存预算（每卡 80 GB，超过 72 GB 打标记）

| GPU | 阶段 | 常驻 | 峰值来源 | 预计 / 实测 |
|---|---|---|---|---|
| 0 | boot 之后 | DA3 + SAM 3 + 各上下文 15.4 GB [M，E9]；再加 SAM 3D 2 个进程空闲时的占用（**未测**） | — | 15.4 GB + SAM 3D 空闲 |
| 0 | 核心（0–25 s） | 同上 | DA3 113 视角 23 GB 保留 [M]；SAM 3 每次 80 对 26–39 GB [M]；E9 全卡峰值 35.5 GB（38 对）[M] | 35.5 GB + 约 10 GB（批变大）[E] + SAM 3D 空闲 → **最紧的一段** |
| 0 | SAM 3D（26 s →） | DA3、SAM 3 权重 | 2 个进程开 MPS，53.5 GB [M] | 约 69 GB（86%）[M+M] |
| 1 | boot 之后 | vLLM 0.35 ≈ 28 GB，加 SAM 3，共 29.4 GB [M] | — | 29.4 GB |
| 1 | 核心 | 同上 | E9 全卡峰值 48.0 GB [M] | 48 GB + 约 10 GB [E] |
| 1 | 泼溅（26 s →） | vLLM + SAM 3 | gsplat 500k，帧放在 GPU 上（**未测**） | 29.4 GB + gsplat，预计 ≤15 GB [E] |

- **vLLM：** `gpu_memory_utilization=0.35`（E9 的值），`max-model-len 16384`，`max-num-seqs 4`，eager 模式。
- **A 的第一次整跑就要量出所有"未测"。** 超过 72 GB 时，按这个顺序调：
  1. GPU0 上 SAM 3 每次前向的对数从 80 降到 40。
  2. 核心阶段把 SAM 3D 的权重放在 CPU 锁页内存里，核心做完再搬上 GPU，约 1–2 s [E]。
  3. DA3 跑完最后一个镜头后，把权重移到 CPU。
  4. vLLM 占比降到 0.30（E9 的建议）。
  5. 事件做完后让 vLLM 进入 sleep 模式，给泼溅腾出约 28 GB。只在泼溅需要时才做。

## 6. 图层和格式

- 每层可以出多个版本，查看器只用最新的那个。
- 下表"约何时写完"一列的时间都是 [E]，由第 4 节推出。

| 层 | 约何时写完 | 类别 | data（小 JSON，内联） | blob |
|---|---|---|---|---|
| `video` | 2–3 s | observed | fps、帧数、尺寸、sha256、[S, E) | MP4 |
| `cameras` | 18–19 s | estimated | 每个镜头：关键帧号、时间、`c2w`（估计米）、504×280 下的 K、源尺寸；`scale` = {mpu, 来源：地面平面 + 假设相机高 1.6 m, status: estimated}；许可标注 | — |
| `room` | 18–19 s | estimated | 每个镜头：三角形数、点数、坐标系 id | 网格：`panoptes-mesh-v1`，stride 9（xyz、法线、rgb，float32，uint32 索引）；点云：GLB 点，带 `pointSizeNative` |
| `people` | 18–19 s | observed + estimated | 轨迹：[{t, 帧, 脚点 xyz}]；`PeopleLoop` 的规则行，带尺度已被闸门挡住时的 NEEDS_REVIEW；备注"快速版没有非人移动物" | 每条轨迹一个带状网格（packed mesh v1） |
| `objects` | 27–30 s | estimated（框）+ inferred（名字） | 每个物体：id、镜头、词、投票、看到的帧数、质心、框（估计米）、最佳视角、状态"检测词，未核" | — |
| `outlines` | 27–30 s | observed（segmented）/ estimated（projected） | 用 `VideoView` 的分析格式：{width, height, frames:[{timeSec, endTimeSec, sourceFrame, objects:[{entityId, label, polygons, source}]}]}，1280×720 像素；投影出的轮廓先用该帧人的掩码切掉 | 分析 JSON（大于 1 MB 时放 blob） |
| `events` | 25–30 s | inferred | `video_events.parse` 的窗口，外加原文 | — |
| `models` | 约 60 s 开始，每接受一个就追加 | generated，只用于显示 | 每个模型：物体 id、在镜头坐标系里的变换、闸门记录的摘要 | GLB，4 万面显示网格 |
| `splat` | 预览约 150 s；full 在后台 | generated，只用于显示 | 格式 splat32、数量、坐标系 id、kind（preview / full）、训练秒数、步数 | `.splat`（splat32，按重要性排序） |
| `timing` | 每次 commit 都重写 | — | 到目前为止的 `run.json`（第 8 节） | — |

**关于 `room` 的一个待定项：** E9 两个镜头的网格合计 189 万三角形，打包后约 59 MB [E]。
- 如果本机看到的时间比写完晚 5 s 以上，C 再加一个 GPU 上抽稀的 `room` 版本，先发。
- 要不要做，以测量为准。

## 7. 补丁存储和 HTTP 端点（来自 E8b）

**存储：** Modal Volume `panoptes-fb-layers`，调用时按需创建。

```
blobs/sha256/<hex>                                 # 永不重写；已存在就跳过
reports/<reportId>/patches/<seq:06d>-<layer>.json  # 补丁
reports/<reportId>/run.json                        # 最后一版计时
sites/<site>/vocab.json                            # 站点词表缓存（第 11 节）
```

补丁的格式：

```json
{"schema": "panoptes-fast-patch-v1", "report": "...", "seq": 7, "layer": "objects", "version": 1,
 "status": "estimated", "labels": ["名字是检测词，未核", "尺度为估计值（地面 + 假设 1.6 m 相机高）"],
 "sent_s": 26.8, "written_s": 28.3, "data": {...},
 "blobs": {"mesh": {"sha256": "...", "bytes": 123, "mediaType": "application/octet-stream", "format": "panoptes-mesh-v1", "byteLayout": {...}}}}
```

**写入器（`layers.Writer`，一个线程）：**
- `put()` 立即返回，从不阻塞流水线。
- 写入线程每轮把积压的所有补丁一起处理：
  1. 先写 blob；
  2. 再写补丁 JSON；
  3. 马上把事件交给 `run()` 去 yield，记 `sent_s`；
  4. 执行一次 `volume.commit()`，把 `written_s` 补进 `timing` 层。
- 多个层同一时刻到齐，就只 commit 一次。例如第 4 节 18 s 那一批。

**本机镜像和端点：**
- CLI 把 yield 出来的补丁和 blob 核对 sha256 后，写到 `runs/fb-<key>-NNN/<reportId>/`。
- `layers.serve(root, port=8793)` 用标准库的 `ThreadingHTTPServer`，只绑 127.0.0.1，提供两个接口：
  - `GET /fast/reports/<id>/patches?after=<seq>`：返回补丁数组；
  - `GET /fast/blobs/<hex>`：带 `Cache-Control: immutable`。
- 每个补丁第一次被取走时，记 `served_s`（两个时钟都记 unix 时间）。
- `web/vite.config.ts` 加一条代理：`"/fast" → http://127.0.0.1:8793`。
- 没有公开 URL，也不需要任何密钥。
- 以后要跨设备演示时，把同一个 `serve` 挂成 Modal web 端点读 Volume（E8b 的 `serve_layers`），并开 `requires_proxy_auth`。现在不做。

## 8. 计时和显存记录（`instrument.py`，run.json）

```python
clock = instrument.Clock()                       # 在 run() 第一行建：t0 = MP4 在容器里
with clock.stage("da3.shot1", gpu=0, n={"views": 113}): ...
clock.external("sam3d.generate", gpu=0, start_unix=..., end_unix=..., n={...})   # 子进程回报
vram = instrument.Vram([0, 1]); vram.start()     # 每 50 ms 采一次整卡用量（mem_get_info，含所有进程）
report = clock.report(vram)                      # 每个阶段都附上窗口内每卡的峰值和 >90% 标记
```

```json
{"schema": "panoptes-fast-run-v1", "report": "...", "site": "me340",
 "video": {"sha256": "...", "frames": 899, "fps": 29.97, "wh": [1280, 720], "window_s": [165, 195]},
 "hardware": {"gpus": ["NVIDIA A100-SXM4-80GB", "..."], "cpu": 32, "mps": true},
 "boot": {"...": "冷启动各段，不计入分析", "first_call_after_boot": true},
 "clock": "从 MP4 字节进容器算起的秒数",
 "stages": [{"stage": "sam3.person", "where": "gpu1", "start_s": 0.2, "end_s": 9.8, "s": 9.6,
             "peak_gb": [30.1, 55.2], "over_90": [false, false], "n": {"chunks": 9}}],
 "gpu_peak": [{"gpu": 0, "total_gb": 80, "peak_gb": 68.9, "at_s": 41.2, "stages_active": ["sam3d.generate"]}],
 "flags": ["gpu0 74.3 GB > 90% at 22.1 s (sam3.vocab.wave2, sam3d idle)"],
 "layers": [{"layer": "cameras", "seq": 2, "version": 1, "sent_s": 17.1, "written_s": 18.6, "served_s_unix": 0, "bytes": 0}],
 "usd_estimate": 0.0}
```

- **阶段名是固定的一套，方便跨次比较：**
  - `decode`、`cuts`
  - `sam3.person@gpuN`、`vlm.vocab`、`sam3.vocab.wave1@gpuN`、`sam3.vocab.wave2@gpuN`
  - `da3.shotK`、`scale.shotK`、`tsdf.shotK`、`people`
  - `lift`、`outlines`、`vlm.events`
  - `sam3d.prepare`、`sam3d.generate`、`sam3d.assess`、`sam3d.decimate`
  - `splat.preview`、`splat.full`
  - `write.<layer>`
- **峰值指的是阶段时间窗内的整卡峰值。** 几个阶段重叠时，它们共享同一个峰值，`stages_active` 会列出当时在跑的阶段。
- 另外记每个进程 torch 的 `max_memory_reserved`，用来分摊：主进程、SAM 3D 的两个进程、泼溅进程各一份。
- 在 MPS 下，如果 `mem_get_info` 取不到整卡数字，就改用 NVML（`nvidia-smi --query-gpu=memory.used`）。

## 9. 查看器

**建议：在现有的 web 应用里加一个新路由 `#/live/<reportId>`，不另写 three.js 页面。** 需要的东西大部分已经有了：
- `mountSceneViewer`：能画打包网格、GLB 点云、box primitive、相机路径和点选；
- `setSplats`：splat32 流式渲染；
- `VideoView`：带实体轮廓的视频，时间和 3D 联动。

**C 要做的：**
1. **`live-report.ts`：** 每 500 ms 轮询一次端点，拿到最新的补丁，拼成一份最小但合法的 `SceneDocument`：
   - 每个镜头一个坐标系；
   - `cameras` 加上图像 asset，带 `sourceFrame` 和 `videoTimestamp`；
   - `room` 转成 observed_surface 和 point_cloud 两种表示；
   - `objects` 转成 box primitive，带标签和状态；
   - `people` 转成带状网格；
   - `models` 转成 generated_mesh；
   - `splat` 转成 `gaussian_splats` 注释；
   - `video` 和 `outlines` 转成 `video_replay` 注释。
   - asset URL 就是 `/fast/blobs/<hex>`。
   - 附一个 node 自检：用 E9 run 003 的输出拼出来的文档，要能通过 `setScene` 的检查。
2. **`LiveReport.tsx`，四块面板：**
   - 3D：房间、物体框、名字，点选后显示信息卡；人的路径；模型和泼溅到了就出现；
   - 视频加轮廓：`projected` 画虚线，`segmented` 画实线；
   - 事件时间线：点一下跳到视频对应时刻；
   - 计时面板：每层的 written_s 和 served_s；每个阶段每卡的峰值，超过 90% 标红；冷启动单独一行，灰色。
3. **三处小改：**
   - `App.tsx`：加路由；
   - `VideoView`：`resolveAsset` 改成可以从 prop 传入，并按 `source` 区分画法；
   - `native-viewer.ts`：`setScene` 现在只要 asset 集合一变就 `release()` 全部。改成保留（entity, representation, asset）没变的 GPU 缓冲，这样新层到来时房间不用重新上传。

**信息卡：**
- **物体：** 名字，标"检测词，未核"；其他候选词；看到的帧数；框的尺寸，标"估计"；是否有模型。
- **人：** 轨迹的时间段、检测次数、规则结论（尺度未测时为 NEEDS_REVIEW）。
- **模型和泼溅：** "生成的显示层，不用于测量"。

## 10. 质量对照（D，`scripts/fast_report_eval.py RUN_DIR --site <me340|samsclub-a2|walmart>`）

**参照物：** 都从 `tests/fixtures/delivered-303/<site>.json` 的节点输出里读，不写死路径。这里衡量的是和今天完整流程的一致度，不是精度。

| 层 | 指标 | 参照 | 标准（跑之前定） |
|---|---|---|---|
| 相机 | 每个镜头按 Sim3 对齐后的 ATE（m，以及占路径长度的 %）；旋转误差中位和最大值 | camera 节点的 DROID `prediction.npz` 加 metric-scale（E9 `evaluate()` 的做法） | ME340 ≤0.057 m（E1 的标准）；另外两段 ≤1.5% 路径长度，只报告，不作阻断 |
| 尺度 | 我们的米数除以参照的米数 | 同上（两边都假设 1.6 m） | 在 0.9–1.1 之间 |
| 物体（2D） | 两种召回：只看 IoU ≥0.5 的位置召回（E2 的规则）；另一种还要求 SAM 3 的词和今天的名字对上（中心词或子串） | names 节点的已命名物体和它们的掩码 | 位置召回 ≥80%；词匹配召回只报告 |
| 物体（3D） | 0.5 m 内的召回；我们的物体里有参照的比例 | object map 的质心 | 报告；和 E9（0.53）、E1 用 SAM 2 时（0.64）比 |
| 词表对照 | 同样的帧，跑一张固定泛词表（约 50 个词），比较两种召回 | 同上 | 每段视频的词表必须在词匹配召回上胜出，才保留做默认 |
| 人 | 地面上的路径差（中位、p90、最大）；R1/R2/R3 按 0.2 s 的一致率 | dynamic layer 和 tracks 节点（`e3_people_eval.py` 的做法） | 中位 ≤0.3 m；不出现 PASS 和 FAIL 互翻 |
| 轮廓 | 抽几帧中间帧，另跑 SAM 3 作参照，算 `projected` 对它的 IoU（均值和按面积加权）；切掉人之前、之后各算一次 | 快速版自己的 SAM 3（只在评估时跑） | 只报告，不宣称精度 |
| 事件 | actor、PPE、安全备注是否一致；标题的 Jaccard | events 节点 | actor、PPE 一致；Jaccard 只报告（时间窗口相同是构造出来的，不算检验） |
| SAM 3D | 接受数；闸门通过率 | SAM 3D 节点；E4 的 30 个物体 | 同一批物体上，和 E4 的 s1cfg12（7/30）差在 ±3 以内 |
| 泼溅 | 留出帧上的 PSNR、SSIM、LPIPS（`eval_holdout`：run 232 的 86 帧） | splat package 节点（30.82 dB） | 预览 ≥27.0 dB（A100 用 DROID 相机时 27.48 [M]）；full 只报告 |

**Sam's Club 的特殊情况：** 交付报告只做了 `samsclub-337-a2` 这个镜头，也就是 samsclub-337 的第 0..B-1 帧，硬链接，帧号不变。所以只对这个镜头比。

## 11. 你问的：用 VLM 做物体的概率判定加速，再加识别缓存，放在 SAM 3 前面

我把"jev / laya"理解成两类模型：
- V-JEPA 一类的视频表征模型：给出"见过没有"或"变了没有"的特征，适合做缓存的键，但不会给物体起名字；
- LLaVA 一类的开源 VLM：我们在 vLLM 里跑的 Qwen3-VL-8B 已经是这一类，而且测过。

如果你指的是别的模型，告诉我。

**能做，分两部分。**

**(a) 不改变结果的缓存，默认打开：**
- **视觉特征：** 每个关键帧只编码一次，两波词表共用（E9 已经在做）。
- **文本特征：** 每个词只编码一次（每套词表 0.015–0.018 s [M]，不用存盘）。
- **站点词表缓存：** 用 `--site`。同一现场第二次来时，上次合并好的词表在 t=0 就进第 1 波；VLM 只补新词，走第 2 波。
  - 最终词集合是并集，所以不丢召回。
  - 省掉的是等 VLM 的约 10 s（第 4 节）。
  - 第一次来的现场省不了。
- **同一段视频重跑：** 按内容寻址命中缓存，但只用于演示。计时一律关掉这个缓存。

**(b) 会改变结果的"概率判定"，默认关闭，要先证明质量不变才能打开（按实验 G1、G2 做）：**
- **G1，逐帧词门控：**
  - 用一个便宜的图文模型（SigLIP 2 或 PE-Core，每帧约 5–10 ms [E]）算每个物体关键帧和词表的相似度。
  - 每帧只让前 K 个词加核心词进 SAM 3。
  - SAM 3 每个（帧，词）对约 7.5 ms，每帧固定成本约 50 ms [M]。
  - 50 个词降到 K=20：第 2 波从约 19 GPU·s 降到约 8 GPU·s，两卡分下来，物体层约早 5–6 s [E]。
- **G2，关键帧新旧缓存：**
  - 物体关键帧和上一个物体关键帧的视野重合度，用 DA3 的位姿和深度算。超过阈值的帧不跑词表，掩码用 'pair' 投影带过来。
  - 省多少取决于相机走得多快，**未测**。
- **两个实验的标准都一样：** 三段视频上，位置召回和词匹配召回都要不低于不开时的值，差距在单次运行的噪声（±1 个物体）以内。达不到就不开。
- **上限：** 就算两个都过，物体层最多早 5–8 s [E]。泼溅和 SAM 3D 闸门决定整份报告的最后时间，这两个实验动不了它们。

## 12. 四个搭建者

- **分支：** 每人从 `fb/spec` 开自己的 worktree，然后把需要的实验分支合进来。
  ```
  git -C /Users/adam/.codex/worktrees/panoptes-phase2-video worktree add /Users/adam/.codex/worktrees/panoptes-phase2-video-fb-KEY -b fb/KEY fb/spec
  ```
- **联调：** 在各自的分支上先用桩代码顶着别人的部分，最后合进 `fb/build`（第 13 节）。

### A：核心流水线（`fb/core`，基于 E9）

- **合入的分支：**
  - `m3/fu-e9-onegpu`：起点；
  - `m3/fu-e2b-vocab`：提示词和 Qwen 采样参数；
  - `m3/fu-e6b-outlines`：`splat`、`outlines` 投影；
  - `m3/exp-e3-people`：去重和 `PeopleLoop` 的接法。
- **负责的文件：** `modal_apps/fast_report.py`、`fast_report/{core,segment,vlm}.py`。镜像也归 A，B 通过 `sam3d.with_envs(image)` 和 `splat.with_envs(image)` 往里加自己的环境。
- **基础镜像：** 改用 `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04`（`add_python="3.11"`）。
  - 原因：B 要在自己的 venv 里编译 pytorch3d，需要 devel 镜像里的 nvcc。
  - E9 的主环境（torch 2.14 cu130 等）装进系统 Python，vLLM 仍在 `/opt/vllm`。
  - A 的第一件事是确认这个组合能建成、E9 的自检能过。
- **做什么：**
  - 把 E9 的探针拆进上面这些模块，行为不变。
  - 词表分两波；Qwen 的输出截到前 50 条，采样参数照模型卡（温度 0.7，top-p 0.8，top-k 20，presence 1.5）；Gemini 中转。
  - 分数下限 0.3，不设上限，每次前向 80 对。
  - 处理洪水式掩码：
    - 同一帧里，不同词的掩码 IoU >0.8 的，只留高分的那个，词记入投票；
    - 一个掩码有一半以上落在同词更高分的掩码里，就删掉；
    - 命名时具体词优先于泛词，按 VLM 给出的顺序排。
  - 字幕带遮掉，用 `complete_video_objects.subtitle_box`。
  - 轮廓先用该帧人的掩码切掉。
  - 保留每个物体的最佳视角，以及它的 SAM 3 低分辨率掩码 logits（288²，fp16），交给 B 放大成全分辨率掩码。
- **对外接口**（都在进程内，张量在 GPU0 上）：
  ```python
  Shot = {"index", "frames": (a, b), "keys": [源帧号], "object_keys": [...], "depth_m": (k,280,504), "K": (k,3,3),
          "c2w_m": (k,4,4), "colors": (k,280,504,3), "person": (k,280,504) bool, "floor": ..., "mpu", "scale_status": "estimated"}
  Obj = {"id", "shot", "word", "votes", "frames", "centroid_m", "box_min_m", "box_max_m", "best_key", "mask_logits_lr"}
  frames_host = np.ndarray (n,720,1280,3) uint8，放在共享内存里，给泼溅进程用
  ```
- **里程碑：**
  - **A1：** 只有 `video`、`cameras`、`room`、`people` 四层（写入器和计时先用桩），ME340 热跑 ≤21 s，ATE ≤0.057 m。
  - **A2：** 加上 `objects`、`outlines`、`events`。
  - **A3：** 量出第 5 节所有"未测"的显存。
- **预算：** $10。

### B：SAM 3D 和泼溅（`fb/sam3d-splat`）

- **合入的分支：** `m3/exp-e4-sam3d`、`m3/fu-e5b-splat`（后者带 `splat_train.py` 的快速 step 改动）。
- **负责的文件：** `fast_report/{sam3d,splat}.py`。
- **环境：**
  - SAM 3D 用 E4 的配方（`sam3d_research.image` 的那套：torch 2.5.1 cu121、pytorch3d、kaolin、flash_attn、spconv），用 venv 的 pip 重新装进 `/opt/sam3d`，不能装进系统 Python，否则会和主环境的 torch 冲突。
  - 泼溅：如果 `/opt/sam3d` 里的 gsplat 正好是 `splat_train` 固定的 1.5.3，就共用这个环境；否则另建 `/opt/splat`（py3.10、torch 2.4.1 cu124）。
  - 两个都是子进程，和主进程之间用管道传。
- **SAM 3D：**
  - boot 时在 GPU0 上起 2 个进程，s1cfg12。
  - 闸门重写两处：
    1. `prepare` 直接用 A 给的 `Shot`、`Obj`，外加内存里的帧，不再读 312 个带位姿的视角；
    2. `assess` 放进进程池。
  - 闸门的判定函数一行都不改。
  - 通过的物体用 Open3D 抽稀到 4 万面，导出 GLB。
  - 排序：核心词和 VLM 列在前面的 EHS 类优先，然后按看到的帧数；前 30 个各试 1 次，其余放后台。
  - 接口：
    ```python
    sam3d.Workers(gpu=0, n=2)                     # boot 时起
    sam3d.gate(objs, shots, frames_host, clock) -> Iterator[{"object", "glb", "transform", "gate"}]
    ```
- **泼溅：**
  - GPU1，SAM 3 队列清空后开始。只练最长的镜头。
  - 先在 ME340 上比两种训练集，在同样的留出帧上选 PSNR 高的：
    - (i) 只用关键帧：视角少，但位姿和人的掩码都是精确的；
    - (ii) 用全部帧：中间帧的位姿插值，开 `splat_train` 的位姿修正，人的掩码用相邻两个关键帧的并集再膨胀。
  - 预览按 `splat_preview_s` 的预算训练，然后在同一进程里续训，出 full 快照。
  - 接口：
    ```python
    splat.Worker(gpu=1)
    .start(frames_host, shot, seeds, budget_s) -> Iterator[{"kind", "seconds", "steps", "splat32", "count"}]
    ```
- **通过标准：**
  - 闸门在 E4 的 30 个物体上，判定和原来的闸门逐个相同（用同样的输入）。
  - 快速输入下的通过率和 s1cfg12 差在 ±3 以内。
  - 每个物体 prepare ≤5 s。
  - 泼溅预览 ≥27.0 dB。
- **预算：** $12。其中包括一次约 30 min 的后台续训。

### C：写层、端点、查看器（`fb/layers-viewer`）

- **合入的分支：** `m3/exp-e8-import`。
- **负责的文件：** `fast_report/layers.py`、`web/src/{LiveReport.tsx,live-report.ts}`，以及第 9 节列的三处小改和 `vite.config.ts`。
- **开发不需要 GPU：** 用 E9 run 003 已有的网格、点云和 JSON，再加交付报告里的 `.splat`，拼出一个假的补丁流。
- **接口：**
  ```python
  w = layers.Writer(volume, report_id, clock)
  w.put(layer, data, blobs={role: (bytes, meta)}, status, labels) -> seq   # 不阻塞
  w.events() -> Iterator[dict]    # run() 原样 yield
  w.close()
  layers.mirror(event, root)      # CLI 用：核对 sha256 后写到本机
  layers.serve(root, port=8793)
  ```
- **通过标准：**
  - 写一层（写 + commit）≤3 s。
  - 端点本机取第一个补丁 ≤0.5 s。
  - 查看器能显示每一层；点选物体出信息卡；视频上有轮廓；计时面板有数字。
  - 同一个补丁流重放两次，画面一致。
- **预算：** $2。

### D：计时、质量对照、三段视频的运行框架（`fb/measure`）

- **合入的分支：** `m3/fu-e9-onegpu`（用它的 `evaluate()`）、`m3/exp-e3-people`（用它的人物评估）。
- **负责的文件：** `fast_report/instrument.py`、`scripts/fast_report_eval.py`、`scripts/fast_report_bench.py`。
- **instrument：** 第 8 节的 `Clock`、`Vram` 和 run.json，附 `--self-check`。
- **eval：** 第 10 节的整张表。参照物从 fixtures 里读。只用本机 numpy，内存要小。
- **bench：** 一次 `app.run()`、一次开机，跑三段视频（ME340 165–195 s、Sam's Club 337–367 s、Walmart 190–220 s，源文件路径取各自 `clip.json` 的 `source.video`）。
  - 每段先跑 1 次首调、再跑 2 次热调。
  - 最后一段加 `background_s=1800`。
  - 汇总成 `runs/fb-bench-NNN/summary.json`，含时间、显存、质量、花费。
- **还要补测复核提出的缺口：** 固定泛词表对照、词匹配召回、轮廓切人前后的对比。
- **预算：** $10。

**机动：** $6。四人合计上限 $40；花到 $35 就停下来问。

## 13. 集成和验收

**合并顺序：** `fb/measure` → `fb/layers-viewer` → `fb/core` → `fb/sam3d-splat`，合进 `fb/build`。
- A 的挂钩一开始是桩，这一步换成真的实现。
- 每合一个，就在 ME340 上整跑一次。

**验收**（`fast_report_bench.py`，三段视频，热启动，双卡 A100）：

| 层 | 写完的目标（从 MP4 进容器算） |
|---|---|
| cameras、room、people | ≤20 s |
| objects、outlines、events | ≤30 s |
| 第一个 SAM 3D 模型 | ≤90 s |
| 泼溅预览 | ≤180 s |

- 每个阶段、每张卡都有峰值；没有超过 90% 的，或者超过了但附有原因和已经用上的调节手段。
- 第 10 节的质量表三段视频都填满。没过的写明原因，不改标准。
- 冷启动、上传、本机看到的时间（served_s）单独列出。
- 花费写进 summary，用 `modal billing report` 对账。

## 14. 规则

- **Python：** `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python`。
- **Modal：**
  - CLI 用 `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/modal`；
  - 只用临时 `modal run`，不部署；
  - 超时写明，retries 0；
  - Mac 内存有限，计算都放在 Modal 上。
- **数据：**
  - 数据根目录是 `/Users/adam/Desktop/panoptes-public/research-notes/phase2`；
  - 只新建 `runs/fb-*` 目录；
  - 不碰 `/Users/adam/.codex/worktrees/panoptes-phase2-video` 本身；
  - 不打开 `*/.platform/imports/*`。
- **日志和密钥：**
  - 日志过滤：`grep -v -i "capabilit\|token\|secret"`；
  - 文件和 URL 里不放密钥；
  - Gemini 只走 `name_video_entities.py` 的机制，本机不碰密钥。
- **提交：**
  - 提交信息结尾：`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`；
  - 不推送。
- **每次运行：**
  - 写明 GPU 型号、花费，以及哪些数字是 [M]、哪些是 [E]；
  - 复核能从保存的输出里重算出所有主要数字。

## 15. 不在快速版里，以后在后台补

- LingBot 稠密点和推测地面；
- 稠密闸门的独立检验；
- SAM 2 全量物体；
- 30 fps 的人轨迹；
- 非人移动物（SAM 3.1 stride 3）；
- 贴图和补洞；
- 插入镜头的配准；
- RecGen；
- 商用档的位姿模型（DA3-BASE 没过；MapAnything-apache 和 VGGT-1B-Commercial 还没测）；
- 从浏览器上传；
- 跨设备的查看端点。
