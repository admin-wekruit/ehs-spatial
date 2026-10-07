# Workcell：空间对象到语义的实际对照实验

[公开交互报告](https://admin-wekruit.github.io/panoptes-workcell-report/workcell-photo-direct/?view=model#semantics)

## 做了什么

保留同一份 52 对象、97 个观察、4 照片的重建。PE-Core-L 与 SigLIP2-so400m 在一个临时 Modal 容器的两张 A100-80GB 上并行推理；没有新训练、生成式 VLM 或模型重建。

借鉴 [HOV-SG](https://hovsg.github.io/) 的局部/全图特征融合及空间关联；[论文](https://arxiv.org/html/2403.17846v1) 和 [特征提取实现](https://github.com/hovsg/HOV-SG/blob/main/hovsg/models/sam_clip_feats_extractor.py) 为参考。这里使用不同编码器、既有 RGB 推断深度、每观察点采样和对象均值融合，并非原版 HOV-SG 复现或其 benchmark。未实现楼层/房间导航图。

冻结协议在 `semantic-match-protocol.json`：24 类、6 个查询；原生坐标邻域 0.04，重合门槛 0.35，特征余弦门槛 0.75。目录名称、期望类别不进入编码器或空间合并。既有目录聚合用于多视角对照；新空间组只根据观测点和图像特征建立，每组每张照片最多一条观察。

## 实测结果及限制

- 两个编码器的 6 个固定查询，首位结果都属于对应的已有目录类别。PE-Core 的“光幕”前两位是两根立柱；“围栏”前两位是两片围栏；“急停”首位为按钮。这只是本组预设查询的检索结果，不是开放世界识别准确率。
- 类别第一名与 52 个目录参考一致：PE-Core 单张 21、多视角 16、空间组 16；SigLIP2 单张 24、多视角 26、空间组 30。类别高度不均衡，25 条是地面标线；目录及用户给出的光幕身份也不是独立真值。
- 跨照片最近邻与目录身份一致：PE-Core 单张 45/97，多视角 45/97；SigLIP2 单张 39/97，多视角 45/97。加重合筛选后两者都是 **42/68**，仅保留 **68/97 = 70.1%** 的观察。61.8% 的比例提升伴随覆盖下降，正确配对绝对数比多视角的 45 下降至 42。
- 上一项是最近邻对照，仅用空间重合筛选；实际空间组额外使用 0.75 语义门槛及每照片唯一约束。不能把最近邻比例当成整簇正确率。
- 明确失败：两根光幕多视角第一名均为“普通黄色柱”；围栏第一名有光幕、墙、机器人等误判。急停按钮和机械臂相对稳定。检索“光幕”能找回它，不代表在所有竞争类别里“光幕”排名第一，也不代表保护功能得到确认。
- 查看实际裁剪可见大围栏裁剪包含背景设备，按钮输入只有较少原始像素。背景、分割、分辨率及外观/功能区别是待分离的误差来源，尚未通过单因素实验确定贡献。多视角平均不保证改善所有对象。

## 延迟与费用

成功 run-b：容器内 **47.62 s**；调用端 **225.18 s**，包含排队/冷启动/传输等。准备 2.35 s，分析 4.33 s。PE-Core 加载 29.69 s、编码进程合计 32.02 s；SigLIP2 加载 22.92 s、进程合计 28.46 s；两个编码进程并行。

原有完整重建 366.93 s 是历史运行，本次没有重跑；47.62 s 是在保存的深度、分割和模型上新增语义实验的时间，不能称为完整 oneshot latency。

初次 run-a 在推理前因编码器导入调度 SDK 失败：容器函数 3.74 s，调用 125.65 s。已将原 Encoder 原样移到 `fast_report/visual_encoder.py`，原入口和实验共用实现，并修复原入口的源码挂载；未增加依赖。

按 [Modal 2026-10-02 标价](https://modal.com/pricing)，请求 2×A100-80GB、8 CPU、32 GiB 为 $0.00156384/s。成功函数窗口估算 $0.07447；两次函数窗口合计 $0.08032。按两次完整调用窗口粗估 $0.54864，包含未必计费的等待，也不包含构建、存储等项目；**实际账单未知，不能当作精确消费额**。公开 `semantic/` 下保留成功及失败 ledger。

## 如何接 EHS

链路为：照片/3D 对象 → 语义候选与跨照片身份 → 尺度、地面及空间事实 → 功能/危险源/作业背景 → 有来源和版本的适用要求 → 检查证据与判定。

本轮只建立可追溯对象级入口，UI 可查实际支持裁剪，并显示可用条件几何和缺失证据。`policyContext.applicability=unknown`、`machineResult=null`；未生成正式规则 finding、阈值、法规适用性或合规判断。候选检查主题只由两编码器的多视角第一名产生，因此认成普通柱子的光幕没有自动提升为保护装置。

地面、模型、物理测量 JSON 完全保留。物理尺度仍未验证；现有 18.55/24.71 cm 是模型端点在条件比例下的估计。语义实验不改善相机、深度、端点或米制精度。机器人不同照片姿态不能合成静态危险包络。

## 复现与发布

源码分支：`codex/workcell-photo-speed`。输入：保存的 `workcell-ground-caliper-2026-10-02`；成功输出 `workcell-semantic-match-2026-10-02-b`，初次失败目录没有 `-b` 后缀。输入哈希在 `input-manifest.json` 与结果 protocol 内。

```sh
PYTHONPATH=.:scripts:modal_apps python scripts/check_workcell_semantic_match.py
PYTHONPATH=.:scripts:modal_apps modal run modal_apps/workcell_semantic_match.py \
  --root "$BASELINE" --out "$FRESH_OUTPUT" \
  --config docs/workcell-photo/semantic-match-protocol.json
PYTHONPATH=.:scripts:modal_apps python scripts/workcell_semantic_report.py \
  --baseline "$BASELINE" --experiment "$FRESH_OUTPUT" --page "$PAGES/workcell-photo-direct"
cd web
node node_modules/typescript/bin/tsc --noEmit
node node_modules/vite/bin/vite.js build --config vite.photo.config.ts
```

**2026-10-03 起上面的 `workcell_semantic_report.py` attach 命令已移除。** 把实验输出放进运行目录的 `semantic-experiment/`（oneshot 用 `--semantic-protocol` 自动完成），`finalize`/`build()` 按内容绑定到该 revision，`workcell_photo_revisions.py` 打包页面；前次失败的账本放在实验目录的 `previous-attempt-spend-ledger.json`。见 [REVISION-SYNC-2026-10-03.md](REVISION-SYNC-2026-10-03.md)。以下为历史说明：若需显示前次失败成本，在 attach 命令加 `--previous-run "$FAILED_OUTPUT"`。输出目录必须全新，禁止覆盖旧账本。把 `web/dist-photo/photo.html` 更新为发布目录的 `index.html` 和 `photo.html`，复制生成的 assets；其余模型与网站文件夹保持现状。

验证：CPU 数据流及标签置换、不合并远处同类、每照片唯一、无效点拒绝、旧 embedding 拒绝检查；Encoder 移动前后 AST 一致；TypeScript 与照片报告构建；独立审查；浏览器查询定位、支持裁剪切换照片及 52 模型加载。

下一步应先分离背景/分割与融合带来的误差，建立独立标注并选择有可见功能证据的代表对象。保留原有三项物理 TODO（多视角真实端点、三尺寸联合标定、完整流程独立验证），不因本实验关闭。
