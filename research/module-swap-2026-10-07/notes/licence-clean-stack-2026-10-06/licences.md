# 许可干净的 on-prem 组件表 / Licence-clean on-prem stack (2026-10-06)

只读调研：一手来源（GitHub LICENSE / README / 源码、Hugging Face 模型卡与 `api/models` 元数据、PyPI、arXiv、vcpkg port 清单）。没有改代码，没有跑 Modal，**花费 $0**。2026-10-06 读取。**不是法律意见。**
Read-only research from primary sources: GitHub licence files, READMEs and source, HF cards and `api/models` metadata, PyPI, arXiv, vcpkg port manifests. No code changed, no Modal runs, **$0 spent**. Read on 2026-10-06. **Not legal advice.**

本文补 `onprem-license-audit-2026-10-05`（它只看了 HF 卡）和 ONPREM.md「许可」一节；几何精度数字来自 `geometry-licence-ab-fair-2026-10-05`。
This extends `onprem-license-audit-2026-10-05` (which read HF cards only) and the Licences section of ONPREM.md. Accuracy numbers come from `geometry-licence-ab-fair-2026-10-05`.

## 结论 / Verdict

**中文**
- **除几何主干外，全套都能 on-prem 商用，权重可以镜像到我们自己的存储、随交付件发给客户，客户不需要 Hugging Face。**
  - 掩码：SAM 3 / 3.1（SAM License，有条件可商用）；备用 SAM 2.1（Apache-2.0）。
  - 检测框：OWLv2（Apache-2.0）或 Grounding DINO（Apache-2.0；要连同 `bert-base-uncased` 一起镜像，它运行时会去 HF 下载）。
  - 补全：SAM 3D Objects（SAM License，替代 RecGen，已在公平 A/B 打平）。
  - 单目核查：MoGe-3（MIT）。
- **几何主干（Pi3X 的位置）仍然没有"许可清楚 + 精度已证"的替代。** 候选按建议顺序：
  1. **VGGT-1B-Commercial**：许可清楚，可商用（README：只排除军事用途），可再分发（同协议 + 附协议副本）。但还**没在公平 A/B 上测过**；HF 人工审批；它的 AUP 里有一条"关键基础设施、交通、**重型机械运行**"相关的禁用（在"有人身伤害风险的活动"项下）。我们做的是机器人工位的安全审计，不是操作机械，但这条要法务看一眼。
  2. **DA3-LARGE-1.1**：精度已证（MAE 1.2 cm，Pi3X 1.4 cm），但许可冲突仍在：HF 卡 apache-2.0，官方 README 表 CC BY-NC 4.0。GitHub issue #259（2026-06-01）问的就是这个，至今 0 回复。要用户向 ByteDance Seed 拿书面确认。
  3. **全宽松许可的经典 SfM**：COLMAP（BSD-3）+ ALIKED 或 DISK + LightGlue（BSD-3 / Apache-2.0；COLMAP ≥ 3.13 已内置 ALIKED + LightGlue 的 ONNX），或 RoMa v1（MIT + DINOv2 Apache-2.0）。用我们自己的 K 求位姿，稠密深度用 MoGe-3（MIT）或 DA3METRIC-LARGE / DA3-BASE（Apache-2.0）对齐，尺度仍用急停。**许可最干净，精度没测**；每个工位只有 2 张照片，SfM 只能做两视图，可能不够。COLMAP 要按下文的编译开关构建（不编 SiftGPU、LSD、CGAL）。
  4. **向 Pi3 作者买商用许可**：HF Pi3 卡写明商用请联系作者。这样能保住已发布报告的精度，可能是最快的路。
  5. map-anything-apache：许可干净，但在第一版 A/B（已被判不公平）里地面 p95 16–18 cm；没在公平 harness 重跑。
  6. DA3-BASE：干净，但 MAE 4.5 cm，不够。
- **不能用（非商用或来源不清）**：Pi3 / Pi3X、RecGen、DA3-LARGE(-1.1，待确认)、DA3-GIANT / NESTED、map-anything（NC 版）、VGGT-1B（原版）、MASt3R / DUSt3R / MUSt3R / CUT3R、SuperPoint、UFM、Fast3R、VGG-T³、DVLT、STream3R、BLASt3R、Argus；ABot-Recon（卡写 Apache，但论文说从 π³ 权重初始化）、AnySplat（从 VGGT-1B 初始化）。HY-WorldMirror 1/2 只能在欧盟、英国、韩国以外用。
- **Meta 系（SAM 3、SAM 3D、DINOv3、VGGT）共同条件**：禁 ITAR、军事 / 战争、**核工业或核应用**、间谍、枪支 / 非法武器；我们不能是制裁对象；禁逆向；我们要为自己的使用和分发**赔偿 Meta**；Meta 可单方面改协议、立即生效。**要问客户站点是否涉及核、军工或 ITAR。**

**English**
- **Everything except the geometry backbone can run on-prem commercially. The weights can be mirrored to our own store and shipped with the delivery, so the customer never needs Hugging Face.**
  - Masks: SAM 3 / 3.1 (SAM License, commercial with conditions); fallback SAM 2.1 (Apache-2.0).
  - Boxes: OWLv2 (Apache-2.0) or Grounding DINO (Apache-2.0; mirror `bert-base-uncased` with it, because the code downloads it from HF at run time).
  - Completion: SAM 3D Objects (SAM License; replaces RecGen, tied in the fair A/B).
  - Monocular check: MoGe-3 (MIT).
- **The geometry backbone (Pi3X's slot) still has no replacement that is both licence-clean and proven accurate.** Candidates, in the order I suggest:
  1. **VGGT-1B-Commercial.** The licence is clear: commercial use allowed (the README excludes only military use), redistribution allowed under the same agreement with a copy attached. It is **not yet tested in the fair A/B**, and HF approval is manual. Its AUP bans use related to "operation of critical infrastructure, transportation technologies, or heavy machinery" under the risk-of-bodily-harm item. We audit robot-cell safety rather than operate machinery, but legal should read that line.
  2. **DA3-LARGE-1.1.** Accuracy is proven (MAE 1.2 cm vs Pi3X 1.4 cm), but the conflict stands: the HF card says apache-2.0, the official README table says CC BY-NC 4.0. GitHub issue #259 (2026-06-01) asks exactly this and has 0 replies. The user needs written confirmation from ByteDance Seed.
  3. **Classic SfM, permissive licences only.** COLMAP (BSD-3) with ALIKED or DISK plus LightGlue (BSD-3 / Apache-2.0; COLMAP ≥ 3.13 ships ALIKED + LightGlue as ONNX), or RoMa v1 (MIT + DINOv2 Apache-2.0). Poses come from our K, dense depth from MoGe-3 (MIT) or DA3METRIC-LARGE / DA3-BASE (Apache-2.0) aligned to the SfM points, and scale from the e-stop as now. **Cleanest licences, accuracy unmeasured.** Each cell has only 2 photos, so SfM is two-view and may be too weak. Build COLMAP with the flags below (no SiftGPU, LSD or CGAL).
  4. **Buy a commercial licence from the Pi3 authors.** The HF Pi3 card says to contact them for commercial use. This keeps the published accuracy and may be the fastest path.
  5. map-anything-apache: clean, but floor p95 was 16–18 cm in the first A/B (judged unfair). Not re-run in the fair harness.
  6. DA3-BASE: clean, but MAE 4.5 cm is not good enough.
- **Not usable (non-commercial or unclear provenance):** Pi3 / Pi3X, RecGen, DA3-LARGE(-1.1, pending), DA3-GIANT / NESTED, map-anything (NC variant), VGGT-1B (original), MASt3R / DUSt3R / MUSt3R / CUT3R, SuperPoint, UFM, Fast3R, VGG-T³, DVLT, STream3R, BLASt3R, Argus. ABot-Recon's card says Apache, but its paper initialises from π³ weights. AnySplat initialises from VGGT-1B. HY-WorldMirror 1/2 is licensed only outside the EU, the UK and South Korea.
- **Conditions shared by the Meta licences (SAM 3, SAM 3D, DINOv3, VGGT).** No ITAR, military or warfare, **nuclear industries or applications**, espionage, or guns and illegal weapons. We must not be a sanctions target. No reverse engineering. We **indemnify Meta** for claims arising from our use or distribution. Meta can change the terms unilaterally, effective immediately. **Ask each customer whether the site involves nuclear, military or ITAR work.**

## 组件表 / Component table

"镜像交付" = 能否把权重放进我们自己的存储并随离线包交给客户。"门控" = HF 上的访问审批。
"Mirror & ship" = may we put the weights in our own store and hand them to the customer in the offline bundle. "Gate" = HF access approval.

| 组件 Component | 用途 Role | 代码 Code | 权重 Weights（pin） | 商用 Commercial | 镜像交付 Mirror & ship | 门控 Gate | 用途限制 Field-of-use | 来源 Sources |
|---|---|---|---|---|---|---|---|---|
| **VGGT-1B-Commercial** | 几何候选 geometry candidate | VGGT License v1（2025-07-29；repo `LICENSE.txt` 同文） | VGGT License v1（`facebook/VGGT-1B-Commercial@ebb29a5`） | **是 Yes** | **是**：§1.b.i 同协议分发 + 附协议副本 / yes, same terms + copy | manual（类 Llama 表单 / Llama-style form） | AUP：军事 / 战争 / 核 / 间谍 / ITAR、武器、毒品；**关键基础设施 / 交通 / 重型机械的运行**（"人身伤害风险"项下）；须向终端用户披露已知风险（AUP 4）；贸易管制 trade controls | [HF](https://huggingface.co/facebook/VGGT-1B-Commercial), [LICENSE](https://huggingface.co/facebook/VGGT-1B-Commercial/blob/main/LICENSE), [README](https://github.com/facebookresearch/vggt#license) |
| VGGT-1B（原版 original） | — | 同上 same | **CC BY-NC 4.0**（`860abec`） | **否 No** | 否 No | 无 none | — | [HF](https://huggingface.co/facebook/VGGT-1B) |
| **MoGe-3** | 单目核查 mono check | MIT（`moge/model/modules/dinov2` 为 Apache-2.0）；FlexGEMM MIT | MIT（`Ruicheng/moge-3-vitl@184008f`；编码器 DINOv2） | 是 Yes | 是（附 MIT 声明）yes, keep notice | 无 none | 无 none | [HF](https://huggingface.co/Ruicheng/moge-3-vitl), [README](https://github.com/microsoft/MoGe/blob/main/README.md) |
| **RoMa v1** | 稠密匹配 dense matcher | MIT（DINOv2 部分 Apache-2.0） | `roma_outdoor.pth` / `roma_indoor.pth`（[Parskatt/storage](https://github.com/Parskatt/storage) 发布，仓库 MIT）+ DINOv2 ViT-L/14（`dl.fbaipublicfiles.com`，Apache-2.0） | 是 Yes | 是 yes | 无 none | 无。来源提示：indoor 模型用 ScanNet 训练（数据集条款非商用），优先用 outdoor（MegaDepth）/ provenance: prefer outdoor | [README](https://github.com/Parskatt/RoMa#license), [model zoo](https://github.com/Parskatt/RoMa/blob/main/romatch/models/model_zoo/__init__.py) |
| **RoMa v2** | 稠密匹配 dense matcher | MIT，DINOv3 代码除外（DINOv3 License） | `romav2.0.1.pt`（GitHub release，1.10 GB）。源码先建空的 DINOv3 再 `load_state_dict(strict)`，所以**检查点里带着 DINOv3 编码器** → 该部分受 DINOv3 License | 是，有条件 Yes* | 是，附 DINOv3 License、同条款（§1.b.i）yes, with DINOv3 licence copy | GitHub 上无门控（DINOv3 在 HF 上是 manual） | DINOv3 §1.b.v：同 SAM（ITAR / 军事 / 核 / 间谍 / 武器）；禁逆向 | [README](https://github.com/Parskatt/RoMaV2#license), [romav2.py](https://github.com/Parskatt/RoMaV2/blob/main/src/romav2/romav2.py), [features.py](https://github.com/Parskatt/RoMaV2/blob/main/src/romav2/features.py), [DINOv3 LICENSE](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) |
| **LightGlue + DISK** | 稀疏匹配 sparse matching | Apache-2.0（LightGlue；kornia Apache-2.0） | `disk_lightglue.pth`（LightGlue v0.1_arxiv，Apache-2.0）；DISK `depth-save.pth`（cvlab-epfl/disk 仓库内，Apache-2.0） | 是 Yes | 是 yes | 无 none | 无 none | [LightGlue README](https://github.com/cvg/LightGlue#license), [DISK](https://github.com/cvlab-epfl/disk), [kornia DISK](https://github.com/kornia/kornia/blob/main/kornia/feature/disk/disk.py) |
| **LightGlue + ALIKED** | 稀疏匹配 sparse matching | Apache-2.0 + BSD-3-Clause（ALIKED） | `aliked_lightglue.pth`（Apache-2.0）；ALIKED `models/*.pth`（BSD-3）。COLMAP 3.13 release 另有 ALIKED + LightGlue 的 ONNX | 是 Yes | 是 yes | 无 none | 无 none | [LightGlue README](https://github.com/cvg/LightGlue#license), [ALIKED](https://github.com/Shiaoming/ALIKED), [COLMAP resources.h](https://github.com/colmap/colmap/blob/main/src/colmap/feature/resources.h) |
| SuperPoint（含 `superpoint_lightglue`） | — | **Magic Leap：学术 / 非营利机构、非商用研究**；禁分发；衍生品归 Magic Leap | 同上（LightGlue 的 SuperPoint 匹配器权重本身标 Apache，但离开 SuperPoint 没法用） | **否 No** | **否 No** | 无 none | — | [LICENSE](https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/LICENSE), [LightGlue README](https://github.com/cvg/LightGlue#license) |
| **COLMAP / pycolmap** | SfM | "new BSD"（BSD-3）。**内含第三方**：SiftGPU（UNC：仅教育 / 研究 / 非营利）、LSD（**AGPL-3.0**）、PoissonRecon MIT、VLFeat BSD-2、Symforce-Caspar Apache-2.0；可选 CGAL（GPL/LGPL）。SIFT 专利 2020 年已过期 | 不需要（可选 ONNX 特征见上） | 是，**按开关构建、不带 SuiteSparse** Yes if built right, no SuiteSparse | 是；PyPI 轮子要履行 GPL / yes; PyPI wheel needs GPL compliance | 无 none | 构建：`-DCUDA_ENABLED=OFF -DGUI_ENABLED=OFF -DLSD_ENABLED=OFF -DCGAL_ENABLED=OFF`（GUI 关会关 OpenGL；CUDA 或 OpenGL 开就会编进 SiftGPU）。PyPI `pycolmap` 4.2.1 的 Linux 构建脚本已关 GUI / CGAL / LSD；`pycolmap-cuda12` 会编进 SiftGPU，别用。**另（2026-10-06 拆轮子核实，原为推断）**：官方轮子**确实**含 GPL-2.0-or-later 代码。`pycolmap-4.2.1-cp311-cp311-manylinux_2_28_x86_64.whl`（sha256 `7627f0ba…dd90f`）的 `_core.so` 静态链接 SuiteSparseQR 和 CHOLMOD Supernodal / MatrixOps（来自 vcpkg `ceres[suitesparse]`）；还静态打包 OpenSSL、libcurl 8.21.0、OpenImageIO、METIS、TIFF 4.7.2、zlib，没带声明。可商用，但分发要履行 GPL。**几何路线已不用它**：BA 换成 `ba_scipy`（numpy / scipy，无 SuiteSparse / CHOLMOD / SPQR / Ceres），4 个配置通过，现场值与 pycolmap 差 ≤ 0.0004 cm（`geometry-backbone-ab-2026-10-06/README.md`）。Verified 2026-10-06: the wheel does carry GPL-2.0+ code (SuiteSparseQR, CHOLMOD Supernodal / MatrixOps, statically linked). The geometry route no longer uses it: its BA is `ba_scipy` (numpy / scipy), within 0.0004 cm of pycolmap | [license](https://colmap.github.io/license.html), [LICENSE](https://github.com/colmap/colmap/blob/main/LICENSE), [thirdparty](https://github.com/colmap/colmap/tree/main/src/thirdparty), [wheel build](https://github.com/colmap/colmap/blob/main/python/ci/install-colmap-almalinux.sh), [vcpkg.json](https://github.com/colmap/colmap/blob/main/vcpkg.json), [cholmod port](https://github.com/microsoft/vcpkg/blob/master/ports/suitesparse-cholmod/vcpkg.json), [spqr port](https://github.com/microsoft/vcpkg/blob/master/ports/suitesparse-spqr/vcpkg.json) |
| GLOMAP | 全局 SfM global SfM | BSD-3；**已停止维护，并入 COLMAP 的 "global" mapper** | — | 是 Yes | 是 yes | 无 none | 同 COLMAP | [README](https://github.com/colmap/glomap) |
| **SAM 2.1** | 掩码备用 mask fallback | Apache-2.0（demo 字体 OFL） | Apache-2.0（README 明写检查点 Apache；`facebook/sam2.1-hiera-large@665f8e2`） | 是 Yes | 是 yes | 无 none | 无 none | [README](https://github.com/facebookresearch/sam2#license), [HF](https://huggingface.co/facebook/sam2.1-hiera-large) |
| **OWLv2** | 开放词表框 open-vocab boxes | Apache-2.0（scenic；transformers） | Apache-2.0（`google/owlv2-base-patch16-ensemble@cfd3195`，large `95e2693`） | 是 Yes | 是 yes | 无 none | 无 none | [HF](https://huggingface.co/google/owlv2-base-patch16-ensemble), [scenic](https://github.com/google-research/scenic) |
| **Grounding DINO** | 开放词表框 open-vocab boxes | Apache-2.0 | Apache-2.0（HF `IDEA-Research/grounding-dino-tiny@a2bb814` / `-base@12bdfa3`；原版 `groundingdino_swint_ogc.pth` 在同一 Apache 仓库的 release）；文本编码器 `bert-base-uncased` Apache-2.0，**运行时会去 HF 下载，要一起镜像** | 是 Yes | 是 yes | 无 none | 无 none | [repo](https://github.com/IDEA-Research/GroundingDINO), [HF](https://huggingface.co/IDEA-Research/grounding-dino-base), [tokenizer code](https://github.com/IDEA-Research/GroundingDINO/blob/main/groundingdino/util/get_tokenlizer.py), [BERT](https://huggingface.co/google-bert/bert-base-uncased) |
| **SAM 3 / 3.1** | 掩码 masks | SAM License（2025-11-19）；transformers 移植 Apache-2.0 | SAM License（`facebook/sam3@3c879f3`，`facebook/sam3.1@daa6319`） | **是，有条件 Yes*** — ScanCode 归类 "Proprietary Free" | **是**：§1.a 授权含复制和分发；§1.b.i 只能按本协议分发，且**随附协议副本** | manual（要法定全名、生日、公司全称） | 见下节 / see next section | [LICENSE](https://github.com/facebookresearch/sam3/blob/main/LICENSE), [HF](https://huggingface.co/facebook/sam3), [ScanCode](https://scancode-licensedb.aboutcode.org/sam-2025-11-19.html) |
| **SAM 3D Objects** | 补全 completion | SAM License（同 2025-11-19 文本，**多一条 §1.a.i 专利许可**） | SAM License（`facebook/sam-3d-objects@2e73555`；README：检查点与代码都是 SAM License） | 是，有条件 Yes* | 是，同上 yes, as above | manual | 同 SAM 3；镜像已去掉 bpy（GPL）、pymeshfix（AGPL）、smplx（NC） | [LICENSE](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE), [README](https://github.com/facebookresearch/sam-3d-objects#license), [HF](https://huggingface.co/facebook/sam-3d-objects) |
| **DA3-BASE** | 几何候选 geometry candidate | Apache-2.0（`3d835ec`） | Apache-2.0（`depth-anything/DA3-BASE@f4a6c9b`；卡和 README 一致） | 是 Yes | 是 yes | 无 none | 无。来源提示：issue #203 问 Apache 小模型的训练数据，维护者未答 | [HF](https://huggingface.co/depth-anything/DA3-BASE), [README table](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md), [#203](https://github.com/ByteDance-Seed/Depth-Anything-3/issues/203) |
| DA3-SMALL / DA3METRIC-LARGE / DA3MONO-LARGE | 深度 depth | Apache-2.0 | Apache-2.0（卡和 README 一致） | 是 Yes | 是 yes | 无 none | 无 none | 同上 same |
| DA3-LARGE-1.1 | 几何候选 geometry candidate | Apache-2.0 | **冲突**：HF 卡 apache-2.0（`0e109ae`），README 表 **CC BY-NC 4.0**；前代 DA3-LARGE 卡也是 NC | **按否处理 treat as No** | 否（待书面确认）no, until confirmed | 无 none | — | [HF](https://huggingface.co/depth-anything/DA3-LARGE-1.1), [README](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/README.md), [#259（0 回复）](https://github.com/ByteDance-Seed/Depth-Anything-3/issues/259) |
| DA3-GIANT(-1.1) / DA3NESTED-GIANT-LARGE(-1.1) | — | Apache-2.0 | CC BY-NC 4.0（卡和 README 一致） | 否 No | 否 No | 无 none | — | 同上 same |
| **map-anything-apache** | 几何候选 geometry candidate | Apache-2.0（UniCeption BSD-3） | Apache-2.0（`facebook/map-anything-apache@00f9c24`，2026-01-20 版；README：商用版，训练数据不同） | 是 Yes | 是 yes | 无 none | 无。依赖卫生：`pyproject` 拉 `plyfile`（**GPL-3.0+**），镜像里去掉 | [README](https://github.com/facebookresearch/map-anything#models), [HF](https://huggingface.co/facebook/map-anything-apache), [plyfile PyPI](https://pypi.org/project/plyfile/) |
| map-anything（默认 default） | — | Apache-2.0 | CC BY-NC 4.0 | 否 No | 否 No | 无 none | — | [HF](https://huggingface.co/facebook/map-anything) |
| MASt3R / DUSt3R（及 CroCo、MUSt3R、CUT3R） | — | **CC BY-NC-SA 4.0** | CC BY-NC-SA 4.0 + 各训练集许可（README：mapfree 尤其严格） | **否 No** | 否 No | 无 none | — | [MASt3R](https://github.com/naver/mast3r/blob/main/LICENSE), [checkpoints](https://github.com/naver/mast3r#checkpoints), [DUSt3R](https://github.com/naver/dust3r/blob/main/LICENSE), [CUT3R](https://github.com/CUT3R/CUT3R/blob/main/LICENSE) |
| Pi3 / Pi3X（现行） | 几何（现行）current geometry | BSD-3 | **CC BY-NC 4.0**（README："Strictly Non-Commercial"，训练数据所致）；HF Pi3 卡标 bsd-2-clause，但正文说商用请联系作者 | **否**，可谈商用许可 No, licence negotiable | 否 No | 无 none | — | [README](https://github.com/yyfz/Pi3/blob/main/README.md), [HF Pi3](https://huggingface.co/yyfz233/Pi3), [HF Pi3X](https://huggingface.co/yyfz233/Pi3X) |

\* 有条件 = 见 Meta 协议一节。 / conditional = see the Meta licence section.

### 2025–2026 新出的多视角前馈几何模型 / New feed-forward multi-view geometry models (2025–2026)

在 HF `image-to-3d` 下载量前 100 和网页检索里找的。真正可商用的只有上表的 VGGT-1B-Commercial、map-anything-apache、DA3 的 Apache 档；下面这些都不行或有地域限制。
From the top 100 HF `image-to-3d` models by downloads plus web search. The only commercially usable ones are VGGT-1B-Commercial, map-anything-apache and the Apache DA3 sizes above. Everything below is blocked or territory-limited.

| 模型 Model | 日期 date | 许可 Licence | 结论 Verdict | 来源 Source |
|---|---|---|---|---|
| HunyuanWorld-Mirror / WorldMirror 2.0（HY-World-2.0） | 2025-10 / 2026-04 | 腾讯社区许可：**不适用于欧盟、英国、韩国**；月活 > 1M 须另签；输出不得用来改进其他模型 | 地域内可商用，**不能当默认** territory-limited | [License](https://github.com/Tencent-Hunyuan/HY-World-2.0/blob/main/License.txt), [HF](https://huggingface.co/tencent/HY-World-2.0) |
| ABot-Recon | 2026-08 | 卡 apache-2.0；但论文 Stage I **从 π³ 公开权重初始化**（CC BY-NC），curope 是 CC BY-NC-SA | **按否处理** treat as No | [HF](https://huggingface.co/acvlab/ABot-Recon), [arXiv 2608.27529](https://arxiv.org/abs/2608.27529), [notices](https://github.com/amap-cvlab/ABot-Recon/blob/main/THIRD_PARTY_NOTICES.md) |
| VGG-T³（nvidia/vgg-ttt） | 2025-12 | NVIDIA OneWay Noncommercial | 否 No | [HF](https://huggingface.co/nvidia/vgg-ttt) |
| DVLT（nvidia/dvlt） | 2026-05 | 权重 NVIDIA License（标签 nvidia-oneway-noncommercial）；代码 Apache-2.0 + VGGT 衍生部分 | 否 No | [HF](https://huggingface.co/nvidia/dvlt) |
| Fast3R | 2025-02 | FAIR NC Research License | 否 No | [HF](https://huggingface.co/jedyang97/Fast3R_ViT_Large_512) |
| STream3R | 2025-08 | NTU S-Lab License 1.0 | 否 No | [HF](https://huggingface.co/yslan/STream3R) |
| BLASt3R matcher | 2026-09 | NAVER 非商用 + DINOv3 | 否 No | [HF](https://huggingface.co/naver/blast3r-matcher) |
| Argus（Realsee） | 2026 | CC BY-NC 4.0，gated | 否 No | [HF](https://huggingface.co/RealseeTechnology/argus-realsee3d) |
| UFM（稠密匹配 dense matcher） | 2025–2026 | 代码 BSD-3；权重 CC BY-NC(-SA)；Apache 版自 2025-06 被请求，未发布 | 否 No | [README](https://github.com/UniFlowMatch/UFM#license), [#1](https://github.com/UniFlowMatch/UFM/issues/1) |
| AnySplat | 2025-06 | 卡 MIT，但从 VGGT-1B（NC）初始化 | 不清楚，按否 unclear → No | `anysplat-ab-2026-10-05` |
| StreamVGGT / MonST3R | 2025 | CC BY-NC 4.0 / CC BY-NC-SA 4.0 | 否 No | [HF](https://huggingface.co/lch01/StreamVGGT) |

## Meta 协议的确切条款 / Meta licence clauses (SAM License 2025-11-19; DINOv3 License 2025-08-19; VGGT License v1 2025-07-29)

三份协议主体结构相同；条款号以 SAM License 为准。下面是转述，原文见上面的链接。
The three share one structure; clause numbers follow the SAM License. Paraphrased below; read the linked texts for exact wording.

| 条款 Clause | SAM 3 / SAM 3D / DINOv3 | VGGT-1B-Commercial |
|---|---|---|
| §1.a 授权 grant | 非独占、全球、**不可转让**、免版税；可使用、复制、分发、衍生、修改。没有营收、用户数或"非商用"限制。SAM 3D 多一条专利许可（§1.a.i） / Non-exclusive, worldwide, non-transferable, royalty-free: use, reproduce, distribute, derive, modify. No revenue, user-count or non-commercial cap. SAM 3D adds a patent licence (§1.a.i). | 同 same |
| §1.b.i 再分发 redistribution | 分发 SAM Materials 或其衍生品给第三方时，**只能按本协议**，并**随附协议副本** / Only under this Agreement, with a copy attached. | 同，且须向第三方提供协议副本 / same |
| §1.b.ii 发表 publication | 发表研究结果要致谢 / Acknowledge in publications. | 同 same |
| §1.b.iii 合规 compliance | 遵守法律、贸易管制、**隐私与数据保护法** / Laws, trade controls, privacy and data-protection law. | 法律、贸易管制 + **AUP（并入协议）** |
| §1.b.iv 逆向 reverse engineering | 禁止逆向、反编译 / Prohibited. | 无此条 not present |
| §1.b.v 用途 field of use | 我们不能是贸易管制对象；**禁 ITAR、军事 / 战争、核工业或核应用、间谍、枪支或非法武器** / Not a sanctions target; no ITAR, military or warfare, nuclear, espionage, guns or illegal weapons. | AUP 第 2 项：军事 / 战争 / 核 / 间谍 / ITAR、武器、毒品、**关键基础设施 / 交通 / 重型机械的运行**、自伤；第 1 项：违法、歧视、无资质执业、未经同意处理敏感个人信息等；第 3 项：欺骗；第 4 项：**须向终端用户披露已知风险** |
| §5.b 诉讼与赔偿 litigation, indemnity | 对 Meta 提起侵权诉讼则许可终止；**我们赔偿 Meta 因我们使用或分发引起的第三方索赔** / Licence ends if we sue Meta for infringement; **we indemnify Meta** for third-party claims from our use or distribution. | 同 same |
| §6 终止 termination | 违约则 Meta 可终止，终止后须删除 / Meta may terminate on breach; then delete. | 同 same |
| §7 法律 law | 加州法律，加州法院专属管辖 / California law and courts. | 同 same |
| §8 修改 changes | Meta 可修改（"精神相似"），**立即生效**，继续使用即视为接受 / Meta may modify, effective immediately; continued use means acceptance. | 同 same |

## 镜像与交付 / Mirroring and shipping the weights

**中文**
- **HF 门控不是许可条件。** HF 文档：门控只是让作者审批谁能从 Hub 下载文件，审批授予个人账户；能做什么由许可证决定。所以用我们已获批的账户下载一次，就可以按许可证放进我们自己的制品库、随离线包交付，客户不需要 HF 账户也不联网。`fetch_weights.py --source` / `--verify` 和 `airgap.sh` 已支持这种流程。
- **可以镜像交付**（附什么）：
  - MIT / BSD：保留版权与许可声明。
  - Apache-2.0：附 LICENSE，上游有 NOTICE 的附 NOTICE，改动过的文件要注明。
  - SAM License / DINOv3 License / VGGT License：附协议副本，按同一协议交付。在客户合同里转述用途限制（ITAR / 军事 / 核 / 间谍 / 武器；VGGT 另加 AUP），因为条款写的是不得允许他人违反（VGGT AUP 原句也是"or allow others to use"）。
- 每个镜像存：repo id、commit sha、每个文件的 sha256、当天的许可证文本及其 sha256（Meta 可随时改协议，存档用于证明我们接受的是哪一版）。
- **不能镜像交付**：所有 CC BY-NC / NC-SA / NVIDIA NC / FAIR NC / S-Lab / NAVER NC 权重（Pi3X、RecGen、DA3-LARGE-1.1 待定、MASt3R 等）；SuperPoint（许可本身就禁止分发）；腾讯 HY 系只能在其许可地域内交付。
- GPL 组件（`plyfile`、vcpkg 构建的 Ceres 里的 SuiteSparse GPL 模块）：可以商用、可以分发，但要履行 GPL（提供源码）；更省事的是从镜像里去掉。

**English**
- **The HF gate is not a licence term.** Per HF's docs, gating only lets authors approve who may download files from the Hub, and access is granted to individual accounts; what you may do with the files is set by the licence. So one download with our approved account is enough; the weights then go into our own artifact store and ship in the offline bundle, and the customer needs neither an HF account nor internet. `fetch_weights.py --source` / `--verify` and `airgap.sh` already support this.
- **May be mirrored and shipped** (with what):
  - MIT / BSD: keep the copyright and licence notice.
  - Apache-2.0: include LICENSE, plus NOTICE where upstream has one; mark modified files.
  - SAM License / DINOv3 License / VGGT License: attach the agreement and ship under the same terms. Pass the use restrictions (ITAR, military, nuclear, espionage, weapons; plus the AUP for VGGT) into the customer contract, because the terms also forbid permitting others to break them.
- For every mirrored model keep the repo id, commit sha, per-file sha256, and that day's licence text with its sha256. Meta can change the terms at any time, so the archive proves which version we accepted.
- **May not be mirrored and shipped:** every CC BY-NC, NC-SA, NVIDIA NC, FAIR NC, S-Lab or NAVER NC weight (Pi3X, RecGen, DA3-LARGE-1.1 pending, MASt3R, …). SuperPoint's licence forbids any distribution. Tencent HY models ship only inside their licensed territory.
- GPL pieces (`plyfile`, SuiteSparse's GPL modules inside a vcpkg-built Ceres) allow commercial use and distribution, but only with GPL compliance (source offer). Dropping them from the images is simpler.

今天读到的许可证文本指纹 / Fingerprints of the licence texts read today (sha256 of the raw file):

| 文本 text | URL | sha256 |
|---|---|---|
| SAM License 2025-11-19（SAM 3） | https://raw.githubusercontent.com/facebookresearch/sam3/main/LICENSE | `4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be` |
| SAM License 2025-11-19（SAM 3D，含专利条款） | https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/LICENSE | `b3a5a0e2d973ab80e6610ccf1cffc40756050d0ace3cd4fec879b3ec290b2e9b` |
| DINOv3 License 2025-08-19 | https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md | `25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e` |
| VGGT License v1（HF 权重仓） | https://huggingface.co/facebook/VGGT-1B-Commercial/resolve/main/LICENSE | `ffbc0091f2f44adfe1a575c4eb6329be446df169e71e1bd9b95ebc420ff06fef` |
| VGGT License v1（GitHub 代码仓，空白不同、正文相同） | https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt | `b94e46746792d1adfa0ebdbccada5d4576d244ae230be9de8a46b5868fff001b` |

## 建议的下一步 / Suggested next steps

1. **用户要做**（需要用户本人发出 / needs the user to send）：(a) 给 ByteDance Seed 发邮件，或在 #259 下跟帖，书面确认 DA3-LARGE-1.1 是否 Apache-2.0；(b) 问 Pi3 作者商用许可报价；(c) 问客户站点有没有核、军工或 ITAR 用途。
   For the user to send: (a) ask ByteDance Seed in writing (email or on #259) whether DA3-LARGE-1.1 is Apache-2.0; (b) ask the Pi3 authors for a commercial licence quote; (c) ask the customer whether any site has nuclear, military or ITAR use.
2. **实验**：把 VGGT-1B-Commercial 放进公平 A/B（同冻结帧、同中性地面、自己的急停尺度、4 个现场值）。先要用有 HF 访问权的账户在 HF 上申请（人工审批）。本机没有 HF token，Modal 上那个 HF secret 有没有 VGGT-Commercial 权限没查。
   Experiment: run VGGT-1B-Commercial through the fair A/B (same frozen frames, neutral floor, own e-stop scale, 4 field values). First request access on HF (manual approval). This Mac has no HF token; whether the Modal HF secret's account has VGGT-Commercial access was not checked.
3. 备选实验：COLMAP（按上面开关）+ ALIKED/LightGlue 或 RoMa v1 + MoGe-3 / DA3METRIC 稠密，两视图，同一评判。map-anything-apache 也可以在公平 harness 里补跑一次。
   Fallback experiment: COLMAP (flags above) with ALIKED/LightGlue or RoMa v1, dense depth from MoGe-3 / DA3METRIC, two views, same judging. map-anything-apache can also be re-run in the fair harness.
4. 法务：VGGT AUP 的"重型机械 / 关键基础设施运行"一条是否覆盖机器人工位安全审计；Meta 协议 §5.b 的赔偿条款。
   Legal: whether VGGT's AUP heavy-machinery / critical-infrastructure line covers robot-cell safety audits, and the §5.b indemnity in the Meta licences.

## 注意 / Caveats

- 不是法律意见；许可按 2026-10-06 读到的版本，以后可能变。Not legal advice; licences as read on 2026-10-06.
- MIT / Apache 权重背后的训练数据来源只能以发布方声明为准，没有审计（DA3 #203 正是在问这个）。Training-data provenance behind permissive weights is the publisher's claim and was not audited.
- pycolmap 轮子含 GPL 代码：原为推断，2026-10-06 已拆 PyPI 4.2.1 cp311 manylinux 轮子核实（见组件表）；几何路线改用 `ba_scipy`。The GPL content of the pycolmap wheel, first inferred, was verified on 2026-10-06 by inspecting the PyPI 4.2.1 cp311 manylinux wheel (see the component table); the geometry route now uses `ba_scipy`.
- VGGT-1B-Commercial、经典 SfM 路线的精度都没测，本文只回答许可。Accuracy of VGGT-1B-Commercial and the classic-SfM route is unmeasured; this note answers licensing only.
- 没审计：CUDA 基础镜像、未点名的 pip / C++ 传递依赖。Not audited: CUDA base images and unnamed pip / C++ transitive dependencies.
