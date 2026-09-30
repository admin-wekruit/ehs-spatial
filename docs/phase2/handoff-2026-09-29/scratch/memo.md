# Panoptes Twin v2 决策备忘（2026-09-28）

[D]=有文档出处；[I]=推断，未验证。

## 1. 资产来源（演示与商用共用一个 CC0/CC-BY 库）

| 场景 | 首选来源 | 许可 |
|---|---|---|
| 仓库（Sam's Club） | NVIDIA SimReady-Warehouse-01（753 个 USD，带碰撞体和质量）+ projectsim 货架 | CC-BY-4.0 [D] |
| 车间（ME340） | Poly Haven（521 个模型，含钻床、工具车、灯具）+ ambientCG 材质 | CC0 [D] |
| 机床、长尾物体 | Objaverse 1.0（只取 BY/CC0）+ Sketchfab CC-BY；都没有就用 SAM 3D 生成并入库 | 逐个模型定 [D] |
| 零售货架（商用） | 加购 DreamVu 超市资产（2,200+ 件） | 商业许可 [D] |

- 非商用，不能入库：ShapeNet、PartNet-Mobility、3D-FUTURE、HSSD、GRScenes、Lightwheel 开源版、GrabCAD [D]。
- Fab、TurboSquid、BlenderKit 和 Isaac Sim 自带资产都禁止分发原文件 [D]。在 three.js 里直接加载 GLB 就算分发 [I]，所以这些资产只能出渲染图或视频。
- CC-BY 资产要逐个署名，写进场景 JSON 和页面 credits [I]。
- SAM 3D 输出可以商用，但受出口管制和军事用途限制 [D]。

## 2. Twin v2：布局与检索

| 步骤 | 方法 | 耗时 |
|---|---|---|
| 布局 | 重力对齐后做 Manhattan 约束的 RANSAC 平面拟合（Open3D）；门窗先在 2D 检测，再投影到墙面 | 每场景几秒，CPU [I] |
| 建库 | 每个资产渲染 12–24 个视角，算 DINOv2/SigLIP2 嵌入，由 VLM 标注类别、正面和尺寸，用 FAISS 检索 | 一次性 [I] |
| 检索 | 按类别过滤 → DINOv2 取 top-20 → 尺寸打分（任一轴要拉伸 >20% 就淘汰）→ 在真实相机视角渲染比对 → VLM 裁决 | 每物体 <1 s，另加 VLM 几秒 [I] |
| 兜底 | 用 SAM 3D 生成 | 每物体 3.5 s；60 个物体约 105–210 s [I] |
| 复用与验收 | 同类物体共用一个资产；每个物体检查轮廓 IoU，再由 VLM 裁判 | [I] |

- 最接近的蓝本是 LiteReality（Apache-2.0），它在 Scan2CAD 上的相似度优于 Digital Cousin [D]。
- SpatialLM 和 SceneScript 的权重是 CC-BY-NC，而且只用住宅数据训练 [D]，只作研究对照。

## 3. GPU 与迁移

| 阶段 | 做法 |
|---|---|
| 1. Modal（现在） | 每个视频一个 A100:2 容器。容器内单节点 Ray，各阶段是占部分 GPU 的 actor。VLM 用 vLLM，走 OpenAI 兼容接口。不跨容器组集群，因为 clustered 还是 beta [D] |
| 起步分配 | GPU0 跑 7–8B VLM、分割和渲染比对；GPU1 跑两个 SAM 3D actor [I]，需要实测 |
| 2. Modal 异构 | 如果用 Isaac Sim，放到单独的 L40S 函数 |
| 3. 本地单机 | 同一个 Docker 镜像，`ray start --head` |
| 4. 本地多机 | K8s + GPU Operator + KubeRay（RayJob 跑批）+ Kueue/KAI；vLLM 单独部署；MIG 只给小模型 |

- **Isaac Sim 不支持 A100/H100**（没有 RT Core）[D]。nvdiffrast 在 A100 上可以跑 [I]。
- Triton、DeepStream、Ray Serve、MPS 暂时都不用 [I]。

## 4. 需要你拍板

1. 演示也只用 CC0/CC-BY 吗？建议是，因为客户演示可能算商用 [I]。
2. 查看器是直接加载 GLB，还是只交付渲染图？
3. 渲染比对用 Isaac Sim（要 L40S/RTX）还是 nvdiffrast（A100 就够）？
4. 是否采购 DreamVu 和 Lightwheel 商用版？预算多少？
5. 法务：接受 nvidia/simready-assets 的访问条款，并请 NVIDIA 书面确认 Omniverse 资产包的条款。
6. MASt3R 权重是 CC BY-NC-SA [D]。如果现有流程在用，是否换成 MapAnything 的 Apache 版？
7. VLM 用 7–8B 单卡，还是 32B 以上单独起服务？
8. 本地 GPU 怎么配：A100/H100 和 RTX PRO 6000/L40S 各多少？

## 未核实
- HomeBody 页面没有写 544 个网格、27 张纹理，代码仍是 "coming soon" [D]。
- Modal 的 `/dev/shm` 大小、NVLink，以及 SAM 3D 单物体峰值显存。

## 来源
https://tml.stanford.edu/homebody/ · https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01 · https://huggingface.co/datasets/projectsim/industrial-parts-and-packaging · https://polyhaven.com/license · https://huggingface.co/datasets/allenai/objaverse · https://natlawreview.com/press-releases/dreamvu-launches-largest-grocery-simready-asset-library-built-real-world · https://raw.githubusercontent.com/facebookresearch/sam-3d-objects/main/LICENSE · https://github.com/LiteReality/LiteReality · https://github.com/manycore-research/SpatialLM · https://github.com/naver/mast3r · https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html · https://modal.com/docs/guide/multi-node-training · https://docs.ray.io/en/latest/cluster/kubernetes/index.html
