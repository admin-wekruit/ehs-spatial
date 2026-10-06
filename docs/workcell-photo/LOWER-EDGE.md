# 下沿的定义 / Lower edge (下沿): the defined measurement

代码 / code: `scripts/workcell_checks/lower_edge.py`（检查模块 / check module），`modal_apps/workcell_part_masks.py`（部位掩码 / part masks），
`scripts/workcell_layer_build.py`（自动生成测量层的「现场对照」/ writes the 现场对照 facts）。
验证 / validation: `research-notes/workcell-lower-edge-2026-10-05`（"点选部位"一节 / section "Click-exemplar parts"）。

## 中文

"物体 X 的部位 P 的下沿"按下面五步算（第 6 条是测量层怎么用它）。部位由一次点选给出（输入），代码里没有按物体类型写的规则。

1. **部位 = 一次点选（点选示例）。** 输入 `{entityId, part（名字）, clickPhoto, clickXY}`：在一张照片里点在这个部位上。
   - 点选像素的射线打到 X 自己的显示模型，第一个交点就是 3D 锚点。
   - 如果模型没打中，或报告的 Pi3X 距离与模型相差 > 5 %，锚点取这条射线上的 Pi3X 点。
   - 锚点同时存成模型自身坐标（`anchorModel`），同一网格的同型物体可以直接复用。
     注意：090 和 030 的立柱是分别生成的网格，坐标系不同；把 090 右立柱的锚点放到 030 的立柱上会落到物体外面。
   - 锚点投到每张有 X 掩码的照片。下面几种照片不出提示点：画面外；别的模型挡在前面；X 自己挡住（锚点在背面）；Pi3X 距离短 5 % 以上（有没建模的东西挡着）；投影不在 X 的掩码上。
2. **每张照片的部位掩码（分割）。** 以投影点为中心切一个 512 px 方块，用 SAM 3 的 tracker（PVS，点提示，与文字模式同一个固定的 `facebook/sam3` 权重）分割。
   - 3 个候选里，取包含提示点、与 X 掩码有交集、并且占方块内 X 掩码 ≤ 80 % 的那个，选预测 IoU 最高的。
   - 占 80 % 以上的是整个物体，不是部位，这张照片就没有部位掩码。
   - 结果裁到 X 的掩码框内，再与 X 的掩码求交。
   - 文字提示的部位（旧的文字模式）仍可用。但同色相邻的两块件，文字分不出来。
   - 实体 id 精确匹配。`part-masks.json` 列出每个目标部位（没有掩码的为 `{}`）；lower_edge 遇到不在其中的部位名直接报错（名字写错），不会当成"没有掩码"。
3. **下边界。** 沿 3D 竖直线的投影（过天底消失点，约 1 px 一条）取区域最低的像素。下面几种线去掉：
   - 区域两侧各 10 %；
   - 被画面截断的线；
   - 下方 1–20 px 内是别的物体掩码的线；
   - 下方 20 px 里一半以上仍是 X、但不属于该部位的线（内部分界）；
   - 下一个像素的射线先打到别的模型的线。
4. **高度：自己的近面，或两视角。** 每个保留像素的下边缘反投影成射线，落到 X 自己的显示模型上（第一个交点；下边缘没打中、像素中心打中，也算直接命中：模型的边就在这个像素里）。
   - 只在上方 3–15 px 才打到模型的线是"擦边线"：报告数量，只用于遮挡判断，**不参与高度**（外推的距离会把高度抬高）。去掉后直接命中的线不足 10 条：不给值（`too_few_direct_lines`）。
   - 模型轮廓的底边在边界下方 ≤ 2 px 以内（一张照片一半以上的线）：标出"下沿＝模型自身的底边"（`edgeIsModelBottom`），不隐藏。
   - 一张照片算"模型可信"，要同时满足：至少一半候选线直接打中；Pi3X 与模型在**同一像素**上的距离相对差（边界上方 10–150 px，每 10 px 一档，取中位数）≤ 5 %。每一档的差都报告（`pi3xGapProfile`）。
   - 只要有一张可用照片不可信，就改用两视角直线三角化，不看模型：两张照片的下边界直线（Huber 拟合）反投影成平面，求交得到 3D 线。
   - 下面几种照片对丢掉：3D 线倾斜 > 45°；两张照片共同看到的线段 < 25 %；任一张照片不到一半的边界像素落在直线 3 px 内。
   - **分割边项。** 沿同一批线，在边界 ±10 px 内找照片最强的亮度台阶（对比度 ≥ 3，亚像素）。一半以上的线有台阶、且偏移彼此相差 ≤ 1 px（IQR/1.349）：这是掩码的系统偏差，校正（模型路径：偏移 × 模型上每 px 的高度；两视角：边界点移到台阶上），只把中位数的标准误计入 σ。偏移不一致：hypot(偏移, 离散) × cm/px 计入 σ。台阶不足：按 ±10 px 窗口均匀分布（10/√3 px）计入 σ。
   - 高度 = n·X + d（报告地面）× nativeToMeters，尺度只读。
5. **±σ 和发布门槛**（取代原来的 15° 夹角规则和 0.5 cm/px 规则）。
   - 模型路径，每张照片：σ = hypot(|Pi3X − 模型距离差| × sin 俯角, IQR / 1.349, 分割边项)。距离差取上面 10–150 px 一带的中位数。
   - 这张照片没有 Pi3X：距离差按"可信"门槛本身取，距离的 5 %（`no_pi3x_5pct_of_range`）。缺证据不会给出小 σ。
   - 两视角，每对照片：σ = hypot(a 线 1 px 引起的 cm × (a 线拟合 RMS 残差, a 的分割边项), b 同理)。
   - 任两张照片（或照片对）的中位数相差 > max(2 cm, 3 × 两者合成 σ)：报"照片间不一致"，不给值。
   - 否则在最低的一档里取（σ ≤ 1 cm 的；没有就 σ ≤ 2.5 cm 的；再没有就全部）：估计 = 中位数，σ = max(σ 的均方根, 中位数极差的一半)。不按张数缩小 σ，误差大多是共有的。
   - **σ ≤ 2.5 cm 就发布估计 ±σ**：σ ≤ 1 cm 标"可信"，否则标"偏大"并写明 σ 主要来自哪一项。
   - 不给值只有三种情况，并写明是哪一种：照片间不一致；只有一张可用视角（或没有可用的线）；σ > 2.5 cm（仍给出估计 ±σ）。
   - 另列、不并入 σ：尺度项（值 × 测量层 `uncertaintyRelative`，090 为 1.7 %，030 为 1.6 %）；地面项（物体附近两视角扫描地面上的高度减报告地面上的高度，来自 clearance B）。
6. **测量层（`workcell_layer_build.py`）。**
   - 模型路径的值与用户现场值相差 ≤ max(2 cm, 2σ)：这个物体的置信度记一次独立验证（"与现场值吻合"）。相差 > max(3 cm, 3σ)：置信度最高为"低"。两视角的值不用模型，不计入模型置信度。
   - "模型下部缺失"只在模型最低点比现场下沿高 > 5 cm 时写（与置信度上限同一规则）。
   - 急停按实体守卫：装急停的实体（有尺度 / 参照物 / 急停复核事实的实体，以及配置里的按钮宿主）除置信度、流程、检查类以外的事实都不能改，不再按"急停"二字判断。
   - 配置里列出的检查结果文件不存在：构建失败，不静默跳过。

**现状（2026-10-06，按复核意见修正后重跑）。**
- 030 右罩壳：**22.5 ±0.85 cm，可信**，比现场 24 cm 低 1.5 cm（在 max(2 cm, 2σ) 内：这个物体的置信度记"与现场值吻合"）。
- 090 右前护板：21.6 ±2.4 cm，偏大（没有现场值）。
- 不给值：090 右罩壳 23.5 ±2.9（σ > 2.5）；030 左罩壳 25.5 ±4.2（σ > 2.5；擦边 12 条不计入；下沿＝模型底边）；030 底横梁 14.4 ±5.2（σ > 2.5）；090 右底横梁（两照片线段重叠 0 %）；090 左罩壳、090 左底横梁（只有一张可用视角）。
- σ 仍主要来自 Pi3X 与模型的距离差：取 10–150 px 中位数后为 090 右 5.2–6.7 cm、030 左 10.3 cm、030 右 0.55 cm。

## English

The lower edge of part P of object X, in five steps (item 6 is how the measurement layer uses it). The part is given by one click (an input); the code knows no object types.

1. **The part is one click (a click exemplar).** Input: `{entityId, part (a label), clickPhoto, clickXY}`, a pixel on the part in one photo.
   - Cast the click ray onto X's own displayed model. The first hit is the 3D anchor.
   - Where the model misses, or the report's Pi3X range there differs by > 5 %, the anchor is the Pi3X point on that ray.
   - The anchor is stored in the model's own frame (`anchorModel`), so it can be reused on an identical object that shares the mesh.
     Caveat: the 090 and 030 posts are separately generated meshes with different frames. The 090R anchor lands off the 030 posts.
   - The anchor is projected into every photo where X has a mask. It is no prompt in a photo when:
     - it falls outside the frame;
     - another model hits in front of it;
     - X itself hits in front of it (a back face);
     - the Pi3X range is more than 5 % shorter (something unmodelled is in front);
     - it misses X's mask.
2. **Part mask per photo (segmentation).** SAM 3's tracker (PVS, point prompt, the same pinned `facebook/sam3` checkpoint as the text mode) segments a 512 px crop centred on the point.
   - Of its 3 candidates, the one with the best predicted IoU wins, among those that contain the point, overlap X's mask, and cover ≤ 80 % of X's mask inside the crop.
   - A mask that covers more is the whole object, not a part, so that photo gets no part mask.
   - The mask is clipped to X's mask box and intersected with X's mask.
   - A text-phrase part (the earlier mode) still works, but text cannot tell two same-coloured neighbouring pieces apart.
   - Entity ids match exactly. `part-masks.json` lists every target part (`{}` when no photo got a mask); lower_edge raises on a part label that is not in it (a typo), instead of reading it as "no mask".
3. **Lower boundary.** Along projected 3D verticals (lines through the nadir vanishing point, about 1 px apart), take the region's lowest pixel. A line is dropped when:
   - it lies in the region's 10 % sides;
   - the frame cuts it;
   - another object's mask is 1–20 px below it;
   - the rest of X fills at least half of the 20 px below it (an internal boundary);
   - the ray through the pixel below meets another model in front.
4. **Height: own near surface, or two-view.** Each kept pixel's bottom edge is cast onto X's own displayed model (first hit; a hit at the pixel's centre also counts as direct: the model's edge lies inside that pixel).
   - A line that meets the model only 3–15 px above is grazed: counted and used for the occlusion test, but **excluded from heights** (the extrapolated range biases them high). Fewer than 10 direct lines left: no value (`too_few_direct_lines`).
   - Where the model's silhouette ends ≤ 2 px below the boundary on at least half of a photo's lines, the result is flagged "edge = the model's own bottom" (`edgeIsModelBottom`), not hidden.
   - A photo is model-backed when at least half of its candidate lines hit the model directly and the Pi3X-vs-model range difference **at the same pixels** (10–150 px above the boundary, every 10 px, median) is ≤ 5 %. The gap per offset is reported (`pi3xGapProfile`).
   - If any usable photo is not model-backed, the method is model-free two-view triangulation: back-project two photos' Huber boundary lines to planes and intersect them in a 3D line.
   - A pair is dropped when the line tilts > 45°, when < 25 % of its stretch is common to both photos, or when < half of a photo's boundary pixels lie within 3 px of its line.
   - **Segmentation-edge term.** Along the same lines, the photo's strongest intensity step within ±10 px of the boundary (contrast ≥ 3, sub-pixel). If at least half the lines have one and their offsets agree within 1 px (IQR / 1.349), it is a mask bias: corrected (model path: offset × the model's cm per px; two-view: the boundary points move onto the step), with only the median's standard error in σ. Offsets that disagree put hypot(offset, spread) × cm/px into σ; too few steps put the ±10 px window as a uniform error (10/√3 px) into σ.
   - Height = n·X + d on the report floor × nativeToMeters. The scale is only read.
5. **±σ and the publish gate.** These replace the 15° and 0.5 cm/px gates.
   - Model path, per photo: σ = hypot(|Pi3X − model range gap| × sin(depression), IQR / 1.349, segmentation term). The gap is the median over the 10–150 px band.
   - A photo without Pi3X there takes the see-through bound itself, 5 % of the range (`no_pi3x_5pct_of_range`): missing evidence never gives a small σ.
   - Two-view, per pair: σ = hypot(cm per px of line a × (line a's RMS fit residual, a's segmentation term), the same for b).
   - If two photos' (or pairs') medians differ by more than max(2 cm, 3 × their combined σ), the result is "photos disagree" with no value.
   - Otherwise, over the lowest tier that has any (σ ≤ 1 cm; else σ ≤ 2.5 cm; else all): estimate = their median, σ = max(RMS of their σ, half the spread of their medians). There is no averaging gain, because the errors are mostly shared.
   - **The estimate ± σ is published whenever σ ≤ 2.5 cm**: labelled "trusted" (可信) at σ ≤ 1 cm, else "large" (偏大) with what dominates σ.
   - No value only when the photos disagree, when there is a single usable view (or no usable lines), or when σ > 2.5 cm (the estimate ± σ is still reported); the text says which.
   - Reported apart, not folded into σ: the scale term (value × the layer's `uncertaintyRelative`: 1.7 % for 090, 1.6 % for 030) and the floor term (height on the two-view sweep floor near the object minus on the report floor, from clearance B).
6. **Measurement layer (`workcell_layer_build.py`).**
   - A model-path value within max(2 cm, 2σ) of the user's field value verifies the object's confidence ("与现场值吻合"); one off by more than max(3 cm, 3σ) caps it at low. A two-view value does not use the model and does not count for the model's confidence.
   - "Model lower part missing" only where the model's lowest point is > 5 cm above the field edge (the confidence cap's rule).
   - The e-stop guard is by entity: every fact of the e-stop's host entities (those with a scale / reference-object / e-stop recheck fact, and the config's button host) except confidence, pipeline and check facts is frozen; no wording rule.
   - A configured check results file that does not exist fails the build instead of being skipped.

**Status (2026-10-06, rerun after the review fixes).**
- 030R housing: **22.5 ±0.85 cm, trusted**, 1.5 cm below the field 24 cm (within max(2 cm, 2σ): the object's confidence records "与现场值吻合").
- 090R front plate: 21.6 ±2.4 cm, large (no field value).
- No value: 090R housing 23.5 ±2.9 (σ > 2.5); 030L housing 25.5 ±4.2 (σ > 2.5; 12 grazed lines excluded; edge = model bottom); 030 rail 14.4 ±5.2 (σ > 2.5); 090R rail (0 % overlap between the two photos); 090L housing and 090L rail (single usable view).
- σ is still dominated by the Pi3X–model range gap: over the 10–150 px band 5.2–6.7 cm at 090R, 10.3 cm at 030L, 0.55 cm at 030R.

```sh
python scripts/workcell_checks/lower_edge.py --view VIEW.json --photos-dir DIR --photo ID=FILE,... --layer-url URL --api ORIGIN \
    --opts CLICKS.json --prompts-out P.json      # CLICKS = {targets: [{entityId, part, clickPhoto, clickXY}]}; CPU
modal run modal_apps/workcell_part_masks.py --view VIEW.json --photos-dir DIR --photo ID=FILE,... --prompts P.json --out PARTS   # L4, ~$0.003
python scripts/workcell_checks/lower_edge.py ... --opts OPTS.json --out LE/results.json
    # OPTS = {targets (as CLICKS), partMasks: PARTS/part-masks.json, sweepFloorAboveReportCm: {entityId: cm}}; CPU, ~20-75 s
python scripts/workcell_layer_build.py --config configs/workcell-layers/090.json --layer IN.json --out OUT.json \
    --data research-notes/workcell-lower-edge-2026-10-05/data --root NOTES=... --root RUNS=...   # durable views + 030 layer patch
```
