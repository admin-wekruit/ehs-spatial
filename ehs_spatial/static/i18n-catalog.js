// Hand-authored interface and report commentary. Original findings, evidence and user text remain verbatim.
window.panoptesTranslations=Object.fromEntries(String.raw`
逐图检测复核	Per-photo detection review
四张来源照片 · 逐图物体范围	Four source photos · Per-photo object bounds
检测覆盖与来源记录 ↗	Detection coverage and provenance ↗
同源照片补充观测区域	Supplementary source-photo observations
完整工位 · 模型与同源观测范围	Full workcell · Models and source-photo bounds
保留原完整模型；补充经照片 SHA 与像素映射验证的逐帧观测。相同标签不是相同实物，未完成背面，不推断不可见对象。	The complete model is retained, supplemented with per-photo observations verified by image hashes and pixel mapping. Matching labels do not establish physical identity. Unseen surfaces have not been completed.
工位重建与物体范围	Workcell reconstruction and object bounds
当前物体没有可靠平面范围；原图证据仍可查看。	No reliable plan bounds are available for this object. Source-photo evidence remains available.
已有照片证据 · 空间范围不可用	Photo evidence available · Spatial bounds unavailable
照片观测范围 · 尚未生成完整模型	Photo-observed bounds · Full model not generated
Blender 参数化资产	Blender parametric asset
原生成资产摆放指标 · 历史实验	Original generated-asset placement metrics · Historical experiment
{scene} 个场景对象 · {observations} 条照片检测记录	{scene} scene objects · {observations} photo detections
{objects} / {inventory} 条原检测记录	{objects} / {inventory} original detection records
来源照片 {label} · 检测 {candidate_id}	Source photo {label} · Detection {candidate_id}
空间证据不足，保留原图中的物体选择。{reason}	Insufficient spatial evidence; the object remains selectable in the source photo. {reason}
已定位照片中的可见表面，可查看观测范围；完整模型尚未生成，遮挡部分未知。	The visible surface is located in the photo and its observed bounds are available. A full model has not been generated; occluded parts are unknown.
当前外形为 Blender 拟合圆柱；下方保留原生成资产的历史摆放实验指标，实际尺寸尚未标定。	The current shape is a Blender-fitted cylinder. Historical placement metrics for the original generated asset remain below. Physical dimensions are uncalibrated.
仅保留照片观测到的表面；未补全不可见范围。	Only photo-observed surfaces are retained; unseen areas have not been completed.
该对象没有生成资产摆放前后对比。	This object has no before-and-after generated-asset placement comparison.
生成资产、Blender 参数化物体与照片观测范围在同一工位查看。点选物体联动原图、3D、高宽深与坐标轴；尚未生成或空间证据不足的物体保留独立状态。	Inspect generated assets, Blender parametric objects and photo-observed bounds in one workcell. Select objects to link source photos, 3D, H/W/D and axes. Objects awaiting generation or lacking spatial evidence retain their own status.
报告缺少场景地址	The report has no scene address
完整工位与 Blender 柱体的实际三维预览	Actual 3D preview of the full workcell and Blender bollards
在完整工位中选择对象，查看原图、观测范围、高宽深和坐标轴。包含 Blender 柱体；检测到的可见表面与生成模型分别标明。	Select objects in the complete workcell to inspect source photos, observed bounds, H/W/D and axes. Blender bollards are included; observed surfaces and generated models are identified separately.
完整工位合并了对象重建、Blender 柱体和同照片的检测范围；原始生成与 Blender 对比保留在下方实验记录。另两个实验场保留各自的照片和坐标来源。切换前请下载 JSON 保存编辑。尺寸尚未标定；当前是实验结果，尚未完成 Lucida 论文复现。	The workcell combines object reconstruction, Blender bollards and detected bounds from the same photos. Original-generation and Blender comparisons remain in the experiments below. The other two playgrounds retain their own photos and coordinates. Download JSON to save edits before switching. Dimensions are uncalibrated; this experiment is not yet a reproduction of Lucida.
场景对象与照片检测记录	Scene objects and photo detections
模型与观测范围投影 · 非实测 CAD	Model and observed bounds projections · Not measured CAD
完整工位 · Blender 参数化与来源空间范围	Full workcell · Blender models and source bounds
保留完整场景及两个 Blender 参数化柱。蓝色虚线范围与 H/W/D 来自同坐标系、同物体来源照片的有效深度；仅代表可见表面。未标定为米，未合并旧四图重建。	The full scene and two Blender bollards are retained. Dashed blue bounds and H/W/D use valid depth from the same object and source coordinate system. They describe visible surfaces only. Units are not calibrated to metres.
打开报告 · 原图 / 3D / Agent →	Open report · Photos / 3D / Agent →
完整工位 + 物体范围	Full workcell + object bounds
原图证据 · 高宽深 · 坐标轴	Photo evidence · H/W/D · Axes
保留完整工位的 10 个场景对象。点选后同时查看模型外形、原图支持的范围、高宽深和坐标轴；两根防撞柱仍为 Blender 拟合圆柱。遮挡部分未知，尺寸未标定。	All 10 scene objects remain in the workcell. Select one to compare its model, photo-supported bounds, H/W/D and axes. The two bollards remain Blender-fitted cylinders. Occluded extent is unknown; dimensions are uncalibrated.
正在读取实验报告…	Loading experiment report…
无效的报告地址。	Invalid report address.
原生重建单位；尺度依据见各项量测	Native reconstruction units; scale provenance is recorded in each measurement
打开报告工作区 · Agent	Open report workspace · Agent
在来源照片上选定区域，通过 Agent 添加物体；空间范围和量测随报告一起更新。	Select a region in the source photo and add it through the agent. Spatial bounds and measurements update with the report.
观测背景必须保留原生坐标	Observed context must retain native coordinates
表面已由已有场景资产覆盖；此区域仅显示观测点范围。	The surface is covered by existing scene assets; this region shows observed-point bounds only.
观测区域 · 只读 · 可见表面，非完整模型 · {model} · 来源 {frames}。{description}	Observed region · Read-only · Visible surface, not a complete model · {model} · Source {frames}. {description}
观测区域 · 只读 · 可见表面，非完整模型 · {model} · 来源 {frames}	Observed region · Read-only · Visible surface, not a complete model · {model} · Source {frames}
工位物体与范围	Workcell objects & bounds
BOR1 · 工位物体与范围	BOR1 · Workcell objects & bounds
原图与 3D 联动 · 包围框与三轴	Linked source photo & 3D · Bounds and XYZ axes
查看各照片中的物体观测范围，在原图和 3D 中点选、查看包围框与三轴。两个按钮保留完整生成网格；其余区域来自分割和深度，尚未合并为跨视角实体。	Explore observed object extents in each photo. Select regions in the source photo or 3D to see bounds and XYZ axes. Two buttons retain full generated meshes; other regions come from segmentation and depth and have not been merged into cross-view entities.
{run} · {generated} 个生成资产 / {regions} 个观测区域 · 单位 {units}	{run} · {generated} generated assets / {regions} observed regions · Units: {units}
场景 Z 轴线夹角 {angle}°	Scene Z-axis angle {angle}°
显示所有范围	Show all bounds
观测区域可重叠，不代表独立完整物体。范围不是安全区域。	Observed regions may overlap and do not represent separate complete objects. Bounds are not safety zones.
无可用空间范围	No supported spatial bounds
观测区域 · 只读 · 可见表面，非完整模型	Observed region · Read-only · Visible surface, not a complete model
观测法向跨度 {height} · 未标定单位	Observed normal span {height} · Uncalibrated units
仅为可见表面范围；未观测部分未知。	Visible surface extent only; unobserved parts are unknown.
烘焙后的场景坐标轴；物体竖直轴未知。仅为可见表面范围；未观测部分未知。	Baked scene coordinates; the object's upright axis is unknown. Visible surface extent only; unobserved parts are unknown.
正在读取观测区域：{name}	Loading observed region: {name}
无效观测区域	Invalid observed region
观测区域面数不匹配	Observed-region face count mismatch
观测区域含无效或重复三角面	Observed region contains invalid or duplicate faces
可选区域过多	Too many selectable regions
机位的观测背景不匹配	The viewpoint's observed context does not match
已显示 {loaded} / {count} 个网格 · 下载 {bytes} / {total} MB	Showing {loaded} / {count} meshes · Downloaded {bytes} / {total} MB
已显示 {loaded} / {count} 个网格 · 加载未完成	Showing {loaded} / {count} meshes · Loading incomplete
未知	Unknown
俯视 · 导航参考	Top · Navigation reference
侧视 · 导航参考	Side · Navigation reference
正视 · 导航参考	Front · Navigation reference
观测范围	Observed extent
来源照片量测	Source-photo measurements
人工定标 · m	Operator-anchored scale · m
估计比例 · m	Estimated scale · m
未标定 · 模型单位	Uncalibrated · Model units
模型单位	model units
高 H {value}	Height H {value}
宽 W {value}	Width W {value}
深 D {value}	Depth D {value}
表面坡度 {value}	Surface slope {value}
主轴离竖直角 {value}	Principal-axis angle to vertical {value}
{value} 模型单位	{value} model units
当前模型 · 高 {height} / 宽 {width} / 深 {depth}；编辑不改变来源证据。	Current model · H {height} / W {width} / D {depth}; edits do not change source evidence.
可见表面；H/W/D 沿地面参考方向，XYZ 为场景参考轴。方向仅为几何估计。	Visible surface; H/W/D follow the floor reference, while XYZ are scene reference axes. Orientation is a geometric estimate.
来源量测来自可见表面；H/W/D 沿地面参考方向，XYZ 为模型参考轴。方向仅为几何估计。	Source measurements use the visible surface. H/W/D follow the floor reference; XYZ are model reference axes. Orientation is a geometric estimate.
未建立地面参考，尺寸与方向未知。	No floor reference is established; dimensions and orientation are unknown.
空间证据不足，无法给出尺寸与方向。	Insufficient spatial evidence for dimensions and orientation.
空间证据不足	Insufficient spatial evidence
无效量测证据	Invalid measurement evidence
尺寸与范围不匹配	Dimensions do not match their bounds
无效方向证据	Invalid orientation evidence
原图叠加	Photo overlay
自由 3D	Free 3D
显示模式	Display mode
模型不透明度	Model opacity
完整工位原照片	Original full workcell photo
原图叠加：点击物体查看坐标轴；自由旋转请切换自由 3D。	Photo overlay: select an object to see its axes. Switch to Free 3D to orbit.
原照片加载失败，请重新加载。	The original photo failed to load. Please reload.
原照片相机字段不完整	The original-photo camera fields are incomplete
此对象没有来源机位。	This object has no source viewpoint.
观测背景必须只读且不可选	Observed context must be read-only and non-selectable
对象的来源机位不匹配	The object source viewpoint does not match
对象的观测背景不匹配	The object's observed context does not match
观测背景不可编辑	Observed context cannot be edited
场景中没有此对象：{id}	Object not found in this scene: {id}
Agent 对话	Agent conversation
对话记录仅在授权工作区查看。	Conversation history is available only in the authorized workspace.
上传与 Agent 工作区 ↗	Uploads & agent workspace ↗
进入完整工作区 ↗	Open full workspace ↗
上传工位照片，或与 Agent 对话补充报告。	Upload workcell photos or add to a report through the agent chat.
来源快照时间	Source snapshot time
“前”是 RecGen 原生生成的网格和预测摆放；“后”是同一网格经过全部可用对象视图共同约束的局部摆放优化。没有替换物体或修改照片。轮廓距离除以图像高度；深度误差是与源几何估计深度比较的中位相对误差，不能解释为实测尺寸精度。	“Before” uses the mesh and placement predicted by RecGen. “After” uses the same mesh with locally optimized placement constrained by all available object views. Objects and photos are unchanged. Boundary distance is normalized by image height. Depth error is the median relative error against the source geometry estimates, not measured dimensional accuracy.
“观测表面”只包含照片可见的三角面。各对象使用的视图逐行列出，本表为输入一致性拟合，不是独立测试集指标。封闭网格仅表示几何拓扑，不证明被遮挡的形状正确。	The observed surface contains only triangles visible in the photos. Each row identifies the views used for that object. These metrics measure consistency with fitting inputs, not performance on an independent test set. A watertight mesh describes topology; it does not prove hidden shapes are correct.
其余 {n} 条候选尚未生成。逐项状态见	The remaining {n} candidates have not been generated. See individual statuses in
小物体补全	Small-object reconstruction
急停与控制按钮 · 独立生成网格	Emergency stop and control · Separate meshes
原 BOR1 检测照片的两个按钮，保留原生网格与独立对象轴。使用原报告几何坐标，未与三照片重建场景混合；物体语义与尺寸仍需核验。	Two buttons from the original BOR1 inspection photos, with native meshes and individual object axes. They retain the inspection coordinate system. Semantics and dimensions need verification.
BOR1 · 小物体补全	BOR1 · Small-object reconstruction
原报告照片的急停与控制按钮，使用原几何坐标；地面为观测表面。尺寸尚未标定。	Emergency stop and control from the inspection photos, in their source geometry coordinates, with observed floor. Scale is uncalibrated.
小物体补全的实际三维预览	Actual 3D preview of small-object reconstruction
小物体补全 · 可交互实验场	Small objects · Interactive playground
全选	Select all
全不选	Clear selection
重置	Reset
内部模型	Interior model
测量点云	Measurement point cloud
语义色 / 照片色	Semantic / photo colors
场景点 开/关	Scene points on/off
物体列表 · 点选 / 多选	Objects · Select / multiselect
拖动旋转 · 滚轮缩放 · 右键拖动平移	Drag to orbit · Scroll to zoom · Right-drag to pan
点击物体名或点云高亮,其余变暗 · Shift 多选 · Alt 隐藏	Click an object name or point cloud to highlight it · Shift to multiselect · Alt to hide
视角锚定在相机 1 的拍摄位姿,绕其正前方 2.5 m 地面点旋转 · 相机 N 切到该帧	View anchored to Camera 1; orbit around the floor point 2.5 m ahead · Camera N selects that frame
内部模型加载中…	Loading interior model…
无法初始化 WebGL	Unable to initialize WebGL
列表量测来自测量点云	List measurements come from the measurement point cloud
拖动旋转 · 滚轮缩放 · 右键拖动平移 · Shift 多选 · Alt 隐藏。照片按钮回到真实源机位；列表量测来自测量点云。	Drag to orbit · Scroll to zoom · Right-drag to pan · Shift to multiselect · Alt to hide. Photo buttons restore source viewpoints; list measurements come from the point cloud.
拖动旋转 · 滚轮缩放 · 右键拖动平移 · Shift 多选 · Alt 隐藏。相机按钮回到测量点云机位。	Drag to orbit · Scroll to zoom · Right-drag to pan · Shift to multiselect · Alt to hide. Camera buttons restore point-cloud viewpoints.
该表面尚未建立对象对应，暂不能联动	This surface has no established object match and cannot link views yet
内部模型未覆盖该对象的机位/关联	The interior model does not cover this object's viewpoint or association
审核结论	Review disposition
分析版本	Analysis version
创建时间	Created at
四联动使用当前库存测量；判定明细沿用已保存的分析结果，重建视图不会重新评定规则。	Linked views use current inventory measurements. Findings retain saved analysis results; reconstruction views do not re-evaluate rules.
交互 3D（内部模型 / 测量点云）	Interactive 3D (interior model / measurement point cloud)
交互 3D（照片色 · 按实例）	Interactive 3D (photo colors · by instance)
尺度来源与置信	Scale source and confidence
物体（平面编号）	Object (plan number)
高 m	Height, m
尺寸 m	Dimensions, m
距相机 m	Distance from camera, m
足迹方法	Footprint method
判定明细	Finding details
无判定结果	No findings
无判定	No finding
备注	Note
阈值已按本 run 调整	Threshold adjusted for this run
VLM 枚举与去向（短语 → 实例数 / unresolved 原因）	VLM inventory and outcomes (phrase → instances / unresolved reason)
VLM 短语	VLM phrase
去向	Outcome
无（缺少 inventory/phrases.json）	Unavailable (inventory/phrases.json missing)
未落地：unresolved 无记录	No instance: no unresolved record
深度渲染	Depth render
点云透视	Point-cloud perspective
点云顶视	Point-cloud top view
测量平面图（CAD 版，全实例）	Measurement plan (CAD style, all instances)
证据图集（每帧 evidence overlay · 深度渲染 · 点云透视/顶视 · 平面图）	Evidence gallery (per-frame overlays · depth render · perspective/top-down point cloud · plan)
无平面对象	No plan objects
物体联动 · 原图 / 交互 3D / CAD 平面图 / 交互平面 — 点一个物体，四格同时选中（多选出间距矩阵）	Linked objects · Photo / 3D / CAD plan / interactive plan — Select an object in all four panels (multiselect for a distance matrix)
原图点选（按机位）	Photo selection (by viewpoint)
CAD 平面图（全实例 · 悬停看名称）	CAD plan (all instances · hover for names)
交互平面（点选出距离矩阵）	Interactive plan (select for a distance matrix)
实体测量（物体 / 数量 / 高 / 尺寸 / 距相机 / 足迹方法）	Entity measurements (object / count / height / dimensions / camera distance / footprint method)
物体	Object
数量	Count
高度	Height
点击取消	Click to deselect
接触边投影	Contact-edge projection
近簇裁剪	Near-cluster crop
共线对齐	Collinear alignment
人工补测	Manual supplemental measurement
法线分割	Normal segmentation
确认机器判定	Confirm automated finding
推翻机器判定	Override automated finding
FAIL 违规	FAIL · Violation
NEEDS REVIEW 待复核	NEEDS REVIEW
感知防护 SENSING/AOPD	Sensing safeguards / AOPD
控制防护 CONTROL	Control safeguards
防护罩/围护 GUARDS	Guards / enclosures
阻挡与引导 IMPEDING	Barriers / guidance
信息标识 INFO	Information / signs
物料/载具 PAYLOAD	Materials / carriers
未见/需现场核实：	Not observed / verify on site:
清单全部检出	All checklist items detected
（缺失清单为空）	(missing-item list is empty)
拒绝项（VLM 出框但裁剪自检未通过，不计入检出）：	Rejected detections (VLM boxes failed crop verification and are not counted):
拒绝项：	Rejected items:
装置检测清单（taxonomy 检测层 · VLM 出框+裁剪自检 → SAM box-prompt）	Device detection checklist (taxonomy layer · VLM boxes + crop verification → SAM box prompt)
装置检测清单	Device detection checklist
回投验证	Reprojection validation
回投验证（平面矩形基线投回照片 · 红点应压在结构接地线上 · 偏差=图高占比）	Reprojection validation (plan-rectangle baseline projected into photos · red points should lie on ground-contact lines · error normalized by image height)
人工框选补测	Manual region measurements
无补测记录	No supplemental measurement records
审核员	Reviewer
决定	Decision
推翻为	Overridden status
理由	Reason
时间	Time
Review 记录	Review record
Review 记录（审核员 · 决定 · 理由 · 时间）	Review record (reviewer · decision · reason · time)
无审核记录（review.json 缺失）	No review recorded (review.json missing)
附录（warnings 全量 · unresolved 全量 · run 参数）	Appendix (all warnings · all unresolved records · run parameters)
run 参数	Run parameters
相机高度：	Camera height:
—（未持久化）	— (not persisted)
policy 集：	Policy set:
后端来源：	Backend sources:
用户	User
已改动 run	Run modified
Agent 对话记录（本 run 上的问答 / 纠错 / 阈值调整，附时间）	Agent conversation (questions / corrections / threshold changes for this run, with timestamps)
无对话记录	No conversation recorded
无词表实体过证据门	No vocabulary entity passed the evidence gate
cell 矩形约束	Cell rectangle constraints
开口	Opening
未闭合（存在开口边）	Not closed (open edges remain)
边	Edge
偏移	Offset
支撑长度	Supported length
来源	Source
内部模型 · {frames} · 尺度未标定	Interior model · {frames} · Uncalibrated scale
测量点云 · {count} 点	Measurement point cloud · {count} points
{count} 三角面 · {matched} 个已关联对象	{count} triangles · {matched} associated objects
已选 {count} 个 · 当前 3D 可显示 {visible} 个	{count} selected · {visible} visible in the current 3D view
已选 {count} 个	{count} selected
内部模型未覆盖 {count} 个已选对象的机位/关联（列表已标注）	The interior model does not cover {count} selected objects' viewpoints or associations (marked in the list)
{name}：内部模型未覆盖该对象的机位/关联	{name}: the interior model does not cover this object's viewpoint or association
{count} 个物体 · {run}	{count} objects · {run}
高 {height} m · {size}	Height {height} m · {size}
距相机 {distance} m · {points} 点	Camera distance {distance} m · {points} points
相机 {number}	Camera {number}
内部模型仅覆盖 {frames}；当前照片机位未覆盖	The interior model covers only {frames}; the current photo viewpoint is not covered
内部模型加载失败：{error}	Interior model failed to load: {error}
当前机位可见 {visible} / {count} 个已选对象	{visible} / {count} selected objects are visible from this viewpoint
此机位可点选 {count} 个对象	{count} objects can be selected from this viewpoint
{message}；当前不可见：{objects}	{message}; not currently visible: {objects}
{message}；{count} 个对象的掩码不可用	{message}; masks unavailable for {count} objects
{count} 实例	{count} instances
{count} 项检出 · 编号=下方图例	{count} detections · Numbers refer to the legend below
scale {scale} · {count} 联动物体 · {entities} 判定实体	scale {scale} · {count} linked objects · {entities} assessed entities
枚举必有交代：每个 VLM 枚举短语要么落地为实例，要么在 unresolved 中记录原因。当前 {phrases} 短语 · {unresolved} 条 unresolved。	Every VLM phrase must produce an instance or an unresolved reason. Current: {phrases} phrases · {unresolved} unresolved records.
无（{file} 缺失）	Unavailable ({file} missing)
正视	Front
侧视	Side
正视 · 地面参考	Front · Floor reference
侧视 · 地面参考	Side · Floor reference
参考坐标轴	Reference axes
参考 {axis} 轴	Reference {axis} axis
网格局部轴；0° 为沿估计地面法线。结构实测倾角未知。完整编辑器可编辑对象。	Mesh-local axes; 0° follows the estimated floor normal. Measured structural tilt is unknown. Edit objects in the full editor.
轴来自圆柱参数；0° 为沿估计地面法线。结构实测倾角未知。完整编辑器可编辑对象。	Axes come from cylinder parameters; 0° follows the estimated floor normal. Measured structural tilt is unknown. Edit objects in the full editor.
烘焙后的场景坐标轴；物体竖直轴未知。结构实测倾角未知。完整编辑器可编辑对象。	Baked scene-coordinate axes; the object's upright axis and measured structural tilt are unknown. Edit objects in the full editor.
选中对象包围框与局部坐标轴	Selected object bounding box and local axes
拖动彩色轴移动 · 方向键微调	Drag a colored axis to move · Arrow keys for fine adjustments
沿参考 {axis} 轴移动物体	Move object along reference {axis} axis
法向高度 {height} · 未标定单位	Normal height {height} · Uncalibrated units
圆柱参考轴夹角 {angle}°	Cylinder reference-axis angle {angle}°
局部 Z 轴线夹角 {angle}°	Local Z axis angle {angle}°
网格局部轴；0° 为沿估计地面法线。结构实测倾角未知。拖动彩色轴可移动对象。	Mesh-local axes; 0° follows the estimated floor normal. Measured structural tilt is unknown. Drag a colored axis to move the object.
轴来自圆柱参数；0° 为沿估计地面法线。结构实测倾角未知。拖动彩色轴可移动对象。	Axes come from cylinder parameters; 0° follows the estimated floor normal. Measured structural tilt is unknown. Drag a colored axis to move the object.
烘焙后的场景坐标轴；物体竖直轴未知。结构实测倾角未知。拖动彩色轴可移动对象。	Baked scene-coordinate axes; the object's upright axis and measured structural tilt are unknown. Drag a colored axis to move the object.
1 份已发布报告	1 published report
Panoptes · 报告历史	Panoptes · Report history
联合报告 ↗	Unified report ↗
已发布的工位报告，保留照片、空间证据和交互实验。	Published workcell reports with their photos, spatial evidence and interactive experiments.
正在读取报告历史…	Loading report history…
此处只列已真实发布的报告。	Only actually published reports are listed here.
已发布	Published
打开报告与 Playgrounds ↗	Open report and playgrounds ↗
发布完成时间	Publication completed
报告目录无效	Invalid report catalog
报告历史加载失败：{error}	Report history failed to load: {error}
{count} 张源照片	{count} source photos
{count} 个 Playgrounds	{count} playgrounds
{count} 份已发布报告	{count} published reports
来源快照 {commit}	Source snapshot {commit}
Panoptes · BOR1 工位联合报告	Panoptes · BOR1 workcell report
PANOPTES · BOR1 工位联合报告	PANOPTES · BOR1 workcell report
Panoptes 报告首页	Panoptes report home
工位空间报告	Workcell spatial report
报告章节	Report sections
报告历史	Report history
四视图报告	Linked views
原检测	Inspection
实验对比	Experiments
现场证据，与三维场景一起看。	See site evidence alongside the 3D scene.
先进入不同实验场探索工位，再从四联动报告查看对象证据、原检测与实验指标。	Explore the workcell in different playgrounds, then inspect object evidence, original findings and experiment metrics in the linked report.
检测来源	Inspection source
重建来源	Reconstruction source
打开完整 3D 编辑器 ↗	Open full 3D editor ↗
同一工位 · 不同实验产物	One workcell · Different experiments
选择实验场	Choose a playground
完整对象重建	Complete object reconstruction
独立网格 · 点选、移动、隐藏	Separate meshes · Select, move, hide
照片表面重建	Photo surface reconstruction
内部 GLB · 原图对照、线框	Interior GLB · Photo comparison, wireframe
参数化柱体试验	Parametric bollard trial
参数化防撞柱试验	Parametric bollard trial
两根拟合圆柱 + 原始场景	Two fitted cylinders + Original scene
完整对象重建的实际三维预览	Actual 3D preview of the object reconstruction
探索 3D ↗	Explore 3D ↗
当前实验场	Current playground
Pi3X + RecGen · 9 个生成资产与 1 块观测地面。旋转场景，点选对象，再调整位置、旋转和缩放。	Pi3X + RecGen · 9 generated assets and 1 observed floor. Orbit the scene, select objects, then edit position, rotation and scale.
全屏	Fullscreen
返回预览	Back to preview
单独打开 ↗	Open separately ↗
展示的是我们的实际实验产物。照片表面重建来自较早的独立 run；参数化试验仅替换两根柱体。切换实验场前，请下载 JSON 保存编辑。各实验场的编辑不会改写下方报告与指标。各实验场均未标定真实尺寸，也不代表已完成 Lucida 论文复现。	These are our actual experiment outputs. The observed surface comes from an earlier independent run; the parametric trial replaces only the two bollards. Download JSON to save edits before switching playgrounds. Playground edits do not change the report or metrics below. Real dimensions remain uncalibrated in all scenes; none is a completed Lucida reproduction.
参考 Lucida 的 Demo 展示方式 ↗	See Lucida's demo presentation ↗
正在读取冻结的报告与场景数据…	Loading the frozen report and scene data…
一个对象，四个视角	One object, four views
重建场景对象	Reconstructed scene objects
读取对象…	Loading objects…
照片与分割证据	Photo and segmentation evidence
原图视角	Source photo view
点击照片中的对象以联动其他视图	Select an object in the photo to link the other views
三张输入照片 · 点击轮廓选择对象	Three input photos · Select an outline
● 当前选择	● Selected
对象级 3D 重建	Object-level 3D reconstruction
完整编辑 ↗	Full editor ↗
可交互工位 3D 场景	Interactive 3D workcell scene
Pi3X + RecGen · Lucida 路线实验	Pi3X + RecGen · Lucida-inspired experiment
可旋转、缩放、点选	Orbit, zoom and select
CAD 式投影图	CAD-style projection
当前重建坐标	Current reconstruction coordinates
CAD 式平面投影，点击对象联动	CAD-style plan projection; select an object to link views
网格投影外包络（凸包）· 非实测 CAD	Projected mesh envelope (convex hull) · Not measured CAD
未标定单位	Uncalibrated units
交互平面	Interactive plan
缩小交互平面	Zoom out the plan
恢复交互平面视角	Reset the plan view
放大交互平面	Zoom in the plan
复位	Reset
可缩放拖动的工位平面	Zoomable and pannable workcell plan
滚轮缩放 · 空白处拖动 · 对象点选	Scroll to zoom · Drag blank space to pan · Select objects
△ 源相机	△ Source cameras
所选对象的逐图对比与对应记录	Per-view comparisons and matches for this object
上面两个平面视图与当前 3D 使用同一套坐标。原检测报告的 CAD 与尺度保留在下方；历史判定未针对新网格或后续编辑重新计算。	Both plans above share coordinates with the current 3D scene. Original inspection CAD and scale are retained below. Historical findings have not been recalculated for the new meshes or subsequent edits.
已有检测报告，保留原始依据	Original inspection, with its evidence preserved
已保存的规则判定	Saved rule findings
原检测 CAD	Original inspection CAD
原报告坐标	Original report coordinates
原检测 CAD，已匹配对象可联动新场景	Original inspection CAD; verified object matches link to the new scene
全部检测记录	All inspection records
不同拍摄视角可能各有一条记录。只有同照片分割证据确认一致的对象，才关联到新场景。	Different camera views may each have a record. Objects link to the new scene only when segmentation evidence from the same photo confirms their identity.
重建改善在哪里，边界在哪里	What improved, and what remains unresolved
查看完整实验记录 ↗	View the full experiment record ↗
对象生成与摆放使用固定三张照片。下表以逐视图均值比较同一生成网格摆放前后与输入照片的一致性；深度参照 Pi3X 估计，不能解释为实测尺寸精度。	Object generation and placement use the same three photos. The table compares mean per-view consistency before and after placing the same generated mesh. Depth references Pi3X estimates, not measured dimensional accuracy.
规则形状，可以由参数生成。	Regular shapes can be generated from parameters.
两根防撞柱从现有分割点云拟合半径、高度与位置，再由 Blender 生成可编辑圆柱。拟合脚本新增 VLM 调用 0 次，计算约 1.38 秒；不包含之前的重建、分割及人工准备成本。	Radius, height and position are fitted to the segmented point clouds for two bollards. Blender then creates editable cylinders. Fitting adds 0 VLM calls and takes about 1.38 seconds, excluding prior reconstruction, segmentation and manual preparation.
右柱轮廓更贴合，但深度误差增加；圆柱也没有底板与螺栓细节。文件保留原场景与参数化试验两个场景，可自行切换检查。	The right bollard's outline fits better, but its depth error increases. Cylinders also lack baseplates and bolt details. The file retains both the original and parametric scenes for comparison.
下载可编辑 .blend	Download editable .blend
参数来源与逐项对比 ↗	Parameter sources and comparisons ↗
Blender 中保留原工位资产并使用两根参数化防撞柱的实际渲染	Actual Blender render retaining the workcell assets with two parametric bollards
Blender 实际渲染 · 参数化柱体试验	Actual Blender render · Parametric bollard trial
地面、相机和尺寸参数从哪里来？	Where do floor, camera and dimension parameters come from?
地面朝向与位置	Floor orientation and position
照片地面分割 ∩ 人工圈定内部范围 → Pi3X 三维点 → 平面拟合。	Photo floor segmentation ∩ manually outlined interior → Pi3X 3D points → plane fitting.
地面范围	Floor extent
照片 3 中可见地面像素三角化，保留空洞，不假定完整厂房边界。	Triangulated visible floor pixels in Photo 3, preserving holes without assuming the full factory boundary.
相机与对象位置	Camera and object positions
Pi3X 联合几何、RecGen 初始摆放，再用多视图轮廓与深度做数值优化。	Pi3X joint geometry and RecGen initial placement, followed by numerical optimization against multi-view silhouettes and depth.
真实米制尺寸	Real metric dimensions
尚未标定。需要可靠实测长度或对应拍照相机高度确定整体比例，局部几何仍需核验。	Uncalibrated. A reliable measured length or the corresponding camera height is needed to establish scale. Local geometry still requires validation.
质量、摩擦、厚度、关节	Mass, friction, thickness and joints
当前没有输入依据，未设置。	Not set; no supporting input is available.
检测、重建、拟合的来源分别保留 · 静态研究快照	Separate inspection, reconstruction and fitting provenance · Static research snapshot
此报告需要启用 JavaScript 才能进行四视图联动。	JavaScript is required for linked interaction across the four views.
打开静态实验记录	Open the static experiment record
工位 · 对象场景编辑	Workcell · Object scene editor
读取场景信息…	Reading scene information…
← 总报告	← Full report
← 联合报告	← Unified report
实验报告 ↗	Experiment report ↗
Blender 文件 ↗	Blender file ↗
恢复原始场景	Restore original scene
导入编辑	Import edits
下载场景 JSON	Download scene JSON
下载原始 GLB	Download original GLB
场景对象	Scene objects
全部显示	Show all
全部隐藏	Hide all
在列表或模型中选择对象，即可调整其变换。	Select an object in the list or model to edit its transform.
聚焦	Focus
恢复此对象	Reset object
同图对比	Same-view comparison
全部可用输入视图共同约束摆放；编辑后的变换尚未重新评估。	All available input views constrain placement. Edited transforms have not been re-evaluated.
原始指标记录	Original metric record
查看逐视图指标 ↗	View per-view metrics ↗
视角	View
整体视角	Overview
俯视	Top view
原图对照	Compare photo
柔和光照	Soft lighting
可编辑3D场景，拖动旋转，Shift拖动平移，滚轮缩放，单击选中对象	Editable 3D scene: drag to orbit, Shift-drag to pan, scroll to zoom, click to select
场景源照片	Scene source photo
拖动旋转 · Shift / 右键拖动平移 · 滚轮缩放 · 单击选中	Drag to orbit · Shift / right-drag to pan · Scroll to zoom · Click to select
生成资产 / 观测表面独立标记	Generated assets / observed surfaces labeled separately
正在读取场景信息。	Reading scene information.
重新加载	Reload
工位重建 · 同图对比	Workcell reconstruction · Same-view comparison
← 返回联合报告与 Playgrounds	← Back to report and playgrounds
同一批照片，同一物体，同一指标	Same photos, same objects, same metrics
“前”是 RecGen 原生生成的网格和预测摆放；“后”是同一网格经过全部可用对象视图共同约束的局部摆放优化。没有替换物体或修改照片。轮廓距离除以图像高度；深度误差是与 Pi3X 估计深度比较的中位相对误差，不能解释为实测尺寸精度。	“Before” uses the mesh and placement predicted by RecGen. “After” uses the same mesh with locally optimized placement constrained by all available object views. Objects and photos are unchanged. Boundary distance is normalized by image height. Depth error is the median relative error against Pi3X estimates, not measured dimensional accuracy.
“观测表面”只包含照片可见的三角面，便于识别完整资产补全带来的误差。三张照片都参与联合几何，已分割的对象视图都参与摆放优化；本表为输入一致性拟合，不是独立测试集指标。封闭网格仅表示几何拓扑，不证明被遮挡的形状正确。	“Observed surface” includes only triangles visible in photos, helping expose errors from asset completion. All three photos contribute to joint geometry, and all segmented object views constrain placement. These are input-consistency fitting metrics, not held-out test results. A closed mesh describes topology; it does not establish that hidden shapes are correct.
这是 RecGen 的实际实验；没有运行未公开的 Lucida GizmoAct 策略。资产为静态可编辑网格，机器人关节和物理碰撞尚未验证。	This is an actual RecGen experiment. The unpublished Lucida GizmoAct policy was not run. Assets are static editable meshes; robot joints and physical collisions have not been validated.
这轮结果能说明什么	What this experiment establishes
三张输入照片得到 9 个生成对象与 1 块观测地面，共 2,359,872 个生成三角面。可在 3D 中选择、隐藏、移动、旋转和缩放对象，切换原图视角，并导出本地编辑。	Three input photos produced 9 generated objects and 1 observed floor, with 2,359,872 generated triangles. Objects can be selected, hidden, moved, rotated and scaled in 3D. Source-photo views and local edit export are available.
当前优先改进：补齐遮挡的机器人支撑与尺度证据；区分透明围栏框架、面板与透射背景；减少人工分割和跨视图身份确认。提高三角面数量本身不能解决这些问题。	Priorities: recover evidence for the occluded robot support and scale; distinguish transparent fence frames, panels and transmitted background; reduce manual segmentation and cross-view identity checks. Increasing triangle counts alone will not resolve these issues.
泛化尚未验证：当前只有一个工位，输入照片参与了优化，且存在人工提示。下一轮应冻结流程，在新工位使用独立留出视角与实测尺寸验收，并记录人工投入。此页面是静态场景研究演示，不是已完成的 Lucida 论文复现。	Generalization has not been validated: this is one workcell, input photos participate in optimization, and manual prompts were used. Next, freeze the pipeline and evaluate new workcells using held-out views and measured dimensions, recording manual effort. This is a static-scene research demo, not a completed Lucida reproduction.
方法来源：	Method sources:
。本轮使用的 RecGen 代码与权重限制非商业用途。	. The RecGen code and weights used in this experiment are restricted to non-commercial use.
完整记录	Full record
输入图	Input view
用途	Role
轮廓 IoU 前 ↑	Silhouette IoU before ↑
后 ↑	After ↑
轮廓距离 前 ↓	Boundary distance before ↓
后 ↓	After ↓
相对深度误差 前 ↓	Relative depth error before ↓
观测表面 IoU	Observed-surface IoU
其它视图 / 参与优化	Other view / Used for optimization
锚定视图 / 参与优化	Anchor view / Used for optimization
下载所有指标	Download all metrics
打开 3D 并下载 GLB	Open 3D and download GLB
Blender：原场景与参数化试验	Blender: original scene and parametric trial
下载可编辑 Blender 文件（约 63 MB）	Download editable Blender file (about 63 MB)
圆柱参数与来源	Cylinder parameters and sources
逐视图对比	Per-view comparison
重新打开验证记录	Reopen verification record
文件包含两个场景。顶部 Scene 下拉菜单切换“01 · 原始重建资产”与“02 · 参数化防撞柱试验”。原场景保留 10 个独立对象和 2,392,713 个三角面；试验场景用照片证据拟合的圆柱替换两柱作对照，每柱 256 个三角面。	The file contains two scenes. Use the top Scene dropdown to switch between “01 · 原始重建资产” (original assets) and “02 · 参数化防撞柱试验” (parametric bollards). The original retains 10 separate objects and 2,392,713 triangles. The trial replaces two bollards with cylinders fitted to photo evidence, each with 256 triangles.
选中 parametric 柱，在 Object Properties → Custom Properties 修改 radius_native / height_native，即可改变半径和高度；位置用 Location 调整。	Select a parametric bollard and edit radius_native / height_native under Object Properties → Custom Properties to change radius and height. Adjust its position using Location.
所有尺寸仍为未标定的模型单位，不是米。	All dimensions remain in uncalibrated model units, not metres.
原始资产在 Blender 的照片三视角渲染	Original assets rendered in Blender from the Photo 3 viewpoint
原始生成资产 · Blender 实际渲染	Original generated assets · Actual Blender render
两根参数化圆柱在同一 Blender 视角的渲染	Two parametric cylinders rendered from the same Blender viewpoint
参数化圆柱候选 · 同一相机和光照	Parametric cylinder candidate · Same camera and lighting
对象	Object
平均轮廓 IoU 前 → 后 ↑	Mean silhouette IoU before → after ↑
平均相对深度误差前 → 后 ↓	Mean relative depth error before → after ↓
左防撞柱	Left bollard
右防撞柱	Right bollard
同 run、同三张照片、同原始 mask 和深度、同 288 高度评估网格。这里的“前”是已优化摆放的生成资产，“后”是拟合圆柱；深度是 Pi3X 估计值，不是测量真值。左柱平均指标改善，但照片 2 的 IoU 略降；右柱深度误差增加。圆柱没有底板、螺栓等细节，也未验证接地/碰撞，因此候选单独保留。	Same run, three photos, original masks and depth, and a 288-pixel-high evaluation grid. Here “before” is the placement-optimized generated asset, and “after” is the fitted cylinder. Depth is estimated by Pi3X, not measured ground truth. The left bollard improves on average, but Photo 2 IoU drops slightly; right-bollard depth error increases. Cylinders omit baseplates and bolts, and ground contact/collisions remain unverified, so the candidate is retained separately.
两柱拟合与评估约 1.382 秒，脚本新增 VLM 调用 0 次、模型推理 0 次。这不包含已有重建、分割、人工证据准备和 Python 启动成本，也不是全流水线成本。	Fitting and evaluation take about 1.382 seconds for both bollards, adding 0 VLM calls and 0 model inferences. This excludes existing reconstruction, segmentation, manual evidence preparation and Python startup; it is not the full pipeline cost.
地面和尺寸参数从哪里来？	Where do floor and dimension parameters come from?
照片 1、3 的地面分割 ∩ 人工内部范围 → Pi3X 点云 → RANSAC + SVD；35,487 个观测点。	Floor segmentation in Photos 1 and 3 ∩ manually outlined interior → Pi3X point cloud → RANSAC + SVD; 35,487 observed points.
照片 3 可见地面像素三角化，保留空洞；不是整个厂区边界。	Triangulated visible floor pixels in Photo 3, preserving holes; not the full site boundary.
相机位置与焦距参数	Camera position and focal parameters
Pi3X 联合三图估计和针孔拟合，非实测相机标定。	Joint Pi3X estimation from three photos and pinhole fitting, not measured camera calibration.
柱体半径、高度、位置	Bollard radius, height and position
分割后的三维点 + 圆柱形状先验，经过多视图轮廓/深度数值拟合。	Segmented 3D points + a cylinder shape prior, numerically fitted against multi-view silhouettes and depth.
估计地面定义的坐标基准，已保存原坐标到 Blender 的可逆变换。	Coordinate reference defined by the estimated floor. The reversible transform from original coordinates to Blender is saved.
尚缺实测标尺；同两点的实测长度 ÷ 模型长度可定整体比例，局部几何仍需核验。	A measured reference is still missing. Measured distance ÷ model distance between the same two points can establish scale; local geometry still needs validation.
厚度、质量、摩擦、关节	Thickness, mass, friction and joints
没有输入依据，未赋值。	Not assigned; no supporting input is available.
Blender Text Editor 中的 PARAMETERS.json、POST_FIT_PARAMETERS.json 保存参数来源和转换。源照片没有打包进下载文件。	PARAMETERS.json and POST_FIT_PARAMETERS.json in Blender's Text Editor retain parameter sources and transforms. Source photos are not packaged in the download.
BOR1 · 观测内部表面 Playground	BOR1 · Observed interior surface playground
BOR1 · 观测内部表面	BOR1 · Observed interior surface
原始照片机位	Source-photo viewpoints
显示控制	Display controls
线框	Wireframe
实体	Solid
整体视图	Overview
独立两照片实验 · 尺度未标定 · 未与其它场景配准	Independent two-photo experiment · Uncalibrated scale · Not aligned with other scenes
工位三维表面	3D workcell surface
当前机位的原始照片	Original photo from the current viewpoint
拖动旋转 · 滚轮缩放 · 右键或双指平移	Drag to orbit · Scroll to zoom · Right-drag or two fingers to pan
正在加载工位重建…	Loading workcell reconstruction…
右围栏	Right fence
工业机器人	Industrial robot
左围栏	Left fence
料车	Parts cart
黄黑护罩	Yellow-black guard
左光幕	Left light curtain
右光幕	Right light curtain
观测地面	Observed floor
生成资产	Generated asset
观测表面	Observed surface
参数化资产	Parametric asset
生成	Generated
观测	Observed
参数化	Parametric
俯视 · 地面法向	Top view · Floor normal
对象聚焦	Object focus
自由视角	Free view
未提供模型名称	Model name unavailable
参数化试验对比	Parametric trial comparison
观测记录	Observation record
此对象没有提供对比指标；编辑后的变换尚未重新评估。	No comparison metrics were provided for this object. Edited transforms have not been re-evaluated.
原生成柱与参数化圆柱的同视图对比；编辑后的变换尚未重新评估。	Same-view comparison of the original generated bollard and parametric cylinder. Edited transforms have not been re-evaluated.
此为照片观测表面，不提供生成资产摆放前后对比。	This is a photo-observed surface; no generated-asset placement comparison is provided.
查看参数化试验对比 ↗	View parametric trial comparison ↗
未标定尺度（非米）	Uncalibrated scale (not metres)
旋转（度 · XYZ）	Rotation (degrees · XYZ)
缩放	Scale
对象变换已更新；下载 JSON 保存编辑。	Object transform updated. Download JSON to save edits.
原照片加载失败；3D 场景仍可操作。	The source photo failed to load. The 3D scene remains interactive.
已恢复加载时的原始场景。	Restored the original scene as loaded.
已恢复同 run 对象的变换与可见性；导入文件中的资源地址未读取。	Restored object transforms and visibility for this run. Asset URLs from the imported file were not read.
3D 显示中断 · 请重新加载	3D display interrupted · Reload required
WebGL 上下文已丢失；请先下载当前编辑，再重新加载。	WebGL context lost. Download your current edits, then reload.
生成资产 / 观测表面 / 参数化资产独立标记	Generated / observed / parametric assets labeled separately
原照片机位使用原始内参与外参；生成资产的完整性和位置以实际验证为准。	Source-photo views use the original camera intrinsics and extrinsics. Generated asset completeness and placement require validation.
实际 Blender 场景 02：只用拟合圆柱替换两根防撞柱，其余八个对象与原相机不变。此候选未替换原始生成场景；浏览器修改仅作用于本地编辑。	Actual Blender Scene 02: only two bollards are replaced by fitted cylinders; the other eight objects and original cameras are unchanged. This candidate does not replace the original generated scene. Browser changes affect local edits only.
冻结 Pi3X 观测点与多视图数值拟合；圆柱先验，无底板。统一深色用于展示，不是恢复的物理材质。	Frozen Pi3X observations and multi-view numerical fitting; cylinder prior without a baseplate. Uniform dark color is for display, not a recovered physical material.
未选择对象	No object selected
照片观测表面	Photo-observed surface
独立生成资产	Separate generated asset
与原检测 inventory 暂无可靠对应	No verified match in the original inspection inventory
点击照片、3D 或平面中的对象	Select an object in the photo, 3D or plan
地面仅保留照片观测到的表面；未补全不可见范围。	The floor retains only photo-observed surfaces; unseen areas are not completed.
四个视图共享同一个对象选择。	All four views share the selected object.
轮廓 IoU · 摆放前 → 后	Silhouette IoU · Placement before → after
相对深度误差 · 前 → 后	Relative depth error · Before → after
同一输入视图	Same input view
轮廓 IoU ↑	Silhouette IoU ↑
相对深度误差 ↓	Relative depth error ↓
边界误差 / 图高 ↓	Boundary error / image height ↓
该对象没有生成资产摆放前后对比。	No generated-asset placement comparison is available for this object.
原报告没有可用 CAD 文件。	No CAD file is available in the original report.
橙色为同一照片分割已核对的原记录。原 CAD 保留原坐标，没有与新重建叠加。	Orange marks original records verified against segmentation in the same photo. Original CAD retains its coordinates and is not overlaid on the new reconstruction.
原 CAD 使用旧报告坐标。当前场景对象没有可靠对应时，不强行关联。	Original CAD uses the earlier report's coordinates. Objects without a verified match remain unlinked.
未通过	Fail
通过	Pass
证据不足	Insufficient evidence
未提供	Not provided
新重建尺度尚未标定	New reconstruction scale is uncalibrated
没有审核记录	No review recorded
查看保存的证据	View saved evidence
原始证据与限制记录 JSON ↗	Original evidence and limitations JSON ↗
记录	Record
原标签	Original label
来源视图	Source view
对应新场景	New scene match
暂无可靠对应	No verified match
同图轮廓 IoU 前 → 后 ↑	Same-view silhouette IoU before → after ↑
相对深度误差前 → 后 ↓	Relative depth error before → after ↓
输入视图	Input views
后侧 ↑ · 按照片 3 方向显示	Rear ↑ · Oriented to Photo 3
模型单位 · 非米	Model units · Not metres
workcell-reconstruction-01 · 较早的内部表面 GLB。切换原始照片机位、原图对照和线框；未拍到的区域保留为空。	workcell-reconstruction-01 · Earlier interior-surface GLB. Switch source-photo viewpoints, photo comparison and wireframe. Unobserved regions remain empty.
原生 Blender 试验的实际几何：仅两根防撞柱换为拟合圆柱，其余 8 个对象保持不变。网页可调整对象变换；半径与高度参数可在下载的 .blend 中修改。	Actual geometry from the native Blender trial: only two bollards are replaced by fitted cylinders; the other 8 objects are unchanged. Edit object transforms on the web; edit radius and height in the downloaded .blend.
当前浏览器未进入全屏，可使用“单独打开”继续操作。	Fullscreen could not be opened. Use “Open separately” to continue.
零长度方向	Zero-length direction
对象变换必须为有限数值，缩放必须大于零。	Object transforms must be finite numbers; scale must be greater than zero.
无效相机内参	Invalid camera intrinsics
无效 camera_to_world	Invalid camera_to_world
相机旋转不是正交矩阵	Camera rotation is not orthogonal
相机坐标手性无效	Invalid camera coordinate handedness
不支持或不完整的场景格式	Unsupported or incomplete scene format
场景说明必须是文字	Scene description must be text
重复相机 ID	Duplicate camera ID
无效对象/索引网格/来源帧	Invalid object, indexed mesh or source frame
对象缺少边界或下载记录	Object bounds or download metadata are missing
对象说明必须是文字	Object description must be text
scene.bin 网格或索引越界	scene.bin mesh or index is out of bounds
网格或索引数据区间重叠	Mesh or index data ranges overlap
网格含非有限数值或无效颜色	Mesh contains non-finite values or invalid colors
三角面索引超出顶点范围	Triangle index exceeds the vertex range
只能导入当前 run 的同版本、同对象场景。	Import requires the same run, scene version and objects.
导入对象 ID 或可见性不匹配。	Imported object IDs or visibility do not match.
资源必须是同源相对路径	Assets must use same-origin relative paths
禁止跨源资源	Cross-origin assets are not allowed
此浏览器无法创建 WebGL 场景。	This browser cannot create a WebGL scene.
此浏览器不支持 uint32 网格索引（OES_element_index_uint）。	This browser does not support uint32 mesh indices (OES_element_index_uint).
3D 显示已中断，请重新加载	3D display interrupted; please reload
编辑 JSON 超过 10MB	Edit JSON exceeds 10 MB
对象边界与原始数据不匹配	Object bounds do not match the original data
浏览器无法分配此对象的图形内存	The browser cannot allocate graphics memory for this object
报告资源必须来自本站	Report assets must come from this site
报告缺少对象或照片数据	The report is missing object or photo data
场景对象或平面数据无效	Invalid scene object or plan data
照片尺寸无效	Invalid photo dimensions
缺少原报告数据	Original report data is missing
未知照片视角	Unknown photo view
缺少平面相机方向	Plan camera direction is missing
平面范围为空	Plan bounds are empty
需要真实照片机位	Source-photo viewpoints are required
缺少有效网格边界	Valid mesh bounds are missing
网格边界为空	Mesh bounds are empty
workcell.glb 不包含三角表面	workcell.glb contains no triangle surfaces
无效对象文件记录	Invalid object file metadata
此浏览器版本不支持模型解压，请更新浏览器后打开	This browser version does not support model decompression. Update your browser to open the scene.
对象压缩文件大小不匹配	Compressed object file size mismatch
对象解压大小超出记录	Decompressed object size exceeds its metadata
对象文件不完整	Object file is incomplete
对象完整性校验失败	Object integrity check failed
下载连续 30 秒没有进展，请检查网络后重新加载	No download progress for 30 seconds. Check your connection and reload.
无效模型文件记录	Invalid model file metadata
无效模型分段	Invalid model chunk
模型文件不完整，请刷新重试	Model file is incomplete. Reload to retry.
模型文件大小不匹配	Model file size mismatch
模型完整性校验失败	Model integrity check failed
准备完整 GLB…	Preparing the full GLB…
{count} 三角面 · 封闭网格：是 ·	{count} triangles · Closed mesh: yes ·
{count} 三角面 · 封闭网格：否 ·	{count} triangles · Closed mesh: no ·
照片 {n}	Photo {n}
照片{n}	Photo {n}
选择{name}	Select {name}
选择检测记录{record}	Select inspection record {record}
照片中的{name}	{name} in the photo
CAD中的{name}	{name} in CAD
平面中的{name}	{name} in the plan
{frame} · {count} 个对象有分割证据	{frame} · Segmentation evidence for {count} objects
{name} · 模型相对单位	{name} · Relative model units
投影外包络宽 {width}	Projected envelope width {width}
已核对原检测记录：{records}	Verified inspection records: {records}
{count} 个输入视图提供分割证据。网格的遮挡背面由模型生成，实际尺寸尚未标定。	{count} input views provide segmentation evidence. Occluded mesh surfaces are generated; real dimensions remain uncalibrated.
原CAD检测记录{record}	Original CAD record {record}
原检测 #{record} · 暂无生成对象对应	Original inspection #{record} · No generated-object match
原检测 #{record} · {name}	Original inspection #{record} · {name}
已选择原记录 #{record}，没有可靠的新场景对应；未在新 3D 中猜测高亮。	Original record #{record} selected. No verified match in the new scene, so no 3D object is highlighted.
原检测：{fails} 项未通过 · {unknown} 项证据不足	Original inspection: {fails} failed · {unknown} insufficient evidence
{count} 个场景对象 · {photos} 张照片	{count} scene objects · {photos} photos
{count} 个场景对象 / {records} 条原检测记录	{count} scene objects / {records} original inspection records
{count} 条检测记录 · {labels} 类标签 · {cad} 个 CAD 投影。原尺度来源：{source}。这些是原 run 已保存的结果。	{count} inspection records · {labels} label categories · {cad} CAD projections. Original scale source: {source}. These are saved results from the original run.
展开全部 {count} 条原检测记录	Expand all {count} original inspection records
{name} ↗	{name} ↗
{name}的实际三维预览	Actual 3D preview of {name}
{name} · 可交互实验场	{name} · Interactive playground
{name} · 工位场景编辑	{name} · Workcell scene editor
{frame} · 同机位对照	{frame} · Same-viewpoint comparison
{frame} · 参考照片（当前为自由视角）	{frame} · Reference photo (currently in free view)
{frame} · 原照片机位	{frame} · Source-photo viewpoint
生成资产 · {model} · 来源 {frames}	Generated asset · {model} · Source {frames}
观测表面 · {model} · 来源 {frames}	Observed surface · {model} · Source {frames}
参数化资产 · {model} · 来源 {frames}。{description}	Parametric asset · {model} · Source {frames}. {description}
参数化资产 · {model} · 来源 {frames}	Parametric asset · {model} · Source {frames}
位置（{units}） {axis}	Position ({units}) {axis}
位置（{units}）	Position ({units})
旋转（度 · XYZ） {axis}	Rotation (degrees · XYZ) {axis}
缩放 {axis}	Scale {axis}
显示 {name}	Show {name}
已下载编辑后的 JSON；原始网格和原始 GLB 保持源资产。	Downloaded edited JSON. Original meshes and GLB remain the source assets.
已下载编辑后的 JSON；原始网格 保持源资产。	Downloaded edited JSON. Original meshes remain the source assets.
已显示 {loaded} / {count} 个对象 · 下载 {bytes} / {total} MB	Showing {loaded} / {count} objects · Downloaded {bytes} / {total} MB
正在载入 {name} · {bytes} / {total} MB	Loading {name} · {bytes} / {total} MB
已显示 {loaded} / {count} 个对象 · 加载未完成	Showing {loaded} / {count} objects · Loading incomplete
{run} · {count} 个对象 · {generated} 个生成 / {parametric} 个参数化 / {observed} 个观测 · 单位 {units}	{run} · {count} objects · {generated} generated / {parametric} parametric / {observed} observed · Units: {units}
{run} · {count} 个对象 · {generated} 个生成 / {observed} 个观测 · 单位 {units}	{run} · {count} objects · {generated} generated / {observed} observed · Units: {units}
场景已完整加载。{description}	Scene fully loaded. {description}
原图 · {frame}	Original photo · {frame}
{count} 张同状态照片 · 照片重建表面 · 未拍到处留空	{count} photos of the same state · Photo-reconstructed surface · Unobserved areas left empty
照片{n} · {id}	Photo {n} · {id}
照片 {n} · {id}	Photo {n} · {id}
报告数据加载失败：{status}	Report data failed to load: {status}
报告未完整加载：{error}	Report loading incomplete: {error}
导入失败：{error}	Import failed: {error}
加载失败：{error}	Loading failed: {error}
对象下载失败：HTTP {status}	Object download failed: HTTP {status}
模型加载失败：{status}	Model failed to load: {status}
scene.json 加载失败：{status}	scene.json failed to load: {status}
机位参数无效：{id}	Invalid viewpoint parameters: {id}
{name}：{error}	{name}: {error}
`.trim().split('\n').map(line=>line.split('\t')));
