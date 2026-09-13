# Browser QA — Spatial Studio

测试日期：2026-09-12。测试者：前端实现子任务。下文记录实际执行结果，不代表真实工厂安全结论或论文质量复现。

## 环境与入口

- 浏览器：独立 Playwright Chromium；开发前端 `http://127.0.0.1:5173/app.html`，代理 `/api` 到 `http://127.0.0.1:8792`。
- 数据：本地 PostgreSQL、实际文件资产和异步任务队列。没有配置付费重建/VLM提供方；本轮没有调用付费模型。
- 生产构建同一入口：`http://127.0.0.1:8792/app.html#/projects`。所有导航是网站内的 hash 路由；资产/API地址不作为导航目标。
- 真实完整工位报告：`/app.html#/reports/c0ffcb96-3eea-48be-8a9f-cd02023a6abc`。
- 真实组件报告：`/app.html#/reports/20962b70-3d10-4180-9d52-257deeb20ec1`。
- 真实完整工位工作台：`/app.html#/projects/a2b7c04d-0162-488b-b7db-37711a37ea62/workbench`。
- 自建验收项目：`/app.html#/projects/ab982c1b-be64-44e6-82c5-d0fd44e842c0/workbench`，名称 `Browser acceptance · BOR1`。
- 验收项目公开固定报告：`/app.html#/reports/9ac4014e-b978-4b9d-b664-2f05c248829e`，名称 `QA reviewed evidence · immutable report`。

管理能力只在原创建浏览器的 IndexedDB 或用户主动导出的管理文件内。本文和 URL 不包含管理密钥。开发站点和8792站点是不同浏览器 origin，不共享 IndexedDB；在另一个 origin 继续管理需导入自己的管理文件，或另存副本。

## 实际通过的交互

| 流程 | 执行与可核对证据 |
| --- | --- |
| 创建与照片上传 | 通过页面创建验收项目，上传现有 BOR1 原图 `observed/images/frame_0003.png`。无相机姿态/几何时仍显示输入照片。服务器自动建立分析任务；未配置提供方返回 `incomplete/provider_not_configured`，没有显示虚构成功结果。 |
| 对象不依赖网格 | 用正式编辑API添加明确命名的QA对象及7×7原图框，页面可选中、改名、保存。真实导入工位的无网格 bollard `0992e0d8-82e3-5449-816d-159ddb48221f` 经列表点击可选中，显示“对象已保留/尚无空间表示”，不会从对象清单消失。 |
| 小按钮 | 真实工位 control button `33e5c934-cbae-5053-9b71-8c7bf63a072c` 经列表点击可选中；来源照片/选择状态和属性同步。原始像素映射及49像素小框优先于包围围栏由可运行核心检查验证；没有把边框命中检查宣称为SAM像素掩膜命中。 |
| 无地面参数建模 | 页面新增 `QA parametric block`，显式创建用户建模坐标系、局部原点、未标定单位和2×1×3参数箱体；地面保持null，来源量测与地面倾角显示未知。实体 `c19d9acc-b080-4c8b-b16a-d939b202809e`。 |
| 修改/撤销/重做/刷新 | 将箱体宽度2改为2.5，Enter提交；Undo回到2，Redo回到2.5；浏览器刷新后宽度仍2.5。保存通过服务端正式版本与逆操作接口。 |
| 规划与副本 | 页面新建 `QA alternative` 规划分支，切换语言保留分支；页面发布报告并另存副本，生成项目 `f1b470f9-6596-4551-9855-e29a33008bdc`，新浏览器能力可编辑该副本。后端后来修正新规划分支建立独立首版本；前端始终读取返回的 `headRevisionId`。 |
| 真实完整场景 | 完整工位自由3D在资产加载完成后显示左右围栏、料车、机器人与地面，大小可阅读。选择右围栏 `ce9516a5-a385-5ecb-b6f5-3d543f3ec941` 时显示XYZ轴/范围；候选放置标签与接受按钮可见。右围栏候选的对齐仍有偏差，未作为已确认结果发布。 |
| 手机布局 | Chromium视口390×844，页面 `scrollWidth === 390`；只有一个canvas，箱体、坐标轴、移动属性页和数值修改可用。切换中英文保留选中对象与宽度2.5。没有测试真实iPhone Safari/WebGL驱动。 |
| 报告及真实回放 | 验收报告绑定固定场景版本，列出真实编辑批次。打开模型后切换真实编辑记录“之后”，前后canvas数量均为1，无页面alert；回放请求保存的不可变版本，不合成演示轨迹。没有相机标定的QA箱体需切到自由3D查看，原图视图不会伪造相机对齐。 |
| EHS规则工作台 | 从Walking-working surfaces来源模板建立JDM规则，运行applicable/not_applicable/unknown三类测试并通过，保存草稿后显式发布。规则ID `8bfdaa5c-2813-492a-815a-5303fc2d5f63`，规则版本 `9a9e54aa-34c0-49cf-8dab-d55dbd5b5b43`。 |
| 适用性与人工证据 | 初始评估 `3507279c...` 为适用性未知。输入具名QA人工负面检查陈述后保存场景版本 `50291f4d...`，重评得到FAIL，旧结果仍在。陈述明确注明是验收测试，不是工厂实测。 |
| 复核与补证 | 保存具名复核，创建补证任务 `9c6e1262-2761-4713-b002-303787221ba5`。之后保存新的具名人工正面检查陈述，产生版本 `fe366c68-977b-421d-806e-580e84f86614`；重新评估 `6a6f7dc5-e40e-4931-ae1c-2f49ca83fac7`：QA对象PASS，未提供检查证据的箱体仍INSUFFICIENT_EVIDENCE。 |
| 补证闭环 | 在补证任务中选择同一实体/规则的新评估与精确人工陈述annotation `a357f232-7a0c-44bf-9194-e9a48c3ab367`，提交后显示“已闭环”和新版本。原始open记录仍为历史；不是覆盖原任务。 |
| 复核报告发布 | 从当前版本发布验收报告 `9ac4014e...`，页面实际显示Walking-working surfaces标题、PASS与证据不足两个结果、`QA reviewer`具名复核及理由。快照固定在 `fe366c68...`。 |
| 冲突保护 | 尝试在历史版本 `50291f4d...` 写新证据被服务端409拒绝，没有覆盖当前版本。页面现将历史补证按钮禁用并提供该分支当前版本链接。 |

## 此前工作台验收路径（报告优先更新见下文）

1. 从“项目→新建”选择场景或单物体，上传1–4张图片。任务状态说明实际处理中、完成、证据不足或配置错误。
2. 工作台左边选对象，中间切原图叠加/自由3D/四视图，右边查看属性或Agent。没有网格的对象也可选；不要把“未知”当作零。
3. 原图圈选区域可作为Agent上下文。Agent提出修改后点击“应用”，使用同一版本编辑接口；旧版本提案需要重新取得当前上下文。
4. 参数模型可通过加号创建。模型尺寸编辑不等于现场标定；来源量测只读。检查候选放置后显式接受，或使用坐标轴/参数校正。
5. 新建规划方案后在独立分支修改。主重建版本不随规划修改。Enter或结束拖拽保存一个批次；命名版本用于检查点。
6. EHS先建立来源明确的规则草稿，运行测试，再显式发布规则版本。评估固定场景版本；补充人工证据会新建版本，需要重新评估。
7. 在评估条目展开“人工复核”，填写署名、理由；需要补证时建立任务。新增证据并重评后，在任务里选择匹配的新评估与证据完成闭环。
8. 工作台“发布报告”产生固定分享链接。报告区分观测、可编辑模型、规划方案；Blender是导出项，不作为重复Playground。访客修改仅在浏览器中；另存副本后拥有新项目。

## 已知边界与未声明通过项

- 没有调用新的真实重建模型或VLM，因此没有质量/速度/成本基准，也没有验证付费Agent推理质量。无提供方的快速失败时长不能当模型分析速度。
- 导入资产的候选模型位置仍需人工对齐；“候选可看见”不等于已达到Lucida重建质量。新版报告模型层显示候选并明确提示待确认，不提升为确认结果。
- 上述早期验收使用原图坐标框。本次补齐了来源多边形的原图映射，轮廓和命中验收见下文；其他缺少映射的来源仍使用其真实原图框。
- 导入数据保留原始实体和待确认关联，可能存在同名或跨视图未合并的实例。没有按名称盲目合并，也没有宣称跨视图自动关联已完成。
- 应用导航、表单、状态和错误使用全局中英文。来源对象名称/条文/用户陈述保留原文；JDM第三方编辑器内部菜单仍为英文，它没有现成的界面本地化接口。
- 真实桌面GLB与原始观测已验证加载；移动端只验证Chromium窄视口。未验证真实低内存iPhone、断网中途大文件恢复或全部GPU驱动。
- Blender导出资产/纹理保真由根任务另一条测试负责，不计入此前端浏览器通过列表。
- PostgreSQL迁移/更换适配器、反复重启期间曾出现503/500或暂时无法连接；在根任务修正PolicyRepository适配后，EHS全链条按上表重跑通过。这些失败没有被隐藏成业务成功。

## 可运行检查

从仓库根目录运行：

```sh
npm --prefix web run check
npm --prefix web run build
node --experimental-strip-types web/checks/renderer.mjs
```

核心检查覆盖原图letterbox映射、49像素对象优先选择、不改变来源的编辑预览、无地面不计算倾角/平面图、公共坐标系投影及姿态数学。原生渲染器检查由根任务维护。浏览器验证数据和结果见上述具体项目/版本，不注入生产演示假结果。

## 集成负责人最终生产构建复核

最终在 Codex 内置浏览器打开8792生产构建，实际切换自由3D与原图叠加，完整工位、当前选择的范围和XYZ仍在。随后刷新到最终脚本 `app-DMNDpVZJ.js`，打开真实项目历史并展开任务：任务 `7c703c1b...` 显示“部分完成”、8.8秒以及 `blender-validation.json`、`manifest.json`、`scene.blend`、`scene.glb` 四个下载入口。API下载的四个文件已另行验证哈希，见 `worker-export.json`。这补充了只读前端解析器检查，证明最终构建里实际显示下载入口。

## 完整报告 UI 更新验证

同日根任务与两条 UI 验证线，使用 Codex 浏览器工具操作实际8792构建。最终报告 `2d77667b-5369-4d66-b5c4-d426e9833e18`，内容清单及边界见 `REPORT-FIRST-UI.md`。

- 直接打开最终报告时，工位全景/四视图、图片理解、EHS、下载、分析版本均在同一页。此次验证未覆盖从报告库辨认并打开工位的完整路径；后续用户反馈暴露了 QA 发布与同名历史入口混杂的问题，见下节。
- 原图可键盘选择真实 control button；选择状态进入照片、CAD/平面与属性，同一实体始终保留。保存轮廓而非大框承担有轮廓对象的命中。
- 浏览器真实拖拽发现并修复 pointer-up 退出圈选后紧接 click 清空区域的问题。回归验证原图框 `2307.2,2440.7,2729.6,3042.3` 保留并带入 Agent 区域。
- 从公开报告点击“保存我的副本并开始复核”，实际新建 `ae9f501e-e15f-4c8b-a19f-833fbd8e88d7`，原图、实体、观察、选框及 Agent 打开状态保留；原发布版本不变，没有发送模型调用。
- 当前来源理解显示31条已关联/62条来源记录；其余31条保留在可展开的来源详情中。9条历史EHS结果可展开原条文及理由，同时当前场景明确显示尚未评估。
- 下载区显示本版本 Blender、GLB、验证和 manifest 四项；历史原始 `.blend` 另保留。当前后端四资产 HTTP 下载均通过大小和 SHA；不把此前原始文件冒充本版本导出。
- 完整后端回归：`562 passed, 28 skipped`。跳过项为未配置的外部条件，未计入通过；测试日志 `/tmp/panoptes-report-pytest.log`。
- 最终点云模式下，原图选择 control button 后，原图与3D都显示XYZ；图层切换不清空对象范围。背景点云不会成为可选业务对象。“适应画布”可明确重置3D视角，选对象本身不抢镜头。
- 最终390×844复测：页面scrollWidth390、一个canvas、只读模式零数字编辑框；中英切换与3D视图标签保留control button和XYZ。手机隐藏文字标签后的图层下拉补了显式accessible name；临时视口已复位。
- 管理型Codex/Chrome浏览器中的下载点击无页面/控制台错误且留在本站，但下载事件15/20秒等待未观察到，实际浏览器落盘文件名未验证。此项没有记为下载端到端通过；服务器下载字节/哈希与Blender重开验证是分别已通过的证据。

此轮未在工厂照片上重新运行付费模型，未宣称新的推理质量或稳定耗时。具体新增前端自检命令在 `REPORT-FIRST-UI.md`。

## 用户反馈后的实际入口复查

2026-09-12，根任务使用 `mcp__cua_repl` 操作同一8792网站，从报告库点击进入，不仅验证指定深链接。

- 用户提供的“1照片、2对象、绿色方块、两个空平面”组合在 QA publication `9ac4014e-b978-4b9d-b664-2f05c248829e` 中实际复现。当前 `2d77667b-5369-4d66-b5c4-d426e9833e18` 的网站和API一致为3照片、68非context记录、55有表示、9有模型；未复现这两个 publication 互相串数据。
- 原库将15次发布平铺，其中同一工位多条同名，且没有内容摘要。现在按项目ID整理为3个工位入口，首项为该工位最新发布，旧版在其历史下保留。不能根据照片或标题合并不同项目。
- 入口照片与计数从每份 publication 的固定快照投影，不读项目当前 head。无对象级观测表面时明确说明；有参数箱体不被称为已完成场景重建。滚动报告仍显示报告名称、照片数和版本。
- 实际点击原图1的 `control button` 成功；放大3D后背景与按钮范围、XYZ保留；实际拖动旋转与点击3D围栏成功。第三张照片中的另一个control button按其独立实体ID保留，没有根据相似名称擅自合并。
- 第三张照片与对象选择经过中英文切换、页面刷新后保留，报告快照仍为518e7869。当前浏览器未记录页面error/warn。
- 实际鼠标点击 CAD 围栏发现重叠轮廓会选中顶层的其他实体。共享 PlanView 现按实际多边形命中列出候选，未确认前不改变选择。最终构建中点选该位置显示12个候选，明确选择 `ce9516a5` 后原图、3D范围和交互平面的选中态一致。交互平面缩放后仍可列候选；首候选获得焦点，Escape关闭并返焦SVG，既有选择不变。
- 390×844下报告与报告库 `scrollWidth=390`，报告仅一个canvas；库入口显示实际缩略图及计数。此处是浏览器窄屏测试，不代表真实iPhone驱动测试。
- 本轮不增加模型调用，不改变历史报告内容、生成资产或安全事实。原有生成姿态偏差、未关联观察、未运行当前EHS评估及模型发布门槛仍须如实保留。

新增可执行检查：`web/tests/report-catalog-check.mjs`、`web/checks/plan-selection.mjs`；后端 `tests/test_platform_publications.py` 的固定摘要回归。最终 TypeScript/Vite 构建通过，publication/backend 23项测试通过，report-catalog、report-context、plan-selection、report-scene及renderer五个前端检查通过。完整模型质量与浏览器文件落盘并未因此获得新的通过声明。

## 管理者报告、地面语义与 CAD 对照复核

2026-09-12，根任务在实际8792构建中使用 CUA 验证；下列浏览器结果与本地导出验证分别记录。

- 报告开头说明本版本是否已有安全评估；模型工作台为主要编辑入口，规则来源与设置保留为次级入口。历史 EHS 的 FAIL、理由和原文完整保留，但不计作本版本合规结论。实际从报告打开模型工作台，固定 revision、照片3及 control 对象上下文均保留。
- 组件报告 `20962b70-3d10-4180-9d52-257deeb20ec1` 的85条记录中，CAD实际可投影81条；其余4条出现在缺失清单，control `b8a871…` 可选。根因是 `planShapes` 漏掉已确认的观测几何；修正采用相同坐标系及对象自身 TRS 投影，没有将缺失几何补成虚构矩形。该报告没有原始 CAD 资产，不引用其他项目的 CAD。
- 原完整报告 `2d77667b-5369-4d66-b5c4-d426e9833e18` 的原始 CAD 容器为 `workcell-original-cad`：来源 `user-bor1-02`，33条区域/对象记录、6条有明确当前实体关联。图内区域坐标为1600×1240像素，保留历史来源坐标与估计尺度；它不是当前模型平面，也不是已验证的现场实测图纸。另27条未关联记录仍未解决，不记为联动修复完成，不强行叠入当前三维坐标。
- CUA在照片1以 Enter 选择 floor：保留表面轮廓，不再展示设备式 XYZ；control仍可选。390×844复测中 `scrollWidth=390`、一个canvas，中英文切换保留选择，临时视口已复位；未记录控制台 warn/error。这些结果不代表真实手机驱动验证。
- 地面角色以新版本保存：revision `45f843bb-066c-4dd5-9cf0-a460455718b6`，publication `f55f9704-0999-460e-863a-4c58cca86fb9`，包含71个实体、67条观察、179项场景资产。未覆盖既有 revision/publication。对应本地 Blender job `44db2d21-3ff1-444b-a7eb-a657d35dac4b` 重开验证通过：48个对象、3台相机；22个对象未放置，整体状态仍为 `incomplete`。4项导出资产通过 SHA 校验，本轮新增模型调用为0。
- 根任务已刷新最终构建 `app-zn2mw1f4.js` 并实查新发布 `f55f9704-0999-460e-863a-4c58cca86fb9`：从对象表点击主地面 `ed46d9e5…` 后，3D表面仍高亮，原图与3D无设备轴/体积框；属性仅有观测宽深，没有模型位置、高度和设备倾角。CAD/交互平面显示68/68，来源投影轮廓单独标示，未按实体是否存在其他坐标系模型来误着色。下载区实际显示固定45f843bb版本的四个导出按钮与完整原始CAD33/6说明。控制台warn/error为空；浏览器下载落盘文件仍未验证，不将文件哈希与重开验证替代该检查。

本轮定向检查在 `web/` 执行：`node tests/report-context-check.mjs`、`node tests/report-review-check.mjs`、`PANOPTES_TEST_PUBLICATION_URL=http://127.0.0.1:8792/api/publications/2d77667b-5369-4d66-b5c4-d426e9833e18 node tests/report-evidence-check.mjs`、`npx tsc --noEmit`，均通过。检查覆盖固定版本入口和选择上下文、历史判定隔离、参考面不显示设备高度/倾角、原始规则/证据可展开，以及原 CAD 计数按来源对象去重并排除无效实体关联。

最终执行：`pytest tests/test_platform_import.py tests/test_platform_report_evidence.py tests/test_platform_geometry_import.py -q` 为10 passed、1 skipped；跳过项为未设置集成测试数据库。生产数据通过同一导入/任务/发布合同另行生成上述新版本，未运行付费推理。`observed-plan.mjs` 对两份原快照验证68/68与81/85；新发布的report-evidence、report-context、report-review、renderer、report-scene、plan-selection检查及TypeScript/Vite构建通过。

模型工作台开源参考已核对官方来源：[Three.js Editor](https://threejs.org/editor/)提供场景导入、对象编辑和GLB等导出；[PlayCanvas Editor](https://github.com/playcanvas/editor)与[PCUI](https://github.com/playcanvas/pcui)可参考对象树/属性面板。当前PlayCanvas Editor README的本地前端流程仍连接托管Editor服务，不将其称为完整独立自托管后端。本轮没有安装或整合上述编辑器；继续复用已有渲染器与实体/版本合同。


## 三栏联动报告与共享几何修正（最新）

2026-09-12，同一 publication `f55f9704-0999-460e-863a-4c58cca86fb9` / revision `45f843bb-066c-4dd5-9cf0-a460455718b6`。本轮更新渲染与通用分析代码，未覆盖报告数据或另建测试内容充当真实工位。此节取代前文的页面布局描述。

- 左侧常驻全部68条对象记录，可搜索名称与短ID；中间原图、3D、CAD、交互平面支持四视图、单视图和独立全屏；右侧为所选对象的观测/模型尺寸、来源与精确判定。理解、历史EHS、原始CAD、Blender和任务历史仍在同一报告下方。
- 根任务使用 `mcp__cua_repl` 在1440×1000实际构建中：左栏搜索并点击照片1 control `33e5c934`，右栏显示该按钮观测范围；放大CAD后用其可访问按钮选围栏 `ce9516a5`，两张平面图和右栏选择一致；Esc回四视图。全屏3D中直接用鼠标点击红色机器人，实体切为 `d5780205`。单视图仍保留两侧栏，主canvas数量为1。
- 切换观测层保留实体选择。共享 `entityGeometryForLayer` 让照片投影、3D选择框和两张平面图消费相同图层、坐标系与来源；模型层才可优先模型。失效表示和无定位依据的生成资产不制造可定位范围。旋转模型的选择框保留其8个原始角点。
- 最新真实快照的模型层68个范围=9模型+46观测+13观测测量；观测层68=46观测+22观测测量。195组3D选择角点与共享几何相同，3,814个照片角点的独立K/c2w软件计算最大浮点残差0.001254px。该值仅说明投影实现一致，不是重建或现场测量精度。原CAD仍是33条历史区域/6条明确关联，不跨坐标系强行合并。
- 从照片3选择floor `f138f2cd`：两个平面图同步选择；原图轴文字与3D轴文字均为空；右栏仅参考面范围，没有设备高度、位置、倾角。
- 在390×844切换“对象→选择照片3 control `45ff37f5`→详情→英文”：实体、照片与观察ID保留，页面scrollWidth=390，canvas数量1。是浏览器窄屏检查，不声称真实手机GPU测试。
- 新分析/resegment共用原图mask保存路径，写精确pixel-edge轮廓、孔洞及单像素；保留原始检测框的证据身份。空mask与复杂度限制显式记录，完整PNG和对象不消失。前端按声明的像素约定处理轮廓，检测框用虚线显示。新capture背景使用sourceContext，原生尺寸使用数组；稀疏有效观测bounds即使不能成面也能选中其范围。
- 右栏只消费当前评估或发布快照，不增加API请求。当前报告尚无本版本评估，9条历史检查缺少可验证对象关联，因此不将其中整体FAIL归给选中按钮。Agent仍须通过已有编辑/副本合同提交。

验证：TypeScript/Vite构建、前端interaction-check，以及report-context、report-review、report-object-findings、report-evidence、report-scene、report-interactions、plan-selection、observed-plan、renderer各检查通过。`pytest tests/test_platform_reconstruction.py tests/test_platform_spatial.py -q` 为34 passed。生产快照保持不可变；新增模型调用为0。新照片真实模型质量、许可及预算发布门槛仍未因此获得通过声明。

## 主窗口显示完整 CAD 图纸

2026-09-13，继续使用同一 `f55f9704-0999-460e-863a-4c58cca86fb9` 发布快照。

- 主 CAD 窗口复用已保存的完整原图 `30accb8b-b3b0-4b4f-8456-c41bdc3a51c7`（1600×1240），保留来源 `user-bor1-02` 的编号、标注、图例和全部33条CAD区域记录。6条区域记录明确关联到3个当前实体，未把当前68条记录宣称为原图覆盖数量。
- `OriginalCadEvidence` 同时用于主窗口和来源证据区；支持拖动、缩放、完整图纸和定位所选。自动定位最高3×以保留上下文；普通滚轮继续滚动报告，Ctrl/⌘滚轮才缩放图纸。未关联对象显示说明，不借用当前场景坐标强行落点。
- 当前交互平面仍使用同源当前几何。没有完整来源CAD资产的Capture仍显示当前投影；本次没有新增完整CAD生成阶段，也没有改写场景、Publication或资产内容。来源图纸是原始PNG，缩放不会增加其分辨率。
- CUA实际检查：主窗口已显示原图而非简化多边形；键盘选择区域13联动围栏 `ce9516a5`；全屏中鼠标点击防撞柱区域21联动 `27d00998` 与右侧尺寸。放大后拖动改变viewBox且不改变所选对象；完整图纸按钮恢复 `0 0 1600 1240`；定位所选宽度为533.333px，符合3×上限。全屏三栏和窄屏CAD标签均能显示图纸，最终控制台warn/error为空。
- `report-evidence-check`（含真实发布快照只读检查）、`report-scene`、`report-context` 与 TypeScript/Vite构建通过；未执行模型推理。


## 当前场景 CAD 的可读性修正（取代上一节主窗口来源图方案）

2026-09-13，仍为 publication `f55f9704-0999-460e-863a-4c58cca86fb9` / revision `45f843bb-066c-4dd5-9cf0-a460455718b6`，不修改快照和资产。

- 原因：历史 PNG 的33条区域仅6条有明确关联，共对应3个当前实体；其长标签本身互相覆盖。将它作为主 CAD 导致大部分对象出现“尚未关联”的提示。现在主 CAD 使用当前实体的共享 `planShapes` 几何，历史原图完整保留在“图纸来源”与下载区，不伪造历史对应关系。
- 报告与模型工作台复用 `CadView`。当前68条记录全部绘制，保留55个保存轮廓和13个观测范围，共1576个轮廓顶点；没有把轮廓改成矩形、删除远端灯或重跑推理。编号按当前实体顺序对应左侧清单，搜索不重排编号；数字搜索可直接定位记录。
- 矢量图按当前地面坐标绘制网格和真实比例尺；本场景未标定，所有投影尺寸显式使用原生单位。所选轮廓高亮、周边线条淡化，地面保持参考层；“定位所选”隐藏拥挤的其他编号并提供宽深标注，“全图”恢复全部轮廓和可分开的短编号。未显示的密集编号不意味着几何被删除。
- 根任务通过 `mcp__cua_repl` 实际操作：从左栏选择control `33e5c934`，CAD显示#30及投影宽0.09071、深0.05638；全屏CAD中实际鼠标点击#06，选择切至料车 `a719e41c`，相应照片3、交互平面与右侧数据同步。随后拖动改变图纸位置但不改变实体。定位按钮显示单一高亮轮廓和双向尺寸，周边保留为淡线。
- 1728×1080浏览器全屏三栏和478px宽面板均实际查看。中英文切换保留对象及图纸坐标；窄屏页面 `scrollWidth=clientWidth=478`，只有一个WebGL canvas。未进行真实手机GPU测试。
- 独立代码复核发现“边缘按下、捕获前离开、在外部松开、重新进入会继续平移”的问题，已修复并执行实际事件处理器回归。重叠命中仍使用共享真实多边形，拖动后不触发选择。

检查：`web/checks/cad-view.mjs`（合成与本报告真实快照）、`report-scene.mjs`、`plan-selection.mjs`、`observed-plan.mjs`、`web/tests/report-context-check.mjs`、`report-evidence-check.mjs`、`npm --prefix web run check`以及TypeScript/Vite构建通过。该图展示当前对象范围/保存轮廓的地面投影；不是新生成的实体工程图，也没有提升模型姿态或现场测量精度。未发生新模型调用。


## Public feedback deployment — 2026-09-13

- Public entry: https://admin-wekruit.github.io/panoptes-workcell-report/app.html#/reports/f55f9704-0999-460e-863a-4c58cca86fb9
- Frozen revision: `45f843bb-066c-4dd5-9cf0-a460455718b6`; 68 object records, 3 photos. Uses the same report renderer and current-scene CAD as the local platform.
- Public API audit fetched all 197 saved metadata responses and all 183 assets (271,985,343 bytes), matching the frozen export and every SHA-256. CORS, HEAD, byte ranges, unknown IDs and write rejection passed. Machine-readable output is in ignored `.platform/public-deployment-check.json`.
- Actual in-app browser on the public website: report loads; fullscreen displays object list, four linked panes and inspector. Selecting control button #30 updates photo 1, CAD #30, plan and observed 3D axes; CAD focus magnifies its saved polygon and dimensions. Selecting cart #06 from CAD updates the photo 3 URL and linked selection. Chinese/English switching preserves the selected object and CAD view. No captured browser console errors.
- 478 px panel uses view tabs; fullscreen at 1728x1080 shows all four panes. Scene assets actually rendered, rather than merely passing metadata checks.
- Export buttons were clicked in browser; the in-app browser download-event wait timed out, so completion through its download UI is not asserted. The Blender and GLB files were successfully downloaded and hash-verified through their actual public HTTP endpoints in the independent asset audit.
- This deployment is a view-only feedback publication. Mutation/Agent/model-workbench routes are not exposed here. The local full platform is unchanged; model quality/release gates remain as documented.
