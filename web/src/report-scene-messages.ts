export const reportSceneMessages: Record<string, [string, string]> = {
  sceneLinked: ["一个工位，四个相连的视角", "One scene, four connected views"],
  sceneReadOnly: [
    "选择对象以核对来源与空间范围。",
    "Select an object to inspect its evidence and spatial bounds.",
  ],
  scenePhoto: ["原图与对象", "Photograph & objects"],
  scene3D: ["完整场景 3D", "Full scene 3D"],
  sceneCAD: ["CAD 投影", "CAD projection"],
  scenePlan: ["交互平面", "Interactive plan"],
  sceneLayers: ["空间表示", "Representation"],
  sceneModel: ["模型与场景", "Model & context"],
  sceneObserved: ["观测表面", "Observed surfaces"],
  sceneNoPoints: ["点云（未附点云）", "Point cloud (not attached)"],
  scenePoints: ["点云", "Point cloud"],
  sceneAllBounds: ["所有对象范围", "All object bounds"],
  sceneFullscreen: ["全屏查看", "View fullscreen"],
  sceneCloseFullscreen: ["返回四视图", "Return to four views"],
  sceneFullscreenUnavailable: [
    "浏览器不支持全屏；已展开此视图。",
    "Fullscreen is unavailable; this view has been expanded.",
  ],
  scenePhotoNumber: ["照片", "Photo"],
  sceneObjects: ["场景对象", "Scene objects"],
  sceneSearch: ["搜索物体名称或编号", "Search object name or ID"],
  sceneNoMatches: ["没有匹配的物体。", "No matching objects."],
  sceneNoObjects: [
    "照片已保留，尚无对象分析结果。可圈选照片区域补充对象。",
    "Photographs are preserved. No object analysis is available yet; select a region to add an object.",
  ],
  sceneEvidenceCount: ["条照片证据", "photo observations"],
  sceneNoGeometry: ["暂无空间表示", "No spatial representation"],
  sceneNoGeometrySelection: [
    "该对象可在原图中选中；当前缺少空间几何。",
    "This object can be selected in the photograph; spatial geometry is not available.",
  ],
  sceneNoRepresentation: [
    "此版本尚无所选类型的空间资产。",
    "This revision has no spatial assets of the selected kind.",
  ],
  sceneCandidateNotice: [
    "含待确认的候选放置；显示位置不代表现场已验证。",
    "Includes candidate placements awaiting confirmation; displayed poses are not verified site evidence.",
  ],
  sceneCandidate: ["放置待确认", "Placement unconfirmed"],
  sceneNoPhotoAxes: [
    "缺少同坐标系的相机或对象几何，原图中仅显示检测范围。",
    "A matching camera or object geometry is unavailable; the photograph shows detection bounds only.",
  ],
  sceneDrawActive: [
    "在原图拖出区域，交给 Agent 补充。",
    "Drag a region in the photograph to give it to the Agent.",
  ],
  sceneSourceAxis: [
    "XYZ 为当前表示的局部轴。",
    "XYZ shows the local axes of the current representation.",
  ],
  sceneNativeAxis: [
    "范围来自观测几何；XYZ 为场景原生坐标轴，非物体结构轴。",
    "Bounds come from observed geometry; XYZ follows the native scene frame, not the object's structural axes.",
  ],
  sceneSelected: ["当前选择", "Selected"],
};
