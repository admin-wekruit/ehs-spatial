import { useEffect, useMemo, useRef, useState } from "react";
import { request } from "./api";
import { activeModel, jsonObject } from "./core";
import { useI18n } from "./i18n";
import type { Revision } from "./types";
import "./spatial-measurements.css";
import { groundMeasurementLabel, measureGroundPoints, threePointAngle, verticalEdgeAngle, type GroundPointMeasurement, type GroundMeasurementKind, type MeasurementScale, type SurfacePick } from "./viewer/native-math";

export type MeasureRegion = { coordinateFrameId: string; nativeToPlane: number[][]; points: number[][] };
export type SceneMeasurement = {
  revisionId: string; kind: string; coordinateFrameId: string; source: string;
  value: number; unit: "deg" | "native" | "native2"; method: string;
  displayLabel?: string;
  /** Exact scene document the measured models belong to. */
  documentSha256?: string;
  caliper?: GroundPointMeasurement;
  measurementScale?: MeasurementScale;
  surfacePicks?: SurfacePick[];
  references: { entityId: string; representationId: string; assetId: string | null; assetSha256: string | null; placementState: string | null; qualityStatus: string | null }[];
  lines: { points: number[][]; color: string }[]; labelPoint: number[];
  /** Extra labels drawn like the main one (e.g. length / width / height / clearance of a selected model). */
  labels?: { point: number[]; text: string }[];
  quality: { areaNative2?: number; angularErrorDeg?: number | null; angularSpreadDeg?: number; rmsResidualNative?: number; classification?: string; surfaceId?: string; surfaceFits?: { areaFraction: number; rmsResidualNative: number }[]; deviationFromVerticalDeg?: number; groundReference?: { normal?: number[] | null }; regionFraction?: number };
};
export type BendAnalysis = { revisionId: string; algorithm: string; items: BendOutcome[] };
export type BendOutcome = { entityId: string; inputSha256: string; status: "measured" | "unsupported" | "skipped" | "failed" | "not_processed"; reason?: string; result?: SceneMeasurement };
export type InclinationSurface = { surfaceId:string; inclinationDeg:number; deviationFromVerticalDeg:number; angularSpreadDeg:number; areaNative2:number; classification:string; result:SceneMeasurement };
export type InclinationOutcome = { entityId:string; status:string; reason?:string; surfaces:InclinationSurface[] };
export type InclinationAnalysis = { revisionId:string; algorithm:string; items:InclinationOutcome[] };
export function SpatialMeasurements({ revision, selectedId, savedBend, savedSurface, inclinationOutcome, onClearSurface, region, drawing, onDraw, onResult, onLocalResult, points, pickingPoints, onPickPoints, analysisAvailable = true, measurementScale, geometryKey }: {
  savedBend?: BendOutcome; savedSurface?: InclinationSurface; inclinationOutcome?: InclinationOutcome; onClearSurface?: () => void;
  points: SurfacePick[]; pickingPoints: boolean; onPickPoints: (start:boolean, count?:1|2|3) => void;
  revision: Revision; selectedId: string; region: MeasureRegion | null; drawing: boolean;
  onDraw: () => void; onResult: (result: SceneMeasurement | null) => void;
  onLocalResult: (result: SceneMeasurement | null) => void;
  analysisAvailable?: boolean; measurementScale?: MeasurementScale; geometryKey: string;
}) {
  const { language } = useI18n(), zh = language === "zh", text = (cn: string, en: string) => zh ? cn : en;
  const [manualKind, setKind] = useState(analysisAvailable ? "bend" : "point_ground"), [target, setTarget] = useState(""), [busy, setBusy] = useState(false),
    [rawResult, setResult] = useState<SceneMeasurement | null>(null), [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  const kind = savedSurface ? "inclination" : manualKind;
  const caliperMode = ["point_ground", "point_distance", "region_ground"].includes(kind), pointMode = caliperMode || kind === "edge_angle" || kind === "edge_vertical";
  const pointCount = kind === "point_ground" ? 1 : kind === "edge_vertical" || kind === "point_distance" ? 2 : 3;
  const selected = revision.document.entities.find(e => e.id === selectedId), model = selected && activeModel(selected);
  const frameScale = revision.document.coordinateFrames.find(f => f.id === (points[0]?.coordinateFrameId ?? model?.coordinateFrameId))?.scale;
  const scale = measurementScale ?? { nativeToMeters: frameScale?.nativeToMeters ?? null, status: frameScale?.status || "unknown", source: "coordinate_frame.scale" };
  const result = useMemo(() => rawResult?.caliper ? { ...rawResult, measurementScale: scale, displayLabel: groundMeasurementLabel(rawResult.caliper, scale) } : rawResult, [rawResult, scale.nativeToMeters, scale.status, scale.source]);
  const objects = revision.document.entities.filter(e => e.id !== selectedId && e.visible !== false && !e.sourceContext && activeModel(e)?.coordinateFrameId === model?.coordinateFrameId && activeModel(e)?.sourceValidity !== "stale" && activeModel(e));
  useEffect(() => {
    const saved = kind === "inclination" && savedSurface ? savedSurface.result : kind === "bend" && savedBend?.status === "measured" ? savedBend.result || null : null;
    pending.current?.abort(); setBusy(false); setResult(saved); setError(""); if (!pointMode) onResult(saved);
    return () => pending.current?.abort();
  }, [revision.id, selectedId, geometryKey, kind, target, region, drawing, points, onResult, savedBend, savedSurface]);
  useEffect(() => { if (pointMode) onLocalResult(result); }, [result, pointMode, onLocalResult]);
  async function calculate() {
    pending.current?.abort(); const controller = new AbortController(); pending.current = controller;
    setBusy(true); setError(""); setResult(null); if (!pointMode) onResult(null);
    const query = new URLSearchParams({ kind, entityA: selectedId });
    if (kind === "occupancy") query.set("region", JSON.stringify(region)); else if (kind === "angle" || kind === "distance") query.set("entityB", target);
    try {
      if(pointMode) {
        if(points.length!==pointCount||points.some(p=>p.coordinateFrameId!==points[0].coordinateFrameId))throw Error("measurement_points_invalid");
        const ground=revision.document.coordinateFrames.find(f=>f.id===points[0].coordinateFrameId)?.ground;
        const references=[...new Map(points.map(p=>[p.representationId,p])).values()].map(p=>{
          const entity=revision.document.entities.find(e=>e.id===p.entityId),rep=entity&&activeModel(entity),asset=revision.document.assets.find(a=>a.id===rep?.assetId);
          if(!rep||rep.id!==p.representationId||rep.sourceValidity==="stale"||entity?.visible===false)throw Error("measurement_model_missing");
          if(rep.coordinateFrameId!==p.coordinateFrameId||(entity?.currentModelTransform||rep.transform).coordinateFrameId!==p.coordinateFrameId)throw Error("measurement_frame_mismatch");
          return {entityId:p.entityId,representationId:rep.id,assetId:rep.assetId||null,assetSha256:asset?.sha256||null,placementState:rep.placementState||null,qualityStatus:typeof jsonObject(rep.qualityEvidence)?.status === "string" ? String(jsonObject(rep.qualityEvidence)?.status) : null};
        });
        if (caliperMode) {
          const caliper = measureGroundPoints(kind as GroundMeasurementKind, points.map(p => p.point), ground);
          const lines = points.map((p, i) => ({ points: [p.point, caliper.feetNative[i]], color: "#f04a3a" }));
          if (pointCount > 1) lines.push({ points: [...caliper.pointsNative, ...(pointCount === 3 ? [points[0].point] : [])], color: "#e36b23" });
          const value: SceneMeasurement = { revisionId: revision.id, documentSha256: revision.documentSha256, kind, coordinateFrameId: points[0].coordinateFrameId, source: "manual_model_points", value: caliper.distanceNative ?? caliper.minHeightNative, unit: "native", method: "model-surface-ground-caliper-v1", references, quality: { groundReference: { ...ground, ...caliper.ground }, ...(kind === "region_ground" ? { classification: "sampled_triangle" } : {}) }, labelPoint: points[0].point, lines, caliper, surfacePicks: points, measurementScale: scale, displayLabel: groundMeasurementLabel(caliper, scale) };
          setResult(value); return;
        }
        const vertical=kind === "edge_vertical" ? verticalEdgeAngle(points.map(p=>p.point),ground?.normal || []) : null;
        const geometry=vertical || threePointAngle(points.map(p=>p.point));
        const value:SceneMeasurement={revisionId:revision.id,kind,coordinateFrameId:points[0].coordinateFrameId,source:"manual_model_points",value:geometry.value,unit:"deg",method:kind === "edge_vertical" ? "model-edge-to-ground-normal-v1" : "three-model-surface-points-v1",references,quality:kind === "edge_vertical" ? {groundReference:ground || undefined} : {},labelPoint:geometry.labelPoint,lines:[{points:points.map(p=>p.point),color:"#e36b23"},...(vertical ? [{points:[points[0].point,vertical.verticalEnd],color:"#f04a3a"}] : []),{points:geometry.arc,color:kind === "edge_vertical" ? "#86e342" : "#edbe38"}]};
        setResult(value);return;
      }
      if (!analysisAvailable) return;
      const value = await request<SceneMeasurement>(`/api/revisions/${revision.id}/measurements?${query}`, { signal: controller.signal });
      if (!controller.signal.aborted) { setResult(value); onResult(value); }
    } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "measurement_failed"); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }
  const errors: Record<string, string> = {
    measurement_no_stable_local_plane: text("未识别到满足面积和平整度条件的局部面。", "No local patch meets the area and flatness criteria."),
    measurement_model_missing: text("当前对象没有有效独立模型。", "No valid independent model for this object."),
    measurement_no_stable_bend: text("未识别到足够稳定、相接的两个板面，无法给出折弯内角。请核对模型的折弯形状。", "Two stable adjoining sheet faces could not be identified. Check the modeled fold geometry."),
    measurement_points_invalid: text("请按提示在同一场景模型中完成取点。", "Complete the requested points in the same scene frame."),
    measurement_points_coincident: text("端点与顶点重合，请重新选点。", "An endpoint coincides with the vertex. Pick the points again."),
    measurement_points_collinear: text("三点共线，无法确定采样区域的平面，请重新选点。", "The three points are collinear. Pick a triangle to define the sampled region."),
    measurement_scale_invalid: text("模型比例无效，请检查比例来源。", "The model scale is invalid. Check its source."),
    measurement_no_stable_plane: text("无法确定稳定板面。请选择平板；曲面或混合部件不适用。", "A stable panel plane could not be fitted. Choose a flat panel, not curved or mixed parts."),
    measurement_complexity_limit: text("网格计算超过本次上限，尚未得到结果。", "Mesh computation exceeded this request's limit. No result was produced."),
    measurement_ground_missing: text("缺少有效地面法向和偏移，无法进行地面测量。", "A valid ground normal and offset are required for ground measurements."),
    measurement_region_invalid: text("区域无效，请在当前 CAD 中重新圈定。", "Invalid region. Draw it again in the current CAD view."),
    measurement_frame_mismatch: text("两个模型不在同一坐标系，不能直接测量。", "The models do not share a coordinate frame."),
  };
  return <section className="spatial-measurements" aria-label={text("空间测量", "Spatial measurements")}>
    <h4>{text("空间测量", "Spatial measurements")} <small>{text("模型估计", "Model estimate")}</small></h4>
    <p>{text("点击模型表面，测量点到地面、两点距离与高差，或三点采样区域。", "Pick model surfaces to measure ground height, point distance and height difference, or a three-point sampled region.")}</p>
    {!model || model.sourceValidity === "stale" ? <p>{text("请先选择有当前模型的对象。", "Select an object with a current model first.")}</p> : <>
      <label>{text("测量类型", "Measurement")}<select value={kind} onChange={e => { if (drawing) onDraw(); onPickPoints(false); onClearSurface?.(); setKind(e.target.value); }}>
        <option value="point_ground">{text("点到地面 · 一点", "Point to ground · one point")}</option>
        <option value="point_distance">{text("两点距离与高差", "Two-point distance and height difference")}</option>
        <option value="region_ground">{text("采样区域离地范围 · 三点", "Sampled region ground heights · three points")}</option>
        {(analysisAvailable || savedBend?.status === "measured") && <option value="bend">{text("板子自身折弯内角 · 两个板面", "Panel interior bend · two sheet faces")}</option>}
        <option value="edge_vertical">{text("斜边与竖直线夹角 · 两点", "Edge angle to vertical · two points")}</option>
        <option value="edge_angle">{text("两边线夹角 · 三点 · 0–180°", "Edge angle · three points · 0–180°")}</option>
        {(analysisAvailable || savedSurface) && <option value="inclination">{text("板面相对地面倾角", "Panel inclination to ground")}</option>}
        {analysisAvailable && <><option value="angle">{text("两板面夹角 · 0–90°", "Panel plane angle · 0–90°")}</option>
        <option value="distance">{text("两表面最短距离", "Minimum surface distance")}</option>
        <option value="occupancy">{text("区域投影占用", "Projected region overlap")}</option></>}
      </select></label>
      <p>{pointMode ? (points.length ? points.map((point, i) => `${i + 1} · ${revision.document.entities.find(entity => entity.id === point.entityId)?.label || point.entityId}`).join("；") : text("可跨物体选择表面点。", "Surface points may come from different objects.")) : <><b>{text("对象 A：", "Object A: ")}</b>{selected?.label}</>}</p>
      {kind === "bend" && savedBend && <p className="saved-bend-status">{text(
        savedBend.status === "measured" ? "处理流程已计算并保存，打开报告自动读取。" : savedBend.status === "unsupported" ? "处理流程已检查：当前模型未检出稳定折弯。可从上方「已识别折弯」选择其他板件。" : savedBend.status === "not_processed" ? "当前模型尚无有效的已保存分析。" : savedBend.status === "skipped" ? "当前对象没有可用于折弯分析的有效独立模型。" : "处理流程未能完成此对象的计算。",
        savedBend.status === "measured" ? "Calculated and saved during processing; loaded with the report." : savedBend.status === "unsupported" ? "Processed: no stable bend detected in this model. Choose a detected bend above." : savedBend.status === "not_processed" ? "No saved analysis for the current model." : savedBend.status === "skipped" ? "No valid independent model for bend analysis." : "Processing could not complete this object's calculation.")}{savedBend.status === "failed" && <small> {errors[savedBend.reason || ""] || savedBend.reason}</small>}</p>}
      {pointMode ? <>
        <p>{kind === "point_ground" ? text("选择一个表面点。红线沿地面法向连接到地面；正值在法向一侧，负值在地面下方。", "Pick a surface point. The red line reaches the ground along its normal; negative height is below ground.") : kind === "point_distance" ? text("依次选择 A、B。直线距离与有符号高差 B−A 分别显示，高差沿地面法向计算。", "Pick A then B. Straight-line distance and signed B−A height difference are measured separately; height follows the ground normal.") : kind === "region_ground" ? text("在目标区域选择三个不共线的表面点。结果是三点张成三角形的离地范围和倾角，仅代表采样区域，不代表整块实体表面。", "Pick three noncollinear surface points. The height range and inclination describe their sampled triangle, not the whole physical face.") : kind === "edge_vertical" ? text("在同一条斜边上选：① 上端点（角的顶点）→ ② 下端点。系统从①画竖直参考线，绿色弧线显示斜边偏离竖直的角度。", "Pick the same sloping edge: ① upper endpoint (angle vertex) → ② lower endpoint. A vertical reference starts at ①; the green arc measures the edge angle to vertical.") : text("依次选择：① 第一条边上的点 → ② 两边交点（顶点）→ ③ 第二条边上的点。测量三维夹角，第二点决定角的位置。", "Pick ① a point on the first edge → ② the shared vertex → ③ a point on the second edge. The second point is the angle vertex.")}</p>
        <button type="button" onClick={()=>onPickPoints(!pickingPoints,pointCount)}>{text(pickingPoints ? "取消取点" : points.length ? "重新取点" : `在 3D 中选择 ${pointCount} 个点`, pickingPoints ? "Cancel picking" : points.length ? "Pick again" : `Pick ${pointCount} point${pointCount === 1 ? "" : "s"} in 3D`)}</button>
        {(points.length > 0 || result) && <button type="button" onClick={() => { onPickPoints(false); setResult(null); setError(""); }}>{text("清除测量并恢复原标注", "Clear and restore annotations")}</button>}
        <p>{text(`已选 ${points.length}/${pointCount} 点。点击模型表面，拖动旋转；也可聚焦画布后用方向键微调光标、Enter 取点。`, `${points.length}/${pointCount} points selected. Click the mesh; drag to rotate. Or focus the canvas, use arrow keys to move the cursor and Enter to pick.`)}</p>
      </> : kind === "bend" ? <p>{text("识别当前板子自身相接的两个板面，在交线处标出内夹角。摊平为 180°，直角折弯为 90°；不使用地面或其他对象作为参考。", "Fit the two adjoining sheet faces of this panel and annotate their interior angle at the hinge. Flat = 180°, a right-angle fold = 90°. No ground or second object is used.")}</p> : kind === "inclination" ? <p>{text("参考：当前场景地面。水平为 0°，竖直为 90°；无需选择第二个对象。", "Reference: scene ground. Horizontal = 0°, vertical = 90°. No second object is needed.")}</p> : kind !== "occupancy" ? <label>{text("参考对象 B", "Reference object B")}<select value={target} onChange={e => setTarget(e.target.value)}>
        <option value="">{text("选择参考对象", "Choose reference object")}</option>
        {objects.map(e => <option key={e.id} value={e.id}>{e.label || e.id}</option>)}
      </select></label> : <><button type="button" onClick={onDraw}>{text(drawing ? "取消圈定" : region ? "重新圈定 CAD 区域" : "圈定 CAD 区域", drawing ? "Cancel drawing" : region ? "Redraw CAD region" : "Draw CAD region")}</button>
        <p>{text(region ? "已圈定临时区域。此区域不代表已确认的安全区。" : "在 CAD 中点击矩形的两个对角；也可用方向键移动光标，Enter 确认。", region ? "Temporary region set. It is not a verified safety zone." : "Click two opposite rectangle corners in CAD, or move the cursor with arrow keys and press Enter.")}</p></>}
      {kind === "inclination" && <p>{savedSurface ? text(`流程已保存 · 局部面 ${savedSurface.surfaceId} · 面积 ${savedSurface.areaNative2.toPrecision(3)} 原生单位²`, `Saved local surface ${savedSurface.surfaceId} · area ${savedSurface.areaNative2.toPrecision(3)} native²`) : inclinationOutcome?.status === "measured" ? text(`已识别 ${inclinationOutcome.surfaces.length} 个局部面，请从上方“倾斜平面”选择。`, `${inclinationOutcome.surfaces.length} local surfaces saved; select one above.`) : text(`自动分析：${inclinationOutcome?.reason ? errors[inclinationOutcome.reason] || inclinationOutcome.reason : "尚无局部面结果"}`, `Automatic analysis: ${inclinationOutcome?.reason || "No saved local surfaces"}`)}</p>}
      {savedSurface && kind === "inclination" && <p>{savedSurface.result.quality.angularErrorDeg == null ? text("地面方向误差未记录；显示模型倾角估计，偏离竖直尚待确认。", "Ground direction error is unavailable; model inclination is an estimate, vertical classification is unverified.") : text(`工程角度误差估计 ${savedSurface.result.quality.angularErrorDeg.toFixed(1)}°，不代表现场标定精度。`, `Engineering angular error estimate ${savedSurface.result.quality.angularErrorDeg.toFixed(1)}°; not field calibration.`)}</p>}
      {(pointMode || analysisAvailable) && <button type="button" className="measure-calculate" disabled={Boolean(kind === "inclination" && savedSurface) || busy || drawing || pickingPoints || (pointMode ? points.length!==pointCount : kind === "occupancy" ? !region : (kind === "angle" || kind === "distance") && !objects.some(e => e.id === target))} onClick={calculate}>{text(kind === "inclination" && savedSurface ? "已保存的局部面测量" : busy ? "正在计算…" : kind === "bend" && savedBend?.status === "measured" ? "重新计算并标注" : "计算并标注", kind === "inclination" && savedSurface ? "Saved local surface measurement" : busy ? "Calculating…" : kind === "bend" && savedBend?.status === "measured" ? "Recalculate & annotate" : "Calculate & annotate")}</button>}
      {error && <p role="alert">{errors[error] || text("测量失败，请重试。", "Measurement failed. Please retry.")} <small>{error}</small></p>}
      {result && <div className="measurement-result" role="status">
        <output className={result.caliper ? "caliper-output" : undefined}>{result.displayLabel || <>{Number(result.value.toPrecision(4))} {result.unit === "deg" ? "°" : text(result.unit === "native2" ? "原生单位²" : "原生单位", result.unit === "native2" ? "native units²" : "native units")}</>}</output>
        <p>{caliperMode ? text(kind === "region_ground" ? "h 为采样三角形内的最小/最大有符号离地高度；角度为采样平面与地面的较小夹角。" : kind === "point_distance" ? "d 为两点直线距离；Δh 为 B−A 的有符号离地高差。" : "h 为沿地面法向的有符号高度。", kind === "region_ground" ? "h gives the sampled triangle’s minimum and maximum signed ground heights; the angle is its inclination to ground." : kind === "point_distance" ? "d is straight-line distance; Δh is the signed B−A ground height difference." : "h is signed height along the ground normal.") : kind === "bend" ? text(`折弯内角 ${result.value.toFixed(1)}°（相对摊平折转 ${(180-result.value).toFixed(1)}°）。橙色、蓝色为同一块板的两个拟合板面；紫色为交线，绿色弧线为内夹角。`, `Interior bend ${result.value.toFixed(1)}° (${(180-result.value).toFixed(1)}° from flat). Orange and blue identify the two fitted sheet faces; purple marks their hinge and green the interior angle.`) : kind === "edge_vertical" ? text("这是斜边与竖直参考线的夹角：红线为竖直方向，橙线为斜边，绿色为所测角度。角的顶点是①；完全竖直为 0°，水平为 90°。", "Edge angle to vertical: red is vertical, orange is the sloping edge, green is the measured angle. Vertex is ①. Vertical = 0°, horizontal = 90°.") : kind === "edge_angle" ? text("所选两条边线在第 ② 点的三维夹角。橙色为两条边线，黄色为角度弧；保留钝角。结果取决于选点和模型准确度。", "3D angle at point ② between the selected edge directions. Orange: the two lines. Yellow: angle arc. Obtuse angles are preserved. Accuracy depends on the points and model.") : kind === "inclination" ? text(`与地面成 ${result.value.toFixed(1)}°；偏离竖直 ${(result.quality.deviationFromVerticalDeg ?? 0).toFixed(1)}°。橙色为拟合板面，蓝色为平行地面的参考面。`, `${result.value.toFixed(1)}° to ground; ${(result.quality.deviationFromVerticalDeg ?? 0).toFixed(1)}° from vertical. Orange: fitted panel. Blue: a reference plane parallel to ground.`) : kind === "angle" ? text("两拟合板面的较小夹角，不是铰链内角或安装倾角。橙色为 A，蓝色为 B。", "The smaller angle between fitted planes, not a hinge interior or installation angle. A is orange; B is blue.") : kind === "distance" ? text("实际网格表面的最短连线。0 表示表面接触或相交。", "Shortest line between mesh surfaces. Zero means contact or intersection.") : text(`占所圈区域 ${((result.quality.regionFraction || 0) * 100).toFixed(1)}%。这是地面投影重叠，不是实体碰撞。`, `${((result.quality.regionFraction || 0) * 100).toFixed(1)}% of the drawn region. This is projected overlap, not solid collision.`)}</p>
        <p>{text("基于当前模型位置与形状，尚未验证为现场实测；不能据此直接判定合规。", "Based on current model geometry and placement, not verified field measurements or a compliance verdict.")}</p>
        {result.caliper && <><p>{scale.nativeToMeters == null ? text("比例未知：保留原生模型单位。", "Scale unknown: native model units.") : text(`模型换算：1 原生单位 = ${Number((scale.nativeToMeters * 100).toPrecision(5))} cm。`, `Model conversion: 1 native unit = ${Number((scale.nativeToMeters * 100).toPrecision(5))} cm.`)} {text("比例状态", "Scale status")}: {scale.status} · {scale.source}</p>
          <button type="button" onClick={() => { const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" })); const a = document.createElement("a"); a.href = url; a.download = `model-measurement-${kind}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }}>{text("下载测量 JSON", "Download measurement JSON")}</button></>}
        {!result.caliper && result.unit !== "deg" && <p>{text("距离与面积以原生模型单位显示。", "Distances and areas use native model units.")}</p>}
        <details><summary>{text("测量依据", "Measurement evidence")}</summary>
          <p>{text("版本", "Revision")}: {result.revisionId.slice(0, 8)}</p>
          {result.quality.groundReference && <p>{text("地面参考坐标系", "Ground reference frame")}: {result.coordinateFrameId.slice(0, 8)}</p>}
          {kind === "bend" && result.quality.surfaceFits?.map((fit,i)=><p key={i}>{text(`板面 ${i+1}：覆盖模型表面积 ${(fit.areaFraction*100).toFixed(1)}%，拟合残差 ${fit.rmsResidualNative.toPrecision(3)} 原生单位。`, `Face ${i+1}: ${(fit.areaFraction*100).toFixed(1)}% of mesh area; RMS residual ${fit.rmsResidualNative.toPrecision(3)} native units.`)}</p>)}
          {result.references.map((ref, i) => <p key={ref.entityId}>{String.fromCharCode(65+i)} · {revision.document.entities.find(e => e.id === ref.entityId)?.label}<br />{text("模型", "Model")}: {ref.representationId.slice(0, 8)} · {ref.placementState === "confirmed" ? text("位置已确认", "Placement confirmed") : text("位置未确认", "Placement unconfirmed")}{ref.qualityStatus === "rejected" && <> · {text("模型质量检查未通过", "Model quality check failed")}</>}{result.quality.surfaceFits?.[i] && <><br />{text("拟合面覆盖", "Fitted surface coverage")}: {(result.quality.surfaceFits[i].areaFraction * 100).toFixed(1)}%</>}</p>)}
        </details>
      </div>}
    </>}
  </section>;
}
