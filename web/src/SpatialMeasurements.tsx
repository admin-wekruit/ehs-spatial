import { useEffect, useRef, useState } from "react";
import { request } from "./api";
import { activeModel, jsonObject } from "./core";
import { useI18n } from "./i18n";
import type { Revision } from "./types";
import "./spatial-measurements.css";
import { threePointAngle, verticalEdgeAngle, type SurfacePick } from "./viewer/native-math";

export type MeasureRegion = { coordinateFrameId: string; nativeToPlane: number[][]; points: number[][] };
export type SceneMeasurement = {
  revisionId: string; kind: string; coordinateFrameId: string; source: string;
  value: number; unit: "deg" | "native" | "native2"; method: string;
  references: { entityId: string; representationId: string; assetId: string | null; assetSha256: string | null; placementState: string | null; qualityStatus: string | null }[];
  lines: { points: number[][]; color: string }[]; labelPoint: number[];
  quality: { surfaceFits?: { areaFraction: number; rmsResidualNative: number }[]; deviationFromVerticalDeg?: number; groundReference?: { normal?: number[] | null }; regionFraction?: number };
};
export function SpatialMeasurements({ revision, selectedId, region, drawing, onDraw, onResult, points, pickingPoints, onPickPoints }: {
  points: SurfacePick[]; pickingPoints: boolean; onPickPoints: (start:boolean, count?:2|3) => void;
  revision: Revision; selectedId: string; region: MeasureRegion | null; drawing: boolean;
  onDraw: () => void; onResult: (result: SceneMeasurement | null) => void;
}) {
  const { language } = useI18n(), zh = language === "zh", text = (cn: string, en: string) => zh ? cn : en;
  const [kind, setKind] = useState("edge_vertical"), [target, setTarget] = useState(""), [busy, setBusy] = useState(false),
    [result, setResult] = useState<SceneMeasurement | null>(null), [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  const pointCount = kind === "edge_vertical" ? 2 : 3;
  const selected = revision.document.entities.find(e => e.id === selectedId), model = selected && activeModel(selected);
  const objects = revision.document.entities.filter(e => e.id !== selectedId && e.visible !== false && !e.sourceContext && activeModel(e)?.coordinateFrameId === model?.coordinateFrameId && activeModel(e)?.sourceValidity !== "stale" && activeModel(e));
  useEffect(() => {
    pending.current?.abort(); setBusy(false); setResult(null); setError(""); onResult(null);
    return () => pending.current?.abort();
  }, [revision.id, selectedId, kind, target, region, drawing, points, onResult]);
  async function calculate() {
    pending.current?.abort(); const controller = new AbortController(); pending.current = controller;
    setBusy(true); setError(""); setResult(null); onResult(null);
    const query = new URLSearchParams({ kind, entityA: selectedId });
    if (kind === "occupancy") query.set("region", JSON.stringify(region)); else if (kind !== "inclination") query.set("entityB", target);
    try {
      if(kind === "edge_angle" || kind === "edge_vertical") {
        if(points.length!==pointCount||points.some(p=>p.coordinateFrameId!==points[0].coordinateFrameId))throw Error("measurement_points_invalid");
        const ground=revision.document.coordinateFrames.find(f=>f.id===points[0].coordinateFrameId)?.ground;
        const vertical=kind === "edge_vertical" ? verticalEdgeAngle(points.map(p=>p.point),ground?.normal || []) : null;
        const geometry=vertical || threePointAngle(points.map(p=>p.point));
        const references=[...new Map(points.map(p=>[p.representationId,p])).values()].map(p=>{
          const entity=revision.document.entities.find(e=>e.id===p.entityId),rep=entity&&activeModel(entity),asset=revision.document.assets.find(a=>a.id===rep?.assetId);
          if(!rep||rep.id!==p.representationId||rep.sourceValidity==="stale")throw Error("measurement_model_missing");
          return {entityId:p.entityId,representationId:rep.id,assetId:rep.assetId||null,assetSha256:asset?.sha256||null,placementState:rep.placementState||null,qualityStatus:typeof jsonObject(rep.qualityEvidence)?.status === "string" ? String(jsonObject(rep.qualityEvidence)?.status) : null};
        });
        const value:SceneMeasurement={revisionId:revision.id,kind,coordinateFrameId:points[0].coordinateFrameId,source:"manual_model_points",value:geometry.value,unit:"deg",method:kind === "edge_vertical" ? "model-edge-to-ground-normal-v1" : "three-model-surface-points-v1",references,quality:kind === "edge_vertical" ? {groundReference:ground || undefined} : {},labelPoint:geometry.labelPoint,lines:[{points:points.map(p=>p.point),color:"#e36b23"},...(vertical ? [{points:[points[0].point,vertical.verticalEnd],color:"#f04a3a"}] : []),{points:geometry.arc,color:kind === "edge_vertical" ? "#86e342" : "#edbe38"}]};
        setResult(value);onResult(value);return;
      }
      const value = await request<SceneMeasurement>(`/api/revisions/${revision.id}/measurements?${query}`, { signal: controller.signal });
      if (!controller.signal.aborted) { setResult(value); onResult(value); }
    } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "measurement_failed"); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }
  const errors: Record<string, string> = {
    measurement_points_invalid: text("请按提示在同一场景模型中完成取点。", "Complete the requested points in the same scene frame."),
    measurement_points_coincident: text("端点与顶点重合，请重新选点。", "An endpoint coincides with the vertex. Pick the points again."),
    measurement_no_stable_plane: text("无法确定稳定板面。请选择平板；曲面或混合部件不适用。", "A stable panel plane could not be fitted. Choose a flat panel, not curved or mixed parts."),
    measurement_complexity_limit: text("网格计算超过本次上限，尚未得到结果。", "Mesh computation exceeded this request's limit. No result was produced."),
    measurement_ground_missing: text("缺少有效地面参考，无法测量相对地面的倾角或投影占用。", "A valid ground reference is required for inclination and projected overlap."),
    measurement_region_invalid: text("区域无效，请在当前 CAD 中重新圈定。", "Invalid region. Draw it again in the current CAD view."),
    measurement_frame_mismatch: text("两个模型不在同一坐标系，不能直接测量。", "The models do not share a coordinate frame."),
    measurement_model_missing: text("当前对象没有有效模型。", "This object has no valid current model."),
  };
  return <section className="spatial-measurements" aria-label={text("空间测量", "Spatial measurements")}>
    <h4>{text("空间测量", "Spatial measurements")} <small>{text("模型估计", "Model estimate")}</small></h4>
    <p>{text("可测两边线夹角、板面倾角、物体间距或 CAD 区域占用。", "Measure edge angles, panel inclination, object distances, or a CAD region.")}</p>
    {!model || model.sourceValidity === "stale" ? <p>{text("请先选择有当前模型的对象。", "Select an object with a current model first.")}</p> : <>
      <label>{text("测量类型", "Measurement")}<select value={kind} onChange={e => { if (drawing) onDraw(); onPickPoints(false); setKind(e.target.value); }}>
        <option value="edge_vertical">{text("斜边与竖直线夹角 · 两点", "Edge angle to vertical · two points")}</option>
        <option value="edge_angle">{text("两边线夹角 · 三点 · 0–180°", "Edge angle · three points · 0–180°")}</option>
        <option value="inclination">{text("板面相对地面倾角", "Panel inclination to ground")}</option>
        <option value="angle">{text("两板面夹角 · 0–90°", "Panel plane angle · 0–90°")}</option>
        <option value="distance">{text("两表面最短距离", "Minimum surface distance")}</option>
        <option value="occupancy">{text("区域投影占用", "Projected region overlap")}</option>
      </select></label>
      <p><b>{text("对象 A：", "Object A: ")}</b>{selected?.label}</p>
      {kind === "edge_angle" || kind === "edge_vertical" ? <>
        <p>{kind === "edge_vertical" ? text("在同一条斜边上选：① 上端点（角的顶点）→ ② 下端点。系统从①画竖直参考线，绿色弧线显示斜边偏离竖直的角度。", "Pick the same sloping edge: ① upper endpoint (angle vertex) → ② lower endpoint. A vertical reference starts at ①; the green arc measures the edge angle to vertical.") : text("依次选择：① 第一条边上的点 → ② 两边交点（顶点）→ ③ 第二条边上的点。测量三维夹角，第二点决定角的位置。", "Pick ① a point on the first edge → ② the shared vertex → ③ a point on the second edge. The second point is the angle vertex.")}</p>
        <button type="button" onClick={()=>onPickPoints(!pickingPoints,pointCount)}>{text(pickingPoints ? "取消取点" : points.length ? "重新取点" : pointCount === 2 ? "选取斜边两端" : "在 3D 中选择三个点", pickingPoints ? "Cancel picking" : points.length ? "Pick again" : pointCount === 2 ? "Pick edge endpoints" : "Pick three points in 3D")}</button>
        <p>{text(`已选 ${points.length}/${pointCount} 点。点击模型表面，拖动旋转；也可聚焦画布后用方向键微调光标、Enter 取点。`, `${points.length}/${pointCount} points selected. Click the mesh; drag to rotate. Or focus the canvas, use arrow keys to move the cursor and Enter to pick.`)}</p>
      </> : kind === "inclination" ? <p>{text("参考：当前场景地面。水平为 0°，竖直为 90°；无需选择第二个对象。", "Reference: scene ground. Horizontal = 0°, vertical = 90°. No second object is needed.")}</p> : kind !== "occupancy" ? <label>{text("参考对象 B", "Reference object B")}<select value={target} onChange={e => setTarget(e.target.value)}>
        <option value="">{text("选择参考对象", "Choose reference object")}</option>
        {objects.map(e => <option key={e.id} value={e.id}>{e.label || e.id}</option>)}
      </select></label> : <><button type="button" onClick={onDraw}>{text(drawing ? "取消圈定" : region ? "重新圈定 CAD 区域" : "圈定 CAD 区域", drawing ? "Cancel drawing" : region ? "Redraw CAD region" : "Draw CAD region")}</button>
        <p>{text(region ? "已圈定临时区域。此区域不代表已确认的安全区。" : "在 CAD 中点击矩形的两个对角；也可用方向键移动光标，Enter 确认。", region ? "Temporary region set. It is not a verified safety zone." : "Click two opposite rectangle corners in CAD, or move the cursor with arrow keys and press Enter.")}</p></>}
      <button type="button" className="measure-calculate" disabled={busy || drawing || pickingPoints || ((kind === "edge_angle" || kind === "edge_vertical") ? points.length!==pointCount : kind === "occupancy" ? !region : kind !== "inclination" && !objects.some(e => e.id === target))} onClick={calculate}>{text(busy ? "正在计算…" : "计算并标注", busy ? "Calculating…" : "Calculate & annotate")}</button>
      {error && <p role="alert">{errors[error] || text("测量失败，请重试。", "Measurement failed. Please retry.")} <small>{error}</small></p>}
      {result && <div className="measurement-result" role="status">
        <output>{Number(result.value.toPrecision(4))} {result.unit === "deg" ? "°" : text(result.unit === "native2" ? "原生单位²" : "原生单位", result.unit === "native2" ? "native units²" : "native units")}</output>
        <p>{kind === "edge_vertical" ? text("这是斜边与竖直参考线的夹角：红线为竖直方向，橙线为斜边，绿色为所测角度。角的顶点是①；完全竖直为 0°，水平为 90°。", "Edge angle to vertical: red is vertical, orange is the sloping edge, green is the measured angle. Vertex is ①. Vertical = 0°, horizontal = 90°.") : kind === "edge_angle" ? text("所选两条边线在第 ② 点的三维夹角。橙色为两条边线，黄色为角度弧；保留钝角。结果取决于选点和模型准确度。", "3D angle at point ② between the selected edge directions. Orange: the two lines. Yellow: angle arc. Obtuse angles are preserved. Accuracy depends on the points and model.") : kind === "inclination" ? text(`与地面成 ${result.value.toFixed(1)}°；偏离竖直 ${(result.quality.deviationFromVerticalDeg ?? 0).toFixed(1)}°。橙色为拟合板面，蓝色为平行地面的参考面。`, `${result.value.toFixed(1)}° to ground; ${(result.quality.deviationFromVerticalDeg ?? 0).toFixed(1)}° from vertical. Orange: fitted panel. Blue: a reference plane parallel to ground.`) : kind === "angle" ? text("两拟合板面的较小夹角，不是铰链内角或安装倾角。橙色为 A，蓝色为 B。", "The smaller angle between fitted planes, not a hinge interior or installation angle. A is orange; B is blue.") : kind === "distance" ? text("实际网格表面的最短连线。0 表示表面接触或相交。", "Shortest line between mesh surfaces. Zero means contact or intersection.") : text(`占所圈区域 ${((result.quality.regionFraction || 0) * 100).toFixed(1)}%。这是地面投影重叠，不是实体碰撞。`, `${((result.quality.regionFraction || 0) * 100).toFixed(1)}% of the drawn region. This is projected overlap, not solid collision.`)}</p>
        <p>{text("基于当前模型位置与形状，尚未验证为现场实测；不能据此直接判定合规。距离与面积暂以原生单位显示，未换算为现场单位。", "Based on current model geometry and placement, not verified field measurements or a compliance verdict. Distances and areas are in native model units, not converted to field units.")}</p>
        <details><summary>{text("测量依据", "Measurement evidence")}</summary>
          <p>{text("版本", "Revision")}: {result.revisionId.slice(0, 8)}</p>
          {result.quality.groundReference && <p>{text("地面参考坐标系", "Ground reference frame")}: {result.coordinateFrameId.slice(0, 8)}</p>}
          {result.references.map((ref, i) => <p key={ref.entityId}>{String.fromCharCode(65+i)} · {revision.document.entities.find(e => e.id === ref.entityId)?.label}<br />{text("模型", "Model")}: {ref.representationId.slice(0, 8)} · {ref.placementState === "confirmed" ? text("位置已确认", "Placement confirmed") : text("位置未确认", "Placement unconfirmed")}{ref.qualityStatus === "rejected" && <> · {text("模型质量检查未通过", "Model quality check failed")}</>}{result.quality.surfaceFits?.[i] && <><br />{text("拟合面覆盖", "Fitted surface coverage")}: {(result.quality.surfaceFits[i].areaFraction * 100).toFixed(1)}%</>}</p>)}
        </details>
      </div>}
    </>}
  </section>;
}
