import { useEffect, useRef, useState } from "react";
import { request } from "./api";
import { activeModel } from "./core";
import { useI18n } from "./i18n";
import type { Revision } from "./types";
import "./spatial-measurements.css";

export type MeasureRegion = { coordinateFrameId: string; nativeToPlane: number[][]; points: number[][] };
export type SceneMeasurement = {
  revisionId: string; kind: string; coordinateFrameId: string; source: string;
  value: number; unit: "deg" | "native" | "native2"; method: string;
  references: { entityId: string; representationId: string; assetId: string | null; assetSha256: string | null; placementState: string | null; qualityStatus: string | null }[];
  lines: { points: number[][]; color: string }[]; labelPoint: number[];
  quality: { surfaceFits?: { areaFraction: number; rmsResidualNative: number }[]; deviationFromVerticalDeg?: number; groundReference?: { normal?: number[] }; regionFraction?: number };
};
export function SpatialMeasurements({ revision, selectedId, region, drawing, onDraw, onResult }: {
  revision: Revision; selectedId: string; region: MeasureRegion | null; drawing: boolean;
  onDraw: () => void; onResult: (result: SceneMeasurement | null) => void;
}) {
  const { language } = useI18n(), zh = language === "zh", text = (cn: string, en: string) => zh ? cn : en;
  const [kind, setKind] = useState("inclination"), [target, setTarget] = useState(""), [busy, setBusy] = useState(false),
    [result, setResult] = useState<SceneMeasurement | null>(null), [error, setError] = useState("");
  const pending = useRef<AbortController | null>(null);
  const selected = revision.document.entities.find(e => e.id === selectedId), model = selected && activeModel(selected);
  const objects = revision.document.entities.filter(e => e.id !== selectedId && e.visible !== false && !e.sourceContext && activeModel(e)?.coordinateFrameId === model?.coordinateFrameId && activeModel(e)?.sourceValidity !== "stale" && activeModel(e));
  useEffect(() => {
    pending.current?.abort(); setBusy(false); setResult(null); setError(""); onResult(null);
    return () => pending.current?.abort();
  }, [revision.id, selectedId, kind, target, region, drawing, onResult]);
  async function calculate() {
    pending.current?.abort(); const controller = new AbortController(); pending.current = controller;
    setBusy(true); setError(""); setResult(null); onResult(null);
    const query = new URLSearchParams({ kind, entityA: selectedId });
    if (kind === "occupancy") query.set("region", JSON.stringify(region)); else if (kind !== "inclination") query.set("entityB", target);
    try {
      const value = await request<SceneMeasurement>(`/api/revisions/${revision.id}/measurements?${query}`, { signal: controller.signal });
      if (!controller.signal.aborted) { setResult(value); onResult(value); }
    } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : "measurement_failed"); }
    finally { if (!controller.signal.aborted) setBusy(false); }
  }
  const errors: Record<string, string> = {
    measurement_no_stable_plane: text("无法确定稳定板面。请选择平板；曲面或混合部件不适用。", "A stable panel plane could not be fitted. Choose a flat panel, not curved or mixed parts."),
    measurement_complexity_limit: text("网格计算超过本次上限，尚未得到结果。", "Mesh computation exceeded this request's limit. No result was produced."),
    measurement_ground_missing: text("缺少有效地面参考，无法测量相对地面的倾角或投影占用。", "A valid ground reference is required for inclination and projected overlap."),
    measurement_region_invalid: text("区域无效，请在当前 CAD 中重新圈定。", "Invalid region. Draw it again in the current CAD view."),
    measurement_frame_mismatch: text("两个模型不在同一坐标系，不能直接测量。", "The models do not share a coordinate frame."),
    measurement_model_missing: text("当前对象没有有效模型。", "This object has no valid current model."),
  };
  return <section className="spatial-measurements" aria-label={text("空间测量", "Spatial measurements")}>
    <h4>{text("空间测量", "Spatial measurements")} <small>{text("模型估计", "Model estimate")}</small></h4>
    <p>{text("可测单板倾角、两物体关系，或 CAD 区域占用。", "Measure panel inclination, two-object geometry, or a region drawn in CAD.")}</p>
    {!model || model.sourceValidity === "stale" ? <p>{text("请先选择有当前模型的对象。", "Select an object with a current model first.")}</p> : <>
      <label>{text("测量类型", "Measurement")}<select value={kind} onChange={e => { if (drawing) onDraw(); setKind(e.target.value); }}>
        <option value="inclination">{text("板面相对地面倾角", "Panel inclination to ground")}</option>
        <option value="angle">{text("两板面夹角 · 0–90°", "Panel plane angle · 0–90°")}</option>
        <option value="distance">{text("两表面最短距离", "Minimum surface distance")}</option>
        <option value="occupancy">{text("区域投影占用", "Projected region overlap")}</option>
      </select></label>
      <p><b>{text("对象 A：", "Object A: ")}</b>{selected?.label}</p>
      {kind === "inclination" ? <p>{text("参考：当前场景地面。水平为 0°，竖直为 90°；无需选择第二个对象。", "Reference: scene ground. Horizontal = 0°, vertical = 90°. No second object is needed.")}</p> : kind !== "occupancy" ? <label>{text("参考对象 B", "Reference object B")}<select value={target} onChange={e => setTarget(e.target.value)}>
        <option value="">{text("选择参考对象", "Choose reference object")}</option>
        {objects.map(e => <option key={e.id} value={e.id}>{e.label || e.id}</option>)}
      </select></label> : <><button type="button" onClick={onDraw}>{text(drawing ? "取消圈定" : region ? "重新圈定 CAD 区域" : "圈定 CAD 区域", drawing ? "Cancel drawing" : region ? "Redraw CAD region" : "Draw CAD region")}</button>
        <p>{text(region ? "已圈定临时区域。此区域不代表已确认的安全区。" : "在 CAD 中点击矩形的两个对角；也可用方向键移动光标，Enter 确认。", region ? "Temporary region set. It is not a verified safety zone." : "Click two opposite rectangle corners in CAD, or move the cursor with arrow keys and press Enter.")}</p></>}
      <button type="button" className="measure-calculate" disabled={busy || drawing || (kind === "occupancy" ? !region : kind !== "inclination" && !objects.some(e => e.id === target))} onClick={calculate}>{text(busy ? "正在计算…" : "计算并标注", busy ? "Calculating…" : "Calculate & annotate")}</button>
      {error && <p role="alert">{errors[error] || text("测量失败，请重试。", "Measurement failed. Please retry.")} <small>{error}</small></p>}
      {result && <div className="measurement-result" role="status">
        <output>{Number(result.value.toPrecision(4))} {result.unit === "deg" ? "°" : text(result.unit === "native2" ? "原生单位²" : "原生单位", result.unit === "native2" ? "native units²" : "native units")}</output>
        <p>{kind === "inclination" ? text(`与地面成 ${result.value.toFixed(1)}°；偏离竖直 ${(result.quality.deviationFromVerticalDeg ?? 0).toFixed(1)}°。橙色为拟合板面，蓝色为平行地面的参考面。`, `${result.value.toFixed(1)}° to ground; ${(result.quality.deviationFromVerticalDeg ?? 0).toFixed(1)}° from vertical. Orange: fitted panel. Blue: a reference plane parallel to ground.`) : kind === "angle" ? text("两拟合板面的较小夹角，不是铰链内角或安装倾角。橙色为 A，蓝色为 B。", "The smaller angle between fitted planes, not a hinge interior or installation angle. A is orange; B is blue.") : kind === "distance" ? text("实际网格表面的最短连线。0 表示表面接触或相交。", "Shortest line between mesh surfaces. Zero means contact or intersection.") : text(`占所圈区域 ${((result.quality.regionFraction || 0) * 100).toFixed(1)}%。这是地面投影重叠，不是实体碰撞。`, `${((result.quality.regionFraction || 0) * 100).toFixed(1)}% of the drawn region. This is projected overlap, not solid collision.`)}</p>
        <p>{text("基于当前模型位置与形状，尚未验证为现场实测；不能据此直接判定合规。距离与面积暂以原生单位显示，未换算为现场单位。", "Based on current model geometry and placement, not verified field measurements or a compliance verdict. Distances and areas are in native model units, not converted to field units.")}</p>
        <details><summary>{text("测量依据", "Measurement evidence")}</summary>
          <p>{text("版本", "Revision")}: {result.revisionId.slice(0, 8)}</p>
          {result.quality.groundReference && <p>{text("地面参考坐标系", "Ground reference frame")}: {result.coordinateFrameId.slice(0, 8)}</p>}
          {result.references.map((ref, i) => <p key={ref.entityId}>{i === 0 ? "A" : "B"} · {revision.document.entities.find(e => e.id === ref.entityId)?.label}<br />{text("模型", "Model")}: {ref.representationId.slice(0, 8)} · {ref.placementState === "confirmed" ? text("位置已确认", "Placement confirmed") : text("位置未确认", "Placement unconfirmed")}{ref.qualityStatus === "rejected" && <> · {text("模型质量检查未通过", "Model quality check failed")}</>}{result.quality.surfaceFits?.[i] && <><br />{text("拟合面覆盖", "Fitted surface coverage")}: {(result.quality.surfaceFits[i].areaFraction * 100).toFixed(1)}%</>}</p>)}
        </details>
      </div>}
    </>}
  </section>;
}
