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
  const { t } = useI18n();
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
    measurement_no_stable_local_plane: t("measure.error.measurement_no_stable_local_plane"),
    measurement_model_missing: t("measure.error.measurement_model_missing"),
    measurement_no_stable_bend: t("measure.error.measurement_no_stable_bend"),
    measurement_points_invalid: t("measure.error.measurement_points_invalid"),
    measurement_points_coincident: t("measure.error.measurement_points_coincident"),
    measurement_points_collinear: t("measure.error.measurement_points_collinear"),
    measurement_scale_invalid: t("measure.error.measurement_scale_invalid"),
    measurement_no_stable_plane: t("measure.error.measurement_no_stable_plane"),
    measurement_complexity_limit: t("measure.error.measurement_complexity_limit"),
    measurement_ground_missing: t("measure.error.measurement_ground_missing"),
    measurement_region_invalid: t("measure.error.measurement_region_invalid"),
    measurement_frame_mismatch: t("measure.error.measurement_frame_mismatch"),
  };
  return <section className="spatial-measurements" aria-label={t("measure.title")}>
    <h4>{t("measure.title")} <small>{t("measure.modelEstimate")}</small></h4>
    <p>{t("measure.intro")}</p>
    {!model || model.sourceValidity === "stale" ? <p>{t("measure.selectModelFirst")}</p> : <>
      <label>{t("measure.kind")}<select value={kind} onChange={e => { if (drawing) onDraw(); onPickPoints(false); onClearSurface?.(); setKind(e.target.value); }}>
        <option value="point_ground">{t("measure.kind.point_ground")}</option>
        <option value="point_distance">{t("measure.kind.point_distance")}</option>
        <option value="region_ground">{t("measure.kind.region_ground")}</option>
        {(analysisAvailable || savedBend?.status === "measured") && <option value="bend">{t("measure.kind.bend")}</option>}
        <option value="edge_vertical">{t("measure.kind.edge_vertical")}</option>
        <option value="edge_angle">{t("measure.kind.edge_angle")}</option>
        {(analysisAvailable || savedSurface) && <option value="inclination">{t("measure.kind.inclination")}</option>}
        {analysisAvailable && <><option value="angle">{t("measure.kind.angle")}</option>
        <option value="distance">{t("measure.kind.distance")}</option>
        <option value="occupancy">{t("measure.kind.occupancy")}</option></>}
      </select></label>
      <p>{pointMode ? (points.length ? points.map((point, i) => `${i + 1} · ${revision.document.entities.find(entity => entity.id === point.entityId)?.label || point.entityId}`).join("；") : t("measure.pointsAnyObject")) : <><b>{t("measure.objectA")}</b>{selected?.label}</>}</p>
      {kind === "bend" && savedBend && <p className="saved-bend-status">{t("measure.savedBend." + savedBend.status)}{savedBend.status === "failed" && <small> {errors[savedBend.reason || ""] || savedBend.reason}</small>}</p>}
      {pointMode ? <>
        <p>{kind === "point_ground" ? t("measure.guide.point_ground") : kind === "point_distance" ? t("measure.guide.point_distance") : kind === "region_ground" ? t("measure.guide.region_ground") : kind === "edge_vertical" ? t("measure.guide.edge_vertical") : t("measure.guide.edge_angle")}</p>
        <button type="button" onClick={()=>onPickPoints(!pickingPoints,pointCount)}>{pickingPoints ? t("measure.cancelPicking") : points.length ? t("measure.pickAgain") : pointCount === 1 ? t("measure.pickOnePoint") : t("measure.pickPoints", { n: pointCount })}</button>
        {(points.length > 0 || result) && <button type="button" onClick={() => { onPickPoints(false); setResult(null); setError(""); }}>{t("measure.clearRestore")}</button>}
        <p>{t("measure.pickedCount", { n: points.length, total: pointCount })}</p>
      </> : kind === "bend" ? <p>{t("measure.bendExplain")}</p> : kind === "inclination" ? <p>{t("measure.inclinationExplain")}</p> : kind !== "occupancy" ? <label>{t("measure.objectB")}<select value={target} onChange={e => setTarget(e.target.value)}>
        <option value="">{t("measure.chooseReference")}</option>
        {objects.map(e => <option key={e.id} value={e.id}>{e.label || e.id}</option>)}
      </select></label> : <><button type="button" onClick={onDraw}>{t(drawing ? "measure.cancelDrawing" : region ? "measure.redrawRegion" : "measure.drawRegion")}</button>
        <p>{t(region ? "measure.regionSet" : "measure.regionHow")}</p></>}
      {kind === "inclination" && <p>{savedSurface ? t("measure.surfaceSaved", { id: savedSurface.surfaceId, area: savedSurface.areaNative2.toPrecision(3) }) : inclinationOutcome?.status === "measured" ? t("measure.surfacesFound", { n: inclinationOutcome.surfaces.length }) : t("measure.autoAnalysis", { reason: inclinationOutcome?.reason ? errors[inclinationOutcome.reason] || inclinationOutcome.reason : t("measure.noSurfaces") })}</p>}
      {savedSurface && kind === "inclination" && <p>{savedSurface.result.quality.angularErrorDeg == null ? t("measure.groundErrorUnknown") : t("measure.angularError", { deg: savedSurface.result.quality.angularErrorDeg.toFixed(1) })}</p>}
      {(pointMode || analysisAvailable) && <button type="button" className="measure-calculate" disabled={Boolean(kind === "inclination" && savedSurface) || busy || drawing || pickingPoints || (pointMode ? points.length!==pointCount : kind === "occupancy" ? !region : (kind === "angle" || kind === "distance") && !objects.some(e => e.id === target))} onClick={calculate}>{t(kind === "inclination" && savedSurface ? "measure.savedSurfaceMeasurement" : busy ? "measure.calculating" : kind === "bend" && savedBend?.status === "measured" ? "measure.recalculate" : "measure.calculate")}</button>}
      {error && <p role="alert">{errors[error] || t("measure.failed")} <small>{error}</small></p>}
      {result && <div className="measurement-result" role="status">
        <output className={result.caliper ? "caliper-output" : undefined}>{result.displayLabel || <>{Number(result.value.toPrecision(4))} {result.unit === "deg" ? "°" : t(result.unit === "native2" ? "measure.unit.native2" : "measure.unit.native")}</>}</output>
        <p>{caliperMode ? t(kind === "region_ground" ? "measure.explain.region_ground" : kind === "point_distance" ? "measure.explain.point_distance" : "measure.explain.point_ground")
          : kind === "bend" ? t("measure.explain.bend", { deg: result.value.toFixed(1), flat: (180 - result.value).toFixed(1) })
          : kind === "inclination" ? t("measure.explain.inclination", { deg: result.value.toFixed(1), dev: (result.quality.deviationFromVerticalDeg ?? 0).toFixed(1) })
          : kind === "occupancy" ? t("measure.explain.occupancy", { pct: ((result.quality.regionFraction || 0) * 100).toFixed(1) })
          : t("measure.explain." + kind)}</p>
        <p>{t("measure.notVerified")}</p>
        {result.caliper && <><p>{scale.nativeToMeters == null ? t("measure.scaleUnknown") : t("measure.scaleConversion", { cm: Number((scale.nativeToMeters * 100).toPrecision(5)) })} {t("measure.scaleStatus")}: {scale.status} · {scale.source}</p>
          <button type="button" onClick={() => { const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" })); const a = document.createElement("a"); a.href = url; a.download = `model-measurement-${kind}.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }}>{t("measure.downloadJson")}</button></>}
        {!result.caliper && result.unit !== "deg" && <p>{t("measure.nativeUnitsNote")}</p>}
        <details><summary>{t("measure.evidence")}</summary>
          <p>{t("measure.revision")}: {result.revisionId.slice(0, 8)}</p>
          {result.quality.groundReference && <p>{t("measure.groundFrame")}: {result.coordinateFrameId.slice(0, 8)}</p>}
          {kind === "bend" && result.quality.surfaceFits?.map((fit,i)=><p key={i}>{t("measure.faceFit", { n: i + 1, pct: (fit.areaFraction * 100).toFixed(1), rms: fit.rmsResidualNative.toPrecision(3) })}</p>)}
          {result.references.map((ref, i) => <p key={ref.entityId}>{String.fromCharCode(65+i)} · {revision.document.entities.find(e => e.id === ref.entityId)?.label}<br />{t("measure.model")}: {ref.representationId.slice(0, 8)} · {ref.placementState === "confirmed" ? t("measure.placementConfirmed") : t("measure.placementUnconfirmed")}{ref.qualityStatus === "rejected" && <> · {t("measure.qualityFailed")}</>}{result.quality.surfaceFits?.[i] && <><br />{t("measure.fitCoverage")}: {(result.quality.surfaceFits[i].areaFraction * 100).toFixed(1)}%</>}</p>)}
        </details>
      </div>}
    </>}
  </section>;
}
