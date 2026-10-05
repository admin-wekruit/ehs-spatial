import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  type IdentitySuggestion,
  downloadAsset,
  downloadJSON,
  downloadReportEdit,
  id,
  owner,
  PUBLICATION_ID,
  prepareOwnedRequest,
  request,
  sendOwnedRequest,
} from "./api";
import {
  cameraForImage,
  publicationReaderURL,
  currentPublicationURL,
  currentEntityId,
  modelScale,
  editableTransform,
  jobResultSummary,
  jsonObject,
  modelGeometry,
  modelTilt,
  observationsFor,
  sourceDimensions,
  sourceScale,
} from "./core";
import { useI18n } from "./i18n";
import { ErrorNotice } from "./App";
import { ModelEvidence } from "./ModelEvidence";
import { IdentityReview } from "./IdentityReview";
import { AgentPanel } from "./AgentPanel";
import { Extent, ReportScene } from "./ReportScene";
import { applyMeasurementLayer, loadMeasurementLayer, withLayerAssets, type MeasurementLayer } from "./measurement-layer";
import { SceneResources, useSceneResources } from "./SceneResources";
import { ReportObjectFindings } from "./ReportObjectFindings";
import { ReportReview, type AssessmentSummary } from "./ReportReview";
import { entityEvidenceStatus, identityCounts, isReferenceSurface } from "./scene-semantics";
import { ReportEvidence } from "./ReportEvidence";
import type {
  Commit,
  EditBatch,
  Entity,
  Evaluation,
  Job,
  Operation,
  ProjectDetail,
  Publication,
  PublicationSummary,
  PublicationView,
  ReportEditSummary,
  Review,
  Revision,
  SceneDocument,
  Selection,
} from "./types";
import "./workcell-report.css";

function ReportEditDownload({publicationId, editId}:{publicationId:string;editId:string}) {
  const {t}=useI18n(), [loading,setLoading]=useState(false), [error,setError]=useState<unknown>();
  return <span className="report-download"><button disabled={loading} onClick={async()=>{
    setLoading(true);setError(undefined);
    try{await downloadReportEdit(publicationId,editId);}catch(error){setError(error);}finally{setLoading(false);}
  }}>{t(loading ? "loading" : "reportTechnical")} JSON ↓</button><ErrorNotice error={error}/></span>;
}

export function ReportDownload({
  assetId,
  children,
}: {
  assetId: string;
  children: React.ReactNode;
}) {
  const [error, setError] = useState<unknown>(),
    [loading, setLoading] = useState(false);
  const { t } = useI18n();
  return (
    <span className="report-download">
      <button
        disabled={loading}
        onClick={async () => {
          setLoading(true);
          setError(undefined);
          try {
            await downloadAsset(assetId);
          } catch (e) {
            setError(e);
          } finally {
            setLoading(false);
          }
        }}
      >
        {children} {loading ? t("loading") : "↓"}
      </button>
      <ErrorNotice error={error} />
    </span>
  );
}
export function ReportDate({ value }: { value: string }) {
  const { language } = useI18n();
  return (
    <time dateTime={value}>
      {new Date(value).toLocaleString(language === "zh" ? "zh-CN" : "en-US", {
        dateStyle: "medium",
        timeStyle: "short",
      })}
    </time>
  );
}
const jobKinds: Record<string, string> = {
  import_scene: "reportImportScene",
  analyze_capture: "createAnalyze",
  generate_object: "generate",
  generate_scene: "generateScene",
  export_blender: "reportExportBlender",
  export_glb: "reportAssets",
  export_json: "reportAssets",
  reassociate_scene: "reportReassociateScene",
};
const reportSections = [
  ["spatial", "reportSpatial"],
  ["understanding", "reportUnderstanding"],
  ["safety", "reportSafety"],
  ["assets", "reportAssets"],
  ["history", "reportHistory"],
] as const;
function jump(section: string) {
  document
    .getElementById("workcell-" + section)
    ?.scrollIntoView({ behavior: "smooth", block: "start" });
}

function contextURL(
  hash: string,
  context: {
    selection: Selection;
    imageId: string | null;
    box: number[] | null;
    reviewMode: boolean;
    agentOpen: boolean;
  },
) {
  const [route, query = ""] = hash.split("?");
  const params = new URLSearchParams(query);
  if (params.has("revision") && context.selection.revisionId) params.set("revision", context.selection.revisionId);
  for (const [key, value] of Object.entries({
    object: context.selection.entityId,
    observation: context.selection.observationId,
    image: context.imageId,
    box: context.box?.join(","),
    review: context.reviewMode ? "1" : null,
    agent: context.agentOpen ? "1" : null,
  })) {
    if (value) params.set(key, value);
    else params.delete(key);
  }
  return route + (params.size ? "?" + params : "");
}

export function WorkcellReport({
  projectId,
  publicationId,
  requestedRevision,
  historical = false,
}: {
  projectId?: string;
  publicationId?: string;
  requestedRevision?: string | null;
  historical?: boolean;
}) {
  const { t } = useI18n();
  const readOnly = !!PUBLICATION_ID;
  const [detail, setDetail] = useState<ProjectDetail>(),
    [publication, setPublication] = useState<Publication>(),
    [layer, setLayer] = useState<MeasurementLayer | null>(null);
  const baseResources = useSceneResources(),
    resources = useMemo(() => withLayerAssets(baseResources, layer), [baseResources, layer]);
  const [error, setError] = useState<unknown>(),
    [canManage, setCanManage] = useState(false),
    [busy, setBusy] = useState(false),
    [notice, setNotice] = useState("");
  const [reviewMode, setReviewMode] = useState(false),
    [agentOpen, setAgentOpen] = useState(false),
    [identitySuggestion, setIdentitySuggestion] = useState<IdentitySuggestion>(),
    [identityEntityIds, setIdentityEntityIds] = useState<[string,string]>(),
    [draw, setDraw] = useState(false),
    [box, setBox] = useState<number[] | null>(null);
  const [selection, setSelection] = useState<Selection>({
    projectId: "",
    revisionId: "",
    entityId: null,
    observationId: null,
    cameraId: null,
  });
  const [imageId, setImageId] = useState<string | null>(null),
    [jobs, setJobs] = useState<Job[]>([]),
    [history, setHistory] = useState<PublicationSummary[]>([]),
    [newestPublication, setNewestPublication] = useState<PublicationSummary>(),
    [events, setEvents] = useState<EditBatch[]>([]),
    [reportEdits, setReportEdits] = useState<ReportEditSummary[]>([]);
  const [objectListRequest, setObjectListRequest] = useState(0),
    [generation, setGeneration] = useState(0);
  const [assessment, setAssessment] = useState<AssessmentSummary>();
  const [evaluationRecords, setEvaluationRecords] = useState<{ revisionId: string; evaluations: Evaluation[] | null }>();
  const saving = useRef(false),
    alive = useRef(true),
    appliedHead = useRef<string | undefined>(undefined),
    startedReviewJobs = useRef(new Set<string>());
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
    };
  }, []);
  useEffect(() => {
    let live = true;
    setError(undefined);
    setNotice("");
    setDetail(undefined);
    setPublication(undefined);
    setLayer(null);
    setJobs([]);
    setHistory([]);
    setNewestPublication(undefined);
    setEvents([]);
    setReportEdits([]);
    startedReviewJobs.current.clear();
    setReviewMode(false);
    setAgentOpen(false);
    setBox(null);
    setDraw(false);
    (async () => {
      const [view, publications] = await Promise.all([
        publicationId ? request<PublicationView>("/api/publications/" + publicationId + "/view") : Promise.resolve(undefined),
        request<{ items: PublicationSummary[] }>("/api/publications"),
      ]);
      const pub = view?.publication;
      const pid = pub?.projectId || projectId!;
      if (pub && publications.items.some(p => p.id === pub.id)) {
        // ponytail: scan newer snapshots in API order; include branchId in summaries if long multi-branch histories make these reads costly.
        for (const item of publications.items.filter(p => p.projectId === pid)) {
          if (item.id === pub.id) break;
          const {publication: candidate} = await request<PublicationView>("/api/publications/" + item.id + "/view");
          if (candidate.projectId !== pub.projectId || candidate.snapshot.revision.branchId !== pub.snapshot.revision.branchId) continue;
          const destination = currentPublicationURL(pub, candidate, location.href);
          if (!live) return;
          if (destination) { location.replace(destination); return; }
          setNewestPublication(item);
          break;
        }
      }
      const d: ProjectDetail = view ? {project:view.project, branch:view.branch, branches:view.branches, revision:view.publication.snapshot.revision}
        : await request<ProjectDetail>("/api/projects/" + pid);
      const measured = pub ? await loadMeasurementLayer(pub.id) : null;
      const revision = applyMeasurementLayer(
        pub?.snapshot.revision ||
        (requestedRevision
          ? await request<Revision>("/api/revisions/" + requestedRevision)
          : d.revision), measured);
      if (!live) return;
      if (pub) {
        const reader = publicationReaderURL(revision.document.schemaVersion, location.href);
        if (reader) { location.replace(reader); return; }
      }
      if (revision.projectId !== pid) throw new Error("invalid_revision");
      const next = {
        ...d,
        revision,
        branch: d.branches.find((b) => b.id === revision.branchId) || d.branch,
      };
      const can = !readOnly && !!(await owner(pid));
      if (!live) return;
      setDetail(next);
      setPublication(pub);
      setLayer(measured?.revisionId === revision.id ? measured : null);
      setReportEdits(view?.edits || []);
      setCanManage(can);
      appliedHead.current = revision.id;
      const params = new URLSearchParams(location.hash.split("?")[1] || "");
      const entity = revision.document.entities.find(
        (e) => e.id === currentEntityId(revision.document, params.get("object") || ""),
      );
      const requestedImage = revision.document.assets.find(
        (a) => a.id === params.get("image") && a.kind === "source_image",
      )?.id;
      const observations = entity
        ? observationsFor(revision.document, entity)
        : [];
      const imageObservations = observations.filter(
        (o) => o.imageId === requestedImage,
      );
      const obs =
        observations.find(
          (o) =>
            o.id === params.get("observation") &&
            (!requestedImage || o.imageId === requestedImage),
        ) ||
        (requestedImage
          ? imageObservations.length === 1
            ? imageObservations[0]
            : undefined
          : observations[0]);
      const camera = cameraForImage(revision.document, requestedImage || obs?.imageId || revision.document.assets.find(a => a.kind === "source_image")?.id);
      const selectedImage =
        requestedImage ||
        camera?.imageId ||
        obs?.imageId ||
        revision.document.assets.find((a) => a.kind === "source_image")?.id ||
        null;
      setImageId(selectedImage);
      setSelection({
        projectId: pid,
        revisionId: revision.id,
        entityId: entity?.id || null,
        observationId: obs?.id || null,
        cameraId: camera?.id || null,
      });
      if (!readOnly && params.get("review") === "1") setReviewMode(true);
      if (!readOnly && params.get("agent") === "1") setAgentOpen(true);
      const inputBox = params.get("box")?.split(",").map(Number);
      const image = revision.document.assets.find(
        (a) => a.id === selectedImage,
      );
      const width = camera?.width || Number(image?.width),
        height = camera?.height || Number(image?.height);
      if (
        !readOnly && inputBox?.length === 4 &&
        inputBox.every(Number.isFinite) &&
        inputBox[0] >= 0 &&
        inputBox[1] >= 0 &&
        inputBox[2] > inputBox[0] &&
        inputBox[3] > inputBox[1] &&
        inputBox[2] <= width &&
        inputBox[3] <= height
      )
        setBox(inputBox);
      const batches = pub ? { items: pub.snapshot.editBatches || [] }
        : await request<{ items: EditBatch[] }>("/api/projects/" + pid + "/edits");
      if (live) {
        setHistory(publications.items.filter((p) => p.projectId === pid));
        setEvents(batches.items);
      }
    })().catch((e) => {
      if (live) setError(e);
    });
    return () => {
      live = false;
    };
  }, [projectId, publicationId, requestedRevision, historical]);
  useEffect(() => {
    if (!detail) return;
    if (publicationId) {
      setJobs(publication?.snapshot.jobs || []);
      return;
    }
    const pid = detail.project.id;
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const response = await request<{ items: Job[] }>(
          "/api/projects/" + pid + "/jobs",
        );
        if (!active) return;
        setJobs(response.items);
        // A pinned view follows only a review explicitly started here; publications remain immutable.
        if (!publicationId && !saving.current) {
          const updated = response.items.some(
            (j) =>
              (!requestedRevision || startedReviewJobs.current.has(j.id)) &&
              j.headAdvanced &&
              j.resultRevisionId &&
              j.resultRevisionId !== appliedHead.current &&
              j.baseRevisionId === appliedHead.current,
          );
          if (updated) {
            const next = await request<ProjectDetail>("/api/projects/" + pid);
            const branch = next.branches.find(
              (b) => b.id === detail!.branch.id,
            );
            if (branch && branch.headRevisionId !== appliedHead.current) {
              const revision = await request<Revision>(
                "/api/revisions/" + branch.headRevisionId,
              );
              if (active) {
                appliedHead.current = revision.id;
                setDetail({ ...next, branch, revision });
              }
            }
          }
        }
        if (
          response.items.some((j) =>
            ["pending_dispatch", "queued", "running"].includes(j.status),
          )
        )
          timer = setTimeout(poll, 4000);
      } catch (e) {
        if (active) setError(e);
      }
    }
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [
    detail?.project.id,
    publicationId,
    publication,
    requestedRevision,
    generation,
  ]);
  useEffect(() => {
    if (detail) setSelection((s) => ({ ...s, revisionId: detail.revision.id }));
  }, [detail?.revision.id]);
  useEffect(() => {
    if (!detail) return;
    const next = contextURL(location.hash, {
      selection,
      imageId,
      box,
      reviewMode,
      agentOpen,
    });
    if (next !== location.hash) window.history.replaceState(null, "", next);
  }, [
    detail?.revision.id,
    selection.revisionId,
    selection.entityId,
    selection.observationId,
    imageId,
    box,
    reviewMode,
    agentOpen,
  ]);
  function select(entityId: string, observationId?: string) {
    if (!detail) return;
    const entity = detail.revision.document.entities.find(
      (e) => e.id === entityId,
    );
    if (!entity) return;
    const observations = observationsFor(detail.revision.document, entity);
    const observation =
      observations.find((o) => o.id === observationId) ||
      observations.find((o) => o.imageId === imageId) ||
      observations[0];
    const camera = cameraForImage(detail.revision.document, observation?.imageId);
    setSelection((s) => ({
      ...s,
      entityId,
      observationId: observation?.id || null,
      cameraId: camera?.id || s.cameraId,
    }));
    if (observation) setImageId(observation.imageId);
    setBox(null);
  }
  function changeCamera(image: string, camera: string | null) {
    if (!detail) return;
    setImageId(image);
    setSelection((s) => {
      const entity = detail.revision.document.entities.find(
        (e) => e.id === s.entityId,
      );
      const observations = entity
        ? observationsFor(detail.revision.document, entity).filter(
            (o) => o.imageId === image,
          )
        : [];
      return {
        ...s,
        cameraId: cameraForImage(detail.revision.document, image)?.id || null,
        observationId:
          observations.find((o) => o.id === s.observationId)?.id ||
          (observations.length === 1 ? observations[0].id : null),
      };
    });
    setBox(null);
  }
  function selectEvidence(context: {
    entityId: string;
    observationId: string | null;
    imageId: string | null;
    cameraId: string | null;
  }) {
    if (!detail) return;
    if (!context.imageId) select(context.entityId);
    else {
      setSelection({
        projectId: detail.project.id,
        revisionId: detail.revision.id,
        entityId: context.entityId,
        observationId: context.observationId,
        cameraId: cameraForImage(detail.revision.document, context.imageId)?.id || null,
      });
      setImageId(context.imageId);
      setBox(null);
    }
    jump("spatial");
  }
  function onSaved(result: Commit) {
    if (!alive.current) return;
    appliedHead.current = result.revision.id;
    setDetail(
      (d) =>
        d && {
          ...d,
          revision: result.revision,
          branch: { ...d.branch, headRevisionId: result.revision.id },
        },
    );
    setSelection(previous => ({...previous,revisionId:result.revision.id, entityId:previous.entityId ? currentEntityId(result.revision.document, previous.entityId) : null}));
    setNotice("reportSaved");
    setDraw(false);
    setBox(null);
    setGeneration((n) => n + 1);
    if (result.editBatch) setEvents((old) => [...old, result.editBatch]);
    const [route, query = ""] = location.hash.split("?");
    const params = new URLSearchParams(query);
    if (params.has("revision")) params.set("revision", result.revision.id);
    params.delete("box");
    window.history.replaceState(
      null,
      "",
      route + (params.size ? "?" + params : ""),
    );
  }
  async function apply(
    operations: Operation[],
    context?: { baseRevisionId: string; agentTurnId?: string },
  ) {
    if (!detail || !canWrite || saving.current) return false;
    if (context && context.baseRevisionId !== detail.revision.id)
      throw new ApiError(409, "conflict");
    saving.current = true;
    setBusy(true);
    try {
      const saved = await request<Commit>(
        "/api/projects/" + detail.project.id + "/edits",
        {
          method: "POST",
          projectId: detail.project.id,
          body: {
            requestId: id(),
            branchId: detail.branch.id,
            baseRevisionId: detail.revision.id,
            operations,
            ...context,
          },
        },
      );
      onSaved(saved);
      return saved;
    } finally {
      saving.current = false;
      if (alive.current) setBusy(false);
    }
  }
  async function reviewModel(entityId: string) {
    if (!detail || !canWrite || saving.current) return;
    setBusy(true);
    setError(undefined);
    try {
      const task = await request<Job>(`/api/projects/${detail.project.id}/jobs`, {
        method: "POST", projectId: detail.project.id,
        body: { requestId: id(), branchId: detail.branch.id, baseRevisionId: detail.revision.id,
          kind: "review_models", inputs: { entityIds: [entityId] }, config: {} },
      });
      if (alive.current) {
        startedReviewJobs.current.add(task.id);
        setJobs(previous => [task, ...previous]);
        setNotice("queued");
        setGeneration(value => value + 1);
      }
    } catch (failure) {
      if (alive.current) setError(failure);
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  async function copy() {
    if (!detail || readOnly) return;
    setBusy(true);
    setError(undefined);
    try {
      const pending = await prepareOwnedRequest(
        "/api/projects/" + detail.project.id + "/forks",
        {
          sourceRevisionId: detail.revision.id,
          title: detail.project.title + " · " + t("copy"),
          operations: [],
        },
      );
      const created = await sendOwnedRequest<ProjectDetail>(pending);
      location.hash = contextURL(
        "/projects/" + created.project.id + "/report",
        {
          selection,
          imageId,
          box,
          reviewMode: true,
          agentOpen,
        },
      );
    } catch (e) {
      setError(e);
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  async function publish() {
    if (!detail || !canWrite) return;
    setBusy(true);
    setError(undefined);
    try {
      const [evaluations, reviews] = await Promise.all([
        request<{ items: Evaluation[] }>(
          "/api/projects/" + detail.project.id + "/evaluations",
        ),
        request<{ items: Review[] }>(
          "/api/projects/" + detail.project.id + "/reviews",
        ),
      ]);
      const evaluationIds = evaluations.items
        .filter((e) => e.sceneRevisionId === detail.revision.id)
        .map((e) => e.id);
      const pub = await request<Publication>(
        "/api/projects/" + detail.project.id + "/publications",
        {
          method: "POST",
          projectId: detail.project.id,
          body: {
            requestId: id(),
            sceneRevisionId: detail.revision.id,
            title: detail.project.title,
            evaluationIds,
            reviewIds: reviews.items
              .filter((r) => evaluationIds.includes(r.evaluationId))
              .map((r) => r.id),
          },
        },
      );
      location.hash = "/reports/" + pub.id;
    } catch (e) {
      setError(e);
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  async function exportBlender() {
    if (!detail || !canWrite) return;
    setBusy(true);
    setError(undefined);
    try {
      await request("/api/projects/" + detail.project.id + "/jobs", {
        method: "POST",
        projectId: detail.project.id,
        body: {
          requestId: id(),
          branchId: detail.branch.id,
          baseRevisionId: detail.revision.id,
          kind: "export_blender",
          inputs: {},
          config: {},
        },
      });
      setGeneration((n) => n + 1);
      jump("history");
    } catch (e) {
      setError(e);
    } finally {
      if (alive.current) setBusy(false);
    }
  }
  const canWrite =
    !readOnly &&
    !!detail &&
    canManage &&
    !publication &&
    detail.revision.id === detail.branch.headRevisionId;
  if (!detail)
    return (
      <div className="workcell-report report-loading">
        <h1>{t("workcellReport")}</h1>
        <ErrorNotice error={error} />
        {!error && <p role="status">{t("loading")}</p>}
      </div>
    );
  const { revision, project } = detail,
    doc = revision.document;
  const entity = doc.entities.find((e) => e.id === selection.entityId);
  const currentJobs = jobs.filter(
    (j) =>
      j.baseRevisionId === revision.id || j.resultRevisionId === revision.id,
  );
  const exportJobs = jobs.filter(
    (j) =>
      j.baseRevisionId === revision.id &&
      ["export_blender", "export_glb", "export_json"].includes(j.kind) &&
      jobResultSummary(j.result).assets.length,
  );
  const objects = doc.entities.filter((e) => !e.sourceContext);
  const sourceImages = doc.assets.filter((a) => a.kind === "source_image");
  const fromReport = history.find(p => p.id === new URLSearchParams(location.hash.split("?")[1] || "").get("fromReport"));
  const newerReport = publication && newestPublication && newestPublication.id !== publication.id
    ? {href: `./app.html?report=${encodeURIComponent(newestPublication.id)}#/reports/${encodeURIComponent(newestPublication.id)}`, title: newestPublication.title} : undefined;
  const modelWorkbenchURL = contextURL("#/projects/" + project.id + "/workbench?revision=" + revision.id, { selection, imageId, box: null, reviewMode: false, agentOpen: false });
  const pendingGeometry = objects.filter(e => !e.representations?.length || e.representations.some(r => r.placementState === "unconfirmed")).length;
  const assessmentState = assessment?.revisionId === revision.id ? assessment.state : "loading";
  const exactEvents: ReportEditSummary[] = publication ? reportEdits :
    events.filter(e => e.revisionId === revision.id).map(e => ({...e, operationTypes:e.operations.map(operation => operation.type)}));
  return (
    <SceneResources.Provider value={resources}>
    <article
      className={"workcell-report" + (reviewMode ? " is-reviewing" : "")}
    >
      <header className="workcell-masthead">
        <div className="report-breadcrumb">
          <a href={readOnly ? "#/reports" : "#/projects"}>{t(readOnly ? "reports" : "projects")}</a>
          <span>/</span>
          <span>{t("workcellReport")}</span>
          <span className="report-status">
            {t(readOnly ? "reportReadOnly" : publication ? "reportReadonlySnapshot" : "reportDraft")}
          </span>
        </div>
        <div className="report-title-row">
          <div>
            <h1>{publication?.title || project.title}</h1>

          </div>
          {!readOnly && <div className="report-header-actions">
            <a className="button primary" href={modelWorkbenchURL}>{t("reportModelWorkbench")} ↗</a>
            <button onClick={() => { setReviewMode(true); jump("safety"); }}>{t("reportReviewAction")}</button>
          <button
            onClick={() => {
              setReviewMode(true);
              setAgentOpen(true);
              setDraw(true);
              setBox(null);
              setSelection((s) => ({
                ...s,
                entityId: null,
                observationId: null,
              }));
              jump("spatial");
            }}
          >
            {t("reportMissingObject")} ＋
          </button>
            <button disabled={busy} onClick={copy}>
              {t("copy")}
            </button>
            {canWrite && (
              <button disabled={busy} onClick={publish}>
                {t("reportPublish")} ↗
              </button>
            )}
          </div>}
        </div>
        <div className="report-meta">
          <ReportDate value={publication?.createdAt || revision.createdAt} />
          <span>
            {sourceImages.length} {t("photos")}
          </span>
          <span>
            {t("version")} {revision.id.slice(0, 8)}
          </span>
          <span>{t(detail.branch.kind)}</span>
        </div>
        {!readOnly && <div className="report-steps" aria-label={t("workcellReport")}>
          <button
            aria-pressed={!reviewMode}
            onClick={() => {
              setReviewMode(false);
              setAgentOpen(false);
              setDraw(false);
            }}
          >
            {t("readReport")}

          </button>
          <button
            aria-pressed={reviewMode}
            onClick={() => {
              setReviewMode(true);
              jump("safety");
            }}
          >
            {t("reviewReport")}

          </button>
        </div>}
      </header>
      <nav className="report-index" aria-label={t("reportContents")}>
        {reportSections.map(([anchor, key]) => (
          <button key={anchor} onClick={() => jump(anchor)}>
            {t(key)}
          </button>
        ))}
      </nav>
      <ErrorNotice error={error} />
      {fromReport && fromReport.id !== publication?.id && <p className="report-notice" role="status">
        {t("reportOpenedCurrent")} {" "}<a href={"#/reports/" + fromReport.id + "?snapshot=1"}>{t("reportOpenOriginalSnapshot")} ↗</a>
      </p>}
      {notice && (
        <p className="report-notice" role="status">
          {t(notice)}
        </p>
      )}
      <section
        id="workcell-spatial"
        className="workcell-section report-visual-section"
      >


        {draw && <p className="report-notice">{t("reportDrawHint")}</p>}
        <ReportScene
          revision={revision}
          newerReport={newerReport}
          selection={selection}
          onSelect={select}
          imageId={imageId}
          cameraId={selection.cameraId}
          onCamera={changeCamera}
          objectListRequest={objectListRequest}
          onFeedback={() => { setReviewMode(true); setAgentOpen(true); }}
          onClearSelection={() => {
            setSelection((value) => ({ ...value, entityId: null, observationId: null }));
            setBox(null); setAgentOpen(false); setReviewMode(false);
          }}
          onOpenSourceCad={jsonObject(jsonObject(doc.reportEvidence)?.historical)?.cad ? () => jump("original-cad") : undefined}
          inspector={<>
            {!(reviewMode && agentOpen) && <div className="report-selection-details">
              {entity ? <>
                {layer?.facts?.[entity.id] && <section className="report-measurement-layer" data-measurement-facts={entity.id}>
                  <h4>照片测量 · 多视角</h4>
                  <dl>{layer.facts[entity.id].map((fact, i) => <div key={i} data-fact-kind={fact.kind}><dt>{fact.label}</dt><dd>{fact.text}</dd></div>)}</dl>
                  {layer.models?.[entity.id] && <p>{layer.models[entity.id].note}</p>}
                  <p>尺度：1 原生单位 = {(layer.scale.nativeToMeters * 100).toFixed(1)} cm{layer.scale.uncertaintyRelative ? `（±${(layer.scale.uncertaintyRelative * 100).toFixed(1)}%，各特征单独定尺度的分散）` : ""}（{layer.scale.source}）。模型估计，未经现场实测验证。</p>
                </section>}
                <ObjectFacts entity={entity} document={doc} /><ModelEvidence entity={entity} onCommit={canWrite ? operations => apply(operations) : undefined}
                  onReview={canWrite ? () => reviewModel(entity.id) : undefined}
                  disabled={busy || jobs.some(task => task.kind === "review_models" && ["pending_dispatch", "queued", "running"].includes(task.status) && Array.isArray(task.inputs.entityIds) && task.inputs.entityIds.includes(entity.id))} />
                <IdentityReview revision={revision} entityId={entity.id} canWrite={canWrite && !busy} publicationId={readOnly ? publication?.id : undefined} onSelect={select} onApply={apply}
                  onSuggest={suggestion => { select(suggestion.entityIds[0]); setIdentitySuggestion(suggestion); setReviewMode(true); setAgentOpen(true); }}
                  onAgent={entityIds => { setIdentityEntityIds(entityIds); setReviewMode(true); setAgentOpen(true); }} />
                <ReportObjectFindings revision={revision} publication={publication} entityId={entity.id}
                  readOnly={readOnly}
                  evaluations={evaluationRecords?.revisionId === revision.id ? evaluationRecords.evaluations ?? undefined : undefined}
                  loading={assessmentState === "loading"}
                  onReview={() => { if (!readOnly) setReviewMode(true); jump("safety"); }} />
              </> : <div className="report-inspector-empty"><h3>{t("reportSelectObject")}</h3><p>{t("reportSelectDetails")}</p><strong>{t("reportAssessment_" + assessmentState)}</strong><p>{t("reportAssessmentScopeHint")}</p></div>}
            </div>}
            {entity && !(reviewMode && agentOpen) && <button className="report-inspector-agent" onClick={() => { setReviewMode(true); setAgentOpen(true); }}>{t("sceneFeedbackTitle")}</button>}
        {reviewMode && agentOpen && (
          <div className="report-correction" id="report-correction">
            <header className="report-feedback-header"><h3>{t("sceneFeedbackTitle")}</h3><button onClick={() => setAgentOpen(false)}>{t("sceneBackDetails")}</button></header>
            {(!readOnly || box) && <div>
              {!readOnly && <p>{t("reportReviewHint")}</p>}
              {box && (
                <p>
                  {t("selectedBox")}: {box.map(Math.round).join(", ")}
                </p>
              )}
            </div>}
            {agentOpen &&
              (canWrite || (readOnly && publication && entity) ? (
                <AgentPanel
                  feedbackPublicationId={readOnly ? publication?.id : undefined}
                  identitySuggestion={identitySuggestion?.entityIds[0] === entity?.id ? identitySuggestion : undefined}
                  identityEntityIds={identityEntityIds?.includes(entity?.id || "") ? identityEntityIds : undefined}
                  projectId={project.id}
                  revision={revision}
                  branch={detail.branch}
                  entityId={selection.entityId}
                  observationId={selection.observationId}
                  imageId={imageId}
                  box={box}
                  canWrite={canWrite && !busy}
                  onApply={apply}
                />
              ) : (
                <div className="report-copy-prompt">
                  <p>{t("reportCopyReason")}</p>
                  <button className="primary" disabled={busy} onClick={copy}>
                    {t("reportCopyToReview")} →
                  </button>
                </div>
              ))}
          </div>
        )}
          </>}
          draw={draw}
          onBox={readOnly ? undefined : (value) => {
            setBox(value);
            setDraw(false);
            setAgentOpen(true);
            setReviewMode(true);
          }}
        />
      </section>
      <section id="workcell-understanding" className="workcell-section">
      <div className="report-summary">
        <div><strong>{identityCounts(doc).records}</strong><span>{t("reportObjectRecords")}</span></div>
        <div><strong>{identityCounts(doc).linkedGroups}</strong><span>{t("identityGroups")}</span></div>
        <div><strong>{identityCounts(doc).pending}</strong><span>{t("identityPending")}</span></div>
        <div><strong>{identityCounts(doc).observations}</strong><span>{t("reportPhotoEvidence")}</span></div>
      </div>

        <div className="report-section-heading">
          <div>
            <span className="report-kicker">02 / {t("sourceEvidence")}</span>
            <h2>{t("reportUnderstanding")}</h2>
          </div>
          <button className="report-open-objects" onClick={() => { setObjectListRequest((request) => request + 1); jump("spatial"); }}>
            {t("sceneOpenObjectList")} · {objects.length} ↑
          </button>
        </div>
        <ReportEvidence
          document={doc}
          section="understanding"
          currentImageId={imageId}
          onSelect={select}
          onSelectEvidence={selectEvidence}
        />
        {!objects.length && (
          <p className="report-notice">{t("reportNoAnalysis")}</p>
        )}
      </section>
      <section id="workcell-safety" className="workcell-section">
      <section className="report-overview" aria-label={t("reportSituation")}>
        <div>
          <span className="report-kicker">{t("reportSituation")}</span>
          <h2>{t("reportAssessment_" + assessmentState)}</h2>
          <p>{t(assessmentState === "assessed" ? "reportAssessmentSavedHint" : "reportAssessmentScopeHint")}</p>
        </div>
        <ul className="report-attention">
          {assessmentState === "assessed" && <li><button onClick={() => jump("safety")}><strong>{assessment!.attentionCount}</strong> {t("reportAttentionFindings")} <span aria-hidden="true">↗</span></button></li>}
          {!!pendingGeometry && <li>{readOnly ? <span><strong>{pendingGeometry}</strong> {t("reportAttentionGeometry")}</span> : <a href={modelWorkbenchURL}><strong>{pendingGeometry}</strong> {t("reportAttentionGeometry")} <span aria-hidden="true">↗</span></a>}</li>}
          {!readOnly && assessmentState === "unassessed" && <li><button onClick={() => { setReviewMode(true); jump("safety"); }}>{t("reportStartAssessment")} <span aria-hidden="true">↗</span></button></li>}
          {assessmentState === "unavailable" && <li><button onClick={() => jump("safety")}>{t("reportAssessmentError")} <span aria-hidden="true">↗</span></button></li>}
        </ul>
      </section>

        <div className="report-section-heading">
          <div>
            <span className="report-kicker">03 / EHS</span>
            <h2>{t("reportSafety")}</h2>
          </div>
          {!readOnly && <button
            aria-pressed={reviewMode}
            onClick={() => setReviewMode((v) => !v)}
          >
            {t("reportDispute")}
          </button>}
        </div>
        {!readOnly && <p className="report-policy-link"><a href="#/policies">{t("reportPolicySettings")} ↗</a></p>}
        {!readOnly && reviewMode && !canWrite && (
          <div className="report-copy-prompt">
            <p>{t("reportCopyReason")}</p>
            <button onClick={copy} className="primary" disabled={busy}>
              {t("reportCopyToReview")}
            </button>
          </div>
        )}
        <ReportReview
          detail={detail}
          publication={publication}
          reviewMode={reviewMode}
          canWrite={canWrite && !busy}
          onSelect={(entity) => {
            select(entity);
            jump("spatial");
          }}
          onSaved={onSaved}
          onSummary={setAssessment}
          onEvaluations={setEvaluationRecords}
        />
        <ReportEvidence
          document={doc}
          section="safety"
          onSelect={(entity) => {
            select(entity);
            jump("spatial");
          }}
        />
      </section>
      <section id="workcell-assets" className="workcell-section">
        <div className="report-section-heading">
          <div>
            <span className="report-kicker">04 / {t("reportAssets")}</span>
            <h2>{t("reportAssets")}</h2>
          </div>
          {!readOnly && <a
            className="button"
            href={modelWorkbenchURL}
          >
            {t("reportAdvanced")} ↗
          </a>}
        </div>
        <p className="report-section-intro">{t("reportExportHint")}</p>
        <div className="report-downloads">
          <button
            onClick={() =>
              downloadJSON(doc, "workcell-" + revision.id + ".json")
            }
          >
            Scene JSON ↓
          </button>
          {sourceImages.map((a, i) => (
            <ReportDownload assetId={a.id} key={a.id}>
              {t("reportSourceDownload")} {i + 1}
            </ReportDownload>
          ))}
          {canWrite && (
            <button disabled={busy} onClick={exportBlender}>
              {t("reportExportBlender")}
            </button>
          )}
        </div>
        {exportJobs.map((job) => (
          <div className="report-export" key={job.id}>
            <strong>{t("reportExactExport")}</strong>
            <span>
              {t(job.status)} · {revision.id.slice(0, 8)}
            </span>
            <div className="report-downloads">
              {jobResultSummary(job.result).assets.map((a) => (
                <ReportDownload key={a.id} assetId={a.id}>
                  {a.name || a.id}
                </ReportDownload>
              ))}
            </div>
          </div>
        ))}
        <ReportEvidence
          document={doc}
          section="assets"
          onSelect={(entity) => {
            select(entity);
            jump("spatial");
          }}
        />
      </section>
      <section id="workcell-history" className="workcell-section">
        <span className="report-kicker">05 / {t("reportHistory")}</span>
        <h2>{t("reportHistory")}</h2>
        <p className="report-section-intro">{t("reportPinnedTasks")}</p>
        {currentJobs.length ? (
          <div className="report-table-scroll">
            <table className="report-tasks">
              <thead>
                <tr>
                  <th>{t("reportStage")}</th>
                  <th>{t("reportResult")}</th>
                  <th>{t("reportTime")}</th>
                  <th>{t("reportTaskOutputs")}</th>
                </tr>
              </thead>
              <tbody>
                {currentJobs.map((j) => {
                  const summary = jobResultSummary(j.result);
                  return (
                    <tr key={j.id}>
                      <td>
                        {t(jobKinds[j.kind] || j.kind)}
                        <small>
                          <ReportDate value={j.createdAt} />
                        </small>
                      </td>
                      <td>
                        <span className={"badge " + j.status}>
                          {t(j.status)}
                        </span>
                        {summary.errors.map((error, i) => (
                          <small key={i}>{t(error)}</small>
                        ))}
                      </td>
                      <td>
                        {typeof summary.workerSeconds === "number"
                          ? summary.workerSeconds.toFixed(2) + " s"
                          : "—"}
                      </td>
                      <td>
                        {summary.assets.map((a) => (
                          <ReportDownload key={a.id} assetId={a.id}>
                            {a.name || a.id.slice(0, 8)}
                          </ReportDownload>
                        ))}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="report-footnote">{t("reportNoTasks")}</p>
        )}
        {!!exactEvents.length && (
          <details className="report-source-details">
            <summary>
              {t("timeline")} · {exactEvents.length}
            </summary>
            <ol>
              {exactEvents.map((e) => (
                <li key={e.id}>
                  <strong>
                    {e.operationTypes.map(type => t(type)).join(" · ") ||
                      t("checkpoint")}
                  </strong>
                  <span>
                    {e.baseRevisionId.slice(0, 8)} → {e.revisionId.slice(0, 8)}
                  </span>
                  <ReportDate value={e.createdAt} />
                  {publication ? <ReportEditDownload publicationId={publication.id} editId={e.id} /> :
                    <button onClick={() => downloadJSON(events.find(event => event.id === e.id), `edit-${e.id}.json`)}>{t("reportTechnical")} JSON ↓</button>}

                </li>
              ))}
            </ol>
          </details>
        )}
        <ReportEvidence document={doc} section="quality" onSelect={select} />
        <details className="report-source-details">
          <summary>
            {t("reportRelated")} · {history.length}
          </summary>
          {history.map((p) => (
            <a
              className="report-history-link"
              key={p.id}
              href={"#/reports/" + p.id + "?snapshot=1"}
            >
              <span>{p.title}</span>
              <span>{p.sceneRevisionId.slice(0, 8)}</span>
              <ReportDate value={p.createdAt} />
              <span>↗</span>
            </a>
          ))}
        </details>
      </section>
      <footer className="report-end">
        <span>PANOPTES · {t("workcellReport")}</span>
        <span>
          {t("reportRevision")} {revision.id.slice(0, 8)}
        </span>
        <button onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}>
          ↑
        </button>
      </footer>
    </article>
    </SceneResources.Provider>
  );
}
export function ObjectFacts({
  entity,
  document,
}: {
  entity: Entity;
  document: SceneDocument;
}) {
  const { t } = useI18n(),
    model = modelGeometry(entity),
    dimensions = sourceDimensions(entity),
    scale = modelScale(document, entity),
    transform = editableTransform(entity, document),
    tilt = modelTilt(document, entity);
  const referenceSurface = isReferenceSurface(document, entity);
  const evidenceStatus = entityEvidenceStatus(document, entity);
  const f = (n: number | undefined | null) =>
    typeof n === "number" && Number.isFinite(n)
      ? (n * (scale?.nativeToMeters || 1)).toFixed(3)
      : "—";
  return (
    <>
      <div className="report-object-title">
        <span>{t("selection")}</span>
        <h3>{entity.label || entity.id}</h3>
        <small>
          {referenceSurface ? t("sceneReferenceSurface") : (entity.representations || []).length
            ? t("sourceEvidence")
            : t("reportMissingGeometry")}
        </small>
      </div>
      <dl className="report-measurements">
        <div>
          <dt>{t("sourceEvidence")}</dt>
          <dd>{t(evidenceStatus.photoKey)}{evidenceStatus.photoCount > 0 && ` · ${evidenceStatus.photoCount}`}</dd>
        </div>
        {evidenceStatus.identityKey && <div>
          <dt>{t("entityIdentity")}</dt>
          <dd>{t(evidenceStatus.identityKey)}</dd>
        </div>}
        <div>
          <dt>{t(referenceSurface ? "reportObservedExtent" : [dimensions.widthNative, dimensions.depthNative, dimensions.groundHeight].every(Number.isFinite) ? "reportGroundExtents" : "reportNativeExtents")}</dt>
          <dd>
            <Extent entity={entity} document={document} />
          </dd>
        </div>
        {!referenceSurface && entity.physicalDimensionsUnknown !== true && <div>
          <dt>
            {t("reportCurrentModel")} · {t("width")} × {t("depth")} ×{" "}
            {t("height")}
          </dt>
          <dd>
            {[model?.width, model?.depth, model?.height].map(f).join(" × ")}
          </dd>
        </div>}
        {!referenceSurface && <div>
          <dt>{t("reportPosition")}</dt>
          <dd>{transform?.position.map(f).join(" / ") || "—"}</dd>
        </div>}
        {!referenceSurface && <div>
          <dt>{t("reportOrientation")}</dt>
          <dd>{entity.modelOrientationUnknown === true || tilt === null ? "—" : tilt.toFixed(1) + "°"}</dd>
        </div>}
      </dl>
      <p className="report-footnote">{t(referenceSurface ? "reportReferenceSurfaceNote" : "reportMeasurementNote")}</p>
      <details className="report-source-details">
        <summary>{t("reportSelectedEvidence")}</summary>
        {observationsFor(document, entity).map((o) => (
          <div key={o.id}>
            <strong>
              {document.cameras.findIndex((c) => c.imageId === o.imageId) + 1} ·{" "}
              {t("photos")}
            </strong>
            <span>
              {o.originalPixelBox?.map((n) => Math.round(n)).join(", ")}
            </span>
            {Array.isArray(o.missingEvidence) && (
              <p>{o.missingEvidence.map((x) => String(x)).join(" · ")}</p>
            )}
          </div>
        ))}
      </details>
    </>
  );
}
