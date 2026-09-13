import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  downloadAsset,
  downloadJSON,
  id,
  owner,
  prepareOwnedRequest,
  request,
  sendOwnedRequest,
} from "./api";
import {
  editableTransform,
  jobResultSummary,
  modelGeometry,
  modelTilt,
  observationsFor,
  sourceDimensions,
  sourceScale,
} from "./core";
import { useI18n } from "./i18n";
import { ErrorNotice } from "./App";
import { AgentPanel } from "./AgentPanel";
import { ReportScene } from "./ReportScene";
import { ReportReview } from "./ReportReview";
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
  Review,
  Revision,
  SceneDocument,
  Selection,
} from "./types";
import "./workcell-report.css";

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

function contextURL(hash: string, context: {
  selection: Selection; imageId: string | null; box: number[] | null;
  reviewMode: boolean; agentOpen: boolean;
}) {
  const [route, query = ""] = hash.split("?");
  const params = new URLSearchParams(query);
  for (const [key, value] of Object.entries({
    object: context.selection.entityId, observation: context.selection.observationId,
    image: context.imageId, box: context.box?.join(","),
    review: context.reviewMode ? "1" : null, agent: context.agentOpen ? "1" : null,
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
}: {
  projectId?: string;
  publicationId?: string;
  requestedRevision?: string | null;
}) {
  const { t } = useI18n();
  const [detail, setDetail] = useState<ProjectDetail>(),
    [publication, setPublication] = useState<Publication>();
  const [error, setError] = useState<unknown>(),
    [canManage, setCanManage] = useState(false),
    [busy, setBusy] = useState(false),
    [notice, setNotice] = useState("");
  const [reviewMode, setReviewMode] = useState(false),
    [agentOpen, setAgentOpen] = useState(false),
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
    [events, setEvents] = useState<EditBatch[]>([]);
  const [allObjects, setAllObjects] = useState(false),
    [search, setSearch] = useState(""),
    [generation, setGeneration] = useState(0);
  const saving = useRef(false),
    alive = useRef(true),
    appliedHead = useRef<string | undefined>(undefined);
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
    setJobs([]);
    setHistory([]);
    setEvents([]);
    setReviewMode(false);
    setAgentOpen(false);
    setBox(null);
    setDraw(false);
    (async () => {
      const pub = publicationId
        ? await request<Publication>("/api/publications/" + publicationId)
        : undefined;
      const pid = pub?.projectId || projectId!;
      const d = await request<ProjectDetail>("/api/projects/" + pid);
      const revision =
        pub?.snapshot.revision ||
        (requestedRevision
          ? await request<Revision>("/api/revisions/" + requestedRevision)
          : d.revision);
      if (revision.projectId !== pid) throw new Error("invalid_revision");
      const next = {
        ...d,
        revision,
        branch: d.branches.find((b) => b.id === revision.branchId) || d.branch,
      };
      const can = !!(await owner(pid));
      if (!live) return;
      setDetail(next);
      setPublication(pub);
      setCanManage(can);
      appliedHead.current = revision.id;
      const params = new URLSearchParams(location.hash.split("?")[1] || "");
      const entity = revision.document.entities.find(
        (e) => e.id === params.get("object"),
      );
      const requestedImage = revision.document.assets.find(
        (a) => a.id === params.get("image") && a.kind === "source_image",
      )?.id;
      const observations = entity
        ? observationsFor(revision.document, entity)
        : [];
      const imageObservations = observations.filter((o) => o.imageId === requestedImage);
      const obs =
        observations.find(
          (o) =>
            o.id === params.get("observation") &&
            (!requestedImage || o.imageId === requestedImage),
        ) ||
        (requestedImage
          ? imageObservations.length === 1 ? imageObservations[0] : undefined
          : observations[0]);
      const camera =
        revision.document.cameras.find(
          (c) => c.imageId === (requestedImage || obs?.imageId),
        ) || (!requestedImage ? revision.document.cameras[0] : undefined);
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
      if (params.get("review") === "1") setReviewMode(true);
      if (params.get("agent") === "1") setAgentOpen(true);
      const inputBox = params.get("box")?.split(",").map(Number);
      const image = revision.document.assets.find(
        (a) => a.id === selectedImage,
      );
      const width = camera?.width || Number(image?.width),
        height = camera?.height || Number(image?.height);
      if (
        inputBox?.length === 4 &&
        inputBox.every(Number.isFinite) &&
        inputBox[0] >= 0 &&
        inputBox[1] >= 0 &&
        inputBox[2] > inputBox[0] &&
        inputBox[3] > inputBox[1] &&
        inputBox[2] <= width &&
        inputBox[3] <= height
      )
        setBox(inputBox);
      const [publications, batches] = await Promise.all([
        request<{ items: PublicationSummary[] }>("/api/publications"),
        pub
          ? Promise.resolve({ items: pub.snapshot.editBatches || [] })
          : request<{ items: EditBatch[] }>("/api/projects/" + pid + "/edits"),
      ]);
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
  }, [projectId, publicationId, requestedRevision]);
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
        // A completed task can advance a live report, never a publication or explicitly pinned revision.
        if (!publicationId && !requestedRevision && !saving.current) {
          const updated = response.items.some(
            (j) =>
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
    const next = contextURL(location.hash, { selection, imageId, box, reviewMode, agentOpen });
    if (next !== location.hash) window.history.replaceState(null, "", next);
  }, [detail?.revision.id, selection.entityId, selection.observationId, imageId, box, reviewMode, agentOpen]);
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
    const camera = detail.revision.document.cameras.find(
      (c) => c.imageId === observation?.imageId,
    );
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
        cameraId: camera,
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
        cameraId: context.cameraId,
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
  async function copy() {
    if (!detail) return;
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
      location.hash = contextURL("/projects/" + created.project.id + "/report", {
        selection, imageId, box, reviewMode: true, agentOpen,
      });
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
  const filteredObjects = objects.filter((e) =>
    (e.label || e.id).toLowerCase().includes(search.toLowerCase()),
  );
  const sourceImages = doc.assets.filter((a) => a.kind === "source_image");
  const exactEvents =
    publication?.snapshot.editBatches ||
    events.filter((e) => e.revisionId === revision.id);
  return (
    <article
      className={"workcell-report" + (reviewMode ? " is-reviewing" : "")}
    >
      <header className="workcell-masthead">
        <div className="report-breadcrumb">
          <a href="#/projects">{t("projects")}</a>
          <span>/</span>
          <span>{t("workcellReport")}</span>
          <span className="report-status">
            {t(publication ? "reportReadonlySnapshot" : "reportDraft")}
          </span>
        </div>
        <div className="report-title-row">
          <div>
            <h1>{publication?.title || project.title}</h1>
            <p>{t("reportFirstHint")}</p>
          </div>
          <div className="report-header-actions">
            <button disabled={busy} onClick={copy}>
              {t("copy")}
            </button>
            {canWrite && (
              <button disabled={busy} className="primary" onClick={publish}>
                {t("reportPublish")} ↗
              </button>
            )}
          </div>
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
        <div className="report-steps" aria-label={t("workcellReport")}>
          <button
            aria-pressed={!reviewMode}
            onClick={() => {
              setReviewMode(false);
              setAgentOpen(false);
              setDraw(false);
            }}
          >
            {t("readReport")}
            <small>{t("reportReadOnly")}</small>
          </button>
          <button
            aria-pressed={reviewMode}
            onClick={() => {
              setReviewMode(true);
              jump("safety");
            }}
          >
            {t("reviewReport")}
            <small>{t("reportOpenAgent")}</small>
          </button>
        </div>
      </header>
      <nav className="report-index" aria-label={t("reportContents")}>
        {reportSections.map(([anchor, key]) => (
          <button key={anchor} onClick={() => jump(anchor)}>
            {t(key)}
          </button>
        ))}
      </nav>
      <ErrorNotice error={error} />
      {notice && (
        <p className="report-notice" role="status">
          {t(notice)}
        </p>
      )}
      <div className="report-summary">
        <div>
          <strong>{objects.length}</strong>
          <span>{t("reportObjectRecords")}</span>
        </div>
        <div>
          <strong>
            {objects.filter((e) => (e.representations || []).length).length}
          </strong>
          <span>{t("reportSpatialObjects")}</span>
        </div>
        <div>
          <strong>
            {
              objects.filter((e) =>
                (e.representations || []).some((r) =>
                  ["generated_mesh", "primitive"].includes(r.kind),
                ),
              ).length
            }
          </strong>
          <span>{t("reportModelObjects")}</span>
        </div>
        <div>
          <strong>{doc.observations.length}</strong>
          <span>{t("reportPhotoEvidence")}</span>
        </div>
      </div>
      <section
        id="workcell-spatial"
        className="workcell-section report-visual-section"
      >
        <div className="report-section-heading">
          <div>
            <span className="report-kicker">01 / {t("reportReadOnly")}</span>
            <h2>{t("reportSpatial")}</h2>
          </div>
          <button
            onClick={() => {
              setReviewMode(true);
              setAgentOpen(true);
              setDraw(true);
              setBox(null);
              setSelection(s=>({...s,entityId:null,observationId:null}));
              jump("spatial");
            }}
          >
            {t("reportMissingObject")} ＋
          </button>
        </div>
        <p className="report-section-intro">{t("reportSpatialHint")}</p>
        {draw && <p className="report-notice">{t("reportDrawHint")}</p>}
        <ReportScene
          revision={revision}
          selection={selection}
          onSelect={select}
          imageId={imageId}
          cameraId={selection.cameraId}
          onCamera={changeCamera}
          draw={draw}
          onBox={(value) => {
            setBox(value);
            setDraw(false);
            setAgentOpen(true);
            setReviewMode(true);
          }}
        />
        <div className="report-selection-details">
          {entity ? (
            <ObjectFacts entity={entity} document={doc} />
          ) : (
            <p>{t("reportSelectDetails")}</p>
          )}
        </div>
        {reviewMode && (
          <div className="report-correction" id="report-correction">
            <div>
              <h3>{t("reportOpenAgent")}</h3>
              <p>{t("reportReviewHint")}</p>
              {box && (
                <p>
                  {t("selectedBox")}: {box.map(Math.round).join(", ")}
                </p>
              )}
              <button onClick={() => setAgentOpen((v) => !v)}>
                {t(agentOpen ? "reportCloseAgent" : "reportOpenAgent")}
              </button>
            </div>
            {agentOpen &&
              (canWrite ? (
                <AgentPanel
                  projectId={project.id}
                  revision={revision}
                  branch={detail.branch}
                  entityId={selection.entityId}
                  observationId={selection.observationId}
                  imageId={imageId}
                  box={box}
                  canWrite={!busy}
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
      </section>
      <section id="workcell-understanding" className="workcell-section">
        <div className="report-section-heading">
          <div>
            <span className="report-kicker">02 / {t("sourceEvidence")}</span>
            <h2>{t("reportUnderstanding")}</h2>
          </div>
          <input
            aria-label={t("search")}
            type="search"
            placeholder={t("search")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        <ReportEvidence
          document={doc}
          section="understanding"
          currentImageId={imageId}
          onSelect={select}
          onSelectEvidence={selectEvidence}
        />
        <p className="report-section-intro">{t("reportUnderstandingHint")}</p>
        <div className="report-table-scroll">
          <table className="report-inventory">
            <thead>
              <tr>
                <th>{t("objects")}</th>
                <th>{t("sourceEvidence")}</th>
                <th>{t("model")}</th>
                <th>{t("reportObservedExtent")}</th>
              </tr>
            </thead>
            <tbody>
              {(allObjects
                ? filteredObjects
                : filteredObjects.slice(0, 10)
              ).map((e) => (
                <tr
                  key={e.id}
                  className={selection.entityId === e.id ? "is-selected" : ""}
                >
                  <td>
                    <button
                      onClick={() => {
                        select(e.id);
                        jump("spatial");
                      }}
                    >
                      {e.label || e.id}
                      <span>↗</span>
                    </button>
                  </td>
                  <td>
                    {observationsFor(doc, e).length} {t("observations")}
                    <small>
                      {t(
                        e.associationState === "confirmed"
                          ? "confirmed"
                          : "pendingAssociation",
                      )}
                    </small>
                  </td>
                  <td>
                    {(e.representations || []).length
                      ? Array.from(
                          new Set(
                            (e.representations || []).map((r) => t(r.kind)),
                          ),
                        ).join(" · ")
                      : t("reportMissingGeometry")}
                  </td>
                  <td>
                    <Extent entity={e} document={doc} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {!objects.length && (
          <p className="report-notice">{t("reportNoAnalysis")}</p>
        )}
        {filteredObjects.length > 10 && (
          <button
            className="report-expand"
            onClick={() => setAllObjects((v) => !v)}
          >
            {t(allObjects ? "reportShowLess" : "reportShowAll")} ·{" "}
            {filteredObjects.length}
          </button>
        )}
        <p className="report-footnote">{t("reportAssociationHint")}</p>
      </section>
      <section id="workcell-safety" className="workcell-section">
        <div className="report-section-heading">
          <div>
            <span className="report-kicker">03 / EHS</span>
            <h2>{t("reportSafety")}</h2>
          </div>
          <button
            aria-pressed={reviewMode}
            onClick={() => setReviewMode((v) => !v)}
          >
            {t("reportDispute")}
          </button>
        </div>
        {reviewMode && !canWrite && (
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
          <a
            className="button"
            href={
              "#/projects/" + project.id + "/workbench?revision=" + revision.id
            }
          >
            {t("reportAdvanced")} ↗
          </a>
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
                    {e.operations.map((o) => t(o.type)).join(" · ") ||
                      t("checkpoint")}
                  </strong>
                  <span>
                    {e.baseRevisionId.slice(0, 8)} → {e.revisionId.slice(0, 8)}
                  </span>
                  <ReportDate value={e.createdAt} />
                  <details>
                    <summary>{t("reportTechnical")}</summary>
                    <pre>
                      {JSON.stringify(
                        { before: e.inverseOperations, after: e.operations },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
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
              href={"#/reports/" + p.id}
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
  );
}
function Extent({
  entity,
  document,
}: {
  entity: Entity;
  document: SceneDocument;
}) {
  const { t } = useI18n(),
    d = sourceDimensions(entity),
    scale = sourceScale(document, entity),
    values = [
      d.widthNative ?? d.extentX,
      d.depthNative ?? d.extentY,
      d.groundHeight ?? d.extentZ,
    ];
  return values.some(Number.isFinite) ? (
    <span className="report-numeric">
      {values
        .map((v) =>
          Number.isFinite(v)
            ? (v! * (scale?.nativeToMeters || 1)).toFixed(2)
            : "—",
        )
        .join(" × ")}
      <small>{scale?.nativeToMeters ? "m" : t("uncalibrated")}</small>
    </span>
  ) : (
    <span>—</span>
  );
}
function ObjectFacts({
  entity,
  document,
}: {
  entity: Entity;
  document: SceneDocument;
}) {
  const { t } = useI18n(),
    model = modelGeometry(entity),
    scale = sourceScale(document, entity),
    transform = editableTransform(entity),
    tilt = modelTilt(document, entity);
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
          {(entity.representations || []).length
            ? t("sourceEvidence")
            : t("reportMissingGeometry")}
        </small>
      </div>
      <dl className="report-measurements">
        <div>
          <dt>{t("reportObservedExtent")}</dt>
          <dd>
            <Extent entity={entity} document={document} />
          </dd>
        </div>
        <div>
          <dt>
            {t("reportCurrentModel")} · {t("width")} × {t("depth")} ×{" "}
            {t("height")}
          </dt>
          <dd>
            {[model?.width, model?.depth, model?.height].map(f).join(" × ")}
          </dd>
        </div>
        <div>
          <dt>{t("reportPosition")}</dt>
          <dd>{transform?.position.map(f).join(" / ") || "—"}</dd>
        </div>
        <div>
          <dt>{t("reportOrientation")}</dt>
          <dd>{tilt === null ? "—" : tilt.toFixed(1) + "°"}</dd>
        </div>
      </dl>
      <p className="report-footnote">{t("reportMeasurementNote")}</p>
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
