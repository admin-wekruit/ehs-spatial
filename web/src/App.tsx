import { CadView } from "./CadView";
import {
  createElement,
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import {
  asset,
  downloadJSON,
  exportCapability,
  id,
  importCapability,
  ownedIds,
  owner,
  pendingRequests,
  PUBLICATION_ID,
  prepareOwnedRequest,
  request,
  resolveAsset,
  sendOwnedRequest,
  ApiError,
  type PendingRequest,
} from "./api";
import {
  editableTransform,
  jobResultSummary,
  jsonObject,
  finiteNumber,
  eulerQuaternion,
  observationsFor,
  photoHits,
  previewOperations,
  quaternionEuler,
  activeModel,
  cameraForImage,
  currentCameras,
  currentEntityId,
  observationOwner,
  modelGeometry,
  modelTilt,
  planShapes,
  planHits,
  planPolygonPath,
  groupPublications,
  sourceDimensions,
  sourceScale,
  modelScale,
} from "./core";
import { useI18n } from "./i18n";
import { WorkcellReport } from "./WorkcellReport";
import { PhotoView } from "./PhotoView";
import { IdentityReview } from "./IdentityReview";
import { ModelEvidence } from "./ModelEvidence";
import { PrimitiveCreator } from "./PrimitiveCreator";
import { AgentPanel } from "./AgentPanel";
import type {
  EditRequest,
  Branch,
  Commit,
  EditBatch,
  Entity,
  Job,
  Operation,
  Project,
  ProjectDetail,
  Publication,
  PublicationSummary,
  Revision,
  SceneDocument,
  Selection,
  Transform,
  Vec3,
} from "./types";
import { mountSceneViewer } from "./viewer/native-viewer";
import { entityEvidenceStatus, isReferenceSurface } from "./scene-semantics";
import "./styles.css";
const PolicyPage = lazy(() => import("./PolicyPage"));
const path = (value: string) => "#" + value;
function navigate(value: string) {
  location.hash = value;
}
function useRoute() {
  const read = () => {
    const route = location.hash.slice(1);
    return PUBLICATION_ID && !/^\/reports(?:\/[^/?#]+)?(?:\?|$)/.test(route)
      ? "/reports/" + PUBLICATION_ID
      : route || "/projects";
  };
  const [value, set] = useState(read);
  useEffect(() => {
    const change = () => {
      const next = read();
      if (PUBLICATION_ID && location.hash.slice(1) !== next)
        window.history.replaceState(null, "", path(next));
      set(next);
    };
    change();
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  const [pathname, query = ""] = value.split("?");
  return { pathname, query: new URLSearchParams(query) };
}
export function ErrorNotice({ error }: { error: unknown }) {
  const { t } = useI18n();
  if (!error) return null;
  const e = error as Error;
  const key =
    error instanceof ApiError && error.status === 409
      ? "conflict"
      : error instanceof ApiError && error.status === 403
        ? "missingKey"
        : e.name === "TypeError"
          ? "networkError"
          : e.message;
  const message = t(key);
  return (
    <div role="alert" className="error-notice">
      {message === key &&
      !["networkError", "conflict", "missingKey"].includes(key) ? (
        <>
          {t("error")} <small>{key}</small>
        </>
      ) : (
        message
      )}
    </div>
  );
}
function Loading() {
  const { t } = useI18n();
  return (
    <div className="loading" role="status">
      <span className="loading-line" />
      {t("loading")}
    </div>
  );
}
function useResource<T>(key: string, loader: () => Promise<T>) {
  const [data, setData] = useState<T>(),
    [error, setError] = useState<unknown>(),
    [nonce, setNonce] = useState(0);
  useEffect(() => {
    let current = true;
    setError(undefined);
    loader()
      .then((value) => {
        if (current) setData(value);
      })
      .catch((e) => {
        if (current) setError(e);
      });
    return () => {
      current = false;
    };
  }, [key, nonce]);
  return { data, error, reload: () => setNonce((n) => n + 1), setData };
}
function AssetImage({
  assetId,
  alt,
  ...props
}: {
  assetId: string;
  alt: string;
  className?: string;
}) {
  const [url, set] = useState<string>();
  useEffect(() => {
    let active = true;
    resolveAsset(assetId)
      .then((u) => {
        if (active) set(u);
      })
      .catch(() => {});
    return () => {
      active = false;
    };
  }, [assetId]);
  return url ? (
    <img src={url} alt={alt} loading="lazy" {...props} />
  ) : (
    <span className="image-placeholder" aria-label={alt} />
  );
}
function Badge({ value }: { value: string }) {
  const { t } = useI18n();
  return <span className={"badge " + value.toLowerCase()}>{t(value)}</span>;
}
function DateLabel({ value }: { value: string }) {
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
export default function App() {
  const route = useRoute(),
    { t, language, setLanguage } = useI18n();
  const [importError, setImportError] = useState<unknown>();
  const parts = route.pathname.split("/").filter(Boolean);
  let content: ReactNode;
  if (parts[0] === "projects" && parts[1] === "new") content = <NewProject />;
  else if (parts[0] === "projects" && parts[1]) {
    const projectId = parts[1];
    content =
      parts[2] === "report" || !parts[2] ? (
        <WorkcellReport
          key={projectId + (route.query.get("revision") || "")}
          projectId={projectId}
          requestedRevision={route.query.get("revision")}
        />
      ) : parts[2] === "history" ? (
        <History projectId={projectId} />
      ) : parts[2] === "safety" ? (
        <Suspense fallback={<Loading />}>
          <PolicyPage
            projectId={projectId}
            revisionId={route.query.get("revision") || undefined}
          />
        </Suspense>
      ) : (
        <Workbench
          key={projectId}
          projectId={projectId}
          requestedRevision={route.query.get("revision")}
          requestedBranch={route.query.get("branch")}
        />
      );
  } else if (parts[0] === "reports" && parts[1])
    content = <WorkcellReport key={parts[1]} publicationId={parts[1]} />;
  else if (parts[0] === "reports") content = <ReportLibrary />;
  else if (parts[0] === "policies")
    content = (
      <Suspense fallback={<Loading />}>
        <PolicyPage policyId={parts[1]} />
      </Suspense>
    );
  else content = <ProjectLibrary />;
  return (
    <>
      <a className="skip-link" href="#main-content">
        {t("open")}
      </a>
      <header className="app-header">
        <a href={path(PUBLICATION_ID ? "/reports/" + PUBLICATION_ID : "/projects")} className="brand">
          <span className="brand-mark">P</span>
          <span>
            PANOPTES<small>SPATIAL STUDIO</small>
          </span>
        </a>
        <nav aria-label="Primary">
          {(PUBLICATION_ID ? ["reports"] : ["projects", "reports"]).map((item) => (
            <a
              key={item}
              href={path("/" + item)}
              aria-current={parts[0] === item ? "page" : undefined}
            >
              {t(item)}
            </a>
          ))}
        </nav>
        <div className="header-actions">
          {!PUBLICATION_ID && <label className="key-import">
            {t("importKey")}
            <input
              type="file"
              accept="application/json"
              onChange={async (e) => {
                try {
                  if (e.target.files?.[0]) {
                    const pid = await importCapability(e.target.files[0]);
                    navigate("/projects/" + pid + "/workbench");
                    location.reload();
                  }
                } catch (error) {
                  setImportError(error);
                }
              }}
            />
          </label>}
          <select
            aria-label="Language"
            value={language}
            onChange={(e) => setLanguage(e.target.value as "zh" | "en")}
          >
            <option value="zh">中文</option>
            <option value="en">EN</option>
          </select>
        </div>
      </header>
      <ErrorNotice error={importError} />
      <main id="main-content">{content}</main>
    </>
  );
}

function ProjectLibrary() {
  const { t } = useI18n(),
    { data, error, reload } = useResource<{
      items: Project[];
      publications: PublicationSummary[];
    }>("projects", async () => {
      const [projects, reports] = await Promise.all([
        request<{ items: Project[] }>("/api/projects"),
        request<{ items: PublicationSummary[] }>("/api/publications"),
      ]);
      return {
        ...projects,
        publications: groupPublications(reports.items).map((group) => group[0]),
      };
    });
  const [filter, setFilter] = useState(""),
    [mine, setMine] = useState(false),
    [owned, setOwned] = useState<Set<string>>(new Set()),
    [pending, setPending] = useState<PendingRequest[]>([]),
    [actionError, setActionError] = useState<unknown>();
  useEffect(() => {
    ownedIds().then(setOwned).catch(setActionError);
    pendingRequests().then(setPending).catch(setActionError);
  }, []);
  async function resume(p: PendingRequest) {
    try {
      const result = await sendOwnedRequest<ProjectDetail>(p);
      navigate("/projects/" + result.project.id + "/report");
    } catch (e) {
      setActionError(e);
    }
  }
  return (
    <section className="library page-width">
      <div className="page-intro">
        <p className="eyebrow">PHOTOGRAPHS → SPATIAL MODELS</p>
        <div className="title-row">
          <h1>{t("studio")}</h1>
          <a className="button primary" href={path("/projects/new")}>
            {t("newProject")} <span>↗</span>
          </a>
        </div>
        <p className="lede">{t("intro")}</p>
        <p className="subtle">{t("allPublic")}</p>
      </div>
      <ErrorNotice error={error || actionError} />
      {pending.length > 0 && (
        <section className="pending-requests">
          <h2>{t("retryPending")}</h2>
          <p>{t("retryPendingCopy")}</p>
          {pending.map((p) => (
            <div key={p.id}>
              <span>{String(p.body.title || p.id)}</span>
              <button onClick={() => resume(p)}>{t("retry")}</button>
            </div>
          ))}
        </section>
      )}
      <div className="list-toolbar">
        <h2>{t("projectLibrary")}</h2>
        <input
          type="search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder={t("search")}
          aria-label={t("search")}
        />
        <label>
          <input
            type="checkbox"
            checked={mine}
            onChange={(e) => setMine(e.target.checked)}
          />
          {t("owned")}
        </label>
        <button onClick={reload}>{t("refresh")}</button>
      </div>
      {!data && !error ? (
        <Loading />
      ) : data?.items.length ? (
        <div className="project-list">
          {data.items
            .filter(
              (p) =>
                (!mine || owned.has(p.id)) &&
                p.title.toLowerCase().includes(filter.toLowerCase()),
            )
            .map((project, i) => {
              const publication = data.publications.find(
                (p) => p.projectId === project.id,
              );
              return (
                <a
                  className="project-row project-report-row"
                  key={project.id}
                  href={path(
                    publication
                      ? "/reports/" + publication.id
                      : "/projects/" + project.id + "/report",
                  )}
                >
                  {publication?.previewImageAssetId ? (
                    <AssetImage
                      assetId={publication.previewImageAssetId}
                      alt={project.title}
                      className="project-cover"
                    />
                  ) : (
                    <span className="project-number">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                  )}
                  <div>
                    <h3>{project.title}</h3>
                    <small>
                      {owned.has(project.id) ? t("owned") : t("browse")}
                    </small>
                    {publication && (
                      <p className="project-content-summary">
                        {publication.photoCount} {t("photos")} ·{" "}
                        {publication.objectCount} {t("reportObjectRecords")} ·{" "}
                        {publication.modelObjectCount} {t("reportModelObjects")}{" "}
                        · {t("version")}{" "}
                        {publication.sceneRevisionId.slice(0, 8)}
                      </p>
                    )}
                  </div>
                  <DateLabel value={project.createdAt} />
                  <span className="row-arrow">↗</span>
                </a>
              );
            })}
        </div>
      ) : (
        <div className="empty-state">
          <span className="large-symbol">＋</span>
          <p>{t("noProjects")}</p>
          <a href={path("/projects/new")} className="button primary">
            {t("choosePhotos")}
          </a>
        </div>
      )}
    </section>
  );
}

function NewProject() {
  const { t } = useI18n(),
    [title, setTitle] = useState(""),
    [target, setTarget] = useState<"scene" | "standalone_object">("scene"),
    [photos, setPhotos] = useState<File[]>([]),
    [previews, setPreviews] = useState<string[]>([]),
    [error, setError] = useState<unknown>(),
    [phase, setPhase] = useState(""),
    [created, setCreated] = useState<ProjectDetail>();
  const captureRequest = useRef(id());
  useEffect(() => {
    const urls = photos.map(URL.createObjectURL);
    setPreviews(urls);
    return () => urls.forEach(URL.revokeObjectURL);
  }, [photos]);
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(undefined);
    if (!title.trim()) {
      setError(new Error("nameRequired"));
      return;
    }
    if (photos.length < 1 || photos.length > 4) {
      setError(new Error("photoLimit"));
      return;
    }
    try {
      setPhase("creating");
      let detail = created;
      if (!detail) {
        const pending = await prepareOwnedRequest("/api/projects", {
          title: title.trim(),
          target,
        });
        detail = await sendOwnedRequest<ProjectDetail>(pending);
        setCreated(detail);
      }
      setPhase("uploading");
      const data = new FormData();
      photos.forEach((file) => data.append("files", file));
      data.set("requestId", captureRequest.current);
      data.set("branchId", detail.branch.id);
      data.set("baseRevisionId", detail.revision.id);
      data.set("target", target);
      data.set("captureMode", "initial");
      await request("/api/projects/" + detail.project.id + "/captures", {
        method: "POST",
        projectId: detail.project.id,
        body: data,
      });
      navigate("/projects/" + detail.project.id + "/report");
    } catch (e) {
      setError(e);
    } finally {
      setPhase("");
    }
  }
  return (
    <section className="new-project page-width narrow">
      <p className="eyebrow">NEW CAPTURE</p>
      <h1>{t("newProject")}</h1>
      <form onSubmit={submit}>
        <label className="field-label">
          {t("title")}
          <input
            required
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            maxLength={200}
          />
        </label>
        <fieldset className="target-options">
          <legend>{t("target")}</legend>
          {(["scene", "standalone_object"] as const).map((value) => (
            <label
              className={
                target === value ? "target-option active" : "target-option"
              }
              key={value}
            >
              <input
                type="radio"
                name="target"
                value={value}
                checked={target === value}
                onChange={() => setTarget(value)}
              />
              <strong>{t(value)}</strong>
              <span>{t(value === "scene" ? "sceneHint" : "objectHint")}</span>
            </label>
          ))}
        </fieldset>
        <label className="upload-zone">
          <span>＋</span>
          <strong>{t("choosePhotos")}</strong>
          <small>{t("photoHint")}</small>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp"
            multiple
            onChange={(e) => {
              const files = Array.from(e.target.files || []);
              if (files.length > 4) {
                setError(new Error("photoLimit"));
                return;
              }
              setPhotos(files);
              setError(undefined);
            }}
          />
        </label>
        <div className="upload-previews">
          {previews.map((url, i) => (
            <figure key={url}>
              <img src={url} alt={`${t("photo")} ${i + 1}`} />
              <figcaption>
                <span>{photos[i]?.name}</span>
                <button
                  type="button"
                  onClick={() =>
                    setPhotos((items) => items.filter((_, n) => i !== n))
                  }
                  aria-label={t("remove")}
                >
                  ×
                </button>
              </figcaption>
            </figure>
          ))}
        </div>
        <p className="subtle">{t("publicUpload")}</p>
        <ErrorNotice error={error} />
        <button className="primary" disabled={!!phase}>
          {t(phase || "createAnalyze")} →
        </button>
      </form>
    </section>
  );
}

function ProjectNav({
  projectId,
  active,
  revisionId,
}: {
  projectId: string;
  active: string;
  revisionId?: string;
}) {
  const { t } = useI18n();
  return (
    <nav className="project-nav">
      {["report", "workbench", "history"].map((tab) => (
        <a
          key={tab}
          href={path(
            `/projects/${projectId}/${tab}${tab !== "history" && revisionId ? "?revision=" + revisionId : ""}`,
          )}
          aria-current={active === tab ? "page" : undefined}
        >
          {t(tab === "report" ? "workcellReport" : tab)}
        </a>
      ))}
    </nav>
  );
}
function Workbench({
  projectId,
  requestedRevision,
  requestedBranch,
}: {
  projectId: string;
  requestedRevision: string | null;
  requestedBranch: string | null;
}) {
  const { t } = useI18n(),
    resource = useResource<ProjectDetail>(projectId, () =>
      request("/api/projects/" + projectId),
    );
  const [detail, setDetail] = useState<ProjectDetail>(),
    [canWrite, setCanWrite] = useState(false),
    [error, setError] = useState<unknown>(),
    [status, setStatus] = useState("saved"),
    [operations, setOperations] = useState<Operation[]>([]),
    [undo, setUndo] = useState<string[]>([]),
    [redo, setRedo] = useState<string[]>([]),
    [action, setAction] = useState(""),
    [name, setName] = useState("");
  const inFlight = useRef(false),
    pendingCommit = useRef<{
      ops: Operation[];
      extra: Record<string, unknown>;
    } | null>(null);
  useEffect(() => {
    owner(projectId)
      .then((cap) => setCanWrite(!!cap))
      .catch(setError);
  }, [projectId]);
  useEffect(() => {
    let active = true;
    const data = resource.data;
    if (data) {
      setDetail((previous) => {
        const branch =
          data.branches.find(
            (b) => b.id === (requestedBranch || previous?.branch.id),
          ) || data.branch;
        const target = requestedRevision || branch.headRevisionId;
        if (target === data.revision.id) return { ...data, branch };
        request<Revision>("/api/revisions/" + target)
          .then((revision) => {
            if (active)
              setDetail({
                ...data,
                revision,
                branch:
                  data.branches.find((b) => b.id === revision.branchId) ||
                  branch,
              });
          })
          .catch(setError);
        return previous || data;
      });
    }
    return () => {
      active = false;
    };
  }, [resource.data, requestedRevision, requestedBranch]);
  const commit = async (
    ops: Operation[],
    extra: Record<string, unknown> = {},
  ) => {
    if (!detail || inFlight.current) return;
    if (!canWrite) {
      setOperations((old) => [...old, ...ops]);
      setStatus("temporary");
      return;
    }
    inFlight.current = true;
    setStatus("saving");
    setError(undefined);
    setOperations(ops);
    const requestId = String(extra.requestId || id());
    pendingCommit.current = { ops, extra: { ...extra, requestId } };
    try {
      const editBody: EditRequest = {
        requestId,
        branchId: detail.branch.id,
        baseRevisionId: detail.revision.id,
        operations: ops,
        ...extra,
      };
      const result = await request<Commit>(
        "/api/projects/" + projectId + "/edits",
        { method: "POST", projectId, body: editBody },
      );
      setDetail({
        ...detail,
        revision: result.revision,
        branch: { ...detail.branch, headRevisionId: result.revision.id },
      });
      setOperations([]);
      if (!extra.undoOf && !extra.redoOf) {
        setUndo((list) => [...list, result.editBatch.id]);
        setRedo([]);
      }
      pendingCommit.current = null;
      setStatus("saved");
      return result;
    } catch (e) {
      setError(e);
      setStatus("saveFailed");
      return false;
    } finally {
      inFlight.current = false;
    }
  };
  async function undoRedo(kind: "undo" | "redo") {
    const stack = kind === "undo" ? undo : redo;
    const batch = stack.at(-1);
    if (!batch) return;
    const result = await commit([], { [kind + "Of"]: batch });
    if (result) {
      if (kind === "undo") {
        setUndo((s) => s.slice(0, -1));
        setRedo((s) => [...s, batch]);
      } else {
        setRedo((s) => s.slice(0, -1));
        setUndo((s) => [...s, result.editBatch.id]);
      }
    }
  }
  async function chooseBranch(branch: Branch) {
    if (!detail) return;
    if (pendingCommit.current) {
      setError(new Error("pendingSave"));
      return;
    }
    try {
      const revision = await request<Revision>(
        "/api/revisions/" + branch.headRevisionId,
      );
      setDetail({ ...detail, branch, revision });
      navigate(`/projects/${projectId}/workbench?branch=${branch.id}`);
      setOperations([]);
      setUndo([]);
      setRedo([]);
    } catch (e) {
      setError(e);
    }
  }
  async function submitAction(e: React.FormEvent) {
    e.preventDefault();
    if (!detail) return;
    try {
      if (action === "copy") {
        const pending = await prepareOwnedRequest(
          `/api/projects/${projectId}/forks`,
          {
            sourceRevisionId: detail.revision.id,
            title: name || detail.project.title + " · " + t("copy"),
            operations,
          },
        );
        const copy = await sendOwnedRequest<ProjectDetail>(pending);
        navigate(`/projects/${copy.project.id}/workbench`);
      } else if (action === "planning") {
        const branch = await request<Branch>(
          `/api/projects/${projectId}/branches`,
          {
            method: "POST",
            projectId,
            body: {
              requestId: id(),
              sourceRevisionId: detail.revision.id,
              title: name,
              kind: "planning",
            },
          },
        );
        const nextRevision = await request<Revision>(
          "/api/revisions/" + branch.headRevisionId,
        );
        setDetail({
          ...detail,
          branches: [...detail.branches, branch],
          branch,
          revision: nextRevision,
        });
        navigate(`/projects/${projectId}/workbench?branch=${branch.id}`);
        setOperations([]);
      } else if (action === "checkpoint") await commit([], { label: name });
      else if (action === "publish") {
        const reviews = await request<{ items: any[] }>(
          `/api/projects/${projectId}/reviews`,
        );
        const evaluations = await request<{ items: any[] }>(
          `/api/projects/${projectId}/evaluations`,
        );
        const publication = await request<Publication>(
          `/api/projects/${projectId}/publications`,
          {
            method: "POST",
            projectId,
            body: {
              requestId: id(),
              sceneRevisionId: detail.revision.id,
              title: name || detail.project.title,
              evaluationIds: evaluations.items
                .filter((e) => e.sceneRevisionId === detail.revision.id)
                .map((e) => e.id),
              reviewIds: reviews.items
                .filter((r) =>
                  evaluations.items.some(
                    (e) =>
                      e.id === r.evaluationId &&
                      e.sceneRevisionId === detail.revision.id,
                  ),
                )
                .map((r) => r.id),
            },
          },
        );
        navigate("/reports/" + publication.id);
      }
      setAction("");
      setName("");
    } catch (e) {
      setError(e);
    }
  }
  async function job(kind: string, entityIds: string[] = []) {
    if (!detail) return;
    try {
      await request(`/api/projects/${projectId}/jobs`, {
        method: "POST",
        projectId,
        body: {
          requestId: id(),
          branchId: detail.branch.id,
          baseRevisionId: detail.revision.id,
          kind,
          inputs: { entityIds },
          config: {},
        },
      });
      setStatus("queued");
    } catch (e) {
      setError(e);
    }
  }
  if (!detail)
    return (
      <>
        <ErrorNotice error={resource.error} />
        <Loading />
      </>
    );
  const revision = operations.length
    ? {
        ...detail.revision,
        document: previewOperations(detail.revision.document, operations),
      }
    : detail.revision;
  return (
    <section className="workbench">
      <div className="workbench-heading">
        <div>
          <a className="breadcrumb" href={path("/projects")}>
            {t("projects")} /
          </a>
          <p className="eyebrow">{t("workbench")}</p>
          <h1>{detail.project.title}</h1>
        </div>
        <ProjectNav
          projectId={projectId}
          active="workbench"
          revisionId={detail.revision.id}
        />
        <div className="workbench-actions">
          <span className={"save-state " + status} role="status">
            {t(status)}
          </span>
          {status === "saveFailed" && pendingCommit.current && (
            <button
              onClick={() =>
                commit(pendingCommit.current!.ops, pendingCommit.current!.extra)
              }
            >
              {t("retry")}
            </button>
          )}
          <button
            disabled={!canWrite || !undo.length || status === "saving"}
            onClick={() => undoRedo("undo")}
            title={t("undo")}
          >
            ↶
          </button>
          <button
            disabled={!canWrite || !redo.length || status === "saving"}
            onClick={() => undoRedo("redo")}
            title={t("redo")}
          >
            ↷
          </button>
          {canWrite ? (
            <>
              <button
                onClick={() =>
                  setAction(action === "checkpoint" ? "" : "checkpoint")
                }
              >
                {t("checkpoint")}
              </button>
              <button
                className="primary"
                onClick={() => setAction(action === "publish" ? "" : "publish")}
              >
                {t("publish")}
              </button>
            </>
          ) : (
            <button className="primary" onClick={() => setAction("copy")}>
              {t("copy")}
            </button>
          )}
        </div>
      </div>
      <div className="branch-bar">
        <Badge value={detail.branch.kind} />
        <select
          aria-label={t("branch")}
          value={detail.branch.id}
          onChange={(e) => {
            const branch = detail.branches.find((b) => b.id === e.target.value);
            if (branch) void chooseBranch(branch);
          }}
        >
          {detail.branches.map((b) => (
            <option key={b.id} value={b.id}>
              {b.title}
            </option>
          ))}
        </select>
        <span className="subtle">
          {t("version")}{" "}
          {detail.revision.label || detail.revision.id.slice(0, 8)}
        </span>
        {canWrite && (
          <button
            onClick={() => setAction(action === "planning" ? "" : "planning")}
          >
            ＋ {t("newPlan")}
          </button>
        )}
        <div className="branch-end">
          {canWrite && (
            <button onClick={() => exportCapability(projectId).catch(setError)}>
              {t("manageKey")}
            </button>
          )}
          <select
            aria-label={t("export")}
            value=""
            onChange={(e) => {
              if (e.target.value === "json")
                downloadJSON(
                  detail.revision.document,
                  "scene-" + detail.revision.id + ".json",
                );
              else if (e.target.value) void job(e.target.value);
            }}
          >
            <option value="">{t("export")} ↗</option>
            <option value="json">Scene JSON</option>
            <option value="export_glb">GLB</option>
            <option value="export_blender">Blender</option>
          </select>
        </div>
      </div>
      {!canWrite && <p className="visitor-note">{t("temporaryCopy")}</p>}
      {action && (
        <form className="inline-action" onSubmit={submitAction}>
          <label>
            {t(
              action === "planning"
                ? "namePlan"
                : action === "copy"
                  ? "copyTitle"
                  : action === "checkpoint"
                    ? "nameCheckpoint"
                    : "title",
            )}
            <input
              autoFocus
              required={action !== "copy"}
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </label>
          <button className="primary">
            {t(
              action === "planning"
                ? "newPlan"
                : action === "checkpoint"
                  ? "checkpoint"
                  : action,
            )}
          </button>
          <button type="button" onClick={() => setAction("")}>
            ×
          </button>
        </form>
      )}
      <ErrorNotice error={error} />
      <Workspace
        revision={revision}
        project={detail.project}
        branch={detail.branch}
        canWrite={canWrite}
        busy={status === "saving" || status === "saveFailed"}
        onCommit={commit}
        onJob={job}
      />
      <TaskStrip projectId={projectId} onComplete={() => resource.reload()} />
    </section>
  );
}

type ViewerMode = "photo" | "free" | "top" | "front" | "side";
export function SpatialView({
  revision,
  selection,
  onSelect,
  onCommit,
  mode,
  cameraId,
  layers,
}: {
  revision: Revision;
  selection: Selection;
  onSelect: (id: string, observationId?: string) => void;
  onCommit: (operations: Operation[]) => unknown;
  mode: ViewerMode;
  cameraId: string | null;
  layers: Record<string, any>;
}) {
  const host = useRef<HTMLDivElement>(null),
    runtime = useRef<any>(null),
    callbacks = useRef({ onSelect, onCommit, revision }),
    { language, t } = useI18n(),
    [status, setStatus] = useState("loadingModel"),
    [error, setError] = useState<unknown>();
  callbacks.current = { onSelect, onCommit, revision };
  useEffect(() => {
    if (!host.current) return;
    let viewer: ReturnType<typeof mountSceneViewer>;
    try {
      viewer = mountSceneViewer(host.current, {
        resolveAsset,
        locale: language,
        onEvent: (event: any) => {
          if (event.type === "selectionIntent") {
            const d = callbacks.current.revision.document,
              c = d.cameras.find((c) => c.id === event.cameraId);
            const hit =
              c && event.originalPixel
                ? photoHits(
                    d,
                    c.imageId,
                    event.originalPixel[0] + 0.5,
                    event.originalPixel[1] + 0.5,
                  )[0]
                : null;
            if (hit)
              callbacks.current.onSelect(hit.entity.id, hit.observation.id);
            else if (event.entityId)
              callbacks.current.onSelect(event.entityId, event.observationId);
          } else if (event.type === "transformCommitIntent" && event.operations)
            callbacks.current.onCommit(event.operations);
          else if (event.type === "loadError" || event.type === "contextLost")
            setError(new Error(event.message || event.code || "error"));
          else if (event.type === "renderReady") setStatus("");
          else if (event.type === "loadProgress")
            setStatus(event.message || "loadingModel");
        },
      });
    } catch (error) {
      setError(error);
      setStatus("");
      return;
    }
    runtime.current = viewer;
    return () => {
      runtime.current = undefined;
      viewer.dispose();
    };
  }, []);
  useEffect(() => {
    setError(undefined);
    Promise.resolve(runtime.current?.setScene(revision)).catch(setError);
  }, [revision]);
  useEffect(() => {
    runtime.current?.setSelection(selection);
  }, [selection.entityId, selection.observationId, selection.revisionId]);
  useEffect(() => {
    runtime.current?.setCamera({ mode, cameraId });
  }, [mode, cameraId]);
  useEffect(() => {
    runtime.current?.setLayers(layers);
  }, [JSON.stringify(layers)]);
  return (
    <div className="spatial-view">
      <div ref={host} className="native-viewer" />
      {mode !== "photo" && (
        <button
          className="spatial-fit"
          onClick={() => runtime.current?.setCamera({ mode, cameraId })}
        >
          {t("fit")}
        </button>
      )}
      {status && (
        <p className="stage-status" role="status">
          {t(status)}
        </p>
      )}
      <ErrorNotice error={error} />
    </div>
  );
}

function Workspace({
  revision,
  project,
  branch,
  canWrite,
  busy = false,
  onCommit,
  onJob,
  report = false,
}: {
  revision: Revision;
  project: Project;
  branch: Pick<Branch, "id">;
  canWrite: boolean;
  busy?: boolean;
  onCommit: (
    ops: Operation[],
    context?: { baseRevisionId: string; agentTurnId?: string },
  ) => unknown;
  onJob?: (kind: string, entityIds?: string[]) => unknown;
  report?: boolean;
}) {
  const document = revision.document,
    objects = document.entities.filter(entity => !entity.sourceContext),
    { t } = useI18n();
  const query = new URLSearchParams(location.hash.split("?")[1] || ""),
    [requestedEntityId, setSelected] = useState<string | null>(query.get("object")),
    [requestedObservationId, setObservation] = useState<string | null>(query.get("observation")),
    [cameraId, setCamera] = useState<string | null>(
      cameraForImage(document, query.get("image"))?.id || currentCameras(document)[0]?.id || null,
    ),
    [sourceImageId, setSourceImage] = useState<string | null>(document.assets.find(a => a.id === query.get("image") && a.kind === "source_image")?.id || document.observations.find(o=>o.id===query.get("observation"))?.imageId || document.assets.find(a=>a.kind==="source_image")?.id || null),
    [mode, setMode] = useState<ViewerMode>("photo"),
    [four, setFour] = useState(false),
    [panel, setPanel] = useState<"properties" | "agent">("properties"),
    [identityEntityIds, setIdentityEntityIds] = useState<[string,string]>(),
    [draw, setDraw] = useState(false),
    [box, setBox] = useState<number[] | null>(null),
    [allBounds, setAllBounds] = useState(false),
    [opacity, setOpacity] = useState(0.45),
    [representation, setRepresentation] = useState("model"),
    [search, setSearch] = useState(""),
    [creatingModel, setCreatingModel] = useState(false),
    [appending, setAppending] = useState(false), [appendFiles, setAppendFiles] = useState<File[]>([]), [captureError, setCaptureError] = useState<unknown>();
  const appendRequest = useRef({id:id(),baseRevisionId:revision.id});
  const selectionEpoch = useRef(0);
  const selectedId = requestedEntityId ? currentEntityId(document, requestedEntityId) : null,
    observationId = selectedId && observationOwner(document, requestedObservationId || "")?.id === selectedId ? requestedObservationId : null,
    selected = objects.find((e) => e.id === selectedId) || null,
    camera =
      currentCameras(document).find((c) => c.id === cameraId),
    imageId =
      sourceImageId ||
      document.observations.find((o) => o.id === observationId)?.imageId ||
      camera?.imageId ||
      document.observations[0]?.imageId ||
      document.assets.find((a) => a.kind === "source_image")?.id ||
      null;
  useEffect(() => {
    // Resolve before rendering children; normalization must not overwrite a newer click.
    setSelected(current => current === requestedEntityId ? selectedId : current);
    setObservation(current => current === requestedObservationId ? observationId : current);
  }, [requestedEntityId, requestedObservationId, selectedId, observationId]);
  useEffect(() => {
    const next = cameraForImage(document, imageId)?.id || null;
    if (next !== cameraId) setCamera(next);
  }, [document.geometryBindings, document.cameras, imageId, cameraId]);
  function select(entityId: string, obsId?: string) {
    const currentId = currentEntityId(document, entityId);
    selectionEpoch.current++;
    setSelected(currentId);
    setBox(null);
    const entity = document.entities.find((e) => e.id === currentId);
    const observations = entity ? observationsFor(document, entity) : [];
    const next =
      observations.find((o) => o.id === obsId) ||
      observations.find((o) => o.imageId === imageId) ||
      observations[0];
    setObservation(next?.id || null);
    if (next && next.imageId !== imageId) {
      setSourceImage(next.imageId);setCamera(cameraForImage(document, next.imageId)?.id || null);
    }
    const url = new URL(location.href);
    const [hashPath, hashQuery = ""] = url.hash.slice(1).split("?");
    const params = new URLSearchParams(hashQuery);
    if (currentId) params.set("object", currentId);
    else params.delete("object");
    history.replaceState(null, "", "#" + hashPath + "?" + params);
  }
  const selection: Selection = {
      projectId: project.id,
      revisionId: revision.id,
      entityId: selectedId,
      observationId,
      cameraId: camera?.id || null,
    },
    layers = {
      observed_surface: representation !== "point_cloud",
      generated_mesh: representation === "model",
      primitive: representation === "model",
      point_cloud: representation === "point_cloud",
      imageId, observations: document.observations,
      allBounds,
      showBounds: allBounds,
      opacity,
      showCandidates: !report,
      editable: !busy,
    };
  return (
    <div className={"workspace-layout " + (report ? "report-workspace" : "")}>
      <aside className="entity-panel">
        <div className="panel-title">
          <h2>{t("objects")}</h2>
          <span>{objects.length}</span>
          {!busy && (
            <button
              className="icon-button"
              onClick={() => setCreatingModel(true)}
              aria-label={t("addModel")}
            >
              ＋
            </button>
          )}
        </div>
        <input
          className="entity-search"
          type="search"
          placeholder={t("search")}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="entity-list">
          {objects
            .filter((e) =>
              (e.label || e.id).toLowerCase().includes(search.toLowerCase()),
            )
            .map((entity) => (
              <div
                className={
                  "entity-row " + (entity.id === selectedId ? "selected" : "")
                }
                key={entity.id}
              >
                <button
                  onClick={() => select(entity.id)}
                  aria-pressed={entity.id === selectedId}
                >
                  <strong>{entity.label || entity.id}</strong>
                  <span>
                    {observationsFor(document, entity).length}{" "}
                    {t("observations")} ·{" "}
                    {t(entityEvidenceStatus(document, entity).modelKey)}
                  </span>
                </button>
                <button className="entity-feedback" aria-label={`${t("sceneFeedback")} · ${entity.label || entity.id}`}
                  onClick={() => { select(entity.id); setCreatingModel(false); setPanel("agent"); }}>{t("sceneFeedback")}</button>
                <input
                  type="checkbox"
                  aria-label={t("visible") + " " + (entity.label || entity.id)}
                  checked={entity.visible !== false}
                  disabled={busy}
                  onChange={(e) =>
                    onCommit([
                      {
                        type: "setVisibility",
                        entityId: entity.id,
                        visible: e.target.checked,
                      },
                    ])
                  }
                />
              </div>
            ))}
          {!objects.length && (
            <p className="empty-copy">{t("noObjects")}</p>
          )}
        </div>
        {canWrite && !report && <div className="capture-append"><label>{t("identityAddPhotos")}
          <input type="file" accept="image/*" multiple disabled={appending||busy} onChange={event=>{
            const files=Array.from(event.target.files||[]);
            if(files.length>4){setCaptureError(Error("photoLimit"));return;}
            setAppendFiles(files);appendRequest.current={id:id(),baseRevisionId:revision.id};setCaptureError(undefined);
          }}/></label>
          {!!appendFiles.length && <button disabled={appending||busy} onClick={async()=>{
            setAppending(true);setCaptureError(undefined);
            try { const data=new FormData();appendFiles.forEach(file=>data.append("files",file));data.set("requestId",appendRequest.current.id);data.set("branchId",branch.id);data.set("baseRevisionId",appendRequest.current.baseRevisionId);data.set("target",document.target);data.set("captureMode","append");
              await request("/api/projects/"+project.id+"/captures",{method:"POST",projectId:project.id,body:data});
              navigate("/projects/"+project.id+"/report");
            }catch(error){setCaptureError(error);}finally{setAppending(false);}
          }}>{t(appending?"uploading":captureError?"retry":"identityAddPhotos")} · {appendFiles.length}</button>}
          <ErrorNotice error={captureError}/>
        </div>}
        <div className="photo-strip">
          {document.assets.filter(asset=>asset.kind==="source_image").map((asset,i)=>(
            <button key={asset.id} onClick={()=>{setSourceImage(asset.id);setCamera(cameraForImage(document,asset.id)?.id||null);setMode("photo");}} className={asset.id===(camera?.imageId||sourceImageId)?"selected":""} aria-label={`${t("photo")} ${i+1}`}>
              <AssetImage assetId={asset.id} alt={`${t("photo")} ${i+1}`}/><span>{i+1}</span>
            </button>
          ))}
        </div>
      </aside>
      <section className="canvas-section">
        <div className="canvas-toolbar">
          <div className="mode-switch">
            <button
              aria-pressed={mode === "photo" && !four}
              onClick={() => {
                setMode("photo");
                setFour(false);
              }}
            >
              {t("photo")}
            </button>
            <button
              aria-pressed={mode === "free" && !four}
              onClick={() => {
                setMode("free");
                setFour(false);
              }}
            >
              {t("free")}
            </button>
            <button aria-pressed={four} onClick={() => setFour(!four)}>
              {t("four")}
            </button>
          </div>
          <select
            aria-label={t("model")}
            value={representation}
            onChange={(e) => setRepresentation(e.target.value)}
          >
            <option value="model">{t("model")}</option>
            <option value="observed_surface">{t("observed_surface")}</option>
            <option value="point_cloud">{t("point_cloud")}</option>
          </select>
          <label className="compact-check">
            <input
              type="checkbox"
              checked={allBounds}
              onChange={(e) => setAllBounds(e.target.checked)}
            />
            {t("sceneShowBorders")}
          </label>
          <select
            aria-label={t("camera")}
            value={mode}
            onChange={(e) => {
              setMode(e.target.value as ViewerMode);
              setFour(false);
            }}
          >
            {(["photo", "free", "top", "front", "side"] as const).map((m) => (
              <option key={m} value={m}>
                {t(m)}
              </option>
            ))}
          </select>
          {mode === "photo" && (
            <label className="opacity-label">
              {t("opacity")}
              <input
                type="range"
                min="0"
                max="1"
                step=".05"
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
            </label>
          )}
        </div>
        <div className={four ? "canvas-grid four" : "canvas-grid"}>
          {four && (
            <div className="canvas-pane">
              <span className="pane-label">01 {t("sourceEvidence")}</span>
              <PhotoView
                document={document}
                imageId={imageId}
                selectedId={selectedId}
                onSelect={select}
                onBox={setBox}
                draw={draw}
                showBounds={allBounds}
              />
            </div>
          )}
          <div className="canvas-pane main-canvas">
            <span className="pane-label">
              {four ? "02 " : ""}
              {t(mode)}
            </span>
            {mode === "photo" && !four && (
              <div
                className={
                  "photo-backdrop " +
                  (draw ? "drawing" : "") +
                  (!camera ? " source-fallback" : "")
                }
              >
                <PhotoView
                  document={document}
                  imageId={imageId}
                  selectedId={selectedId}
                  onSelect={select}
                  onBox={setBox}
                  draw={draw}
                  showBounds={allBounds}
                />
              </div>
            )}
            <div
              className={
                mode === "photo" && !four ? "scene-overlay" : "scene-solid"
              }
              style={
                mode === "photo" && !four
                  ? { pointerEvents: draw || !camera ? "none" : "auto" }
                  : undefined
              }
            >
              {(mode !== "photo" || !!camera) && (
                <SpatialView
                  revision={revision}
                  selection={selection}
                  onSelect={select}
                  onCommit={onCommit}
                  mode={mode}
                  cameraId={camera?.id || null}
                  layers={layers}
                />
              )}
            </div>
            {selected && !(selected.representations || []).length && (
              <div className="selection-status">
                <strong>{selected.label || selected.id}</strong>
                <span>{t("noGeometry")}</span>
              </div>
            )}
          </div>
          {four && (
            <>
              <div className="canvas-pane">
                <span className="pane-label">03 {t("cad")}</span>
                <CadView
                  key={revision.id}
                  document={document}
                  selectedId={selectedId}
                  onSelect={select}
                  geometryOptions={{ layer: representation as "model" | "observed_surface" | "point_cloud", frameId: camera?.coordinateFrameId || "", imageId: camera?.imageId, observations: document.observations, showCandidates: !report }}
                />
              </div>
              <div className="canvas-pane">
                <span className="pane-label">04 {t("plan")}</span>
                <PlanView
                  document={document}
                  selectedId={selectedId}
                  onSelect={select}
                  geometryOptions={{ layer: representation as "model" | "observed_surface" | "point_cloud", frameId: camera?.coordinateFrameId || "", imageId: camera?.imageId, observations: document.observations, showCandidates: !report }}
                  interactive
                />
              </div>
            </>
          )}
        </div>
        <div className="canvas-footer">
          <span>{selected?.label || t("selectObject")}</span>
          <button
            className={draw ? "active" : ""}
            onClick={() => {
              setDraw(!draw);
              setMode("photo");
              setPanel("agent");
            }}
          >
            {t("drawBox")}
          </button>
          {box && <span>{t("selectedBox")}</span>}
        </div>
      </section>
      <aside className="inspector-panel">
        <div className="inspector-tabs">
          {(["properties", "agent"] as const).map((p) => (
            <button
              key={p}
              aria-pressed={panel === p}
              onClick={() => setPanel(p)}
            >
              {t(p)}
            </button>
          ))}
        </div>
        {creatingModel ? (
          <PrimitiveCreator
            document={document}
            selected={selected}
            onCommit={async (ops) => {
              const epoch = selectionEpoch.current;
              const result = await onCommit(ops);
              if (result !== false && epoch === selectionEpoch.current) {
                const entityId =
                  ops
                    .slice()
                    .reverse()
                    .find((o) => o.entityId)?.entityId ||
                  ops.find((o) => o.type === "addEntity")?.entity;
                if (typeof entityId === "string") setSelected(entityId);
                setMode("free");
              }
              return result;
            }}
            onClose={() => setCreatingModel(false)}
          />
        ) : panel === "properties" ? (
          <><EntityInspector
            entity={selected}
            document={document}
            onCommit={onCommit}
            busy={busy}
            onGenerate={
              canWrite && onJob
                ? () => onJob("generate_object", selected ? [selected.id] : [])
                : undefined
            }
          />
          {selected && <IdentityReview revision={revision} entityId={selected.id} canWrite={canWrite&&!busy} onSelect={select} onApply={async(ops,context)=>onCommit(ops,context)} onSuggest={()=>{}} onAgent={ids=>{setIdentityEntityIds(ids);setPanel("agent");}}/>}</>
        ) : (
          <AgentPanel
            identityEntityIds={identityEntityIds?.includes(selectedId||"") ? identityEntityIds : undefined}
            projectId={project.id}
            revision={revision}
            branch={branch}
            entityId={selectedId}
            observationId={observationId}
            imageId={imageId}
            box={box}
            canWrite={canWrite}
            onApply={onCommit}
          />
        )}
      </aside>
    </div>
  );
}

function EntityInspector({
  entity,
  document,
  onCommit,
  busy,
  onGenerate,
}: {
  entity: Entity | null;
  document: SceneDocument;
  onCommit: (
    ops: Operation[],
    context?: { baseRevisionId: string; agentTurnId?: string },
  ) => unknown;
  busy: boolean;
  onGenerate?: () => unknown;
}) {
  const { t } = useI18n(),
    [label, setLabel] = useState("");
  useEffect(() => setLabel(entity?.label || ""), [entity?.id, entity?.label]);
  if (!entity) return <div className="empty-copy">{t("selectObject")}</div>;
  const candidate = (entity.representations || []).some(
    (r) =>
      r.id === entity.activeModelRepresentationId && r.sourceValidity !== "stale" && r.placementState === "unconfirmed" &&
      ["imported_proposal", "requires_alignment_confirmation"].includes(
        r.placementReason || "",
      ),
  );
  const referenceSurface = isReferenceSurface(document, entity),
    evidenceStatus = entityEvidenceStatus(document, entity),
    transform = referenceSurface ? null : editableTransform(entity),
    dimensions = sourceDimensions(entity),
    scale = sourceScale(document, entity),
    model = modelGeometry(entity),
    modelUnits = modelScale(document, entity),
    tilt = modelTilt(document, entity);
  const display = (value: number | undefined, units = scale) =>
    Number.isFinite(value)
      ? `${(value! * (units?.nativeToMeters || 1)).toFixed(3)} ${units?.nativeToMeters ? "m" : ""}`
      : t("unknown");
  return (
    <div className="entity-inspector">
      <p className="eyebrow">{t("selection")}</p>
      <h2>{entity.label || entity.id}</h2>
      <ModelEvidence entity={entity} onCommit={onCommit} disabled={busy} />
      {referenceSurface && <p className="evidence-note">{t("sceneReferenceSurface")}</p>}
      <Badge value={evidenceStatus.photoKey} />
      {evidenceStatus.identityKey && <p className="evidence-note">{t(evidenceStatus.identityKey)}</p>}
      <label className="field-label">
        {t("label")}
        <input
          value={label}
          disabled={busy}
          onChange={(e) => setLabel(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && label.trim() && label !== entity.label)
              onCommit([
                { type: "setLabel", entityId: entity.id, label: label.trim() },
              ]);
          }}
        />
      </label>
      {!(entity.representations || []).length && (
        <p className="evidence-note">{t("noGeometryCopy")}</p>
      )}
      {candidate && (
        <p className="evidence-note">
          {t("candidatePlacement")}
          {!busy && transform && (
            <button
              onClick={() =>
                onCommit([
                  { type: "setTransform", entityId: entity.id, transform },
                ])
              }
            >
              {t("acceptPlacement")}
            </button>
          )}
        </p>
      )}
      {transform && (
        <>
          <h3>{t("currentMeasurements")}</h3>
          <dl className="measurement-list">
            {(["height", "width", "depth"] as const).map((k) => (
              <div key={k}>
                <dt>{t(k)}</dt>
                <dd>{display(model?.[k], modelUnits)}</dd>
              </div>
            ))}
            <div>
              <dt>{t("tilt")}</dt>
              <dd>{tilt === null ? t("unknown") : tilt.toFixed(1) + "°"}</dd>
            </div>
          </dl>
          <PrimitiveFields entity={entity} busy={busy} onCommit={onCommit} />
          {(["position", "rotation", "scale"] as const).map((kind) => (
            <TransformFields
              key={entity.id + kind}
              value={
                kind === "rotation"
                  ? quaternionEuler(transform.quaternion)
                  : transform[kind]
              }
              kind={kind}
              disabled={busy}
              onChange={(values) =>
                onCommit([
                  {
                    type: "setTransform",
                    entityId: entity.id,
                    transform: {
                      ...transform,
                      ...(kind === "rotation"
                        ? { quaternion: eulerQuaternion(values) }
                        : { [kind]: values }),
                    },
                  },
                ])
              }
            />
          ))}
        </>
      )}
      <section className="measurement-block">
        <h3>{t("sourceMeasurements")}</h3>
        <Badge value={scale?.status || "uncalibrated"} />
        <dl className="measurement-list">
          {(referenceSurface ? ["widthNative", "depthNative"] as const : ["groundHeight", "extentX", "extentY", "extentZ"] as const).map(
            (key) => (
              <div key={key}>
                <dt>{t(key === "widthNative" ? "width" : key === "depthNative" ? "depth" : key)}</dt>
                <dd>{display(dimensions[key])}</dd>
              </div>
            ),
          )}
        </dl>
        <p className="subtle">{t("sourceNote")}</p>
      </section>
      {onGenerate && !referenceSurface && (
        <button className="wide" onClick={onGenerate}>
          {t("generate")} ↗
        </button>
      )}
      <details>
        <summary>{t("evidence")}</summary>
        {observationsFor(document, entity).map((o) => (
          <p key={o.id} className="evidence-id">
            {o.id}
            <br />
            {o.originalPixelBox?.map((n) => Math.round(n)).join(" · ")}
          </p>
        ))}
      </details>
    </div>
  );
}
function PrimitiveFields({
  entity,
  busy,
  onCommit,
}: {
  entity: Entity;
  busy: boolean;
  onCommit: (ops: Operation[]) => unknown;
}) {
  const { t } = useI18n();
  const rep = activeModel(entity)?.kind === "primitive" ? activeModel(entity) : null,
    spec: any = rep?.primitive,
    p = spec,
    kind = spec?.kind || spec?.type;
  const valueFor = (key: string) =>
    kind === "box"
      ? p.dimensions[["width", "depth", "height"].indexOf(key)]
      : p[key];
  if (!rep || !p) return null;
  return (
    <fieldset className="primitive-fields">
      <legend>{t("primitive")}</legend>
      {(kind === "box"
        ? ["width", "depth", "height"]
        : kind === "cylinder"
          ? ["radius", "height"]
          : []
      ).map((key) => (
        <label className="field-label" key={entity.id + key}>
          {t(key)}
          <input
            type="number"
            min="0.000001"
            step="any"
            disabled={busy}
            defaultValue={valueFor(key)}
            key={entity.id + key + valueFor(key)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                const value = Number(e.currentTarget.value);
                if (
                  Number.isFinite(value) &&
                  value > 0 &&
                  value !== valueFor(key)
                )
                  onCommit([
                    {
                      type: "setPrimitive",
                      entityId: entity.id,
                      primitive: {
                        ...spec,
                        ...(kind === "box"
                          ? {
                              dimensions: p.dimensions.map(
                                (v: number, i: number) =>
                                  i ===
                                  ["width", "depth", "height"].indexOf(key)
                                    ? value
                                    : v,
                              ),
                            }
                          : { [key]: value }),
                      },
                      transform: editableTransform(entity),
                    },
                  ]);
              }
            }}
          />
        </label>
      ))}
    </fieldset>
  );
}
function TransformFields({
  value,
  kind,
  disabled,
  onChange,
}: {
  value: Vec3;
  kind: string;
  disabled: boolean;
  onChange: (next: Vec3) => void;
}) {
  const { t } = useI18n(),
    [fields, setFields] = useState(value.map(String));
  useEffect(
    () => setFields(value.map((n) => String(Math.round(n * 1e5) / 1e5))),
    [JSON.stringify(value)],
  );
  function commit() {
    const next = fields.map(Number) as Vec3;
    if (
      next.every(Number.isFinite) &&
      (kind !== "scale" || next.every((n) => n > 0)) &&
      next.some((n, i) => n !== value[i])
    )
      onChange(next);
  }
  return (
    <fieldset className="transform-fields">
      <legend>{t(kind)}</legend>
      {fields.map((v, i) => (
        <label key={i} className={"axis-" + i}>
          <span>{"XYZ"[i]}</span>
          <input
            type="number"
            step="any"
            value={v}
            disabled={disabled}
            min={kind === "scale" ? 0.000001 : undefined}
            onChange={(e) =>
              setFields((items) =>
                items.map((x, n) => (n === i ? e.target.value : x)),
              )
            }
            onKeyDown={(e) => {
              if (e.key === "Enter") commit();
            }}
          />
        </label>
      ))}
    </fieldset>
  );
}

export function PlanView({
  document,
  selectedId,
  onSelect,
  interactive = false,
  geometryOptions,
}: {
  document: SceneDocument;
  selectedId: string | null;
  onSelect: (id: string) => void;
  interactive?: boolean;
  geometryOptions?: Parameters<typeof planShapes>[1];
}) {
  const { t } = useI18n(),
    [zoom, setZoom] = useState(1),
    [candidates, setCandidates] = useState<string[]>([]);
  const drawing = useRef<SVGGElement>(null),
    svg = useRef<SVGSVGElement>(null),
    picker = useRef<HTMLDivElement>(null),
    pointer = useRef<{ x: number; y: number; moved: boolean } | null>(null);
  useEffect(() => { setCandidates([]); pointer.current = null; }, [document, selectedId, zoom, geometryOptions?.layer, geometryOptions?.frameId, geometryOptions?.imageId]);
  useEffect(() => { if (candidates.length) picker.current?.querySelector<HTMLButtonElement>("button[data-candidate]")?.focus(); }, [candidates]);
  function choose(id: string) {
    setCandidates([]);
    onSelect(id);
    svg.current?.focus();
  }
  const shapes = planShapes(document, geometryOptions);
  if (!shapes.length)
    return <div className="empty-stage">{t("emptyPlan")}</div>;
  const min = [
      Math.min(...shapes.map((s) => s.min[0])),
      Math.min(...shapes.map((s) => s.min[1])),
    ],
    max = [
      Math.max(...shapes.map((s) => s.max[0])),
      Math.max(...shapes.map((s) => s.max[1])),
    ],
    factor = Math.min(
      520 / Math.max(max[0] - min[0], 1e-6),
      320 / Math.max(max[1] - min[1], 1e-6),
    );
  return (
    <div className="plan-view" onKeyDown={(event) => {
      if (event.key === "Escape" && candidates.length) {
        event.preventDefault();
        event.stopPropagation();
        setCandidates([]);
        svg.current?.focus();
      }
    }}>
      <div className="plan-coverage">
        <strong>{t(interactive ? "planScope" : "cadScope")}</strong>
        <span>{shapes.length} / {document.entities.filter((entity) => entity.visible !== false && !entity.sourceContext).length} {t("planCoverage")}</span>
      </div>
      {interactive && (
        <div className="plan-tools">
          <button onClick={() => setZoom((z) => Math.max(0.5, z / 1.2))}>
            −
          </button>
          <button onClick={() => setZoom(1)}>{t("fit")}</button>
          <button onClick={() => setZoom((z) => Math.min(4, z * 1.2))}>
            ＋
          </button>
        </div>
      )}
      <svg ref={svg} tabIndex={-1} viewBox="0 0 600 400" aria-label={t(interactive ? "plan" : "cad")}
        onPointerDown={(event) => { pointer.current = event.button === 0 ? { x: event.clientX, y: event.clientY, moved: false } : null; }}
        onPointerMove={(event) => { const start = pointer.current; if (start && Math.hypot(event.clientX - start.x, event.clientY - start.y) > 4) start.moved = true; }}
        onPointerCancel={() => { pointer.current = null; }}
        onClick={(event) => {
          const start = pointer.current;
          pointer.current = null;
          if (start?.moved) return;
          const named = (event.target as Element).closest("[data-plan-entity]")?.getAttribute("data-plan-entity");
          if (event.detail === 0 && named) { choose(named); return; }
          const matrix = drawing.current?.getScreenCTM();
          if (!matrix) return;
          const local = new DOMPoint(event.clientX, event.clientY).matrixTransform(matrix.inverse());
          const hits = planHits(shapes, local.x / factor + min[0], max[1] - local.y / factor, 4 / (factor * zoom));
          if (hits.length === 1) choose(hits[0].entity.id);
          else setCandidates(hits.map((hit) => hit.entity.id));
        }}>
        {!interactive && <g className="plan-grid" aria-hidden="true">
          {Array.from({ length: 13 }, (_, index) => <line key={"x" + index} x1={index * 50} y1="0" x2={index * 50} y2="400" />)}
          {Array.from({ length: 9 }, (_, index) => <line key={"y" + index} x1="0" y1={index * 50} x2="600" y2={index * 50} />)}
        </g>}
        <g
          ref={drawing}
          transform={`translate(300 200) scale(${zoom}) translate(${(-(max[0] - min[0]) * factor) / 2} ${(-(max[1] - min[1]) * factor) / 2})`}
        >
          {shapes
            .slice()
            .sort(
              (a, b) =>
                (b.max[0] - b.min[0]) * (b.max[1] - b.min[1]) -
                (a.max[0] - a.min[0]) * (a.max[1] - a.min[1]),
            )
            .map(({ entity, min: lo, max: hi, polygons, lines, projectionSource, geometryKind }) => (
              <g
                key={entity.id}
                role="button"
                tabIndex={0}
                data-plan-entity={entity.id}
                aria-label={entity.label || entity.id}
                aria-pressed={entity.id === selectedId}
                onKeyDown={(e) => {
                  if (["Enter", " "].includes(e.key)) {
                    e.preventDefault();
                    e.stopPropagation();
                    choose(entity.id);
                  }
                }}
              >
                {polygons.map((polygon, index) => <path key={"area-" + index}
                  d={planPolygonPath(polygon, p => [(p[0] - min[0]) * factor, (max[1] - p[1]) * factor])}
                  fillRule="evenodd"
                  className={[entity.id === selectedId ? "selected" : "", geometryKind === "model" ? "model-footprint" : projectionSource === "saved_hull" ? "saved-footprint" : "observed-footprint"].join(" ")}
                />)}
                {lines.map((line, index) => <path key={"line-" + index}
                  d={"M " + line.map(p => [(p[0] - min[0]) * factor, (max[1] - p[1]) * factor].join(",")).join(" L ")}
                  style={{fill: "none"}} className={entity.id === selectedId ? "selected" : "observed-footprint"} />)}
                <title>{entity.label || entity.id}</title>
                {entity.id === selectedId && (
                  <text
                    x={(lo[0] - min[0]) * factor + 5}
                    y={(max[1] - hi[1]) * factor + 16}
                  >
                    {entity.label || entity.id}
                  </text>
                )}
              </g>
            ))}
        </g>
      </svg>
      <div className="plan-legend"><span className="observed-key">{t("planObserved")}</span><span className="model-key">{t("planModeled")}</span><span className="saved-key">{t("planSaved")}</span><span>{t("planNativeUnits")}</span></div>
      {!interactive && <details className="plan-object-index"><summary>{t("planIndex")}</summary>
        <div>{document.entities.filter((entity) => entity.visible !== false && !entity.sourceContext).map((entity) => <button key={entity.id} type="button" aria-pressed={entity.id === selectedId} onClick={() => choose(entity.id)}><span>{entity.label || entity.id}</span><small>{shapes.some((shape) => shape.entity.id === entity.id) ? t("planProjected") : t("planMissing")}</small></button>)}</div>
      </details>}
      {candidates.length > 1 && (
        <div ref={picker} className="plan-hit-picker" role="group" aria-label={t("scenePlanOverlap")}>
          <header><strong>{t("scenePlanOverlap")} · {candidates.length}</strong><button type="button" aria-label={t("close")} onClick={() => { setCandidates([]); svg.current?.focus(); }}>×</button></header>
          <p>{t("sceneChoosePlanObject")}</p>
          <div className="plan-hit-options">
            {candidates.map((id) => {
              const entity = shapes.find((shape) => shape.entity.id === id)?.entity;
              return entity && <button type="button" key={id} data-candidate={id} aria-pressed={id === selectedId} onClick={() => choose(id)}><span>{entity.label || id}</span><small>{id.slice(0, 8)}</small></button>;
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function TaskStrip({
  projectId,
  onComplete,
}: {
  projectId: string;
  onComplete: () => void;
}) {
  const { t } = useI18n(),
    [jobs, setJobs] = useState<Job[]>([]),
    [error, setError] = useState<unknown>(),
    callback = useRef(onComplete),
    previous = useRef(new Map<string, string>());
  callback.current = onComplete;
  useEffect(() => {
    let stopped = false,
      timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const data = await request<{ items: Job[] }>(
          `/api/projects/${projectId}/jobs`,
        );
        if (stopped) return;
        setJobs(data.items);
        for (const job of data.items) {
          if (
            previous.current.has(job.id) &&
            previous.current.get(job.id) !== job.status &&
            ["succeeded", "incomplete"].includes(job.status)
          )
            callback.current();
          previous.current.set(job.id, job.status);
        }
      } catch (e) {
        if (!stopped) setError(e);
      }
      if (
        !stopped &&
        previous.current.size &&
        [...previous.current.values()].some((s) =>
          [
            "pending_dispatch",
            "queued",
            "running",
            "cancel_requested",
          ].includes(s),
        )
      )
        timer = setTimeout(poll, 4000);
    }
    const changed = () => {
      clearTimeout(timer);
      void poll();
    };
    window.addEventListener("panoptes:mutation", changed);
    void poll();
    return () => {
      stopped = true;
      clearTimeout(timer);
      window.removeEventListener("panoptes:mutation", changed);
    };
  }, [projectId]);
  return (
    <details className="task-strip">
      <summary>
        {t("taskProgress")}{" "}
        <span>
          {
            jobs.filter((j) =>
              ["pending_dispatch", "queued", "running"].includes(j.status),
            ).length
          }{" "}
          {t("running")}
        </span>
      </summary>
      <ErrorNotice error={error} />
      {jobs.length ? (
        jobs.map((job) => {
          const result = jobResultSummary(job.result);
          return (
            <div className="job-row" key={job.id}>
              <span>{job.kind}</span>
              <Badge value={job.status} />
              {!!result.errors.length && (
                <span className="job-error">
                  {result.errors.map(t).join(" · ")}
                </span>
              )}
              {result.workerSeconds !== undefined && (
                <span>
                  {result.workerSeconds.toFixed(1)} {t("seconds")}
                </span>
              )}
              {result.assets.map((asset) => (
                <button
                  key={asset.id}
                  onClick={() =>
                    resolveAsset(asset.id)
                      .then((url) => window.open(url, "_blank", "noopener"))
                      .catch(setError)
                  }
                >
                  {asset.name || t("download")}
                </button>
              ))}
              {["queued", "running", "pending_dispatch"].includes(
                job.status,
              ) && (
                <button
                  onClick={() =>
                    request("/api/jobs/" + job.id + "/cancel", {
                      method: "POST",
                      projectId,
                      body: { requestId: id() },
                    }).catch(setError)
                  }
                >
                  {t("cancel")}
                </button>
              )}
            </div>
          );
        })
      ) : (
        <p>{t("noTasks")}</p>
      )}
    </details>
  );
}

function History({ projectId }: { projectId: string }) {
  const { t } = useI18n(),
    project = useResource<ProjectDetail>(projectId, () =>
      request("/api/projects/" + projectId),
    ),
    revisions = useResource<{ items: Revision[] }>(
      projectId + "revisions",
      () => request(`/api/projects/${projectId}/revisions`),
    ),
    publications = useResource<{ items: PublicationSummary[] }>(
      projectId + "publications",
      () => request("/api/publications"),
    );
  return (
    <section className="page-width">
      <div className="title-row">
        <h1>{project.data?.project.title || t("history")}</h1>
        <ProjectNav projectId={projectId} active="history" />
      </div>
      <ErrorNotice
        error={project.error || revisions.error || publications.error}
      />
      <h2>{t("versions")}</h2>
      <div className="history-list">
        {revisions.data?.items.map((revision) => (
          <a
            key={revision.id}
            href={path(
              `/projects/${projectId}/workbench?revision=${revision.id}`,
            )}
          >
            <span className="history-dot" />
            <div>
              <strong>{revision.label || revision.id.slice(0, 8)}</strong>
              <small>
                {revision.document.entities.length} {t("objects")}
              </small>
            </div>
            <DateLabel value={revision.createdAt} />
            <span>↗</span>
          </a>
        ))}
      </div>
      <h2>{t("reports")}</h2>
      {publications.data?.items
        .filter((p) => p.projectId === projectId)
        .map((p) => (
          <a
            className="report-history"
            key={p.id}
            href={path("/reports/" + p.id)}
          >
            <strong>{p.title}</strong>
            <DateLabel value={p.createdAt} />
            <span>↗</span>
          </a>
        ))}
      <TaskStrip projectId={projectId} onComplete={revisions.reload} />
    </section>
  );
}
function ReportLibrary() {
  const { t } = useI18n(),
    { data, error } = useResource<{ items: PublicationSummary[] }>(
      "publications",
      () => request("/api/publications"),
    );
  const [search, setSearch] = useState("");
  const groups = groupPublications(data?.items || []).filter((group) =>
    group.some((p) =>
      p.title.toLocaleLowerCase().includes(search.toLocaleLowerCase()),
    ),
  );
  return (
    <section className="page-width report-catalog">
      <h1>{t("reportLibrary")}</h1>
      <p className="lede">{t("catalogIntro")}</p>
      <ErrorNotice error={error} />
      <div className="list-toolbar">
        <input
          type="search"
          aria-label={t("search")}
          placeholder={t("search")}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        {data && (
          <span>
            {groupPublications(data.items).length} {t("catalogWorkcells")} ·{" "}
            {data.items.length} {t("catalogPublications")}
          </span>
        )}
      </div>
      {!data && !error ? (
        <Loading />
      ) : data?.items.length ? (
        <div className="workcell-catalog">
          {!groups.length && <p className="empty-state">{t("noResults")}</p>}
          {groups.map(([p, ...history]) => (
            <article
              className="catalog-workcell"
              key={p.projectId}
              aria-label={p.title}
            >
              <a
                className="catalog-report-link"
                href={path("/reports/" + p.id)}
              >
                <div className="catalog-cover">
                  {p.previewImageAssetId ? (
                    <AssetImage assetId={p.previewImageAssetId} alt={p.title} />
                  ) : (
                    <span>{t("noPhoto")}</span>
                  )}
                </div>
                <div className="catalog-report-body">
                  <p className="catalog-caption">
                    {t("catalogLatest")} · <DateLabel value={p.createdAt} />
                  </p>
                  <h2>{p.title}</h2>
                  <p className="catalog-revision">
                    {t("version")} {p.sceneRevisionId.slice(0, 8)} ·{" "}
                    {t("reportReadOnly")}
                  </p>
                  <dl className="catalog-content">
                    {(
                      [
                        ["photos", p.photoCount],
                        ["reportObjectRecords", p.objectCount],
                        ["reportSpatialObjects", p.spatialObjectCount],
                        ["reportModelObjects", p.modelObjectCount],
                      ] as const
                    ).map(([key, count]) => (
                      <div key={key}>
                        <dt>{t(key)}</dt>
                        <dd>{count}</dd>
                      </div>
                    ))}
                  </dl>
                  {p.observedSurfaceObjectCount === 0 && (
                    <p className="catalog-missing">
                      {t("catalogNoObservedScene")}
                    </p>
                  )}
                  <span className="catalog-open">
                    {t("viewReport")} <span aria-hidden="true">↗</span>
                  </span>
                </div>
              </a>
              {history.length > 0 && (
                <details className="catalog-history">
                  <summary>
                    {t("catalogHistory")} · {history.length}
                  </summary>
                  <ol>
                    {history.map((old) => (
                      <li key={old.id}>
                        <a href={path("/reports/" + old.id)}>
                          <span>{old.title}</span>
                          <small>
                            {t("version")} {old.sceneRevisionId.slice(0, 8)} ·{" "}
                            {old.photoCount} {t("photos")} · {old.objectCount}{" "}
                            {t("reportObjectRecords")}
                          </small>
                          <DateLabel value={old.createdAt} />
                        </a>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </article>
          ))}
        </div>
      ) : (
        <div className="empty-state">{t("noReports")}</div>
      )}
    </section>
  );
}
