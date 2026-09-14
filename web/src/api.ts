import type { Asset } from "./types";
export type PendingRequest = {
  id: string;
  url: string;
  body: Record<string, unknown>;
  capability: string;
  createdAt: string;
};
type Owner = { id: string; capability: string };
export type FeedbackInput = {
  requestId: string; conversationId: string; message: string; language: "zh" | "en";
  imageId?: string | null; observationId?: string | null;
};
type FeedbackSession = Owner & { conversationId: string; pending?: FeedbackInput };
export const PUBLICATION_ID = import.meta.env?.VITE_PUBLICATION_ID?.trim() || "";
const DB = "panoptes-platform",
  API_ORIGIN = (import.meta.env?.VITE_API_ORIGIN || "").replace(/\/$/, "");
let opened: Promise<IDBDatabase> | undefined;
function database() {
  return (opened ??= new Promise((resolve, reject) => {
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore("owners", { keyPath: "id" });
      request.result.createObjectStore("pending", { keyPath: "id" });
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(new Error("browser_storage_unavailable"));
  }));
}
async function read<T>(store: string, id?: string): Promise<T> {
  const db = await database();
  return new Promise((resolve, reject) => {
    const request = id
      ? db.transaction(store).objectStore(store).get(id)
      : db.transaction(store).objectStore(store).getAll();
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(new Error("browser_storage_unavailable"));
  });
}
async function put(store: string, value: unknown) {
  const db = await database();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(store, "readwrite");
    tx.objectStore(store).put(value);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(new Error("browser_storage_unavailable"));
    tx.onabort = () => reject(new Error("browser_storage_unavailable"));
  });
}
export const owner = async (id: string) =>
  (await read<Owner | undefined>("owners", id))?.capability;
export async function feedbackSession(publicationId: string, entityId: string, pending?: FeedbackInput | null): Promise<FeedbackSession> {
  const db = await database(), key = `feedback:${publicationId}:${entityId}`;
  return new Promise((resolve, reject) => {
    const tx = db.transaction("owners", "readwrite"), store = tx.objectStore("owners"), found = store.get(key);
    let session: FeedbackSession;
    found.onsuccess = () => {
      session = found.result;
      if (!session) {
        const bytes = crypto.getRandomValues(new Uint8Array(32));
        session = { id: key, capability: btoa(String.fromCharCode(...bytes)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, ""), conversationId: crypto.randomUUID() };
      }
      if (pending !== undefined) session = { ...session, pending: pending || undefined };
      store.put(session);
    };
    tx.oncomplete = () => resolve(session);
    tx.onerror = tx.onabort = () => reject(new Error("browser_storage_unavailable"));
  });
}
export const ownedIds = async () =>
  new Set((await read<Owner[]>("owners")).filter((o) => !o.id.startsWith("feedback:")).map((o) => o.id));
export const pendingRequests = () => read<PendingRequest[]>("pending");
export class ApiError extends Error {
  status: number;
  params: Record<string, any>;
  constructor(status: number, code: string, params: Record<string, any> = {}) {
    super(code);
    this.status = status;
    this.params = params;
  }
}
export async function request<T>(
  path: string,
  options: {
    method?: string;
    body?: unknown;
    projectId?: string;
    capability?: string;
    feedbackCapability?: string;
    signal?: AbortSignal;
  } = {},
): Promise<T> {
  if (!path.startsWith("/api/")) throw new Error("invalid_api_path");
  const method = options.method || "GET",
    headers = new Headers();
  if (options.feedbackCapability && (options.projectId || options.capability)) throw new Error("mixed_capability_context");
  const capability =
    options.capability ||
    (options.projectId ? await owner(options.projectId) : undefined);
  if (method !== "GET" && options.projectId && !capability)
    throw new ApiError(403, "owner_capability_required");
  if (capability && method !== "GET")
    headers.set("Authorization", "Capability " + capability);
  if (options.feedbackCapability) headers.set("Authorization", "Feedback " + options.feedbackCapability);
  const multipart = options.body instanceof FormData;
  if (options.body !== undefined && !multipart)
    headers.set("Content-Type", "application/json");
  const response = await fetch(API_ORIGIN + path, {
    method,
    headers,
    body:
      options.body === undefined
        ? undefined
        : multipart
          ? (options.body as FormData)
          : JSON.stringify(options.body),
    signal: options.signal,
    credentials: "omit",
    redirect: "error",
  });
  const value = await response
    .json()
    .catch(() => ({ error: { code: "invalid_response" } }));
  if (!response.ok)
    throw new ApiError(
      response.status,
      value.error?.code || "request_failed",
      value.error?.params || {},
    );
  if (method !== "GET")
    window.dispatchEvent(
      new CustomEvent("panoptes:mutation", { detail: path }),
    );
  return value as T;
}
export async function prepareOwnedRequest(
  url: string,
  body: Record<string, unknown>,
) {
  const bytes = crypto.getRandomValues(new Uint8Array(32));
  const capability =
    "pcap_v1_" +
    btoa(String.fromCharCode(...bytes))
      .replace(/\+/g, "-")
      .replace(/\//g, "_")
      .replace(/=+$/, "");
  const id = crypto.randomUUID();
  const pending = {
    id,
    url,
    body: { ...body, requestId: id },
    capability,
    createdAt: new Date().toISOString(),
  };
  await put("pending", pending);
  return pending;
}
export async function sendOwnedRequest<T extends { project: { id: string } }>(
  pending: PendingRequest,
): Promise<T> {
  const result = await request<T>(pending.url, {
    method: "POST",
    body: pending.body,
    capability: pending.capability,
  });
  const db = await database();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(["owners", "pending"], "readwrite");
    tx.objectStore("owners").put({
      id: result.project.id,
      capability: pending.capability,
    });
    tx.objectStore("pending").delete(pending.id);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(new Error("browser_storage_unavailable"));
    tx.onabort = () => reject(new Error("browser_storage_unavailable"));
  });
  return result;
}
export async function importCapability(file: File) {
  if (file.size > 8192) throw new Error("invalid_management_key");
  const value = JSON.parse(await file.text());
  if (
    value.schemaVersion !== 1 ||
    typeof value.projectId !== "string" ||
    !/^pcap_v1_[A-Za-z0-9_-]{43}$/.test(value.capability)
  )
    throw new Error("invalid_management_key");
  await put("owners", { id: value.projectId, capability: value.capability });
  return value.projectId as string;
}
export async function exportCapability(projectId: string) {
  const capability = await owner(projectId);
  if (!capability) throw new ApiError(403, "owner_capability_required");
  downloadJSON(
    { schemaVersion: 1, projectId, capability },
    "panoptes-management-key.json",
  );
}
export function downloadJSON(data: unknown, name: string) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
const assets = new Map<string, Promise<Asset>>();
export function asset(id: string) {
  if (!assets.has(id))
    assets.set(
      id,
      request<Asset>("/api/assets/" + encodeURIComponent(id))
        .then((value) => {
          const parsed = new URL(value.url, API_ORIGIN || location.origin);
          if (
            !["http:", "https:"].includes(parsed.protocol) ||
            parsed.username ||
            parsed.password
          )
            throw new Error("invalid_asset_url");
          return { ...value, url: parsed.href };
        })
        .finally(() => {
          // ponytail: deduplicate in-flight lookups only; signed URLs expire.
          assets.delete(id);
        }),
    );
  return assets.get(id)!;
}
export const resolveAsset = async (id: string) => (await asset(id)).url;
export async function downloadAsset(id: string) {
  const value = await asset(id);
  // A local blob URL keeps cross-origin signed assets from navigating away from the report.
  const response = await fetch(
    API_ORIGIN + "/api/assets/" + encodeURIComponent(id) + "/content",
    {
      credentials: "omit",
      redirect: "error",
    },
  );
  if (!response.ok)
    throw new ApiError(response.status, "asset_download_failed");
  const blob = await response.blob();
  if (blob.size !== value.sizeBytes) throw new Error("asset_size_mismatch");
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const extension: Record<string, string> = {
    "application/x-blender": ".blend",
    "model/gltf-binary": ".glb",
    "application/json": ".json",
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/svg+xml": ".svg",
    "application/pdf": ".pdf",
  };
  a.download =
    typeof value.metadata.name === "string"
      ? value.metadata.name
      : id + (extension[value.mediaType] || "");
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
export const id = () => crypto.randomUUID();
