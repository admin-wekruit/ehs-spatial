import { useState } from "react";
import { id } from "./api";
import { useI18n } from "./i18n";
import type { Entity, Operation, SceneDocument } from "./types";
export function PrimitiveCreator({
  document,
  selected,
  onCommit,
  onClose,
}: {
  document: SceneDocument;
  selected: Entity | null;
  onCommit: (operations: Operation[]) => unknown;
  onClose: () => void;
}) {
  const { t } = useI18n(),
    [label, setLabel] = useState(
      selected && !(selected.representations || []).length
        ? selected.label || ""
        : "",
    ),
    [attach, setAttach] = useState(
      !!selected && !(selected.representations || []).length,
    ),
    [kind, setKind] = useState("box"),
    [frame, setFrame] = useState(document.coordinateFrames[0]?.id || "new"),
    [dims, setDims] = useState({ width: 1, depth: 1, height: 1, radius: 0.5 }),
    [position, setPosition] = useState([0, 0, 0]),
    [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (
      !label.trim() ||
      !Object.values(dims).every((n) => Number.isFinite(n) && n > 0) ||
      !position.every(Number.isFinite)
    )
      return;
    const frameId = frame === "new" ? id() : frame,
      entityId = attach && selected ? selected.id : id(),
      transform = {
        coordinateFrameId: frameId,
        position,
        quaternion: [0, 0, 0, 1],
        scale: [1, 1, 1],
      },
      operations: Operation[] = [];
    if (frame === "new")
      operations.push({
        type: "addCoordinateFrame",
        frame: {
          id: frameId,
          convention: "opencv",
          source: "manual_assertion",
          scale: {
            status: "uncalibrated",
            nativeToMeters: null,
            sourceRefs: [],
          },
          ground: null,
        },
      });
    if (!attach)
      operations.push({
        type: "addEntity",
        entity: {
          id: entityId,
          label: label.trim(),
          associationState: "association_pending",
          observationRefs: [],
          representations: [],
          measurements: {},
          currentModelTransform: null,
        },
      });
    operations.push({
      type: "setPrimitive",
      entityId,
      transform,
      primitive:
        kind === "box"
          ? { kind, dimensions: [dims.width, dims.depth, dims.height] }
          : { kind, radius: dims.radius, height: dims.height },
    });
    setBusy(true);
    try {
      const saved = await onCommit(operations);
      if (saved !== false) onClose();
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="primitive-creator" onSubmit={submit}>
      <div className="list-toolbar">
        <h2>{t("addModel")}</h2>
        <button type="button" onClick={onClose} aria-label={t("close")}>
          ×
        </button>
      </div>
      <p className="subtle">{t("modelFrameNote")}</p>
      {selected && !(selected.representations || []).length && (
        <label>
          <input
            type="checkbox"
            checked={attach}
            onChange={(e) => setAttach(e.target.checked)}
          />
          {t("attachSelected")} · {selected.label || selected.id}
        </label>
      )}
      <label className="field-label">
        {t("label")}
        <input
          required
          value={label}
          onChange={(e) => setLabel(e.target.value)}
        />
      </label>
      <label className="field-label">
        {t("primitive")}
        <select value={kind} onChange={(e) => setKind(e.target.value)}>
          <option value="box">{t("box")}</option>
          <option value="cylinder">{t("cylinder")}</option>
        </select>
      </label>
      <label className="field-label">
        {t("coordinateFrame")}
        <select value={frame} onChange={(e) => setFrame(e.target.value)}>
          {document.coordinateFrames.map((f) => (
            <option key={f.id} value={f.id}>
              {f.id.slice(0, 8)}
            </option>
          ))}
          <option value="new">{t("newModelFrame")}</option>
        </select>
      </label>
      <div className="primitive-dimensions">
        {(kind === "box"
          ? ["width", "depth", "height"]
          : ["radius", "height"]
        ).map((key) => (
          <label className="field-label" key={key}>
            {t(key)}
            <input
              type="number"
              min="0.000001"
              step="any"
              required
              value={dims[key as keyof typeof dims]}
              onChange={(e) =>
                setDims((old) => ({ ...old, [key]: Number(e.target.value) }))
              }
            />
          </label>
        ))}
      </div>
      <fieldset className="transform-fields">
        <legend>{t("position")}</legend>
        {position.map((v, i) => (
          <label key={i}>
            <span>{"XYZ"[i]}</span>
            <input
              type="number"
              step="any"
              value={v}
              onChange={(e) =>
                setPosition((p) =>
                  p.map((x, j) => (j === i ? Number(e.target.value) : x)),
                )
              }
            />
          </label>
        ))}
      </fieldset>
      <button className="primary" disabled={busy}>
        {t("addModel")}
      </button>
    </form>
  );
}
