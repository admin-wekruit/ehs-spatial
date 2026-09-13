import { useEffect, useRef, useState } from "react";
import { resolveAsset } from "./api";
import {
  observationsFor,
  originalPixel,
  photoHits,
  observationPolygons,
} from "./core";
import { useI18n } from "./i18n";
import type { SceneDocument } from "./types";

export function PhotoView({
  document,
  imageId,
  selectedId,
  onSelect,
  onBox,
  draw = false,
}: {
  document: SceneDocument;
  imageId: string | null;
  selectedId: string | null;
  onSelect: (id: string, observationId?: string) => void;
  onBox?: (box: number[] | null) => void;
  draw?: boolean;
}) {
  const { t } = useI18n(),
    [url, setURL] = useState<string>(),
    [size, setSize] = useState<[number, number]>([1, 1]),
    [error, setError] = useState(false),
    [box, setBox] = useState<number[] | null>(null),
    ref = useRef<SVGSVGElement>(null),
    start = useRef<[number, number] | null>(null),
    drawingGesture = useRef(false);
  useEffect(() => {
    let live = true;
    setURL(undefined);
    setError(false);
    setBox(null);
    if (imageId)
      resolveAsset(imageId)
        .then((url) => {
          if (live) setURL(url);
        })
        .catch(() => {
          if (live) setError(true);
        });
    return () => {
      live = false;
    };
  }, [imageId]);
  const camera = document.cameras.find((c) => c.imageId === imageId),
    width = camera?.width || size[0],
    height = camera?.height || size[1];
  function coordinates(e: React.PointerEvent | React.MouseEvent) {
    return ref.current
      ? originalPixel(
          e.clientX,
          e.clientY,
          ref.current.getBoundingClientRect(),
          width,
          height,
        )
      : null;
  }
  const overlays = document.entities
    .flatMap((entity) =>
      observationsFor(document, entity)
        .filter(
          (o) =>
            o.imageId === imageId &&
            o.originalPixelBox &&
            entity.visible !== false,
        )
        .map((observation) => ({
          entity,
          observation,
          box: observation.originalPixelBox!,
        })),
    )
    .sort(
      (a, b) =>
        (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]) -
        (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]),
    );
  if (!imageId || error)
    return <div className="empty-stage">{t("noPhoto")}</div>;
  return (
    <div className={"photo-view " + (draw ? "draw-mode" : "")}>
      {!url ? (
        <p className="stage-status">{t("loading")}</p>
      ) : (
        <svg
          ref={ref}
          viewBox={`0 0 ${width} ${height}`}
          preserveAspectRatio="xMidYMid meet"
          role="group"
          aria-label={t("sourceEvidence")}
          onPointerDown={(e) => {
            drawingGesture.current = draw;
            if (!draw) return;
            start.current = coordinates(e);
            if (start.current) {
              ref.current?.setPointerCapture(e.pointerId);
              setBox([...start.current, ...start.current]);
            }
          }}
          onPointerMove={(e) => {
            if (!start.current) return;
            const end = coordinates(e);
            if (end)
              setBox([
                Math.min(start.current[0], end[0]),
                Math.min(start.current[1], end[1]),
                Math.max(start.current[0], end[0]),
                Math.max(start.current[1], end[1]),
              ]);
          }}
          onPointerUp={(e) => {
            const end = coordinates(e);
            const region =
              start.current && end
                ? [
                    Math.min(start.current[0], end[0]),
                    Math.min(start.current[1], end[1]),
                    Math.max(start.current[0], end[0]),
                    Math.max(start.current[1], end[1]),
                  ]
                : box;
            if (
              start.current &&
              region &&
              region[2] - region[0] > 2 &&
              region[3] - region[1] > 2
            )
              onBox?.(region);
            start.current = null;
          }}
          onPointerCancel={() => {
            start.current = null;
            setBox(null);
          }}
          onClick={(e) => {
            // Pointer-up may leave drawing mode before its click event arrives.
            if (drawingGesture.current) {
              drawingGesture.current = false;
              return;
            }
            if (draw) return;
            const xy = coordinates(e);
            if (!xy) return;
            const hit = photoHits(document, imageId, xy[0], xy[1])[0];
            if (hit) onSelect(hit.entity.id, hit.observation.id);
          }}
        >
          <image
            href={url}
            width={width}
            height={height}
            onLoad={() => {
              const img = new Image();
              img.onload = () => setSize([img.naturalWidth, img.naturalHeight]);
              img.src = url;
            }}
          />
          {overlays.map(({ entity, observation, box: b }) => {
            const polygons = observationPolygons(observation);
            const d = polygons.length
              ? polygons
                  .map(
                    (p) => "M" + p.map((xy) => xy.join(",")).join(" L") + " Z",
                  )
                  .join(" ")
              : `M${b[0]},${b[1]} H${b[2]} V${b[3]} H${b[0]} Z`;
            return (
              <path
                key={entity.id + observation.id}
                d={d}
                fillRule="evenodd"
                data-evidence-kind={polygons.length ? "segmentation" : "detection_box"}
                className={
                  entity.id === selectedId
                    ? "photo-bound selected"
                    : "photo-bound"
                }
                vectorEffect="non-scaling-stroke"
                role="button"
                tabIndex={0}
                aria-label={entity.label || entity.id}
                onKeyDown={(e) => {
                  if (["Enter", " "].includes(e.key)) {
                    e.preventDefault();
                    onSelect(entity.id, observation.id);
                  }
                }}
              >
                <title>{entity.label || entity.id}</title>
              </path>
            );
          })}
          {box && (
            <rect
              x={box[0]}
              y={box[1]}
              width={box[2] - box[0]}
              height={box[3] - box[1]}
              className="drawn-box"
              vectorEffect="non-scaling-stroke"
            />
          )}
        </svg>
      )}
    </div>
  );
}
