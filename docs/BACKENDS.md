# Model backend interface

Every heavyweight model call routes through an env-selected backend. All
backends for a service speak the SAME request/response schema, so moving
from a vendor API to our Modal deployment to an internal GPU cluster is a
config change — no code changes, no schema drift.

Reference implementation of the switch: `ehs_spatial/backends.py`
(`service_backend`, `http_json`).

## Switches

| Service      | Env var            | Values (default first)        | Router |
|--------------|--------------------|-------------------------------|--------|
| SAM 3        | `SAM3_BACKEND`     | `fal` \| `modal` \| `http`    | `ehs_spatial/providers/sam3.py::sam_subscribe` |
| MapAnything  | `GEOMETRY_BACKEND` | `replicate` \| `modal` \| `http` | `ehs_spatial/providers/map_anything.py::_default_runner` |
| MoGe-3       | `MOGE_BACKEND`     | `modal` \| `replicate` \| `http` | `ehs_spatial/providers/moge.py::_default_runner` |
| VLM (Gemini) | — (adapter seam)   | swap `ehs_spatial/providers/gemini.py` adapter for any OpenAI-compatible endpoint | pipeline takes the adapter as a parameter |

`modal` backends resolve `modal.Cls.from_name(app, cls)`:

- SAM 3: `("sam3-inference", "Sam3")` — deployed (`modal_apps/sam3_app.py`)
- MoGe-3: `("moge3-inference", "MoGe3")` — deployed
- MapAnything: `("mapanything-inference", "MapAnything")` — deployed and
  adapter-validated (masks 96-98% after depth-edge trim, matching the
  replicate wrapper; frames decode through parse_frame_json; GLB written).
  `replicate` stays default until a full-pipeline A/B signs off; flip with
  `GEOMETRY_BACKEND=modal`. Note: Modal serving keeps the source aspect
  (518x392 on 3:4 captures) where replicate squares to 518x518, and raw
  metric scale differs ~1.4x — the MoGe anchor re-gauges either way.

## HTTP contract (internal GPU case)

One JSON POST per call. An internal serving team implements these three
endpoints and sets the URLs; nothing else in the pipeline moves.

### `SAM3_HTTP_URL`

```
POST {"image_b64": <base64 image>, "prompt": <text>}
POST {"image_b64": <base64 image>, "box": [x1, y1, x2, y2]}
->   {"rle": [<COCO object RLE>, ...], "scores": [float, ...]}
```

### `GEOMETRY_HTTP_URL`

```
POST {"inputs": [<data URI>, ...], **flags}     # replicate MapAnything input schema
->   replicate MapAnything response schema (per-frame geometry)
```

### `MOGE_HTTP_URL`

```
POST {"image_b64": <base64 image>}
->   {"ply_b64": <base64 PLY>, "intrinsics": [[...]], "fov_x_deg": float}
```

### RLE format

COCO object form everywhere: `{"size": [H, W], "counts": [...]}` —
column-major, zero run first. Reference codecs:
`ehs_spatial.providers.sam3.decode_coco_rle` / `encode_coco_rle`.

## Guarantees

- Schemas above are frozen; tests in `tests/test_backends.py` pin the
  routing (each env value dispatches to the right transport, unknown
  values fail loudly for SAM).
- Every router imports its heavy client (`fal_client`, `replicate`,
  `modal`) lazily inside the branch — an internal deployment using only
  `http` needs none of them installed.
