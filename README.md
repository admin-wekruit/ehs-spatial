# EHS Spatial MVP

A local, single-process inspection workbench that turns exactly four workcell
images into approximate 3D fence-clearance evidence, a deterministic demo
assessment, and fact-grounded follow-up answers. The fixed `0.6 m` criterion is
a demo rule, not an official EHS standard. This is not a certified safety tool,
CAD/SLAM system, production monitoring service, or replacement for an EHS
professional.

## Stack and provider IDs

- Python `3.12`; Gradio `6.20.0`; NumPy `2.5.1`; Pydantic `2.13.4`;
  Open3D `0.19.0`; Shapely `2.1.2`; Pillow `12.3.0`.
- Provider clients: `replicate==1.0.7`, `fal-client==1.0.0`, and
  `google-genai==2.11.0`.
- Replicate Map Anything:
  `vufinder/map-anything:bb68c254a65d3ce6b173909181d2dfbd044300b3ebca25ab63f07aa7eb1eebff`
  using the `map-anything-apache` checkpoint.
- fal SAM 3.1 endpoint: `fal-ai/sam-3-1/image-rle`.
- Gemini model: `gemini-3.5-flash` through the Interactions API.

Provider pages:
[Map Anything](https://replicate.com/vufinder/map-anything),
[SAM 3.1 image RLE](https://fal.ai/models/fal-ai/sam-3-1/image-rle), and
[Gemini 3.5 Flash](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash).

## Install and run

```bash
uv sync --extra dev --frozen
cp .env.example .env
```

Fill `REPLICATE_API_TOKEN`, `FAL_KEY`, and `GEMINI_API_KEY` in `.env`, then use
uv's native environment-file support:

- `REPLICATE_API_TOKEN`: [Replicate API tokens](https://replicate.com/account/api-tokens)
- `FAL_KEY`: [fal API keys](https://fal.ai/dashboard/keys)
- `GEMINI_API_KEY`: [Google AI Studio API keys](https://aistudio.google.com/apikey)

```bash
uv run --env-file .env python app.py
```

No custom `.env` loader is used.

## Reproducible EHS evaluation

Generate the four calibrated CPU scenes and validate the deterministic
SceneMap/rule path without any provider calls:

```bash
uv run python scripts/ehs_eval.py generate --output outputs/ehs_v1
uv run python scripts/ehs_eval.py offline --pack outputs/ehs_v1
```

Then run one explicit paid end-to-end case:

```bash
uv run --env-file .env python scripts/ehs_eval.py live \
  --pack outputs/ehs_v1 --case ladder_050 --live
```

See [the calibrated eval guide](eval/README.md) for truth boundaries, exact
artifacts, call counts, and the opt-in four-case benchmark command.

## Capture and evidence

Upload four local images in the labeled front/right/rear/left slots. Capture the
same workcell from four sides with useful overlap, stable lighting, a visible
factory floor, safety fence, and staged movable objects. Enter the measured lens
height above the floor. The app intentionally rejects incomplete four-view runs.

Each successful run remains local under `runs/{run_id}/`, including copied
inputs, provider geometry, masks, `point_cloud.glb`, `observations.json`,
`scene.json`, `assessment.json`, `topdown.png`, and `chat.jsonl`. Geometry and
rendering run on CPU; CUDA is neither required nor configured.

There is no fallback path. A provider, decoding, grounding, or artifact error is
shown as an error and the run does not invent a result.

## Cost, privacy, and data restrictions

One analysis makes **1 paid Map Anything call, 28 paid fal segmentation calls
(7 object labels × 4 frames; the floor is fitted geometrically, not segmented)
plus up to 3 extra calls per frame-label whose canonical prompt returns nothing
(synonym fallback), and at least 1 paid Gemini interaction**; each chat question
adds another Gemini interaction. fal listed SAM 3.1 at `$0.01/request` on
2026-07-15, making segmentation roughly `$0.28–0.5/case`. Review current
provider pricing before use.

Gemini requests use `store=True` so follow-up questions can chain through the
stored interaction ID. Provider retention therefore applies. Do not upload Tesla
images or any real factory, employee, customer, confidential, or regulated data
until the relevant vendor terms, retention policy, and internal approvals have
been reviewed and accepted.

## Tests

Offline verification never calls a provider:

```bash
uv run pytest -q
uv run python -m compileall -q app.py ehs_spatial scripts tests
uv lock --check
uv pip check
```

The live smoke is paid and opt-in. It requires the three API variables from
`.env` and exactly four explicit local image variables:

```bash
EHS_LIVE_SMOKE=1 \
EHS_SMOKE_IMAGE_1=/absolute/path/front.jpg \
EHS_SMOKE_IMAGE_2=/absolute/path/right.jpg \
EHS_SMOKE_IMAGE_3=/absolute/path/rear.jpg \
EHS_SMOKE_IMAGE_4=/absolute/path/left.jpg \
uv run --env-file .env pytest -q tests/test_live_smoke.py
```
