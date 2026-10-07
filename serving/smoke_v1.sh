#!/usr/bin/env bash
# Smoke of the v1 contract (docs/BACKENDS-v1.md) against a running service: healthz, info, and for sam3d one synchronous call
# plus one job round trip (POST /v1/sam3d/jobs -> 202 -> GET /v1/jobs/{id} until done). The input is the synthetic scene of
# modal_apps/sam3d_research.debug (tests/contract/inputs.py), so it also works against the real model.
#   PANOPTES_SERVICE_API_KEY=... serving/smoke_v1.sh [URL]       URL default http://127.0.0.1:8805 (a sam3d service)
#   a geometry-mvs URL (…:8804) gets healthz + info only.
set -euo pipefail
URL=${1:-http://127.0.0.1:8805}
KEY=${PANOPTES_SERVICE_API_KEY:?set PANOPTES_SERVICE_API_KEY}
here=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
H=(-sS -H "X-API-Key: $KEY" -H 'Content-Type: application/json')
json() { "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d$1)"; }

echo "== healthz"; health=$(curl "${H[@]}" "$URL/healthz"); echo "$health"
model=$(echo "$health" | json "['model']")
echo "== info";    curl "${H[@]}" "$URL/v1/info" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print({k: d[k] for k in ('model_id','model_revision','code_sha','seed_policy')}, list(d['weights_sha256']))"
echo "== no key -> 401"; code=$(curl -s -o /dev/null -w '%{http_code}' "$URL/healthz"); [ "$code" = 401 ] || { echo "expected 401, got $code"; exit 1; }
if [ "$model" != sam3d ]; then echo "$model: healthz + info ok"; exit 0; fi

body=$(mktemp); trap 'rm -f "$body"' EXIT
PYTHONPATH="$here/../tests/contract" "$PY" -c "import inputs, json; print(json.dumps(inputs.sam3d_body(size=${SMOKE_SIZE:-256})))" > "$body"
echo "== POST /v1/sam3d (sync)"
curl "${H[@]}" -X POST "$URL/v1/sam3d" --data-binary "@$body" | "$PY" -c "import json,sys; d=json.load(sys.stdin); assert 'mesh_npz_b64' in d, d; print({k: d[k] for k in ('vertices','faces','seconds','gpu','pins')})"
echo "== POST /v1/sam3d/jobs -> poll"
sed 's/"input_sha256": "\([0-9a-f]\{63\}\)[0-9a-f]"/"input_sha256": "\1f"/' "$body" > "$body.job"  # a second key: a fresh job, not the cached result
job=$(curl "${H[@]}" -X POST "$URL/v1/sam3d/jobs" --data-binary "@$body.job"); echo "$job"; rm -f "$body.job"
id=$(echo "$job" | json "['job_id']")
for _ in $(seq 1 "${SMOKE_WAIT:-120}"); do
  rec=$(curl "${H[@]}" "$URL/v1/jobs/$id"); status=$(echo "$rec" | json "['status']")
  case $status in
    done)  echo "$rec" | json "['result']['vertices'], d['result']['faces'], d['seconds'], d['gpu'], d['model_info']['model_revision']"; echo "smoke ok"; exit 0;;
    error) echo "$rec"; exit 1;;
  esac
  sleep 1
done
echo "job $id still $status"; exit 1
