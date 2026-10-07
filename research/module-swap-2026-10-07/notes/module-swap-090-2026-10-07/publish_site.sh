#!/bin/zsh
# Publish locally exported publications to the live report service (Modal app behind the Pages viewer): the live catalog is the
# three published reports + the listed variants; the catalog is cloned (APFS) into one folder, verified / pre-encoded by
# scripts/prepare_publication_site.py, and deployed with the platform's modal_apps/publication_site.py under the live app name
# (the viewer's API base is https://wekruit-livekit-agents--panoptes-publications-estop-web.modal.run).
#   ./publish_site.sh VARIANT [VARIANT ...]      VARIANT = a folder of .platform/swap-20261007 (v2-mvs-fill-sam3d, ...)
set -e
SP=${SWAP_SCRATCH:-/private/tmp/claude-501/-Users-adam-Desktop-panoptes-public/1fd9a1db-e580-4bfc-8110-119a1cc38a99/scratchpad}
PLAT=${PANOPTES_PLATFORM:-/Users/adam/Desktop/Tesla/panoptes-platform}
PY=${PY:-$PLAT/.venv/bin/python}
LIVE="25686138-ba21-475f-81f0-c2818bc4ab6a cd84d3fb-7d1f-4736-8ffa-d44855e59fab 4b58dbd2-3846-47f2-af97-57eaa108753c"
CAT=$SP/publication-catalog-live
HTTP=$SP/publication-http-live
filt() { grep -viE "capabilit|token|secret|password|postgres://"; }

IDS="$LIVE"
for V in "$@"; do
  ID=$($PY -c "import json; print(json.load(open('$PLAT/.platform/swap-20261007/$V/result.json'))['publicationId'])")
  test -d $PLAT/.platform/publication-catalog/$ID
  IDS="$IDS $ID"; echo "$V -> $ID"
done
rm -rf "${CAT:?}" "${HTTP:?}"; mkdir -p $CAT
for ID in ${=IDS}; do cp -c -R $PLAT/.platform/publication-catalog/$ID $CAT/$ID; done
echo "catalog: $(ls $CAT | wc -l | tr -d ' ') publications"
cd $PLAT
$PY scripts/prepare_publication_site.py --catalog $CAT --output $HTTP 2>&1 | filt | tail -2
PANOPTES_PUBLICATION_APP=panoptes-publications-estop PANOPTES_PUBLICATION_CATALOG=$CAT PANOPTES_PUBLICATION_HTTP=$HTTP \
  $PLAT/.venv/bin/modal deploy modal_apps/publication_site.py 2>&1 | filt | grep -vE "^\s*$|Created mount|^│|^├|^└" | tail -6
$PY -c "import time; time.sleep(8)"
curl -s -m 30 "https://wekruit-livekit-agents--panoptes-publications-estop-web.modal.run/api/publications" | $PY -c "
import sys, json; d = json.load(sys.stdin); items = d if isinstance(d, list) else d.get('publications', d.get('items', []))
print('live now:', len(items)); [print(' ', x.get('id', '')[:8], '|', (x.get('title') or '')[:80]) for x in items]"
