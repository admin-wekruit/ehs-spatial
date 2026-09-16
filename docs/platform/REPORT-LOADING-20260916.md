# Report loading correction — 2026-09-16

Affected public report: `d6c2d4d3-4769-4526-a0f7-73de34fa2f5b`.

## Cause

Initial rendering fetched the entire publication (97,989,774 decoded bytes),
then the project with a second copy of its scene (21,474,588 bytes).
The publication included 76 MB of editing history. React formatted the complete
inverse/forward operations into hidden `pre` elements even when their containing
details were closed, repeating that work on subsequent report renders.

The public service also parsed all historical bundles and verified every asset
each time its container started. This verification belongs in release preparation
for the immutable deployment, rather than on a reader's first request.

## Change

- The shared report-view endpoint retains the full current scene, evidence and
  asset manifest, plus lightweight edit summaries. It excludes full edit bodies.
- Complete audit events remain frozen and downloadable individually on click.
  The original publication endpoint remains unchanged.
- Release preparation validates every existing publication and blob, then writes
  compressed response files and a small serving index. Runtime streams those
  verified files instead of parsing historical catalogs or serializing each GET.
- No object, mesh, CAD contour or photograph was removed to reduce loading cost.

## Measurements

Single measured report; these are not p95 or cross-device guarantees.

| Initial report metadata | Before | After |
| --- | ---: | ---: |
| Decoded JSON, publication + project | 119,464,362 B | 21,563,681 B |
| gzip HTTP body, compression level 3 | 31,132,129 B | 5,741,324 B |
| Full editing history on initial load | 5 complete events | 0 complete events |

Real-catalog preparation: 59.433 s, 984 routes, 875 asset IDs.
Local prepared application construction: 0.012 s (excludes Python imports).
First HTTP request after Modal redeployment: 6.828 s to first byte, 8.041 s
complete. Subsequent request: 0.292 s to first byte, 0.882 s complete. These HTTP
timings cover metadata only, not model downloads or GPU upload.

The model view still downloads approximately 134 MB of geometry; this change
does not claim that all geometry loads in the metadata request time.

## Checks

- `tests/check_report_loading.py`: real API DTO response; no runtime catalog
  scan; no inline inverse history; lossless on-demand events; unchanged original
  routes; gzip, identity, HEAD, ETag, CORS, read-only writes, corrupt asset rejection.
- `tests/check_publication_site.py`: immutable revision sharing/conflicts,
  history, latest project, old assets, content verification, ranges and CORS.
- `tests/test_publication_feedback.py`: 19 passed.
- Web TypeScript/Vite build and existing interaction/viewer scheduling checks pass.
- Local computer-use check: 26/26 projection records; select material cart across
  photo/CAD/model; drag rotates scene; no console warnings/errors. Total page text
  including closed details after selection was 82,161 characters, of which 21,255
  were in `pre` elements, rather than tens of megabytes of audit JSON.
- Public browser check after deployment: refresh to visible report heading in
  2.307 s in the existing browser session (not a cold-cache benchmark); all 24
  independent models subsequently loaded. Actual drag changed scene orientation;
  selecting the cart retained its identity in CAD and photograph; CAD remained
  26/26. No console errors or warnings were reported during this check. Pages
  deployment `95ef8c2251ff306c1c149f0b272367255466b4eb` succeeded.

Database-backed publication tests were skipped without `PANOPTES_TEST_DATABASE_URL`;
the changed API projection is covered with an injected repository in the runnable
loading check above. Existing frozen catalog contents were verified during build.
