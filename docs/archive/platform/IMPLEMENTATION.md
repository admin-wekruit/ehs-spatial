# Panoptes platform implementation

Approved source: user's detailed implementation specification, 2026-09-12.
Product source is this ehs-spatial worktree. Existing public pages remain archives until all release gates pass. No dual writes or provider fallback. No accounts, organizations, billing product, or generic plugin framework.

## Shared contracts (schemaVersion 1)

Use camelCase on the wire and in scene documents; UUID business IDs. New Python modules live in `ehs_spatial/platform/`. Existing geometry, renderer and provider code are reused. Frontend lives in `web/`, builds `app.html` with hash routes. API is FastAPI with injected repository/blob/executor/providers; PostgreSQL via psycopg, versioned SQL; local and S3 blobs; local and Modal execution.

Scene document:
```
{schemaVersion:1, captureId:null, target:"scene"|"standalone_object",
 coordinateFrames:[{id, convention:"opencv", scale:{status:"uncalibrated"|"model_estimated"|"operator_anchored",nativeToMeters:null,sourceRefs:[]},ground:null}],
 cameras:[{id,imageId,coordinateFrameId,width,height,K,cameraToWorld}],
 observations:[{id,revision:1,imageId,originalPixelBox:[x0,y0,x1,y1],maskAssetId:null,pixelMapping:[],labelEvidence:[],geometrySupport:null,sourceRefs:[]}],
 entities:[{id,label,observationRefs:[],associationState:"association_pending"|"confirmed",representations:[],currentModelTransform:null,measurements:{},groupId:null,lineage:[]}],
 assets:[], annotations:[]}
```
Representations: `{id,kind:"observed_surface"|"point_cloud"|"generated_mesh"|"primitive",assetId:null,coordinateFrameId,transform,primitive:null,placementState:"unconfirmed"|"confirmed",sourceRefs:[]}`. Transform: `{coordinateFrameId,position:[x,y,z],quaternion:[x,y,z,w],scale:[x,y,z]}`. Quaternions normalized, finite values, positive scales. Unknown geometry and measurements are null, never fictitious coordinates/metres.

Selection lives in host app: `{projectId,revisionId,entityId,observationId,cameraId}`. Entity exists and remains selectable without representations. Source image/3D/CAD/plan select same identity; no mesh-only filtering.

Commands: `{requestId,branchId,baseRevisionId,operations}`. Operations discriminated by `type`: setTransform (fields above + entityId), setLabel(entityId,label), setVisibility(entityId,visible), setMaterial(entityId,material), addEntity(entity), removeEntity(entityId), addObservation(observation,entityId), setPrimitive(entityId,primitive), mergeEntities(entityIds,survivorId), splitEntity(entityId,groups), setCalibration(coordinateFrameId,scale), addAnnotation(annotation), removeAnnotation(annotationId). Server generates inverse operations. Evidence and model/planning values remain separate. Commit returns `{revision,editBatch,headAdvanced:true}`. Conflict 409 includes base/current IDs, no implicit merge.

Project `{id,title,defaultBranchId,createdAt,forkSourceRevisionId}`. Branch `{id,projectId,kind,title,sourceRevisionId,headRevisionId}`. Revision `{id,projectId,branchId,parentRevisionId,sourceRevisionId,document,documentSha256,label,createdAt}`. Lists `{items:[...]}`. Create/fork response `{project,branch,revision}`. GET project returns same + branches. Errors `{error:{code,params}}`, never raw credentials/provider exceptions. Jobs and public data never expose capabilities.

Browser creates 32 random bytes as `pcap_v1_<base64url>`, stores pending request in IndexedDB BEFORE send; Authorization `Capability <token>`. Server stores only SHA. Create/fork unique(capability SHA, requestId), body mismatch409. Fork auth is NEW project's capability; source public. All original project writes require matching capability. Reads public. No capability in URLs/model context/logs.

API routes /api/projects (GET POST), /api/projects/{pid} GET, captures POST multipart(files,target,requestId,branchId,baseRevisionId), branches POST, revisions GET, /api/revisions/{rid} GET, edits POST, forks POST, jobs POST, /api/jobs/{jid} GET + /cancel POST, agent-turns POST GET, publications POST, /api/publications GET + /{id} GET, /api/assets/{id} GET (asset metadata+url). Job POST `{requestId,branchId,baseRevisionId,kind,inputs,config}`. Publication `{requestId,sceneRevisionId,evaluationIds:[],reviewIds:[],title}`. Agent request `{requestId,conversationId,branchId,baseRevisionId,message,entityId:null,observationId:null,box:null,language}`. Names are not identities.

10 core tables: projects,scene_branches,captures,scene_revisions,edit_batches,assets,jobs,model_calls,publications,agent_turns. Composite ownership foreign keys. Branch lock + CAS for edits; immutable revisions/publications. Jobs are outbox pending_dispatch→queued→running→succeeded/incomplete/failed/outcome_unknown/cancelled; cancel_requested separate. Attempts fenced. Reserve model call before external execution; unknown paid outcome must not be retried. No paid budget configuration = no paid call. Asset upload and hash check before DB reference. Late result saved, no head overwrite. Batch children assets only, parent one revision.

## Work packages
- W0: preserve inherited changes/hashes, baseline tests.
- W1/W2: migrations, repository transaction contracts, DTO/API, immutable storage, persistent jobs, capabilities, branches, publication/fork, budgets.
- W3: generic normalization/evidence/entity association/cache/model adapters; preserve small objects; no floor prerequisite for standalone.
- W4/W7: React workbench/report/history, shared renderer/selection/edit lifecycle, bilingual/mobile, distinct playgrounds, visitor fork.
- W5: Deep Chat + scoped persistent agent tool loop, proposals through same edit endpoint.
- W6: generic box/cylinder and version-bound GLB/Blender exports, reopen check and camera representability.
- W8: JDM/Zen policy source/revision/evaluation/review/evidence workflow; sole shared Python measurement evaluator.
- W9: migration, swap drills, browser E2E, fixed model experiments/licenses, release gates.

## Mandatory verification
Idempotency mismatch and lost create response; cross-project writes rejected; concurrent branch CAS; independent planning branches; old/duplicate/cancelled worker result; no duplicate paid call; immutable blob hash/size; missing mesh selectable; overlap picks smallest eligible observation; rapid A→B and late asset; arbitrary object count; no floor standalone; camera rotation/skew and nonunit scale; undo/redo append; GUI/Agent same commit; report immutable/fork isolated; applicability unknown not PASS; m² vs m; no hidden small-object filtering; website hash refresh; renderer dispose; mobile preview before GLB.

Publication cutover requires real model + license + quality + replacement + browser gates, not just unit tests. Preserve historical fixed experiments; corrected product baseline separately reruns old and candidate same metrics. No claimed metrology or speed without evidence. Detailed user's specification remains authoritative where this coordination summary omits detail.
