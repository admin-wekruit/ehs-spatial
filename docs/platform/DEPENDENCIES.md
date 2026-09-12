# Dependency and model release evidence

Checked 2026-09-12. Exact application versions are in `uv.lock` and `web/package-lock.json`; `dependency-inventory.json` records installed Python and locked npm declarations. Metadata is inventory, not a complete license clearance.

| Component | Evidence | State |
| --- | --- | --- |
| React, Vite, Deep Chat, JDM Editor | Installed/locked packages declare MIT | Integrated locally; retain their notices when distributing builds |
| FastAPI, uvicorn, boto3, pypdf | Installed declarations: MIT, BSD-3-Clause, Apache-2.0, BSD-3-Clause respectively | Integrated locally |
| psycopg | Installed declaration LGPL-3.0-only | Unmodified library dependency; distribution obligations remain applicable |
| ZEN | [Official license](https://github.com/gorules/zen/blob/master/LICENSE) is MIT text | Python binding executed in local policy checks |
| MapAnything Apache | [Official model card](https://huggingface.co/facebook/map-anything-apache) declares Apache-2.0 | Target adapter present. This does not clear the separate original noncommercial weights or the product quality gate |
| MoGe-3 | [Official weight repository](https://huggingface.co/Ruicheng/moge-3-vitl) declares MIT | Target adapter present; exact runtime and quality evaluation remain unverified |
| SAM 3D Objects | [Official SAM License](https://github.com/facebookresearch/sam-3d-objects/blob/main/LICENSE), a custom license | Do not label Apache/MIT. Gated access, runtime dependencies, pose/pointmap fixtures and quality gate remain unverified |
| Blender | Official local Blender 4.5.9 executed and reopened export fixtures | Server-side executable; user does not install it to use the web editor |

Existing imported research meshes remain labelled imported proposals with source provenance. Importing a mesh neither changes its provenance nor demonstrates a commercially cleared model pipeline.

No production model manifest is marked approved in this implementation. Audited immutable GPU images, exact weight/code pins, complete runtime dependency notices, and measured candidate comparisons are required before enabling a product model. Missing evidence blocks the stage; the worker does not switch to another model.

## S3 subset verification

Supabase's current [PUT route](https://github.com/supabase/storage/blob/755986d5a8d915296d0fedafe03011e2616ae563/src/http/routes/s3/commands/put-object.ts) and [S3 handler](https://github.com/supabase/storage/blob/755986d5a8d915296d0fedafe03011e2616ae563/src/storage/protocols/s3/s3-handler.ts) do not enforce the conditional PUT/checksum fields previously assumed by the adapter. The application instead validates content-derived keys before writing, reuses verified existing bytes, writes only after a definite missing-object response, and verifies downloaded bytes before DB registration. Concurrent valid writes for the same SHA256 key contain the same bytes. This is application content immutability, not provider WORM or object lock.

The client disables optional automatic SDK checksums with the documented [Boto3 configuration](https://docs.aws.amazon.com/boto3/latest/guide/configuration.html); all transfers still use SigV4 and application SHA256 verification. `test_platform_storage.py` checks actual SDK request shapes without network access. Live hosted storage verification remains pending.
