# ehs-spatial（Panoptes）

工位照片 → 3D 重建（对象、盒、尺寸、离地高、不确定度）→ 测量报告。企业版交接 1.0.0-rc1（2026-10-07）：一个仓库、客户自己的 GPU 与存储。

**读的顺序（新 session 从这里开始，不要跳）**

1. `docs/STATE.md` — 系统现在是什么：流程、服务与端口、已发布结果、什么不在交付里、正在做的研究。有冲突以它为准。
2. `HANDOFF.md` — 怎么跑：clone（`-b main`）、`.env`、GPU 机 `make up GPU=a|b`、`make smoke`、`panoptes run --cell 090`、验收 A1–A6。
3. `research/module-swap-2026-10-07/REPRODUCE-PROMPT.md` — 怎么复现已发布的数字（本机 GPU，不用 Modal）。
4. `docs/MILESTONES.md` — 阶段与里程碑；`CHANGELOG.md` — 变更。

**其它目录**：`docs/research/`（研究与提案，未交付；索引 `docs/research/README.md`）；`docs/archive/`（历史，只读；索引 `docs/archive/README.md`）；
`research/module-swap-2026-10-07/notes/`（冻结实验记录）；`docs/platform/OPERATIONS.md`（平台运维）；`docs/BACKENDS-v1.md` / `docs/BACKENDS.md`（服务契约）；`docs/STORAGE.md`。

**给 agent / 自动化 session 的硬规则**：`CLAUDE.md`（同 `AGENTS.md`）。

Quick start:

```bash
git clone -b main https://github.com/admin-wekruit/ehs-spatial.git && cd ehs-spatial
make env            # -> .env; fill credentials + jump endpoints; make check-env
make up GPU=a       # on GPU card A: sam3d :8805 + sam3 :8801      (card B: geometry-mvs :8804 + moge :8803 + mapanything :8802)
make smoke
make run CELL=090   # ~40 min; prints the report URL
```
