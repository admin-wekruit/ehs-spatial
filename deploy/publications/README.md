# 报告站（发布层，WP5）

`panoptes run` 的 S7 把一次运行导入平台、发布、导出成一个**发布目录** `catalog/<publicationId>/`（`ehs-spatial/scripts/export_platform_publication.py` 的格式）。
看报告需要三样东西，放在同一个 origin 下：

| 东西 | 来源 | 查看器怎么请求 |
|---|---|---|
| 查看器 bundle `app.html` + `assets/` | `ehs-spatial/web` 一次构建：`VITE_PUBLICATION_ID=/ VITE_API_ORIGIN= npx vite build --outDir $PANOPTES_WEB_ROOT`（`VITE_API_ORIGIN` 空 = 同源 `/api/`） | 直接打开 `<base>/app.html#/reports/<publicationId>` |
| 发布 API（只读） | `ehs-spatial/scripts/prepare_publication_site.py --catalog $PANOPTES_PUBLICATION_CATALOG --output $PANOPTES_PUBLICATION_HTTP` 的输出 + 发布服务进程 | `/api/publications`、`/api/publications/<id>`、`/api/revisions/<id>…`、`/api/assets/<id>/content` |
| 测量层 | 流程写到 `$PANOPTES_PAGES`：`<publicationId>.json`、`pipeline-*.jpg`、网格 `.bin` | 相对路径 `./measurement-layer/<publicationId>.json` |

**`prepare_publication_site.py` 的输出不是静态目录树**：它是 `index.json`（路由 → 文件、资产 → catalog 内路径）+ 按内容寻址的 `<sha256>.json.gz` 响应体，
资产本身仍在 catalog 里；把路由映射成 HTTP 的是发布服务进程（`ehs-spatial/containers/onprem/serve_publications.py`，= Modal 上
`modal_apps/publication_site.py` 的 `create_app`）。所以两种托管方案里，发布 API 都是那一个小容器；"静态"指查看器 bundle 和测量层。

## 方案 1（建议，2026-10-05 已离线验证 29/29）：nginx 容器 — `compose.nginx.yml` + `nginx.conf`

= `ehs-spatial/containers/onprem/` 的栈去掉平台 API + Postgres（S7 由流程机做）。`web`（nginx）同源服务 bundle、`measurement-layer/`，
并把 `/api/` 代理到 `publications`；不需要 CORS，任何主机名都行，每次发布不用重建 bundle。

```sh
# 一次：镜像 + bundle（联网机器；镜像可用 ehs-spatial/scripts/onprem/airgap.sh 打包离线装）
cd ehs-spatial && docker build -f containers/onprem/platform.Dockerfile -t panoptes-platform:onprem .
cd web && npm ci && VITE_PUBLICATION_ID=/ VITE_API_ORIGIN= node node_modules/vite/bin/vite.js build --outDir $PANOPTES_WEB_ROOT --emptyOutDir
mkdir -p $PANOPTES_PUBLICATION_CATALOG $PANOPTES_PUBLICATION_HTTP $PANOPTES_PAGES     # .env 第 4 节的三个目录（PANOPTES_PAGES 不要放在 bundle 的 --outDir 里再 --emptyOutDir）

# 起站（仓库根目录，同一份 .env）
docker compose --env-file .env -f deploy/publications/compose.nginx.yml up -d
curl -s http://localhost:8080/api/publications | head -c 200          # 空 catalog 也能起：{"items":[]}
# 每次 panoptes run 之后（S7 已导出到 catalog）
uv run --env-file .env python $PANOPTES_WORKCELL/scripts/prepare_publication_site.py --catalog $PANOPTES_PUBLICATION_CATALOG --output $PANOPTES_PUBLICATION_HTTP
docker compose --env-file .env -f deploy/publications/compose.nginx.yml restart publications
```

`.env`：`PANOPTES_PUBLIC_BASE_URL=http://<这台机器>:8080`（compose 固定映射 8080；改端口两处一起改）。

## 方案 2：S3 静态托管（bundle + 测量层进桶，发布 API 仍是容器）

适合已有对象存储静态站点能力（AWS S3 website、Ceph RGW website）的环境。**MinIO 没有 website 模式**——MinIO 前面还得放 nginx，那就直接用方案 1。

```sh
# 桶与策略（bucket-policy.json：把 BUCKET / PREFIX 换掉；S3 兼容端点都加 --endpoint-url $AWS_ENDPOINT_URL）
aws s3 mb s3://BUCKET
sed 's/BUCKET/BUCKET/; s/PREFIX/reports/' deploy/publications/bucket-policy.json > /tmp/policy.json
aws s3api put-bucket-policy --bucket BUCKET --policy file:///tmp/policy.json
aws s3 website s3://BUCKET/ --index-document app.html --error-document app.html

# 发布 API：只起 publications 容器并暴露 8793，允许静态站 origin 跨域
PANOPTES_PUBLICATION_ORIGINS=http://BUCKET.s3-website-REGION.amazonaws.com \
  docker compose --env-file .env -f deploy/publications/compose.nginx.yml up -d publications    # 再 -p 8793:8793 或 ports 覆盖

# bundle 指向发布 API 的 origin（这一步和方案 1 唯一不同），然后同步
VITE_PUBLICATION_ID=/ VITE_API_ORIGIN=http://<api 主机>:8793 node node_modules/vite/bin/vite.js build --outDir $PANOPTES_WEB_ROOT --emptyOutDir
aws s3 sync $PANOPTES_WEB_ROOT/assets s3://BUCKET/reports/assets --cache-control "public,max-age=31536000,immutable"
aws s3 cp   $PANOPTES_WEB_ROOT/app.html s3://BUCKET/reports/app.html --cache-control no-cache --content-type text/html
# 测量层：每次 panoptes run 之后
aws s3 sync $PANOPTES_PAGES s3://BUCKET/reports/measurement-layer --cache-control no-cache
```

布局（`PREFIX=reports`）：

```
s3://BUCKET/reports/app.html
s3://BUCKET/reports/assets/*                      # 不可变，长缓存
s3://BUCKET/reports/measurement-layer/<publicationId>.json, pipeline-<eid>.jpg, *.bin
```

`.env`：`PANOPTES_PUBLIC_BASE_URL=http://BUCKET.s3-website-REGION.amazonaws.com/reports`。
发布目录本身（`publications/<id>/`）走 `panoptes artifacts push`（WP1，`PANOPTES_BLOB_ROOT=s3://…` 下的 `publications/` 前缀）进同一个桶备份，
发布 API 读的仍是本机 catalog（`panoptes artifacts pull publications/<id>` 可恢复）。静态托管下访客反馈的 POST 不可用（只读站点）。

## `PANOPTES_PUBLIC_BASE_URL` 怎么进查看器

| 读它的 | 用法 |
|---|---|
| `panoptes run`（`ehs_spatial/cli.py`） | 结束时打印 `<base>/app.html#/reports/<publicationId>`；`panoptes status` 列出每次运行的地址 |
| `S3BlobStore.url()`（ehs-spatial，WP1） | 资产的公开 URL：`<base>/api/assets/<id>/content`（方案 1）或 presigned（桶私有时） |
| 查看器 bundle | **不读它**。bundle 在构建时只认 `VITE_API_ORIGIN`（方案 1 空 = 同源；方案 2 = 发布 API 的 origin）；测量层按相对路径 `./measurement-layer/` 取，所以 bundle 和测量层必须在同一 origin + 前缀下，这正是 `PANOPTES_PUBLIC_BASE_URL` 指的地方 |

## 测量层

`build_swap_layer.py` 把层写到 `$PANOPTES_PAGES`（env.sh 默认 `$SWAP_SCRATCH/measurement-layer`；`.env` 里直接指到 nginx 服务的目录，省一次拷贝）。
一个层只对它命名的 revision 生效：把同一次采集的发布搬到另一个平台实例（id 变了）时，用 `ehs-spatial/scripts/onprem_layer.py --dry-run` 校验并 rebase
（`ehs-spatial/containers/onprem/README.md` §5），不要手改 id。
