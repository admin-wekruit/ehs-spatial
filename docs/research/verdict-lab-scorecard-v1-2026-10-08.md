# 判定层 lab 记分卡 v1（2026-10-08 下午）

命令（仓库根目录；`runs/` 不入库，随时可重跑）：

```
PANOPTES_FAKE_MODEL=1 PANOPTES_WORKCELL=$PWD .venv/bin/python -m ehs_spatial.cli verdict matrix --benchmark ehs_spatial/verdict/benchmark/v0 \
  --configs ehs_spatial/verdict/configs/{baseline,python-engine,stpl-k1,clause-kg-ts,funclib-ts,html-report}.yaml \
  --runs-dir runs/lab-v1 --jobs 2 --run-id lab-v1
```

和 v0 的差别：引擎 `clingo@2` / `python@2`、规则包 `handwritten@2`（包围规则没有 coverage 时出 CANNOT_DETERMINE，经已观察地面的缺口出 FAIL）、
`relations@2`（光幕 / 区域扫描仪的 `vertical` / `horizontal`）；记分卡多了两列：规则包状态（compiled / needs_input / vocabulary_gap / refused =
L5 方案的编译率）和 LLM 调用数。金标仍是试跑判定（provisional），键是规则 id + 对象，所以换了规则包（function-library）的行按定义对不上金标——那两行看的是
编译率和判定分布，不是一致率。

| run | item | L2 | L3 | L4 | L5 | L6 | L7 | rules c/n/g/r | PASS | FAIL | NEEDS_M | NEEDS_I | CANNOT | gold | diff vs previous row |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lab-v1-baseline-030 | 030 | relations@2 | none@1 | none@1 | handwritten@2 | clingo@2 | markdown@1 | 5/1/0/0 | 9 | 1 | 1 | 6 | 1 | 18/18 | — |
| lab-v1-baseline-090 | 090 | relations@2 | none@1 | none@1 | handwritten@2 | clingo@2 | markdown@1 | 5/1/0/0 | 10 | 2 | 2 | 7 | 1 | 22/22 | item |
| lab-v1-python-engine-030 | 030 | relations@2 | none@1 | none@1 | handwritten@2 | python@2 | markdown@1 | 5/1/0/0 | 9 | 1 | 1 | 6 | 1 | 18/18 | L6 clingo→python |
| lab-v1-python-engine-090 | 090 | relations@2 | none@1 | none@1 | handwritten@2 | python@2 | markdown@1 | 5/1/0/0 | 10 | 2 | 2 | 7 | 1 | 22/22 | item |
| lab-v1-stpl-k1-030 | 030 | relations@2 | stpl@1 | none@1 | handwritten@2 | clingo@2 | markdown@1 | 5/1/0/0 | 9 | 1 | 1 | 6 | 1 | 18/18 | L3 none→stpl; L6 k 2→1 |
| lab-v1-stpl-k1-090 | 090 | relations@2 | stpl@1 | none@1 | handwritten@2 | clingo@2 | markdown@1 | 5/1/0/0 | 11 | 2 | 1 | 7 | 1 | 21/22 | item |
| lab-v1-clause-kg-ts-030 | 030 | relations@2 | none@1 | clause-kg@0 | handwritten@2 | clingo@2 | markdown@1 | 5/1/0/0 | 9 | 1 | 1 | 6 | 1 | 18/18 | L4 none→clause-kg (ISO v0 + TS) |
| lab-v1-clause-kg-ts-090 | 090 | relations@2 | none@1 | clause-kg@0 | handwritten@2 | clingo@2 | markdown@1 | 5/1/0/0 | 10 | 2 | 2 | 7 | 1 | 22/22 | item |
| lab-v1-funclib-ts-090 | 090 | relations@2 | none@1 | clause-kg@0 | function-library@0 | clingo@2 | markdown@1 | 9/9/0/6 | 0 | 0 | 0 | 22 | 0 | n/a | L5 handwritten→function-library |
| lab-v1-funclib-ts-030 | 030 | relations@2 | none@1 | clause-kg@0 | function-library@0 | clingo@2 | markdown@1 | 9/9/0/6 | 0 | 0 | 0 | 18 | 0 | n/a | item |
| lab-v1-html-report-030 | 030 | relations@2 | none@1 | none@1 | handwritten@2 | clingo@2 | html@1 | 5/1/0/0 | 9 | 1 | 1 | 6 | 1 | 18/18 | L7 markdown→html |
| lab-v1-html-report-090 | 090 | relations@2 | none@1 | none@1 | handwritten@2 | clingo@2 | html@1 | 5/1/0/0 | 10 | 2 | 2 | 7 | 1 | 22/22 | item |

读法：

- 基线 / python 引擎 / stpl-k1 / clause-kg / html 五组数字和 v0 完全一样（引擎 v2 只改了包围规则没有 coverage 的读法，两个工位本来就没有 coverage，以前也是 CANNOT_DETERMINE；html@1 只换报告形式）。
- **function-library@0**（无 LLM，条款 → 规则）在 090 检索到的 24 条条款里：9 条编译（ISO 13857 4.4、Table2-min-height；TS 8.1.4、8.1.5、8.1.6、9.1.1、9.1.2、9.1.6、9.1.7）、9 条 needs_input（表 / 公式 / 声明输入：ISO 13857 Table 2 / Table 4、ISO 13855 公式 13 与 9.3、ISO 13854 Table 1、ISO 10218-2 两条、TS 10.6.3）、6 条 refused（`exists`、适用性里带比较、三变量、属性型要求、不可照片判定）。整个合并图 38 条：12 / 9 / 0 / 17。
- 这 9 条编译规则在 090 / 030 上**一条都没绑定到对象**：它们的适用性要 `perimeter_of(F, Z)`（围栏在危险区周界上）、`covers_opening(L, C)`（光幕盖住开口）或 `horizontal(H)`（水平光幕；两个工位的光幕都是竖的，所以 9.1.7 正确地不适用——修正前它把竖光幕判成 FAIL）。关系层今天不算 perimeter_of / covers_opening：**下一步是 L2 从声明的危险区 / 限制空间多边形（报告标注面板的 `restricted_space` → `labels declared` → `scene-json declared:`）算这两个谓词**，之后这 9 条会出和手写包同样的判定（综合测试已证明：在合成工位上加上这两个谓词，4.4 / Table2-min-height / 9.1.2 与手写 floor_gap / fence_height / lc_lowest_beam 在两个引擎下逐对象一致）。
- 22 / 18 条 NEEDS_INPUT 来自表 / 公式规则对每对（危险源, 固定结构）的绑定：缺的声明输入 = body_part、resolution_mm、restricted_space、risk_level、stop_time_ms、table_lookup、zone_depth_mm——正是 html@1 标注面板的"声明输入"表单要收的。
- `code-synthesis@0` / `redundant-translation@0` / `llm-extract@0` 三行没有跑：本机没有 Claude 凭证（没有 `ANTHROPIC_API_KEY`、没有 `ant` 配置文件）。配置和首跑命令都在各自 README；预计首跑 24 / 48 / 11 次 Haiku 调用，之后全部命中缓存（`runs/llm-cache`），`PANOPTES_FAKE_MODEL=1` 也能重放。
- diff 列里 "L6 decision None→guard_band" 一类是默认值未归一化的噪声（配置写不写默认值的区别），和 v0 一样待修。
