# 判定层 lab 记分卡 v0（2026-10-08）

命令（从仓库根目录）：

```
PANOPTES_FAKE_MODEL=1 python -m ehs_spatial.cli verdict matrix --benchmark ehs_spatial/verdict/benchmark/v0 \
  --configs ehs_spatial/verdict/configs/baseline.yaml ehs_spatial/verdict/configs/python-engine.yaml ehs_spatial/verdict/configs/stpl-k1.yaml \
  --runs-dir runs/lab-v0 --jobs 2 --run-id lab-v0
```

基准 v0 = 090 / 030 的试跑场景（金标是 2026-10-07 试跑的判定，标 provisional，不是人工标注）。规则包 = handwritten@1（试跑的 5 条 + reach_over 输入规则）。

| run | item | L1 | L2 | L3 | L4 | L5 | L6 | L7 | PASS | FAIL | NEEDS_MEASUREMENT | NEEDS_INPUT | CANNOT_DETERMINE | gold agreement | diff vs previous row |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lab-v0-baseline-030 | 030 | scene-json@1 | relations@1 | none@1 | none@1 | handwritten@1 | clingo@1 | markdown@1 | 9 | 1 | 1 | 6 | 1 | PASS 9/9 · FAIL 1/1 · NEEDS_MEASUREMENT 1/1 · NEEDS_INPUT 6/6 · CANNOT_DETERMINE 1/1 (provisional) | — |
| lab-v0-baseline-090 | 090 | scene-json@1 | relations@1 | none@1 | none@1 | handwritten@1 | clingo@1 | markdown@1 | 10 | 2 | 2 | 7 | 1 | PASS 10/10 · FAIL 2/2 · NEEDS_MEASUREMENT 2/2 · NEEDS_INPUT 7/7 · CANNOT_DETERMINE 1/1 (provisional) | item 030→090 |
| lab-v0-python-engine-030 | 030 | scene-json@1 | relations@1 | none@1 | none@1 | handwritten@1 | python@1 | markdown@1 | 9 | 1 | 1 | 6 | 1 | PASS 9/9 · FAIL 1/1 · NEEDS_MEASUREMENT 1/1 · NEEDS_INPUT 6/6 · CANNOT_DETERMINE 1/1 (provisional) | item 090→030; L6 clingo→python; L6 decision None→guard_band |
| lab-v0-python-engine-090 | 090 | scene-json@1 | relations@1 | none@1 | none@1 | handwritten@1 | python@1 | markdown@1 | 10 | 2 | 2 | 7 | 1 | PASS 10/10 · FAIL 2/2 · NEEDS_MEASUREMENT 2/2 · NEEDS_INPUT 7/7 · CANNOT_DETERMINE 1/1 (provisional) | item 030→090 |
| lab-v0-stpl-k1-030 | 030 | scene-json@1 | relations@1 | stpl@1 | none@1 | handwritten@1 | clingo@1 | markdown@1 | 9 | 1 | 1 | 6 | 1 | PASS 9/9 · FAIL 1/1 · NEEDS_MEASUREMENT 1/1 · NEEDS_INPUT 6/6 · CANNOT_DETERMINE 1/1 (provisional) | item 090→030; L3 none→stpl; L6 python→clingo; L6 decision guard_band→None; L6 k 2→1 |
| lab-v0-stpl-k1-090 | 090 | scene-json@1 | relations@1 | stpl@1 | none@1 | handwritten@1 | clingo@1 | markdown@1 | 11 | 2 | 1 | 7 | 1 | PASS 10/10 · FAIL 2/2 · NEEDS_MEASUREMENT 1/2 · NEEDS_INPUT 7/7 · CANNOT_DETERMINE 1/1 (provisional) | item 030→090 |


读法：python@1 与 clingo@1 在 40 条判定上状态、测量值、U、裕量全部一致（等价测试通过）；stpl-k1 行：感知规约没有标出不可信对象（计数不变），k=1 把护带减半，090 右围栏离地缝 84 ± 100 → [34, 134] ≤ 180 变 PASS，机器人 ↔ 左围栏 533 ± 143 → [462, 604] 仍跨 500 保持 NEEDS_MEASUREMENT。diff 列里的 "decision None→guard_band" 是默认值未归一化的噪声（待修）。
