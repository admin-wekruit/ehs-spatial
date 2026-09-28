# E3 人和移动物（快速版验证，ME340，2026-09-28）

[M] 实测，[E] 估计。GPU 都是 Modal A100-80GB，按列表价 $0.000694/s 计。

## 怎么重跑

```
# 1) GPU（一个临时 app，两个容器同时跑，retries 0，超时 900/1200 s）
python modal_apps/e3_people_probe.py --output runs/m3-exp-e3-people-probe-NEW
# 2) CPU 打分（本机，约 45 s，峰值约 0.6 GB）
python scripts/e3_people_eval.py --probe runs/m3-exp-e3-people-probe-NEW --output runs/m3-exp-e3-people-eval-NEW
python scripts/e3_people_eval.py --self-check
```

python 用 `/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python`；runs 在 `research-notes/phase2/runs/`。
本次结果：`m3-exp-e3-people-probe`（GPU 输出）、`m3-exp-e3-people-eval-v3`（最终打分；v1、v2 是中间版本）。

## 做法

- 相机、深度、地面都和今天一样（DROID 位姿 + DA3 posed 深度 + 1.6 m 携带高度），只换检测器、帧率和关联。
  所以这里量的是"检测 + 5 fps + 关联"的差别；DA3 any-view 位姿带来的误差看 E1。
- 规则用 `ehs_spatial.live_people.PeopleLoop`（`video.judge_frame`），通过 `scripts/replay_people_stream.py` 的流回放。
- 区域用 m0 回放用过的测试区域 `POLYGON ((1.8 -3.9, 2.8 -3.9, 2.8 -2.9, 1.8 -2.9))`。

## 结果

| 项 | 数 |
|---|---|
| (a) SAM 3 'person' 5 fps 对今天质心，同帧，地面距离 | 中位 0.007 m，p90 0.022 m，最大 0.136 m [M] |
| (a) 同上，插值到今天 10 fps 的帧 | 中位 0.011 m，p90 0.035 m，最大 0.15 m [M] |
| (a) 检测对今天掩码 | 116/116 匹配，IoU 中位 0.955；多出 24 个掩码 [M] |
| (a) 规则，过尺度闸门后，对 ref10 逐 0.2 s | R1 100%，R2 100%（都没有移动物，NO_DATA），R3 96.5% [M] |
| (a) 规则，过闸门前 | R1 96.5%，R3 96.5%：7.74–8.34 s 一段，今天 FAIL，快速版 NEEDS_REVIEW [M] |
| (a) 身体质心速度（R3 同一规则） | 102/102 结论相同，速度差中位 0.012 m/s [M] |
| (a) SAM 3 耗时 | 只找人 113 ms/帧，8 张一批 81 ms/张；加 4 个移动物词 281 ms/帧 [M] |
| (b) SAM 3.1 stride 3，热 | 传播 9.9 fps；加 init 8.1 fps；224 帧共 27.8 s [M] |
| (b) SAM 3.1 stride 1 | 7.9 fps；670 帧共 100 s [M] |
| (b) stride 3 对 stride 1 的掩码 | IoU 中位 0.995，p10 0.985 [M] |
| (c) 今天保留的 2 条运动轨迹 | 1 条是主讲人（SAM 3 人覆盖 98.7%），1 条是烧录字幕（180-motion-4）[M] |
| (c) 移动物词表 | forklift/cart/vehicle 0 次；pallet jack 23 帧，全是误检（静止绿色物体）[M] |
| (c) 静态场景重投影残差 | CPU 63 ms/帧；≥2% 的种子 24/110 帧，其中 93% 像素在人身上，9% 在字幕带 [M] |
| 冷启动 | 发起到函数开始约 20 s；模型加载 SAM 3 5.6 s，SAM 3.1 21.3 s [M] |
| 花费 | 两个容器合计 342 s，GPU 列表价上限 $0.24（加 CPU/内存约 $0.26）[M 秒数，E 价格] |

## 要注意

- SAM 3 有时把主讲人返回两次（分数 0.46），5 fps 关联因此把一条轨迹拆成两条；丢掉"一半以上落在更高分掩码里"的掩码就修好了（`fast5_dedupe`）。
- SAM 3 还找到 6 个很小的远处"人"（130–210 像素，分数 0.41–0.64），今天的层里没有；是真人还是误检没核实。
- ME340 的脚几乎一直被工作台挡住，两边都算不出脚速度；尺度是假设的，所有过闸门的结论都是 NEEDS_REVIEW；没有真的移动物。所以这段视频对规则的检验偏弱，R2 完全没被检验。
- ME340 没有叉车这类移动物，"快速版会漏多少非人移动物"在这段视频上测不出来（分母是 0）。
