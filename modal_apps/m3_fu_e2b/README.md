# E2b: a per-video vocabulary for SAM 3 (fast path follow-up)

`vocab_probe.py` tests whether a vocabulary written for each video can replace the fixed 20 EHS words of E2. In round 1
those 20 words found 51% of today's named objects on ME340. Round 1 reached 73-81% only with 40 words picked after
seeing the misses.

The probe has three steps:

1. **Ask a VLM for the object list.** A VLM sees 3 or 5 walk-shot frames and lists the object types it sees. The
   frames are evenly spaced over the shot, which means evenly along the path at walking pace. The list puts
   EHS-relevant types first, as short nouns.
   - Gemini runs through the existing report container, the `scripts/name_video_entities.py` mechanism. No key is
     handled here.
   - Qwen3-VL-8B runs through vLLM on a Modal A100. This is the on-prem option.
2. **Run SAM 3 on the list.** The list goes to SAM 3, alone and with the fixed EHS core list: fire extinguisher, exit
   sign, forklift, ladder, spill, cable, hose, guard.
   - It uses sam3_app's pinned model. Vision is encoded once per frame and text once per list. Frames x prompts go
     through in batches of 80 pairs.
   - It runs on the 45 round-1 keyframes (2 fps) and on 56 keyframes (every 4th frame of today's 10 fps grid, "~60").
     Together that is 89 frames.
3. **Score recall with round 1's rule.** A named object is found when some SAM 3 mask has IoU >= 0.5 with one of its
   observed masks on a shared keyframe. There is no top-12 cap, and the score floor is 0.3, 0.4 or 0.5. Round 1's
   production rule (floor 0.4, top 12 per word) is also scored, for reference.

The prompt is generic and was never changed after seeing ME340's names or any recall number. There are two versions:
- **v1:** "at most 50 entries". The cap of 50 comes from the speed budget: about 0.0075 s per word per frame.
- **v2:** "be exhaustive: aim for 40 to 50". It was added after v1's Gemini lists came back short (21-25 entries),
  before any recall had been computed.

Qwen's lists are cut to the first 50 entries, as the prompt asks.

## Rerun

```
PY=/Users/adam/Desktop/Tesla/panoptes-platform/.venv/bin/python
RUN=/Users/adam/Desktop/panoptes-public/research-notes/phase2/runs/m3-fu-e2b-vocab-1
$PY modal_apps/m3_fu_e2b/vocab_probe.py --self-check                          # no GPU, no network
$PY modal_apps/m3_fu_e2b/vocab_probe.py vlm  --out $RUN                        # Gemini (answers already in RUN are reused) + Qwen3-VL
$PY modal_apps/m3_fu_e2b/vocab_probe.py sam3 --out $RUN --only controls,gemini --tag a
$PY modal_apps/m3_fu_e2b/vocab_probe.py sam3 --out $RUN --only qwen --tag b
$PY modal_apps/m3_fu_e2b/vocab_probe.py sam3 --out $RUN --only unions --tag c   # two calls (3 + 5 frames) merged + core
$PY modal_apps/m3_fu_e2b/vocab_probe.py summary --out $RUN
```

These files are in `runs/m3-fu-e2b-vocab-1`:
- `gemini-*/`: the request manifests, provider events and outputs.
- `vlm.json`: every VLM answer, its timing and the parsed lists.
- `vlm-qwen-greedy.json`: the first Qwen run, with greedy decoding.
- `sam3-{a,b,c}.json`: per set, the words, timings, masks per frame and each observation's best IoU.
- `summary.json`: recall tables, misses, timings and USD.

## Results, 2026-09-28 (ME340 walk shot)

M = measured, E = estimated. Recall is given as found / pool.
- The pool is today's named walk objects that were observed on the keyframes: 93 on the 45 keyframes, 95 on the 56.
- SAM 3 s/frame is warm, on an A100-80GB. Run a and run c got a PCIe card and run b (all the single-call Qwen rows)
  an SXM4 card. By the linear fit below, the SXM4 card was about 10% faster at the same word count (E).
- s/frame includes text encoding (0.015-0.018 s per list, M), the vision encoder, the decoder, and post-processing to
  full-size masks.

**Control.** The 20 EHS words on the 45 keyframes reproduce round 1:
- production rule: 47/93, the same as round 1;
- floor 0.3 with no cap: 51/93, against round 1's 52/93.

### Recall, the 56 keyframes (the 45-keyframe numbers in brackets)

| Vocabulary | Words | SAM 3 s/frame (M) | Floor 0.3 (M) | Floor 0.4 (M) | Production rule (M) | Masks/frame at 0.3 / 0.4 (M) |
|---|---|---|---|---|---|---|
| 20 EHS words (round 1) | 20 | 0.192 | 52/95 = 55% (55%) | 46% (51%) | 46% (51%) | 83 / 46 |
| core 8 only | 8 | 0.110 | 12% (10%) | 7% (8%) | 7% (8%) | 23 / 11 |
| Gemini, 1 call, 3 frames, v1 (+ core) | 25 (33) | 0.230 (0.289) | 64% (65%) [68-69%] | 60% | 56% | 166 / 95 |
| Gemini, 1 call, 5 frames, v1 (+ core) | 21 (29) | 0.204 (0.264) | 76% (77%) [75-78%] | 67-68% | 60-61% | 151 / 88 |
| Gemini, 1 call, 3 frames, v2 (+ core) | 30 (38) | 0.274 (0.327) | 76% (76%) [73%] | 66% | 64% | 146 / 75 |
| Gemini, 1 call, 5 frames, v2 (+ core) | 28 (36) | 0.258 (0.314) | 66% (67%) [66-68%] | 56-57% | 54-55% | 133 / 70 |
| **Gemini, 2 calls (3 + 5 frames) merged + core, v1** | 47 | 0.389 | **76/95 = 80%** (84%) | 73% (75%) | 68% (72%) | 310 / 176 |
| **Gemini, 2 calls (3 + 5 frames) merged + core, v2** | 59 | 0.480 | **81/95 = 85%** (85%) | 73% (75%) | 69% (73%) | 261 / 135 |
| **Qwen3-VL, 1 call, 3 frames, v1, first 50 (+ core)** | 50 (56) | 0.385 (0.429) | **85/95 = 89%** (89%) | 87% (85%) | 79% (81%) | 309 / 179 |
| **Qwen3-VL, 1 call, 5 frames, v1, first 50 (+ core)** | 50 (57) | 0.387 (0.435) | **85/95 = 89%** (89%) | 86% (85%) | 77% (81%) | 307 / 166 |
| Qwen3-VL, 1 call, 3 frames, v2, first 50 (+ core) | 50 (56) | 0.386 (0.426) | 74% (78%) | 64-65% | 62-63% | 229 / 128 |
| Qwen3-VL, 1 call, 5 frames, v2, first 50 (+ core) | 50 (56) | 0.387 (0.428) | 81% (81%) | 69-71% | 68-69% | 304 / 168 |
| Qwen3-VL, 2 calls merged + core, v1 / v2 | 81 / 85 | 0.664 / 0.694 | 89% / 83% | 87% / 74% | 79% / 73% | 388 / 210 (v1) |

- Adding {person, floor}, which production always runs, changes no row by more than one object.
- SAM 3 time is linear in the number of words (M, 2 to 85 words): about 0.05 + 0.0075 x words s/frame. The target of
  0.5 s/frame therefore allows about 60 words.
- Peak GPU memory for SAM 3 at 80 pairs per forward was 26-39 GB (M). Round 1 measured 19.8 GB at 40 pairs, with the
  same speed.

### VLM time (M)

| VLM | Call | Time | Notes |
|---|---|---|---|
| Gemini 3.5 Flash (thinking LOW), through the report container | 1 request, 3 or 5 frames at 640x480 | 4.9-11.3 s inside the container, 7-14 s client wall | 3.5k / 5.6k input tokens. The slow calls spent 565-1197 tokens thinking. 21-30 entries every time, even when asked for 40-50. |
| Qwen3-VL-8B, vLLM 0.11 eager, 0.45 of the card | warm call | 9.4-9.8 s | Every call ran to the 600-token cap: after about 30 real entries it free-associates (kitchen items, "cat8 cable", "seismograph"). Load 53-63 s; container start to function entry 5-9 s. |
| Qwen3-VL-8B, greedy (first run) | warm call | 33 s | Looped ("tool tray", "tool box", ...) to the 1024-token cap. The rerun used the model card's sampling (T 0.7, top-p 0.8, top-k 20, presence penalty 1.5). |

Two calls of the same VLM run in parallel, so they add no wall time. Qwen with CUDA graphs and a stop after 50 entries
should take about 3-6 s (E).

### Verdict against the target (>= 80% recall at <= 0.5 s/frame)

- **Met at score floor 0.3 (M):**
  - Gemini, two parallel calls (3 + 5 frames) merged with the core list: 80-85%, 47-59 words, 0.39-0.48 s/frame.
  - Qwen3-VL v1, one call, first 50 entries: 89%, 0.39 s/frame; with the core list 0.43 s/frame.
- **Met at floor 0.4 (M):** only the Qwen v1 single call, at 86-87%. Gemini's two-call lists drop to 73%.
- **Not met:**
  - A single Gemini call: 64-77%, 0.20-0.33 s/frame.
  - Qwen with the v2 prompt: 74-81%.
- **Answers vary from call to call.** The same prompt with different frames swings recall by about 10 points
  (Gemini 64-76%; Qwen v1 89% against v2 74-81%). One ME340 run per configuration cannot rank the VLMs.
- **Merging two calls helps.** It is the cheapest way to steady the list: all six pairs of Gemini lists score 76-85%
  on the 56 keyframes at floor 0.3 (computed from the per-list results, M; the two measured merges agree within one object).
- **The core list adds 0-3 points** on this clip. On its own it finds 12%; ME340 has no forklift, exit sign or spill.
- **Still missed, even by the best list:** CNC control panel, handles, light fixtures, metal plates, plastic bins,
  plastic guard, table leg, vise, wood block.

### Caveats

- **The recall rule ignores the word.** An object counts as found by any mask that overlaps it, whatever word produced
  that mask.
  - Generic nouns such as "tool", "machine", "container" or "metal part" produce many masks. At floor 0.3 that is
    150-390 masks per frame against 83 for the 20 EHS words. This is partly why Qwen's long, generic list scores high.
  - A check that the SAM 3 word matches today's name would give lower numbers.
  - Downstream lifting has to handle the mask count. Floor 0.4 halves it but costs 4-12 points of recall with the
    Gemini lists.
- **Hardware and runs.** SAM 3 seconds come from one A100-80GB, PCIe or SXM4 as Modal assigned. Each configuration was
  run once.
- **Cold start and transfer are extra (M):**
  - SAM 3 container: 73-74 s from the call to function entry in two runs, and 320 s in a third that queued for an
    A100-80GB;
  - SAM 3 model load: 4-5 s;
  - upload of the 89 PNGs: included in the client wall time.
- **Fitting both models on one card (E).** Qwen3-VL (vLLM at 0.45, about 36 GB) plus SAM 3 at 40 pairs per forward
  (about 20 GB) fit one A100-80GB. At 80 pairs, which peaked at 26-39 GB, the two together would need about 62-75 GB.
- **USD (E), about $1.7, well under the $5 cap:**
  - SAM 3 runs: about $0.68;
  - Qwen: $0.43;
  - one SAM 3 run that crashed on an empty-mask edge case (fixed): about $0.1;
  - about 10 min of a Qwen container crash-looping on an import-path bug (fixed): up to about $0.4;
  - Gemini: 4 requests, under $0.02.
