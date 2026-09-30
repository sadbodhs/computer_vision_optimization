# Results — capacity and latency

[← index](../README.md) · prev: [Methodology](methodology.md) · next: [DeepStream](deepstream.md)

All flows run the **same YOLOv8s FP16 model**. Flow IDs are defined in the
[contenders table](../README.md#the-contenders). Read
[how to read the tables](methodology.md#how-to-read-the-tables) first.

---

![Latency versus throughput for every flow, swept over concurrency 1-16](img/pareto-latency-throughput.png)

*The whole study in one plot.* Up and to the left is better. **A2** reaches the
top-left corner — lowest latency at high throughput. **D** climbs highest but
walks right as it does: its client keeps eight frames in flight per stream, so its
latency is mostly that self-built queue (Little's law), and past concurrency 4 it
exceeds 33 ms. A live camera would not see this
([live cameras](#live-cameras-paced-mode)). Given the same batch-8 engine, A2
reaches D's throughput at about a quarter of its latency
([correction](batching.md#correction-the-throughput-lead-is-batching-not-triton)).
The grey flows (A1/B1/C1) are
dominated everywhere — they are what "just use the server" or "just use Python"
costs if you feed them naively.

## Capacity results

**Reading the cells.** `↑` is throughput (higher better), `↓` is per-frame
latency p50 (lower better). Flow D's cell differs: its `↓` is end-to-end latency
with the client holding **8 frames in flight per stream**, so it is mostly those
frames waiting behind each other (Little's law), *not* the batch window — Triton's
own queue accounts for ~1.3 ms of D's 6.2 ms at one stream. The bracketed figure
is the engine's batch-8 cost per frame, a floor rather than a measurement. Full conventions: [notation](methodology.md#notation).

!!! warning "The bracketed `0.61` is the engine's floor, not a measurement"

    The same `0.61` appears on every D row because it is the **`trtexec`
    batch-8 per-frame cost** — the engine's floor — not a per-concurrency
    measurement of the server. Triton's own counters show D does not reach
    batch 8 until concurrency 8: it forms batches of **exactly 4.00** at
    concurrency 1-4, so the real service cost there is higher than 0.61.
    Measured in [batching](batching.md#what-batch-size-does-d-actually-form).


**Streams** are client streams. Frames in flight equal the stream count for every
flow except D, where they are **8 × streams** — so each row compares equal
*streams*, not equal load. See [how each flow feeds the GPU](../README.md#how-each-flow-feeds-the-gpu).

Two numbers per cell. **`fps↑` = throughput (higher is better) · `ms↓` = per-frame
latency p50 (lower is better)**. The best fps and the best latency in each row are
**bold**.

| Streams | A1 — C++ TRT, CPU path | A2 — C++ TRT, full-CUDA | B1 — Triton+C++ gRPC | B2 — Triton+C++ CUDA-shm | C1 — Triton+PyTorch | C2 — Triton+Py numpy+shm | D — Triton async, dynamic ≤8 |
|---|---|---|---|---|---|---|---|
| 1 | 456↑ · 1.32↓ | 809↑ · **1.23↓** | 222↑ · 3.27↓ | 654↑ · 1.28↓ | 144↑ · 6.4↓ | 469↑ · 1.69↓ | **1041↑** · 6.2↓ (engine floor 0.61) |
| 2 | 793↑ · **1.50↓** | **1219↑** · 1.60↓ | 368↑ · 4.03↓ | 951↑ · 1.83↓ | 198↑ · 9.6↓ | 736↑ · 2.20↓ | 1136↑ · 11.1↓ (engine floor 0.61) |
| 4 | 956↑ · 3.31↓ | 1175↑ · 3.40↓ | 472↑ · 6.90↓ | 1092↑ · 4.02↓ | 218↑ · 10.8↓ | 986↑ · **3.15↓** | **1378↑** · 19.2↓ (engine floor 0.61) |
| 8 | 1055↑ · **6.43↓** | 1187↑ · 6.72↓ | 488↑ · 14.6↓ | 1131↑ · 6.60↓ | 222↑ · 12.0↓ | 1037↑ · 6.86↓ | **1640↑** · 36.6↓ (engine floor 0.61) |
| 16 | 1205↑ · **11.8↓** | 1160↑ · 13.8↓ | 496↑ · 30.4↓ | 1128↑ · 13.5↓ | 225↑ · 14.7↓ | 1038↑ · 14.5↓ | **1665↑** · 73.9↓ (engine floor 0.61) |

**Row winners.** *(All flows here at batch 1 except D. Given the same batch-8 engine,
A2 ties D at ~1,620 fps with a quarter of its latency:
[correction](batching.md#correction-the-throughput-lead-is-batching-not-triton). On
three models a batched in-process loop beats Triton's multi-model D, 2,280 vs
1,780 fps: [measured](batching.md#the-multi-model-lead-reverses).)*
For throughput: D (1665) > A2 (1219, saturates ~1200 from conc=2)
> B2 (1131) > C2 (1038) > B1 (496) > C1 (225). For latency: A2 and B2 lead at one
stream (1.23 / 1.28 ms — not like-for-like: A2's clock starts before the frame's
upload, B2's after it; on one clock A2 is 0.3–0.9 ms faster,
[measured](live-batching.md#6-choosing-a2-b2-or-b3-at-each-load)). From two streams
up the batch-1 flows A1, A2, B2 and C2 sit within ~1 ms of one another up to 8
streams (2.7 ms at 16), and D's
6.2–73.9 ms is its client's in-flight queue.

**A2's notable shape**: it hits its ~1200 fps plateau already at conc=2 (multiple
CUDA streams overlap H2D/compute/D2H) and latency then scales linearly with N.
Overlapping streams lift it past the single-stream `trtexec` figure (1028 qps),
but no further than ~1,200: the batch-1 engine is the limit, and part of that
ceiling is launch overhead ([CUDA graphs](cuda-graphs.md)).

**D's latency is visible in the same row**: at conc=16 a frame takes 74 ms — but
that is 128 frames in flight queuing behind each other (8 per stream), not the
batch window, which is capped at 5 ms. A live camera with one frame in flight pays
~0.5 ms for batching with a zero window and ~5.5 ms with D's 5 ms one
([live cameras](#live-cameras-paced-mode)). See
[batching](batching.md#where-ds-latency-actually-goes).

GPU util at conc=16: A2 81% · B2 84% · C2 82% · D 79% (Triton-without-shm was 52%).

![Throughput against concurrency for each flow](img/concurrency-scaling.png)

Read against the dashed engine cap: **A2 plateaus immediately** (it is at its
ceiling by concurrency 2 and never improves), while **D keeps climbing** because
batching amortises the GPU pass across 8 frames. B2 tracks A2 from below —
150–270 fps behind at 1–2 streams, 30–85 behind from 4. Everything below the cap line is losing to the engine, not to the GPU.

## Live cameras (paced mode)

The table above says how much each pipeline can process. It does not say how long
a live camera's frame waits: capacity mode is closed-loop, and D's client holds
eight frames per stream. [Paced mode](live-batching.md) measures that instead —
virtual 30 fps cameras, one frame in flight each, every frame timed from the moment
it is due, upload included. The budget is one frame interval, **33.3 ms**.

| 30 fps cameras | A2 | B2 | B3 · 0 µs | B3 · 500 µs | D's settings (5 ms window) |
|---|---|---|---|---|---|
| 8, unsynchronised (240 fps) | **1.48 / 2.12** | 1.92 / 2.86 | 2.33 / 3.12 | 2.85 / 3.31 | 7.33 / 8.41 |
| 32, unsynchronised (960 fps) | **2.07 / 4.27** | 4.35 / 7.92 | 6.15 / 9.09 | 4.97 / 7.06 | 5.60 / 9.89 |
| 48, unsynchronised (1,440 fps) | 827 / 1,567 | 2,124 / 3,879 | 13.04 / 17.70 | 13.03 / 17.76 | **8.95 / 12.59** |
| 16, synchronised burst | 14.03 / 14.54 | **9.65** / 15.39 | 11.78 / 13.00 | 11.90 / 13.44 | 11.65 / **12.79** |

*p50 / p99 ms per frame, median of 3 seeded repeats
([`b3_paced.tsv`](../results/v3/b3_paced.tsv), [`a2_paced.tsv`](../results/v3/a2_paced.tsv)).
B3 is B2's synchronous client in front of a short-window dynamic batcher.*

- **Below capacity, one frame at a time wins**, and in-process wins by another
  0.3–0.9 ms. Batching can only add waiting there: ~0.5 ms with a zero window
  (almost all of it the batch-8 engine running one frame), ~5.5 ms with D's 5 ms one.
- **Past a one-frame-at-a-time pipeline's capacity, its queue grows without
  limit** — A2 is 0.8 s and B2 2.1 s late at 48 cameras — while every batching
  config stays at 9–13 ms. Capacities under live traffic: A2 ~1,290 fps, B2
  ~1,100, batching ~1,650. At 56 cameras (1,680 fps) nothing keeps up.
- **When cameras fire together**, batching bounds the tail (13–26% lower p99 in
  synchronised bursts) and A2 falls behind: its per-camera contexts share the GPU
  and all finish late together.

![Lowest per-frame turnaround at each load](img/selection-paced.png)

*Circled: the pipeline with the lowest turnaround at each load — it changes with
the load. Per-load picks: [choosing A2, B2 or B3](live-batching.md#6-choosing-a2-b2-or-b3-at-each-load).*

**Other camera rates.** Capacity is total frames per second, so the same GPU
serves more slow cameras: at the safe loads, one 3090 carries about 96 cameras at
10 fps on A2 (32 at 30 fps, 16 at 60) and 144 at 10 fps with batching (48 at 30,
24 at 60). What changes with the rate is the budget: a 5 ms batching window is
noise against a 10 fps camera's 100 ms and fatal against a 120 fps camera's
8.3 ms. Tables and worked examples, derived from these 30 fps measurements:
[sizing rules by camera frame rate](sizing-rules.md#by-camera-frame-rate).

DeepStream's single-stream latency (1.50 ms, RTSP mode) is on the
[DeepStream](deepstream.md) page; it was not run in paced mode.

## Per-frame efficiency — the fair unit for D

Per-frame GPU cost = GPU pass time ÷ frames per pass:

| Flow | Frames per GPU pass | GPU time/pass | **GPU cost per frame** | Client+server overhead per frame |
|---|---|---|---|---|
| A2 | 1 | 0.97 ms | **0.97 ms** | ~0.26 ms (in-process) |
| B2 | 1 | 0.97 ms | **0.97 ms** | ~0.31 ms (CUDA shm round trip) |
| C2 | 1 | 0.97 ms | **0.97 ms** | ~0.72 ms (sys-shm round trip) |
| D  | 8* | 4.84 ms | **0.61 ms** | batch window ≤ 5 ms + round trip |

*\*The engine's ceiling. D actually forms batches of 4.00 at 1–4 streams and ~8
only from 8 streams ([measured](batching.md#what-batch-size-does-d-actually-form)),
so below that its real per-frame cost is higher than 0.61 ms.*

D's GPU is **37% cheaper per frame** (0.61 vs 0.97 ms) because one pass shares
weights/memory across 8 frames. That efficiency is the entire reason its fps is
higher — and it comes from the batch size, not the server: A2 given the same
batch-8 engine ties it
([correction](batching.md#correction-the-throughput-lead-is-batching-not-triton)).
What it costs a frame depends on how frames arrive. In capacity mode D's 6–74 ms
is mostly its client's 8-deep in-flight window; for a live camera, batching costs
~0.5 ms with a zero window and ~5.5 ms with D's 5 ms one
([live cameras](#live-cameras-paced-mode)).

## One line per flow

- **A2** — lowest latency, in-process control. 809 fps single, 1.23 ms. At batch 8:
  1,622 fps at 10 ms on one model, 2,280 fps on three.
- **B2** — Triton with most of the tax removed (CUDA shm): 654–1131 fps, 1.28 ms
  (on one clock with A2 it is 0.3–0.9 ms slower per frame).
- **B3** — B2's client with a 0–500 µs dynamic-batching window: for a live camera
  ~0.5 ms over B2 below capacity, and still 9–13 ms at 48 cameras where B2 is
  seconds late ([live traffic](live-batching.md)).
- **C2** — Python without the tax (numpy + sys-shm + processes): 1038 fps, 1.69 ms.
- **D** — GPU-efficient batching: 0.61 ms/frame GPU cost, 1665 fps; 6–74 ms p50
  in capacity mode, mostly its client's in-flight window. A2 at batch 8 ties it.

Raw data: [`results/`](../results/README.md) · full fair tables:
[`results/comparison_tables.md`](../results/comparison_tables.md).

---

[← index](../README.md) · prev: [Methodology](methodology.md) · next: [DeepStream](deepstream.md)
