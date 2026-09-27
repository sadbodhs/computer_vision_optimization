# Live camera traffic — batching that does not make frames wait (B3)

[← index](../README.md) · prev: [Batching](batching.md) · next: [Triton tuning](triton-tuning.md)

Every latency this study published for batching comes from **capacity mode**: a
closed-loop client holding 8 frames in flight per stream. That is the right way to
measure throughput, and it makes latency a queue the client builds itself (see
[batching](batching.md)). A live camera has **at most one frame in flight**, and
nobody had measured what batching does to it.

This page does, with a new flow and a new measurement mode.

Script: [`b3_paced.sh`](../scripts/b3_paced.sh) · client:
[`grpc_client_cuda.cu`](../cpp/src/grpc_client_cuda.cu) (`--mode paced`) · figure:
[`plot_b3.py`](../scripts/plot_b3.py) · raw data:
[`results/v3/b3_paced.tsv`](../results/v3/b3_paced.tsv) — 5 configs × 2 arrival
patterns × 7 camera counts × 3 interleaved repeats, no foreign GPU process in any
run.

!!! abstract "The short version"

    - **Below capacity, batching never makes a frame faster — only slower.** D's
      configuration adds **~5.5 ms** to every live frame. B3 · 0 µs adds **~0.5 ms**
      (B3 · 500 µs ~1 ms). B2 (no batching) is still the fastest by that margin.
    - **In a burst, batching bounds the tail** — up to 24% lower p99.
    - **Past B2's capacity, batching is the difference between working and not**
      — at 48 cameras B2's median frame is **2.1 seconds** late; every batching
      config stays at **9–13 ms**.
    - **The right batch window grows with load.** There is no single best value.

---

## Setup

**Paced mode** (`trt_grpc_cuda --mode paced`): N virtual cameras, each producing a
frame every 33.3 ms on its own schedule and sending it the moment it is due —
one frame in flight per camera, as a real camera would. Turnaround is measured
from each frame's **due time**, not from when the request left, so a slow server
cannot hide its own delay by making the client send later (the error known as
coordinated omission). It includes the upload into the shared-memory region,
because a real frame has to get there too.

Two arrival patterns: **unsynchronised** (each camera at a random, seeded phase —
the same phases for every config) and **synchronised** (all cameras fire together
— the worst-case burst).

| Config | Model | Server batching | Role |
|---|---|---|---|
| **B2** | `yolov8s` — batch-1 engine | none (`max_batch_size: 0`) | the no-batching baseline |
| **D config** | `yolov8s_dyn` — batch-8 engine | preferred `[4, 8]`, **5,000 µs** | the published D setting, now under live traffic |
| **B3 · 500 µs** | `yolov8s_dyn` | preferred `[2, 4, 8]`, **500 µs** | batch only frames that arrive close together |
| **B3 · 0 µs** | `yolov8s_dyn` | **0 µs**, no preferred sizes | batch only when both instances are busy |

All four use CUDA shared memory and 2 Triton instances. **B3** is B2's client
pointed at a short-window batcher: batching happens only when frames are already
waiting together, and no frame is held back to be batched.

![Per-frame p99 turnaround against offered load, unsynchronised and synchronised cameras](img/b3-paced.png)

## 1. Below capacity, batching only adds latency — the window decides how much

Unsynchronised cameras, p50 turnaround (ms):

| Cameras | B2 | B3 · 0 µs | B3 · 500 µs | D config |
|---:|---:|---:|---:|---:|
| 1 | **2.50** | 3.10 | 3.69 | 8.39 |
| 4 | **1.91** | 2.49 | 3.06 | 7.58 |
| 8 | **1.92** | 2.33 | 2.85 | 7.33 |
| 16 | **2.16** | 2.67 | 3.04 | 5.81 |

Every batching config is slower than B2, and the gap splits cleanly into two parts:

- **~0.6 ms is the engine, not the batching.** The batch-8 engine running a single
  frame is slower than a dedicated batch-1 engine — measured at 1 camera, where
  every config forms batches of exactly 1.00. TensorRT tunes kernels for the
  engine's optimisation shape (batch 8); batch 1 runs on them correctly but less
  efficiently.
- **The rest is the window.** 500 µs costs ~0.5 ms; D's 5,000 µs costs ~5 ms. A
  frame arriving alone cannot form a preferred batch, so it waits out the whole
  window before it is sent.

So D's configuration charges every live frame **5.4–5.9 ms** for batches that, at
these loads, barely form. B3 · 0 µs cuts that to **0.4–0.6 ms**, nearly all of
which is the engine.

## 2. In a burst, batching bounds the tail

Synchronised cameras — every frame lands at once:

| Cameras | B2 p99 | B3 · 0 µs p99 | B3 · 500 µs p99 | B2 mean | B3 · 500 µs mean |
|---:|---:|---:|---:|---:|---:|
| 8 | 9.66 | **8.11** | 8.21 | **6.81** | 7.10 |
| 16 | 15.39 | **13.00** | 13.44 | **9.85** | 10.51 |
| 32 | 29.26 | 22.26 | **22.21** | 17.53 | **15.99** |

With B2, a burst of N frames queues behind two instances and the last frame waits
for all the others. Batching sends the burst as a few batches, so the worst frame
finishes sooner — **p99 falls 13–24%**. The price is paid in the mean at 8 and 16
cameras, where every frame in a batch waits for the whole batch; by 32 cameras
batching wins both.

Note how close B2's burst p99 at 32 cameras (29.3 ms) comes to one frame interval
(33.3 ms). Batching keeps a synchronised rig well inside it.

## 3. Past B2's capacity, batching is the difference between working and not

Unsynchronised cameras:

| Cameras (fps) | B2 p50 | B2 p99 | batching configs p50 | batching configs p99 |
|---|---:|---:|---:|---:|
| 32 (960) | 4.35 | 7.92 | 4.97–6.15 | 7.06–9.89 |
| **48 (1,440)** | **2,124** | **3,879** | **8.95–13.04** | **12.59–17.76** |
| 56 (1,680) | 3,357 | 6,121 | 84–94 | 146–157 |

48 cameras sits between B2's capacity (~1,131 fps) and D's (~1,650 fps). B2 cannot
keep up, so its queue grows for the whole run and frames arrive **seconds** late.
Batching raises capacity enough to stay on the flat part of the curve. At 56
cameras (1,680 fps) every config is past capacity and degrades — batching only
moves the wall, it does not remove it.

## 4. The right window grows with load

| Load | Best config (p50) | Mean batch formed |
|---|---|---|
| ≤ 16 cameras | **B3 · 0 µs** | ~1.0 — nothing to batch; any wait is pure cost |
| 32 cameras | **B3 · 500 µs** (4.97 ms, vs 6.15 at 0 µs) | 1.45 vs 1.27 — the window catches enough batch-mates to pay for itself |
| 48 cameras | **D config** (8.95 ms, vs 13.0 for B3) | 4.01 vs ~3.5 — bigger batches are more GPU-efficient, and near capacity efficiency *is* latency |

A window is a bet that company is coming. At low load it rarely is, and the wait
is wasted; near capacity it always is, and the larger batches free enough GPU time
to shorten everyone's queue. **There is no single best `max_queue_delay` — only a
best one for a given load.**

## 5. Triton turns batching on even when you don't ask

The design had a fifth arm: the batch-8 engine with the `dynamic_batching` block
removed, meant as "batching off". It batched — 3.4 per execution in a burst,
7.99 at 56 cameras — identically to B3 · 0 µs. Triton's own report of the
configuration it loaded settles it:

```
dynamic_batching: {"preferred_batch_size": [8], "max_queue_delay_microseconds": 0, ...}
```

**Config auto-complete adds a zero-delay dynamic batcher to any model with
`max_batch_size > 0` that does not name a scheduler.** Leaving the block out does
not disable batching. The reliable ways are `max_batch_size: 0`, or starting the
server with `--disable-auto-complete-config`.

The arm is kept in the data and the figure as what it turned out to be: an
independent second run of the 0 µs config, which it matches. At 1 camera, where
it formed batches of exactly 1.00, it still validly measures the engine's own cost
(+0.64 ms). No published result is affected — every model behind one is either
`max_batch_size: 0` or has batching configured explicitly.

## Predictions, stated in advance

| # | Prediction | Result |
|---|---|---|
| 1 | Low load: B3 within ~0.5 ms of B2; D config ~4–5 ms worse | **Partly wrong.** B3 · 0 µs +0.4–0.6 ms ✓; D config +5.4–5.9 ms ✓ (slightly more). B3 · 500 µs +0.9–1.2 ms ✗ — the prediction left out the batch-8 engine's own ~0.6 ms |
| 2 | Bursts: batching bounds p99, B2 keeps the better mean | ✓ at 8 and 16 cameras; at 32 batching wins both |
| 3 | 48 cameras: B2 unbounded, batching bounded | ✓ 2.1 s against ≤ 13 ms |
| 4 | B3 · 0 µs batches ~1.0 at low load, rising with load | ✓ 1.00 → 1.01 → 1.27 → 3.57 → 7.98 |

Not predicted: that the best window grows with load (§4), and that Triton enables
batching unasked (§5).

## What changes in the advice

| Situation | Use | Why |
|---|---|---|
| Few cameras, well below capacity, latency is everything | **B2** | ~0.5 ms faster than anything that batches |
| Bursts possible, or load may approach B2's ~1,131 fps | **B3 · 0 µs** | +0.5 ms at low load buys a bounded tail in bursts and survival past B2's capacity |
| Sustained load near capacity | **longer window** (D's config) | bigger batches free GPU time; best p50 at 48 cameras |
| Live cameras at low load | **never D's config** | +5.5 ms on every frame for batches that do not form |

The earlier [use-cases](use-cases.md) advice — *"No batching, ever"* for closed
loops, justified by D's 74 ms — survives in its conclusion and loses its reason.
The real cost of batching a live camera below capacity is **~0.5 ms, not 74 ms**,
and B3 · 0 µs is cheap enough to be worth it wherever a burst or a load spike is
possible.

## Three harness defects this run exposed

- **Every stream loaded its own copy of `frames.bin` (1.47 GB).** Published
  16-stream capacity runs were quietly using 23.5 GB of host RAM; at 32 streams the
  kernel killed the client before it printed anything (exit 137). The B2 client now
  shares one copy. **The A2 and D clients still have the same pattern.**
- **The sweep's signal handler restored state and then kept running** — unlocked,
  with its backup deleted. The handler now exits, restores from git, and was
  tested on exactly that failure.
- **The container's baked source had drifted from the repo** — it still carried the
  nearest-neighbour resize the [accuracy](accuracy.md) work replaced. Paced mode
  never runs that kernel, so no result here is affected, and the rebuilt binary is
  current.

## Scope

- YOLOv8s FP16 at 640×640, 30 fps cameras, 2 Triton instances, one RTX 3090.
- Turnaround includes the upload from **pageable** host memory and runs on a GPU
  that is idle most of each frame interval at low load, so absolute paced-mode
  numbers are higher than capacity-mode ones (B2: 2.5 ms here vs 1.28 ms there).
  **Compare configs within this page, not against capacity-mode tables.**
- Synchronised phases are the worst case; real rigs sit between the two patterns.
- An obvious follow-up: build the batch-8 engine with a second, batch-1
  optimisation profile — that could remove most of the 0.6 ms engine cost that
  §1 attributes to every batching config at low load.

---

[← index](../README.md) · prev: [Batching](batching.md) · next: [Triton tuning](triton-tuning.md)
