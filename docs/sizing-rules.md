# Sizing rules — what to run, at what load, with what latency

[← index](../README.md) · prev: [Use cases](use-cases.md) · next: [Methodology](methodology.md)

The rest of this site explains *why*. This page is the *what*: the study's
measurements reduced to rules you can size a deployment with. Every number is
measured, not estimated, and links to where it comes from.

**Measured on:** one RTX 3090, YOLOv8s FP16 at 640×640, live 30 fps cameras in
[paced mode](live-batching.md) (one frame in flight per camera, timed from the
moment each frame is due, upload included). Latencies are medians of three seeded
repeats; GPU utilisation is `nvidia-smi` from one run
([§7](live-batching.md#7-how-busy-the-gpu-was)). Cameras are unsynchronised unless
a rule says otherwise. The [camera frame rate](#by-camera-frame-rate) section
scales these 30 fps measurements to other camera rates and says where that holds.
On another GPU the fps figures move, but the rules hold when
load is read as a percentage of *your* measured capacity — see the last section.

---

## The rules

1. **Plan from capacity, not from latency.** Below capacity every pipeline here
   answers well inside a 33 ms frame; past it, its queue grows without limit and
   frames arrive seconds late. Measured capacity on this 3090: **A2 ~1,290 fps
   (43 cameras), B2 ~1,105 fps (36), batching ~1,660 fps (55)**
   ([delivered fps once overloaded](live-batching.md#3-past-b2s-capacity-batching-is-the-difference-between-working-and-not)).

2. **Run one-frame-at-a-time pipelines (A2, B2) at no more than ~75% of their
   capacity — about 80% `nvidia-smi` utilisation.** There the median stays within
   **1.4×** and the p99 within **2×** of their light-load values (A2 at 960 fps:
   2.07 / 4.27 ms, GPU 77% busy). At ~87% of capacity it is **2.3× and 2.8×** (B2
   at 960 fps: 4.35 / 7.92 ms, 90% busy). Past 100%, seconds.

3. **Below that, use A2. Add a server only if you need one, and add batching only
   if load can spike.** A2 answers in 1.5–2.1 ms p50. B2's server costs **0.3–0.9
   ms per frame** on the same clock. At light load batching can only add waiting:
   **+0.5 ms** with a 0 µs window, **~+1 ms** with 500 µs, **+5.5 ms** with D's
   5 ms ([live traffic §1](live-batching.md#1-below-capacity-batching-only-adds-latency-the-window-decides-how-much)).

4. **Past ~75% of one-frame capacity, batch — and lengthen the window as load
   grows:** 0 µs up to ~480 fps (16 cameras), 500 µs around 960 fps (32), 5 ms
   around 1,440 fps (48) ([§4](live-batching.md#4-the-right-window-grows-with-load)).
   At 1,440 fps D's 5 ms settings answer in **8.95 / 12.59 ms**, where A2 is 0.8 s
   and B2 2.1 s behind.

5. **For batching configs, GPU utilisation is not the signal — delivered fps is.**
   They read 78–94% busy at ~58% of their capacity and 100% at 87%, and still
   answer in 5–13 ms there, because a busier GPU just forms bigger batches. Watch delivered fps against
   capacity instead: fine at **87%** (1,440 fps), collapsed at **101%** (1,680 fps,
   ≥84 ms and rising).

6. **Above ~87% of batching capacity (~1,450 fps, ~48 cameras here), add
   capacity** rather than tuning: a second GPU, calibrated INT8 (**+32.8%**
   throughput for **−1.55 mAP**, [precision](precision.md#calibrated-int8-the-actual-result)),
   or a smaller model ([model cost](model-scaling.md)). The 3090 is also at its
   power limit there (348 of 350 W).

7. **If cameras fire together, batch from ~8 cameras.** In synchronised bursts
   batching gives the lowest p99 and A2 falls behind: at 16 cameras D's settings
   answer in 11.65 / 12.79 ms against A2's 14.03 / 14.54. At 32 synchronised
   cameras the one-frame pipelines reach **28–29 ms p99** — nearly the whole frame
   budget — against 22 ms for batching
   ([§2](live-batching.md#2-in-a-burst-batching-bounds-the-tail)).

8. **Offline, where nothing waits, run batch 8.** A2 at batch 8 and D tie at
   **~1,620 fps** on one model, A2 at a quarter of D's latency; three models in one
   batched in-process loop reach **2,280 fps** against Triton's 1,780
   ([batching](batching.md#correction-the-throughput-lead-is-batching-not-triton)).
   Choose D for a shared server, not for speed.

9. **Size the decoder as well as the detector.** At 1080p NVDEC tops out at
   **~768 fps (~25 cameras)** — below every detector capacity above, so at 1080p
   it is the first limit ([decoder capacity](nvdec.md)).

10. **Two things that cost more than any tuning.** Never ship raw tensors over
    gRPC (**3–3.6×** throughput lost: [transport](transport.md)), and never share
    the GPU during a latency-critical run (a second GPU process raised A2's
    latency **69%**: [contention](contention.md)).

## The load bands

The rules above as one table, for this 3090. "GPU busy" is for the pipeline in
the *Run* column.

| Load (30 fps cameras) | GPU busy | Run | Expect p50 / p99 | Move on when |
|---|---|---|---|---|
| ≤ 480 fps (≤ 16) | ≤ ~50% | **A2** · B2 if you need a server | A2 1.5 / 3.4 ms · B2 2.2 / 4.0 ms | load can spike → add a 0 µs window (+0.5 ms) |
| 480–960 fps (16–32) | 50–80% (A2) | **A2** · B3 · 500 µs if you need a server | A2 2.1 / 4.3 ms · B3 · 500 µs 5.0 / 7.1 ms | A2 above ~80% busy; B2 already at 2.8× its light-load p99 here |
| 960–1,450 fps (32–48) | ~100% | **Batching, 5 ms window** (D's settings) | 9.0 / 12.6 ms at 1,440 fps | delivered fps falls below offered |
| > ~1,450 fps (> 48) | 100%, 348/350 W | **Another GPU**, INT8 or a smaller model | ≥ 84 ms at 1,680 fps, rising | — |
| Synchronised, ≥ 8 cameras | — | **Batching** (D's settings: lowest p99) | 16 cameras: 11.7 / 12.8 ms | — |
| Offline | — | **Batch 8**: A2 in-process, or D | ~1,620 fps (1 model) · 2,280 (3 models) | — |

## By camera frame rate

Every measurement here used 30 fps cameras. Camera rate changes two things, and
they scale differently:

- **The work** is total frames per second, cameras × fps. The GPU does the same
  work per frame whatever the camera's rate, so the capacities above hold in fps:
  100 cameras at 10 fps are the same 1,000 fps load as 33 at 30.
- **The budget** is the frame interval: 200 ms at 5 fps, 33 ms at 30, 8.3 ms at
  120. A batching window that is noise at 10 fps can use up the whole frame at 120.

**Cameras per 3090**, from the safe loads in the rules (one-frame pipelines at
~75% of capacity, batching at ~87%):

| Camera fps | Frame budget | A2 (≤ 960 fps) | B2 (≤ ~830 fps) | Batching (≤ 1,440 fps) | Batching window that fits |
|---|---|---|---|---|---|
| 5 | 200 ms | 192 | 166 | 288 | any, including D's 5 ms |
| 10 | 100 ms | 96 | 83 | 144 | any |
| 15 | 67 ms | 64 | 55 | 96 | any |
| 30 | 33 ms | 32 | 27 | 48 | up to 5 ms (measured) |
| 60 | 16.7 ms | 16 | 13 | 24 | 0–500 µs; D's settings take half the budget or more (p99 8.4–12.6 ms) |
| 120 | 8.3 ms | 8 | 6 | 12 | 0 µs, or no batching (A2) |

**Which pipeline a camera rate allows.** Keep a pipeline's p99 under half the
frame interval, so a late frame still lands before the next one. From the measured
p99s below capacity:

| Pipeline | p99 below capacity (measured) | Suits cameras up to |
|---|---|---|
| A2 | 2.0–4.3 ms | ~115 fps |
| B2 | 2.8–4.0 ms at light load · 7.9 ms at 960 fps | ~120 fps light · ~60 fps near its limit |
| B3 · 0 µs | 3.1–4.8 ms at light load · 9.1 ms at 960 fps | ~100 fps light · ~55 fps near its limit |
| D's settings (5 ms window) | 8.4–9.9 ms · 12.6 ms at 1,440 fps | ~50 fps · ~40 fps near capacity |

So a slow camera can afford any pipeline and should be chosen on operations (do
you need a server?) and on total load; a fast camera rules out long batching
windows, whatever the load.

### Worked examples

| Deployment | Total load | Run | Expect p50 / p99 | Budget | Watch for |
|---|---|---|---|---|---|
| Robot, 2 cameras × 60 fps | 120 fps | **A2** | 1.6 / 2.0 ms | 16.7 ms | nothing: 9% of A2's capacity |
| Retail, 20 cameras × 15 fps | 300 fps | **A2**, or B2 / B3 if you want a server | A2 ~1.5 / ~2.5 ms | 67 ms | nothing: GPU ~30% busy |
| High-speed inspection, 4 cameras × 120 fps | 480 fps | **A2** | 1.5 / 3.4 ms | 8.3 ms | D is ruled out (5.8 / 8.5 ms at this load); B3 · 0 µs (2.7 / 4.8 ms) fits only just |
| Traffic, 48 cameras × 30 fps | 1,440 fps | **Batching, D's settings** | 9.0 / 12.6 ms | 33 ms | measured point; A2 and B2 are seconds behind here |
| Warehouse, 100 cameras × 10 fps | 1,000 fps | **Batching, 500 µs window** (B3) | ~5.6 / ~8 ms | 100 ms | the decoder: NVDEC tops out at ~768 fps at 1080p, so use 720p or lower streams, or a second GPU ([decoder capacity](nvdec.md)) |
| City, 300 cameras × 5 fps | 1,500 fps | **Two GPUs**, or batching on INT8 | — | 200 ms | past 87% of batching capacity on one 3090; INT8's engine is 32.8% faster, but INT8 through the pipelines is not measured |

*Measured values where a row's total load was measured (120, 480, 1,440 fps);
"~" marks an interpolation between the two nearest measured loads.*

**Where this scaling is only approximate.** Capacity and light-load latency carry
over by total fps, because they depend on the work per frame. Three things do not:

- **Latency near capacity depends on how frames arrive.** Many slow cameras arrive
  more smoothly than a few fast ones, so near capacity these figures are likely a
  little pessimistic for 5–15 fps cameras and a little optimistic for 60–120 fps.
- **Synchronised bursts scale with the number of cameras, not total fps.** 100
  cameras triggered together at 10 fps is a burst of 100 frames; the largest burst
  measured was 48.
- **A2 runs one execution context per camera.** Beyond the 56 cameras measured
  (192 at 5 fps, say), its memory and scheduling are untested; one A2 process that
  batches across cameras would avoid it.

None of these rows was run at its own camera rate. Treat the table as sizing, and
confirm on your own traffic before committing hardware.

## How latency grows with load

The pattern behind rules 2 and 5, as multiples of each pipeline's p50 / p99 at 8
cameras (240 fps):

| Pipeline (capacity) | at ~37–43% of capacity | at ~58% | at ~74–87% | past 100% |
|---|---|---|---|---|
| A2 (~1,290 fps) | ×1.0 / ×1.6 | — | ×1.4 / ×2.0 (74%) | seconds |
| B2 (~1,105 fps) | ×1.1 / ×1.4 | — | ×2.3 / ×2.8 (87%) | seconds |
| Batching, 5 ms window (~1,660 fps) | — | ×0.8 / ×1.2 | ×1.2 / ×1.5 (87%) | ≥ 84 ms |
| Batching, 500 µs window | — | ×1.7 / ×2.1 | ×4.6 / ×5.4 (87%) | ≥ 84 ms |

The long window looks flat because it already pays its 5 ms at light load; near
capacity that is exactly why it wins. The thresholds are **measured points, not
fitted limits**: A2 was fine at 74% and gone at 112%, B2 fine at 87% and gone at
130%, batching fine at 87% and gone at 101%. The true edges lie in between; the
rules use the highest point measured safe.

## On another GPU

The fps figures are this card's. To use the rules elsewhere:

1. Measure your one-frame capacity — about 20 minutes, with the quick estimate in
   [On another GPU](other-gpus.md#quick-estimate-about-20-minutes-of-gpu-time).
2. Read every rule as a percentage of that capacity: one-frame pipelines up to
   ~75%, then batching. On this 3090 batching capacity was **1.3–1.5×** the
   one-frame capacity (1,660 vs 1,290 for A2 and 1,105 for B2); measure yours
   before relying on the ratio, because how much batch 8 saves depends on the GPU
   and on the model's architecture ([model zoo](model-zoo.md)).
3. The window costs (+0.5 / +1 / +5.5 ms at light load) are timers and carry over
   almost unchanged; the engine part (~0.5 ms here) does not.

## What these rules do not cover

One model at one resolution, one GPU, 30 fps cameras. Utilisation was sampled once
per load, not three times. The rules say nothing about accuracy, which is on
[its own page](accuracy.md), or about heavier models, where the plumbing matters
less and batching may not help at all ([model zoo](model-zoo.md)).

---

[← index](../README.md) · prev: [Use cases](use-cases.md) · next: [Methodology](methodology.md)
