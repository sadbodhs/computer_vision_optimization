# On another GPU — what transfers, and how to redo it

[← index](../README.md) · prev: [Reproduce](reproduce.md) · next: [Decoder capacity](nvdec.md)

Every number in this study was measured on **one RTX 3090** (2,100 MHz max SM
clock, 350 W power limit, PCIe Gen4 x16). A reader with a different card needs to
know two things: which findings they can use as they are, and how to find their
own numbers without repeating two days of benchmarking.

---

## What transfers and what does not

| Finding | Transfers? | Why |
|---|---|---|
| D's 5 ms batch window adds ~5 ms to a frame that arrives alone | **Yes — in absolute terms** | it is a timer, not compute; 5 ms is 5 ms on any card |
| A 500 µs window adds ≤ 0.5 ms to a lone frame | **Yes** | same reason |
| Triton switches batching on for any model with `max_batch_size > 0` and no scheduler | **Yes** | a software behaviour, not a hardware one |
| One-frame-at-a-time pipelines (A2, B2) are flat until near capacity, then hit a wall | **Yes — the shape** | it is queueing, and queueing looks the same on every GPU |
| Batching raises the wall | **Yes — the direction.** How far: no | depends on how much faster your GPU runs batch 8 than batch 1 |
| In synchronised bursts, per-camera contexts (A2) lose to first-come-first-served or batching | **Yes — the shape** | concurrent contexts share any GPU and finish late together |
| **Where** the wall is, in cameras or fps | **No** | depends on compute, clocks, and — as the 3090 shows — the power limit |
| Absolute latencies in ms | **No** | |
| The batch-8 engine's penalty at batch 1 (~0.6 ms here) | Size: **no** | TensorRT tunes kernels per GPU; measure it |
| The A2–B2 gap (0.3–0.9 ms here) | **Partly** | much of Triton's overhead is CPU and gRPC, so the host CPU matters as much as the GPU |
| NVDEC has no session limit | **Yes** | driver behaviour; how many frames per second it decodes does not transfer — see [decoder capacity](nvdec.md) |

The rule of thumb that falls out: **timers and software behaviour carry over
exactly; shapes carry over; positions do not.** Everything in the last group is what
the process below measures.

## The indicator that travels: GPU utilisation

Camera counts are a property of the 3090. How busy the GPU was at a given latency is
not — see [how busy the GPU was](live-batching.md#7-how-busy-the-gpu-was). On the
3090, measured with `nvidia-smi`:

- Utilisation grows **roughly in proportion to load**. Per 30 fps camera, A2 costs
  ~2.4–4.0% of the GPU, B2 ~2.8–4.5% and the B3 configs ~2.8–6.1% — the high end at
  one camera, the low end at 32, because the cost per camera falls as the GPU clocks
  up under load.
- **A2 and B2 stay flat below ~80% utilisation**, start queueing between 80 and 95%,
  and hit their wall near 100%.
- **For batching configs, 100% is not the wall** — at 100% they still turn frames
  around in 9–13 ms, because a busier GPU forms bigger batches. `nvidia-smi` measures
  "some kernel was running", not "no capacity left".

So on your GPU: **keep one-frame-at-a-time pipelines under ~80% utilisation**, and
treat the approach to 100% as the signal to switch to batching.

## Quick estimate — about 20 minutes of GPU time

```bash
scripts/export_models.sh                     # ALWAYS: TensorRT engines are tied to one GPU model
scripts/make_frames.sh videos/real.mp4 500   # the replay input
docker start triton-server
CAMS="4 8 16" ARMS_LIST="a2 b2" scripts/util_paced.sh results/v3/gpu_util_mygpu.tsv
```

Adjust `CAMS` so the highest count lands at roughly 50–80% utilisation: a faster card
may need `"8 16 32"`, a smaller one `"2 4 8"`.

From the output, take **utilisation per camera** at the highest count that stayed
under ~80%, then:

```
cameras before the wall  ≈  100 / (utilisation % per camera)
comfortable ceiling      ≈   80 / (utilisation % per camera)
```

On the 3090 this predicts A2's wall at 35–41 cameras from measurements at 16–32
cameras; the measured wall is ~43. **The estimate is conservative**, because the
GPU clocks up and overlaps work better as load rises — measure at the highest load
you can while staying under 80%.

## Full process — two to three hours of GPU time

1. **Get the GPU to yourself.** Nothing else running; on a shared box use
   `scripts/gpu_lock.sh acquire <name>`. Sharing the machine measurably distorts every
   number: a second GPU process raises A2's latency 69% ([contention](contention.md)),
   and one unrelated CPU job made B2 read 23–29% slow
   ([re-check](live-batching.md#re-checked-after-a-client-leak)).
2. **Rebuild engines** with `scripts/export_models.sh` and re-baseline A2 in capacity
   mode (see [reproduce](reproduce.md)).
3. **Find the wall `W`** in cameras with the quick estimate above.
4. **Choose camera counts as fractions of `W`**, not the 3090's list — for example
   5%, 10%, 25%, 50%, 75%, 90%, 110% and 130% of `W`. The interesting behaviour lives
   between 75% and 130%, and a 3090-sized list would miss it on a faster or slower card.
5. **Run the three sweeps** with that list, writing to your own directory so the
   3090's data is untouched:

    ```bash
    D=results/rtx4090; mkdir -p $D
    CAMS="…" scripts/b3_paced.sh   $D/b3_paced.tsv
    CAMS="…" scripts/a2_paced.sh   $D/a2_paced.tsv
    CAMS="…" scripts/util_paced.sh $D/gpu_util_paced.tsv
    ```

6. **Plot** with `RESULTS_DIR=$D python3 scripts/plot_selection.py` and
   `RESULTS_DIR=$D python3 scripts/plot_util.py`.

The result is your own selection chart: the same questions, answered for your card.

## Classes of GPU that behave differently

| GPU class | What changes |
|---|---|
| **Power-capped or laptop GPUs** | The wall comes earlier. The 3090 reaches its wall at 348 W of its 350 W limit, with the SM clock dropping from ~1,950 to ~1,760 MHz — part of its wall is a power wall. Lower the limit and the wall moves |
| **Jetson / Orin** | Unified memory: there is no PCIe upload, so the transport findings and the upload share of turnaround do not apply. `nvidia-smi` is replaced by `tegrastats`, and power modes (`nvpmodel`) move every number |
| **Data-centre GPUs** (A100, H100, L4, L40) | MIG splits a card into slices, each with its own wall. The batch-8 speedup may differ (unmeasured here), so measure where batching starts to pay off |
| **Any card, 1080p cameras** | The decoder may be the wall before the detector: on the 3090, NVDEC tops out at ~768 fps at 1080p (~25 cameras at 30 fps), below what the detector serves. NVDEC count and generation differ per card — rerun `scripts/nvdec_capacity.sh` ([decoder capacity](nvdec.md)) |
| **Other TensorRT or Triton versions** | Engines must be rebuilt, kernel choices differ, and the batch-1 penalty of a batch-8 engine may change. The auto-batching behaviour should be re-checked at `/v2/models/<model>/config` |

## Contributing a result

A run on another GPU is welcome in the repo under `results/<gpu-name>/`, with the
same bar as everything else: the GPU used exclusively, engines rebuilt, A2
re-baselined, the camera counts chosen relative to that GPU's wall, and the three
TSVs plus a note of the card's power limit and driver. See
[roadmap](roadmap.md#contributing-a-measurement).

---

[← index](../README.md) · prev: [Reproduce](reproduce.md) · next: [Decoder capacity](nvdec.md)
