# In-graph NMS — paying 19% of the engine to save ~1.2 ms of wire

[← index](../README.md) · prev: [CUDA graphs](cuda-graphs.md) · next: [Contention](contention.md)

Every flow in this study returns the raw detection head — `[1,84,8400]` FP32,
**2.82 MB per frame** — and does NMS afterwards. Folding NMS into the engine
instead makes the output `[1,300,6]`, **7.2 KB**: a **392× smaller** payload.

The trade is not obvious in advance. In-graph NMS makes the *engine* slower,
because the network now carries TopK and NonMaxSuppression work it did not have
before. The question is whether the transport saving is bigger than that penalty.

Script: [`scripts/probe_transport.py`](../scripts/probe_transport.py) ·
raw data: [`results/v3/in_graph_nms.tsv`](../results/v3/in_graph_nms.tsv).

---

## The two halves of the trade

**The engine gets slower** (`trtexec`, batch-1 FP16, no client involved):

| Engine | Throughput |
|---|---|
| stock `[1,84,8400]` | 1023.26 qps |
| in-graph NMS `[1,300,6]` | **833.38 qps (−18.6%)** |

**The round trip gets faster** — same input, same server, raw gRPC, no shared
memory. fps is the median of 3 repeats; latency was recorded for one run only:

| Model | Output | fps | p50 (1 run) |
|---|---|---|---|
| `yolov8s` | 2.82 MB | 258.9 | 3.789 ms |
| `yolov8s_nms` | **7.2 KB** | **344.7 (+33%)** | **2.816 ms (−26%)** |

*Corrected 2026-09-29: this table first showed a 2.862 ms p50 and p95 values that
are not in the raw data; the p50 is now the recorded one, and p95 was not recorded.*

## Reading it

The engine penalty is about **0.22 ms/frame** (1/833 − 1/1023). The measured
round-trip saving is **~0.97 ms/frame** at p50 (0.96 ms from the fps medians). So
shipping that 2.82 MB output costs roughly **1.2 ms per frame** on a raw gRPC path — and paying 0.22 ms of GPU to
avoid it is a clearly good deal: **+33% throughput net**.

> **On a raw-gRPC path, returning the 2.82 MB output costs ~1.2 ms per frame —
> more than an 18.6% slower engine.** The 4.9 MB input was not varied, so its share
> of the round trip is not measured here.

## Which flows this actually helps

This matters in proportion to how much of the output crosses a boundary:

- **B1 (raw gRPC) and C1 (Python gRPC)** — the full 2.82 MB is serialised
  and copied. These are the flows the +33% applies to.
- **C2 (Python, system shared memory)** — skips serialisation but still copies
  2.82 MB per frame through host shared memory, so it would gain less. Not
  measured.
- **A2 (in-process)** — never sends the output anywhere; a GPU compact kernel
  already reduces it to a few KB before it reaches the host
  ([stage decomposition](stage-decomposition.md)). In-graph NMS would replace that
  kernel, not add to it, and would cost 18.6% of engine throughput for a saving
  A2 has already made.
- **B2/D (CUDA shared memory)** — the output stays in GPU memory and the client's
  compact kernel reads it there, so the transport is already avoided. The engine
  penalty would likely dominate. **Not measured** — the C++ clients expect
  `[1,84,8400]` and would need reworking for the `[1,300,6]` shape.

So the honest summary is: **in-graph NMS is a large win for the naive paths and
probably a loss for the zero-copy ones.** It is a different solution to the same
problem the compact kernel already solves, and the flows that took the trouble to
solve it properly gain nothing.

## Caveats

- The ONNX ultralytics produces uses **ONNX-native NMS ops** (TopK +
  NonMaxSuppression), not the `EfficientNMS_TRT` plugin. A plugin build would very
  likely shrink the 18.6% engine penalty, and is untested here.
- Fixed `[1,300,6]` output means a hard cap of 300 detections per image. Fine for
  these scenes; not universally.
- The round-trip probe deliberately does **no postprocessing**, so it isolates
  transport. Real clients would additionally save the CPU-side NMS the stock path
  still has to run — this measurement understates the total benefit.
- Accuracy of the in-graph NMS build was not measured; its NMS parameters are
  ultralytics' defaults, not the study's conf 0.25 / IoU 0.45 contract.

---

[← index](../README.md) · prev: [CUDA graphs](cuda-graphs.md) · next: [Contention](contention.md)
