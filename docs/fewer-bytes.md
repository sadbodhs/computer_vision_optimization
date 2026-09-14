# Moving fewer bytes — shrink the tensor, not the transfer

[← index](../README.md) · prev: [Transport](transport.md) · next: [Stage decomposition](stage-decomposition.md)

[Transport](transport.md) asks **how to move bytes faster** — system shared memory
on CPU paths, CUDA IPC on GPU paths, 3-3.6x either way. This page asks a question
it never did: **why are there that many bytes?**

Every flow in this study ships `[1,3,640,640]` FP32 — **4.92 MB per frame** —
because the client does the letterbox *and* the divide-by-255 before handing the
tensor over. But the pixels started life as 8-bit, and the divide is one
multiply. We are paying 4x the wire cost to deliver data that only needs 8 bits.

Scripts: [`fold_norm.py`](../scripts/fold_norm.py) ·
[`io_precision.py`](../scripts/io_precision.py) ·
raw data: [`results/v3/fewer_bytes.tsv`](../results/v3/fewer_bytes.tsv)

---

## The lever

Setting `--inputIOFormats=uint8:chw` alone does **not** work:

```
Error 3: /model.0/conv/Conv: only activation types allowed as input
```

The cast has to happen *inside* the graph. So splice a two-node head onto the
front of the ONNX — `Cast(uint8→float)` then `Div(255)` — reusing the original
input name as the Div output, so nothing downstream is rewired.
[`fold_norm.py`](../scripts/fold_norm.py) does it in about thirty lines.

| Batch-1 engine | H2D | GPU compute | D2H |
|---|---:|---:|---:|
| FP32 in, FP32 out *(every flow today)* | 0.1896 ms | 0.9792 ms | 0.1107 ms |
| **UINT8 in + in-graph ÷255** | **0.0509 ms** | 0.9750 ms | 0.1107 ms |
| **+ FP16 out** | 0.0510 ms | 0.9678 ms | **0.0581 ms** |

**H2D falls 3.72x, and the normalisation is free.** GPU compute goes 0.9792 →
0.9750 ms — unchanged. TensorRT folds the scale into the first convolution, so
you add two graph nodes and pay nothing for them.

Stacked with the [FP16 output binding](model-zoo.md#6-the-output-binding-is-fp32-and-that-is-a-choice),
per-frame transport drops **0.300 ms → 0.109 ms, a 2.75x cut**, and transport's
share of the frame falls from **23.5% to 10.1%**.

## Batching makes it work *better*

Per batch of 8:

| Batch-8 engine | H2D | GPU compute | D2H |
|---|---:|---:|---:|
| FP32 in, FP32 out | 1.4741 ms | 4.8494 ms | 0.8502 ms |
| UINT8 in | **0.3730 ms** | 4.8799 ms | 0.8489 ms |
| + FP16 out | 0.3722 ms | 4.8413 ms | **0.4309 ms** |

H2D improves **3.95x** here against 3.72x at batch 1 — closer to the theoretical
4x, because batching amortises the fixed per-transfer cost over 8x the bytes.
Transport per frame: **0.291 ms → 0.100 ms**.

## It costs no accuracy

500 COCO val2017 images, identical letterbox, conf 0.001:

| | mAP50-95 | mAP50 | mAP75 | detections |
|---|---:|---:|---:|---:|
| FP32 input | 0.47348 | 0.64524 | 0.51426 | 26,619 |
| **UINT8 + in-graph ÷255** | **0.47357** | 0.64483 | 0.51421 | 26,642 |

**+0.00009 mAP50-95.** Noise.

It is worth being precise about *why* this is free, because the intuition that it
should cost something is reasonable. `cv2.resize` on a uint8 image **already
returns uint8** — the rounding to 8 bits happens in today's pipeline too, just
before the `/255`. Shipping uint8 does not add a quantisation step; it moves the
divide from the client to the graph. The result is not bit-identical only because
the two engines were built separately and TensorRT chose slightly different
tactics.

!!! warning "This reasoning is specific to the numpy path"

    A2 and B2 letterbox in a **CUDA kernel that interpolates in float**. Writing
    uint8 from that kernel *would* add a rounding step that does not exist today.
    The numbers above are measured on the numpy path; the CUDA path needs its own
    accuracy run before the same claim is made for it.

## Where it pays — and where it cannot

The engine-level throughput barely moved: **1019 → 1031 qps, +1.2%.** Which, by
now, should be predictable — yolov8s is GPU-bound, so shrinking transport removes
something that was already hidden behind compute.

So the lever was taken to the flow where the payload **is** the constraint. B1
ships its tensor over raw gRPC with no shared memory:

| Raw gRPC, concurrency 1 | throughput | latency |
|---|---:|---:|
| FP32 input (4.92 MB) | 179.3 infer/s | 3787 us |
| **UINT8 input (1.23 MB)** | **336.3 infer/s** | **2226 us** |
| | **+87.6%** | **−41%** |

And the flow where it **cannot** help, which is the more useful half of the result:

> Batch-8 engine compute is 4.8494 ms ÷ 8 = **0.6062 ms/frame → a 1650 fps
> ceiling.** D measures **1665 fps**. D is already at the engine ceiling.

D's transport is entirely hidden behind compute, so removing two thirds of it
removes something that was costing nothing — **0% gain, measured.** The same
change, on the same GPU, with the same engine: **+87.6% on B1, 0% on D.**

That is the study's central rule in its sharpest form yet. The lever is not good
or bad; it is good exactly where the thing it shrinks was the constraint.

## Hidden in the middle, exposed at both ends

The engine-level throughput barely moved (+1.2%) while B1 nearly doubled, and
those look contradictory until you separate **latency** from **throughput**.

Single stream, one request at a time, nothing pipelined — the single-instance
case:

| | latency p50 | throughput |
|---|---:|---:|
| FP32 in, FP32 out | 1.2859 ms | 1010.1 qps |
| UINT8 in | 1.1255 ms | 1033.3 qps |
| **UINT8 in + FP16 out** | **1.0763 ms** | 1030.0 qps |
| | **−16.3%** | +2.0% |

**Latency falls 16.3%; throughput does not move.** With one request in flight
there is nothing to hide the copy behind, so the saving lands in full on the
frame. Add pipelining and request A's transfer overlaps request B's compute, and
the same saving becomes invisible.

Note the two numbers that are easily confused. The **transfer** is 3.72x cheaper.
The **frame** is 16.3% cheaper — because transport was 0.300 ms of a 1.286 ms
frame, 23.4%, and removing three quarters of 23.4% leaves ~16%. Quote the second
to a user; the first is an implementation detail.

So the benefit is not simply "large at low concurrency, shrinking as concurrency
rises". It is **U-shaped**:

| Regime | Example | Benefit |
|---|---|---:|
| Single inference, latency-bound | one camera, closed loop | **−16.3% latency** |
| Pipelined, compute-bound | D at batch-8 | **0%** |
| Payload- or bandwidth-bound | B1 raw gRPC; PCIe saturated | **+87.6%** |

Copies are shared until the bus itself becomes the constraint. D currently moves
~12.9 GB/s on a ~24 GB/s PCIe 4.0 link — about half. Add streams, add models, or
put a second pipeline on the same card, and the bytes stop being free again.

## Which models benefit: input bytes per millisecond of compute

The intuitive answer — "models with big inputs" — is wrong. **SAM ViT-H has the
largest input in the whole sweep at 12.58 MB, 2.5x a YOLO frame, and benefits
least of anything measured (0.4%).**

The predictor is input size *relative to compute*:

| Best candidates | MB/ms | saves |
|---|---:|---:|
| YOLO11n | 6.11 | **12.5%** |
| YOLO11n-seg | 5.26 | 9.9% |
| YOLOv8s | 4.98 | 10.8% |
| DeepLabV3-MNv3 | 4.39 | 5.7% |

| Worst candidates | MB/ms | saves |
|---|---:|---:|
| DINOv2-L | 0.24 | 0.7% |
| Depth Anything V2-L | 0.19 | 0.6% |
| SAM ViT-H | 0.15 | **0.4%** |

### A 3D CNN is a *worse* candidate than a 2D detector

Worth testing, because the intuition that video models — many frames per
inference — should benefit most is a reasonable one. They do not.

| `r3d_18`, 16 frames at 112x112 | H2D | GPU compute | H2D share |
|---|---:|---:|---:|
| FP32 | 0.0944 ms | 1.1646 ms | 7.5% |
| UINT8 | 0.0260 ms | 1.1675 ms | 2.2% |

It saves **5.4%** of frame time. YOLOv8s saves **10.8%** — twice as much, from a
workload that looks smaller. r3d_18 ships 2.41 MB and spends 1.165 ms on it;
YOLOv8s ships 4.92 MB and spends 0.979 ms.

3D convolutions exist to do a great deal of compute per input byte, so they hide
their own transport by construction. **Heavy architectures are poor candidates
however large their input.**

This is the [model-cost crossover](model-scaling.md) arriving from the input
side: a fixed transport cost divided by a growing compute denominator.

## The three levers, together

| Lever | Shrinks | By | Costs |
|---|---|---|---|
| [In-graph NMS](in-graph-nms.md) | output | 392x | 18.6% slower engine |
| [FP16 output binding](model-zoo.md#6-the-output-binding-is-fp32-and-that-is-a-choice) | output | 2x | nothing measured; mAP untested |
| **UINT8 input + in-graph norm** | **input** | **4x** | **nothing — +0.0001 mAP** |

The third is the only one that is unambiguously free, and it is the one nobody
reaches for. It is also what DeepStream has been doing all along: `nvinfer`
takes NVMM uint8 buffers and applies `net-scale-factor` on the GPU. Flow E never
shipped a float tensor across PCIe, which is part of why
[its per-frame cost looks the way it does](deepstream.md).

## Scope

- Measured on **yolov8s at 640x640, FP16**, on one RTX 3090; the model-selection
  ranking extrapolates each model's measured H2D, assuming UINT8 removes 73% of it
  (the figure measured directly on yolov8s at batch 1 and 8).
- The B1 A/B is `perf_analyzer` at **concurrency 1**; it would not stabilise at
  higher concurrency in synchronous mode, so the sweep is one point, not a curve.
- The accuracy run is the **numpy** preprocessing path — see the warning above.
- The C++ clients still send FP32; adopting this in A2/B2/D needs a kernel that
  writes uint8, which is not done here.

---

[← index](../README.md) · prev: [Transport](transport.md) · next: [Stage decomposition](stage-decomposition.md)
