# Transport — which shared memory, and when

[← index](../README.md) · prev: [DeepStream](deepstream.md) · next: [Moving fewer bytes](fewer-bytes.md)

The question: once your frames are preprocessed, how do you get them into the
server without paying for it twice?

---

## The finding

| Variant | Python client (CPU data) | C++ client (GPU data) | Why |
|---|---|---|---|
| raw gRPC | 130 fps · 7.1 ms | 222 fps · 3.3 ms | protobuf serialize + socket copies |
| **system shm** | **469 fps · 1.69 ms** | — | server reads CPU shm directly; 2 copies gone |
| **CUDA shm** | 330 fps · 2.47 ms | **654 fps · 1.28 ms** | zero-copy only when bytes already on GPU |

*One stream, capacity mode (closed loop, preprocessed frames replayed). Not
live-camera latencies; see [live traffic](live-batching.md).*

**Pick by data location, not by which sounds faster:**

- Data on GPU (C++ CUDA kernels) → **CUDA IPC shm** (3× vs raw).
- Data on CPU (numpy) → **system shm** (3.6× vs raw). CUDA shm here is *worse*
  than system shm — it just moves the H2D copy to the client side.

We went in believing CUDA shared memory would always win. It doesn't. The copy you
eliminate has to be the copy that matched where the data already lived.

## Why raw gRPC costs what it does

Triton's own server-side counters, from a separate raw-gRPC run, showed per
request (the stages overlap, so they do not sum to B1's 3.27 ms client-side p50):

- ~1.27 ms copying the input to the GPU
- ~1.25 ms copying the output back
- ~1.2 ms of actual inference

The framework was spending more time on copies than on the model. Giving the
client a CUDA shared-memory region — a buffer the client's kernel writes and the
server reads through an IPC handle, with **no copy in either direction** — took B2
from 222 to **654 fps** (one stream, capacity mode) and its latency to 1.28 ms.

That looks like 0.05 ms above A2's 1.23 ms in-process, but the two clocks differ:
A2's timer starts before its upload, B2's after it. Measured on one clock (live
30 fps cameras, timed from each frame's due time, upload included), A2 is
**0.3–0.9 ms faster per frame** (8 cameras: 1.48 vs 1.92 ms p50;
[live traffic](live-batching.md#6-choosing-a2-b2-or-b3-at-each-load)).

> **Zero-copy removes almost all of the copy cost; it does not make Triton free.**
> With CUDA shm, what remains is 0.3–0.9 ms per frame of gRPC round trip and
> scheduling, on the same clock as A2 *(corrected 2026-09-29: this box first said
> Triton's whole framework costs about a tenth of a millisecond)*.

## Region management (the part people get wrong)

One `cudaMalloc` + one IPC handle **per stream, for the stream's lifetime**. The
shm region *is* the preprocessing kernel's output buffer — the kernel writes
straight into the region the server will read.

**Never allocate or register per frame.** Registration is a server round trip;
doing it per frame reintroduces exactly the overhead shm exists to remove.

**And unregister it on every exit path.** This study's own clients did not: Triton
kept each run's regions mapped after the client exited (+10 MiB of server GPU
memory per camera per run) until the fix of 2026-09-28. A re-check found no
published number moved ([re-check](live-batching.md#re-checked-after-a-client-leak)).

## The zero-copy chain in live / RTSP mode (A2 / B2 / D)

In capacity mode the clients replay preprocessed tensors from `frames.bin`
instead, and A2's upload of each one is inside its timer.

```
NVDEC NV12 (GPU) → fused kernel reads in place → writes TRT input / shm (GPU)
                 → infer → compact kernel (GPU) → only ~KB of candidate boxes to host
```

One CUDA stream end-to-end. The full `[1,84,8400]` output tensor (2.8 MB) never
crosses to the host — a GPU compact kernel reduces it to the handful of candidate
boxes above threshold first.

## Caveat

Multi-process C2 requires unique shm region names per process (handled in
`client_v2.py` via `run_tag`); sharing region names across threads caused a data
race that invalidated the N≥4 in-process numbers. See
[contention](contention.md#caveats).

---

[← index](../README.md) · prev: [DeepStream](deepstream.md) · next: [Moving fewer bytes](fewer-bytes.md)
