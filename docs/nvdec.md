# Decoder capacity — how many cameras NVDEC can feed

[← index](../README.md) · prev: [On another GPU](other-gpus.md) · next: [Roadmap](roadmap.md)

Every flow in this study reports frames per second of inference. A deployment counts
**cameras**, and before the detector sees a frame, the GPU's hardware video decoder
(NVDEC) has to decode it. If the decoder runs out first, the detector's capacity is
irrelevant. This page measures the decoder alone: *N* concurrent decode sessions,
each decoding as fast as it can, with frames kept on the GPU and no inference.

Script: [`nvdec_capacity.sh`](../scripts/nvdec_capacity.sh) · data:
[`results/v3/nvdec_capacity.tsv`](../results/v3/nvdec_capacity.tsv)

---

## One decoder, two ceilings

Sources: the study's own video (H.264, 640×360) and a 1080p H.264 version of it.
Sessions 1–32, 3 interleaved repeats, the decoder's utilisation sampled every 250 ms:

| Sessions | 640×360: fps (cameras at 30 fps) | 1080p: fps (cameras at 30 fps) |
|---:|---:|---:|
| 1 | 2,270 (75.7)* | 725 (24.2) |
| 2 | 2,387 (79.6) | 755 (25.2) |
| 4 | 2,476 (82.5) | 766 (25.5) |
| 8 | 2,524 (84.1) | 767 (25.6) |
| 16 | 2,532 (84.4) | 769 (25.6) |
| 32 | 2,526 (84.2) | 768 (25.6) |

\* One single-session 640×360 repeat reached only 1,303 fps with the decoder 69% busy;
the median of the three is shown. Every other run held the decoder at 98–100%.

**There is no session limit.** Thirty-two concurrent sessions all ran. (The cap on
consumer cards applies to NVENC, the encoder, not to NVDEC.) **The decoder is the
limit, not the number of sessions**: one 1080p stream already keeps it 100% busy, and
adding sessions only shares a fixed throughput. At 1080p a 3090 decodes about
**768 frames a second, 25 cameras at 30 fps**; at 640×360 about **2,530, or 84
cameras**. The ratio, 3.3× for 9× fewer pixels, says the decoder has a per-frame cost
as well as a per-pixel one.

## At 1080p, the decoder runs out before the detector

The same GPU runs yolov8s at 640 at about 1,190 fps (A2, batch 1) and about 1,620 fps
batched ([results](results.md), [batching](batching.md#correction-the-throughput-lead-is-batching-not-triton)).
Against 768 fps of 1080p decoding, **a 1080p camera fleet is decoder-bound at about 25
cameras per 3090**, with only half to two-thirds of the detector's capacity in use. At the study's
640×360, the decoder (84 cameras) outruns the detector (40–54 cameras), which is why
the rest of this study never hit it.

For sizing: count **decoded pixels** as well as inference. A card with more NVDEC
engines (data-centre parts carry several), or cameras streaming at lower resolution,
moves the ceiling; on this card, at 1080p, nothing on the inference side does.

## How it was measured, and two attempts that were not measurements

- **Hardware decoding only.** Each session is `ffmpeg -hwaccel cuda
  -hwaccel_output_format cuda -c:v h264_cuvid`, frames kept on the GPU, output
  discarded.
- **The first attempt measured the CPU.** ffmpeg's generic `-hwaccel cuda` path failed
  to create a hardware decoder in this container (`cuvidCreateDecoder:
  CUDA_ERROR_INVALID_VALUE`) and **silently fell back to CPU decoding, still exiting
  successfully**. The decoder sat at 0% while the runs reported ~12,800 fps. The
  harness now forces the CUVID decoder and rejects any run whose decoder utilisation
  never passes 50%. Anyone measuring NVDEC with ffmpeg should check the decoder
  utilisation, not the exit code. The study's own C++ NVDEC path was checked
  (A2 on a live RTSP camera) and works.
- **Shared host, capped at 6 cores.** The host was busy for hours with an unrelated CPU
  job (load 1–24 during this run, recorded per row). Every ffmpeg ran under
  `taskset` on 6 fixed cores, and the test used at most **1.27 cores**, measured per
  run from the container's CPU accounting. A validity rule was committed before the
  run: the 4-session points had to land within 5% of their values on a quiet host
  (2,462 and 761 fps). They landed at +0.6% and +0.7%. The decoder, not the CPU, sets
  these numbers.

Both earlier attempts are kept in `results/v3/` and marked excluded.

## What was predicted

| | Prediction | Measured | |
|---|---|---|---|
| P1 | No session limit: 32 concurrent sessions all run | 32 ran, none failed | **held** |
| P2 | At 1080p the decoder saturates at 700–1,200 fps (25–40 cameras) | 768 fps, 25.6 cameras | **held** |
| P3 | At 640×360 the aggregate is ≥ 3,000 fps, limited by the CPU, not NVDEC | ~2,530 fps with the decoder at 100% | **failed**: the decoder is the limit there too |
| P4 | At 1080p the decoder, not the detector, caps one GPU | 768 fps decoded vs ~1,190–1,620 fps detected | **held** |

The 4-session measurements from the diagnosis (the quiet-host reference values) were
taken before the full sweep; the predictions above were committed earlier and not
changed.

---

[← index](../README.md) · prev: [On another GPU](other-gpus.md) · next: [Roadmap](roadmap.md)
