# results/ — provenance

[← index](../README.md)

What each file is, and how it maps to the tables in the top-level README/STORY.

## Layout

| Path | Pass | What it is |
|---|---|---|
| `*_single_*.json`, `*_3stream_*.json` | **v1 (flawed)** | The first, source-capped pass (STORY §3). Kept for honesty. `latency_ms_est` here is `1000/fps`, a derived reciprocal — **not** a measured p50. Do not cite these as pipeline capacity. |
| `v2/all_*.tsv` | **v2 (corrected)** | The 4-arm capacity + RTSP sweep from `scripts/benchmark_v2.sh`. One JSON per repeat, `arm|json`. |
| `v2/gpu_*.csv`, `gpu_*.csv`, `v3/gpu_*.csv` | v2/v3 | GPU util/mem samples (`scripts/gpu_sample.sh`) taken during runs. |
| `capacity_table.tsv` | — | The canonical capacity numbers (flow × concurrency → fps, latency) as data, so `scripts/make_plots.py` can regenerate the figures. Mirrors the table in `docs/results.md`. |
| `stage_decomposition.tsv` | — | Per-stage per-frame times behind the stage-decomposition figure. |
| `comparison_tables.md` | v2 | The fair tables (per-frame latency, per-frame GPU cost) + Flow E (DeepStream). |
| `v3/parallel_contention_N3.tsv` | v3 | Multi-instance contention at N=3, MPS off (`scripts/parallel_test.sh`): A2, B2 and C2, one row per instance, one repeat. One B2 instance ran at 4.46 ms p50 against 2.22-2.32 for the other two. |
| `v3/accuracy.tsv` | v3 | COCO val2017 mAP (500 imgs): study chain vs ultralytics reference (`scripts/accuracy_eval.py`, `scripts/accuracy_reference.py`). Includes the nearest-resize row (the original bug) and the calibrated INT8 row. |
| `v3/deepstream_capacity.tsv` | v3 | E2 capacity-mode attempt. **Harness-bound, not a DeepStream ceiling** - GPU median 0% at 8 streams. Do not quote 501 fps as E2 capacity. |
| `v3/model_scaling.tsv` | v3 | The model-cost ladder (YOLO11 n/s/m/l/x + yolov8s control) through A2, 3 interleaved repeats (`scripts/model_scaling.sh`). `non_engine_ms` is p50 minus engine time - the column the page is about. |
| `v3/fewer_bytes.tsv` | v3 | Engine-level A/B of UINT8 input + in-graph /255 (`scripts/fold_norm.py`), batch 1 and 8, against FP32 and against the FP16 output binding. |
| `v3/fewer_bytes_flows.tsv` | v3 | The same change taken to the flows: +87.6% on B1 (payload-bound, `perf_analyzer` conc=1) and 0% possible on D, which already sits at the 1650 fps batch-8 engine ceiling. |
| `v3/fewer_bytes_latency.tsv` | v3 | Single-stream, nothing pipelined: the case where the saving is fully exposed. Latency -16.3%, throughput +2.0% - the same change, read two ways. |
| `v3/fewer_bytes_3dcnn.tsv` | v3 | Tests the intuition that video models benefit most. They do not: r3d_18 saves 5.4% of frame time against yolov8s's 10.8%, because 3D convolutions do far more compute per input byte. |
| `v3/fewer_bytes_accuracy.tsv` | v3 | COCO val2017 (500 imgs) for the UINT8 path: +0.00009 mAP50-95. Free because cv2.resize already returns uint8 - the divide moves, no quantisation is added. Numpy path only; the CUDA-kernel path is untested. |
| `v3/gpu_util_paced.tsv` | v3 | GPU utilisation, SM clock and power at each paced load, all five configs (`scripts/util_paced.sh`, nvidia-smi every 200 ms inside the measured window, one repeat, unsynchronised). `utilization.gpu` is the fraction of time any kernel ran, not SM occupancy. |
| `v3/a2_paced.tsv` | v3 | A2 in paced mode (`scripts/a2_paced.sh`, `trt_pipeline_cuda --mode paced`): same cameras, seeded phases and clock as `b3_paced.tsv`, plus `b2_anchor` cells re-run in this sweep to show the two files compare (5 of 6 within 0.2-4.1%, one burst cell 12%). |
| `v3/leakfix_b3_slice.tsv`, `v3/leakfix_a2_slice.tsv` | v3 | Re-check of `b3_paced.tsv` / `a2_paced.tsv` after fixing the Triton clients' CUDA shared-memory leak (`scripts/leakfix_check.sh`, predictions committed first; compared by `scripts/leakfix_compare.py`): b2, b3_0, dnow and A2 (control) at 8/32/48 cameras, both phases, seeds 1-3. All cells within 10% of published; one tie at 48 synchronised cameras swapped order. |
| `v3/leakfix_b3_slice_aborted_cpuload.tsv` | v3 | **Excluded.** First attempt at the re-check above, stopped because an unrelated CPU-only job (load ~12) was running on the host; kept only as the record of why. |
| `v3/batch8_a2_vs_d.tsv` | v3 | Part B: A2 at batch 1, A2 at batch 8 (`trt_pipeline_cuda --batch 8`, dynamic-batch engine, capacity mode) and D (8/16 in flight), one session, 3 interleaved repeats (`scripts/batch8_a2_vs_d.sh`, predictions committed first). Batch-8 A2 ties D (1622 vs 1624 fps) at 10 ms p50 against D's 37 ms: the published single-model lead was the batch size. One row per run. |
| `v3/multimodel_a2_vs_d.tsv` | v3 | Three models at once (yolov8n, yolov8s, yolo11n): D's multi-model serving (9/18 streams) against A2 running the same three in one process (`trt_pipeline_cuda --engines`, batch 8 and batch 1), one session after restarting triton-server onto the rebuilt engines, 3 interleaved repeats (`scripts/multimodel_a2_vs_d.sh`, predictions committed first). Batch-8 A2 2,280 fps vs D 1,780: the published multi-model lead reverses. One row per run. |
| `v3/partb_rebuild.tsv` | v3 | Part B: every engine rebuilt by `scripts/export_models.sh` (after fixing it to install headless OpenCV), the rebuilt `yolov8s_u8` timed with trtexec (0.980 ms vs published 0.9750) and A2 re-measured on the rebuilt `yolov8s` (within 0.8-2.2% of the old engine). Old engines backed up outside the repo. |
| `v3/partb_rebuild_aborted_exportfailed.tsv` | v3 | **Excluded.** First attempt: the export failed (missing `libxcb`) and the check timed the OLD engines under the label "rebuilt". Kept as the record of why the check now refuses stale engines. |
| `v3/nvdec_capacity.tsv` | v3 | Part B: NVDEC decode capacity (`scripts/nvdec_capacity.sh`, CUVID decoder, frames on the GPU): 640x360 and 1080p H.264, 1-32 concurrent sessions, 3 interleaved repeats, decoder utilisation sampled per run. Shared-host mode: ffmpeg pinned to 6 cores (used at most 1.27), load and cores used recorded per row; the 4-session controls landed within 0.7% of their quiet-host values (pre-set 5% rule). |
| `v3/nvdec_capacity_aborted_cpufallback.tsv`, `v3/nvdec_capacity_aborted_cpuload.tsv` | v3 | **Excluded.** NVDEC attempts: the first measured CPU decoding (ffmpeg's `-hwaccel cuda` fell back silently, decoder at 0%); the second was overrun by an unrelated job (load 82). |
| `v3/b3_paced.tsv` | v3 | Live camera traffic (`scripts/b3_paced.sh`, client `--mode paced`): 5 server configs x unsynchronised/synchronised phases x 1-56 cameras x 3 interleaved repeats. Turnaround is measured from each frame's due time and includes the upload from pageable memory, so it is NOT comparable to capacity-mode latencies. The `nobatch` arm is not batching-off: Triton auto-completes a 0 us dynamic batcher for it, so it replicates `b3_0`. |
| `v3/instance_grid.tsv` | v3 | `instance_group count` x concurrency x flow (`scripts/instance_grid.sh`), with mean batch size per cell. Instances buy B2 up to +45.9% and D at most +8.3%, because batching already does the same job; D's batch size falls as instances rise, which is the trade made visible. |
| `v3/d_latency_decomposition.tsv` | v3 | Flow D's client-observed p50 split against Triton's own per-request counters (`scripts/d_decompose.sh`). Shows the batching queue is 5.5-20% of the latency at concurrency 1-4, and that `8N / fps` (Little's law on the client's in-flight depth) predicts the p50 across a 12x range. |
| `v3/batch_achieved.tsv` | v3 | What batch size flow D actually forms, from Triton's metrics counters either side of each run (`scripts/batch_achieved.sh`). Mean batch is **4.00** at concurrency 1-4, not the 8 the config allows. The `svc_us_per_frame` and `queue_us_per_frame` columns are Triton's own attribution, which credits every request in a batch with the full batch time - they are batch-execution times, not per-frame. |
| `v3/io_precision.tsv` | v3 | Output-binding A/B: FP32 (trtexec default) vs FP16 `--outputIOFormats`, 7 models (`scripts/io_precision.py`). D2H halves everywhere; throughput moves only on DeepLabV3, the one model where D2H exceeded GPU compute. |
| `v3/model_zoo.tsv` | v3 | 22 models across classification, detection, segmentation and dense backbones, engine-level (`scripts/engine_sweep.py`). H2D/GPU/D2H from `trtexec`; output shapes parsed from its build log. `params` is counted from ONNX initializer dims and `input_px` is H*W, so the table supports asking what model size predicts. **Not comparable to the A2 columns in `model_scaling.tsv`.** |
| `v3/inspection_encoders.tsv` | v3 | Manufacturing inspection, Phase 1-pre: six stage-2 anomaly encoders (WRN50+PatchCore head, EfficientAD-S with in-graph map, DINOv2 ViT-S/B patch tokens, ResNet-18 L1-3, U-Net R34 reference) at 256/512 (ViTs 252/504), batch 1 and 8, **random weights** (`scripts/inspection_encoders.py`). Engine-level `trtexec`, each engine built once and timed in 3 interleaved rounds; medians, with `gpu_ms_min/max` across repeats. `*_per_frame` columns divide by batch. Comparable to `model_zoo.tsv`. Continued, with every later phase, in [sadbodhs/manufacturing_inspection](https://github.com/sadbodhs/manufacturing_inspection). |
| `v3/inspection_encoders_raw.tsv` | v3 | Every repeat behind `inspection_encoders.tsv` (18 engines x 3 rounds). |
| `v3/in_graph_nms.tsv` | v3 | Output-size A/B: engine throughput and raw-gRPC round trip for the stock `[1,84,8400]` head vs an in-graph-NMS `[1,300,6]` build (`scripts/probe_transport.py`). |
| `v3/batching_knobs.tsv` | v3 | `preferred_batch_size` x `max_queue_delay_microseconds` sweep on flow D at concurrency 1 and 8 (`scripts/batching_knobs.sh`). |
| `v3/triton_knobs.tsv` | v3 | `instance_group count` x CUDA graphs sweep on B2 (`scripts/triton_knobs.sh`), 3 repeats per cell. |
| `v3/precision_ceilings.tsv` | v3 | INT8 / sparsity **speed ceilings** (`scripts/precision_ceilings.sh`), uncalibrated — accuracy invalid — plus one `int8_calibrated` row (1358.72 qps) whose mAP is in `accuracy.tsv`. |
| `v3/cuda_graphs.tsv` | v3 | CUDA Graphs A/B across all three engines (`scripts/cuda_graphs.sh`), 3 repeats per cell. |
| `v3/cuda_graphs_pipeline.tsv` | v3 | CUDA graphs inside A2 (`trt_pipeline_cuda --cuda-graph`, `scripts/cuda_graphs_pipeline.sh`), capacity mode: graph off/on at 1, 2 and 4 streams, 5 repeats at 1 stream and 3 above. +11.8% at 1 stream (~799 → ~894 fps, means); plateau at 4 streams ~1167 → ~1249. |
| `v3/mps_contention_N3.tsv` | v3 | CUDA MPS A/B at N=3 for A2 and B2 (`scripts/mps_contention.sh`), 3 repeats per condition, 3 instances each. Medians — A2: 951.8 fps / 3.145 ms off, 1258.9 fps / 2.487 ms on. B2: 977.2 / 2.295 off, 993.6 / 2.760 on (B2 p95 4.51 → 2.80 ms). |
| `v3/reproduction_repeats.tsv` | v3 | The five re-runs of the two out-of-variance cells discussed below (A2 @ conc 2, D @ conc 4). |

## Reading the v2 TSVs (important)

`arm|json`. In the **first** committed `all_*.tsv` files the `A_cap_*` and
`trtexec_*` rows are **blank** — that harness (v2.1) invoked the CPU-path binary
name (`trt_pipeline`) and an empty engine-cap capture, and those arms produced no
JSON. The A2 and engine-ceiling numbers in the README were taken from **direct
binary runs**, not those rows.

`scripts/benchmark_v2.sh` has since been corrected (v2.2) to invoke the exact
binaries the published tables report — `trt_pipeline_cuda` (A2), `trt_grpc_cuda`
(B2), `client_v2.py` (C2), `trt_grpc_async` (D) — and two bugs that produced
those blank rows were fixed: the `trtexec` cap parsed a fixed awk column (which
picked up the `[I]` log-level tag), and `client_v2.py` deadlocked the C2 arm
(see the commit for `for...else` / shm-lifecycle details).

## The complete run — `v2/all_20260910_084552.tsv`

**Every arm populated, no blank rows.** 82 rows: A2/B2/C2/D × concurrency
1/2/4/8/16 × 3 repeats, plus RTSP arms, 3-process arms, and engine caps.

Engine ceilings now captured (vs the README's published figures):

| Engine | measured | published |
|---|---|---|
| yolov8s (batch-1) | **1022.98** qps | 1028 |
| yolov8n | **1490.79** qps | 1490 |
| yolo11n | **1252.58** qps | 1259 |

Capacity medians vs the published table — **18 of 20 cells within ±2%**, the
stated run-to-run variance:

| Flow | conc 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| A2 | 796 (−1.6%) | 1084 (**−11.1%**) | 1168 (−0.6%) | 1178 (−0.7%) | 1154 (−0.5%) |
| B2 | 645 (−1.4%) | 938 (−1.4%) | 1090 (−0.2%) | 1109 (−2.0%) | 1114 (−1.2%) |
| C2 | 464 (−1.0%) | 747 (+1.5%) | 1000 (+1.4%) | 1047 (+1.0%) | 1026 (−1.2%) |
| D | 1047 (+0.6%) | 1129 (−0.6%) | 1276 (**−7.4%**) | 1617 (−1.4%) | 1672 (+0.4%) |

**Two cells fell outside variance. Both have since been re-run 5 times, and they
turned out to have different explanations:**

**A2 @ conc=2 — the cell is unstable, not wrong.** Five repeats:
`1226 · 1213 · 1087 · 1101 · 1107` fps. It is *bimodal*: it either reaches ~1220
or settles near ~1100, spread **138.6 fps (12.5%)**. Both the published 1219 and
the 1084 re-run are inside its range, so neither is an error — but this cell does
not honour the study's stated "<±2% variance", and it is the least reliable number
in the capacity table. That matters because it is the cell behind the claim that
**A2 saturates at concurrency 2**; on the low runs the plateau does not begin
until concurrency 4.

**D @ conc=4 — stable, and the published figure is ~3% high.** Five repeats:
`1343 · 1337 · 1336 · 1335 · 1349` fps, spread **13.6 fps (1.0%)**. The reliable
value is **~1337**, against 1378 published. Well within the kind of drift you get
across separate sessions, and the shape of D's curve is unaffected.

The published tables are left as they are — they are a real run, and re-running
does not make an earlier honest measurement retroactively wrong. What changes is
the confidence attached: treat A2 @ conc=2 as ±12%, and D @ conc=4 as ~1337.

## Regenerating

```bash
scripts/make_frames.sh videos/real.mp4 500   # capacity-replay input
scripts/benchmark_v2.sh 10 3                  # full sweep -> results/v2/all_<ts>.tsv
scripts/parallel_test.sh 8 3                  # contention -> results/v3/
```
