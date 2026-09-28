#!/usr/bin/env bash
# Re-check of the live-traffic (paced) results after the CUDA shared-memory leak fix.
#
# THE BUG. trt_grpc_cuda (B2/B3) and trt_grpc_async (D) registered their CUDA
# shared-memory regions with Triton and never unregistered them. Triton then
# kept each run's GPU buffers mapped after the client exited: measured on
# 2026-09-28, 2 regions and +10 MiB of server GPU memory per camera per run.
# The sweeps behind b3_paced.tsv, a2_paced.tsv and gpu_util_paced.tsv ran with
# it. They restart Triton whenever the batching config changes, so the leak
# peaked at a few GB within one server lifetime - held memory, not compute. No
# drift is visible across repeats in the published data (median p50 of repeat 3
# vs repeat 1: B2 0.97, B3 0 us 0.99, D config 1.00; A2 control 1.00), but that
# is an inference. This re-runs a slice with the fixed clients to measure it.
#
# THE SLICE. Same scripts, seeds (1-3), durations and interleaving as the
# published sweeps, at 8, 32 and 48 cameras, both phases:
#   b3_paced.sh  arms b2, b3_0, dnow            -> results/v3/leakfix_b3_slice.tsv
#   a2_paced.sh  A2 (never used Triton: control) + B2 anchors
#                                               -> results/v3/leakfix_a2_slice.tsv
# Compared by scripts/leakfix_compare.py against the published medians.
#
# PREDICTIONS (written before running):
#   P1  The fix works: no region is left registered after any run (shm_clear.sh
#       never reports one) and Triton's GPU memory ends where it started.
#   P2  No correction needed: every Triton cell below capacity (median of 3 p50)
#       lands within +-10% of the published median. The cross-sweep anchors
#       reproduced within 0.2-4.1%, one burst cell at 12%, so 10% is the noise
#       bar. Overloaded cells (B2 and A2 at 48 cameras) are only checked for
#       staying overloaded (p50 > 100 ms): their queue grows with run length.
#   P3  The A2 control meets the same bar against a2_paced.tsv. If P2 fails and
#       P3 holds, the Triton side moved (the leak, or something else there) and
#       a correction is published. If both fail, the rig drifted.
#   P4  The fastest pipeline (lowest p50) at each load and phase in the slice is
#       the one the published selection chart names.
#
# The caller holds the GPU lock as "bench". The sub-scripts release it on exit,
# so it is re-taken (waiting if another tab got in first) between them.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$ROOT/scripts/gpu_lock.sh"
C=${CONTAINER:-triton-server}
B3OUT=$ROOT/results/v3/leakfix_b3_slice.tsv
A2OUT=$ROOT/results/v3/leakfix_a2_slice.tsv
CAMS_SLICE="8 32 48"

# Paced timings depend on the host CPU as much as the GPU, and a CPU-only job
# never takes the GPU lock. The first attempt ran beside a 7-core job
# (load ~12) and was stopped: B2 read 23-29% slow on the same seeds.
MAX_LOAD=${MAX_LOAD:-3}
load=$(cut -d" " -f1 /proc/loadavg)
if awk -v l="$load" -v m="$MAX_LOAD" "BEGIN{exit !(l>m)}"; then
  echo "refusing: host load average $load > $MAX_LOAD - paced timings would be distorted" >&2
  ps -eo pcpu,etime,args --sort=-pcpu | head -4 >&2
  bash "$LOCK" release bench >/dev/null 2>&1
  exit 1
fi

server_mib() {
  local p; p=$(docker top "$C" -eo pid,comm 2>/dev/null | awk '$2=="tritonserver"{print $1; exit}')
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' -v p="$p" '$1==p{print $2}'
}

echo "--- Triton GPU memory before: $(server_mib) MiB"
CAMS="$CAMS_SLICE" ARMS_LIST="b2 b3_0 dnow" bash "$ROOT/scripts/b3_paced.sh" "$B3OUT" 2>&1 | tee "$ROOT/results/v3/leakfix_b3_slice.log"
echo "--- Triton GPU memory after B3 slice: $(server_mib) MiB"

bash "$LOCK" wait bench "leakfix check: A2 control slice"
CAMS="$CAMS_SLICE" ANCHOR_CAMS="$CAMS_SLICE" bash "$ROOT/scripts/a2_paced.sh" "$A2OUT" 2>&1 | tee "$ROOT/results/v3/leakfix_a2_slice.log"
echo "--- Triton GPU memory after A2 slice: $(server_mib) MiB"
echo "--- leftover-region warnings in the logs (P1 expects 0):"
grep -c "still registered" "$ROOT"/results/v3/leakfix_*_slice.log

python3 "$ROOT/scripts/leakfix_compare.py" "$ROOT"
