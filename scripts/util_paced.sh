#!/usr/bin/env bash
# GPU utilisation, SM clock and power at each load, for every paced-mode config.
#
# The paced sweeps (b3_paced.sh, a2_paced.sh) say which pipeline is fastest at
# each camera count ON AN RTX 3090. Camera counts do not transfer to another GPU;
# how busy the GPU was at that load does. This re-runs each cell while sampling
# nvidia-smi every 200 ms, keeping only samples from the measured window (after
# the client's start line and warmup).
#
# Caveat recorded with the data: nvidia-smi's utilization.gpu is the fraction of
# time ANY kernel was running, not how full the SMs were. It reaches 100% before
# the GPU runs out of compute, so it marks "never idle", not "no headroom left".
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:-$ROOT/results/v3/gpu_util_paced.tsv}
C=${CONTAINER:-triton-server}
CAMS=${CAMS:-"1 4 8 16 32 48"}
ARMS_LIST=${ARMS_LIST:-"a2 b2 b3_0 b3_500 dnow"}
DURATION=${DURATION:-13}; WARMUP=${WARMUP:-1}; FPS=${FPS:-30}; SETTLE=${SETTLE:-3}
CFG=$ROOT/triton/models/yolov8s_dyn/config.pbtxt
if ! git -C "$ROOT" diff --quiet -- triton/models/yolov8s_dyn/config.pbtxt; then
  echo "refusing: yolov8s_dyn/config.pbtxt has uncommitted changes" >&2; exit 1
fi
ready() { for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready)" = 200 ] && return 0; sleep 2; done; return 1; }
RESTORED=0
restore() {
  [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  echo "--- restoring yolov8s_dyn config from git and releasing the GPU lock ---"
  git -C "$ROOT" checkout -- triton/models/yolov8s_dyn/config.pbtxt
  docker restart "$C" >/dev/null 2>&1; ready || echo "WARNING: server not ready after restore"
  bash "$ROOT/scripts/gpu_lock.sh" release bench || true
}
trap restore EXIT
trap 'echo "--- interrupted ---"; exit 130' INT TERM

write_dyn() {
  local DB
  case $1 in
    dnow)   DB='dynamic_batching { preferred_batch_size: [ 4, 8 ] max_queue_delay_microseconds: 5000 }' ;;
    b3_500) DB='dynamic_batching { preferred_batch_size: [ 2, 4, 8 ] max_queue_delay_microseconds: 500 }' ;;
    b3_0)   DB='dynamic_batching { max_queue_delay_microseconds: 0 }' ;;
  esac
  cat > "$CFG" <<CFGEOF
name: "yolov8s_dyn"
platform: "tensorrt_plan"
default_model_filename: "model.plan"
max_batch_size: 8
$DB
instance_group [ { count: 2 kind: KIND_GPU } ]
input [ { name: "images" data_type: TYPE_FP32 dims: [ 3, 640, 640 ] } ]
output [ { name: "output0" data_type: TYPE_FP32 dims: [ 84, 8400 ] } ]
CFGEOF
}

client() {  # $1 arm  $2 cameras  $3 duration
  local common="--mode paced --fps $FPS --phase random --seed 1 --warmup $WARMUP --file frames.bin --streams $2 --duration $3"
  case $1 in
    a2) docker exec "$C" bash -lc "cd /work/cpp/build && ./trt_pipeline_cuda --engine /models/yolov8s/1/model.plan $common" ;;
    b2) docker exec "$C" bash -lc "cd /work/cpp/build && ./trt_grpc_cuda --model yolov8s $common" ;;
    *)  docker exec "$C" bash -lc "cd /work/cpp/build && ./trt_grpc_cuda --model yolov8s_dyn $common" ;;
  esac
}

mkdir -p "$(dirname "$OUT")"
printf 'arm\tcameras\toffered_fps\tfps\tp50_ms\tp99_ms\tutil_mean_pct\tutil_p95_pct\tsm_clock_mean_mhz\tpower_mean_w\tsamples\n' > "$OUT"

for arm in $ARMS_LIST; do
  case $arm in a2|b2) ;; *) write_dyn "$arm"; docker restart "$C" >/dev/null; ready || { echo "server down: $arm" >&2; continue; }; sleep 3 ;; esac
  client "$arm" 4 3 >/dev/null 2>&1          # discarded warm run
  for N in $CAMS; do
    sleep "$SETTLE"
    S=/tmp/util_${arm}_${N}.csv
    nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw --format=csv,noheader,nounits -lms 200 > "$S" &
    SPID=$!
    T0=$(date +%s.%N)
    J=$(client "$arm" "$N" "$DURATION" 2>/dev/null)
    T1=$(date +%s.%N)
    kill "$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
    [ -z "$J" ] && { echo "  $arm N=$N FAILED" >&2; continue; }
    CELL_JSON="$J" python3 - "$arm" "$N" "$T0" "$T1" "$S" "$OUT" "$WARMUP" <<'PY'
import datetime, json, os, statistics as st, sys
arm, N, T0, T1, S, out, warm = sys.argv[1:8]
N, T0, T1, warm = int(N), float(T0), float(T1), float(warm)
d = json.loads(os.environ["CELL_JSON"])
lo = T0 + 2.0 + 0.05 * N + warm + 1.0      # client start line + warmup + engine-load margin
hi = T1 - 1.0
u, clk, pw = [], [], []
for line in open(S):
    p = [x.strip() for x in line.split(",")]
    if len(p) < 4:
        continue
    try:
        ts = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
        if lo <= ts <= hi:
            u.append(float(p[1])); clk.append(float(p[2])); pw.append(float(p[3]))
    except ValueError:
        pass
if not u:
    print("  %s N=%d: no samples in window" % (arm, N)); sys.exit()
u.sort()
p95 = u[min(int(0.95 * len(u)), len(u) - 1)]
row = [arm, str(N), "%.0f" % d["offered_fps"], "%.1f" % d["fps"], "%.3f" % d["lat_ms_p50"],
       "%.3f" % d["lat_ms_p99"], "%.1f" % st.mean(u), "%.0f" % p95, "%.0f" % st.mean(clk),
       "%.1f" % st.mean(pw), str(len(u))]
open(out, "a").write("\t".join(row) + "\n")
print("  %-7s N=%-3d offered %5.0f  p50 %8.3f ms  GPU util %5.1f%% (p95 %3.0f%%)  SM %4.0f MHz  %5.1f W  [%d samples]"
      % (arm, N, d["offered_fps"], d["lat_ms_p50"], st.mean(u), p95, st.mean(clk), st.mean(pw), len(u)))
PY
    rm -f "$S"
  done
done
echo "wrote $OUT"
