#!/usr/bin/env bash
# B3: per-frame turnaround of LIVE, camera-paced traffic under each server config.
#
# Every D latency in this repo comes from capacity mode - a closed-loop client
# holding 8 requests in flight per stream - so it measures a queue the client
# builds itself. A live camera has at most one frame in flight. This runs N
# open-loop virtual cameras at 30 fps (trt_grpc_cuda --mode paced) and measures
# turnaround from each frame's DUE time, against five server configurations of
# the same engine family:
#
#   b2       yolov8s,     batch-1 engine, no batching           (published B2)
#   nobatch  yolov8s_dyn, batching OFF                           (engine effect only)
#   dnow     yolov8s_dyn, preferred [4,8], 5000 us               (published D config)
#   b3_500   yolov8s_dyn, preferred [2,4,8], 500 us              (B3)
#   b3_0     yolov8s_dyn, 0 us, no preferred sizes               (B3, no window)
#
# All Triton arms: 2 instances. yolov8s_dyn's config is swapped per arm and the
# original is restored on exit, including on error. Arm order rotates per repeat
# so drift cannot accumulate in whichever runs last. Any GPU process that is not
# ours is recorded against the row it overlapped.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:-$ROOT/results/v3/b3_paced.tsv}
C=${CONTAINER:-triton-server}
REPEATS=${REPEATS:-3}
DURATION=${DURATION:-13}           # includes WARMUP
WARMUP=${WARMUP:-1}
FPS=${FPS:-30}
SETTLE=${SETTLE:-3}
CAMS=${CAMS:-"1 4 8 16 32 48 56"}
PHASES=${PHASES:-"random sync"}
ARMS=(b2 nobatch dnow b3_500 b3_0)

CFG=$ROOT/triton/models/yolov8s_dyn/config.pbtxt
# The config is tracked in git, so git is the restore source - not a temp copy
# that the restore itself deletes. Refuse to start over uncommitted edits, which
# the restore would otherwise silently discard.
if ! git -C "$ROOT" diff --quiet -- triton/models/yolov8s_dyn/config.pbtxt; then
  echo "refusing: triton/models/yolov8s_dyn/config.pbtxt has uncommitted changes" >&2; exit 1
fi
ready() { for i in $(seq 1 60); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v2/health/ready)" = 200 ] && return 0
  sleep 2; done; return 1; }
RESTORED=0
restore() {
  [ "$RESTORED" = 1 ] && return 0; RESTORED=1
  echo "--- restoring yolov8s_dyn config from git and releasing the GPU lock ---"
  git -C "$ROOT" checkout -- triton/models/yolov8s_dyn/config.pbtxt
  docker restart "$C" >/dev/null 2>&1; ready || echo "WARNING: server not ready after restore"
  bash "$ROOT/scripts/gpu_lock.sh" release bench || true
}
# A signal handler must EXIT. An earlier version ran restore on TERM and then
# carried on - sweeping unlocked, with its backup already deleted.
trap restore EXIT
trap 'echo "--- interrupted ---"; exit 130' INT TERM

write_dyn() {  # $1 = arm
  local DB
  case $1 in
    nobatch) DB="" ;;
    dnow)    DB='dynamic_batching { preferred_batch_size: [ 4, 8 ] max_queue_delay_microseconds: 5000 }' ;;
    b3_500)  DB='dynamic_batching { preferred_batch_size: [ 2, 4, 8 ] max_queue_delay_microseconds: 500 }' ;;
    b3_0)    DB='dynamic_batching { max_queue_delay_microseconds: 0 }' ;;
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

met() { curl -s localhost:8002/metrics | awk -v m="$1" -v mod="$2" '$0 ~ "^"m"\\{model=\""mod"\"" {print $2}'; }
foreign() {  # GPU processes that are not this experiment
  nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
    | awk -F/ '{print $NF}' | grep -v -E '^(tritonserver|trt_grpc_cuda)$' | sort -u | paste -sd, -
}

mkdir -p "$(dirname "$OUT")"
printf 'arm\tmodel\trepeat\tphase\tcameras\toffered_fps\tfps\tp50_ms\tp95_ms\tp99_ms\tmax_ms\tmean_ms\tlate_frames\tframes\tmean_batch\tqueue_us_per_req\tforeign_gpu\n' > "$OUT"

current=""
for r in $(seq 1 "$REPEATS"); do
  n=${#ARMS[@]}; order=()
  for i in $(seq 0 $((n-1))); do order+=("${ARMS[$(( (i + r - 1) % n ))]}"); done
  echo "=== repeat $r: ${order[*]} ==="
  for arm in "${order[@]}"; do
    if [ "$arm" = b2 ]; then MODEL=yolov8s; else MODEL=yolov8s_dyn
      if [ "$current" != "$arm" ]; then
        write_dyn "$arm"; docker restart "$C" >/dev/null; ready || { echo "server down for $arm" >&2; continue; }
        current=$arm; sleep 3
      fi
    fi
    # discarded warm run so the first measured cell does not pay for a cold model
    docker exec "$C" bash -lc "cd /work/cpp/build && ./trt_grpc_cuda --mode paced --fps $FPS --model $MODEL --file frames.bin --streams 4 --duration 3 --warmup 3" >/dev/null 2>&1
    for phase in $PHASES; do
      for N in $CAMS; do
        sleep "$SETTLE"
        c0=$(met nv_inference_count $MODEL); e0=$(met nv_inference_exec_count $MODEL)
        q0=$(met nv_inference_queue_duration_us $MODEL); s0=$(met nv_inference_request_success $MODEL)
        f0=$(foreign)
        J=$(docker exec "$C" bash -lc "cd /work/cpp/build && ./trt_grpc_cuda --mode paced --fps $FPS --phase $phase --seed $r --warmup $WARMUP --model $MODEL --file frames.bin --streams $N --duration $DURATION" 2>/dev/null)
        f1=$(foreign)
        c1=$(met nv_inference_count $MODEL); e1=$(met nv_inference_exec_count $MODEL)
        q1=$(met nv_inference_queue_duration_us $MODEL); s1=$(met nv_inference_request_success $MODEL)
        [ -z "$J" ] && { echo "  $arm $phase N=$N FAILED" >&2; continue; }
        FG=$(printf '%s,%s' "$f0" "$f1" | tr ',' '\n' | grep -v '^$' | sort -u | paste -sd, -)
        # JSON goes via the environment: stdin is taken by the heredoc below
        B3_JSON="$J" python3 - "$arm" "$MODEL" "$r" "$phase" "$N" "${c0:-0}" "${c1:-0}" "${e0:-0}" "${e1:-0}" "${q0:-0}" "${q1:-0}" "${s0:-0}" "${s1:-0}" "${FG:-}" "$OUT" <<'PY'
import json, os, sys
(arm, model, r, phase, N, c0, c1, e0, e1, q0, q1, s0, s1, fg, out) = sys.argv[1:16]
d = json.loads(os.environ["B3_JSON"])
cnt, ex = float(c1) - float(c0), float(e1) - float(e0)
req, que = float(s1) - float(s0), float(q1) - float(q0)
batch = cnt / ex if ex else 0.0
qreq = que / req if req else 0.0
row = [arm, model, r, phase, N, "%.0f" % d["offered_fps"], "%.1f" % d["fps"],
       "%.3f" % d["lat_ms_p50"], "%.3f" % d["lat_ms_p95"], "%.3f" % d["lat_ms_p99"],
       "%.3f" % d["lat_ms_max"], "%.3f" % d["lat_ms_mean"], str(d["late_frames"]),
       str(d["frames"]), "%.2f" % batch, "%.0f" % qreq, fg]
open(out, "a").write("\t".join(row) + "\n")
print("  %-8s %-6s N=%-3s offered=%5.0f  p50 %7.3f  p99 %8.3f  max %8.3f ms  batch %.2f  late %s%s"
      % (arm, phase, N, d["offered_fps"], d["lat_ms_p50"], d["lat_ms_p99"], d["lat_ms_max"],
         batch, d["late_frames"], ("  FOREIGN:" + fg) if fg else ""))
PY
      done
    done
  done
done
echo "wrote $OUT"
