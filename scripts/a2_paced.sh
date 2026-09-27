#!/usr/bin/env bash
# A2 in paced mode - the same open-loop virtual cameras as scripts/b3_paced.sh,
# driven through the in-process C++ pipeline instead of Triton.
#
# This runs in a separate sweep from b3_paced.tsv, so comparability has to be
# shown, not assumed: each repeat also re-runs a few B2 cells ("b2_anchor"). If
# they reproduce the B2 rows of b3_paced.tsv, A2 can be read against that file.
#
# Same cameras, same seeded phases (--seed = repeat index), same warmup, same
# turnaround definition (from each frame's due time, upload included).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=${1:-$ROOT/results/v3/a2_paced.tsv}
C=${CONTAINER:-triton-server}
REPEATS=${REPEATS:-3}
DURATION=${DURATION:-13}
WARMUP=${WARMUP:-1}
FPS=${FPS:-30}
SETTLE=${SETTLE:-3}
CAMS=${CAMS:-"1 4 8 16 32 48 56"}
ANCHOR_CAMS=${ANCHOR_CAMS:-"1 8 32"}
PHASES=${PHASES:-"random sync"}
ENGINE=/models/yolov8s/1/model.plan

cleanup() { bash "$ROOT/scripts/gpu_lock.sh" release bench || true; }
trap cleanup EXIT
trap 'echo "--- interrupted ---"; exit 130' INT TERM

met() { curl -s localhost:8002/metrics | awk -v m="$1" -v mod="$2" '$0 ~ "^"m"\\{model=\""mod"\"" {print $2}'; }
foreign() {
  nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null \
    | awk -F/ '{print $NF}' | grep -v -E '^(tritonserver|trt_grpc_cuda|trt_pipeline_cuda)$' | sort -u | paste -sd, -
}

mkdir -p "$(dirname "$OUT")"
printf 'arm\tmodel\trepeat\tphase\tcameras\toffered_fps\tfps\tp50_ms\tp95_ms\tp99_ms\tmax_ms\tmean_ms\tlate_frames\tframes\tmean_batch\tqueue_us_per_req\tforeign_gpu\n' > "$OUT"

cell() {  # $1 arm  $2 repeat  $3 phase  $4 cameras
  local arm=$1 r=$2 phase=$3 N=$4 J c0 e0 c1 e1 f0 f1 FG
  sleep "$SETTLE"
  f0=$(foreign)
  if [ "$arm" = a2 ]; then
    J=$(docker exec "$C" bash -lc "cd /work/cpp/build && ./trt_pipeline_cuda --engine $ENGINE --mode paced --fps $FPS --phase $phase --seed $r --warmup $WARMUP --file frames.bin --streams $N --duration $DURATION" 2>/dev/null)
    c0=0; c1=0; e0=0; e1=0
  else
    c0=$(met nv_inference_count yolov8s); e0=$(met nv_inference_exec_count yolov8s)
    J=$(docker exec "$C" bash -lc "cd /work/cpp/build && ./trt_grpc_cuda --mode paced --fps $FPS --phase $phase --seed $r --warmup $WARMUP --model yolov8s --file frames.bin --streams $N --duration $DURATION" 2>/dev/null)
    c1=$(met nv_inference_count yolov8s); e1=$(met nv_inference_exec_count yolov8s)
  fi
  f1=$(foreign)
  [ -z "$J" ] && { echo "  $arm $phase N=$N FAILED" >&2; return; }
  FG=$(printf '%s,%s' "$f0" "$f1" | tr ',' '\n' | grep -v '^$' | sort -u | paste -sd, -)
  CELL_JSON="$J" python3 - "$arm" "$r" "$phase" "$N" "${c0:-0}" "${c1:-0}" "${e0:-0}" "${e1:-0}" "${FG:-}" "$OUT" <<'PY'
import json, os, sys
arm, r, phase, N, c0, c1, e0, e1, fg, out = sys.argv[1:11]
d = json.loads(os.environ["CELL_JSON"])
ex = float(e1) - float(e0)
batch = (float(c1) - float(c0)) / ex if ex else 1.0     # A2 never batches: 1.00
model = "yolov8s (in-process engine)" if arm == "a2" else "yolov8s"
row = [arm, model, r, phase, N, "%.0f" % d["offered_fps"], "%.1f" % d["fps"],
       "%.3f" % d["lat_ms_p50"], "%.3f" % d["lat_ms_p95"], "%.3f" % d["lat_ms_p99"],
       "%.3f" % d["lat_ms_max"], "%.3f" % d["lat_ms_mean"], str(d["late_frames"]),
       str(d["frames"]), "%.2f" % batch, "0", fg]
open(out, "a").write("\t".join(row) + "\n")
print("  %-9s %-6s N=%-3s offered=%5.0f  p50 %8.3f  p99 %9.3f  max %9.3f ms  late %s%s"
      % (arm, phase, N, d["offered_fps"], d["lat_ms_p50"], d["lat_ms_p99"], d["lat_ms_max"],
         d["late_frames"], ("  FOREIGN:" + fg) if fg else ""))
PY
}

for r in $(seq 1 "$REPEATS"); do
  echo "=== repeat $r ==="
  for phase in $PHASES; do
    # alternate which goes first so neither side always runs on a warmer card
    if [ $((r % 2)) -eq 1 ]; then
      for N in $CAMS; do cell a2 "$r" "$phase" "$N"; done
      for N in $ANCHOR_CAMS; do cell b2_anchor "$r" "$phase" "$N"; done
    else
      for N in $ANCHOR_CAMS; do cell b2_anchor "$r" "$phase" "$N"; done
      for N in $CAMS; do cell a2 "$r" "$phase" "$N"; done
    fi
  done
done
echo "wrote $OUT"
