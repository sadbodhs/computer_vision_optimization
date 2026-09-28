#!/usr/bin/env bash
# Part B, item 1: is D's throughput lead Triton, or batching?
#
# The published comparison puts D (Triton async + dynamic batching, batch-8
# engine) at 1665 fps against A2's plateau of ~1205 fps (docs/batching.md:
# "+38% vs A2"), and the README recommends D for maximum throughput. But A2 ran
# at batch 1 (cpp/src/main_cuda.cu was hardcoded to it) and D at batch 8, and
# docs/results.md already says batching "is the entire reason its fps is
# higher". This gives A2 the same batch-8 engine (--batch 8, capacity mode) and
# measures both in one session, so the comparison is batching-for-batching.
#
# Arms (capacity mode, frames.bin, 10 s each, 3 repeats, interleaved, 5 s pause):
#   A2 batch 1  yolov8s engine          streams 1, 2, 4   (the published A2)
#   A2 batch 8  yolov8s_dyn engine      streams 1, 2, 4   (new: --batch 8)
#   D           trt_grpc_async, yolov8s_dyn, 8 and 16 in flight (the published D)
#
# PREDICTIONS (written 2026-09-28, before any batch-8 A2 measurement; a 3-second
# functional check confirmed it runs and finds the same detections per frame)
#   P1  A2 at batch 8 reaches >= 1550 fps at 2-4 streams, within 5% of D's peak
#       measured in the same session (published 1665). At 1 stream it is lower
#       (~1000-1250 fps), because one stream serialises the upload of 8 frames
#       with the inference.
#   P2  So D's throughput lead over A2 is batching, not Triton: batch-8 A2 ties
#       or beats D. If it holds, the README's "maximum throughput -> D" row and
#       batching.md's "+38% vs A2" are corrected to say so.
#   P3  A2 at batch 1 reproduces its published plateau (~1205 fps) within 5%.
#
# Output: results/v3/batch8_a2_vs_d.tsv (one row per run). Caller holds the lock.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/v3/batch8_a2_vs_d.tsv"
C=triton-server
BIN=/work/cpp/build
DUR=10

run() {  # arm streams -> one TSV row
  local arm=$1 s=$2 cmd
  case $arm in
    a2_b1) cmd="$BIN/trt_pipeline_cuda --engine /models/yolov8s/1/model.plan --mode file --file frames.bin --streams $s --duration $DUR" ;;
    a2_b8) cmd="$BIN/trt_pipeline_cuda --engine /models/yolov8s_dyn/1/model.plan --batch 8 --mode file --file frames.bin --streams $s --duration $DUR" ;;
    d)     cmd="$BIN/trt_grpc_async --model yolov8s_dyn --file frames.bin --streams $s --duration $DUR" ;;
  esac
  local j
  j=$(docker exec $C bash -c "cd $BIN && $cmd" 2>/dev/null | grep '^{' | tail -1)
  python3 - "$arm" "$s" "$REP" "$j" >> "$OUT" <<'PY'
import json, sys
arm, s, rep, j = sys.argv[1:5]
try:
    r = json.loads(j)
    print("\t".join([arm, s, rep, "%.1f" % r["fps"], "%.3f" % r.get("lat_ms_p50", 0),
                     "%.3f" % r.get("lat_ms_p99", 0), str(r.get("frames", 0))]))
except Exception as e:
    print("\t".join([arm, s, rep, "ERROR", "", "", str(e)[:80]]))
PY
}

[ -f "$OUT" ] || printf "arm\tstreams\trep\tfps\tlat_ms_p50\tlat_ms_p99\tframes\n" > "$OUT"
CONFIGS="a2_b1:1 a2_b1:2 a2_b1:4 a2_b8:1 a2_b8:2 a2_b8:4 d:8 d:16"
for REP in 1 2 3; do
  # a different, fixed order each repeat, so no arm always runs first or last
  ORDER=$(echo $CONFIGS | tr ' ' '\n' | python3 -c "import random,sys; l=sys.stdin.read().split(); random.Random($REP).shuffle(l); print(' '.join(l))")
  for c in $ORDER; do
    run "${c%%:*}" "${c##*:}"
    tail -1 "$OUT"
    sleep 5   # let queues drain: no run inherits the previous one's backlog
  done
done
