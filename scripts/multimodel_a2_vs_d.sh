#!/usr/bin/env bash
# Is Triton's multi-model lead batching or the scheduler?
#
# After Part B's correction (batch-8 A2 ties D on one model), one claim was left
# marked untested: D serving THREE models at once (yolov8n, yolov8s, yolo11n,
# async, dynamic batching) at 1,799-1,816 fps, "+49-51% vs A2" and "Triton's
# scheduler has no hand-rolled equivalent". That compared it with A2 running
# yolov8s alone at batch 1 (1,205 fps). This builds the hand-rolled equivalent
# (trt_pipeline_cuda --engines a,b,c: one process, one CUDA context, streams
# dealt round-robin across the three models, no scheduler) and measures it
# against D in one session.
#
# Setup: triton-server is RESTARTED first (not recreated). It was last started
# before the Part B engine rebuild, so it was still serving the old engines from
# memory while A2 would load the rebuilt files; after the restart both sides run
# the same rebuilt engines. From here on the server's baseline is the rebuilt set.
#
# Arms (capacity mode, 10 s, 3 interleaved repeats, 5 s pause):
#   d3       trt_grpc_async --models yolov8n_dyn,yolov8s_dyn,yolo11n_dyn, 9 and 18 streams
#   a2m_b8   trt_pipeline_cuda --engines (the three _dyn engines) --batch 8, 3 and 6 streams
#   a2m_b1   trt_pipeline_cuda --engines (the three batch-1 engines), 3 and 6 streams
#
# PREDICTIONS (written 2026-09-29, before any multi-model A2 measurement)
#   M1  A2 at batch 8 on the three models (6 streams) comes within 5% of D's
#       multi-model peak in the same session, at a fraction of its latency: the
#       multi-model lead is batching too, not the scheduler. If A2 falls more than
#       5% short, the scheduler earns the credit and the page says so.
#   M2  A2 at batch 1 on the three models reaches >= 1,450 fps, above the 1,205
#       fps of yolov8s-only A2 the published "+49-51%" was measured against,
#       because two of the three models are lighter: that gap overstated even the
#       batch-1 difference.
#   M3  D multi-model reproduces its published 1,799-1,816 fps within 10%.
#
# Output: results/v3/multimodel_a2_vs_d.tsv. Caller holds the lock; quiet host.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/v3/multimodel_a2_vs_d.tsv"
C=triton-server
BIN=/work/cpp/build
DUR=10
M=/models
DYN="$M/yolov8n_dyn/1/model.plan,$M/yolov8s_dyn/1/model.plan,$M/yolo11n_dyn/1/model.plan"
FIX="$M/yolov8n/1/model.plan,$M/yolov8s/1/model.plan,$M/yolo11n/1/model.plan"

docker restart $C >/dev/null
for i in $(seq 1 90); do curl -sf localhost:8000/v2/health/ready >/dev/null && break; sleep 2; done
curl -sf localhost:8000/v2/health/ready >/dev/null || { echo "triton-server not ready after restart" >&2; exit 1; }
for m in yolov8n_dyn yolov8s_dyn yolo11n_dyn; do
  printf '%s %s\n' $m "$(curl -s -o /dev/null -w %{http_code} localhost:8000/v2/models/$m/ready)"
done
sleep 10

run() {  # arm streams
  local arm=$1 s=$2 cmd j
  case $arm in
    d3)     cmd="$BIN/trt_grpc_async --models yolov8n_dyn,yolov8s_dyn,yolo11n_dyn --file frames.bin --streams $s --duration $DUR" ;;
    a2m_b8) cmd="$BIN/trt_pipeline_cuda --engines $DYN --batch 8 --mode file --file frames.bin --streams $s --duration $DUR" ;;
    a2m_b1) cmd="$BIN/trt_pipeline_cuda --engines $FIX --mode file --file frames.bin --streams $s --duration $DUR" ;;
  esac
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
CONFIGS="d3:9 d3:18 a2m_b8:3 a2m_b8:6 a2m_b1:3 a2m_b1:6"
for REP in 1 2 3; do
  ORDER=$(echo $CONFIGS | tr ' ' '\n' | python3 -c "import random,sys; l=sys.stdin.read().split(); random.Random(10+$REP).shuffle(l); print(' '.join(l))")
  for c in $ORDER; do
    run "${c%%:*}" "${c##*:}"
    tail -1 "$OUT"
    sleep 5
  done
done
