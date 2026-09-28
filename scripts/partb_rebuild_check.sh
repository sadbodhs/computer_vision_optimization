#!/usr/bin/env bash
# Part B, item 2: rebuild every engine from the repo, and check nothing moved.
#
# scripts/export_models.sh is the documented way to regenerate the engines, and
# since Part A it also builds yolov8s_u8 (fold_norm.py + a UINT8 input binding),
# which had only ever been built by hand. It has never been run since. This:
#   1. backs up every current engine (outside the model repository),
#   2. runs export_models.sh, which now works in the export container's /tmp,
#      so a crash can no longer leave a folder inside triton/models,
#   3. times the rebuilt yolov8s_u8 with trtexec, against the published engine,
#   4. re-measures A2 (batch 1, streams 1/2/4, 3 repeats) on the rebuilt yolov8s
#      engine, against item 1's A2 batch-1 runs on the old engine, same session.
#
# PREDICTIONS (written 2026-09-28, before the rebuild)
#   R1  export_models.sh completes and writes all 7 engines, yolov8s_u8
#       included: its build is reproducible from the repo for the first time.
#   R2  The rebuilt u8 engine's GPU compute is within 3% of the published
#       0.9750 ms (results/v3/fewer_bytes.tsv, batch 1).
#   R3  A2 on the rebuilt yolov8s engine is within 3% of the old engine. The
#       ultralytics release that exports the ONNX is newer than the one behind
#       the published engines; the inspection study found an export path alone
#       can move an engine by 9%, so a larger shift here would be that effect.
#
# Output: results/v3/partb_rebuild.tsv. Caller holds the lock.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/v3/partb_rebuild.tsv"
C=triton-server
BIN=/work/cpp/build
BACKUP=/home/suchi/sadbodh/model_exports/engine_backup_$(date +%Y%m%d_%H%M)

mkdir -p "$BACKUP"
for d in "$ROOT"/triton/models/*/1/model.plan; do
  m=$(basename "$(dirname "$(dirname "$d")")")
  mkdir -p "$BACKUP/$m/1" && cp -p "$d" "$BACKUP/$m/1/model.plan"
done
echo "engines backed up to $BACKUP"

STAMP="$BACKUP/.started"; touch "$STAMP"
"$ROOT/scripts/export_models.sh" 2>&1 | tail -20
export_rc=${PIPESTATUS[0]}
# Measure nothing unless every engine really was rebuilt: the first attempt's
# export failed and the timings below then ran on the OLD engines.
stale=0
for m in yolov8n yolov8s yolo11n yolov8n_dyn yolov8s_dyn yolo11n_dyn yolov8s_u8; do
  p="$ROOT/triton/models/$m/1/model.plan"
  if [ ! "$p" -nt "$STAMP" ]; then echo "NOT REBUILT: $m"; stale=1; fi
done
if [ "$export_rc" -ne 0 ] || [ "$stale" -ne 0 ]; then
  echo "export failed (rc=$export_rc) or left engines unrebuilt: nothing measured" >&2
  exit 1
fi
ls -la "$ROOT"/triton/models/{yolov8n,yolov8s,yolo11n,yolov8n_dyn,yolov8s_dyn,yolo11n_dyn,yolov8s_u8}/1/model.plan

printf "what\tarm\tstreams\trep\tvalue\tunit\n" > "$OUT"
# R2: trtexec on the rebuilt u8 engine, 3 runs
for rep in 1 2 3; do
  g=$(docker exec $C /usr/src/tensorrt/bin/trtexec --loadEngine=/models/yolov8s_u8/1/model.plan 2>&1 |
      grep -oP "GPU Compute Time:.*?mean = \K[0-9.]+")
  printf "u8_gpu_compute\trebuilt\t1\t%s\t%s\tms\n" "$rep" "$g" >> "$OUT"; tail -1 "$OUT"
done
# R3: A2 batch 1 on the rebuilt engine
for rep in 1 2 3; do
  for s in 1 2 4; do
    f=$(docker exec $C bash -c "cd $BIN && ./trt_pipeline_cuda --engine /models/yolov8s/1/model.plan --mode file --file frames.bin --streams $s --duration 10" 2>/dev/null |
        grep '^{' | tail -1 | python3 -c "import json,sys; print('%.1f' % json.load(sys.stdin)['fps'])")
    printf "a2_fps\trebuilt_engine\t%s\t%s\t%s\tfps\n" "$s" "$rep" "$f" >> "$OUT"; tail -1 "$OUT"
    sleep 5
  done
done
