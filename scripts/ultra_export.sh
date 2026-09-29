#!/bin/bash
# ultralytics <name> -> ONNX at <imgsz>, for the model zoo (docs/model-zoo.md).
#
# Runs INSIDE the triton-server container, which has ultralytics. The export is
# done in a scratch directory and the ONNX lands in $OUT (default /tmp/ultra_exp),
# never under /models. Copy it out to the exports folder on the host with:
#   docker cp triton-server:/tmp/ultra_exp/<name>.onnx /home/suchi/sadbodh/model_exports/_exp/
#
# Why not /models: it is Triton's --model-repository, and Triton loads EVERY
# subdirectory of it as a model. This script used to write to /models/_exp; that
# folder has no version subdirectory, so the next server restart fails to load it
# and, with the default --exit-on-error, takes the whole server down - days later,
# for an unrelated reason. ultralytics also writes <name>.onnx next to the
# weights, which is why the weights are copied into the scratch directory first.
#
# Usage: scripts/ultra_export.sh <name> [imgsz]   (weights: /models/<name>.pt if
#        present, otherwise ultralytics downloads them into the scratch directory)
set -e
m="$1"; sz="${2:-640}"; OUT="${OUT:-/tmp/ultra_exp}"
[ -n "$m" ] || { echo "usage: $0 <name> [imgsz]" >&2; exit 1; }
case "$OUT" in /models|/models/*) echo "refusing: OUT=$OUT is inside the model repository" >&2; exit 1 ;; esac

W=$(mktemp -d /tmp/ultra_work.XXXXXX)
trap 'rm -rf "$W"' EXIT
[ -f "/models/$m.pt" ] && cp "/models/$m.pt" "$W/"
cd "$W"
yolo export model="$m.pt" format=onnx imgsz="$sz" dynamic=false simplify=True >"/tmp/${m}_exp.log" 2>&1 \
  || { echo "$m EXPORT_FAILED: $(grep -iE 'error|not supported' "/tmp/${m}_exp.log" | tail -1)"; exit 0; }
mkdir -p "$OUT"
mv -f "$W/$m.onnx" "$OUT/$m.onnx"
echo "$m exported at $sz -> $OUT/$m.onnx"
