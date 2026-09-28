#!/usr/bin/env bash
# Part B runner: wait for a quiet host CPU, then run the three pre-registered
# steps under the GPU lock. Re-checks the CPU before each step and waits again
# if something else has started: capacity runs are CPU-sensitive.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# The same bar as leakfix_check.sh, whose published re-check used it: load
# average under 3 on this 16-core host.
quiet() {
  awk -v l="$(cut -d" " -f1 /proc/loadavg)" 'BEGIN{exit !(l < 3.0)}'
}
wait_quiet() {
  until quiet; do echo "$(date +%T) waiting for a quiet CPU (load $(cut -d' ' -f1 /proc/loadavg))"; sleep 60; done
  echo "$(date +%T) CPU quiet"
}
# Steps default to all three; pass script names to run a subset.
STEPS=("$@"); [ ${#STEPS[@]} -eq 0 ] && STEPS=(batch8_a2_vs_d.sh partb_rebuild_check.sh nvdec_capacity.sh)
for step in "${STEPS[@]}"; do
  wait_quiet
  "$ROOT/scripts/gpu_lock.sh" wait bench "Part B: $step"
  echo "=== $step ==="
  "$ROOT/scripts/$step"
  "$ROOT/scripts/gpu_lock.sh" release bench
done
echo "=== PART B DONE ==="
