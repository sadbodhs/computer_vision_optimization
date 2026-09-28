#!/bin/bash
# Safety net run before every client run in the paced sweeps: unregister any
# CUDA shared-memory region still registered with Triton.
#
# The clients unregister their own regions on exit, but a client that crashes
# or is killed does not, and Triton then keeps that run's GPU buffers mapped
# (CUDA IPC) for the rest of its life - conditions that drift run by run. A
# non-zero count here is therefore worth a line in the log: it means the
# previous run did not clean up after itself.
#
# Usage: scripts/shm_clear.sh [http_port]   (default 8000)
PORT="${1:-8000}"
n=$(curl -s "localhost:$PORT/v2/cudasharedmemory/status" \
      | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "?")
if [ "$n" != "0" ]; then
  echo "shm_clear: $n region(s) still registered; unregistering" >&2
  curl -s -X POST "localhost:$PORT/v2/cudasharedmemory/unregister" >/dev/null
fi
