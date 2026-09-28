#!/usr/bin/env bash
# Part B, item 3: how many camera streams can the GPU's hardware decoder take?
#
# Every flow here reports frames per second. A deployment counts cameras, and
# before the detector sees a frame, NVDEC has to decode it. This measures the
# decoder alone: N concurrent ffmpeg sessions decoding on NVDEC
# (-hwaccel cuda, frames kept on the GPU, no display, no inference), as fast as
# they can, and reports the aggregate decode rate and what it means in 30 fps
# cameras.
#
# Sources: videos/real.mp4 (the study's own, H.264 640x360, 300 frames) and a
# 1920x1080 H.264 version of it made once with NVENC (videos/real_1080p.mp4,
# gitignored), because deployed cameras are rarely 360p. Each session decodes
# its source looped 40 times (12,300 frames). Sessions N = 1, 2, 4, 8, 16, 32;
# 3 repeats, interleaved. nvidia-smi's decoder utilisation is sampled each run.
#
# PREDICTIONS (written 2026-09-28, before any of this ran)
#   P1  There is no session limit: 32 concurrent NVDEC sessions all run (the
#       consumer-card session cap applies to NVENC, the encoder, not NVDEC).
#   P2  At 1080p the decoder saturates at 700-1,200 fps aggregate, i.e. 25-40
#       cameras at 30 fps, with decoder utilisation near 100%.
#   P3  At 640x360 the aggregate is several times higher (>= 3,000 fps), and it is
#       limited by the ffmpeg processes (CPU), not by NVDEC: decoder utilisation
#       stays well below 100% at saturation.
#   P4  So at 1080p the decoder, not the detector, caps a single GPU: A2 alone
#       processes ~1200 fps of yolov8s at 640, more than NVDEC can feed at 1080p.
#
# DECODER PATH. The first attempt used ffmpeg's generic `-hwaccel cuda`. In this
# container that path fails to create a decoder (cuvidCreateDecoder:
# CUDA_ERROR_INVALID_VALUE) and ffmpeg silently falls back to CPU decoding, still
# exiting 0: decoder utilisation stayed at 0% while "decoding" 12,800 fps. The
# harness now forces the CUVID hardware decoder (-c:v h264_cuvid, frames kept on
# the GPU), which works (100% decoder utilisation, no errors), and marks any run
# whose decoder utilisation never passes 50% as invalid rather than recording it.
# SHARED-HOST MODE (BUSY_OK=1), added 2026-09-28. The host was busy for hours
# with an unrelated job. This run is decoder-bound (NVDEC at 100%), so it may not
# need a quiet host, but that is checked, not assumed:
#   - every ffmpeg runs under `taskset -c $PIN` (default 10-15): the test can
#     never use more than 6 cores, and each row records the cores its container
#     actually used (from the container's cgroup cpu.stat);
#   - VALIDITY RULE, written before the run: the two 4-session points must land
#     within 5% of their quiet-host values (640x360: 2,462 fps; 1080p: 761 fps,
#     measured during the diagnosis). If either misses, the whole run is
#     excluded and the sweep waits for a quiet host instead.
# Output: results/v3/nvdec_capacity.tsv. Caller holds the lock.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/v3/nvdec_capacity.tsv"
C=triton-server
LOOPS=40
FRAMES_PER_SESSION=$((300 * (LOOPS + 1)))

# the 1080p source, made once: scaled on the CPU (this ffmpeg build has no
# scale_cuda filter), encoded on NVENC. A 0-byte file from a failed attempt is
# replaced, not reused.
if ! docker exec $C test -s /work/videos/real_1080p.mp4; then
  docker exec $C ffmpeg -hide_banner -loglevel error -y -i /work/videos/real.mp4 \
    -vf scale=1920:1080 -c:v h264_nvenc -b:v 8M -g 30 /work/videos/real_1080p.mp4
fi

BUSY_OK=${BUSY_OK:-0}
PIN=${PIN:-}; [ "$BUSY_OK" = 1 ] && PIN=${PIN:-10-15}
TS=""; [ -n "$PIN" ] && TS="taskset -c $PIN"
CG=/sys/fs/cgroup/system.slice/docker-$(docker inspect -f '{{.Id}}' $C).scope/cpu.stat
cg_usec() { awk '$1=="usage_usec"{print $2}' "$CG" 2>/dev/null || echo 0; }
[ -f "$OUT" ] || printf "source\tsessions\trep\twall_s\tagg_fps\tcameras_30fps\tfailed\tdec_util_max\tvalid\tload_start\tload_end\tcores_used\tpinned\n" > "$OUT"
# Every run waits for a quiet host (the same bar as leakfix_check.sh: 1-minute
# load under 3) and records the load it started and ended under: an earlier
# attempt was overrun mid-sweep by an unrelated job pushing the load to 82.
wait_quiet() {
  [ "$BUSY_OK" = 1 ] && return 0
  until awk -v l="$(cut -d' ' -f1 /proc/loadavg)" 'BEGIN{exit !(l < 3.0)}'; do sleep 30; done
}
for REP in 1 2 3; do
  ORDER=$(for s in real.mp4 real_1080p.mp4; do for n in 1 2 4 8 16 32; do echo "$s:$n"; done; done |
          python3 -c "import random,sys; l=sys.stdin.read().split(); random.Random($REP).shuffle(l); print(' '.join(l))")
  for c in $ORDER; do
    src=${c%%:*}; n=${c##*:}
    wait_quiet
    L0=$(cut -d' ' -f1 /proc/loadavg)
    # sample decoder utilisation in the background while the sessions run
    nvidia-smi --query-gpu=utilization.decoder --format=csv,noheader,nounits -lms 200 > /tmp/nvdec_util.$$ &
    SMI=$!
    u0=$(cg_usec)
    t0=$(python3 -c "import time; print(time.time())")
    failed=$(docker exec $C bash -c "
      fail=0
      for i in \$(seq 1 $n); do
        $TS ffmpeg -hide_banner -loglevel error -nostats -hwaccel cuda -hwaccel_output_format cuda \
          -c:v h264_cuvid -stream_loop $LOOPS -i /work/videos/$src -f null - &
      done
      for p in \$(jobs -p); do wait \$p || fail=\$((fail+1)); done
      echo \$fail")
    t1=$(python3 -c "import time; print(time.time())")
    u1=$(cg_usec)
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    wall=$(python3 -c "print($t1 - $t0)")
    umax=$(sort -n /tmp/nvdec_util.$$ | tail -1); rm -f /tmp/nvdec_util.$$
    python3 -c "
w=float('$wall'); n=int('$n'); f=int('$failed')
agg=(n-f)*$FRAMES_PER_SESSION/w
u=int('${umax:-0}' or 0)
print('\t'.join(['$src', str(n), '$REP', '%.2f'%w, '%.0f'%agg, '%.1f'%(agg/30), str(f), str(u),
                 'yes' if (u > 50 and f == 0) else 'NO', '$L0', '$(cut -d' ' -f1 /proc/loadavg)',
                 '%.2f' % (($u1 - $u0) / 1e6 / w), '${PIN:-none}']))" >> "$OUT"
    tail -1 "$OUT"
    sleep 3
  done
done
