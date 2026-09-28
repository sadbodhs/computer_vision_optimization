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
# Output: results/v3/nvdec_capacity.tsv. Caller holds the lock; needs a quiet CPU.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/v3/nvdec_capacity.tsv"
C=triton-server
LOOPS=40
FRAMES_PER_SESSION=$((300 * (LOOPS + 1)))

# the 1080p source, made once on NVENC (not the CPU)
if ! docker exec $C test -f /work/videos/real_1080p.mp4; then
  docker exec $C ffmpeg -hide_banner -loglevel error -y -hwaccel cuda -hwaccel_output_format cuda \
    -i /work/videos/real.mp4 -vf scale_cuda=1920:1080 -c:v h264_nvenc -b:v 8M -g 30 \
    /work/videos/real_1080p.mp4
fi

[ -f "$OUT" ] || printf "source\tsessions\trep\twall_s\tagg_fps\tcameras_30fps\tfailed\tdec_util_max\n" > "$OUT"
for REP in 1 2 3; do
  ORDER=$(for s in real.mp4 real_1080p.mp4; do for n in 1 2 4 8 16 32; do echo "$s:$n"; done; done |
          python3 -c "import random,sys; l=sys.stdin.read().split(); random.Random($REP).shuffle(l); print(' '.join(l))")
  for c in $ORDER; do
    src=${c%%:*}; n=${c##*:}
    # sample decoder utilisation in the background while the sessions run
    nvidia-smi --query-gpu=utilization.decoder --format=csv,noheader,nounits -lms 200 > /tmp/nvdec_util.$$ &
    SMI=$!
    t0=$(python3 -c "import time; print(time.time())")
    failed=$(docker exec $C bash -c "
      fail=0
      for i in \$(seq 1 $n); do
        ffmpeg -hide_banner -loglevel error -nostats -hwaccel cuda -hwaccel_output_format cuda \
          -stream_loop $LOOPS -i /work/videos/$src -f null - &
      done
      for p in \$(jobs -p); do wait \$p || fail=\$((fail+1)); done
      echo \$fail")
    t1=$(python3 -c "import time; print(time.time())")
    kill $SMI 2>/dev/null; wait $SMI 2>/dev/null
    wall=$(python3 -c "print($t1 - $t0)")
    umax=$(sort -n /tmp/nvdec_util.$$ | tail -1); rm -f /tmp/nvdec_util.$$
    python3 -c "
w=float('$wall'); n=int('$n'); f=int('$failed')
agg=(n-f)*$FRAMES_PER_SESSION/w
print('\t'.join(['$src', str(n), '$REP', '%.2f'%w, '%.0f'%agg, '%.1f'%(agg/30), str(f), '${umax:-}']))" >> "$OUT"
    tail -1 "$OUT"
    sleep 3
  done
done
