#!/usr/bin/env python3
"""Summary + figure for B3: per-frame turnaround of live, camera-paced traffic.

Reads results/v3/b3_paced.tsv (scripts/b3_paced.sh). Each cell is repeated;
the summary takes the MEDIAN across repeats of each statistic, so one noisy
repeat cannot carry a conclusion. Turnaround is measured from each frame's due
time (open loop), so an overloaded arm shows its queue instead of hiding it.

Usage: python3 scripts/plot_b3.py [repo_root]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
TSV = os.path.join(REPO, "results", "v3", "b3_paced.tsv")
OUT = os.path.join(REPO, "docs", "img", "b3-paced.png")

ARMS = ["b2", "nobatch", "dnow", "b3_500", "b3_0"]
LABEL = {
    "b2": "B2 — batch-1 engine, no batching",
    # Intended as "batching off" (no dynamic_batching block), but Triton's config
    # auto-complete adds dynamic_batching {preferred [8], 0 us} to any model with
    # max_batch_size > 0 - verified from /v2/models/yolov8s_dyn/config. It is
    # therefore a second, independent run of the 0 us config.
    "nobatch": "no scheduler set → Triton auto-enables 0 µs (replicates B3 · 0 µs)",
    "dnow": "D config — [4,8], 5 ms window",
    "b3_500": "B3 · 500 µs — [2,4,8]",
    "b3_0": "B3 · 0 µs",
}
COLOR = {"b2": "#1a73e8", "nobatch": "#9aa0a6", "dnow": "#a142f4",
         "b3_500": "#e8710a", "b3_0": "#12a150"}
B2_CEIL, D_CEIL = 1131, 1650   # published capacity-mode ceilings, fps

cells = defaultdict(list)
foreign = []
for r in csv.DictReader(open(TSV), delimiter="\t"):
    cells[(r["arm"], r["phase"], int(r["cameras"]))].append(r)
    if r.get("foreign_gpu"):
        foreign.append((r["arm"], r["phase"], r["cameras"], r["repeat"], r["foreign_gpu"]))


def med(rows, k):
    return st.median(float(x[k]) for x in rows)


cams = sorted({c for (_, _, c) in cells})
phases = [p for p in ("random", "sync") if any(k[1] == p for k in cells)]

for phase in phases:
    print("\n=== phase: %s — median across repeats (p50 / p99 ms, mean batch) ===" % phase)
    print("%-5s %7s " % ("cams", "offered") + " ".join("%-22s" % a for a in ARMS))
    for c in cams:
        line = "%-5d %7d " % (c, c * 30)
        for a in ARMS:
            rows = cells.get((a, phase, c))
            if not rows:
                line += "%-22s " % "-"
                continue
            line += "%-22s " % ("%7.2f /%8.2f  b%.2f" % (med(rows, "p50_ms"), med(rows, "p99_ms"),
                                                         med(rows, "mean_batch")))
        print(line)

if foreign:
    print("\nWARNING: rows overlapped with a foreign GPU process:")
    for f in foreign:
        print("  ", f)
else:
    print("\nno foreign GPU process overlapped any row")

# ---- predictions, checked mechanically
def p(arm, phase, c, k="p50_ms"):
    rows = cells.get((arm, phase, c))
    return med(rows, k) if rows else float("nan")

print("\n=== predictions ===")
for c in (1, 4, 8):
    print("P1  %d cams random: b3_500-b2 %+.2f ms | b3_0-b2 %+.2f ms | dnow-b2 %+.2f ms | nobatch-b2 %+.2f ms"
          % (c, p("b3_500", "random", c) - p("b2", "random", c),
             p("b3_0", "random", c) - p("b2", "random", c),
             p("dnow", "random", c) - p("b2", "random", c),
             p("nobatch", "random", c) - p("b2", "random", c)))
for c in (8, 16, 32):
    print("P2  %d cams sync p99: b2 %.2f | b3_500 %.2f | b3_0 %.2f   (mean: b2 %.2f | b3_500 %.2f)"
          % (c, p("b2", "sync", c, "p99_ms"), p("b3_500", "sync", c, "p99_ms"), p("b3_0", "sync", c, "p99_ms"),
             p("b2", "sync", c, "mean_ms"), p("b3_500", "sync", c, "mean_ms")))
for c in (48, 56):
    print("P3  %d cams random p50: b2 %.1f | nobatch %.1f | dnow %.1f | b3_500 %.1f | b3_0 %.1f ms"
          % (c, p("b2", "random", c), p("nobatch", "random", c), p("dnow", "random", c),
             p("b3_500", "random", c), p("b3_0", "random", c)))
print("P4  b3_0 mean batch (random): " + ", ".join(
    "%d cams %.2f" % (c, p("b3_0", "random", c, "mean_batch")) for c in cams))

# ---- figure: p99 turnaround vs offered load, one panel per phase
fig, axes = plt.subplots(1, len(phases), figsize=(6.4 * len(phases), 4.8), sharey=True)
if len(phases) == 1:
    axes = [axes]
for ax, phase in zip(axes, phases):
    for a in ARMS:
        xs, ys = [], []
        for c in cams:
            rows = cells.get((a, phase, c))
            if rows:
                xs.append(c * 30)
                ys.append(med(rows, "p99_ms"))
        if xs:
            ax.plot(xs, ys, "o--" if a == "nobatch" else "o-", lw=1.2 if a == "nobatch" else 2,
                    ms=3 if a == "nobatch" else 5, color=COLOR[a], label=LABEL[a])
    for x, lab in ((B2_CEIL, "B2 capacity"), (D_CEIL, "D capacity")):
        ax.axvline(x, ls="--", lw=1, color="#888")
        ax.annotate(lab, (x, 0.5), xycoords=("data", "axes fraction"), rotation=90,
                    fontsize=8, color="#666", ha="right", va="center")
    ax.axhline(33.3, ls=":", lw=1, color="#c00")
    ax.annotate("one frame at 30 fps (33.3 ms)", (30, 36), fontsize=8, color="#c00")
    ax.set_yscale("log")
    ax.set_xlabel("offered load (fps) — cameras × 30")
    ax.set_title("%s camera phases" % ("Unsynchronised" if phase == "random" else "Synchronised (burst)"),
                 fontsize=11)
    ax.grid(alpha=.25, which="both")
axes[0].set_ylabel("p99 turnaround (ms, log) — from frame due time")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Live camera traffic: per-frame turnaround by server configuration", fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("\nwrote", os.path.relpath(OUT, REPO))
