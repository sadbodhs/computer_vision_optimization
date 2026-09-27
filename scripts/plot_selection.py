#!/usr/bin/env python3
"""Selection chart: which pipeline has the lowest per-frame turnaround at each load.

Combines results/v3/b3_paced.tsv (B2, D config, B3) with results/v3/a2_paced.tsv
(A2, plus B2 "anchor" cells re-run in the same sweep). The anchors are checked
first: A2 is only read against the B3 file if B2 reproduces across the two
sweeps. Every statistic is the median across repeats.

Two answers per load, because A2 has no server: the overall winner, and the
winner among the Triton configurations for anyone who needs one.

Usage: python3 scripts/plot_selection.py [repo_root]
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
R = os.path.join(REPO, "results", "v3")
OUT = os.path.join(REPO, "docs", "img", "selection-paced.png")

ARMS = ["a2", "b2", "b3_0", "b3_500", "dnow"]           # 'nobatch' is a b3_0 replicate
SERVER = ["b2", "b3_0", "b3_500", "dnow"]
LABEL = {"a2": "A2 — in-process C++, no server", "b2": "B2 — Triton, no batching",
         "b3_0": "B3 · 0 µs", "b3_500": "B3 · 500 µs", "dnow": "D config (5 ms window)"}
SHORT = {"a2": "A2", "b2": "B2", "b3_0": "B3·0µs", "b3_500": "B3·500µs", "dnow": "D-cfg"}
COLOR = {"a2": "#e8710a", "b2": "#1a73e8", "b3_0": "#12a150", "b3_500": "#f9ab00",
         "dnow": "#a142f4"}

cells = defaultdict(list)
for fn in ("b3_paced.tsv", "a2_paced.tsv"):
    for r in csv.DictReader(open(os.path.join(R, fn)), delimiter="\t"):
        cells[(r["arm"], r["phase"], int(r["cameras"]))].append(r)


def med(arm, phase, c, k):
    rows = cells.get((arm, phase, c))
    return st.median(float(x[k]) for x in rows) if rows else None


# ---- 1. comparability: B2 anchors (A2 sweep) vs B2 (B3 sweep)
print("=== B2 anchor check (same cells, two sweeps) ===")
worst = 0.0
for phase in ("random", "sync"):
    for c in (1, 8, 32):
        a, b = med("b2_anchor", phase, c, "p50_ms"), med("b2", phase, c, "p50_ms")
        if a is None or b is None:
            continue
        rel = abs(a - b) / b
        worst = max(worst, rel)
        print("  %-6s %2d cams  p50: B3-sweep %.3f  A2-sweep %.3f  (%+.1f%%)" % (phase, c, b, a, 100 * (a - b) / b))
print("  worst p50 disagreement: %.1f%%" % (100 * worst))

# ---- 2. selection table
cams = sorted({c for (arm, _, c) in cells if arm in ARMS})
print("\n=== lowest turnaround at each load (median of repeats) ===")
for phase in ("random", "sync"):
    print("\n-- %s --" % phase)
    print("%-5s %-7s | %-28s %-28s | %-28s" % ("cams", "fps", "best p50 (overall)", "best p99 (overall)",
                                                "best p99 with a server"))
    for c in cams:
        def best(k, pool):
            vals = [(med(a, phase, c, k), a) for a in pool if med(a, phase, c, k) is not None]
            vals.sort()
            if not vals:
                return "-"
            (v, a), rest = vals[0], vals[1:]
            margin = (" (+%.2f to %s)" % (rest[0][0] - v, SHORT[rest[0][1]])) if rest else ""
            return "%s %.2f%s" % (SHORT[a], v, margin)
        print("%-5d %-7d | %-28s %-28s | %-28s" % (c, c * 30, best("p50_ms", ARMS), best("p99_ms", ARMS),
                                                    best("p99_ms", SERVER)))

# ---- 3. figure: 2x2 (phase x metric), categorical camera axis, winner ringed
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), sharex=True)
xs = list(range(len(cams)))
for row, phase in enumerate(("random", "sync")):
    for col, (k, kname) in enumerate((("p50_ms", "median (p50)"), ("p99_ms", "worst 1% (p99)"))):
        ax = axes[row][col]
        for a in ARMS:
            ys = [med(a, phase, c, k) for c in cams]
            ax.plot([x for x, y in zip(xs, ys) if y is not None], [y for y in ys if y is not None],
                    "o-", lw=2 if a in ("a2", "b2", "b3_0") else 1.4, ms=5, color=COLOR[a], label=LABEL[a])
        for x, c in zip(xs, cams):                       # ring the winner at each load
            vals = [(med(a, phase, c, k), a) for a in ARMS if med(a, phase, c, k) is not None]
            if vals:
                v, a = min(vals)
                ax.plot([x], [v], "o", ms=15, mfc="none", mec=COLOR[a], mew=2.5)
        ax.axhline(33.3, ls=":", lw=1, color="#c00")
        ax.set_yscale("log")
        ax.set_title("%s cameras — %s" % ("Unsynchronised" if phase == "random" else "Synchronised (burst)",
                                         kname), fontsize=11)
        ax.grid(alpha=.25, which="both")
        if col == 0:
            ax.set_ylabel("turnaround (ms, log)")
        if row == 1:
            ax.set_xticks(xs)
            ax.set_xticklabels(["%d\n%d fps" % (c, c * 30) for c in cams], fontsize=9)
            ax.set_xlabel("cameras at 30 fps")
axes[0][0].legend(fontsize=8, loc="upper left")
axes[0][0].annotate("33.3 ms = one frame", (0, 36), fontsize=8, color="#c00")
fig.suptitle("Which pipeline gives the lowest per-frame turnaround? — circled: lowest at each load",
             fontsize=12)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("\nwrote", os.path.relpath(OUT, REPO))
