#!/usr/bin/env python3
"""Home-page chart for live cameras: which pipeline is fastest depends on the load.

One panel, the four flows a reader chooses between, unsynchronised cameras,
median per-frame turnaround from each frame's due time (paced mode). The full
2x2 view with bursts, p99 and both B3 windows is docs/img/selection-paced.png on
the live-traffic page; this is its one-glance summary for the README.

Data: results/v3/b3_paced.tsv (B2, B3 0 us, D config) and a2_paced.tsv (A2),
median of 3 seeded repeats. Colours match every other figure in the repo.

Usage: python3 scripts/plot_live_headline.py [repo_root]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
R = os.environ.get("RESULTS_DIR", os.path.join(REPO, "results", "v3"))
OUT = os.path.join(REPO, "docs", "img", "live-cameras-headline.png")

ARMS = ["a2", "b2", "b3_0", "dnow"]
LABEL = {"a2": "A2 · in-process C++, one frame at a time",
         "b2": "B2 · Triton, one frame at a time",
         "b3_0": "B3 · Triton, batching with no wait",
         "dnow": "D's settings · batching, 5 ms wait"}
SHORT = {"a2": "A2", "b2": "B2", "b3_0": "B3", "dnow": "D settings"}
COLOR = {"a2": "#e8710a", "b2": "#1a73e8", "b3_0": "#12a150", "dnow": "#a142f4"}
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"

cells = defaultdict(list)
for fn in ("b3_paced.tsv", "a2_paced.tsv"):
    for r in csv.DictReader(open(os.path.join(R, fn)), delimiter="\t"):
        if r["phase"] == "random":
            cells[(r["arm"], int(r["cameras"]))].append(float(r["p50_ms"]))
cams = sorted({c for (a, c) in cells if a in ARMS})
med = {k: st.median(v) for k, v in cells.items()}
xs = {c: i for i, c in enumerate(cams)}

fig, ax = plt.subplots(figsize=(10.5, 5.6))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(MUTED)

# where one-frame-at-a-time stops keeping up, and where nothing does
i48, i56 = xs[48], xs[56]
ax.axvspan(i48 - 0.5, i56 - 0.5, color="#f1f1f1", zorder=0)
ax.axvspan(i56 - 0.5, i56 + 0.5, color="#e4e4e4", zorder=0)
ax.text(i48, 1.02, "past one-frame-at-a-time\ncapacity: only batching\nkeeps up", ha="center",
        va="bottom", fontsize=8.5, color=MUTED)
ax.text(i56, 1.02, "nothing\nkeeps up", ha="center", va="bottom", fontsize=8.5, color=MUTED)

ax.axhline(33.3, ls=":", lw=1.2, color="#c5221f", zorder=1)
ax.text(-0.45, 38, "33 ms: a new frame arrives - fall behind this and you never catch up",
        fontsize=8.5, color="#c5221f", va="bottom")

for a in ARMS:
    pts = [(xs[c], med[(a, c)]) for c in cams if (a, c) in med]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", lw=2, ms=6, color=COLOR[a],
            label=LABEL[a], zorder=3, markeredgecolor="white", markeredgewidth=1.2)
    x0, y0 = pts[0]                                  # direct label at the 1-camera end
    ax.text(x0 - 0.12, y0, SHORT[a], ha="right", va="center", fontsize=9, color=INK)

# ring the fastest at each load
for c in cams:
    vals = [(med[(a, c)], a) for a in ARMS if (a, c) in med]
    v, a = min(vals)
    if c < 56:
        ax.plot([xs[c]], [v], "o", ms=15, mfc="none", mec=COLOR[a], mew=2, zorder=4)

ax.set_yscale("log")
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_ylim(0.8, 6000)
ax.set_xlim(-0.9, len(cams) - 0.5)
ax.set_xticks(range(len(cams)))
ax.set_xticklabels(["%d\n%d fps" % (c, 30 * c) for c in cams], fontsize=9, color=INK)
ax.set_xlabel("30 fps cameras, unsynchronised (total frames per second)", color=INK)
ax.set_ylabel("median time per frame, ms (log scale)", color=INK)
ax.set_yticks([1, 2, 5, 10, 20, 50, 100, 1000])
ax.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "1,000"], color=INK)
ax.grid(axis="y", color=GRID, lw=0.8, which="major")
ax.tick_params(colors=MUTED)
ax.legend(loc="upper left", fontsize=8.8, frameon=False, bbox_to_anchor=(0.0, 0.93))
ax.set_title("Live cameras: the fastest pipeline depends on the load  (ringed: fastest at each load)",
             fontsize=11.5, color=INK, loc="left", pad=12)
fig.text(0.01, 0.005, "YOLOv8s FP16, RTX 3090. Paced mode: time from each frame's due time to its answer, "
         "upload included. Median of 3 seeded repeats.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 1))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140, facecolor="white")
for c in cams:
    print("%3d cams  " % c + "  ".join("%s %8.2f" % (a, med[(a, c)]) for a in ARMS if (a, c) in med))
print("wrote", os.path.relpath(OUT, REPO))
