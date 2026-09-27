#!/usr/bin/env python3
"""GPU utilisation behind the paced-mode results — the view that transfers.

Camera counts are specific to the RTX 3090. How busy the GPU was at a given
turnaround is not: read the right-hand panel with your own GPU's utilisation and
you know roughly where you sit on the curve, whatever the card.

nvidia-smi's utilization.gpu is the fraction of time ANY kernel was running, not
how full the SMs were — it reaches 100% before the GPU truly runs out of compute.
Read it as "how often the GPU is never idle", and treat the approach to 100% as
the warning, not 100% itself.

Usage: python3 scripts/plot_util.py [repo_root]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
# RESULTS_DIR lets another GPU's data sit beside the 3090's (see docs/other-gpus.md)
TSV = os.path.join(os.environ.get("RESULTS_DIR", os.path.join(REPO, "results", "v3")), "gpu_util_paced.tsv")
OUT = os.path.join(REPO, "docs", "img", "gpu-util-paced.png")

ARMS = ["a2", "b2", "b3_0", "b3_500", "dnow"]
LABEL = {"a2": "A2 — in-process C++", "b2": "B2 — Triton, no batching", "b3_0": "B3 · 0 µs",
         "b3_500": "B3 · 500 µs", "dnow": "D config (5 ms window)"}
COLOR = {"a2": "#e8710a", "b2": "#1a73e8", "b3_0": "#12a150", "b3_500": "#f9ab00", "dnow": "#a142f4"}

rows = defaultdict(dict)
for r in csv.DictReader(open(TSV), delimiter="\t"):
    rows[r["arm"]][int(r["cameras"])] = r
cams = sorted({c for a in rows for c in rows[a]})

print("%-8s " % "cameras" + " ".join("%-24s" % a for a in ARMS))
for c in cams:
    line = "%-8d " % c
    for a in ARMS:
        r = rows.get(a, {}).get(c)
        line += "%-24s " % (("util %5.1f%%  %4s MHz" % (float(r["util_mean_pct"]), r["sm_clock_mean_mhz"]))
                            if r else "-")
    print(line)

fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 4.8))
xs = list(range(len(cams)))
for a in ARMS:
    pts = [(x, rows[a][c]) for x, c in zip(xs, cams) if c in rows.get(a, {})]
    if not pts:
        continue
    a1.plot([x for x, _ in pts], [float(r["util_mean_pct"]) for _, r in pts], "o-", lw=2, ms=5,
            color=COLOR[a], label=LABEL[a])
    a3.plot([x for x, _ in pts], [float(r["sm_clock_mean_mhz"]) for _, r in pts], "o-", lw=2, ms=5,
            color=COLOR[a], label=LABEL[a])
    # turnaround against utilisation: stop at the first overloaded point, where
    # utilisation is pinned and turnaround is a queue, not a pipeline property
    u, t = [], []
    for _, r in pts:
        u.append(float(r["util_mean_pct"])); t.append(float(r["p50_ms"]))
        if float(r["p50_ms"]) > 100:
            break
    a2.plot(u, t, "o-", lw=2, ms=5, color=COLOR[a], label=LABEL[a])

for ax in (a1, a3):
    ax.set_xticks(xs)
    ax.set_xticklabels(["%d\n%d fps" % (c, c * 30) for c in cams], fontsize=9)
    ax.set_xlabel("cameras at 30 fps (RTX 3090)")
    ax.grid(alpha=.25)
a1.axhline(100, ls="--", lw=1, color="#888")
a1.set_ylabel("GPU utilisation (%, nvidia-smi)")
a1.set_title("How busy the 3090 was at each load", fontsize=11)
a1.legend(fontsize=8, loc="upper left")
a2.set_yscale("log")
a2.set_xlabel("GPU utilisation (%) — measure this on your own GPU")
a2.set_ylabel("p50 turnaround (ms, log)")
a2.set_title("Turnaround against utilisation — the view that transfers", fontsize=11)
a2.axhline(33.3, ls=":", lw=1, color="#c00")
a2.grid(alpha=.25, which="both")
a3.set_ylabel("mean SM clock (MHz)")
a3.set_title("SM clock: base when idle, boost mid-load, power-limited at the wall", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=140)
print("wrote", os.path.relpath(OUT, REPO))
