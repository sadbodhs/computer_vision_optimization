#!/usr/bin/env python3
"""Figures for pages that were text-only.

Three panels, one per page that needed a visual, each from data already
committed under results/v3/:

  batching-contention.png  -> docs/batching.md
      D's client latency split into "what the server admits" (queue +
      inference) and "what the client adds" (its own in-flight window), with
      Little's law (8N/fps) as the predicted line. Shows the queue is never
      the whole wait: the client's own window is most of it, and past
      concurrency 4 the queue share collapses while the window keeps growing.

  contention-scaling.png  -> docs/contention.md
      p50 latency of A2 and B2 as N independent instances share the GPU
      (data from parallel_contention_N3.tsv, N=1..3, median of the JSON
      repeats), with and without MPS (mps_contention_N3.tsv, N=3). Shows the
      inversion: MPS cuts A2's latency but adds nothing for B2.

  fewer-bytes-flows.png   -> docs/fewer-bytes.md
      Wire cost vs delivered gain, per flow, for the UINT8-input engine
      against FP32 (data from fewer_bytes_flows.tsv): H2D time falls ~4x
      everywhere, but only the raw-gRPC flow converts it into throughput,
      and D, already at the engine ceiling, gains nothing.

Usage: python3 scripts/plot_page_figs.py [repo_root]
"""
import csv
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
V3 = os.path.join(REPO, "results", "v3")
IMG = os.path.join(REPO, "docs", "img")

ACCENT = "#e8710a"
BLUE = "#1a73e8"
PURPLE = "#a142f4"
GREY = "#9aa0a6"
GREEN = "#12a150"
RED = "#d93025"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# ---------------------------------------------------------------- batching
def fig_batching():
    rows = read_tsv(os.path.join(V3, "d_latency_decomposition.tsv"))
    conc = [int(r["concurrency"]) for r in rows]
    client = [float(r["client_p50_ms"]) for r in rows]
    queue = [float(r["srv_queue_ms"]) for r in rows]
    infer = [float(r["srv_infer_ms"]) for r in rows]
    little = [float(r["littles_law_ms"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.62
    ax.bar(conc, queue, width, color=PURPLE, label="server queue wait")
    ax.bar(conc, infer, width, bottom=queue, color=ACCENT, label="server inference")
    own = [c - q - i for c, q, i in zip(client, queue, infer)]
    ax.bar(conc, own, width, bottom=[q + i for q, i in zip(queue, infer)],
           color=GREY, label="client's own in-flight window")
    ax.plot(conc, little, "o--", color=BLUE, lw=1.5, ms=5,
            label="Little's law: 8N / fps")
    ax.set_xlabel("concurrency (streams × 8 in flight)")
    ax.set_ylabel("client p50 latency (ms)")
    ax.set_xticks(conc)
    ax.set_title("D's wait is mostly the client's own window, not the queue")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "batching-contention.png"))
    plt.close(fig)


# --------------------------------------------------------------- contention
def _p50s_by_n(path, key):
    """median p50 per (flow, N) from the parallel-contention JSON rows."""
    out = {}
    for r in read_tsv(path):
        j = json.loads(r[key])
        k = (j["pipeline"], r.get("instance") or r.get("condition"))
        out.setdefault(k, []).append(j["lat_ms_p50"])
    return out


def fig_contention():
    # header is "flow,instance,json" and rows are comma-separated with the JSON
    # as the final field (split on the first two commas only)
    path = os.path.join(V3, "parallel_contention_N3.tsv")

    # rows: flow= A2/B2..., instance = N (1..3)
    by = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            flow, inst, js = line.rstrip("\n").split(",", 2)
            n = int(inst)
            by.setdefault((flow, n), []).append(json.loads(js)["lat_ms_p50"])

    def series(flow):
        xs = sorted(n for (f, n) in by if f == flow)
        return xs, [statistics.median(by[(flow, n)]) for n in xs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))

    for flow, label, color in (("A2", "A2 · C++ TRT", ACCENT),
                               ("B2", "B2 · Triton + CUDA-shm", BLUE)):
        try:
            xs, ys = series(flow)
        except KeyError:
            continue
        ax1.plot(xs, ys, "o-", color=color, label=label)
    ax1.set_xlabel("instances sharing the GPU")
    ax1.set_ylabel("p50 latency (ms)")
    ax1.set_title("Without MPS: every instance pays")
    ax1.legend(fontsize=9, frameon=False)

    # MPS panel: mps_contention_N3.tsv, condition = <flow>_mps_<off/on>
    mps_path = os.path.join(V3, "mps_contention_N3.tsv")
    med = {}
    for r in read_tsv(mps_path):
        cond = r["condition"]
        flow, _, state = cond.rpartition("_")
        j = json.loads(r["json"])
        med.setdefault((flow, state), []).append(j["lat_ms_p50"])
    labels, offs, ons = [], [], []
    pretty = {"a2": "A2", "b2": "B2"}
    for flow in ("a2", "b2"):
        if (flow, "off") in med and (flow, "on") in med:
            labels.append(pretty[flow])
            offs.append(statistics.median(med[(flow, "off")]))
            ons.append(statistics.median(med[(flow, "on")]))
    x = range(len(labels))
    w = 0.36
    ax2.bar([i - w / 2 for i in x], offs, w, color=GREY, label="MPS off")
    ax2.bar([i + w / 2 for i in x], ons, w, color=GREEN, label="MPS on")
    for i, (o, n) in enumerate(zip(offs, ons)):
        d = (n - o) / o * 100
        ax2.annotate(f"{d:+.0f}%", (i, min(o, n)),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("p50 latency (ms)")
    ax2.set_title("With MPS at N=3: A2 wins, B2 does not")
    ax2.legend(fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "contention-scaling.png"))
    plt.close(fig)


# ------------------------------------------------------------- fewer-bytes
def fig_fewer_bytes():
    path = os.path.join(V3, "fewer_bytes_flows.tsv")
    rows = read_tsv(path)

    # rows are long-form: one row per (flow, input_dtype)
    by = {}
    for r in rows:
        by.setdefault(r["flow"], {})[r["input_dtype"]] = (
            float(r["throughput"]), r.get("source", ""))
    flows = sorted(by)
    fp32, u8, skip = [], [], []
    for f in flows:
        d = by[f]
        if "fp32" in d and "uint8" in d:
            fp32.append(d["fp32"][0])
            u8.append(d["uint8"][0])
            skip.append(False)
        else:  # D: measured 0% because it is already at the engine ceiling
            base = d.get("fp32", (0, ""))[0]
            fp32.append(base)
            u8.append(base)
            skip.append(True)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = list(range(len(flows)))
    w = 0.36
    ax.bar([i - w / 2 for i in x], fp32, w, color=GREY,
           label="FP32 input (4.92 MB)")
    ax.bar([i + w / 2 for i in x], u8, w, color=ACCENT,
           label="UINT8 input (1.23 MB)")
    for i, (a, b, s) in enumerate(zip(fp32, u8, skip)):
        d = (b - a) / a * 100 if a else 0
        txt = "0%\n(ceiling)" if s else f"{d:+.0f}%"
        ax.annotate(txt, (i, max(a, b)), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(flows)
    ax.set_ylabel("throughput (fps)")
    ax.set_title("Same lever, opposite outcomes: only the transport-bound flow gains")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fewer-bytes-flows.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    x = range(len(flows))
    w = 0.36
    ax.bar([i - w / 2 for i in x], fp32, w, color=GREY, label="FP32 input (4.92 MB)")
    ax.bar([i + w / 2 for i in x], u8, w, color=ACCENT, label="UINT8 input (1.23 MB)")
    for i, (a, b) in enumerate(zip(fp32, u8)):
        d = (b - a) / a * 100
        ax.annotate(f"{d:+.0f}%" if abs(d) >= 0.5 else "0%",
                    (i, max(a, b)), ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(flows)
    ax.set_ylabel("throughput (fps)")
    ax.set_title("Same lever, opposite outcomes: only the transport-bound flow gains")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fewer-bytes-flows.png"))
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(IMG, exist_ok=True)
    fig_batching()
    print("batching-contention.png written")
    fig_contention()
    print("contention-scaling.png written")
    fig_fewer_bytes()