#!/usr/bin/env python3
"""Compare the post-fix slice (leakfix_*_slice.tsv) with the published paced results.

Scores predictions P2-P4 of scripts/leakfix_check.sh. Every statistic is the
median across the three seeded repeats, on both sides.

Usage: python3 scripts/leakfix_compare.py [repo_root]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
R = os.path.join(REPO, "results", "v3")
TOL = 0.10          # P2/P3 bar: the cross-sweep anchor spread
OVERLOAD_MS = 100.0


def load(fn):
    d = defaultdict(list)
    for r in csv.DictReader(open(os.path.join(R, fn)), delimiter="\t"):
        d[(r["arm"], r["phase"], int(r["cameras"]))].append(float(r["p50_ms"]))
    return {k: st.median(v) for k, v in d.items()}


pub = {**load("b3_paced.tsv"), **load("a2_paced.tsv")}
new = {**load("leakfix_b3_slice.tsv"), **load("leakfix_a2_slice.tsv")}

verdict = {"P2": True, "P3": True}
print("%-10s %-6s %4s %12s %12s %8s  %s" % ("arm", "phase", "cams", "published", "after fix", "change", "check"))
for k in sorted(new):
    arm, phase, cams = k
    if k not in pub:
        continue
    a, b = pub[k], new[k]
    if a > OVERLOAD_MS:
        ok = b > OVERLOAD_MS
        check = "stays overloaded" if ok else "NO LONGER OVERLOADED"
    else:
        ok = abs(b - a) / a <= TOL
        check = "within 10%" if ok else "MOVED"
    p = "P3" if arm == "a2" else "P2"
    verdict[p] = verdict[p] and ok
    print("%-10s %-6s %4d %12.3f %12.3f %+7.1f%%  %s" % (arm, phase, cams, a, b, 100 * (b - a) / a, check))

# P4: fastest pipeline per (phase, cameras) among the arms re-run
arms = ("a2", "b2", "b3_0", "dnow")
p4 = True
print("\nfastest p50 per load (P4):")
for phase in ("random", "sync"):
    for cams in (8, 32, 48):
        def best(src):
            vals = [(src[(a, phase, cams)], a) for a in arms if (a, phase, cams) in src]
            return min(vals)[1] if vals else "-"
        bp, bn = best(pub), best(new)
        p4 = p4 and bp == bn
        print("  %-6s %2d cams: published %-5s after fix %-5s %s" % (phase, cams, bp, bn, "" if bp == bn else "CHANGED"))

print("\nP2 (Triton cells within 10%%): %s" % ("held" if verdict["P2"] else "FAILED"))
print("P3 (A2 control within 10%%):   %s" % ("held" if verdict["P3"] else "FAILED"))
print("P4 (same fastest pipeline):    %s" % ("held" if p4 else "FAILED"))
