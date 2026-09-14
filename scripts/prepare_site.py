#!/usr/bin/env python3
"""Stage the repo's markdown for MkDocs without duplicating anything in git.

The committed docs use relative links that are correct when browsing the repo on
GitHub (../scripts/foo.sh, ../README.md). On a published site those resolve
outside the site root and 404. This rewrites them at BUILD time only:

  docs/introduction.md    -> index.md             (the site LANDS on the intro)
  ../README.md            -> overview.md          (README is a page, not the root)
  ../<non-md path>        -> absolute GitHub blob URL
  docs/foo.md (in README) -> foo.md

Why the intro is the root: GitHub always shows README.md first and there is no
changing that, but a site visitor has no such constraint, and someone arriving
cold is better served by "here is the problem" than by a table of numbers. The
README keeps its job as the repo front page and becomes /overview/ on the site.

Committed files are never modified.
"""
import os
import re
import shutil
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/site/src"
BLOB = "https://github.com/sadbodhs/computer_vision_optimization/blob/main"

HOME = "introduction.md"      # the doc promoted to index.md
README_PAGE = "overview.md"   # where README.md lands in the site tree

shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

link_re = re.compile(r"\]\(([^)]+)\)")


def fix_docs_link(m):
    href = m.group(1)
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return m.group(0)
    target, hashmark, frag = href.partition("#")
    if target == "../README.md":
        return "](%s%s%s)" % (README_PAGE, hashmark, frag)
    if target == HOME:
        return "](index.md%s%s)" % (hashmark, frag)
    if href.startswith("../"):
        # .md files that live outside docs/ (e.g. ../results/README.md) also
        # aren't in the site tree -> send them to GitHub too
        return f"]({BLOB}/{href[3:]})"
    return m.group(0)


def fix_readme_link(m):
    href = m.group(1)
    if href.startswith(("http://", "https://", "#", "mailto:")):
        return m.group(0)
    target, hashmark, frag = href.partition("#")
    if target == "docs/" + HOME:
        return "](index.md%s%s)" % (hashmark, frag)
    if href.startswith("docs/"):
        return f"]({href[5:]})"          # docs/precision.md -> precision.md
    return f"]({BLOB}/{href})"           # STORY.md, scripts/, results/, triton/


# docs/*.md -> site root, with introduction.md promoted to index.md
n = 0
for name in sorted(os.listdir(os.path.join(REPO, "docs"))):
    if not name.endswith(".md"):
        continue
    text = open(os.path.join(REPO, "docs", name), encoding="utf-8").read()
    out_name = "index.md" if name == HOME else name
    open(os.path.join(OUT, out_name), "w", encoding="utf-8").write(
        link_re.sub(fix_docs_link, text))
    n += 1

# non-markdown assets under docs/ (figures) must come along, or they 404
for sub in ("img",):
    src_dir = os.path.join(REPO, "docs", sub)
    if os.path.isdir(src_dir):
        shutil.copytree(src_dir, os.path.join(OUT, sub))
        print(f"copied docs/{sub}/")

# On the site, swap the static Pareto PNG for the interactive Plotly version.
# Drawn INTO the page rather than into an iframe: an iframe is its own document
# with its own canvas, so it rendered as a white box on the dark theme, and its
# raw-HTML src is not rewritten by MkDocs so the relative depth had to be right
# by hand -- which it once wasn't, silently loading the 404 page. See the header
# of scripts/make_interactive.py.
# GitHub markdown cannot run JavaScript, so the committed .md keeps the PNG and
# only the published site gets the chart. Same data either way.
#
# The ../ below is still hand-maintained: both pages that take the swap are
# served at <page>/index.html, one level down. Assert it rather than trust it.
CHART = (
    '<div id="pareto-chart" style="width:100%; height:540px;"></div>\n'
    '<script src="../img/plotly-2.35.2.min.js" charset="utf-8"></script>\n'
    '<script src="../img/pareto_chart.js"></script>'
)
CAPTION = (
    '\n\n*Interactive: drag to zoom, click a legend entry to isolate a flow, hover a\n'
    'point for its exact concurrency, throughput and latency. The static version of\n'
    'this chart is in the repository.*'
)

PARETO_PNG = "![Latency versus throughput for every flow, swept over concurrency 1-16](img/pareto-latency-throughput.png)"
results_md = os.path.join(OUT, "results.md")
if os.path.exists(results_md):
    _t = open(results_md, encoding="utf-8").read()
    if PARETO_PNG in _t:
        open(results_md, "w", encoding="utf-8").write(
            _t.replace(PARETO_PNG, CHART + CAPTION))
        print("swapped the Pareto PNG for the interactive chart")
    else:
        print("WARNING: results.md Pareto PNG not matched - it will stay static")

# README.md -> overview.md
readme = open(os.path.join(REPO, "README.md"), encoding="utf-8").read()
overview_md = link_re.sub(fix_readme_link, readme)

# The README's "read this as a site" link is for repo visitors; on the site it
# points at the site you are already on, so drop it.
SITE_LINK = ("**\U0001F4D6 [Read this as a site](%s)** — searchable, with an interactive version\n"
             "of the chart below.\n\n" % "https://sadbodhs.github.io/computer_vision_optimization/overview/")
if SITE_LINK in overview_md:
    overview_md = overview_md.replace(SITE_LINK, "")
    print("dropped the self-referential site link from the overview")

# Same swap on the overview: served at /overview/index.html, the same depth as
# results.md, hence the same ../img/ src.
INDEX_PNG = "![Latency versus throughput for every flow](img/pareto-latency-throughput.png)"
if INDEX_PNG in overview_md:
    overview_md = overview_md.replace(INDEX_PNG, CHART)
    print("swapped the overview hero PNG for the interactive chart")
else:
    print("WARNING: overview hero PNG not matched - it will stay static")


# The site LANDS on the introduction (index.md), so it needs the hero chart
# most of all. It is served at the site root, so the asset src has no ../.
ROOT_CHART = (
    '<div id="pareto-chart" style="width:100%; height:540px;"></div>\n'
    '<script src="img/plotly-2.35.2.min.js" charset="utf-8"></script>\n'
    '<script src="img/pareto_chart.js"></script>'
)
index_md = os.path.join(OUT, "index.md")
if os.path.exists(index_md):
    _t = open(index_md, encoding="utf-8").read()
    # insert the hero under the first paragraph break, so the visitor sees it
    # before any wall of text
    parts = _t.split("\n---\n", 1)
    if len(parts) == 2:
        _t = parts[0] + "\n---\n\n" + ROOT_CHART + "\n\n" + parts[1]
        open(index_md, "w", encoding="utf-8").write(_t)
        print("hero chart inserted into the landing page")
    else:
        print("WARNING: index.md has no --- separator; hero not inserted")

open(os.path.join(OUT, README_PAGE), "w", encoding="utf-8").write(overview_md)

for _p in ("results.md", README_PAGE):
    assert _p != "index.md", "the chart src assumes depth 1, not the site root"


# --- social preview metadata on every page ---------------------------------
# Material does not emit og:image by default; link-preview crawlers (GitHub, X,
# Slack) read og: meta from the served HTML. The page <head> is generated, but
# meta tags in the markdown body are honoured by the major crawlers, and this
# is the only hook that needs no MkDocs plugin.
SITE = "https://sadbodhs.github.io/computer_vision_optimization"
OG_BLOCK = (
    '<meta property="og:title" content="Triton vs TensorRT vs DeepStream"/>\n'
    '<meta property="og:description" content="Measured comparison of CV '
    'inference serving pipelines on an RTX 3090"/>\n'
    '<meta property="og:image" content="%s/img/pareto-latency-throughput.png"/>\n'
    '<meta name="twitter:card" content="summary_large_image"/>\n' % SITE
)
for _name in os.listdir(OUT):
    if not _name.endswith(".md"):
        continue
    _p = os.path.join(OUT, _name)
    _t = open(_p, encoding="utf-8").read()
    open(_p, "w", encoding="utf-8").write(_t.rstrip() + "\n\n" + OG_BLOCK)

print(f"staged {n} docs (introduction -> index.md) + {README_PAGE} into {OUT}")
