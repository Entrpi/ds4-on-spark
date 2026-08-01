#!/usr/bin/env python3
"""Two-panel branch-vs-upstream throughput overlay (prefill + generation) as SVG.

Reads two ds4-bench CSVs (same ctx grid) — an upstream baseline and a branch —
and draws the side-by-side overlay used in the ds4-on-spark README: a gray
upstream line and a green branch line per panel, with the band between them
shaded green where the branch is faster and red where it is slower, plus
endpoint callouts (prefill as a x-ratio, generation as a +%). Standard library
only; mirrors the visual style of docs/gb10-uplift-vs-upstream.svg.

Usage:
  python3 scripts/plot_overlay.py BRANCH.csv UPSTREAM.csv \
      --out docs/gb10-uplift-vs-upstream.svg \
      --title "DGX Spark (GB10) throughput: upstream antirez/ds4 vs the ds4-on-spark branch" \
      --branch-label "ds4-on-spark branch (5625a99)" \
      --upstream-label "upstream antirez/ds4 main" \
      --subtitle "V4 Flash IQ2XXS imatrix · ds4-bench ctx 2k-64k step 2k gen=128" \
      --source "ds4-bench · GB10 (sm_121) · captured decode · 2026-06-01"
"""

import argparse
import csv
import html
import math
from pathlib import Path

UPSTREAM = "#7a8497"
BRANCH = "#2fa97a"


def read_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            try:
                rows.append((int(r["ctx_tokens"]),
                             float(r["prefill_tps"]),
                             float(r["gen_tps"])))
            except (KeyError, ValueError):
                continue
    rows.sort(key=lambda x: x[0])
    return rows


def nice_ceil(v):
    if v <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(v))
    n = v / mag
    for s in (1, 1.5, 2, 2.5, 3, 4, 5, 10):
        if n <= s:
            return s * mag
    return 10 * mag


def fmt_ctx(t):
    return f"{t // 1024}k" if t % 1024 == 0 else f"{t/1024:.0f}k"


def panel(rows_b, rows_u, key, x0, x1, y0, y1, title, fmt_call, n_yticks=5):
    """Return SVG fragment for one panel. key: 1=prefill_tps, 2=gen_tps."""
    xs = [r[0] for r in rows_b]
    vb = [r[key] for r in rows_b]
    vu = [r[key] for r in rows_u]
    n = len(xs)
    vmax = nice_ceil(max(max(vb), max(vu)) * 1.05)
    out = []

    def px(i):
        return x0 + (x1 - x0) * (i / (n - 1)) if n > 1 else x0

    def py(v):
        return y1 - (y1 - y0) * (v / vmax)

    # horizontal grid + y labels
    step = vmax / n_yticks
    for k in range(n_yticks + 1):
        val = step * k
        yy = py(val)
        out.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}"/>')
        out.append(f'<text class="axis-label" x="{x0-8}" y="{yy+3:.1f}" text-anchor="end">{val:g}</text>')
    # vertical grid at ~5 positions
    label_idx = {0: fmt_ctx(xs[0]), n - 1: fmt_ctx(xs[-1])}
    for frac in (0.25, 0.5, 0.75):
        label_idx[round(frac * (n - 1))] = fmt_ctx(xs[round(frac * (n - 1))])
    for i in range(n):
        if i in label_idx:
            xx = px(i)
            out.append(f'<line class="grid" x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}"/>')
            out.append(f'<text class="axis-label" x="{xx:.1f}" y="{y1+16}" text-anchor="middle">{label_idx[i]}</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    cy = (y0 + y1) / 2
    out.append(f'<text class="axis-label" x="{x0-52}" y="{cy:.1f}" text-anchor="middle" transform="rotate(-90 {x0-52} {cy:.1f})">tok/s</text>')
    out.append(f'<text class="axis-label" x="{(x0+x1)/2:.0f}" y="{y1+34}" text-anchor="middle">context tokens</text>')
    out.append(f'<text class="panel-title" x="{(x0+x1)/2:.0f}" y="{y0-12}" text-anchor="middle">{html.escape(title)}</text>')

    # gain/loss band: split into runs of same sign
    i = 0
    while i < n:
        sign = vb[i] >= vu[i]
        j = i
        while j + 1 < n and (vb[j + 1] >= vu[j + 1]) == sign:
            j += 1
        seg = list(range(i, j + 1))
        pts = [f"{px(k):.1f},{py(vb[k]):.1f}" for k in seg]
        pts += [f"{px(k):.1f},{py(vu[k]):.1f}" for k in reversed(seg)]
        cls = "gain" if sign else "loss"
        out.append(f'<polygon class="{cls}" points="{" ".join(pts)}"/>')
        i = j + 1

    up_pts = " ".join(f"{px(i):.1f},{py(vu[i]):.1f}" for i in range(n))
    br_pts = " ".join(f"{px(i):.1f},{py(vb[i]):.1f}" for i in range(n))
    out.append(f'<polyline class="line-upstream" points="{up_pts}"/>')
    out.append(f'<polyline class="line-fork" points="{br_pts}"/>')
    out.append(f'<circle class="dot-upstream" cx="{px(0):.1f}" cy="{py(vu[0]):.1f}" r="3"/>')
    out.append(f'<circle class="dot-fork" cx="{px(0):.1f}" cy="{py(vb[0]):.1f}" r="3"/>')

    def callout(i, anchor_end):
        c = fmt_call(vb[i], vu[i])
        good = vb[i] >= vu[i]
        cls = "callout" if good else "callout-bad"
        xx = px(i) + (-8 if anchor_end else 8)
        yy = min(py(vb[i]), py(vu[i])) - 6
        a = 'text-anchor="end"' if anchor_end else ""
        return f'<text class="{cls}" x="{xx:.1f}" y="{yy:.1f}" {a}>{c} @ {fmt_ctx(xs[i])}</text>'

    out.append(callout(0, False))
    out.append(callout(n - 1, True))

    # per-series mean/max stats block, bottom-left inside the plot area
    # (the band between the lines never reaches this strip: both series'
    # minima sit well above the x-axis at every stamp so far)
    def stats(vals):
        return sum(vals) / len(vals), max(vals)
    bm, bx = stats(vb)
    um, ux = stats(vu)
    sx = x0 + 10
    out.append(f'<text class="stat-fork" x="{sx}" y="{y1-24:.1f}">mean {bm:,.1f} · max {bx:,.1f}</text>')
    out.append(f'<text class="stat-upstream" x="{sx}" y="{y1-9:.1f}">mean {um:,.1f} · max {ux:,.1f}</text>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("branch_csv")
    ap.add_argument("upstream_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="throughput: upstream vs branch")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--source", default="")
    ap.add_argument("--branch-label", default="branch")
    ap.add_argument("--upstream-label", default="upstream")
    ap.add_argument("--prefill-title", default="prefill")
    ap.add_argument("--gen-title", default="generation")
    a = ap.parse_args()

    rb, ru = read_rows(a.branch_csv), read_rows(a.upstream_csv)
    n = min(len(rb), len(ru))
    rb, ru = rb[:n], ru[:n]

    pre = panel(rb, ru, 1, 78, 595, 130, 370, a.prefill_title,
                lambda b, u: f"{b/u:.2f}x" if u else "n/a")
    gen = panel(rb, ru, 2, 655, 1172, 130, 370, a.gen_title,
                lambda b, u: f"{(b/u-1)*100:+.0f}%" if u else "n/a")

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 470" width="1200" height="470" role="img" aria-label="ds4 throughput overlay">
<style>
  .bg {{ fill: #ffffff; }}
  .title    {{ font: 600 17px -apple-system, Segoe UI, Inter, Helvetica, Arial, sans-serif; fill: #1f2329; }}
  .subtitle {{ font: 12.5px -apple-system, Segoe UI, Inter, Helvetica, Arial, sans-serif; fill: #566174; }}
  .panel-title {{ font: 600 14px -apple-system, Segoe UI, Inter, Helvetica, Arial, sans-serif; fill: #1f2329; }}
  .axis-label  {{ font: 11px -apple-system, Segoe UI, Inter, Helvetica, Arial, sans-serif; fill: #566174; }}
  .legend      {{ font: 12px -apple-system, Segoe UI, Inter, Helvetica, Arial, sans-serif; fill: #1f2329; }}
  .callout     {{ font: 700 12px -apple-system, Segoe UI, Inter, Helvetica, Arial, sans-serif; fill: #1f8a5a; }}
  .callout-bad {{ font: 700 12px -apple-system, Segoe UI, Inter, Helvetica, Arial, sans-serif; fill: #c0392b; }}
  .grid {{ stroke: #eef0f3; stroke-width: 0.8; }}
  .axis {{ stroke: #c8ccd2; stroke-width: 1; }}
  .line-upstream {{ fill: none; stroke: {UPSTREAM}; stroke-width: 2; }}
  .line-fork     {{ fill: none; stroke: {BRANCH}; stroke-width: 2.4; }}
  .dot-upstream  {{ fill: {UPSTREAM}; }}
  .dot-fork      {{ fill: {BRANCH}; }}
  .gain {{ fill: {BRANCH}; fill-opacity: 0.18; stroke: none; }}
  .loss {{ fill: #d64545; fill-opacity: 0.16; stroke: none; }}
  .stat-fork     {{ font: 600 11.5px ui-monospace, SFMono-Regular, Menlo, monospace; fill: {BRANCH}; }}
  .stat-upstream {{ font: 600 11.5px ui-monospace, SFMono-Regular, Menlo, monospace; fill: {UPSTREAM}; }}
</style>
<rect class="bg" x="0" y="0" width="1200" height="470"/>
<text class="title" x="600" y="30" text-anchor="middle">{html.escape(a.title)}</text>
<text class="subtitle" x="600" y="52" text-anchor="middle">{html.escape(a.subtitle)}</text>
<g transform="translate(300,66)">
<line class="line-upstream" x1="0" y1="8" x2="32" y2="8"/>
<text class="legend" x="40" y="12">{html.escape(a.upstream_label)}</text>
<line class="line-fork" x1="300" y1="8" x2="332" y2="8"/>
<text class="legend" x="340" y="12">{html.escape(a.branch_label)}</text>
</g>
{pre}
{gen}
<text class="subtitle" x="600" y="454" text-anchor="middle">{html.escape(a.source)}</text>
</svg>
'''
    Path(a.out).write_text(svg, encoding="utf-8")
    print(f"wrote {a.out}  ({n} frontiers/panel)")


if __name__ == "__main__":
    main()
