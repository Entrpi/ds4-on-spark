#!/usr/bin/env python3
"""Decode throughput vs context, one line per device. Standard library only.

Input CSV columns: device,ctx_tokens,gen_tps  (one row per measured point).
Draws a clean multi-series line chart (log-x context, linear-y tok/s) with
endpoint value labels — the ds4 fastest-path (CLI captured) decode curve.

Usage:
  python3 scripts/plot_decode_ctx.py decode_points.csv --out docs/decode-vs-ctx.svg \
      --title "ds4 fastest-path decode vs context (5625a99, captured)"
"""
import argparse, csv, html, math
from collections import OrderedDict
from pathlib import Path

COLORS = {"PRO 6000": "#2fa97a", "GB10": "#3b6fd6"}
DEFAULT = ["#2fa97a", "#3b6fd6", "#d68a2f", "#a04fd0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="ds4 fastest-path decode vs context")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--source", default="")
    a = ap.parse_args()

    series = OrderedDict()
    with open(a.csv, encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            series.setdefault(r["device"], []).append((float(r["ctx_tokens"]), float(r["gen_tps"])))
    for k in series:
        series[k].sort()

    W, H = 980, 470
    x0, x1, y0, y1 = 92, 930, 92, 384
    all_ctx = [c for pts in series.values() for c, _ in pts]
    all_tps = [t for pts in series.values() for _, t in pts]
    lxmin, lxmax = math.log10(min(all_ctx)), math.log10(max(all_ctx))
    ymax = math.ceil(max(all_tps) / 10) * 10

    def px(c):
        return x0 + (x1 - x0) * (math.log10(c) - lxmin) / (lxmax - lxmin)

    def py(t):
        return y1 - (y1 - y0) * (t / ymax)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="ds4 decode vs context">',
        '<style>',
        '.bg{fill:#fff}',
        '.title{font:600 17px -apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif;fill:#1f2329}',
        '.subtitle{font:12.5px -apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif;fill:#566174}',
        '.axis-label{font:11px -apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif;fill:#566174}',
        '.legend{font:600 13px -apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif}',
        '.val{font:700 12px -apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif}',
        '.grid{stroke:#eef0f3;stroke-width:.8}.axis{stroke:#c8ccd2;stroke-width:1}',
        '</style>',
        f'<rect class="bg" x="0" y="0" width="{W}" height="{H}"/>',
        f'<text class="title" x="{W/2}" y="30" text-anchor="middle">{html.escape(a.title)}</text>',
        f'<text class="subtitle" x="{W/2}" y="52" text-anchor="middle">{html.escape(a.subtitle)}</text>',
    ]
    # y grid
    for k in range(0, ymax + 1, max(10, ymax // 6 // 10 * 10 or 10)):
        yy = py(k)
        out.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}"/>')
        out.append(f'<text class="axis-label" x="{x0-8}" y="{yy+3:.1f}" text-anchor="end">{k}</text>')
    # x grid at powers + key ctx
    for c in [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        if lxmin - 0.05 <= math.log10(c) <= lxmax + 0.05:
            xx = px(c)
            out.append(f'<line class="grid" x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y1}"/>')
            out.append(f'<text class="axis-label" x="{xx:.1f}" y="{y1+16}" text-anchor="middle">{c//1024}k</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    out.append(f'<text class="axis-label" x="40" y="{(y0+y1)/2:.0f}" text-anchor="middle" transform="rotate(-90 40 {(y0+y1)/2:.0f})">decode tok/s</text>')
    out.append(f'<text class="axis-label" x="{(x0+x1)/2:.0f}" y="{y1+34}" text-anchor="middle">context tokens (log scale)</text>')

    lx = x0 + 16
    for i, (dev, pts) in enumerate(series.items()):
        col = COLORS.get(dev, DEFAULT[i % len(DEFAULT)])
        poly = " ".join(f"{px(c):.1f},{py(t):.1f}" for c, t in pts)
        out.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="2.6"/>')
        for c, t in pts:
            out.append(f'<circle cx="{px(c):.1f}" cy="{py(t):.1f}" r="3.4" fill="{col}"/>')
        # endpoint labels
        c0, t0 = pts[0]; cN, tN = pts[-1]
        out.append(f'<text class="val" x="{px(c0)+8:.1f}" y="{py(t0)-7:.1f}" fill="{col}">{t0:.1f}</text>')
        out.append(f'<text class="val" x="{px(cN)-8:.1f}" y="{py(tN)-7:.1f}" fill="{col}" text-anchor="end">{tN:.1f}</text>')
        # legend
        out.append(f'<line x1="{lx}" y1="74" x2="{lx+26}" y2="74" stroke="{col}" stroke-width="3"/>')
        out.append(f'<text class="legend" x="{lx+32}" y="78" fill="{col}">{html.escape(dev)}</text>')
        lx += 32 + 9 * len(dev) + 40
    out.append(f'<text class="subtitle" x="{W/2}" y="454" text-anchor="middle">{html.escape(a.source)}</text>')
    out.append('</svg>')
    Path(a.out).write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
