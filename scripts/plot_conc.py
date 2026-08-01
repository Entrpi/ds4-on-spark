#!/usr/bin/env python3
"""Served aggregate throughput vs concurrency as SVG. Standard library only.

Input CSV columns: concurrency,agg_tps,per_stream_tps  (one row per N).
Draws the aggregate line with per-point value labels and a per-stream
reference line — the ds4-on-spark README concurrency chart.

Usage:
  python3 scripts/plot_conc.py conc_points.csv --out docs/v050_conc_throughput.svg \
      --title "ds4-server aggregate decode throughput vs concurrency" \
      --subtitle "DSpark ship config · 192-token completions · repeats x3, warmup discarded" \
      --source "conc_sweep.py · GB10 · v0.5.0"
"""
import argparse
import csv
import html

AGG = "#2fa97a"
PER = "#7a8497"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="ds4-server throughput vs concurrency")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--source", default="")
    a = ap.parse_args()

    rows = []
    with open(a.csv, encoding="utf-8-sig", newline="") as fp:
        for r in csv.DictReader(fp):
            rows.append((int(r["concurrency"]), float(r["agg_tps"]), float(r["per_stream_tps"])))
    rows.sort()

    W, H = 760, 430
    # right margin leaves room for the in-chart series labels at px(last)+6
    x0, x1, y0, y1 = 84, 636, 84, 344
    ns = [n for n, _, _ in rows]
    ymax = max(agg for _, agg, _ in rows) * 1.18

    def px(i):
        return x0 + (x1 - x0) * i / (len(rows) - 1)

    def py(v):
        return y1 - (y1 - y0) * (v / ymax)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="throughput vs concurrency">']
    o.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    o.append(f'<text x="{x0}" y="34" font-family="system-ui,sans-serif" font-size="17" font-weight="600" fill="#1f2933">{html.escape(a.title)}</text>')
    if a.subtitle:
        o.append(f'<text x="{x0}" y="54" font-family="system-ui,sans-serif" font-size="12" fill="#64748b">{html.escape(a.subtitle)}</text>')
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        v = ymax * frac
        y = py(v)
        o.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        o.append(f'<text x="{x0-8}" y="{y+4:.1f}" text-anchor="end" font-family="ui-monospace,monospace" font-size="11" fill="#64748b">{v:.0f}</text>')
    for i, (n, _, _) in enumerate(rows):
        o.append(f'<text x="{px(i):.1f}" y="{y1+20}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="12" fill="#334155">{n}</text>')
    o.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1+44}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#64748b">concurrent requests</text>')
    o.append(f'<text x="20" y="{(y0+y1)/2:.0f}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="12" fill="#64748b" transform="rotate(-90 20 {(y0+y1)/2:.0f})">tokens / second</text>')

    def line(vals, color, width):
        pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(vals))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"/>')

    line([per for _, _, per in rows], PER, 1.6)
    line([agg for _, agg, _ in rows], AGG, 2.6)
    for i, (n, agg, per) in enumerate(rows):
        o.append(f'<circle cx="{px(i):.1f}" cy="{py(agg):.1f}" r="4" fill="{AGG}"/>')
        o.append(f'<text x="{px(i):.1f}" y="{py(agg)-10:.1f}" text-anchor="middle" font-family="ui-monospace,monospace" font-size="11.5" font-weight="600" fill="#1f2933">{agg:.1f}</text>')
        o.append(f'<circle cx="{px(i):.1f}" cy="{py(per):.1f}" r="3" fill="{PER}"/>')
    o.append(f'<text x="{px(len(rows)-1)+6:.1f}" y="{py(rows[-1][1])+4:.1f}" font-family="system-ui,sans-serif" font-size="11.5" fill="{AGG}">aggregate</text>')
    o.append(f'<text x="{px(len(rows)-1)+6:.1f}" y="{py(rows[-1][2])+4:.1f}" font-family="system-ui,sans-serif" font-size="11.5" fill="{PER}">per stream</text>')
    if a.source:
        o.append(f'<text x="{x0}" y="{H-14}" font-family="ui-monospace,monospace" font-size="10.5" fill="#94a3b8">{html.escape(a.source)}</text>')
    o.append("</svg>")
    with open(a.out, "w", encoding="utf-8") as fp:
        fp.write("\n".join(o) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
