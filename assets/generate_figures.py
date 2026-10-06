"""Generate SVG figures from MEASURED results (no hand-drawn numbers).

Reads results/04_curve.csv, results/*.json and writes assets/*.svg.
Pure stdlib (no extra dependency) so Docker stays slim.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(HERE, "results")
OUT = os.path.join(HERE, "assets")
os.makedirs(OUT, exist_ok=True)


def svg_line_chart(path, xs, series, title, xlabel, ylabel, markers=None):
    W, H, P = 720, 380, 56
    w, h = W - 2 * P, H - 2 * P
    xmax = max(xs)
    import math
    xs_log = [math.log10(x) for x in xs]
    x0, x1 = min(xs_log), max(xs_log)
    all_y = [y for s in series for y in s["ys"]]
    y1 = max(1.0, max(all_y) * 1.05)
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706"]

    def X(v):
        return P + (math.log10(v) - x0) / (x1 - x0) * w

    def Y(v):
        return P + h - (v / y1) * h

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" role="img">']
    parts.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    parts.append(f'<text x="{W//2}" y="28" text-anchor="middle" font-size="17" font-family="sans-serif" font-weight="bold">{title}</text>')
    # grid + axes
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = frac * y1
        parts.append(f'<line x1="{P}" y1="{Y(yv):.1f}" x2="{P+w}" y2="{Y(yv):.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{P-8}" y="{Y(yv)+4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#4b5563">{yv:.2f}</text>')
    for x in xs:
        parts.append(f'<text x="{X(x):.1f}" y="{P+h+18}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#4b5563">{x}</text>')
    parts.append(f'<text x="{W//2}" y="{H-6}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#374151">{xlabel}</text>')
    for i, s in enumerate(series):
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, s["ys"]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{colors[i % len(colors)]}" stroke-width="2.5"/>')
        for x, y in zip(xs, s["ys"]):
            parts.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" fill="{colors[i % len(colors)]}"/>')
        parts.append(f'<text x="{P + 10 + i*190}" y="{P - 14}" font-size="12" font-family="sans-serif" fill="{colors[i % len(colors)]}">\u25a0 {s["label"]}</text>')
    if markers:
        for mx, label in markers:
            parts.append(f'<line x1="{X(mx):.1f}" y1="{P}" x2="{X(mx):.1f}" y2="{P+h}" stroke="#9ca3af" stroke-dasharray="5,4"/>')
            parts.append(f'<text x="{X(mx):.1f}" y="{P-2}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#6b7280">{label}</text>')
    parts.append("</svg>")
    open(path, "w").write("\n".join(parts))
    return path


def svg_bars(path, labels, values, title, ylabel):
    W, H, P = 680, 360, 60
    w, h = W - 2 * P, H - 2 * P
    y1 = max(values) * 1.15
    n = len(labels)
    slot = w / n
    bw = min(120, slot * 0.55)
    cols = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2"]
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" role="img">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W//2}" y="28" text-anchor="middle" font-size="17" font-family="sans-serif" font-weight="bold">{title}</text>']
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yv = frac * y1
        yy = P + h - (yv / y1) * h
        p.append(f'<line x1="{P}" y1="{yy:.1f}" x2="{P+w}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        p.append(f'<text x="{P-8}" y="{yy+4:.1f}" text-anchor="end" font-size="11" font-family="sans-serif" fill="#4b5563">{yv:.2f}</text>')
    for i, (lb, v) in enumerate(zip(labels, values)):
        x = P + slot * i + (slot - bw) / 2
        y = P + h - (v / y1) * h
        p.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{P+h-y:.1f}" fill="{cols[i % len(cols)]}" rx="4"/>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" text-anchor="middle" font-size="12" font-family="sans-serif" font-weight="bold">{v:.3f}</text>')
        p.append(f'<text x="{x+bw/2:.1f}" y="{P+h+18}" text-anchor="middle" font-size="11" font-family="sans-serif" fill="#374151">{lb}</text>')
    p.append(f'<text x="{W//2}" y="{H-4}" text-anchor="middle" font-size="12" font-family="sans-serif" fill="#374151">{ylabel}</text>')
    p.append("</svg>")
    open(path, "w").write("\n".join(p))
    return path


# --- Fig 1: EXP-04 hit-rate curve (real CSV) ---
xs, matt = [], []
with open(os.path.join(RES, "04_curve.csv")) as f:
    for row in csv.DictReader(f):
        xs.append(int(row["capacity"]))
        matt.append(float(row["mattson"]))
svg_line_chart(os.path.join(OUT, "fig_hitrate_curve.svg"), xs,
               [{"label": "hit rate (Mattson = naive, diff 0.0)", "ys": matt}],
               "Hit rate vs cache capacity (EXP-04, 930 accesses)",
               "capacity (pages, log scale)", "hit rate",
               markers=[(122, "ws70=122"), (256, "knee=256")])

# --- Fig 2: EXP-06 locality gap (real JSON) ---
d06 = json.load(open(os.path.join(RES, "06.json")))
h = d06["H1_session_locality"]
svg_bars(os.path.join(OUT, "fig_locality_gap.svg"),
         ["sequential", "interleaved"],
         [h["sequential_hit"], h["interleaved_hit"]],
         "Session locality decides reuse (EXP-06, 64-page cache)",
         "token hit rate")

# --- Fig 3: EXP-05 live replay curve (real JSON) ---
d05 = json.load(open(os.path.join(RES, "05.json")))
c = d05["curve"]
xs5 = sorted(int(k) for k in c)
svg_line_chart(os.path.join(OUT, "fig_live_replay.svg"), xs5,
               [{"label": "live-scale replay hit rate", "ys": [c[str(x)] for x in xs5]}],
               "Live BurstGPT-scale replay (EXP-05, n=2000 live rows)",
               "capacity (pages, log scale)", "hit rate",
               markers=[(d05["working_set_80"], "ws80=404")])

print("wrote:", sorted(os.listdir(OUT)))
