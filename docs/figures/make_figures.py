"""Draws the README figures as static SVG, one light and one dark variant each.

    python docs/figures/make_figures.py

Palette: three categorical hues (blue = Model A, orange = Model B, aqua = ensemble; grey = reference), checked for
colour-vision-deficiency separation and contrast on both surfaces. Text always uses the text colours.
"""
from pathlib import Path

OUT = Path(__file__).resolve().parent
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
THEMES = {
    "light": dict(bg="#fcfcfb", border="#e2e1dc", t1="#0b0b0b", t2="#52514e", grid="#e6e5e0", axis="#b9b8b2",
                  blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", fillA="#eaf2fd", fillB="#fdeee8", fillE="#e6f6f0", fillN="#f3f2ef", neutral="#8a8983"),
    "dark": dict(bg="#1a1a19", border="#33332f", t1="#ffffff", t2="#c3c2b7", grid="#2e2e2b", axis="#55544f",
                 blue="#3987e5", orange="#d95926", aqua="#199e70", fillA="#16273d", fillB="#3a2016", fillE="#12302a", fillN="#262624", neutral="#8f8e86"),
}


def svg(w, h, body, c, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}" '
            f'font-family="{FONT}">\n<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="10" fill="{c["bg"]}" stroke="{c["border"]}"/>\n{body}\n</svg>\n')


def text(x, y, s, c, size=13, weight=400, anchor="start", fill="t1"):
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{c[fill]}">{s}</text>'


def bar_right(x, y, w, h, color, r=4):
    """Horizontal bar anchored at x, rounded on the data end only."""
    w = max(w, r)
    return f'<path d="M{x},{y} h{w-r} a{r},{r} 0 0 1 {r},{r} v{h-2*r} a{r},{r} 0 0 1 -{r},{r} h-{w-r} z" fill="{color}"/>'


def header(c, title, sub):
    return text(28, 38, title, c, 17, 600) + "\n" + text(28, 60, sub, c, 13, 400, fill="t2")


# --------------------------------------------------------------------------- 1. pipeline
def pipeline(c):
    W, H = 980, 470
    b = [header(c, "How a record gets matched", "Two different matchers score every candidate pair. A third model combines them, then one rule decides.")]
    b.append(f'<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="{c["t2"]}"/></marker></defs>')

    def box(x, y, w, h, t, s, stroke, fill, s2=None):
        o = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{c[fill]}" stroke="{c[stroke]}" stroke-width="1.5"/>'
        o += text(x + w / 2, y + 26, t, c, 13.5, 600, "middle")
        o += text(x + w / 2, y + 46, s, c, 11.5, 400, "middle", "t2")
        if s2:
            o += text(x + w / 2, y + 62, s2, c, 11.5, 400, "middle", "t2")
        return o

    def arrow(x1, y1, x2, y2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{c["t2"]}" stroke-width="1.4" marker-end="url(#ar)"/>'

    def path(d):
        return f'<path d="{d}" fill="none" stroke="{c["t2"]}" stroke-width="1.4" marker-end="url(#ar)"/>'

    b.append(box(28, 205, 140, 80, "Raw records", "3 sources, 24 million", "axis", "fillN", "name, address, country"))
    ya, yb, bw, bh, gap = 110, 300, 166, 80, 20
    xs = [208 + i * (bw + gap) for i in range(3)]
    b.append(text(208, ya - 14, "MODEL A  reads the two records together", c, 11.5, 600, fill="t2"))
    b.append(text(208, yb + bh + 24, "MODEL B  measures the two records with engineered features", c, 11.5, 600, fill="t2"))
    A = [("Contrastive encoder", "multilingual-e5-small", "same business, close vectors"), ("Exact GPU search", "32 candidates per record", "99.65% of true pairs kept"),
         ("Cross-encoder judge", "XLM-RoBERTa + LightGBM", "probability per pair")]
    B = [("Learned normalisation", "script dictionary,", "legal forms, regions"), ("Keys + neighbours", "ranker keeps 12 per record", "99.43% of true pairs kept"),
         ("Two-pass LightGBM", "113 pair features", "probability per pair")]
    for i, (t, s, s2) in enumerate(A):
        b.append(box(xs[i], ya, bw, bh, t, s, "blue", "fillA", s2))
    for i, (t, s, s2) in enumerate(B):
        b.append(box(xs[i], yb, bw, bh, t, s, "orange", "fillB", s2))
    for y in (ya, yb):
        for i in range(2):
            b.append(arrow(xs[i] + bw, y + bh / 2, xs[i + 1] - 3, y + bh / 2))
    b.append(path(f"M168,230 C190,230 190,{ya + bh/2} {xs[0]-3},{ya + bh/2}"))
    b.append(path(f"M168,260 C190,260 190,{yb + bh/2} {xs[0]-3},{yb + bh/2}"))
    ex = xs[2] + bw + 42
    b.append(box(ex, 140, 170, 80, "Stacker", "LightGBM, 22 features", "aqua", "fillE", "both scores + context"))
    b.append(box(ex, 270, 170, 80, "Decision rule", "one owner per record", "aqua", "fillE", "score at least 0.70"))
    b.append(path(f"M{xs[2]+bw},{ya + bh/2} C{ex-22},{ya + bh/2} {ex-22},170 {ex-3},170"))
    b.append(path(f"M{xs[2]+bw},{yb + bh/2} C{ex-22},{yb + bh/2} {ex-22},192 {ex-3},192"))
    b.append(arrow(ex + 85, 220, ex + 85, 267))
    b.append(text(ex + 85, 386, "Matches per record", c, 13.5, 600, "middle"))
    b.append(text(ex + 85, 404, "or an empty list", c, 11.5, 400, "middle", "t2"))
    b.append(arrow(ex + 85, 350, ex + 85, 369))
    b.append(text(28, 448, "Every model is trained on four fifths of the labelled records. The remaining fifth is only scored, and the stacker learns from those scores.", c, 11.5, fill="t2"))
    return svg(W, H, "\n".join(b), c, "Pipeline: two matchers, a stacker and a decision rule")


# --------------------------------------------------------------------------- 2. blocking
def blocking(c):
    W, H = 980, 392
    rows = [("Keyword keys only", "12 candidates per record", 3.00, "orange"),
            ("Embedding search, approximate index", "33 candidates per record", 1.12, "blue"),
            ("Keys + embedding neighbours + ranker  (Model B)", "12 candidates per record", 0.57, "orange"),
            ("Embedding search, exact  (Model A)", "33 candidates per record", 0.35, "blue"),
            ("Union of Model A and Model B", "36 candidates per record", 0.19, "aqua")]
    x0, x1, y0, rh = 360, 900, 100, 46
    b = [header(c, "True matches lost before matching starts", "Share of true pairs missing from the candidate list, validation fold. Lower is better.")]
    for v in (0, 1, 2, 3):
        x = x0 + (x1 - x0) * v / 3.2
        b.append(f'<line x1="{x}" y1="{y0-8}" x2="{x}" y2="{y0 + rh*len(rows)}" stroke="{c["grid"] if v else c["axis"]}" stroke-width="1"/>')
        b.append(text(x, y0 + rh * len(rows) + 18, f"{v}%", c, 11.5, 400, "middle", "t2"))
    for i, (n, s, v, col) in enumerate(rows):
        y = y0 + i * rh
        b.append(text(x0 - 14, y + 15, n, c, 12.5, 500, "end"))
        b.append(text(x0 - 14, y + 31, s, c, 11, 400, "end", "t2"))
        w = (x1 - x0) * v / 3.2
        b.append(bar_right(x0, y + 8, w, 20, c[col]))
        b.append(text(x0 + w + 8, y + 23, f"{v:.2f}%", c, 12.5, 600))
    ly = y0 + rh * len(rows) + 44
    for i, (col, lab) in enumerate((("blue", "Model A"), ("orange", "Model B"), ("aqua", "Both"))):
        b.append(f'<rect x="{x0 + i*110}" y="{ly-10}" width="12" height="12" rx="3" fill="{c[col]}"/>')
        b.append(text(x0 + i * 110 + 18, ly, lab, c, 12, fill="t2"))
    return svg(W, H, "\n".join(b), c, "Share of true pairs missed by each candidate generation method")


# --------------------------------------------------------------------------- 3. scores
def scores(c):
    W, H = 980, 330
    rows = [("Model A", "cross-encoder pipeline", 0.9824, None), ("Model B", "feature pipeline", 0.9883, 0.9801), ("Stacked ensemble", "both models combined", 0.9904, 0.9866)]
    x0, x1, y0, rh, lo, hi = 250, 900, 108, 52, 0.978, 0.992
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    b = [header(c, "Macro F0.5 by model", "One score per Source 1 record, averaged. A wrong merge costs more than a missed match. Higher is better.")]
    for v in (0.980, 0.984, 0.988, 0.992):
        b.append(f'<line x1="{sx(v)}" y1="{y0-10}" x2="{sx(v)}" y2="{y0 + rh*len(rows) - 8}" stroke="{c["grid"]}" stroke-width="1"/>')
        b.append(text(sx(v), y0 + rh * len(rows) + 10, f"{v:.3f}", c, 11.5, 400, "middle", "t2"))
    for i, (n, s, val, tst) in enumerate(rows):
        y = y0 + i * rh + 14
        b.append(text(x0 - 22, y - 1, n, c, 12.5, 500, "end"))
        b.append(text(x0 - 22, y + 15, s, c, 11, 400, "end", "t2"))
        if tst is not None:
            b.append(f'<line x1="{sx(tst)}" y1="{y+2}" x2="{sx(val)}" y2="{y+2}" stroke="{c["axis"]}" stroke-width="2"/>')
            b.append(f'<circle cx="{sx(tst)}" cy="{y+2}" r="6" fill="{c["orange"]}" stroke="{c["bg"]}" stroke-width="2"/>')
            b.append(text(sx(tst), y - 12, f"{tst:.4f}", c, 12, 600, "middle"))
        else:
            b.append(text(sx(0.9801), y + 6, "not measured", c, 11, 400, "middle", "t2"))
        b.append(f'<circle cx="{sx(val)}" cy="{y+2}" r="6" fill="{c["blue"]}" stroke="{c["bg"]}" stroke-width="2"/>')
        b.append(text(sx(val), y - 12, f"{val:.4f}", c, 12, 600, "middle"))
    ly = y0 + rh * len(rows) + 42
    b.append(f'<circle cx="{x0+6}" cy="{ly-4}" r="6" fill="{c["blue"]}"/>' + text(x0 + 20, ly, "Validation fold, 442,001 records, labels known", c, 12, fill="t2"))
    b.append(f'<circle cx="{x0+346}" cy="{ly-4}" r="6" fill="{c["orange"]}"/>' + text(x0 + 360, ly, "Test set, 1,732,544 records, labels hidden", c, 12, fill="t2"))
    return svg(W, H, "\n".join(b), c, "Macro F0.5 of each model on the validation fold and on the hidden test set")


# --------------------------------------------------------------------------- 4. errors
def errors(c):
    W, H = 980, 330
    names = ["Model A", "Model B", "Average of A and B", "Stacked ensemble"]
    cols = ["blue", "orange", "neutral", "aqua"]
    panels = [("Wrong merges", "pairs accepted that are not matches", [9781, 2782, 2839, 1807], 10000),
              ("Missed matches", "true pairs that were not returned", [59289, 46090, 46538, 41941], 60000)]
    b = [header(c, "Errors on the validation fold", "1,530,716 true pairs across 442,001 records. Lower is better.")]
    for p, (t, s, vals, mx) in enumerate(panels):
        px = 28 + p * 476
        x0, x1, y0, rh = px + 150, px + 390, 128, 40
        b.append(text(px, 96, t, c, 13.5, 600) + text(px + 8 + len(t) * 8.3, 96, s, c, 11.5, 400, fill="t2"))
        b.append(f'<line x1="{x0}" y1="{y0-6}" x2="{x0}" y2="{y0 + rh*4}" stroke="{c["axis"]}" stroke-width="1"/>')
        for i, (n, v) in enumerate(zip(names, vals)):
            y = y0 + i * rh
            b.append(text(x0 - 12, y + 21, n, c, 12.5, 500 if i == 3 else 400, "end"))
            w = (x1 - x0) * v / mx
            b.append(bar_right(x0, y + 7, w, 20, c[cols[i]]))
            b.append(text(x0 + w + 8, y + 22, f"{v:,}", c, 12.5, 600))
    b.append(text(28, 312, "The average of the two scores barely helps. The stacker learns where each model can be trusted, and cuts both kinds of error.", c, 11.5, fill="t2"))
    return svg(W, H, "\n".join(b), c, "Wrong merges and missed matches per model on the validation fold")


if __name__ == "__main__":
    for name, fn in (("pipeline", pipeline), ("blocking", blocking), ("scores", scores), ("errors", errors)):
        for mode, c in THEMES.items():
            (OUT / f"{name}-{mode}.svg").write_text(fn(c))
    print("figures written to", OUT)
