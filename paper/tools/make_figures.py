"""Generate the manuscript figures as SVG, using only the standard library.

Figure 1 is the 4.0 stage graph (haplotype_reconstruction/stages/graph.py).
Figure 2 is drawn from the "Kernel ladders" table of docs/developer/costs.md, so it
always shows the slopes recorded there; nothing is measured or typed in here.

Run from anywhere: python3 paper/tools/make_figures.py
"""

from __future__ import annotations

import re
from pathlib import Path
from xml.sax.saxutils import escape

PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
FIGURES = PAPER / "figures"
COSTS = REPO / "docs" / "developer" / "costs.md"

FONT = "DejaVu Sans, Helvetica, Arial, sans-serif"
INK = "#1a1a1a"
MUTED = "#5f6368"
CONTIG_FILL, CONTIG_EDGE = "#e8f0fb", "#2f5d9e"
GENOME_FILL, GENOME_EDGE = "#fdf0e3", "#b8621b"
INPUT_FILL, INPUT_EDGE = "#f1f3f4", "#5f6368"
CUBIC, EXEMPT = "#2f5d9e", "#c0571b"


def svg(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="{FONT}">\n'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK}"/>'
        "</marker></defs>\n"
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n' + "\n".join(body) + "\n</svg>\n"
    )


def text(x: float, y: float, s: str, size: float = 11, weight: str = "normal", anchor: str = "middle",
         fill: str = INK, style: str = "normal") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
            f'font-style="{style}" text-anchor="{anchor}" fill="{fill}">{escape(s)}</text>')


def box(x: float, y: float, w: float, h: float, title: str, lines: list[str], fill: str, edge: str) -> list[str]:
    out = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{fill}" stroke="{edge}" stroke-width="1.4"/>']
    top = y + h / 2 - (len(lines) * 12) / 2 + (2 if lines else 4)
    out.append(text(x + w / 2, top, title, 11.5, "bold"))
    for i, line in enumerate(lines):
        out.append(text(x + w / 2, top + 14 + 12 * i, line, 9.5, fill=MUTED))
    return out


def path(points: list[tuple[float, float]], dashed: bool = False) -> str:
    d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in points)
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    return f'<path d="{d}" fill="none" stroke="{INK}" stroke-width="1.3"{dash} marker-end="url(#arrow)"/>'


def figure_pipeline() -> str:
    width, height = 990, 425
    body: list[str] = []
    bands = [(8, 142, "Per contig (contigs run concurrently)", CONTIG_FILL),
             (156, 266, "Genome-wide", GENOME_FILL),
             (280, 415, "Per contig", CONTIG_FILL)]
    for top, bottom, label, fill in bands:
        body.append(f'<rect x="8" y="{top}" width="{width - 16}" height="{bottom - top}" rx="4" '
                    f'fill="{fill}" fill-opacity="0.35" stroke="none"/>')
        body.append(text(16, top + 14, label, 10, "bold", "start", MUTED, "italic"))

    y, h = 40, 82
    body += box(16, y, 118, h, "Inputs", ["VCF/BCF with", "allele depths (AD);", "genetic map"], INPUT_FILL, INPUT_EDGE)
    body += box(150, y, 128, h, "Evidence", ["read model,", "genotype likelihoods,", "site and sample QC"],
                CONTIG_FILL, CONTIG_EDGE)
    body += box(294, y, 120, h, "Discovery", ["local founder", "panels in blocks", "(missing-aware)"],
                CONTIG_FILL, CONTIG_EDGE)
    body += box(430, y, 150, h, "Local selection", ["rounds initial, L1, L2,", "each fed by an", "assembly context"],
                CONTIG_FILL, CONTIG_EDGE)
    body += box(596, y, 138, h, "Assembly", ["joins, founder count,", "completion,", "refinement"],
                CONTIG_FILL, CONTIG_EDGE)
    body += box(750, y, 106, h, "Painting", ["diploid founder", "mosaics"], CONTIG_FILL, CONTIG_EDGE)
    body += box(872, y, 100, h, "Refit", ["founder alleles", "from paintings"], CONTIG_FILL, CONTIG_EDGE)

    yb, hb = 182, 64
    body += box(150, yb, 220, hb, "Relatedness", ["pairwise IBD, full-sib families", "from every contig's evidence"],
                GENOME_FILL, GENOME_EDGE)
    body += box(596, yb, 260, hb, "Pedigree", ["every sample a candidate parent;", "sibling guard, release tiers"],
                GENOME_FILL, GENOME_EDGE)

    yc, hc = 316, 78
    body += box(560, yc, 150, hc, "Family phase", ["phase conditional on", "the pedigree;", "meiosis chains"],
                CONTIG_FILL, CONTIG_EDGE)
    body += box(726, yc, 124, hc, "Recombination", ["maps from", "the final phase"], CONTIG_FILL, CONTIG_EDGE)
    body += box(866, yc, 106, hc, "Export", ["founder and", "phased-sample", "VCFs"], CONTIG_FILL, CONTIG_EDGE)

    mid = y + h / 2
    for x0, x1 in ((134, 150), (278, 294), (414, 430), (580, 596), (734, 750), (856, 872)):
        body.append(path([(x0, mid), (x1 - 1, mid)]))
    body.append(path([(214, y + h), (214, yb - 1)]))                          # evidence -> relatedness
    body.append(path([(370, yb + hb / 2), (595, yb + hb / 2)]))               # relatedness -> pedigree
    body.append(path([(803, y + h), (803, yb - 1)]))                          # painting -> pedigree
    body.append(path([(640, yb + hb), (640, yc - 1)]))                        # pedigree -> phase
    body.append(path([(788, yb + hb), (788, yc - 1)]))                        # pedigree -> recombination
    body.append(path([(940, y + h), (940, 288), (690, 288), (690, yc - 1)]))  # refit -> phase
    for x0, x1 in ((710, 726), (850, 866)):
        body.append(path([(x0, yc + hc / 2), (x1 - 1, yc + hc / 2)]))
    body.append(text(width - 16, height - 6,
                     "The evidence of each contig also feeds pedigree and family phase; arrows omitted for clarity.",
                     9, anchor="end", fill=MUTED, style="italic"))
    return svg(width, height, body)


LADDER_ROW = re.compile(
    r"^\| (?P<stage>[\w.]+) \| `(?P<probe>[\w.]+)` \| (?P<what>.*?) \| "
    r"(?P<k>-?\d+\.\d+) ± (?P<kse>\d+\.\d+) over [^(]*\((?P<kgate>[^)]*)\) \| "
    r"(?P<n>-?\d+\.\d+) ± (?P<nse>\d+\.\d+) over [^(]*\((?P<ngate>[^)]*)\) \|$"
)


def read_ladders() -> list[dict[str, object]]:
    rows = []
    in_table = False
    for line in COSTS.read_text(encoding="utf-8").splitlines():
        if line.startswith("## Kernel ladders"):
            in_table = True
            continue
        if in_table and line.startswith("## "):
            break
        match = LADDER_ROW.match(line) if in_table else None
        if match and match["kgate"] != "size proxy":
            rows.append(dict(stage=match["stage"], probe=match["probe"], k=float(match["k"]),
                             kse=float(match["kse"]), n=float(match["n"]), nse=float(match["nse"]),
                             exempt=match["kgate"] == "exempt"))
    if not rows:
        raise SystemExit(f"no kernel-ladder rows parsed from {COSTS}")
    return rows


def figure_scaling(rows: list[dict[str, object]]) -> str:
    row_h, top, label_w = 15, 24, 300
    panel_w, gap = 285, 50
    height = top + row_h * len(rows) + 58
    width = label_w + 2 * panel_w + gap + 20
    panels = [("k", "kse", "Fitted K exponent (K = 5–30, N = 200)", -0.5, 4.5, [1, 2, 3], 3.3, [0, 1, 2, 3, 4]),
              ("n", "nse", "Fitted N exponent (N = 250–2,500, K = 15)", -0.5, 2.5, [1, 2], None,
               [0, 0.5, 1, 1.5, 2, 2.5])]
    body: list[str] = []
    bottom = top + row_h * len(rows)
    for i, row in enumerate(rows):
        yy = top + row_h * i + row_h / 2
        if i % 2 == 0:
            body.append(f'<rect x="8" y="{yy - row_h / 2:.1f}" width="{width - 16}" height="{row_h}" '
                        'fill="#f4f6f8"/>')
        body.append(text(label_w - 8, yy + 3.5, str(row["probe"]), 9.2, anchor="end"))
    for p, (key, se, title, lo, hi, refs, limit, ticks) in enumerate(panels):
        x0 = label_w + p * (panel_w + gap)

        def sx(v: float, x0: float = x0, lo: float = lo, hi: float = hi) -> float:
            return x0 + (v - lo) / (hi - lo) * panel_w

        body.append(f'<rect x="{x0}" y="{top}" width="{panel_w}" height="{bottom - top}" fill="none" '
                    f'stroke="{MUTED}" stroke-width="0.8"/>')
        for tick in ticks:
            xt = sx(tick)
            body.append(f'<line x1="{xt:.1f}" y1="{bottom}" x2="{xt:.1f}" y2="{bottom + 4}" stroke="{MUTED}"/>')
            body.append(text(xt, bottom + 15, f"{tick:g}", 9, fill=MUTED))
        for ref in refs:
            body.append(f'<line x1="{sx(ref):.1f}" y1="{top}" x2="{sx(ref):.1f}" y2="{bottom}" '
                        'stroke="#c7ccd1" stroke-width="0.9"/>')
        if limit is not None:
            body.append(f'<line x1="{sx(limit):.1f}" y1="{top}" x2="{sx(limit):.1f}" y2="{bottom}" '
                        f'stroke="{EXEMPT}" stroke-width="1" stroke-dasharray="4,3"/>')
            body.append(text(sx(limit) + 3, top - 4, "3.3 gate", 8.5, anchor="start", fill=EXEMPT))
        body.append(text(x0 + panel_w / 2, bottom + 32, title, 10, "bold"))
        for i, row in enumerate(rows):
            yy = top + row_h * i + row_h / 2
            v, e = float(row[key]), float(row[se])
            colour = EXEMPT if row["exempt"] else CUBIC
            body.append(f'<line x1="{sx(max(lo, v - e)):.1f}" y1="{yy:.1f}" x2="{sx(min(hi, v + e)):.1f}" '
                        f'y2="{yy:.1f}" stroke="{colour}" stroke-width="1.4"/>')
            body.append(f'<circle cx="{sx(v):.1f}" cy="{yy:.1f}" r="3.2" fill="{colour}"/>')
    legend_y = height - 10
    body.append(f'<circle cx="{label_w + 6}" cy="{legend_y - 3.5}" r="3.2" fill="{CUBIC}"/>')
    body.append(text(label_w + 14, legend_y, "kernel gated at most cubic in K", 9.5, anchor="start"))
    body.append(f'<circle cx="{label_w + 236}" cy="{legend_y - 3.5}" r="3.2" fill="{EXEMPT}"/>')
    body.append(text(label_w + 244, legend_y, "kernel under a decided cost exemption (0, 0b, 0c, 0d)", 9.5,
                     anchor="start"))
    return svg(int(width), int(height), body)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "fig1_pipeline.svg").write_text(figure_pipeline(), encoding="utf-8")
    rows = read_ladders()
    (FIGURES / "fig2_scaling.svg").write_text(figure_scaling(rows), encoding="utf-8")
    print(f"figures: fig1_pipeline.svg, fig2_scaling.svg ({len(rows)} probed kernels)")


if __name__ == "__main__":
    main()
