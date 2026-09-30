"""Draw the analysis-pipeline graphic for slides (16:9): docs/pipeline.png, .svg and .pdf.

    python docs/pipeline_figure.py

Cohort size, bootstrap count, alpha and thresholds are read from the `main` setting, so the
graphic always matches the code. Colour means data modality and nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from adni_fci import PRESETS, build_cohort  # noqa: E402

plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42})

W, H = 13.333, 7.5  # inches; the axes use inch units, so font sizes are true slide points
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
SURFACE, CARD, BORDER, FAINT = "#ffffff", "#f7f7f5", "#dddcd5", "#c3c2b7"
MRI, FLUID, COGNITION, DEMOGRAPHIC = "#2a78d6", "#eb6834", "#1baf7a", "#a09e96"

# Schematic causal graph used in steps 4 and 5: name -> (x, y in the unit square, colour).
NODES = {
    "age": (0.05, 0.62, DEMOGRAPHIC), "apoe": (0.05, 0.10, DEMOGRAPHIC),
    "nfl": (0.40, 0.95, FLUID), "ptau": (0.46, 0.48, FLUID), "abeta": (0.38, 0.05, FLUID),
    "hippo": (0.74, 0.82, MRI), "cog": (0.96, 0.40, COGNITION),
}
ONE_RUN = [("age", "nfl"), ("age", "hippo"), ("apoe", "abeta"), ("abeta", "ptau"),
           ("ptau", "hippo"), ("ptau", "cog"), ("hippo", "cog")]
CONSENSUS = [("abeta", "cog", 0.25), ("apoe", "ptau", 0.3), ("age", "nfl", 1.0), ("age", "hippo", 1.0),
             ("apoe", "abeta", 1.0), ("abeta", "ptau", 0.55), ("ptau", "hippo", 0.7), ("ptau", "cog", 1.0),
             ("hippo", "cog", 1.0)]


def text(ax, x, y, s, size=11.5, color=INK_2, weight="normal", ha="left", va="baseline", **kw):
    return ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=6, **kw)


def box(ax, x, y, w, h, fill=CARD, edge=BORDER, radius=0.12, lw=0.9, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=fill, ec=edge, lw=lw, zorder=z))


def arrow(ax, start, end, color=INK_2, lw=1.8, head=13, shrink=0.0, z=3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
                                 shrinkA=shrink, shrinkB=shrink, zorder=z, capstyle="round", joinstyle="round"))


def graph(ax, x, y, w, h, edges, radius=0.07):
    """Schematic causal graph in the box (x, y, w, h); edges are (source, target[, frequency])."""
    at = {name: (x + nx * w, y + ny * h) for name, (nx, ny, _) in NODES.items()}
    for source, target, *freq in edges:
        freq = freq[0] if freq else None
        stable = freq is None or freq >= 0.5
        arrow(ax, at[source], at[target], color=INK if stable else FAINT,
              lw=1.3 if freq is None else 0.7 + 2.2 * freq, head=7.5 if freq is None else 6 + 4 * freq,
              shrink=radius * 72 + 1.5, z=3 if stable else 2)
    for name, (_, _, color) in NODES.items():
        ax.add_patch(Circle(at[name], radius, fc=color, ec=SURFACE, lw=1.4, zorder=4))


def card(ax, x, y, w, h, step, title):
    box(ax, x, y, w, h)
    text(ax, x + 0.22, y + h - 0.36, str(step), 11, MUTED, "bold")
    text(ax, x + 0.22, y + h - 0.72, title, 16, INK, "bold")


def pills(ax, x, y, labels, size=11, height=0.38, gap=0.12, pad=0.17):
    """A row of rounded tags starting at x, vertically centred on y."""
    renderer = ax.figure.canvas.get_renderer()
    for label in labels:
        t = text(ax, 0, y, label, size, INK_2, va="center_baseline")
        width = t.get_window_extent(renderer).width / ax.figure.dpi
        box(ax, x, y - height / 2, width + 2 * pad, height, fill=SURFACE, radius=height / 2, lw=0.9)
        t.set_x(x + pad)
        x += width + 2 * pad + gap


def main() -> None:
    setting = PRESETS["main"]
    cohort = build_cohort(setting)
    groups = cohort["DX_GROUP"].value_counts()
    n_boot, alpha = setting.n_bootstraps, f"{setting.alpha:g}"

    fig = plt.figure(figsize=(W, H), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(0, H))
    ax.axis("off")

    y0, h, gap = 2.22, 4.72, 0.34
    widths = [2.50, 1.94, 1.72, 2.85, 2.12]
    lefts = np.cumsum([0.42, *(w + gap for w in widths[:-1])])
    top = y0 + h
    for left in lefts[1:]:  # flow arrows in the gaps between cards
        arrow(ax, (left - gap + 0.03, y0 + h / 2), (left - 0.03, y0 + h / 2))

    # 1  Data ---------------------------------------------------------------------------
    x, w = lefts[0], widths[0]
    card(ax, x, y0, w, h, 1, "ADNI baseline data")
    modalities = [(MRI, "MRI volumes", ["ICV, hippocampus"]),
                  (FLUID, "Plasma biomarkers", ["Aβ42/40, pTau217,", "NfL, GFAP"]),
                  (COGNITION, "Cognition", ["ADAS-Cog13"]),
                  (DEMOGRAPHIC, "Demographics", ["age, sex, education,", "APOE4"])]
    y = top - 1.30
    for color, name, lines in modalities:
        ax.add_patch(Circle((x + 0.32, y + 0.055), 0.08, fc=color, ec="none", zorder=4))
        text(ax, x + 0.52, y, name, 12.5, INK, "bold")
        for i, line in enumerate(lines, start=1):
            text(ax, x + 0.52, y - 0.245 * i, line, 11.5)
        y -= 0.40 + 0.245 * len(lines)
    text(ax, x + 0.22, y0 + 0.28, f"{len(setting.variables)} variables", 11.5, MUTED)

    # 2  Cohort -------------------------------------------------------------------------
    x, w = lefts[1], widths[1]
    card(ax, x, y0, w, h, 2, "Cohort")
    columns = [MRI] * 2 + [FLUID] * 4 + [COGNITION] + [DEMOGRAPHIC] * 5  # the 12 variables
    cell = (w - 0.44) / len(columns)
    table_top = top - 1.12
    for j, color in enumerate(columns):
        ax.add_patch(Rectangle((x + 0.22 + j * cell, table_top - 0.12), cell - 0.02, 0.12, fc=color, ec="none", zorder=3))
        for i in range(5):  # participants, fading out
            ax.add_patch(Rectangle((x + 0.22 + j * cell, table_top - 0.32 - 0.17 * i), cell - 0.02, 0.13,
                                   fc=FAINT, ec="none", alpha=0.75 - 0.14 * i, zorder=3))
    text(ax, x + 0.22, top - 2.80, f"{len(cohort)}", 34, INK, "bold")
    text(ax, x + 0.22, top - 3.08, "participants", 12)
    for k, group in enumerate(("CN", "MCI", "AD")):
        text(ax, x + 0.22 + k * 0.56, top - 3.52, str(groups.get(group, 0)), 13, INK, "bold")
        text(ax, x + 0.22 + k * 0.56, top - 3.73, group, 10.5, MUTED)
    text(ax, x + 0.22, y0 + 0.50, f"MRI within ±{setting.match_window_months} months", 11.5)
    text(ax, x + 0.22, y0 + 0.26, "of the plasma sample", 11.5)

    # 3  Normalise ----------------------------------------------------------------------
    x, w = lefts[2], widths[2]
    card(ax, x, y0, w, h, 3, "Normalise")
    t = np.linspace(0, 1, 80)
    skewed = t ** 1.1 * np.exp(-7.5 * t)
    bell = np.exp(-((t - 0.5) ** 2) / (2 * 0.17 ** 2))
    base = top - 2.35
    cw = 0.45  # width of each density curve
    for offset, density in ((0.22, skewed), (w - 0.22 - cw, bell)):
        points = np.column_stack([x + offset + cw * t, base + 0.95 * density / density.max()])
        ax.add_patch(Polygon(np.vstack([[x + offset, base], points, [x + offset + cw, base]]),
                             closed=True, fc=FAINT, ec=INK_2, lw=1.3, alpha=0.9, zorder=3, joinstyle="round"))
    arrow(ax, (x + 0.22 + cw + 0.04, base + 0.36), (x + w - 0.22 - cw - 0.04, base + 0.36), lw=1.5, head=11)
    for i, line in enumerate(("Yeo-Johnson for", "skewed variables,", "then z-scores")):
        text(ax, x + 0.22, top - 2.95 - 0.25 * i, line, 12)
    text(ax, x + 0.22, y0 + 0.50, "Binary variables", 11.5, MUTED)
    text(ax, x + 0.22, y0 + 0.26, "stay 0/1", 11.5, MUTED)

    # 4  Bootstrapped FCI ---------------------------------------------------------------
    x, w = lefts[3], widths[3]
    card(ax, x, y0, w, h, 4, "Bootstrapped FCI")
    gx, gy, gw, gh = x + 0.22, top - 2.42, 1.34, 1.12
    for k in (2, 1, 0):  # a stack of graphs, one per bootstrap
        box(ax, gx + 0.08 * k, gy + 0.08 * k, gw, gh, fill=SURFACE, radius=0.08, z=2)
    graph(ax, gx + 0.14, gy + 0.15, gw - 0.28, gh - 0.30, ONE_RUN, radius=0.055)
    text(ax, x + w - 0.22, top - 1.70, f"× {n_boot}", 22, INK, "bold", ha="right")
    text(ax, x + w - 0.22, top - 1.97, "bootstrap", 11.5, ha="right")
    text(ax, x + w - 0.22, top - 2.20, "resamples", 11.5, ha="right")
    for i, line in enumerate(("Resample participants", "Shuffle variable order", f"FCI with KCI test, α = {alpha}")):
        text(ax, x + 0.22, top - 2.95 - 0.25 * i, line, 12)
    box(ax, x + 0.16, y0 + 0.16, w - 0.32, 0.92, fill=SURFACE, radius=0.08)
    text(ax, x + 0.30, y0 + 0.80, "Prior knowledge", 11.5, INK, "bold")
    text(ax, x + 0.30, y0 + 0.54, "Demographics are root nodes", 11.5)
    text(ax, x + 0.30, y0 + 0.30, "Cognition is a sink node", 11.5)

    # 5  Edge stability -----------------------------------------------------------------
    x, w = lefts[4], widths[4]
    card(ax, x, y0, w, h, 5, "Edge stability")
    graph(ax, x + 0.28, top - 2.42, w - 0.56, 1.30, CONSENSUS, radius=0.075)
    for i, line in enumerate(("Frequency of each", "edge across the", f"{n_boot} graphs")):
        text(ax, x + 0.22, top - 2.95 - 0.25 * i, line, 12)
    for i, (lw, color, label) in enumerate(((2.9, INK, f"stable: ≥ {setting.stable_edge_prob:g}"),
                                           (1.2, FAINT, "unstable"))):
        y = y0 + 0.78 - 0.26 * i
        ax.plot([x + 0.22, x + 0.55], [y + 0.05, y + 0.05], color=color, lw=lw, solid_capstyle="round", zorder=3)
        text(ax, x + 0.68, y, label, 11.5)
    text(ax, x + 0.22, y0 + 0.26, f"below {setting.min_edge_prob:g}: not shown", 11.5, MUTED)

    # Footer ----------------------------------------------------------------------------
    sens = ", ".join(f"{a:g}" for a in setting.sensitivity_alphas)
    text(ax, 0.42, 1.50, "Follow-up analyses", 12.5, INK, "bold", va="center_baseline")
    pills(ax, 2.72, 1.50, [f"Sensitivity to α ({sens})", "Separating sets: biomarker vs hippocampus",
                           "Stratified by diagnosis (CN, MCI, AD)"])
    text(ax, 0.42, 0.90, "Alternative settings", 12.5, INK, "bold", va="center_baseline")
    pills(ax, 2.72, 0.90, ["CSF instead of plasma", "Tau PET instead of pTau217", "Four cognitive tests",
                           "Amygdala volume added", "Fisher-z test"])

    out = Path(__file__).resolve().parent
    fig.savefig(out / "pipeline.png", dpi=300)
    fig.savefig(out / "pipeline.svg")
    fig.savefig(out / "pipeline.pdf")
    print(f"wrote {out / 'pipeline.png'}, .svg and .pdf")


if __name__ == "__main__":
    main()
