"""Figures: PAGs (graphviz) and heatmaps (matplotlib)."""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydot
import seaborn as sns

from .knowledge import forbidden_edges
from .settings import Setting
from .summaries import endpoint_frequencies

DISPLAY_NAMES = {
    "Intracranial Volume": "ICV",
    "Hippocampus": "Hippocampal Volume",
    "Amygdala": "Amygdala Volume",
    "AB42_AB40_ratio": "Aβ42/40 Ratio",
    "pT217_F": "pTau217",
    "NfL_Q": "NfL",
    "GFAP_Q": "GFAP",
    "ASSAY_FUJIREBIO": "NfL/GFAP assay (Fujirebio=1)",
    "ABETA42_ABETA40_ratio": "CSF Aβ42/40 Ratio",
    "PTAU": "CSF pTau181",
    "META_TEMPORAL_SUVR": "Tau PET (meta-temporal)",
    "TOTAL13": "ADAS-Cog13",
    "MMSCORE": "MMSE",
    "MOCA": "MoCA",
    "TRABSCOR": "TMT-B (s)",
    "PTGENDER": "Sex (F=1, M=0)",
    "PTEDUCAT": "Education (years)",
    "AGE": "Age (years)",
    "APOE4_0": "APOE4 (0 copies)",
    "APOE4_1": "APOE4 (1 copy)",
}

# graphviz attributes per edge; dir="both" so that circle marks at both ends are drawn.
_EDGE_STYLE = {
    "-->": dict(style="solid", arrowtail="none", arrowhead="normal"),
    "o->": dict(style="dashed", arrowtail="odot", arrowhead="normal"),
    "<->": dict(style="solid", arrowtail="normal", arrowhead="normal"),
    "o-o": dict(style="dotted", arrowtail="odot", arrowhead="odot"),
    "---": dict(style="solid", arrowtail="none", arrowhead="none"),
    "o--": dict(style="dotted", arrowtail="odot", arrowhead="none"),
}
_LEGEND = "Edge labels: bootstrap frequency.  → cause   ↔ latent confounder   o→ / o–o orientation undetermined"


def label(column: str, display_names: bool = True) -> str:
    return DISPLAY_NAMES.get(column, column) if display_names else column


def _ensure_graphviz() -> None:
    """pydot needs graphviz's ``dot``; in a conda env that is not activated it sits next to python."""
    env_bin = Path(sys.executable).parent
    if shutil.which("dot") is None and (env_bin / "dot").exists():
        os.environ["PATH"] = f"{env_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def pag_graph(edges: pd.DataFrame, columns=None, min_prob: float = 0.1, display_names: bool = True,
              title: str | None = None, legend: bool = True) -> pydot.Dot:
    """Graphviz PAG of all edges with frequency >= ``min_prob``.

    Line width and labels show the bootstrap frequency. ``columns`` fixes the node set
    (so variables without edges still appear); by default it is taken from ``edges``.
    """
    graph = pydot.Dot(graph_type="digraph", rankdir="LR", fontsize="10")
    nodes = list(columns) if columns is not None else list(dict.fromkeys([*edges["source"], *edges["target"]]))
    for node in nodes:
        graph.add_node(pydot.Node(node, label=label(node, display_names), shape="ellipse"))
    for e in edges[edges["probability"] >= min_prob].itertuples():
        if e.edge in _EDGE_STYLE:
            graph.add_edge(pydot.Edge(e.source, e.target, dir="both", penwidth=f"{1 + 4 * e.probability:.2f}",
                                      label=f"{e.probability:.2f}", fontsize="8", **_EDGE_STYLE[e.edge]))
    caption = [text for text in (title, f"edges with frequency ≥ {min_prob:g}", _LEGEND if legend else None) if text]
    graph.set_label("\n".join(caption))
    graph.set_labelloc("t")
    return graph


def save_pag(edges: pd.DataFrame, path, **kwargs) -> Path:
    """Write :func:`pag_graph` to ``path``; the format follows the extension (.png, .pdf, .svg)."""
    _ensure_graphviz()
    path = Path(path)
    pag_graph(edges, **kwargs).write(str(path), format=path.suffix.lstrip(".") or "png")
    return path


def show_pag(edges: pd.DataFrame, **kwargs):
    """Display :func:`pag_graph` inline in a notebook."""
    from IPython.display import Image

    _ensure_graphviz()
    return Image(pag_graph(edges, **kwargs).create_png())


def _heatmap(ax, frame: pd.DataFrame, display_names: bool, **kwargs) -> None:
    sns.heatmap(frame.to_numpy(dtype=float), cmap="YlOrRd", vmin=0, vmax=1, annot=True, fmt=".2f", ax=ax,
                xticklabels=[label(c, display_names) for c in frame.columns],
                yticklabels=[label(str(r), display_names) for r in frame.index], **kwargs)


def plot_endpoint_heatmaps(pags: np.ndarray, columns, display_names: bool = True, path=None) -> plt.Figure:
    """Three heatmaps: P(arrowhead / tail / circle at the column variable's end of the row–column edge)."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    titles = {"arrowhead": "Arrowhead at column  (row *→ column)",
              "tail": "Tail at column  (row *— column)",
              "circle": "Circle at column  (row *–o column)"}
    for ax, (name, frame) in zip(axes, endpoint_frequencies(pags, columns).items()):
        _heatmap(ax, frame, display_names, annot_kws={"size": 7})
        ax.set_title(titles[name])
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_sepsets(summary: pd.DataFrame, target: str, display_names: bool = True, path=None) -> plt.Figure:
    """Heatmap of P(variable in separating set | biomarker separated from ``target``)."""
    members = summary.drop(columns="p_separated")
    fig, ax = plt.subplots(figsize=(1.1 * members.shape[1] + 3, 0.6 * len(members) + 2))
    _heatmap(ax, members, display_names, annot_kws={"size": 9})
    ax.set_yticklabels([f"{label(b, display_names)}  (separated {p:.0%})" for b, p in summary["p_separated"].items()],
                       rotation=0)
    ax.set_title(f"P(variable in separating set | biomarker ⊥ {label(target, display_names)})")
    ax.set_xlabel("Separating-set member")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_distributions(X: np.ndarray, columns, display_names: bool = True, bins: int = 25) -> plt.Figure:
    """Histograms of each (preprocessed) variable."""
    columns = list(columns)
    ncols = 6
    nrows = int(np.ceil(len(columns) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.4 * nrows), squeeze=False)
    for ax, (i, col) in zip(axes.flat, enumerate(columns)):
        ax.hist(X[:, i], bins=bins, edgecolor="black", alpha=0.7)
        ax.set_title(label(col, display_names), fontsize=9)
    for ax in axes.flat[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_background_knowledge(setting: Setting, display_names: bool = True) -> plt.Figure:
    """Which directed edges background knowledge forbids (row = cause, column = effect)."""
    columns = setting.variables
    forbidden = forbidden_edges(columns, setting)
    matrix = pd.DataFrame([[float((c, e) in forbidden) for e in columns] for c in columns],
                          index=columns, columns=columns)
    fig, ax = plt.subplots(figsize=(0.55 * len(columns) + 3, 0.5 * len(columns) + 2))
    sns.heatmap(matrix, cmap="Greys", vmin=0, vmax=1.6, cbar=False, linewidths=0.5, linecolor="lightgrey", ax=ax,
                xticklabels=[label(c, display_names) for c in columns],
                yticklabels=[label(c, display_names) for c in columns])
    ax.set_xlabel("effect")
    ax.set_ylabel("cause")
    ax.set_title(f"Forbidden edges in '{setting.name}' (dark = cause → effect ruled out)")
    fig.tight_layout()
    return fig
