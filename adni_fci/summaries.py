"""Turn bootstrap PAGs into tables of edge frequencies."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

# (mark at source, mark at target) -> edge, read left to right. Marks: -1 tail, 1 arrowhead, 2 circle.
EDGE_SYMBOLS = {(-1, 1): "-->", (2, 1): "o->", (1, 1): "<->", (2, 2): "o-o", (-1, -1): "---", (2, -1): "o--"}
# Mirror images are written with source and target swapped (A <-- B becomes B --> A).
_MIRRORED = {(1, -1), (1, 2), (-1, 2)}

EDGE_MEANINGS = {
    "-->": "A causes B (A is an ancestor of B, B is not an ancestor of A)",
    "<->": "neither causes the other: a latent confounder of A and B",
    "o->": "B does not cause A; A causes B and/or they share a latent confounder",
    "o-o": "adjacent, orientation undetermined",
}


def edge_frequencies(pags: np.ndarray, columns) -> pd.DataFrame:
    """How often each edge type links each pair of variables across bootstrap PAGs.

    Returns one row per (pair, edge type) with columns ``source, edge, target, count,
    probability, p_adjacent`` (``p_adjacent``: share of bootstraps with any edge between the pair).
    """
    columns = list(columns)
    n_boot, p, _ = pags.shape
    rows = []
    for i in range(p):
        for j in range(i + 1, p):
            marks = Counter(zip(pags[:, i, j].tolist(), pags[:, j, i].tolist()))
            p_adjacent = 1 - marks.pop((0, 0), 0) / n_boot
            for (mark_i, mark_j), count in marks.items():
                source, target = columns[i], columns[j]
                if (mark_i, mark_j) in _MIRRORED:
                    mark_i, mark_j, source, target = mark_j, mark_i, target, source
                rows.append({"source": source, "edge": EDGE_SYMBOLS.get((mark_i, mark_j), f"{mark_i}/{mark_j}"),
                             "target": target, "count": count, "probability": count / n_boot,
                             "p_adjacent": p_adjacent})
    edges = pd.DataFrame(rows, columns=["source", "edge", "target", "count", "probability", "p_adjacent"])
    return edges.sort_values(["probability", "p_adjacent"], ascending=False, kind="stable").reset_index(drop=True)


def format_edges(edges: pd.DataFrame, min_prob: float = 0.0) -> str:
    """Edges at or above ``min_prob`` as aligned text, strongest first."""
    shown = edges[edges["probability"] >= min_prob]
    if shown.empty:
        return f"  (no edges with frequency >= {min_prob})"
    w_src, w_tgt = shown["source"].str.len().max(), shown["target"].str.len().max()
    return "\n".join(f"  {e.source:<{w_src}}  {e.edge}  {e.target:<{w_tgt}}  {e.probability:.2f}"
                     for e in shown.itertuples())


def compare_edge_tables(tables: dict[str, pd.DataFrame], min_prob: float = 0.3) -> pd.DataFrame:
    """Edge probabilities side by side, e.g. across alphas, diagnostic groups or settings.

    Keeps edges reaching ``min_prob`` in at least one table; an edge absent from a table
    counts as probability 0.
    """
    if not tables:
        return pd.DataFrame()
    wide = pd.concat({name: t.set_index(["source", "edge", "target"])["probability"]
                      for name, t in tables.items()}, axis=1).fillna(0.0)
    strongest = wide.max(axis=1)
    return wide[strongest >= min_prob].loc[strongest[strongest >= min_prob].sort_values(ascending=False).index]


def endpoint_frequencies(pags: np.ndarray, columns) -> dict[str, pd.DataFrame]:
    """Per mark type, P(that mark sits at the column variable's end of the row–column edge).

    For example ``endpoint_frequencies(...)["arrowhead"].loc[A, B]`` is P(A *-> B).
    """
    columns = list(columns)
    # pags[:, i, j] is the mark at i, so the mark at the column variable j is pags[:, j, i].
    return {name: pd.DataFrame((pags == code).mean(axis=0).T, index=columns, columns=columns)
            for name, code in (("arrowhead", 1), ("tail", -1), ("circle", 2))}


def sepset_summary(sepsets: list[dict], columns, biomarkers, target: str) -> pd.DataFrame:
    """How often FCI separated each biomarker from ``target``, and by which variables.

    One row per biomarker: ``p_separated`` is the share of bootstraps in which the pair is
    non-adjacent; every other column is P(variable in the separating set | separated).
    """
    columns = list(columns)
    t = columns.index(target)
    rows = {}
    for biomarker in biomarkers:
        if biomarker == target or biomarker not in columns:
            continue
        i = columns.index(biomarker)
        key = (min(i, t), max(i, t))
        found = [s[key] for s in sepsets if key in s]
        row = {"p_separated": len(found) / len(sepsets)}
        for k, name in enumerate(columns):
            if k not in (i, t):
                row[name] = float(np.mean([k in s for s in found])) if found else np.nan
        rows[biomarker] = row
    summary = pd.DataFrame.from_dict(rows, orient="index").rename_axis("biomarker")
    return summary.reindex(columns=["p_separated", *(c for c in columns if c in summary.columns)])
