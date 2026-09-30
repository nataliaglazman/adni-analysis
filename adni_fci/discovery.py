"""Bootstrapped FCI.

Each bootstrap resamples subjects with replacement and shuffles the column order (FCI's
output can depend on variable order). Bootstrap ``b`` draws from the random stream
``(seed, stream, b)``, so results do not depend on ``n_jobs``, and every alpha of a
sensitivity analysis is evaluated on identical resamples.
"""
from __future__ import annotations

import contextlib
import io
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import causallearn.search.ConstraintBased.FCI as _fci_module
import numpy as np
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import CIT_Base
from joblib import Parallel, delayed
from tqdm import tqdm

from .knowledge import build_background_knowledge, make_ci_test
from .settings import Setting

# causal-learn iterates over sets of nodes (e.g. in the possible-D-sep search) and GraphNode
# hashes its name, which Python salts per process. FCI's output therefore depended on the
# process (~5% of bootstrap PAGs changed between runs despite fixed seeds). A fixed hash
# makes runs exactly reproducible.
GraphNode.__hash__ = lambda node: zlib.crc32(node.name.encode())


@contextlib.contextmanager
def _fci_hooks(captured_sepsets: list):
    """Let causal-learn's ``fci()`` take a ready-made CI test and expose its separating sets.

    ``fci()`` always builds its own test from a string via ``CIT()``, but we need to pass a
    RestrictedCIT and reuse one p-value cache across alphas. It also discards the separating
    sets; we keep the dict returned by ``fas()``, which the later possible-D-sep step
    updates in place, so it holds FCI's final separating sets.
    """
    original_cit, original_fas = _fci_module.CIT, _fci_module.fas

    def cit(data, method="fisherz", **kwargs):
        return method if isinstance(method, CIT_Base) else original_cit(data, method, **kwargs)

    def fas(*args, **kwargs):
        result = original_fas(*args, **kwargs)
        captured_sepsets.append(result[1])
        return result

    _fci_module.CIT, _fci_module.fas = cit, fas
    try:
        yield
    finally:
        _fci_module.CIT, _fci_module.fas = original_cit, original_fas


def run_fci(data: np.ndarray, ci_test, alpha: float, background_knowledge=None):
    """One FCI run on ``data`` with a CI-test object (or causal-learn test name).

    Returns the PAG endpoint matrix (``G[i, j]`` is the mark at i on the i–j edge:
    -1 tail, 1 arrowhead, 2 circle, 0 no edge) and the separating sets
    ``{(i, j): {k, ...}}`` of non-adjacent pairs.
    """
    captured: list = []
    with _fci_hooks(captured), contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # spurious numpy matmul warnings inside KCI
        graph, _ = _fci_module.fci(data, independence_test_method=ci_test, alpha=float(alpha),
                                   background_knowledge=background_knowledge, show_progress=False)
    return graph.graph.astype(np.int8), captured[0]


def _draw_rows(data: np.ndarray, rng: np.random.Generator, max_tries: int = 100):
    """Row indices of one bootstrap resample, redrawing resamples with redundant variables.

    In a small group a resample can make a variable constant or an exact combination of
    others: without any APOE4 homozygote, ``APOE4_0 == 1 - APOE4_1``. CI tests are undefined
    on such data (Fisher-z raises), so the resample is drawn again from the same stream.
    Returns the rows and the number of redraws.
    """
    n, p = data.shape
    for redraws in range(max_tries):
        rows = rng.integers(0, n, size=n)
        sample = data[rows]
        if sample.std(axis=0).min() > 0 and np.linalg.matrix_rank(np.corrcoef(sample.T)) == p:
            return rows, redraws
    raise ValueError(f"No usable bootstrap resample in {max_tries} draws: too few subjects (n={n}) "
                     "for these variables.")


def _bootstrap_worker(data, columns, setting: Setting, alphas, b: int, stream: int):
    rng = np.random.default_rng([setting.seed, stream, b])
    p = data.shape[1]
    rows, redraws = _draw_rows(data, rng)
    order = rng.permutation(p)
    sample = data[np.ix_(rows, order)]
    names = [columns[k] for k in order]

    bk = build_background_knowledge(names, setting)
    ci_test = make_ci_test(sample, names, setting)  # shared across alphas, so p-values are reused
    # FCI records the conditioning set it asked for; drop variables a RestrictedCIT never conditioned on.
    never_conditioned = getattr(ci_test, "forbidden", frozenset())

    pags, sepsets = {}, {}
    for alpha in alphas:
        pag, seps = run_fci(sample, ci_test, alpha, bk)
        unpermuted = np.empty_like(pag)
        unpermuted[np.ix_(order, order)] = pag
        pags[alpha] = unpermuted
        sepsets[alpha] = {
            tuple(sorted((int(order[i]), int(order[j])))):
                frozenset(int(order[k]) for k in s if k not in never_conditioned)
            for (i, j), s in seps.items()
        }
    return pags, sepsets, redraws


@dataclass
class BootstrapResult:
    """PAGs from all bootstraps, with node order ``columns``."""

    columns: list[str]
    pags: dict[float, np.ndarray]  # alpha -> (n_bootstraps, p, p) endpoint matrices
    sepsets: dict[float, list[dict]] = field(default_factory=dict)  # alpha -> per-bootstrap sepsets
    n_redrawn: int = 0  # degenerate resamples that were drawn again (see _draw_rows)

    @property
    def n_bootstraps(self) -> int:
        return len(next(iter(self.pags.values())))

    def save(self, path) -> None:
        """Save the PAGs (not the separating sets) to a ``.npz`` file."""
        alphas = list(self.pags)
        np.savez_compressed(path, columns=np.array(self.columns), alphas=np.array(alphas),
                            pags=np.stack([self.pags[a] for a in alphas]))

    @classmethod
    def load(cls, path) -> "BootstrapResult":
        with np.load(Path(path)) as f:
            return cls(columns=f["columns"].tolist(),
                       pags={float(a): pags for a, pags in zip(f["alphas"], f["pags"])})


def run_bootstrap_fci(data: np.ndarray, columns, setting: Setting, *, alphas=None, n_bootstraps=None,
                      n_jobs: int = -1, stream: int = 0, progress: bool = True, desc: str | None = None
                      ) -> BootstrapResult:
    """Run FCI on ``n_bootstraps`` resamples of ``data`` for each alpha (default: ``setting.alpha``).

    ``stream`` separates independent analyses on the same seed (e.g. diagnostic groups).
    """
    columns = list(columns)
    alphas = list(dict.fromkeys(float(a) for a in (alphas if alphas is not None else [setting.alpha])))
    n_bootstraps = n_bootstraps or setting.n_bootstraps

    tasks = (delayed(_bootstrap_worker)(data, columns, setting, alphas, b, stream) for b in range(n_bootstraps))
    results = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
    if progress:
        results = tqdm(results, total=n_bootstraps, desc=desc or setting.name, unit="boot")
    results = list(results)

    return BootstrapResult(
        columns=columns,
        pags={a: np.stack([r[0][a] for r in results]) for a in alphas},
        sepsets={a: [r[1][a] for r in results] for a in alphas},
        n_redrawn=sum(r[2] for r in results),
    )
