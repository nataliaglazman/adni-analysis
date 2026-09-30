"""End-to-end analysis of one setting, writing everything to ``results/<setting name>/``.

Steps
-----
main         bootstrapped FCI at ``setting.alpha``: edge frequencies, PAG figures, heatmaps
sensitivity  the same bootstraps re-run at ``setting.sensitivity_alphas``
sepsets      separating sets of each biomarker vs ``setting.sepset_target``
stratified   bootstrapped FCI within CN, MCI and AD (cognitive scores excluded)
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data import build_cohort
from .discovery import BootstrapResult, run_bootstrap_fci
from .plotting import plot_endpoint_heatmaps, plot_sepsets, save_pag
from .preprocessing import preprocess
from .settings import RESULTS_DIR, Setting
from .summaries import compare_edge_tables, edge_frequencies, format_edges, sepset_summary

STEPS = ("main", "sensitivity", "sepsets", "stratified")
DX_GROUPS = ("CN", "MCI", "AD")


def _section(title: str) -> None:
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def _report_redraws(boot: BootstrapResult, label: str) -> None:
    if boot.n_redrawn:
        print(f"{label}: {boot.n_redrawn} resample(s) had redundant variables and were redrawn")


def _save_pags(edges, out_dir: Path, stem: str, columns, setting: Setting, title: str) -> None:
    save_pag(edges, out_dir / f"{stem}.png", columns=columns, min_prob=setting.min_edge_prob, title=title)
    save_pag(edges, out_dir / f"{stem}_stable.png", columns=columns, min_prob=setting.stable_edge_prob, title=title)


def run_setting(setting: Setting, steps=STEPS, *, n_jobs: int = -1, results_dir=RESULTS_DIR,
                progress: bool = True) -> dict:
    """Run the chosen ``steps`` for ``setting`` and return the results as a dict."""
    steps = set(steps) | {"main"}
    if unknown := steps - set(STEPS):
        raise ValueError(f"Unknown steps {sorted(unknown)}; choose from {STEPS}")
    out = Path(results_dir) / setting.name
    out.mkdir(parents=True, exist_ok=True)
    variables = setting.variables

    _section(f"[{setting.name}] {setting.description}")
    cohort = build_cohort(setting)
    X, prep = preprocess(cohort, variables)
    dx_counts = cohort["DX_GROUP"].value_counts().reindex(DX_GROUPS, fill_value=0).to_dict()
    print(f"Cohort: {len(cohort)} subjects {dx_counts}, {len(variables)} variables")
    print(f"CI test: {setting.ci_test}, alpha = {setting.alpha}, {setting.n_bootstraps} bootstraps")
    (out / "setting.json").write_text(json.dumps(setting.to_dict(), indent=2, ensure_ascii=False))
    cohort.to_csv(out / "cohort.csv", index=False)
    prep.to_csv(out / "preprocessing.csv")

    alphas = [setting.alpha, *(setting.sensitivity_alphas if "sensitivity" in steps else ())]
    boot = run_bootstrap_fci(X, variables, setting, alphas=alphas, n_jobs=n_jobs, progress=progress)
    _report_redraws(boot, setting.name)
    boot.save(out / "bootstrap_pags.npz")

    edges = edge_frequencies(boot.pags[setting.alpha], variables)
    edges.to_csv(out / "edge_frequencies.csv", index=False)
    print(f"\nStable edges (frequency >= {setting.stable_edge_prob}):")
    print(format_edges(edges, setting.stable_edge_prob))
    _save_pags(edges, out, "pag", variables, setting, f"{setting.name} (n={len(cohort)})")
    plt.close(plot_endpoint_heatmaps(boot.pags[setting.alpha], variables, path=out / "endpoint_heatmaps.png"))
    results = {"setting": setting, "cohort": cohort, "preprocessing": prep, "bootstrap": boot, "edges": edges}

    if "sensitivity" in steps:
        tables = {f"alpha={a:g}": edge_frequencies(boot.pags[a], variables) for a in sorted(boot.pags)}
        sensitivity = compare_edge_tables(tables, min_prob=setting.min_edge_prob)
        sensitivity.to_csv(out / "sensitivity_alpha.csv")
        print(f"\nSensitivity to alpha (edges reaching {setting.stable_edge_prob} at some alpha):")
        print(sensitivity[sensitivity.max(axis=1) >= setting.stable_edge_prob].round(2).to_string())
        results["sensitivity"] = sensitivity

    if "sepsets" in steps:
        target = setting.sepset_target
        if target in variables:
            seps = sepset_summary(boot.sepsets[setting.alpha], variables, setting.biomarkers, target)
            seps.to_csv(out / "sepsets.csv")
            plt.close(plot_sepsets(seps, target, path=out / "sepsets.png"))
            print(f"\nSeparating sets, biomarker vs {target} (P(member | separated)):")
            print(seps.round(2).to_string())
            results["sepsets"] = seps
        else:
            print(f"\nSkipping sepsets: {target!r} is not a variable of this setting.")

    if "stratified" in steps:
        results["stratified"] = run_stratified(cohort, setting, out / "stratified", n_jobs=n_jobs, progress=progress)

    print(f"\nResults written to {out}")
    return results


def run_stratified(cohort: pd.DataFrame, setting: Setting, out_dir, *, n_jobs: int = -1,
                   progress: bool = True) -> pd.DataFrame | None:
    """Bootstrapped FCI within each diagnostic group, without cognitive scores (they largely track diagnosis)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [c for c in setting.variables if c not in setting.cognitive]
    tables = {}
    for stream, group in enumerate(DX_GROUPS, start=1):
        subset = cohort[cohort["DX_GROUP"] == group]
        if len(subset) < setting.min_group_size:
            print(f"\n{group}: skipped (n={len(subset)} < {setting.min_group_size})")
            continue
        cols = [c for c in columns if subset[c].nunique() > 1]
        if constant := [c for c in columns if c not in cols]:
            print(f"\n{group}: dropping constant columns {constant}")
        X, _ = preprocess(subset, cols)
        boot = run_bootstrap_fci(X, cols, setting, n_bootstraps=setting.n_bootstraps_stratified, stream=stream,
                                 n_jobs=n_jobs, progress=progress, desc=f"{setting.name} {group}")
        _report_redraws(boot, group)
        edges = edge_frequencies(boot.pags[setting.alpha], cols)
        edges.to_csv(out_dir / f"edge_frequencies_{group}.csv", index=False)
        _save_pags(edges, out_dir, f"pag_{group}", cols, setting, f"{setting.name}: {group} (n={len(subset)})")
        tables[f"{group} (n={len(subset)})"] = edges

    if not tables:
        return None
    comparison = compare_edge_tables(tables, min_prob=setting.min_edge_prob)
    comparison.to_csv(out_dir / "comparison.csv")
    print(f"\nStratified by diagnosis (edges reaching {setting.stable_edge_prob} in some group):")
    print(comparison[comparison.max(axis=1) >= setting.stable_edge_prob].round(2).to_string())
    return comparison


def setting_from_dict(values: dict) -> Setting:
    """Rebuild a Setting saved as ``setting.json`` (JSON turns tuples into lists)."""
    names = {f.name for f in fields(Setting)}
    return Setting(**{k: tuple(v) if isinstance(v, list) else v for k, v in values.items() if k in names})


def load_results(name: str, results_dir=RESULTS_DIR) -> dict:
    """Read the saved outputs of a finished run, e.g. to re-plot or compare without re-running FCI."""
    folder = Path(results_dir) / name
    if not (folder / "edge_frequencies.csv").exists():
        raise FileNotFoundError(f"No results for {name!r} in {folder.parent}. Run: python run_analysis.py {name}")
    results = {
        "setting": setting_from_dict(json.loads((folder / "setting.json").read_text())),
        "cohort": pd.read_csv(folder / "cohort.csv"),
        "edges": pd.read_csv(folder / "edge_frequencies.csv"),
        "bootstrap": BootstrapResult.load(folder / "bootstrap_pags.npz"),
    }
    for key, file, index_cols in (("sensitivity", "sensitivity_alpha.csv", [0, 1, 2]),
                                  ("sepsets", "sepsets.csv", 0),
                                  ("stratified", "stratified/comparison.csv", [0, 1, 2])):
        if (folder / file).exists():
            results[key] = pd.read_csv(folder / file, index_col=index_cols)
    return results


def compare_settings(names, results_dir=RESULTS_DIR, min_prob: float = 0.3) -> pd.DataFrame:
    """Edge probabilities of several finished runs side by side."""
    return compare_edge_tables({n: load_results(n, results_dir)["edges"] for n in names}, min_prob=min_prob)
