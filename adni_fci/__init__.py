"""Bootstrapped FCI causal discovery on ADNI plasma, CSF, PET, MRI and cognitive data.

Typical use::

    from adni_fci import PRESETS, run_setting
    results = run_setting(PRESETS["main"])

See ``tutorial.ipynb`` for a walkthrough and ``run_analysis.py`` for the command line.
"""
from .data import build_cohort
from .discovery import BootstrapResult, run_bootstrap_fci, run_fci
from .knowledge import RestrictedCIT, build_background_knowledge, forbidden_edges, make_ci_test
from .pipeline import STEPS, compare_settings, load_results, run_setting, run_stratified
from .plotting import (DISPLAY_NAMES, pag_graph, plot_background_knowledge, plot_distributions,
                       plot_endpoint_heatmaps, plot_sepsets, save_pag, show_pag)
from .preprocessing import preprocess
from .settings import PRESETS, RESULTS_DIR, Setting, get_setting
from .summaries import (EDGE_MEANINGS, compare_edge_tables, edge_frequencies, endpoint_frequencies,
                        format_edges, sepset_summary)
