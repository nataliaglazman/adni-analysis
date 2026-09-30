#!/usr/bin/env python
"""Run the bootstrapped-FCI analysis for one or more settings (defined in adni_fci/settings.py).

Examples
--------
  python run_analysis.py --list                       # show available settings
  python run_analysis.py main                         # everything for the main setting
  python run_analysis.py main csf tau_pet --steps main
  python run_analysis.py all --n-bootstraps 50        # quick pass over every setting
  python run_analysis.py main --ci-test fisherz --alpha 0.01

Results go to results/<setting>/. Overriding --n-bootstraps, --alpha or --ci-test writes to a
separate folder (e.g. results/main_B50/) so that full runs are not overwritten.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")  # figures are only written to files

from adni_fci import PRESETS, RESULTS_DIR, STEPS, get_setting, run_setting  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("settings", nargs="*", help="setting name(s), or 'all'")
    parser.add_argument("--list", action="store_true", help="list the available settings and exit")
    parser.add_argument("--steps", default=",".join(STEPS),
                        help=f"comma-separated subset of {','.join(STEPS)} (default: all; main always runs)")
    parser.add_argument("--n-bootstraps", type=int, help="override the number of bootstraps")
    parser.add_argument("--alpha", type=float, help="override the significance level")
    parser.add_argument("--ci-test", help="override the CI test (kci, fisherz, rcit, fastkci, ...)")
    parser.add_argument("--tag", help="suffix for the output folder, results/<setting>_<tag>/")
    parser.add_argument("--n-jobs", type=int, default=-1, help="parallel workers (default: all cores)")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help=f"output root (default: {RESULTS_DIR})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list or not args.settings:
        width = max(map(len, PRESETS))
        print("Available settings:\n" + "\n".join(f"  {n:<{width}}  {s.description}" for n, s in PRESETS.items()))
        return

    names = list(PRESETS) if args.settings == ["all"] else args.settings
    settings = [get_setting(n) for n in names]
    steps = [s.strip() for s in args.steps.split(",") if s.strip()]

    overrides = {k: v for k, v in (("n_bootstraps", args.n_bootstraps), ("alpha", args.alpha),
                                   ("ci_test", args.ci_test)) if v is not None}
    tag = args.tag or "_".join([*([f"B{args.n_bootstraps}"] if args.n_bootstraps else []),
                                *([f"alpha{args.alpha:g}"] if args.alpha else []),
                                *([args.ci_test] if args.ci_test else [])])

    for setting in settings:
        if overrides or tag:
            setting = replace(setting, name=f"{setting.name}_{tag}" if tag else setting.name, **overrides)
        start = time.time()
        run_setting(setting, steps, n_jobs=args.n_jobs, results_dir=args.results_dir)
        print(f"[{setting.name}] finished in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
