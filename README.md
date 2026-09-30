# Causal discovery on ADNI multimodal biomarkers

Bootstrapped FCI ([causal-learn](https://github.com/py-why/causal-learn)) on baseline ADNI data: plasma or CSF
biomarkers, tau PET, MRI volumes, cognition and demographics. The `main` setting is the analysis in
`docs/AAIC abstract.docx`.

## Quick start

```bash
conda activate causallearn        # Python 3.13, packages in requirements.txt (+ graphviz for figures)
python run_analysis.py --list     # available settings
python run_analysis.py main       # full main analysis -> results/main/
```

A full KCI run of one setting (200 bootstraps, all steps) takes roughly 1–1.5 h on the 14-core M4 Pro; Fisher-z
runs take seconds. `tutorial.ipynb` walks through every step: cohort, preprocessing, background knowledge,
bootstrapped FCI and the summaries.

## Settings

Every setting is defined once in `adni_fci/settings.py`; together they replace the old per-setting notebooks.

| Setting | Differs from `main` by | 
|---|---|---|
| `main` | (plasma Aβ42/40, pTau217, NfL, GFAP; ICV, hippocampus; ADAS-Cog13; KCI, α = 0.05, 200 bootstraps) | 
| `fisherz` | Fisher-z test instead of KCI |
| `csf` | CSF Aβ42/40 and pTau181 instead of plasma |
| `cognition` | adds MMSE, TMT-B and MoCA; no edges among cognitive scores | 
| `cognition_amygdala` | `cognition` plus amygdala volume | 
| `tau_pet` | tau PET (FTP meta-temporal SUVR) instead of plasma pTau217 | 
| `plasma_assays` | Mar 2026 plasma release: NfL/GFAP from Quanterix or Fujirebio, assay platform as a covariate | `siemens_clean_assays`* |
| `no_mri_to_plasma` | MRI volumes may not cause fluid biomarkers | 

To add a setting, add a `replace(MAIN, name=..., ...)` entry to `PRESETS` (see the end of the tutorial).

## Running

```bash
python run_analysis.py main csf tau_pet          # several settings
python run_analysis.py all --steps main          # every setting, main step only
python run_analysis.py main --n-bootstraps 50    # writes results/main_B50/, so full runs are never overwritten
python run_analysis.py main --ci-test fisherz --alpha 0.01
```

`--steps` (default: all) selects from:

- `main`: bootstrapped FCI, edge frequencies and PAG figures;
- `sensitivity`: α = 0.01, 0.05 and 0.1 on the same resamples;
- `sepsets`: separating sets of each biomarker vs hippocampal volume;
- `stratified`: CN, MCI and AD separately, without cognitive scores.

Bootstraps run in parallel on all cores (`--n-jobs`). Each one draws from its own random stream, so results do not
depend on the number of workers.

## Outputs (`results/<setting>/`)

| File | Content |
|---|---|
| `setting.json` | every parameter of the run |
| `cohort.csv`, `preprocessing.csv` | analysis dataset; skewness and transforms |
| `edge_frequencies.csv` | frequency of every edge type for every pair of variables |
| `pag.png`, `pag_stable.png` | PAG with edges of frequency ≥ 0.1 and ≥ 0.5 |
| `endpoint_heatmaps.png` | P(arrowhead / tail / circle) at each endpoint |
| `bootstrap_pags.npz` | raw bootstrap PAGs (`BootstrapResult.load`) |
| `sensitivity_alpha.csv` | edge frequencies per α |
| `sepsets.csv`, `sepsets.png` | separating sets, biomarker vs hippocampus |
| `stratified/` | per-group edge frequencies, PAGs and a comparison table |

`load_results(name)` and `compare_settings([...])` read finished runs back in, e.g. to re-plot with other thresholds
without re-running FCI.

## Layout

```
adni_fci/            analysis package
  settings.py        all settings (PRESETS) and data paths
  data.py            ADNI loading and cohort assembly
  preprocessing.py   skew correction and scaling
  knowledge.py       background knowledge and the restricted CI test
  discovery.py       bootstrapped FCI
  summaries.py       edge frequencies, separating sets, comparisons
  plotting.py        PAG figures and heatmaps
  pipeline.py        end-to-end run of one setting
run_analysis.py      command line
tutorial.ipynb       walkthrough
data/                the 14 files the code reads: git-ignored, never commit (ADNI Data Use Agreement)
data_unused/         every other data file, not read by the code (git-ignored for the same reason)
results/             outputs (git-ignored, contain participant-level tables)
archive/             old notebooks, figures and outputs (git-ignored)
docs/                AAIC abstract; pipeline graphic for slides (pipeline.png/.svg/.pdf, made by pipeline_figure.py)
```

