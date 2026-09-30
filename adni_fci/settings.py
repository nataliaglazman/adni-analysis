"""Every analysis setting in one place.

Each of the old notebooks (``siemens_clean_csf.ipynb``, ``siemens_clean_cog.ipynb``, ...)
differed from the main analysis in a handful of choices. A :class:`Setting` holds all of
those choices, and :data:`PRESETS` lists the named settings. To try a new variant, add an
entry to ``PRESETS`` (or build one on the fly with ``dataclasses.replace``) instead of
copying a notebook.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"  # every file the analysis reads: ADNI tables and the MRI region dictionaries
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(frozen=True)
class Setting:
    """All choices that define one analysis. Variable names refer to cohort columns (see data.py)."""

    name: str
    description: str = ""

    # --- Variables in the causal graph -------------------------------------------------
    mri: tuple[str, ...] = ("Intracranial Volume", "Hippocampus")
    biomarkers: tuple[str, ...] = ("AB42_AB40_ratio", "pT217_F", "NfL_Q", "GFAP_Q")
    cognitive: tuple[str, ...] = ("TOTAL13",)
    demographics: tuple[str, ...] = ("PTGENDER", "PTEDUCAT", "AGE", "APOE4_0", "APOE4_1")
    # Technical variables treated like demographics (no incoming edges), e.g. assay platform.
    covariates: tuple[str, ...] = ()

    # --- Data sources ------------------------------------------------------------------
    plasma_release: str = "2025"     # "2025": UPenn Apr 2025; "2026": Mar 2026 (NfL/GFAP on two platforms)
    match_window_months: int = 3     # plasma / CSF / tau PET must lie within ± this of the MRI scan
    tau_tracer: str = "FTP"          # SUVRs are not comparable across tracers

    # --- Causal discovery --------------------------------------------------------------
    ci_test: str = "kci"             # any causal-learn CI test: "kci", "fisherz", "rcit", "fastkci", ...
    ci_test_kwargs: dict = field(default_factory=dict)
    alpha: float = 0.05
    n_bootstraps: int = 200
    seed: int = 42

    # --- Background knowledge (see knowledge.py) ---------------------------------------
    restrict_conditioning_on_cognition: bool = True  # cognitive scores never enter conditioning sets
    cognition_is_sink: bool = True                   # cognition cannot cause biomarkers or MRI
    forbid_cognition_to_cognition: bool = False      # removes all edges among cognitive scores
    forbid_mri_to_biomarkers: bool = False
    amyloid_tau_cascade: bool = False                # NfL/GFAP cannot cause amyloid or tau markers

    # --- Secondary analyses ------------------------------------------------------------
    sensitivity_alphas: tuple[float, ...] = (0.01, 0.05, 0.1)
    sepset_target: str = "Hippocampus"   # separating sets are reported for (biomarker, target) pairs
    n_bootstraps_stratified: int = 50
    min_group_size: int = 30             # diagnostic groups smaller than this are skipped

    # --- Reporting ---------------------------------------------------------------------
    min_edge_prob: float = 0.1       # edges below this are hidden in the full PAG figure
    stable_edge_prob: float = 0.5    # edges at or above this count as stable

    @property
    def variables(self) -> list[str]:
        return [*self.mri, *self.biomarkers, *self.cognitive, *self.demographics, *self.covariates]

    @property
    def roots(self) -> list[str]:
        """Variables that nothing else in the graph may cause."""
        return [*self.demographics, *self.covariates]

    def to_dict(self) -> dict:
        return asdict(self)


MAIN = Setting(
    name="main",
    description="Plasma Aβ42/40, pTau217, NfL, GFAP + ICV and hippocampus + ADAS-Cog13 + "
                "demographics, KCI test.",
)

COGNITION = replace(
    MAIN,
    name="cognition",
    description="Four cognitive outcomes (ADAS-Cog13, MMSE, TMT-B, MoCA); no edges among them.",
    cognitive=("TOTAL13", "MMSCORE", "TRABSCOR", "MOCA"),
    forbid_cognition_to_cognition=True,
)

# tau_pet, cognition_amygdala and plasma_assays were rebuilt from the variables shown in the
# old figures (pag_stability_corrected_tau_pet*.png, *_cog_amy.png, *_q_f*.png): their
# notebooks had been overwritten by copies of another notebook, so the original code was lost.
PRESETS: dict[str, Setting] = {s.name: s for s in (
    MAIN,
    replace(MAIN, name="fisherz", ci_test="fisherz",
            description="Main setting with the linear-Gaussian Fisher-z test instead of KCI."),
    replace(MAIN, name="csf", biomarkers=("ABETA42_ABETA40_ratio", "PTAU"),
            description="CSF Aβ42/40 and pTau181 in place of the plasma panel."),
    replace(MAIN, name="tau_pet", biomarkers=("AB42_AB40_ratio", "META_TEMPORAL_SUVR", "NfL_Q", "GFAP_Q"),
            description="Tau PET (flortaucipir meta-temporal SUVR) in place of plasma pTau217."),
    COGNITION,
    replace(COGNITION, name="cognition_amygdala", mri=("Intracranial Volume", "Hippocampus", "Amygdala"),
            description="Cognition setting with amygdala volume added."),
    replace(MAIN, name="plasma_assays", plasma_release="2026",
            biomarkers=("AB42_AB40_ratio", "pT217_F", "NfL", "GFAP"), covariates=("ASSAY_FUJIREBIO",),
            description="Mar 2026 plasma release: NfL/GFAP from Quanterix or Fujirebio, "
                        "with the assay platform as an exogenous covariate."),
    replace(MAIN, name="no_mri_to_plasma", forbid_mri_to_biomarkers=True,
            description="Main setting with MRI volumes forbidden from causing fluid biomarkers."),
)}


def get_setting(name: str) -> Setting:
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"Unknown setting {name!r}. Available: {', '.join(PRESETS)}") from None
