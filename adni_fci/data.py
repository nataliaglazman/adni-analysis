"""Load the ADNI tables and assemble the analysis cohort (one row per subject).

The MRI scan anchors each subject: fluid biomarkers (plasma or CSF) and tau PET must lie
within ± ``Setting.match_window_months`` of it. Subject-level tables (demographics, APOE,
cognition, diagnosis) are joined on RID. Only the tables a setting needs are loaded.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .settings import DATA_DIR, Setting

PLASMA_FILES = {
    "2025": "UPenn Plasma Quanterix Apr 25 2025.csv",
    "2026": "UPenn Plasma Quanterix Mar 25 2026.csv",
}
PLASMA_VARS = {"AB42_F", "AB40_F", "AB42_AB40_ratio", "pT217_F", "NfL_Q", "GFAP_Q",
               "NfL_F", "GFAP_F", "NfL", "GFAP", "ASSAY_FUJIREBIO"}
CSF_VARS = {"ABETA42", "ABETA40", "ABETA42_ABETA40_ratio", "PTAU", "TAU"}
TAU_PET_VARS = {"META_TEMPORAL_SUVR"}
DEMOGRAPHIC_VARS = {"PTGENDER", "PTEDUCAT", "AGE", "APOE4_0", "APOE4_1"}
BINARY_VARS = ("PTGENDER", "APOE4_0", "APOE4_1", "ASSAY_FUJIREBIO")

# Columns the old pipeline removed before summing FreeSurfer volumes into regions.
EXCLUDED_MRI_COLUMNS = ["IMAGEUID", "RUNDATE", "STATUS", "ST8SV", "ST68SV", "ST125SV", "ST66SV"]


# --- MRI -------------------------------------------------------------------------------

def mri_region_columns(columns) -> dict[str, list[str]]:
    """Map broad brain regions (e.g. 'Hippocampus') to their FreeSurfer volume columns."""
    datadic = pd.read_csv(DATA_DIR / "datadic2.csv")
    datadic["Grouping"] = datadic["Broader Region"].str.split("(").str[0].str.rstrip()
    datadic["Brain Region"] = datadic["Brain Region"].str.split("(").str[0].str.replace(" ", "")

    dic = pd.read_csv(DATA_DIR / "dictionary_mri.csv", header=None)
    dic["Brain Region"] = dic.iloc[:, 1].str.split(" ").str[-1]
    dic = dic[~dic.iloc[:, 1].str.contains("Thickness|Surface Area")]

    mapping = pd.merge(dic, datadic, on="Brain Region", how="right")
    col_to_region = dict(zip(mapping[0], mapping["Grouping"]))

    regions = defaultdict(list)
    for col in columns:
        region = col_to_region.get(col)
        if ("SV" in col or "CV" in col) and isinstance(region, str):
            regions[region].append(col)
    return dict(regions)


def load_mri(regions) -> pd.DataFrame:
    """UCSF FreeSurfer volumes summed into broad regions; one row per scan, earliest first."""
    raw = pd.read_csv(DATA_DIR / "freesurfer.csv", low_memory=False)
    mri = raw[raw.isnull().sum(axis=1) < 25]  # basic QC: drop scans with many missing values
    mri = mri.dropna(subset=["EXAMDATE", "RID"]).sort_values("EXAMDATE")
    mri = mri.drop(columns=[c for c in EXCLUDED_MRI_COLUMNS if c in mri.columns])

    region_cols = mri_region_columns(mri.columns)
    out = mri[["RID", "EXAMDATE"]].copy()
    for region in regions:
        if region not in region_cols:
            raise KeyError(f"Unknown MRI region {region!r}. Available: {sorted(region_cols)}")
        cols = region_cols[region]
        out[region] = mri[cols].sum(axis=1, min_count=len(cols))  # NaN if any part is missing
    out["EXAMDATE"] = pd.to_datetime(out["EXAMDATE"])
    return out.reset_index(drop=True)


# --- Fluid biomarkers and tau PET (matched to the MRI scan by date) --------------------

def load_plasma(release: str = "2025") -> pd.DataFrame:
    """Baseline UPenn plasma panel: Fujirebio Aβ42, Aβ40, pTau217 and Quanterix NfL/GFAP.

    In the 2026 release NfL/GFAP were measured on Quanterix for some samples and on
    Fujirebio for others (on different scales). ``NfL``/``GFAP`` combine the two, and
    ``ASSAY_FUJIREBIO`` flags samples measured on Fujirebio.
    """
    if release not in PLASMA_FILES:
        raise ValueError(f"Unknown plasma release {release!r}. Available: {list(PLASMA_FILES)}")
    df = pd.read_csv(DATA_DIR / PLASMA_FILES[release])
    df = df[df["VISCODE2"] == "bl"].copy()

    markers = [c for c in ("AB42_F", "AB40_F", "pT217_F", "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F") if c in df]
    df[markers] = df[markers].mask(df[markers] == -4)  # -4 = failed assay / below detection
    df["AB42_AB40_ratio"] = df["AB42_F"] / df["AB40_F"]

    if {"NfL_F", "GFAP_F"} <= set(df.columns):
        on_quanterix = df[["NfL_Q", "GFAP_Q"]].notna().all(axis=1)
        on_fujirebio = ~on_quanterix & df[["NfL_F", "GFAP_F"]].notna().all(axis=1)
        for marker in ("NfL", "GFAP"):
            df[marker] = df[f"{marker}_Q"].where(on_quanterix, df[f"{marker}_F"].where(on_fujirebio))
        df["ASSAY_FUJIREBIO"] = np.where(on_quanterix, 0.0, np.where(on_fujirebio, 1.0, np.nan))

    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"])
    return df[["RID", "EXAMDATE", *[c for c in df.columns if c in PLASMA_VARS]]]


def load_csf() -> pd.DataFrame:
    """Baseline CSF Aβ42, Aβ40, pTau181 and total tau (complete panels only)."""
    df = pd.read_csv(DATA_DIR / "csf.csv")
    df = df[df["VISCODE2"] == "bl"][["RID", "EXAMDATE", "ABETA42", "ABETA40", "PTAU", "TAU"]].copy()
    df["ABETA42_ABETA40_ratio"] = df["ABETA42"] / df["ABETA40"]
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"])
    return df.dropna()


def load_tau_pet(tracer: str = "FTP") -> pd.DataFrame:
    """Tau PET meta-temporal SUVR for one tracer, one row per scan."""
    df = pd.read_csv(DATA_DIR / "tau_pet.csv", low_memory=False)
    df = df[df["TRACER"] == tracer][["RID", "SCANDATE", "META_TEMPORAL_SUVR"]]
    df = df.rename(columns={"SCANDATE": "EXAMDATE"})
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"])
    return df.dropna()


def match_to_scan(cohort: pd.DataFrame, measurements: pd.DataFrame, window_months: int) -> pd.DataFrame:
    """Attach measurements taken within ± ``window_months`` of each subject's MRI.

    Keeps each subject's earliest eligible MRI (``cohort`` is date-sorted) and, if several
    measurements qualify, the one closest in time to it.
    """
    merged = cohort.assign(_row=np.arange(len(cohort))).merge(
        measurements.rename(columns={"EXAMDATE": "_date"}), on="RID")
    offset = pd.DateOffset(months=window_months)
    merged = merged[(merged["_date"] >= merged["EXAMDATE"] - offset)
                    & (merged["_date"] <= merged["EXAMDATE"] + offset)]
    merged = merged.assign(_gap=(merged["_date"] - merged["EXAMDATE"]).abs())
    merged = merged.sort_values(["_row", "_gap"], kind="stable").drop_duplicates("RID")
    return merged.drop(columns=["_row", "_date", "_gap"]).reset_index(drop=True)


# --- Subject-level tables (joined on RID) ----------------------------------------------

def _first_valid(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Drop ADNI missing-value codes (negative values) and keep the first row per subject."""
    df = df[["RID", column]].copy()
    df[column] = df[column].mask(df[column] < 0)
    return df.dropna().drop_duplicates("RID")


def load_demographics() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "demographics.csv", low_memory=False)
    df = df[["RID", "PTGENDER", "PTEDUCAT", "PTDOB"]].drop_duplicates("RID")  # first record per subject
    df[["PTGENDER", "PTEDUCAT"]] = df[["PTGENDER", "PTEDUCAT"]].mask(df[["PTGENDER", "PTEDUCAT"]] < 0)
    return df


def load_apoe() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "apoe.csv")[["RID", "GENOTYPE"]].dropna()
    df["APOE4"] = df["GENOTYPE"].str.count("4")
    return df[["RID", "APOE4"]].drop_duplicates("RID")


def load_adas() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "adas.csv", low_memory=False)
    return _first_valid(df[df["VISCODE2"] == "bl"], "TOTAL13")


def load_mmse() -> pd.DataFrame:
    """Earliest available MMSE (visit codes are not used, as in the original analysis)."""
    df = pd.read_csv(DATA_DIR / "mmse.csv", low_memory=False)
    df = df.assign(VISDATE=pd.to_datetime(df["VISDATE"])).sort_values("VISDATE", kind="stable")
    return _first_valid(df, "MMSCORE")


def load_moca() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "moca.csv", low_memory=False)
    return _first_valid(df[df["VISCODE2"] == "bl"], "MOCA")


def load_trails_b() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "neurobat.csv", low_memory=False)
    return _first_valid(df[df["VISCODE2"] == "bl"], "TRABSCOR")


def load_diagnosis() -> pd.DataFrame:
    """Baseline diagnosis (CN / MCI / AD).

    Uses VISCODE2 == 'bl' like every other table; the raw VISCODE is only 'bl' for ADNI1,
    which left ~13% of the cohort without a diagnosis in the original notebooks.
    """
    df = pd.read_csv(DATA_DIR / "diagnosis.csv", low_memory=False)
    df = df[df["VISCODE2"] == "bl"].dropna(subset=["DIAGNOSIS"]).drop_duplicates("RID")
    df["DX_GROUP"] = df["DIAGNOSIS"].map({1: "CN", 2: "MCI", 3: "AD"})
    return df[["RID", "DX_GROUP"]]


COGNITIVE_LOADERS = {
    "TOTAL13": load_adas,     # ADAS-Cog13 (higher = worse)
    "MMSCORE": load_mmse,     # MMSE
    "MOCA": load_moca,        # MoCA
    "TRABSCOR": load_trails_b,  # Trail Making Test B, seconds (higher = worse)
}


# --- Cohort ----------------------------------------------------------------------------

def _check_variables(setting: Setting) -> None:
    known = PLASMA_VARS | CSF_VARS | TAU_PET_VARS
    problems = [v for v in (*setting.biomarkers, *setting.covariates) if v not in known]
    problems += [v for v in setting.cognitive if v not in COGNITIVE_LOADERS]
    problems += [v for v in setting.demographics if v not in DEMOGRAPHIC_VARS]
    if problems:
        raise ValueError(f"Setting {setting.name!r} uses unknown variables {problems}. "
                         "Add a loader for them in adni_fci/data.py.")


def build_cohort(setting: Setting) -> pd.DataFrame:
    """One row per subject with every variable of ``setting`` observed.

    Columns: RID, EXAMDATE (MRI date), DX_GROUP (baseline diagnosis, may be missing),
    followed by ``setting.variables``.
    """
    _check_variables(setting)
    variables = setting.variables
    cohort = load_mri(setting.mri)

    fluid_sources = (
        (PLASMA_VARS, lambda: load_plasma(setting.plasma_release)),
        (CSF_VARS, load_csf),
        (TAU_PET_VARS, lambda: load_tau_pet(setting.tau_tracer)),
    )
    for source_vars, load in fluid_sources:
        needed = [v for v in variables if v in source_vars]
        if needed:
            cohort = match_to_scan(cohort, load().dropna(subset=needed), setting.match_window_months)

    for table in (load_demographics(), load_apoe(), *(COGNITIVE_LOADERS[c]() for c in setting.cognitive)):
        cohort = cohort.merge(table, on="RID", how="inner")
    cohort = cohort.merge(load_diagnosis(), on="RID", how="left")

    cohort["AGE"] = cohort["EXAMDATE"].dt.year - pd.to_datetime(cohort["PTDOB"], format="%m/%Y").dt.year
    cohort["PTGENDER"] = cohort["PTGENDER"].map({1: 0, 2: 1})  # 1 = female
    cohort["APOE4_0"] = (cohort["APOE4"] == 0).astype(int)     # reference category: two APOE4 copies
    cohort["APOE4_1"] = (cohort["APOE4"] == 1).astype(int)

    cohort = cohort.dropna(subset=variables).reset_index(drop=True)
    binary = [c for c in BINARY_VARS if c in variables]
    cohort[binary] = cohort[binary].astype(int)
    return cohort[["RID", "EXAMDATE", "DX_GROUP", *variables]]
