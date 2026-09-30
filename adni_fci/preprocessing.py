"""Put variables on a comparable scale before running conditional-independence tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer, StandardScaler


def preprocess(df: pd.DataFrame, columns, skew_threshold: float = 1.0) -> tuple[np.ndarray, pd.DataFrame]:
    """Yeo-Johnson-transform skewed continuous columns, then z-score all continuous columns.

    Columns with at most two distinct values are treated as binary and left as 0/1.
    Returns the data matrix (column order = ``columns``) and a per-column report.
    """
    columns = list(columns)
    X = df[columns].to_numpy(dtype=float, copy=True)
    continuous = [i for i, c in enumerate(columns) if df[c].nunique() > 2]
    skew_before = {i: stats.skew(X[:, i]) for i in continuous}
    skewed = [i for i in continuous if abs(skew_before[i]) > skew_threshold]

    if skewed:
        X[:, skewed] = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(X[:, skewed])
    if continuous:
        X[:, continuous] = StandardScaler().fit_transform(X[:, continuous])

    report = pd.DataFrame({
        "type": ["continuous" if i in skew_before else "binary" for i in range(len(columns))],
        "skew_before": [skew_before.get(i, np.nan) for i in range(len(columns))],
        "yeo_johnson": [i in skewed for i in range(len(columns))],
        "skew_after": [stats.skew(X[:, i]) if i in skew_before else np.nan for i in range(len(columns))],
    }, index=pd.Index(columns, name="variable"))
    return X, report
