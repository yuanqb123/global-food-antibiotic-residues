"""Shared design matrices, cluster covariance and multiplicity adjustment."""

import numpy as np
import pandas as pd
from scipy import stats, linalg
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ii = np.where(np.isfinite(p))[0]
    o = ii[np.argsort(p[ii])]
    q[o] = np.minimum.accumulate((p[o] * len(o) / np.arange(1, len(o) + 1))[::-1])[
        ::-1
    ].clip(0, 1)
    return q


def design(g, cats, numeric=None):
    X = pd.get_dummies(
        g[cats].fillna("missing").astype(str), drop_first=True, dtype=float
    )
    for k, v in (numeric or {}).items():
        X[k] = np.asarray(v, float)
    X = sm.add_constant(X, has_constant="add")
    _, _, p = linalg.qr(X.to_numpy(), mode="economic", pivoting=True)
    r = np.linalg.matrix_rank(X)
    return X.iloc[:, sorted(p[:r])]


def wls(g, y, X, w, cluster):
    fit = sm.WLS(y, X, weights=w).fit()
    G = g[cluster].nunique()
    cov = cov_cluster(fit, g[cluster])
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return fit, cov, se, G
