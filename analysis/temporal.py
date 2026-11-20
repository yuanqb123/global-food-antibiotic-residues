"""Adjusted temporal trends and the complete food-by-antibiotic inventory."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog
from scipy.special import expit
from scipy.stats import norm, chi2
from .common import read, save
from .weighting import block_weights

BASE_FIELDS = [
    ("food_group", 6),
    ("antibiotic_class_std", 10),
    ("GBD_super_region", 7),
    ("sample_source", 5),
    ("analytical_method", 5),
    ("sampling_design", 4),
]


def categorical_design(g, fields):
    X = pd.DataFrame({"intercept": np.ones(len(g))}, index=g.index)
    for col, maximum in fields:
        s = g[col].fillna("missing").astype(str)
        top = s.value_counts().index[:maximum]
        s = s.where(s.isin(top), "other")
        dum = pd.get_dummies(s, prefix=col, drop_first=True, dtype=float)
        X = pd.concat((X, dum), axis=1)
    return X


def fit_binomial(g, X, clusters=("country_iso3", "source_id")):
    A = np.asarray(X, float)
    n = g.sample_size_n.to_numpy(float)
    k = g.positive_count_n.to_numpy(float)
    w = block_weights(g)

    def obj(b):
        eta = A @ b
        return -np.sum(w * (k * eta - n * np.logaddexp(0, eta))), -A.T @ (
            w * (k - n * expit(eta))
        )

    ans = minimize(
        obj,
        np.zeros(A.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 800, "ftol": 1e-10},
    )
    p = expit(A @ ans.x)
    H = A.T @ ((w * n * p * (1 - p))[:, None] * A)
    Hinv = np.linalg.pinv(H, rcond=1e-10)
    covs = []
    for col in clusters:
        ids = g[col].fillna("missing").astype(str).to_numpy()
        unique = np.unique(ids)
        scores = np.asarray(
            [
                A[ids == u].T
                @ (w[ids == u] * (k[ids == u] - n[ids == u] * p[ids == u]))
                for u in unique
            ]
        )
        adj = len(unique) / (len(unique) - 1) if len(unique) > 1 else 1
        covs.append(adj * Hinv @ (scores.T @ scores) @ Hinv)
    # Conservative diagonal envelope across country and DOI cluster covariances.
    ses = np.sqrt(np.maximum.reduce([np.maximum(np.diag(c), 0) for c in covs]))
    ll = -obj(ans.x)[0]
    return {
        "beta": ans.x,
        "se": ses,
        "covs": covs,
        "p": p,
        "ll": ll,
        "success": ans.success,
        "columns": list(X),
        "n_units": len(g),
        "n_countries": g.country_iso3.nunique(),
        "n_DOI": g.source_id.nunique(),
    }


def estimate(res, key, scale=1):
    i = res["columns"].index(key)
    b = res["beta"][i] * scale
    se = res["se"][i] * abs(scale)
    z = b / se if se > 0 else np.nan
    return {
        "OR": np.exp(b),
        "CI_low": np.exp(b - 1.96 * se),
        "CI_high": np.exp(b + 1.96 * se),
        "p": 2 * norm.sf(abs(z)) if np.isfinite(z) else np.nan,
        "beta": b,
        "se": se,
    }


def bh(p):
    a = np.asarray(p, float)
    q = np.full(len(a), np.nan)
    valid = np.where(np.isfinite(a))[0]
    if not len(valid):
        return q
    order = valid[np.argsort(a[valid])]
    ranks = np.arange(1, len(order) + 1)
    q[order] = np.minimum.accumulate((a[order] * len(order) / ranks)[::-1])[::-1].clip(
        0, 1
    )
    return q


def temporal(g, fields=BASE_FIELDS, nonlinear=False):
    g = g.reset_index(drop=True)
    X = categorical_design(g, fields)
    X["year_5"] = (g.year.to_numpy() - 2010) / 5
    if nonlinear:
        # Restricted (natural) cubic spline with four fixed knots, scaled to
        # decades to keep the optimizer well conditioned.
        t = (g.year.to_numpy() - 2010) / 10
        knots = np.array([1998, 2008, 2017, 2024], float)
        knots = (knots - 2010) / 10

        def d(v, k):
            return np.maximum(v - k, 0) ** 3

        for j, k in enumerate(knots[:-2]):
            X[f"rcs_{j+1}"] = (
                d(t, k)
                - d(t, knots[-1]) * (knots[-1] - k) / (knots[-1] - knots[-2])
                + d(t, knots[-2]) * (knots[-2] - k) / (knots[-1] - knots[-2])
            ) / 10
    return fit_binomial(g, X), X, g


def log_concentration_model(g, country_fixed=False):
    g = g.reset_index(drop=True)
    fields = [("food_group", 6), ("antibiotic_std", 20)] + (
        [("country_iso3", 100)] if country_fixed else [("GBD_super_region", 7)]
    )
    X = categorical_design(g, fields)
    X["year_5"] = (g.year_mid.to_numpy() - 2010) / 5
    A = X.to_numpy(float)
    y = np.log(g.concentration_ugkg.to_numpy(float))
    # Equal weight to country, then source/frame block, then concentration rows.
    country = g.country_iso3.fillna("unknown")
    block = country + "|" + g.source_id + "|" + g.sampling_frame_id.fillna("unknown")
    w = 1 / (
        block.groupby(country).transform("nunique")
        * block.groupby(block).transform("size")
    ).to_numpy(float)
    inv = np.linalg.pinv(A.T @ (w[:, None] * A))
    beta = inv @ (A.T @ (w * y))
    err = y - A @ beta
    covs = []
    for col in ["country_iso3", "source_id"]:
        ids = g[col].fillna("missing").to_numpy()
        u = np.unique(ids)
        scores = np.array([A[ids == v].T @ (w[ids == v] * err[ids == v]) for v in u])
        covs.append(
            inv @ scores.T @ scores @ inv * (len(u) / (len(u) - 1) if len(u) > 1 else 1)
        )
    idx = list(X).index("year_5")
    b = beta[idx]
    se = max(np.sqrt(max(v[idx, idx], 0)) for v in covs)
    return {
        "n_points": len(g),
        "n_DOI": g.DOI.nunique(),
        "n_countries": g.country_iso3.nunique(),
        "year_min": g.year_mid.min(),
        "year_max": g.year_mid.max(),
        "ratio_per_5y": np.exp(b),
        "CI_low": np.exp(b - 1.96 * se),
        "CI_high": np.exp(b + 1.96 * se),
        "p": 2 * norm.sf(abs(b / se)),
        "beta": b,
        "se": se,
        "rank": np.linalg.matrix_rank(A),
        "n_parameters": A.shape[1],
    }


def run():
    base = read("occurrence_units.csv").merge(
        read("regression_regions.csv"), on="country_iso3", validate="many_to_one"
    )
    base["source_id"] = base.DOI.fillna("").where(
        base.DOI.fillna("").ne(""), base.study_id.fillna("unknown")
    )
    base = base.merge(
        read("measurement_covariates.csv"), on="canonical_row", validate="one_to_one"
    )
    base["year"] = base.year_mid
    base = base[base.year.between(1990, 2026) & base.country_iso3.notna()].reset_index(
        drop=True
    )
    linear, X, gb = temporal(base)
    spline, Xs, gs = temporal(base, nonlinear=True)
    save(
        pd.DataFrame(
            [
                dict(
                    n_units=len(base),
                    n_countries=base.country_iso3.nunique(),
                    n_DOI=base.DOI.nunique(),
                    success=linear["success"],
                    **estimate(linear, "year_5"),
                )
            ]
        ),
        "detection_primary_trend.csv",
    )
    idx = [
        spline["columns"].index(v) for v in spline["columns"] if v.startswith("rcs_")
    ]
    B = spline["beta"][idx]
    ps = [
        float(chi2.sf(B @ np.linalg.pinv(c[np.ix_(idx, idx)]) @ B, len(idx)))
        for c in spline["covs"]
    ]
    save(
        pd.DataFrame(
            [
                dict(
                    p_conservative=max(ps),
                    p_country=ps[0],
                    p_source=ps[1],
                    success=spline["success"],
                )
            ]
        ),
        "detection_nonlinearity.csv",
    )
    cs = (
        read("dated_positive_means.csv")
        .drop(columns=["GBD_super_region"], errors="ignore")
        .merge(read("regression_regions.csv"), on="country_iso3", how="left")
    )
    rows = []
    for title, z in [
        ("all_available_years", cs),
        ("through_2024", cs[cs.year_mid.le(2024)]),
        ("exclude_C_above_10000", cs[cs.concentration_ugkg.le(10000)]),
    ]:
        rows.append(dict(analysis=title, **log_concentration_model(z)))
    rows.append(
        dict(
            analysis="country_fixed_effects",
            **log_concentration_model(cs, country_fixed=True),
        )
    )
    save(pd.DataFrame(rows), "concentration_primary_trends.csv")
    rows = []
    annual = []
    rng = np.random.default_rng(101006)
    for (food, drug), g in base.groupby(["food_group", "antibiotic_std"]):
        info = dict(
            food_group=food,
            antibiotic_std=drug,
            n_units=len(g),
            n_sources=g.source_id.nunique(),
            n_DOI=g.DOI.nunique(),
            n_countries=g.country_iso3.nunique(),
            n_years=g.year.nunique(),
            year_min=g.year.min(),
            year_max=g.year.max(),
        )
        for year, z in g.groupby("year"):
            pooled = (
                z.groupby("source_id")[["positive_count_n", "sample_size_n"]]
                .sum()
                .to_numpy()
            )
            k, n = pooled.sum(axis=0)
            p = k / n
            if len(pooled) > 1:
                ix = rng.integers(0, len(pooled), size=(1000, len(pooled)))
                bs = pooled[ix].sum(axis=1)
                lo, hi = np.quantile(bs[:, 0] / bs[:, 1], [0.025, 0.975])
                method = "source_bootstrap_1000"
            else:
                zz = 1.96
                mid = (p + zz * zz / (2 * n)) / (1 + zz * zz / n)
                half = (
                    zz
                    * np.sqrt(p * (1 - p) / n + zz * zz / (4 * n * n))
                    / (1 + zz * zz / n)
                )
                lo, hi = max(0, mid - half), min(1, mid + half)
                method = "conditional_Wilson_single_source"
            annual.append(
                info
                | dict(
                    year_mid=year,
                    k=k,
                    n=n,
                    rate=p,
                    CI_low=lo,
                    CI_high=hi,
                    interval_method=method,
                )
            )
        if (
            g.year.nunique() < 2
            or g.positive_count_n.sum() == 0
            or (g.sample_size_n - g.positive_count_n).sum() == 0
        ):
            rows.append(info | dict(status="constant_or_no_time_variation"))
            continue
        g = g.reset_index(drop=True)
        XX = categorical_design(g, [("GBD_super_region", 7)])
        XX["year_5"] = (g.year - 2010) / 5
        A = XX.to_numpy()
        if np.linalg.matrix_rank(A) < A.shape[1]:
            rows.append(info | dict(status="rank_deficient"))
            continue
        signs = np.r_[
            A[g.positive_count_n.gt(0)], -A[g.positive_count_n.lt(g.sample_size_n)]
        ]
        lp = linprog(
            -signs.sum(axis=0),
            A_ub=-signs,
            b_ub=np.zeros(len(signs)),
            bounds=[(-1, 1)] * A.shape[1],
            method="highs",
        )
        if lp.success and -lp.fun > 1e-7:
            rows.append(info | dict(status="complete_or_quasi_separation"))
            continue
        rr = fit_binomial(g, XX)
        e = estimate(rr, "year_5")
        good = (
            rr["success"]
            and np.isfinite(e["beta"])
            and abs(e["beta"]) < 18
            and np.isfinite(e["se"])
            and e["se"] > 0
        )
        status = "estimable" if good else "nonconverged_or_boundary"
        if good:
            w = block_weights(g)
            pp = rr["p"]
            H = A.T @ ((w * g.sample_size_n.to_numpy() * pp * (1 - pp))[:, None] * A)
            model_se = np.sqrt(max(np.linalg.pinv(H)[-1, -1], 0))
            if e["se"] < 1e-3 * model_se:
                status = "point_only_collapsed_cluster_variance"
                e.update(CI_low=np.nan, CI_high=np.nan, p=np.nan, se=np.nan)
        rows.append(info | dict(status=status, **(e if good else {})))
    tr = pd.DataFrame(rows)
    qmask = tr.status.eq("estimable")
    tr["q"] = bh(tr.p.fillna(1))
    tr.loc[~qmask, "q"] = np.nan
    tr["positive_estimable"] = tr.status.isin(
        ["estimable", "point_only_collapsed_cluster_variance"]
    ) & tr.OR.gt(1)
    tr["positive_FDR_significant"] = tr.positive_estimable & tr.q.lt(0.05)
    save(tr, "food_antibiotic_trends.csv")
    save(pd.DataFrame(annual), "food_antibiotic_annual.csv")
