"""Country-level burden and clinical-resistance association analyses."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, rankdata, norm
from scipy.optimize import minimize, linprog
from scipy.optimize._numdiff import approx_derivative
from scipy.special import expit
from .common import read, save


def ols_hc3(x, y):
    A = x.to_numpy(float)
    v = y.to_numpy(float)
    beta = np.linalg.pinv(A.T @ A) @ (A.T @ v)
    resid = v - A @ beta
    hat = np.sum(A * (A @ np.linalg.pinv(A.T @ A)), axis=1)
    meat = A.T @ np.diag((resid / (1 - np.clip(hat, 0, 0.99))) ** 2) @ A
    inv = np.linalg.pinv(A.T @ A)
    cov = inv @ meat @ inv
    return (
        beta,
        np.sqrt(np.maximum(np.diag(cov), 0)),
        1 - np.sum(resid**2) / np.sum((v - v.mean()) ** 2),
    )


def ordinal_fit(X, y, w=None, clusters=None, inference=True):
    """Proportional odds logit; positive beta means higher ordered category.
    No intercept in X; ordered cutpoints supply intercepts. Rank/separation
    failures are disclosed; no ridge added to manufacture estimable effects.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    n, p = X.shape
    levels = np.unique(y)
    K = len(levels)
    if K < 2:
        return dict(success=False, status="constant_outcome")
    yy = np.searchsorted(levels, y)
    if np.linalg.matrix_rank(np.c_[np.ones(n), X]) < p + 1:
        return dict(success=False, status="rank_deficient")
    # Any cumulative split that admits separation invalidates finite PO inference.
    if inference:
        for cut in range(K - 1):
            signs = np.where(yy > cut, 1.0, -1.0)
            A = signs[:, None] * np.c_[np.ones(n), X]
            lp = linprog(
                -A.sum(axis=0),
                A_ub=-A,
                b_ub=np.zeros(n),
                bounds=[(-1, 1)] * (p + 1),
                method="highs",
            )
            if lp.success and -lp.fun > 1e-7:
                return dict(
                    success=False, status="cumulative_complete_or_quasi_separation"
                )
    w = np.ones(n) if w is None else np.asarray(w, float)
    w = w / w.mean()
    freq = np.bincount(yy, weights=w, minlength=K)
    cum = np.cumsum(freq)[:-1] / freq.sum()
    initial = np.log(cum / (1 - cum))
    theta = np.r_[np.zeros(p), initial[0], np.log(np.diff(initial))]

    def unpack(t):
        return np.r_[t[p], t[p] + np.cumsum(np.exp(t[p + 1 :]))]

    def probabilities(t, A):
        cs = expit(unpack(t)[None, :] - A @ t[:p, None])
        return np.diff(np.c_[np.zeros(len(A)), cs, np.ones(len(A))], axis=1)

    def logrows(t):
        return np.log(np.clip(probabilities(t, X)[np.arange(n), yy], 1e-14, 1))

    def obj(t):
        return -np.sum(w * logrows(t))

    ans = minimize(obj, theta, method="BFGS", options={"gtol": 2e-5, "maxiter": 700})
    gradient = approx_derivative(lambda t: np.array([obj(t)]), ans.x).ravel()
    if (
        not np.all(np.isfinite(ans.x))
        or np.max(np.abs(ans.x[:p])) > 18
        or (not ans.success and np.max(np.abs(gradient)) > 1e-3)
    ):
        return dict(success=False, status="nonconverged_or_boundary")
    result = dict(
        success=True,
        status="converged",
        beta=ans.x[:p],
        theta=ans.x,
        levels=levels,
        probabilities=lambda A: probabilities(ans.x, np.asarray(A, float)),
    )
    if inference:
        H = approx_derivative(
            lambda t: approx_derivative(lambda u: np.array([obj(u)]), t).ravel(), ans.x
        )
        H = (H + H.T) / 2
        ev = np.linalg.eigvalsh(H)
        if ev.min() <= 1e-7 or np.linalg.cond(H) > 1e10:
            return dict(success=False, status="singular_information")
        inv = np.linalg.inv(H)
        score = approx_derivative(logrows, ans.x) * w[:, None]
        covs = []
        for ids in clusters:
            ids = np.asarray(ids).astype(str)
            u = np.unique(ids)
            if len(u) < 2:
                return dict(success=False, status="single_cluster")
            ss = np.array([score[ids == k].sum(axis=0) for k in u])
            covs.append(len(u) / (len(u) - 1) * inv @ (ss.T @ ss) @ inv)
        se = np.sqrt(np.maximum.reduce([np.diag(c) for c in covs]))[:p]
        result.update(se=se, pvalue=2 * norm.sf(np.abs(result["beta"] / se)))
    return result


def run():
    who = read("country_foodborne_burden.csv")
    amr = read("country_AMR_mortality_categories.csv")
    ecorows = []
    loco = []
    for metric, g, xcol, ycol, log in [
        (
            "foodborne_DALY",
            who,
            "residue_all_years_rate",
            "foodborne_DALY_rate_100k",
            True,
        ),
        (
            "foodborne_death",
            who,
            "residue_all_years_rate",
            "foodborne_death_rate_100k",
            True,
        ),
        ("AMR_ordinal", amr, "raw_analyte_test_rate", "ordinal_bin", False),
    ]:
        g = (
            g.dropna(subset=[xcol, ycol])
            .drop_duplicates("country_iso3")
            .reset_index(drop=True)
        )
        xx = g[xcol].to_numpy()
        yy = g[ycol].to_numpy()
        rho, pval = spearmanr(xx, yy)
        for omitted in [None] + g.country_iso3.tolist():
            keep = (
                np.ones(len(g), bool)
                if omitted is None
                else g.country_iso3.ne(omitted).to_numpy()
            )
            rr, pp = spearmanr(xx[keep], yy[keep])
            y = np.log(yy[keep]) if log else yy[keep]
            f = sm.OLS(y, sm.add_constant(xx[keep] * 10)).fit(cov_type="HC3")
            ci = f.conf_int()[1]
            rec = dict(
                metric=metric,
                omitted_country=omitted or "none",
                n_countries=int(keep.sum()),
                spearman_rho=rr,
                spearman_p=pp,
                slope_per_10pp=f.params[1],
                CI_low_HC3=ci[0],
                CI_high_HC3=ci[1],
                p_HC3=f.pvalues[1],
                effect_scale=(
                    "log burden"
                    if log
                    else "ordinal score diagnostic only; not mortality rate"
                ),
            )
            if omitted is None:
                ecorows.append(rec)
                inf = sm.OLS(y, sm.add_constant(xx[keep] * 10)).fit().get_influence()
                gg = g[["country_iso3", xcol, ycol]].copy()
                gg["cooks_distance"] = inf.cooks_distance[0]
                gg["studentized_external_residual"] = inf.resid_studentized_external
                save(gg, "ecological_influence_" + metric)
            else:
                loco.append(rec)
            if metric != "AMR_ordinal":
                f2 = sm.OLS(yy[keep], sm.add_constant(xx[keep] * 10)).fit(
                    cov_type="HC3"
                )
                if omitted is None:
                    ecorows.append(
                        dict(
                            metric=metric + "_untransformed_linear",
                            omitted_country="none",
                            n_countries=int(keep.sum()),
                            slope_per_10pp=f2.params[1],
                            CI_low_HC3=f2.conf_int()[1, 0],
                            CI_high_HC3=f2.conf_int()[1, 1],
                            p_HC3=f2.pvalues[1],
                            effect_scale="burden rate per 100000",
                        )
                    )
    save(pd.DataFrame(ecorows), "ecological_sensitivity")
    save(pd.DataFrame(loco), "ecological_leave_one_country_out")
    country = read("country_foodborne_covariates.csv")
    models = []
    for outcome in ("foodborne_DALY_rate_100k", "foodborne_death_rate_100k"):
        for signal in ("residue_all_years_rate", "adjusted_occurrence_exploratory"):
            for adjustment, amu in [
                ("unadjusted", None),
                ("GDP_region_evidence_production_regulation", None),
                ("plus_reported_AMU_kg", "reported_AMU_kg"),
                ("plus_modelled_AMU_mgPCU", "modelled_AMU_mgPCU"),
            ]:
                cols = [outcome, signal]
                if adjustment != "unadjusted":
                    cols += [
                        "GDP_per_capita_current_USD",
                        "meat_tonnes_per_1000_people",
                        "TrACSS_vet_regulatory_stage_score",
                        "n_DOI",
                    ]
                if amu:
                    cols.append(amu)
                g = country.dropna(subset=cols).copy()
                g = g[g[outcome] > 0]
                if len(g) < 20:
                    continue
                X = pd.DataFrame(
                    {
                        "intercept": np.ones(len(g)),
                        "residue_signal_10pp": g[signal].to_numpy(float) * 10,
                    }
                )
                if adjustment != "unadjusted":
                    for c in [
                        "GDP_per_capita_current_USD",
                        "meat_tonnes_per_1000_people",
                        "TrACSS_vet_regulatory_stage_score",
                        "n_DOI",
                    ]:
                        X[c] = np.log1p(g[c].clip(lower=0)).to_numpy(float)
                    if amu:
                        X[amu] = np.log1p(g[amu].clip(lower=0)).to_numpy(float)
                    reg = pd.get_dummies(
                        g.continent, prefix="region", drop_first=True, dtype=float
                    ).reset_index(drop=True)
                    X = pd.concat([X, reg], axis=1)
                y = np.log(g[outcome]).reset_index(drop=True)
                b, se, r2 = ols_hc3(X, y)
                rho, p = spearmanr(g[signal], g[outcome])
                models.append(
                    {
                        "outcome": outcome,
                        "signal": signal,
                        "adjustment": adjustment,
                        "n_countries": len(g),
                        "coef_log_burden_per_10pp_residue": b[1],
                        "CI_low_HC3": b[1] - 1.96 * se[1],
                        "CI_high_HC3": b[1] + 1.96 * se[1],
                        "burden_ratio_per_10pp": np.exp(b[1]),
                        "burden_ratio_CI_low_HC3": np.exp(b[1] - 1.96 * se[1]),
                        "burden_ratio_CI_high_HC3": np.exp(b[1] + 1.96 * se[1]),
                        "r_squared": r2,
                        "spearman_rho_unadjusted_same_subset": rho,
                        "spearman_p_naive": p,
                        "interpretation": "cross-sectional ecological association; no causal attribution",
                    }
                )
    save(pd.DataFrame(models), "ecological_adjusted_associations.csv")
    allg = read("clinical_resistance_matches.csv")
    rng = np.random.default_rng(20261117)
    rows = []
    for species, g in allg.groupby("species", sort=False):
        x = g.food_detection_pct.to_numpy()
        y = g.resistance_pct.to_numpy()
        n = len(g)
        rho, p = spearmanr(x, y)
        ii = rng.integers(0, n, (2000, n))
        a = rankdata(x[ii], axis=1)
        b = rankdata(y[ii], axis=1)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        br = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        lo, hi = np.quantile(br, [0.025, 0.975])
        rows.append(
            dict(
                species=species,
                metric=g.metric.iloc[0],
                n_countries=n,
                rho=rho,
                p=p,
                CI_low=lo,
                CI_high=hi,
                year_min=int(g.AMR_year.min()),
                year_max=int(g.AMR_year.max()),
                adjustment="none",
                bootstrap_resamples=2000,
            )
        )
    save(pd.DataFrame(rows), "clinical_resistance_correlations.csv")
    for species in ["Escherichia coli", "Klebsiella pneumoniae"]:
        z = allg[allg.species.eq(species)]
        assert np.allclose(
            z.resistance_pct,
            np.sqrt(z.third_generation_cephalosporins_pct * z.fluoroquinolones_pct),
        )
    oc = read("occurrence_units.csv")
    oc = (
        oc.groupby("country_iso3")
        .agg(
            k=("positive_count_n", "sum"),
            n=("sample_size_n", "sum"),
            n_sources=("study_id", "nunique"),
        )
        .reset_index()
    )
    oc["value"] = oc.k / oc.n
    cc = read("concentration_statistics.csv")
    cc = cc[
        cc.statistic_type.eq("positive_mean")
        & cc.quantitative_eligible
        & ~cc.source_unit_caution.fillna("").astype(str).str.lower().isin(["true", "1"])
        & cc.concentration_ugkg.gt(0)
    ]
    cc = (
        cc.groupby("country_iso3")
        .agg(
            value=("concentration_ugkg", "median"),
            n_statistics=("statistic_id", "size"),
            n_sources=("source_id", "nunique"),
        )
        .reset_index()
    )
    values = []
    for ep, z in [("occurrence", oc), ("concentration", cc)]:
        z = z.merge(
            read("country_income_classification.csv"),
            on="country_iso3",
            how="left",
            validate="one_to_one",
        )
        z["endpoint"] = ep
        values.append(z)
    income = pd.concat(values, ignore_index=True)
    save(income, "country_income_values.csv")
    ref = read("country_income_analysis.csv").set_index(["endpoint", "country_iso3"])
    actual = income.set_index(["endpoint", "country_iso3"])
    assert np.allclose(actual.value, ref.reindex(actual.index).value, equal_nan=True)
    records = []
    from scipy.stats import kruskal

    for ep, z in income.groupby("endpoint"):
        values = [
            g.value.to_numpy()
            for _, g in z.dropna(subset=["income_group"]).groupby("income_group")
        ]
        result = kruskal(*values)
        for name, g in z.groupby("income_group"):
            records.append(
                dict(
                    endpoint=ep,
                    income_group=name,
                    n_countries=len(g),
                    median=g.value.median(),
                    q25=g.value.quantile(0.25),
                    q75=g.value.quantile(0.75),
                    Kruskal_Wallis_H=result.statistic,
                    Kruskal_Wallis_p=result.pvalue,
                )
            )
    save(pd.DataFrame(records), "income_group_comparisons.csv")


def curves():
    rng = np.random.default_rng(81005)
    curve = []
    fits = []
    who = read("country_foodborne_burden.csv")
    for field in ["foodborne_DALY_rate_100k", "foodborne_death_rate_100k"]:
        d = who.dropna(subset=["residue_all_years_rate", field])
        x = d.residue_all_years_rate.to_numpy() * 100
        y = np.log(d[field].to_numpy())
        grid = np.linspace(x.min(), x.max(), 120)
        A = np.c_[np.ones(len(x)), x / 10]
        G = np.c_[np.ones(len(grid)), grid / 10]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        bs = []
        for _ in range(1500):
            ix = rng.integers(0, len(d), len(d))
            bb = np.linalg.lstsq(A[ix], y[ix], rcond=None)[0]
            bs.append(np.exp(G @ bb))
        lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
        for xx, yy, ll, hh in zip(grid, np.exp(G @ coef), lo, hi):
            curve.append(
                dict(
                    endpoint=field,
                    residue_rate_pct=xx,
                    fitted=yy,
                    CI_low=ll,
                    CI_high=hh,
                    method="log_linear_country_bootstrap_1500_geometric_mean",
                )
            )
        fits.append(
            dict(
                endpoint=field,
                n_countries=len(d),
                effect_per_10pp=np.exp(coef[1]),
                bootstrap_successes=len(bs),
                model="unadjusted_log_linear",
            )
        )
    amr = (
        read("country_AMR_mortality_categories.csv")
        .dropna(subset=["raw_analyte_test_rate", "ordinal_bin"])
        .drop_duplicates("country_iso3")
    )
    x = amr.raw_analyte_test_rate.to_numpy() * 100
    y = amr.ordinal_bin.to_numpy(int)
    grid = np.linspace(x.min(), x.max(), 120)
    f = ordinal_fit(x[:, None] / 10, y, inference=False)
    if not f["success"]:
        raise RuntimeError("AMR ordinal fit failed")
    bs = []
    for _ in range(1500):
        ix = rng.integers(0, len(y), len(y))
        fb = ordinal_fit(x[ix, None] / 10, y[ix], inference=False)
        if fb["success"]:
            bs.append(fb["probabilities"](grid[:, None] / 10) @ fb["levels"])
    if len(bs) < 1200:
        raise RuntimeError("AMR bootstrap insufficient converged replicates")
    lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
    pred = f["probabilities"](grid[:, None] / 10) @ f["levels"]
    for xx, yy, ll, hh in zip(grid, pred, lo, hi):
        curve.append(
            dict(
                endpoint="AMR_ordinal",
                residue_rate_pct=xx,
                fitted=yy,
                CI_low=ll,
                CI_high=hh,
                method="proportional_odds_expected_ordinal_score_country_bootstrap_1500",
            )
        )
    fits.append(
        dict(
            endpoint="AMR_ordinal",
            n_countries=len(y),
            effect_per_10pp=np.exp(f["beta"][0]),
            bootstrap_successes=len(bs),
            model="unadjusted_proportional_odds",
        )
    )
    save(pd.DataFrame(curve), "ecological_fitted_curves.csv")
    save(pd.DataFrame(fits), "ecological_curve_models.csv")
