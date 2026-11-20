"""Independent governance association sensitivity retained in the SI."""

import warnings
import numpy as np
import pandas as pd
from scipy import linalg, stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_cluster, cov_cluster_2groups
from .common import read, save, OUTPUTS
from .weighting import weights

CC = "WGI_Control_of_Corruption"
END = ["occurrence", "joint_risk_trigger", "RQ_res"]
GE = "WGI_Government_Effectiveness"
CORE = [
    "GDP_per_capita_current_USD",
    "access_to_electricity_pct",
    "safely_managed_sanitation_pct",
    "year_mid",
]
BACK = [
    "production_intensity_proxy",
    "rural_population_pct",
    "agriculture_value_added_pct_GDP",
]
SCENARIOS = [
    "primary",
    "water_subset_base",
    "add_water",
    "add_production_structure",
    "add_sampling_method",
    "equal_record_weight",
    "drug_class_control",
]


def write(rows, name):
    save(pd.DataFrame(rows), name)


def wsd(x, w):
    m = np.average(x, weights=w)
    s = np.sqrt(np.average((x - m) ** 2, weights=w))
    return m, s


def outcome(g, e):
    if e == "occurrence":
        return g.positive_count_n.to_numpy(float) / g.sample_size_n.to_numpy(float)
    if e == "joint_risk_trigger":
        return g.available_path_attention_trigger.to_numpy(float)
    return np.log10(1 + g.RQ_res.to_numpy(float))


def select(raw, e, scenario):
    cols = CORE + [GE, CC]
    if scenario in ["water_subset_base", "add_water"]:
        cols += ["safely_managed_drinking_water_pct"]
    if scenario == "add_production_structure":
        cols += BACK
    g = raw[raw.WGI_pair_complete.eq(True)].copy()
    g = g.replace([np.inf, -np.inf], np.nan).dropna(subset=cols).reset_index(drop=True)
    assert not g.statistic_id.duplicated().any()
    w = (
        weights(g, e == "occurrence")
        if scenario != "equal_record_weight"
        else np.ones(len(g))
    )
    w = np.array(w, dtype=float, copy=True)
    w /= w.mean()
    return g, w


def design(g, e, scenario, w, vars):
    X = pd.DataFrame({"Intercept": np.ones(len(g))})
    scales = []
    cov = CORE.copy()
    if scenario == "add_water":
        cov += ["safely_managed_drinking_water_pct"]
    if scenario == "add_production_structure":
        cov += BACK
    for c in cov + vars:
        x = g[c].to_numpy(float)
        if c in ["GDP_per_capita_current_USD", "production_intensity_proxy"]:
            x = np.log1p(x)
        m, s = wsd(x, w)
        if s > 1e-12:
            X[c] = (x - m) / s
        scales.append({"variable": c, "mean_transformed": m, "SD_transformed": s})
    cats = [
        "food_group",
        (
            "antibiotic_class_broad"
            if e == "occurrence" or scenario == "drug_class_control"
            else "antibiotic_std"
        ),
    ]
    if e != "occurrence":
        cats += ["statistic_type"]
    if e == "joint_risk_trigger":
        cats += ["MRL_path_assessed"]
    if scenario == "add_sampling_method":
        cats += ["sample_source", "sampling_design", "analytical_method"]
    for c in cats:
        s = g[c].fillna("unreported").astype(str)
        X = pd.concat(
            [X, pd.get_dummies(s, prefix=c, drop_first=True, dtype=float)], axis=1
        )
    # Keep numerics/WGI first. Drop only nuisance dummy columns that have no independent contrast.
    keep = []
    rank = 0
    dropped = []
    for c in X:
        trial = np.asarray(X[keep + [c]]) * np.sqrt(w)[:, None]
        r = np.linalg.matrix_rank(trial)
        if r > rank:
            keep.append(c)
            rank = r
        else:
            dropped.append(c)
    assert all(v in keep for v in vars)
    return X[keep], scales, dropped


def covariance(fit, g):
    co = pd.factorize(g.country_iso3)[0]
    sr = pd.factorize(g.source_id)[0]
    country = cov_cluster(fit, co, use_correction=True)
    source = cov_cluster(fit, sr, use_correction=True)
    two, _, _ = cov_cluster_2groups(fit, co, sr, use_correction=True)
    return country, source, two


def infer(b, se, df):
    p = 2 * stats.t.sf(abs(b / se), df) if se > 0 else np.nan
    q = stats.t.ppf(0.975, df)
    return b - q * se, b + q * se, p


def wild_p(X, y, w, g, j, bread, beta, B=3999):
    A = X.to_numpy(float)
    col = A[:, j]
    A0 = np.delete(A, j, axis=1)
    bn = np.linalg.lstsq(A0 * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    fit0 = A0 @ bn
    e0 = y - fit0
    inv = bread
    v = inv[:, j]
    co = pd.factorize(g.country_iso3)[0]
    G = len(np.unique(co))
    N = len(y)
    P = A.shape[1]
    K = inv @ (A.T * w[None, :])
    base = K @ fit0
    perturb = np.column_stack([K[:, co == k] @ e0[co == k] for k in range(G)])
    qbase = np.array(
        [np.sum(w[co == k] * col[co == k] * 0) for k in range(G)]
    )  # reserved diagnostic shape
    av = A @ v
    L = np.array([np.sum(w[co == k] * av[co == k] * fit0[co == k]) for k in range(G)])
    U = np.array([np.sum(w[co == k] * av[co == k] * e0[co == k]) for k in range(G)])
    H = np.array([A[co == k].T @ (w[co == k] * av[co == k]) for k in range(G)])
    correction = G / (G - 1) * (N - 1) / (N - P)
    e = y - A @ beta
    sobs = np.array([np.sum(w[co == k] * av[co == k] * e[co == k]) for k in range(G)])
    observed = abs(beta[j] / np.sqrt(correction * np.sum(sobs**2)))
    rng = np.random.default_rng(10122026)
    signs = rng.choice([-1.0, 1.0], size=(G, B))
    bs = base[:, None] + perturb @ signs
    scores = L[:, None] + U[:, None] * signs - H @ bs
    sest = np.sqrt(correction * np.sum(scores**2, axis=0))
    ts = abs(bs[j] / sest)
    return (1 + np.sum(ts >= observed)) / (B + 1)


def run():
    results = []
    audits = []
    coverage = []
    scaling = []
    vifs = []
    loco = []
    jack = []
    glmrows = []
    allcoefs = []
    matches = []
    for e in END:
        raw = read("governance_" + e + "_input.csv")
        matched = raw[raw.WGI_pair_complete.eq(True)].copy()
        primary, wp = select(raw, e, "primary")
        matched["independent_analysis_disposition"] = np.where(
            matched.statistic_id.isin(primary.statistic_id),
            "included_primary",
            "missing_GDP_electricity_or_sanitation",
        )
        matched.to_csv(
            OUTPUTS / f"{e}_matched_disposition.csv", index=False, encoding="utf-8-sig"
        )
        for scenario in SCENARIOS:
            g, w = select(raw, e, scenario)
            yraw = outcome(g, e)
            ym, ys = wsd(yraw, w)
            y = (yraw - ym) / ys
            coverage.append(
                {
                    "endpoint": e,
                    "scenario": scenario,
                    "original_records": len(raw),
                    "governance_matched": len(matched),
                    "records": len(g),
                    "countries": g.country_iso3.nunique(),
                    "sources": g.source_id.nunique(),
                    "unique_canonical_rows": g.canonical_row.nunique(),
                    "positive_joint_rows": (
                        int(yraw.sum()) if e == "joint_risk_trigger" else np.nan
                    ),
                    "kish_weight_n": w.sum() ** 2 / np.sum(w * w),
                    "outcome_weighted_mean": ym,
                    "outcome_weighted_SD": ys,
                }
            )
            if scenario == "primary":
                aid = g[
                    [
                        "statistic_id",
                        "canonical_row",
                        "country_iso3",
                        "source_id",
                        "sampling_frame_id",
                        GE,
                        CC,
                    ]
                    + CORE
                ].copy()
                aid["analysis_weight"] = w
                aid["original_outcome"] = yraw
                aid["standardized_outcome"] = y
                aid.to_csv(
                    OUTPUTS / f"{e}_primary_analysis_rows.csv",
                    index=False,
                    encoding="utf-8-sig",
                )
            for mode, vars in [
                ("GE_only", [GE]),
                ("CC_only", [CC]),
                ("joint", [GE, CC]),
            ]:
                X, scale, dropped = design(g, e, scenario, w, vars)
                A = X.to_numpy(float)
                P = A.shape[1]
                fit = sm.WLS(y, A, weights=w).fit()
                b = np.asarray(fit.params)
                bread = np.asarray(fit.normalized_cov_params)
                covco, covsr, covtwo = covariance(fit, g)
                G = g.country_iso3.nunique()
                df = G - 1
                audits.append(
                    {
                        "endpoint": e,
                        "scenario": scenario,
                        "mode": mode,
                        "n": len(g),
                        "countries": G,
                        "sources": g.source_id.nunique(),
                        "columns": P,
                        "rank": np.linalg.matrix_rank(A * np.sqrt(w)[:, None]),
                        "condition_number": np.linalg.cond(A * np.sqrt(w)[:, None]),
                        "R2_in_sample": fit.rsquared,
                        "R2_adjusted": fit.rsquared_adj,
                        "weighted_fit_out_of_0_1": (
                            np.average(
                                ((A @ b * ys + ym) < 0) | ((A @ b * ys + ym) > 1),
                                weights=w,
                            )
                            if e != "RQ_res"
                            else np.nan
                        ),
                        "dropped_redundant_nuisance_columns": "|".join(dropped),
                        "normal_equation_max_error": float(
                            np.max(abs(A.T @ (w * (y - A @ b))))
                        ),
                        "two_way_cov_min_eigenvalue": float(
                            np.linalg.eigvalsh(covtwo).min()
                        ),
                    }
                )
                if scenario == "primary":
                    pd.DataFrame(A, columns=list(X)).to_csv(
                        OUTPUTS / f"{e}_{mode}_primary_design.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    for s in scale:
                        scaling.append(
                            dict(endpoint=e, scenario=scenario, mode=mode, **s)
                        )
                    scaling.append(
                        {
                            "endpoint": e,
                            "scenario": scenario,
                            "mode": mode,
                            "variable": "OUTCOME",
                            "mean_transformed": ym,
                            "SD_transformed": ys,
                        }
                    )
                for k, c in enumerate(X):
                    allcoefs.append(
                        {
                            "endpoint": e,
                            "scenario": scenario,
                            "mode": mode,
                            "term": c,
                            "beta": b[k],
                            "SE_country": np.sqrt(max(covco[k, k], 0)),
                        }
                    )
                for v in vars:
                    j = list(X).index(v)
                    se = np.sqrt(max(covco[j, j], 0))
                    lo, hi, p = infer(b[j], se, df)
                    ser = np.sqrt(max(covsr[j, j], 0))
                    setwo = np.sqrt(covtwo[j, j]) if covtwo[j, j] > 0 else np.nan
                    r = {
                        "endpoint": e,
                        "scenario": scenario,
                        "mode": mode,
                        "indicator": v,
                        "beta_standardized": b[j],
                        "CI95_low": lo,
                        "CI95_high": hi,
                        "SE_country": se,
                        "p_country": p,
                        "direction": "positive" if b[j] > 0 else "negative",
                        "effect_original_outcome_per_WGI_SD": b[j] * ys,
                        "effect_original_low": lo * ys,
                        "effect_original_high": hi * ys,
                        "n_records": len(g),
                        "n_countries": G,
                        "n_sources": g.source_id.nunique(),
                        "df": df,
                        "SE_source": ser,
                        "p_source": infer(b[j], ser, g.source_id.nunique() - 1)[2],
                        "SE_two_way": setwo,
                        "p_two_way": (
                            infer(b[j], setwo, df)[2] if np.isfinite(setwo) else np.nan
                        ),
                    }
                    # Conditional VIF uses weighted projection on all other actual columns.
                    rest = np.delete(A, j, axis=1)
                    z = A[:, j]
                    bz = np.linalg.lstsq(
                        rest * np.sqrt(w)[:, None], z * np.sqrt(w), rcond=None
                    )[0]
                    zerr = z - rest @ bz
                    vif = np.sum(w * (z - np.average(z, weights=w)) ** 2) / np.sum(
                        w * zerr * zerr
                    )
                    r["conditional_VIF"] = vif
                    r["partial_R2"] = (
                        b[j] ** 2
                        * np.sum(w * zerr * zerr)
                        / (
                            np.sum(w * fit.resid**2)
                            + b[j] ** 2 * np.sum(w * zerr * zerr)
                        )
                    )
                    vifs.append(
                        {
                            "endpoint": e,
                            "scenario": scenario,
                            "mode": mode,
                            "indicator": v,
                            "VIF": vif,
                            "residualized_SD": np.sqrt(
                                np.average(zerr * zerr, weights=w)
                            ),
                        }
                    )
                    if scenario == "primary":
                        r["p_wild_country_3999"] = wild_p(X, y, w, g, j, bread, b)
                    results.append(r)
                if scenario == "primary":
                    reduced = []
                    for country in sorted(g.country_iso3.unique()):
                        keep = g.country_iso3.ne(country).to_numpy()
                        AA = A[keep]
                        ww = w[keep]
                        yy = y[keep]
                        br = np.linalg.lstsq(
                            AA * np.sqrt(ww)[:, None], yy * np.sqrt(ww), rcond=None
                        )[0]
                        for v in vars:
                            j = list(X).index(v)
                            other = np.delete(AA, j, axis=1)
                            z = AA[:, j]
                            zz = (
                                z
                                - other
                                @ np.linalg.lstsq(
                                    other * np.sqrt(ww)[:, None],
                                    z * np.sqrt(ww),
                                    rcond=None,
                                )[0]
                            )
                            identified = np.sum(ww * zz * zz) > 1e-9
                            loco.append(
                                {
                                    "endpoint": e,
                                    "mode": mode,
                                    "indicator": v,
                                    "omitted_country": country,
                                    "beta_standardized": (
                                        br[j] if identified else np.nan
                                    ),
                                    "identifiable": identified,
                                    "n_records_remaining": sum(keep),
                                    "design_rank": np.linalg.matrix_rank(
                                        AA * np.sqrt(ww)[:, None]
                                    ),
                                }
                            )
                    for v in vars:
                        vals = np.array(
                            [
                                row["beta_standardized"]
                                for row in loco
                                if row["endpoint"] == e
                                and row["mode"] == mode
                                and row["indicator"] == v
                            ],
                            float,
                        )
                        j = list(X).index(v)
                        if np.isfinite(vals).all():
                            sej = np.sqrt(
                                (G - 1) / G * np.sum((vals - vals.mean()) ** 2)
                            )
                            cl, ch, pj = infer(b[j], sej, G - 1)
                        else:
                            sej = cl = ch = pj = np.nan
                        jack.append(
                            {
                                "endpoint": e,
                                "mode": mode,
                                "indicator": v,
                                "beta_standardized": b[j],
                                "country_jackknife_SE": sej,
                                "CI95_low": cl,
                                "CI95_high": ch,
                                "p_country_jackknife": pj,
                                "LOCO_min": np.nanmin(vals),
                                "LOCO_max": np.nanmax(vals),
                                "LOCO_same_sign_fraction": np.nanmean(
                                    np.sign(vals) == np.sign(b[j])
                                ),
                                "all_LOCO_identifiable": bool(np.isfinite(vals).all()),
                            }
                        )
                print(e, scenario, mode, len(g), G, flush=True)
        write(results, "governance_associations_stage.csv")
        write(audits, "governance_fit_diagnostics.csv")
        write(coverage, "governance_analysis_coverage.csv")
        write(vifs, "governance_conditional_VIF.csv")
        write(loco, "governance_leave_one_country_out.csv")
        write(jack, "governance_country_jackknife.csv")
        # Link function sensitivity, same complete covariate subset and category design.
        if e != "RQ_res":
            g, w = select(raw, e, "primary")
            yr = outcome(g, e)
            for mode, vars in [
                ("GE_only", [GE]),
                ("CC_only", [CC]),
                ("joint", [GE, CC]),
            ]:
                X, _, _ = design(g, e, "primary", w, vars)
                A = X.to_numpy(float)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = sm.GLM(
                        yr, A, family=sm.families.Binomial(), freq_weights=w
                    ).fit(maxiter=300, tol=1e-10)
                covco, covsr, covtwo = covariance(fit, g)
                df = g.country_iso3.nunique() - 1
                for v in vars:
                    j = list(X).index(v)
                    se = np.sqrt(max(covco[j, j], 0))
                    lo, hi, p = infer(fit.params[j], se, df)
                    glmrows.append(
                        {
                            "endpoint": e,
                            "mode": mode,
                            "indicator": v,
                            "logit_coefficient_per_WGI_SD": fit.params[j],
                            "ratio": np.exp(fit.params[j]),
                            "CI95_ratio_low": np.exp(lo),
                            "CI95_ratio_high": np.exp(hi),
                            "p_country": p,
                            "converged": fit.converged,
                            "max_abs_coefficient": np.max(abs(fit.params)),
                            "warnings": "|".join(
                                sorted(set(str(x.message) for x in caught))
                            ),
                            "n_records": len(g),
                            "n_countries": g.country_iso3.nunique(),
                            "interpretation": (
                                "fractional-mean odds ratio"
                                if e == "occurrence"
                                else "odds ratio"
                            ),
                            "risk_of_separation": bool(np.max(abs(fit.params)) > 20),
                        }
                    )
    res = pd.DataFrame(results)
    res["q_BH"] = np.nan
    for joint in [False, True]:
        idx = res.index[
            (res.scenario == "primary") & (res["mode"].eq("joint") == joint)
        ]
        res.loc[idx, "q_BH"] = multipletests(
            res.loc[idx, "p_country"], method="fdr_bh"
        )[1]
    save(res, "governance_adjusted_associations.csv")
    write(audits, "governance_fit_diagnostics.csv")
    write(coverage, "governance_analysis_coverage.csv")
    write(scaling, "governance_standardization_parameters.csv")
    write(vifs, "governance_conditional_VIF.csv")
    write(loco, "governance_leave_one_country_out.csv")
    write(jack, "governance_country_jackknife.csv")
    write(glmrows, "governance_logit_link_sensitivity.csv")
    write(allcoefs, "governance_all_coefficients.csv")
    # Country-period descriptive correlations, de-duplicated by exact matched exposure/covariate period.
    corr = []
    for e in END:
        g, w = select(read("governance_" + e + "_input.csv"), e, "primary")
        p = g.drop_duplicates(
            ["country_iso3", "sampling_year_start", "sampling_year_end", GE, CC] + CORE
        )
        for a in [GE, CC]:
            for b in [GE, CC] + CORE[:-1]:
                if a == b:
                    continue
                corr.append(
                    {
                        "endpoint": e,
                        "x": a,
                        "y": b,
                        "Spearman_rho": stats.spearmanr(p[a], p[b]).statistic,
                        "n_country_periods": len(p),
                        "n_countries": p.country_iso3.nunique(),
                    }
                )
    write(corr, "governance_country_period_correlations.csv")
