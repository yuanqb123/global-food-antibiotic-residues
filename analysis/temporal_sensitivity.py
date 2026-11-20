"""Global and China food-specific trends and interaction sensitivity."""

import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from .common import read, save
from .weighting import weights
from .statistics import design, wls, bh


def run():
    oc = read("occurrence_units.csv")
    oc["source_id"] = oc.DOI.fillna(oc.study_id)
    oc["statistic_id"] = oc.canonical_row.astype(str)
    who = read("WHO_regions.csv")[["country_iso3", "display_region"]]
    oc = oc.merge(who, on="country_iso3", how="left", validate="many_to_one")
    c = read("dated_positive_means.csv")
    c = c[c.country_iso3.notna() & c.source_id.notna()].copy()
    if "display_region" not in c:
        c = c.merge(who, on="country_iso3", how="left")
    a = oc[oc.year_mid.notna() & oc.country_iso3.notna()].copy()
    trends = []
    interactions = []
    annuals = []
    for ep, allg in [("detection", a), ("concentration", c)]:
        for scope, g0 in [
            ("global", allg),
            ("China", allg[allg.country_iso3.eq("CHN")]),
        ]:
            g0 = g0.reset_index(drop=True)
            save(g0, "" + ep + "_" + scope + "_records")
            for food, g in [("all_foods", g0)] + list(g0.groupby("food_group")):
                g = g.reset_index(drop=True)
                base = dict(
                    endpoint=ep,
                    scope=scope,
                    food_group=food,
                    n_records=len(g),
                    countries=g.country_iso3.nunique(),
                    sources=g.source_id.nunique(),
                    years=g.year_mid.nunique(),
                    first=g.year_mid.min(),
                    last=g.year_mid.max(),
                )
                cats = (
                    ["antibiotic_class_broad"]
                    + (["display_region"] if scope == "global" else [])
                    + (["food_group"] if food == "all_foods" else [])
                )
                X = design(g, cats, {"year5": (g.year_mid - 2010) / 5})
                w = weights(g, ep == "detection")
                y = (
                    g.rate.to_numpy()
                    if ep == "detection"
                    else np.log(g.concentration_ugkg.to_numpy())
                )
                cluster = "country_iso3" if scope == "global" else "source_id"
                G = g[cluster].nunique()
                try:
                    if G < 4 or "year5" not in X:
                        raise ValueError(
                            "fewer than 4 inference clusters or unidentified year effect"
                        )
                    if ep == "detection":
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fit = sm.GLM(
                                y, X, family=sm.families.Binomial(), freq_weights=w
                            ).fit(maxiter=200)
                        A = X.to_numpy()
                        mu = fit.fittedvalues.to_numpy()
                        bread = np.linalg.pinv(A.T @ ((w * mu * (1 - mu))[:, None] * A))
                        sc = A * (w * (y - mu))[:, None]
                        cs = (
                            pd.DataFrame(sc)
                            .groupby(g[cluster].to_numpy())
                            .sum()
                            .to_numpy()
                        )
                        cov = (
                            bread
                            @ (cs.T @ cs)
                            @ bread
                            * G
                            / (G - 1)
                            * (len(g) - 1)
                            / (len(g) - X.shape[1])
                        )
                        se = np.sqrt(np.maximum(np.diag(cov), 0))
                        converged = fit.converged
                    else:
                        fit, cov, se, G = wls(g, y, X, w, cluster)
                        converged = True
                    i = X.columns.get_loc("year5")
                    b = fit.params.iloc[i]
                    cr = stats.t.ppf(0.975, G - 1)
                    status = (
                        "estimable"
                        if converged and np.isfinite(se[i]) and 0 < se[i] < 10
                        else "unstable"
                    )
                    trends.append(
                        {
                            **base,
                            "status": status,
                            "cluster": cluster,
                            "clusters": G,
                            "beta": b,
                            "SE": se[i],
                            "ratio_per_5years": np.exp(b),
                            "CI_low": np.exp(b - cr * se[i]),
                            "CI_high": np.exp(b + cr * se[i]),
                            "p": 2 * stats.t.sf(abs(b / se[i]), G - 1),
                            "effect": (
                                "odds ratio"
                                if ep == "detection"
                                else "geometric mean concentration ratio"
                            ),
                            "design_rank": np.linalg.matrix_rank(X),
                            "design_columns": X.shape[1],
                        }
                    )
                except Exception as e:
                    trends.append({**base, "status": str(e)})
            # Explicit joint interaction: this tests differences between food slopes.
            g = g0
            cats = ["food_group", "antibiotic_class_broad"] + (
                ["display_region"] if scope == "global" else []
            )
            num = {"year5": (g.year_mid - 2010) / 5}
            levels = sorted(g.food_group.unique())
            ref = levels[0]
            for food in levels[1:]:
                num["interaction_" + food] = num["year5"] * g.food_group.eq(food)
            X = design(g, cats, num)
            w = weights(g, ep == "detection")
            y = (
                g.rate.to_numpy()
                if ep == "detection"
                else np.log(g.concentration_ugkg.to_numpy())
            )
            cluster = "country_iso3" if scope == "global" else "source_id"
            G = g[cluster].nunique()
            if ep == "detection":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = sm.GLM(
                        y, X, family=sm.families.Binomial(), freq_weights=w
                    ).fit(maxiter=200)
                A = X.to_numpy()
                mu = fit.fittedvalues.to_numpy()
                br = np.linalg.pinv(A.T @ ((w * mu * (1 - mu))[:, None] * A))
                sc = A * (w * (y - mu))[:, None]
                cs = pd.DataFrame(sc).groupby(g[cluster].to_numpy()).sum().to_numpy()
                cov = (
                    br
                    @ (cs.T @ cs)
                    @ br
                    * G
                    / (G - 1)
                    * (len(g) - 1)
                    / (len(g) - X.shape[1])
                )
            else:
                fit, cov, se, G = wls(g, y, X, w, cluster)
            ix = [i for i, v in enumerate(X.columns) if v.startswith("interaction_")]
            b = fit.params.to_numpy()[ix]
            cv = cov[np.ix_(ix, ix)]
            rank = np.linalg.matrix_rank(cv)
            Fstat = float(b @ np.linalg.pinv(cv) @ b / len(ix))
            interactions.append(
                dict(
                    endpoint=ep,
                    scope=scope,
                    n_records=len(g),
                    countries=g.country_iso3.nunique(),
                    sources=g.source_id.nunique(),
                    reference_food=ref,
                    F=Fstat,
                    df1=len(ix),
                    df2=G - 1,
                    cov_rank=rank,
                    p=stats.f.sf(Fstat, len(ix), G - 1),
                    status=(
                        "estimable"
                        if rank == len(ix)
                        else "singular_cluster_covariance"
                    ),
                )
            )
            # Continuous sampling midpoint displayed without false exact dates; connect annual calendar buckets.
            z = g0.copy()
            z["display_year"] = np.floor(z.year_mid).astype(int)
            for food, h in [("all_foods", z)] + list(z.groupby("food_group")):
                for yr, j in h.groupby("display_year"):
                    annuals.append(
                        dict(
                            endpoint=ep,
                            scope=scope,
                            food_group=food,
                            year=yr,
                            n_records=len(j),
                            sources=j.source_id.nunique(),
                            countries=j.country_iso3.nunique(),
                            estimate=(
                                j.positive_count_n.sum() / j.sample_size_n.sum()
                                if ep == "detection"
                                else j.concentration_ugkg.median()
                            ),
                            Q25=(
                                j.rate.quantile(0.25)
                                if ep == "detection"
                                else j.concentration_ugkg.quantile(0.25)
                            ),
                            Q75=(
                                j.rate.quantile(0.75)
                                if ep == "detection"
                                else j.concentration_ugkg.quantile(0.75)
                            ),
                            tests=(
                                j.sample_size_n.sum() if ep == "detection" else np.nan
                            ),
                        )
                    )
    tr = pd.DataFrame(trends)
    tr["q_BH_28_trends"] = bh(tr.p.where(tr.status.eq("estimable")))
    it = pd.DataFrame(interactions)
    it["q_BH_4_interactions"] = bh(it.p.where(it.status.eq("estimable")))
    an = pd.DataFrame(annuals)
    save(an, "annual_display_summary")

    g = tr
    if not (
        (g.endpoint == "concentration")
        & (g.scope == "China")
        & (g.food_group == "other animal-derived foods")
    ).any():
        g = pd.concat(
            [
                g,
                pd.DataFrame(
                    [
                        dict(
                            endpoint="concentration",
                            scope="China",
                            food_group="other animal-derived foods",
                            n_records=0,
                            countries=0,
                            sources=0,
                            years=0,
                            status="not represented",
                        )
                    ]
                ),
            ],
            ignore_index=True,
        )
    bad = g.status.eq("estimable") & (
        (g.SE.le(1e-8)) | ((g.n_records - g.design_rank).lt(10))
    )
    for field in ["p", "ratio_per_5years", "CI_low", "CI_high"]:
        g["raw_fit_" + field] = g[field]
    g.loc[bad, "status"] = "degenerate_cluster_variance_or_insufficient_residual_df"
    g.loc[~g.status.eq("estimable"), ["p", "ratio_per_5years", "CI_low", "CI_high"]] = (
        np.nan
    )
    g["q_BH_28_trends"] = bh(g.p.where(g.status.eq("estimable")).fillna(1))
    g.loc[~g.status.eq("estimable"), "q_BH_28_trends"] = np.nan
    assert len(g) == 28
    save(g, "annual_trend_effects")

    save(it, "food_year_interactions")
