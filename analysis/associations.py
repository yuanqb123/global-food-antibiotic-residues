"""Weighted fixed-effect associations with country-cluster inference."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.linalg import qr
from statsmodels.stats.multitest import multipletests
from .common import load_config, save
from .models import load
from .weighting import weights

CFG = load_config()
vv = CFG["national_indicators"]
trans = CFG["indicator_transforms"]
aq = CFG["aquatic_only"]


def run():
    rows = []
    for ep in ["RQ_res", "HQ_MRL"]:
        g = load(ep)
        ww = weights(g)
        nat = g[vv].copy()
        cats_list = [
            "food_group",
            "antibiotic_std",
            "statistic_type",
            "sample_source",
            "analytical_method",
            "sampling_design",
        ]
        for v in vv:
            if trans.get(v) == "log1p_z":
                nat[v] = np.log1p(nat[v])
        # Separate observed-indicator associations adjusted for all original categorical controls and year.
        for v in vv:
            ok = nat[v].notna()
            h = g.loc[ok].copy()
            x = nat.loc[ok, v]
            sd = x.std()
            nc = h.country_iso3.nunique()
            cats = pd.get_dummies(
                h[cats_list].fillna("missing").astype(str), drop_first=True, dtype=float
            )
            other = h[["year_mid"]].fillna(h.year_mid.median())
            xx = pd.concat(
                [
                    pd.Series(1.0, index=h.index, name="intercept"),
                    ((x - x.mean()) / sd).rename("indicator_SD"),
                    cats,
                    other,
                ],
                axis=1,
            )
            # Retain the target column first; remove linearly dependent controls deterministically.
            keep = []
            rank = 0
            for j in range(xx.shape[1]):
                rr = np.linalg.matrix_rank(xx.iloc[:, keep + [j]].to_numpy(), tol=1e-8)
                if rr > rank:
                    keep.append(j)
                    rank = rr
            xx = xx.iloc[:, keep]
            y = (
                np.log10(1 + h.RQ_res)
                if ep == "RQ_res"
                else h.available_path_attention_trigger.astype(float)
            )
            fit = sm.WLS(y, xx, weights=ww[ok]).fit(
                cov_type="cluster",
                cov_kwds={"groups": h.country_iso3, "use_correction": True},
                use_t=True,
            )
            ci = fit.conf_int().loc["indicator_SD"]
            mul = 1 if ep == "RQ_res" else 100
            if nc < 10 or len(h) - xx.shape[1] < nc:
                ci.iloc[:] = np.nan
            rows.append(
                dict(
                    endpoint=ep,
                    variable=v,
                    n=len(h),
                    countries=nc,
                    sources=h.source_id.nunique(),
                    effect=fit.params.indicator_SD * mul,
                    low=ci.iloc[0] * mul,
                    high=ci.iloc[1] * mul,
                    p=(
                        np.nan
                        if nc < 10 or len(h) - xx.shape[1] < nc
                        else fit.pvalues.indicator_SD
                    ),
                    parameters=xx.shape[1],
                    country_df=nc - 1,
                    sd=sd,
                )
            )
    res = pd.DataFrame(rows)
    for ep in res.endpoint.unique():
        ix = res.endpoint.eq(ep)
        res.loc[ix, "q_BH"] = multipletests(
            res.loc[ix, "p"].fillna(1), method="fdr_bh"
        )[1]
        res.loc[ix & res.p.isna(), "q_BH"] = np.nan
    save(res, "fixed_effect_associations.csv")
