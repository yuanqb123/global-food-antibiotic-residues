"""Matched-food contrasts and the controlled common-intake sensitivity."""

import numpy as np
import pandas as pd
from scipy import stats
from .common import read, save
from .weighting import weights
from .statistics import design, wls


def run():
    who = read("WHO_regions.csv")[["country_iso3", "display_region"]]
    contrast = []
    profiles = []
    for endpoint, g, value in [
        ("detection", read("dairy_meat_detection.csv"), "rate"),
        (
            "positive_concentration",
            read("dairy_meat_concentration.csv"),
            "concentration_ugkg",
        ),
    ]:
        z = g[
            g.food_group.isin(["milk/dairy", "meat"]) & g.antibiotic_std.notna()
        ].copy()
        for food, h in z.groupby("food_group"):
            for drug, j in h.groupby("antibiotic_std"):
                profiles.append(
                    dict(
                        endpoint=endpoint,
                        food_group=food,
                        drug=drug,
                        units=len(j),
                        sources=j.source_id.nunique(),
                        tests=(
                            j.sample_size_n.sum() if endpoint == "detection" else np.nan
                        ),
                        k=(
                            j.positive_count_n.sum()
                            if endpoint == "detection"
                            else np.nan
                        ),
                        median=j[value].median(),
                    )
                )
        common = set(z[z.food_group.eq("meat")].antibiotic_std) & set(
            z[z.food_group.eq("milk/dairy")].antibiotic_std
        )
        z = z[
            z.antibiotic_std.isin(common) & z.country_iso3.notna() & z.source_id.notna()
        ].reset_index(drop=True)
        if "display_region" not in z:
            z = z.merge(who, on="country_iso3", how="left")
        for region in ["drug_only", "drug_region_year"]:
            cats = ["antibiotic_std"] + (
                ["display_region"] if region == "drug_region_year" else []
            )
            num = {"dairy": z.food_group.eq("milk/dairy").astype(float)}
            if region == "drug_region_year":
                num.update(
                    year5=(z.year_mid.fillna(2010) - 2010) / 5,
                    year_missing=z.year_mid.isna().astype(float),
                )
            X = design(z, cats, num)
            y = (
                z.rate.to_numpy()
                if endpoint == "detection"
                else np.log(z.concentration_ugkg.to_numpy())
            )
            w = weights(z, endpoint == "detection")
            for cluster in ["source_id", "country_iso3"]:
                fit, cov, se, G = wls(z, y, X, w, cluster)
                i = X.columns.get_loc("dairy")
                b = fit.params.iloc[i]
                crit = stats.t.ppf(0.975, G - 1)
                lo = b - crit * se[i]
                hi = b + crit * se[i]
                contrast.append(
                    dict(
                        endpoint=endpoint,
                        specification=region,
                        inference_cluster=cluster,
                        n_records=len(z),
                        n_sources=z.source_id.nunique(),
                        n_countries=z.country_iso3.nunique(),
                        common_drugs=len(common),
                        beta=b,
                        CI_low=lo,
                        CI_high=hi,
                        p=2 * stats.t.sf(abs(b / se[i]), G - 1),
                        effect=100 * b if endpoint == "detection" else np.exp(b),
                        effect_low=100 * lo if endpoint == "detection" else np.exp(lo),
                        effect_high=100 * hi if endpoint == "detection" else np.exp(hi),
                        effect_scale=(
                            "adjusted percentage points dairy-minus-meat"
                            if endpoint == "detection"
                            else "geometric mean concentration ratio dairy/meat"
                        ),
                    )
                )
        save(z, "dairy_meat_" + endpoint + "_common_drug_records")
    save(pd.DataFrame(profiles), "dairy_meat_drug_profiles")
    save(pd.DataFrame(contrast), "dairy_meat_contrasts.csv")
    z = read("dairy_meat_intake.csv")
    rows = []
    for metric, field in [("RQ_res", "RQ_res"), ("HQ", "HQ_same_mean")]:
        q = z[z[field].notna() & z[field].gt(0)].reset_index(drop=True)
        w = weights(q)
        X = design(
            q,
            ["antibiotic_std", "display_region"],
            dict(
                dairy=q.food_group.eq("milk/dairy").astype(float),
                year5=(q.year_mid.fillna(2010) - 2010) / 5,
                year_missing=q.year_mid.isna().astype(float),
            ),
        )
        for scenario, y in [
            ("actual_reference_intakes", np.log(q[field])),
            ("common_0_3kg_intake", np.log(q[field] * 0.3 / q.food_intake_kg_day)),
            ("intake_contribution", np.log(q.food_intake_kg_day / 0.3)),
        ]:
            for cluster in ["country_iso3", "source_id"]:
                fit, cov, se, G = wls(q, y, X, w, cluster)
                i = X.columns.get_loc("dairy")
                b = fit.params.iloc[i]
                crit = stats.t.ppf(0.975, G - 1)
                rows.append(
                    dict(
                        metric=metric,
                        scenario=scenario,
                        cluster=cluster,
                        n_records=len(q),
                        countries=q.country_iso3.nunique(),
                        sources=q.source_id.nunique(),
                        common_drugs=q.antibiotic_std.nunique(),
                        beta_log=b,
                        ratio_dairy_meat=np.exp(b),
                        CI_low=np.exp(b - crit * se[i]),
                        CI_high=np.exp(b + crit * se[i]),
                        p=2 * stats.t.sf(abs(b / se[i]), G - 1),
                    )
                )
    save(pd.DataFrame(rows), "dairy_meat_intake_contrasts.csv")
