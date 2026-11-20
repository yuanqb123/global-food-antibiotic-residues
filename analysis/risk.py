"""Dietary, jurisdictional MRL and exploratory resistance-selection screening."""

import numpy as np
import pandas as pd
from .common import read, save, load_config


def run():
    h = read("HQ_positive_means.csv")
    h["EDI_ugkg_bw_day"] = h.concentration_ugkg * h.food_intake_kg_day / h.BW_kg
    h["HQ"] = h.EDI_ugkg_bw_day / h.ADI_ugkg_bw_day
    assert np.allclose(h.HQ, h.HQ_current, rtol=1e-12, atol=1e-12)
    save(h, "dietary_HQ_records.csv")
    m = read("MRL_mean_statistics.csv")
    m = m[m.statistic_type.eq("positive_mean")].copy()
    m["C_MRL"] = m.concentration_ugkg / m.MRL_ugkg
    assert np.allclose(m.C_MRL, m.MRL_quotient, rtol=1e-12, atol=1e-12)
    save(m, "regulatory_MRL_records.csv")
    r = read("RQ_positive_means.csv")
    r["RQ"] = (
        r.concentration_ugkg
        * r.food_intake_kg_day
        / (r.V_colon_L_day * r.F * r.PNEC_res_ug_L)
    )
    assert np.allclose(r.RQ, r.RQ_res, rtol=1e-12, atol=1e-12)
    summaries = []
    for F in [1, 10, 100, 1000]:
        z = r.copy()
        z["scenario_F"] = F
        z["RQ"] = (
            z.concentration_ugkg
            * z.food_intake_kg_day
            / (z.V_colon_L_day * F * z.PNEC_res_ug_L)
        )
        save(z, "resistance_selection_F" + str(F) + ".csv")
        summaries.append(
            dict(
                endpoint="RQ_res",
                F=F,
                n=len(z),
                above_0_1=int((z.RQ > 0.1).sum()),
                above_1=int((z.RQ > 1).sum()),
                median=z.RQ.median(),
            )
        )
    summaries += [
        dict(
            endpoint="HQ",
            n=len(h),
            above_0_1=int((h.HQ > 0.1).sum()),
            above_1=int((h.HQ > 1).sum()),
            median=h.HQ.median(),
        ),
        dict(endpoint="C_MRL", n=len(m), above_1=int((m.C_MRL > 1).sum())),
    ]
    save(pd.DataFrame(summaries), "risk_summary.csv")
    paired = read("paired_HQ_RQ_means.csv")
    hq = paired.HQ_current > 1
    rq = paired.RQ_res > 1
    paired["category"] = np.select(
        [hq & rq, hq & ~rq, ~hq & rq], ["both", "HQ_only", "RQ_only"], default="neither"
    )
    save(paired, "paired_risk_classifications.csv")
    counts = paired.category.value_counts().reindex(
        ["neither", "RQ_only", "HQ_only", "both"], fill_value=0
    )
    assert counts.to_list() == [265, 167, 0, 14]
    assert len(h) == 512 and len(m) == 57 and len(r) == 658
    assert int((h.HQ > 1).sum()) == 14 and int((m.C_MRL > 1).sum()) == 12
    save(
        counts.rename_axis("category").reset_index(name="records"),
        "paired_risk_counts.csv",
    )
    # Independent threshold-reference sensitivity; environmental PNECs remain scenario references.
    thresholds = read("resistance_selection_thresholds.csv")
    comparisons = []
    for version, t in thresholds.groupby("version"):
        p = t.set_index("antibiotic_std").PNEC_res_ug_L
        for F in [1, 10, 100, 1000]:
            z = r.copy()
            z["reference_version"] = version
            z["scenario_F"] = F
            z["PNEC_scenario_ug_L"] = z.antibiotic_std.map(p)
            z["RQ_scenario"] = (
                z.concentration_ugkg
                * z.food_intake_kg_day
                / (0.5 * F * z.PNEC_scenario_ug_L)
            )
            eligible = z.RQ_scenario.notna()
            comparisons.append(
                dict(
                    reference_version=version,
                    F=F,
                    eligible=int(eligible.sum()),
                    above_1=int(z.RQ_scenario.gt(1).sum()),
                    above_0_1=int(z.RQ_scenario.gt(0.1).sum()),
                    median=z.RQ_scenario.median(),
                )
            )
    save(pd.DataFrame(comparisons), "PNEC_reference_sensitivity.csv")
    allmean = read("HQ_all_sample_means.csv")
    allmean["HQ"] = (
        allmean.concentration_ugkg
        * allmean.food_intake_kg_day
        / (allmean.BW_kg * allmean.ADI_ugkg_bw_day)
    )
    assert np.allclose(allmean.HQ, allmean.HQ_current)
    save(allmean, "all_sample_mean_HQ_sensitivity.csv")
    mean = read("risk_mean_eligibility.csv")
    save(
        mean.groupby(["statistic_type", "final_HQ_disposition"], dropna=False)
        .size()
        .reset_index(name="statistics"),
        "risk_mean_eligibility_counts.csv",
    )
    rules = read("national_MRL_rules.csv")
    save(
        rules.groupby("country_iso3")
        .agg(rules=("MRL_ugkg", "size"), standards=("standard_version", "nunique"))
        .reset_index(),
        "national_MRL_inventory_counts.csv",
    )
    # Endpoint labels must agree with the explicit current risk parameters.
    model = read("HQ_MRL_model_input.csv")
    hq = (
        pd.concat([h[["statistic_id", "HQ"]], allmean[["statistic_id", "HQ"]]])
        .set_index("statistic_id")
        .HQ
    )
    allm = read("MRL_mean_statistics.csv")
    mr = (allm.concentration_ugkg / allm.MRL_ugkg).set_axis(allm.statistic_id)
    flag = model.statistic_id.map(hq).gt(0.1) | model.statistic_id.map(mr).gt(1)
    assert np.array_equal(flag, model.available_path_attention_trigger.astype(bool))
    rqmodel = read("RQ_res_model_input.csv")
    rq = r.set_index("statistic_id").RQ
    recomputed = rqmodel.statistic_id.map(rq)
    assert recomputed.notna().all() and np.allclose(
        recomputed, rqmodel.RQ_res, rtol=1e-12, atol=1e-12
    )
    save(
        pd.DataFrame(
            [
                dict(
                    endpoint="HQ_MRL",
                    model_records=len(model),
                    matched_records=len(model),
                    inconsistent_targets=int(
                        flag.ne(
                            model.available_path_attention_trigger.astype(bool)
                        ).sum()
                    ),
                ),
                dict(
                    endpoint="RQ_res",
                    model_records=len(rqmodel),
                    matched_records=int(recomputed.notna().sum()),
                    inconsistent_targets=int(
                        (
                            ~np.isclose(
                                recomputed, rqmodel.RQ_res, rtol=1e-12, atol=1e-12
                            )
                        ).sum()
                    ),
                ),
            ]
        ),
        "model_target_risk_reconciliation.csv",
    )
    return summaries
