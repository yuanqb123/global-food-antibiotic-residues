"""Spatial composition standardisation and five-year descriptive summaries."""

import numpy as np
import pandas as pd
from .common import read, save


def run():
    occ = read("occurrence_units.csv").fillna(
        {"food_class_std": "", "antibiotic_std": "", "DOI": ""}
    )
    cs = occ[(occ.food_class_std.ne("")) & (occ.antibiotic_std.ne(""))].copy()
    ref = (
        cs.groupby(["food_group", "antibiotic_std"])
        .agg(
            positive_count=("positive_count_n", "sum"),
            sample_size=("sample_size_n", "sum"),
        )
        .reset_index()
    )
    ref["reference_rate"] = (ref.positive_count + 0.5) / (ref.sample_size + 1)
    cs = cs.merge(
        ref[["food_group", "antibiotic_std", "reference_rate"]],
        on=["food_group", "antibiotic_std"],
    )
    global_rate = cs.positive_count_n.sum() / cs.sample_size_n.sum()
    adj = []
    rng = np.random.default_rng(20260930)
    for iso, g in cs.groupby("country_iso3"):
        observed = g.positive_count_n.sum()
        expected = (g.sample_size_n * g.reference_rate).sum()
        p = min(1.0, global_rate * observed / expected) if expected > 0 else np.nan
        block = (
            g.assign(expected_component=g.sample_size_n * g.reference_rate)
            .groupby("study_id")
            .agg(
                k=("positive_count_n", "sum"),
                n=("sample_size_n", "sum"),
                e=("expected_component", "sum"),
            )
        )
        bk = block.k.to_numpy()
        bn = block.n.to_numpy()
        be = block.e.to_numpy()
        m = len(block)
        draw_idx = rng.integers(0, m, size=(400, m))
        beta_draw = rng.beta(bk[draw_idx] + 0.5, bn[draw_idx] - bk[draw_idx] + 0.5)
        numerator = (beta_draw * bn[draw_idx]).sum(axis=1)
        denominator = be[draw_idx].sum(axis=1)
        draws = np.minimum(1.0, global_rate * numerator / denominator)
        ci_low, ci_high = np.quantile(draws, [0.025, 0.975])
        ci_low = min(ci_low, p)
        ci_high = max(ci_high, p)
        adj.append(
            {
                "country_iso3": iso,
                "adjusted_occurrence_exploratory": p,
                "raw_rate": observed / g.sample_size_n.sum(),
                "CI_95_low_exploratory": ci_low,
                "CI_95_high_exploratory": ci_high,
                "positive_count": observed,
                "sample_size": g.sample_size_n.sum(),
                "n_DOI": g.DOI[g.DOI.ne("")].nunique(),
                "n_studies": g.study_id.nunique(),
                "n_frames": g[["study_id", "sampling_frame_id"]]
                .drop_duplicates()
                .shape[0],
                "n_units": len(g),
                "uncertainty_scope": (
                    "single-study within-sample only"
                    if m == 1
                    else "study-block bootstrap plus within-block beta"
                ),
                "method": "indirect food x antibiotic standardization; all countries shown; uncertainty depends on study/frame coverage",
            }
        )
    save(pd.DataFrame(adj), "country_standardised_detection.csv")

    t = occ[occ.year_mid.notna()].copy()
    t["source_id"] = t.DOI.where(t.DOI.ne(""), t.study_id)
    t["period_start"] = (t.year_mid // 5 * 5).astype(int)
    c = read("dated_positive_means.csv")
    c["period_start"] = (c.year_mid // 5 * 5).astype(int)
    rng = np.random.default_rng(71004)
    rows = []
    intervals = []
    for period, g in t.groupby("period_start"):
        a = (
            g.groupby("source_id")[["positive_count_n", "sample_size_n"]]
            .sum()
            .to_numpy(float)
        )
        k, n = a.sum(axis=0)
        rate = k / n
        if len(a) < 2:
            z = 1.96
            m = (rate + z * z / (2 * n)) / (1 + z * z / n)
            h = (
                z
                * np.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n))
                / (1 + z * z / n)
            )
            lo, hi = np.clip(m - h, 0, 1), np.clip(m + h, 0, 1)
            method = "Wilson_conditional_single_source"
        else:
            ix = rng.integers(0, len(a), size=(1500, len(a)))
            bs = a[ix].sum(axis=1)
            lo, hi = np.quantile(bs[:, 0] / bs[:, 1], [0.025, 0.975])
            method = "source_cluster_bootstrap_1500"
        intervals.append(
            dict(
                period_start=period,
                rate=rate,
                CI_low=lo,
                CI_high=hi,
                n_sources=g.source_id.nunique(),
                interval_method=method,
            )
        )
    for ep, g in [("detection", t), ("concentration", c)]:
        for food, z in [("all_foods", g)] + list(g.groupby("food_group")):
            for period, h in z.groupby("period_start"):
                d = dict(
                    endpoint=ep,
                    food_group=food,
                    period_start=int(period),
                    n_records=len(h),
                    n_countries=h.country_iso3.nunique(),
                )
                if ep == "detection":
                    d.update(
                        k=h.positive_count_n.sum(),
                        n=h.sample_size_n.sum(),
                        estimate=h.positive_count_n.sum() / h.sample_size_n.sum(),
                    )
                else:
                    d.update(
                        estimate=h.concentration_ugkg.median(),
                        q25=h.concentration_ugkg.quantile(0.25),
                        q75=h.concentration_ugkg.quantile(0.75),
                    )
                rows.append(d)
    save(pd.DataFrame(rows), "five_year_food_summaries.csv")
    save(pd.DataFrame(intervals), "five_year_detection_intervals.csv")
