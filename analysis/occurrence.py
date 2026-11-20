"""Pooled analyte-test detection and the prespecified common-drug panel."""

import numpy as np
import pandas as pd
from .common import read, save


def run():
    oc = read("occurrence_units.csv")
    oc["source_id"] = oc.DOI.fillna(oc.study_id)
    oc["statistic_id"] = oc.canonical_row.astype(str)
    oc = oc.merge(
        read("WHO_regions.csv")[["country_iso3", "display_region"]],
        on="country_iso3",
        how="left",
        validate="many_to_one",
    )
    for group in [
        "food_group",
        "antibiotic_std",
        "antibiotic_class_broad",
        "display_region",
        "country_iso3",
    ]:
        table = (
            oc.groupby(group)
            .agg(
                k=("positive_count_n", "sum"),
                n=("sample_size_n", "sum"),
                units=("canonical_row", "size"),
                sources=("source_id", "nunique"),
            )
            .reset_index()
        )
        table["detection_rate"] = table.k / table.n
        p = table.detection_rate
        n = table.n
        z = 1.96
        mid = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        table["CI_low"] = (mid - half).clip(0, 1)
        table["CI_high"] = (mid + half).clip(0, 1)
        save(table, "pooled_" + group + ".csv")
    r = (
        oc[oc.antibiotic_std.notna()]
        .groupby("antibiotic_std")
        .agg(
            tests=("sample_size_n", "sum"),
            regions=("display_region", "nunique"),
            foods=("food_group", "nunique"),
            sources=("source_id", "nunique"),
        )
        .sort_values(["tests"], ascending=False)
    )
    panel = r[(r.regions == 6) & (r.foods == 6)].head(3).index.tolist()
    assert panel == ["tetracycline", "doxycycline", "chloramphenicol"]
    r["selected"] = r.index.isin(panel)
    save(r.reset_index(), "common_panel_selection_without_outcomes")
    pp = oc[oc.antibiotic_std.isin(panel)].copy()
    save(pp, "common_panel_three_drug_records")
    cells = []
    out = []
    for dim in ["display_region", "food_group"]:
        for val, g in oc.groupby(dim):
            z = g[g.antibiotic_std.isin(panel)]
            a = (
                z.groupby("antibiotic_std")
                .agg(
                    k=("positive_count_n", "sum"),
                    n=("sample_size_n", "sum"),
                    units=("canonical_row", "size"),
                    sources=("source_id", "nunique"),
                )
                .reindex(panel)
            )
            assert a.n.notna().all()
            for drug, row in a.iterrows():
                cells.append(
                    dict(
                        dimension=dim,
                        group=val,
                        drug=drug,
                        **row.to_dict(),
                        rate=row.k / row.n
                    )
                )
            est = float((a.k / a.n).mean())
            raw = g.positive_count_n.sum() / g.sample_size_n.sum()
            # Resample entire contributing source blocks including all drug rows; require complete panel in replicate.
            block = z.groupby(["source_id", "antibiotic_std"])[
                ["positive_count_n", "sample_size_n"]
            ].sum()
            keys = sorted(z.source_id.unique())
            K = np.zeros((len(keys), 3))
            N = K.copy()
            for i, s in enumerate(keys):
                for j, d in enumerate(panel):
                    if (s, d) in block.index:
                        K[i, j], N[i, j] = block.loc[(s, d)].to_numpy()
            rng = np.random.default_rng(20261022)
            bs = []
            for b in range(1999):
                ix = rng.integers(0, len(keys), len(keys))
                kk = K[ix].sum(0)
                nn = N[ix].sum(0)
                if (nn > 0).all():
                    bs.append(np.mean(kk / nn))
            out.append(
                dict(
                    dimension=dim,
                    group=val,
                    panel_rate=est,
                    CI_low=np.quantile(bs, 0.025),
                    CI_high=np.quantile(bs, 0.975),
                    all_analyte_rate=raw,
                    difference_pp=100 * (est - raw),
                    tests=a.n.sum(),
                    units=len(z),
                    sources=z.source_id.nunique(),
                    countries=z.country_iso3.nunique(),
                    drugs=3,
                    bootstrap_valid=len(bs),
                )
            )
    out = pd.DataFrame(out)
    for dim in out.dimension.unique():
        ix = out.dimension.eq(dim)
        out.loc[ix, "rank_all"] = out.loc[ix, "all_analyte_rate"].rank(
            ascending=False, method="min"
        )
        out.loc[ix, "rank_panel"] = out.loc[ix, "panel_rate"].rank(
            ascending=False, method="min"
        )
    save(pd.DataFrame(cells), "common_panel_drug_cells")
    save(out, "common_panel_panel_comparison")
    print("Common-drug panel", len(pp), len(out), flush=True)
