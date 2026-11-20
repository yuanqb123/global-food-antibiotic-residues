"""Complete exploratory pathogen/drug family and prespecified within-species baskets."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from statsmodels.stats.multitest import multipletests
from .common import read, save


def run():
    match = read("clinical_all_country_matches.csv")
    raw = read("clinical_resistance_components.csv")
    raw = raw[
        raw.value.between(0, 100) & raw.ISO3.notna() & raw.year.notna()
    ].drop_duplicates(
        [
            "OrganismId",
            "AntimicrobialId",
            "ISO3",
            "year",
            "Source",
            "value",
            "IsolatesTested",
        ]
    )
    oc = read("occurrence_units.csv")
    food = oc.groupby("country_iso3").agg(
        k=("positive_count_n", "sum"), n=("sample_size_n", "sum")
    )
    food["raw_analyte_test_rate"] = food.k / food.n
    mrsa = match[
        match.Organism.str.contains("Staphylococcus aureus", case=False)
        & match.Antimicrobial.str.contains("Oxacillin|MRSA", case=False)
    ]
    primary = [
        (3, 7),
        (3, 6),
        (int(mrsa.OrganismId.iloc[0]), int(mrsa.AntimicrobialId.iloc[0])),
    ]
    rng = np.random.default_rng(20261117)
    rows = []
    sensitivity = []
    for (org, abx), g in match.groupby(["OrganismId", "AntimicrobialId"], sort=True):
        g = g.copy()
        n = len(g)
        if n < 10:
            continue
        x = g.food_detection_pct.to_numpy()
        y = g.value.to_numpy()
        rho, p = spearmanr(x, y)
        idx = rng.integers(0, n, size=(2000, n))
        xx = rankdata(x[idx], axis=1)
        yy = rankdata(y[idx], axis=1)
        xx -= xx.mean(axis=1, keepdims=True)
        yy -= yy.mean(axis=1, keepdims=True)
        boot = (xx * yy).sum(axis=1) / np.sqrt(
            (xx * xx).sum(axis=1) * (yy * yy).sum(axis=1)
        )
        lo, hi = np.nanquantile(boot, [0.025, 0.975])
        loo = [spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(n)]
        row = dict(
            organism=g.Organism.iloc[0],
            antibiotic=g.Antimicrobial.iloc[0],
            org_id=int(org),
            abx_id=int(abx),
            countries=n,
            rho=rho,
            p=p,
            CI_low=lo,
            CI_high=hi,
            AMR_year_min=int(g.year.min()),
            AMR_year_max=int(g.year.max()),
            food_units=int(g.n_units.sum()),
            food_sources=int(g.n_studies.sum()),
            LOO_min=min(loo),
            LOO_max=max(loo),
            primary=(org, abx) in primary,
        )
        rows.append(row)
    res = pd.DataFrame(rows)
    res["q_all_pairs"] = multipletests(res.p.fillna(1), method="fdr_bh")[1]
    res["q_primary3"] = np.nan
    ix = res.primary
    res.loc[ix, "q_primary3"] = multipletests(res.loc[ix, "p"], method="fdr_bh")[1]
    save(res, "clinical_all_phenotype_correlations.csv")
    baskets = [
        (
            "Escherichia coli",
            "3GC + carbapenems",
            ["Cephalosporins (3rd gen)", "Carbapenems"],
        ),
        (
            "Klebsiella pneumoniae",
            "3GC + carbapenems",
            ["Cephalosporins (3rd gen)", "Carbapenems"],
        ),
        (
            "Escherichia coli",
            "3GC + fluoroquinolones",
            ["Cephalosporins (3rd gen)", "Fluoroquinolones"],
        ),
        (
            "Klebsiella pneumoniae",
            "3GC + fluoroquinolones",
            ["Cephalosporins (3rd gen)", "Fluoroquinolones"],
        ),
    ]
    rows = []
    countryrows = []
    rng = np.random.default_rng(20261117)
    audit = []
    for species, basket, drugs in baskets:
        z = raw[(raw.Organism == species) & raw.Antimicrobial.isin(drugs)]
        assert not z.duplicated(["ISO3", "Antimicrobial"]).any()
        v = z.pivot(index="ISO3", columns="Antimicrobial", values="value").reindex(
            columns=drugs
        )
        yr = z.pivot(index="ISO3", columns="Antimicrobial", values="year").reindex(
            columns=drugs
        )
        c = v.copy()
        c["n_drugs"] = v.notna().sum(axis=1)
        c["year_min"] = yr.min(axis=1)
        c["year_max"] = yr.max(axis=1)
        c["food_pct"] = food.raw_analyte_test_rate.reindex(c.index) * 100
        c["GM_pct"] = np.prod(v.to_numpy(), axis=1) ** (1 / len(drugs))
        c["zero_components"] = v.eq(0).sum(axis=1)
        ok = c.n_drugs.eq(2) & c.year_min.eq(c.year_max) & c.food_pct.notna()
        c["eligible"] = ok
        c["species"] = species
        c["basket"] = basket
        countryrows.append(c.reset_index())
        g = c[ok]
        x = g.food_pct.to_numpy()
        y = g.GM_pct.to_numpy()
        r, p = spearmanr(x, y)
        n = len(g)
        ii = rng.integers(0, n, (2000, n))
        a = rankdata(x[ii], axis=1)
        b = rankdata(y[ii], axis=1)
        a -= a.mean(axis=1, keepdims=True)
        b -= b.mean(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (a * b).sum(axis=1) / np.sqrt(
                (a * a).sum(axis=1) * (b * b).sum(axis=1)
            )
        lo, hi = np.nanquantile(boot, [0.025, 0.975])
        loo = [spearmanr(np.delete(x, i), np.delete(y, i)).statistic for i in range(n)]
        rows.append(
            dict(
                species=species,
                basket=basket,
                n=n,
                rho=r,
                p=p,
                CI_low=lo,
                CI_high=hi,
                zero_countries=int(g.zero_components.gt(0).sum()),
                year_min=int(g.year_min.min()),
                year_max=int(g.year_max.max()),
                LOO_min=min(loo),
                LOO_max=max(loo),
            )
        )
        audit.append(
            dict(
                species=species,
                basket=basket,
                all_available=len(c),
                complete_components=int(c.n_drugs.eq(2).sum()),
                complete_same_year=int(
                    (c.n_drugs.eq(2) & c.year_min.eq(c.year_max)).sum()
                ),
                food_matched=n,
            )
        )
    res = pd.DataFrame(rows)
    res["q_four_composites"] = multipletests(res.p, method="fdr_bh")[1]
    save(res, "clinical_species_composite_correlations.csv")
    save(
        pd.concat(countryrows, ignore_index=True),
        "clinical_species_composite_countries.csv",
    )
    save(pd.DataFrame(audit), "clinical_species_match_flow.csv")
