"""Analysis-cohort and national-indicator descriptive summaries."""

import numpy as np
import pandas as pd
from .common import read, save, OUTPUTS
from .models import load, vv


def coverage():
    dictionary = read("national_indicator_definitions.csv").set_index("variable")
    rows = []
    corr = []
    missing = []
    for ep in ["RQ_res", "HQ_MRL"]:
        g = load(ep)
        for v in vv:
            a = g[v].notna()
            x = g.loc[a, v]
            rows.append(
                dict(
                    endpoint=ep,
                    variable=v,
                    label=dictionary.loc[v, "label"],
                    observed_records=int(a.sum()),
                    countries_with_observed_values=g.loc[a, "country_iso3"].nunique(),
                    total_records=len(g),
                    missing_or_inapplicable_records=int((~a).sum()),
                    minimum=x.min(),
                    median=x.median(),
                    maximum=x.max(),
                )
            )
        national = g.drop_duplicates(["country_iso3", "year_mid"] + vv)
        for i, a in enumerate(vv):
            for b in vv[i + 1 :]:
                z = national[[a, b, "country_iso3"]].dropna()
                corr.append(
                    dict(
                        endpoint=ep,
                        first=a,
                        second=b,
                        country_periods=len(z),
                        countries=z.country_iso3.nunique(),
                        spearman=(
                            z[a].corr(z[b], method="spearman") if len(z) > 2 else np.nan
                        ),
                    )
                )
        path = OUTPUTS / f"{ep}_SHAP_all_records.csv"
        if path.exists():
            h = pd.read_csv(path)
            for v in vv:
                a = g[v].isna().to_numpy()
                s = abs(h[v + "_SHAP"].to_numpy()) * h.weight.to_numpy()
                missing.append(
                    dict(
                        endpoint=ep,
                        variable=v,
                        missing_records=int(a.sum()),
                        missing_fraction_of_abs_group_SHAP=s[a].sum() / s.sum(),
                    )
                )
    save(pd.DataFrame(rows), "indicator_observed_coverage.csv")
    save(pd.DataFrame(corr), "indicator_country_period_correlations.csv")
    if missing:
        save(pd.DataFrame(missing), "missingness_contributions.csv")


def cohort_sizes():
    rows = []
    for name, unit in [
        ("occurrence_units.csv", "valid analyte-test units"),
        ("concentration_statistics.csv", "concentration statistics"),
        ("dated_positive_means.csv", "dated positive-mean statistics"),
        ("HQ_positive_means.csv", "positive-mean HQ statistics"),
        ("MRL_mean_statistics.csv", "matched mean-statistic regulatory values"),
        ("RQ_positive_means.csv", "positive-mean RQ statistics"),
        ("paired_HQ_RQ_means.csv", "exactly paired statistic IDs"),
        ("RQ_res_model_input.csv", "SVR record statistics"),
        ("HQ_MRL_model_input.csv", "SVC record statistics"),
    ]:
        g = read(name)
        source = g.source_id if "source_id" in g else g.DOI.fillna(g.study_id)
        rows.append(
            dict(
                cohort=name,
                unit=unit,
                rows=len(g),
                countries=g.country_iso3.nunique(),
                source_groups=source.nunique(),
                underlying_records=g.canonical_row.nunique(),
                unique_DOIs=g.DOI.dropna().nunique() if "DOI" in g else np.nan,
            )
        )
    save(pd.DataFrame(rows), "cohort_sizes.csv")
    coordinates = []
    for name, field, total in [
        ("detection_map_display.csv", "rate", 7321),
        ("concentration_map_display.csv", "concentration_ugkg", 1211),
    ]:
        g = read(name)
        xy = g[["display_longitude_beautified", "display_latitude_beautified"]]
        coordinates.append(
            dict(
                input=name,
                total_eligible=total,
                mapped=len(g),
                missing_coordinates=total - len(g),
                unique_display_positions=len(xy.drop_duplicates()),
                unique_underlying_records=g.canonical_row.nunique(),
                point_values_missing=int(g[field].isna().sum()),
            )
        )
    save(pd.DataFrame(coordinates), "coordinate_audit.csv")


def run():
    coverage()
    cohort_sizes()
