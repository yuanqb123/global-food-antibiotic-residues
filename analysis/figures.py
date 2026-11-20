"""Main figures from calculated statistics and display coordinates."""

import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from .common import read, OUTPUTS, INPUTS

F = OUTPUTS / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 15,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def world_polygons():
    geo = json.loads((INPUTS / "world_countries.geojson").read_text(encoding="utf-8"))
    world = []
    for f in geo["features"]:
        g = f["geometry"]
        items = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        world.extend([np.array(p[0]) for p in items if p and len(p[0]) >= 3])
    return world


def figure2():
    pool = []
    for dim in ["food_group", "antibiotic_class_broad", "antibiotic_std"]:
        z = pd.read_csv(OUTPUTS / ("pooled_" + dim + ".csv")).rename(
            columns={dim: "group", "detection_rate": "rate"}
        )
        z["dimension"] = dim
        pool.append(z)
    pool = pd.concat(pool, ignore_index=True)
    oc = read("occurrence_units.csv")
    food = {
        "meat": "Meat",
        "milk/dairy": "Milk and dairy",
        "egg": "Eggs",
        "aquatic products": "Aquatic foods",
        "honey": "Honey",
        "other animal-derived foods": "Other animal foods",
    }
    classes = [
        "beta-lactams",
        "quinolones",
        "tetracyclines",
        "sulfonamides",
        "amphenicols",
        "macrolides",
        "aminoglycosides",
    ]
    cl = {
        "beta-lactams": "Beta-lactams",
        "quinolones": "Quinolones",
        "tetracyclines": "Tetracyclines",
        "sulfonamides": "Sulfonamides",
        "amphenicols": "Amphenicols",
        "macrolides": "Macrolides",
        "aminoglycosides": "Aminoglycosides",
        "ionophores": "Ionophores",
        "diaminopyrimidines": "Diaminopyrimidines",
        "lincosamides": "Lincosamides",
        "nitrofurans": "Nitrofurans",
    }
    # Draw all panels on a common canvas.
    fig, axs = plt.subplots(
        2,
        2,
        figsize=(12.5, 12),
        gridspec_kw={"height_ratios": [1, 1.3]},
        layout="constrained",
    )
    for ax, dim, title, color, mapper in [
        (axs[0, 1], "antibiotic_class_broad", "B  Antibiotic classes", "#386785", cl),
        (
            axs[1, 0],
            "antibiotic_std",
            "C  Top 20 individual antibiotics",
            "#c77b53",
            {},
        ),
        (axs[1, 1], "food_group", "D  Food categories", "#368c89", food),
    ]:
        z = pool[pool.dimension.eq(dim)].copy()
        if dim == "antibiotic_class_broad":
            z = z[z.group.isin(list(cl))]
        if dim == "antibiotic_std":
            z = z.nlargest(20, "n")
        z = z.sort_values("rate")
        y = np.arange(len(z))
        v = z.rate * 100
        ax.barh(y, v, color=color, alpha=0.9, height=0.65)
        ax.errorbar(
            v,
            y,
            xerr=[v - z.CI_low * 100, z.CI_high * 100 - v],
            fmt="none",
            ecolor="#263b45",
            capsize=2,
            lw=0.8,
        )
        ax.set(
            yticks=y,
            yticklabels=[mapper.get(k, k.capitalize()) for k in z.group],
            xlabel="Pooled detection frequency (%)",
            xlim=(0, max(v) * 1.3 + 1),
        )
        ax.tick_params(labelsize=15)
        ax.set_title(title, loc="left", weight="bold", fontsize=16)
        ax.grid(axis="x", alpha=0.12)
        ax.set_axisbelow(True)
        for yy, row in zip(y, z.itertuples()):
            ax.text(
                row.CI_high * 100 + max(v) * 0.03,
                yy,
                f"{row.rate*100:.1f}",
                va="center",
                fontsize=15,
            )
    a = axs[0, 0]
    foods = list(food)
    g = oc.groupby(["food_group", "antibiotic_class_broad"]).agg(
        k=("positive_count_n", "sum"), n=("sample_size_n", "sum")
    )
    mat = np.array(
        [
            [
                (
                    g.loc[(f, c), "k"] / g.loc[(f, c), "n"] * 100
                    if (f, c) in g.index
                    else np.nan
                )
                for c in classes
            ]
            for f in foods
        ]
    )
    im = a.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=45)
    a.set(
        yticks=range(6),
        yticklabels=[food[f] for f in foods],
        xticks=range(7),
        xticklabels=[cl[c] for c in classes],
    )
    a.tick_params(axis="x", rotation=45, labelsize=14)
    a.tick_params(axis="y", labelsize=15)
    a.set_title("A  Food and antibiotic class", loc="left", weight="bold", fontsize=16)
    for i, j in np.ndindex(mat.shape):
        a.text(
            j,
            i,
            f"{mat[i,j]:.1f}",
            ha="center",
            va="center",
            fontsize=15,
            color="white" if mat[i, j] > 30 else "#222",
        )
    fig.colorbar(im, ax=a, shrink=0.7, label="Detection frequency (%)")
    fig.savefig(F / "Figure2.png", dpi=240, facecolor="white")
    fig.savefig(F / "Figure2.pdf", facecolor="white")
    plt.close(fig)


def figure3():
    fig = plt.figure(figsize=(18, 24))
    world = world_polygons()
    cm = ListedColormap(
        ["#e6c43c", "#c4d77d", "#a6d8a7", "#64c4b3", "#32aeb9", "#278db3", "#245f9c"]
    )
    alphas = np.array([0.64, 0.67, 0.71, 0.77, 0.83, 0.89, 0.94])
    checks = []
    for j, (field, title, labs, n) in enumerate(
        [
            (
                "rate",
                "A  Global detection frequency",
                ["<1%", "1–<5%", "5–<10%", "10–<20%", "20–<35%", "35–<50%", "≥50%"],
                7321,
            ),
            (
                "concentration_ugkg",
                "B  Global positive-sample mean concentration",
                ["<1", "1–<10", "10–<50", "50–<100", "100–<500", "500–<1000", "≥1000"],
                1206,
            ),
        ]
    ):
        ax = fig.add_axes([0.035, 0.76 - 0.25 * j, 0.835, 0.20])
        z = read(
            "detection_map_display.csv"
            if field == "rate"
            else "concentration_map_display.csv"
        )
        assert len(z) == n
        levels = z.display_bin.to_numpy()
        xy = z[
            ["display_longitude_beautified", "display_latitude_beautified"]
        ].to_numpy()
        rgba = cm(levels)
        rgba[:, 3] = alphas[levels]
        order = np.argsort(levels, kind="stable")
        ax.add_collection(
            PolyCollection(
                world, facecolors="#f0f3f4", edgecolors="#d0d8dc", linewidths=0.35
            )
        )
        pts = ax.scatter(
            xy[order, 0],
            xy[order, 1],
            c=rgba[order],
            s=44,
            edgecolors="white",
            linewidths=0.10,
            rasterized=True,
        )
        ax.set(xlim=(-180, 180), ylim=(-58, 83))
        ax.set_axis_off()
        ax.set_title(title, loc="left", fontsize=20, weight="bold", pad=12)
        assert np.array_equal(np.asarray(pts.get_offsets()), xy[order])
        assert len(pts.get_offsets()) == n
        cbax = fig.add_axes([0.88, 0.78 - 0.25 * j, 0.013, 0.16])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cm, norm=BoundaryNorm(np.arange(8), 7)),
            cax=cbax,
            ticks=np.arange(7) + 0.5,
        )
        cb.ax.set_yticklabels(labs, fontsize=16)
        cb.set_label(
            "Detection frequency" if j == 0 else "Concentration (µg/kg)", fontsize=18
        )
        checks.append(
            dict(panel=title[:1], points=n, display_coordinates_and_bins_identical=True)
        )
    allc = read("concentration_statistics.csv")
    conc = allc[
        allc.quantitative_eligible
        & allc.statistic_type.eq("positive_mean")
        & allc.concentration_ugkg.gt(0)
    ].copy()
    assert len(conc) == 1211
    cross = read("WHO_regions.csv")[["country_iso3", "display_region"]]
    cc = conc.merge(cross, on="country_iso3", how="left")
    reg = (
        pd.read_csv(OUTPUTS / "pooled_display_region.csv")
        .rename(columns={"detection_rate": "occurrence_rate"})
        .sort_values("occurrence_rate")
    )
    adj = (
        pd.read_csv(OUTPUTS / "country_standardised_detection.csv")
        .dropna(subset=["adjusted_occurrence_exploratory"])
        .nlargest(10, "adjusted_occurrence_exploratory")
        .iloc[::-1]
    )
    co = cc.groupby("country_iso3").concentration_ugkg.median().nlargest(10).iloc[::-1]
    REG = {
        "AFR": "African Region",
        "AMR": "Region of the Americas",
        "EMR": "Eastern Mediterranean Region",
        "EUR": "European Region",
        "SEAR": "South-East Asia Region",
        "WPR": "Western Pacific Region",
    }
    country = {
        "SEN": "Senegal",
        "NOR": "Norway",
        "ROU": "Romania",
        "LKA": "Sri Lanka",
        "CZE": "Czechia",
        "POL": "Poland",
        "MYS": "Malaysia",
        "PHL": "Philippines",
        "ARG": "Argentina",
        "BEN": "Benin",
        "THA": "Thailand",
        "HUN": "Hungary",
        "MEX": "Mexico",
        "SAU": "Saudi Arabia",
        "EGY": "Egypt",
        "MOZ": "Mozambique",
        "KWT": "Kuwait",
        "ETH": "Ethiopia",
        "IRN": "Iran",
        "CMR": "Cameroon",
    }
    axs = [
        fig.add_axes(p)
        for p in [
            [0.215, 0.285, 0.29, 0.16],
            [0.63, 0.285, 0.34, 0.16],
            [0.215, 0.055, 0.29, 0.16],
            [0.63, 0.055, 0.34, 0.16],
        ]
    ]
    axs[0].barh(reg.display_region.map(REG), reg.occurrence_rate * 100, color="#148a83")
    axs[0].set_xlabel("Pooled analyte-test detection frequency (%)", fontsize=18)
    x = adj.adjusted_occurrence_exploratory.to_numpy() * 100
    yy = np.arange(len(adj))
    axs[1].errorbar(
        x,
        yy,
        xerr=[
            np.maximum(0, x - adj.CI_95_low_exploratory * 100),
            np.maximum(0, adj.CI_95_high_exploratory * 100 - x),
        ],
        fmt="o",
        color="#264c68",
        capsize=2,
    )
    axs[1].set(yticks=yy, yticklabels=[country[k] for k in adj.country_iso3])
    axs[1].set_xlabel("Exploratory adjusted detection frequency (%)", fontsize=18)

    def box(ax, data, labels, color):
        bb = ax.boxplot(
            data,
            orientation="horizontal",
            patch_artist=True,
            tick_labels=labels,
            showfliers=False,
            widths=0.55,
            medianprops={"color": "#1b2730", "linewidth": 1.5},
            whiskerprops={"color": color},
            capprops={"color": color},
        )
        for b in bb["boxes"]:
            b.set(facecolor=color, alpha=0.3, edgecolor=color)
        ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.18)
        ax.set_xlabel("Positive-sample mean concentration (µg/kg)", fontsize=18)

    box(
        axs[2],
        [
            cc.loc[cc.display_region.eq(k), "concentration_ugkg"]
            for k in reg.display_region
        ],
        [REG[k] for k in reg.display_region],
        "#148a83",
    )
    box(
        axs[3],
        [cc.loc[cc.country_iso3.eq(k), "concentration_ugkg"] for k in co.index],
        [country[k] for k in co.index],
        "#d77943",
    )
    for ax, title in zip(
        axs,
        [
            "C  Detection by WHO region",
            "D  Countries with higher\ndetection frequencies",
            "E  Concentration by WHO region",
            "F  Countries with higher\nconcentrations",
        ],
    ):
        ax.set_title(title, loc="left", fontsize=20, weight="bold", pad=12)
        ax.tick_params(labelsize=16)
    checks.append(
        dict(
            panels="C–F",
            data="Original WHO6 summaries, frozen adjusted estimates and 1211 positive means",
            all_panel_title_fontsize=20,
            region_order=reg.display_region.tolist(),
            detection_countries=adj.country_iso3.tolist(),
            concentration_countries=co.index.tolist(),
        )
    )
    fig.savefig(F / "Figure3.png", dpi=300, facecolor="white")
    fig.savefig(F / "Figure3.pdf", facecolor="white")
    plt.close(fig)


def figure6():
    who = read("country_foodborne_burden.csv")
    amr = (
        read("country_AMR_mortality_categories.csv")
        .dropna(subset=["raw_analyte_test_rate", "ordinal_bin"])
        .drop_duplicates("country_iso3")
    )
    curves = pd.read_csv(OUTPUTS / "ecological_fitted_curves.csv")
    data = read("clinical_resistance_matches.csv")
    stats = pd.read_csv(OUTPUTS / "clinical_resistance_correlations.csv")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Arial",
            "mathtext.it": "Arial:italic",
            "mathtext.bfit": "Arial:bold:italic",
        }
    )
    fig, axs = plt.subplots(2, 3, figsize=(20, 13.0), layout="constrained")
    records = []
    for ax, field, df, xf, col, title in [
        (
            axs[0, 0],
            "foodborne_death_rate_100k",
            who,
            "residue_all_years_rate",
            "#3B75AF",
            "A  Detection and\nfoodborne mortality rate",
        ),
        (
            axs[0, 1],
            "foodborne_DALY_rate_100k",
            who,
            "residue_all_years_rate",
            "#3B75AF",
            "B  Detection and\nfoodborne DALY rate",
        ),
        (
            axs[0, 2],
            "AMR_ordinal",
            amr,
            "raw_analyte_test_rate",
            "#3B75AF",
            "C  Detection and AMR\nmortality category",
        ),
    ]:
        yf = "ordinal_bin" if field == "AMR_ordinal" else field
        g = df.dropna(subset=[xf, yf])
        line = curves[curves.endpoint.eq(field)]
        ax.fill_between(
            line.residue_rate_pct, line.CI_low, line.CI_high, color=col, alpha=0.18
        )
        ax.plot(line.residue_rate_pct, line.fitted, color=col, lw=2)
        ax.scatter(
            g[xf] * 100, g[yf], s=48, color=col, alpha=0.7, edgecolors="white", zorder=3
        )
        rho, p = spearmanr(g[xf], g[yf])
        ax.text(
            0.97,
            0.97,
            f"n = {len(g)}; ρ = {rho:.3f}\np = {p:.3g}",
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=17,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
        )
        if field == "AMR_ordinal":
            t = (
                g[["ordinal_bin", "legend_interval_per_100k"]]
                .drop_duplicates()
                .sort_values("ordinal_bin")
            )
            ax.set(
                yticks=t.ordinal_bin,
                yticklabels=t.legend_interval_per_100k,
                ylabel="Mortality map category\n(per 100,000)",
            )
        else:
            ax.set(yscale="log", ylabel="Rate per 100,000 (log scale)")
        ax.set_title(title, loc="left", fontsize=20, weight="bold", pad=15)
    for i, (ax, (_, row)) in enumerate(zip(axs[1], stats.iterrows())):
        g = data[data.species.eq(row.species)]
        x = g.food_detection_pct.to_numpy()
        y = g.resistance_pct.to_numpy()
        n = len(x)
        grid = np.linspace(x.min(), x.max(), 201)
        fit = np.polyfit(x, y, 1)
        pred = fit[0] * grid + fit[1]
        rng = np.random.default_rng(20261117 + i)
        ii = rng.integers(0, n, (2000, n))
        xx = x[ii]
        yy = y[ii]
        mx = xx.mean(axis=1)
        my = yy.mean(axis=1)
        den = ((xx - mx[:, None]) ** 2).sum(axis=1)
        slopes = ((xx - mx[:, None]) * (yy - my[:, None])).sum(axis=1) / den
        intercepts = my - slopes * mx
        bs = intercepts[:, None] + slopes[:, None] * grid
        lo, hi = np.quantile(bs, [0.025, 0.975], axis=0)
        col = "#168C86"
        ax.fill_between(grid, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(grid, pred, color=col, lw=2)
        ax.scatter(x, y, s=48, color=col, alpha=0.7, edgecolors="white", zorder=3)
        ax.text(
            0.97,
            0.97,
            f"n = {n}; ρ = {row.rho:.3f}\np = {row.p:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=17,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=2),
        )
        name = [
            r"$\mathbfit{E.\ coli}$",
            r"$\mathbfit{K.\ pneumoniae}$",
            r"$\mathbfit{S.\ aureus}$",
        ][i]
        ax.set_title(
            chr(68 + i) + "  Detection and\n" + name + " resistance proportion",
            loc="left",
            fontsize=20,
            weight="bold",
            pad=15,
        )
        ax.set_ylabel("Clinical non-susceptibility (%)")
        lower = min(-2, float(lo.min() - 2))
        upper = max(102, float(hi.max() + 2))
        ax.set_ylim(lower, upper)
        for a, b, c, d in zip(grid, pred, lo, hi):
            records.append(
                dict(
                    species=row.species,
                    food_detection_pct=a,
                    fitted_mean_pct=b,
                    CI_low=c,
                    CI_high=d,
                    slope=fit[0],
                    intercept=fit[1],
                )
            )
    for ax in axs.flat:
        ax.set_xlabel("Pooled detection frequency (%)", fontsize=17)
        ax.yaxis.label.set_size(17)
        ax.tick_params(labelsize=16)
    fig.savefig(F / "Figure6.png", dpi=300, facecolor="white")
    fig.savefig(F / "Figure6.pdf", facecolor="white")
    plt.close(fig)
    pd.DataFrame(records).to_csv(OUTPUTS / "clinical_fitted_curves.csv", index=False)


def figure7():
    con = pd.concat(
        [
            pd.read_csv(OUTPUTS / f"{ep}_indicator_contributions.csv")
            for ep in ["RQ_res", "HQ_MRL"]
        ],
        ignore_index=True,
    )
    profiles = pd.read_csv(OUTPUTS / "response_profiles.csv")
    names = {
        "reported_vet_AMU_kg_food_scope": "Veterinary AMU: reported total",
        "modelled_2020_vet_AMU_mg_PCU": "Veterinary AMU: modelled 2020",
        "EMA_vet_AMU_mg_PCU": "Veterinary AMU: EMA intensity",
        "production_intensity_proxy": "Food production / 1,000 people",
        "aquaculture_production_tonnes": "Aquaculture production",
        "aquaculture_share_pct": "Aquaculture share",
        "TrACSS_vet_regulatory_stage_score": "Veterinary regulation (TrACSS)",
        "GDP_per_capita_current_USD": "GDP per capita",
        "rural_population_pct": "Rural population share",
        "agriculture_value_added_pct_GDP": "Agricultural value added",
        "access_to_electricity_pct": "Electricity access",
        "safely_managed_drinking_water_pct": "Safely managed drinking water",
        "safely_managed_sanitation_pct": "Safely managed sanitation",
        "human_total_DDD_per_1000_day": "Human antimicrobial use",
        "freshwater_withdrawal_stress_pct": "Freshwater withdrawal stress",
        "WGI_Government_Effectiveness": "Government Effectiveness",
        "WGI_Control_of_Corruption": "Control of Corruption",
    }
    themes = {
        "Antimicrobial use": [
            "reported_vet_AMU_kg_food_scope",
            "modelled_2020_vet_AMU_mg_PCU",
            "EMA_vet_AMU_mg_PCU",
            "human_total_DDD_per_1000_day",
        ],
        "Food and aquaculture\nproduction": [
            "production_intensity_proxy",
            "aquaculture_production_tonnes",
            "aquaculture_share_pct",
        ],
        "Regulation and\ngovernance": [
            "TrACSS_vet_regulatory_stage_score",
            "WGI_Government_Effectiveness",
            "WGI_Control_of_Corruption",
        ],
        "Socioeconomic\nconditions": [
            "GDP_per_capita_current_USD",
            "rural_population_pct",
            "agriculture_value_added_pct_GDP",
            "access_to_electricity_pct",
        ],
        "Water, sanitation and\nenvironmental pressure": [
            "safely_managed_drinking_water_pct",
            "safely_managed_sanitation_pct",
            "freshwater_withdrawal_stress_pct",
        ],
    }
    mean = con.groupby("variable")["share"].mean()
    themes = {
        k: sorted(v, key=lambda x: mean[x], reverse=True) for k, v in themes.items()
    }
    order = [v for vs in themes.values() for v in vs]
    assert len(set(order)) == 17
    ys = {}
    groups = []
    y = 0
    for k, vs in themes.items():
        start = y
        for v in vs:
            ys[v] = y
            y += 1
        groups.append((k, start, y - 1))
        y += 0.65
    lim = y - 0.65 - 0.3
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 13,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def draw(fig, pos, titley):
        ax = fig.add_axes(pos)
        ax.set_ylim(lim, -0.7)
        ax.set_xlim(0, 25)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([ys[v] for v in order])
        ax.set_yticklabels([names[v] for v in order], fontsize=12.5)
        ax.tick_params(axis="y", length=0, pad=11)
        ax.set_xticks([0, 5, 10, 15, 20, 25])
        ax.set_xlabel(
            "Relative contribution among national indicators (%)",
            fontsize=13,
            labelpad=9,
        )
        ax.grid(axis="x", alpha=0.12)
        ax.set_axisbelow(True)
        for j, (k, lo, hi) in enumerate(groups):
            if j % 2 == 0:
                ax.axhspan(lo - 0.5, hi + 0.5, color="#f3f5f7", lw=0, zorder=0)
            fy = fig.transFigure.inverted().transform(
                ax.transData.transform((0, (lo + hi) / 2))
            )[1]
            fig.text(
                0.025,
                fy,
                k,
                fontsize=13,
                weight="bold",
                ha="left",
                va="center",
                linespacing=1.4,
            )
            f0 = fig.transFigure.inverted().transform(
                ax.transData.transform((0, lo - 0.4))
            )[1]
            f1 = fig.transFigure.inverted().transform(
                ax.transData.transform((0, hi + 0.4))
            )[1]
            fig.add_artist(
                plt.Line2D(
                    [0.228, 0.228],
                    [f0, f1],
                    transform=fig.transFigure,
                    color="#b9c2c9",
                    lw=1.5,
                )
            )
        for ep, off, c, label in [
            ("RQ_res", -0.17, "#3978A8", "RQ_res"),
            ("HQ_MRL", 0.17, "#C77C3D", "HQ/MRL"),
        ]:
            z = con[con.endpoint.eq(ep)].set_index("variable").loc[order]
            vals = z.share.to_numpy() * 100
            lo = z.share_CI_low.to_numpy() * 100
            hi = z.share_CI_high.to_numpy() * 100
            yy = np.array([ys[v] for v in order]) + off
            ax.barh(yy, vals, height=0.29, color=c, alpha=0.86, label=label, zorder=2)
            ax.errorbar(
                vals,
                yy,
                xerr=np.vstack([vals - lo, hi - vals]),
                fmt="none",
                ecolor=c,
                lw=1.1,
                capsize=2,
                zorder=3,
            )
        fig.text(
            0.025,
            titley,
            "A  National-indicator contributions by factor group",
            fontsize=17,
            weight="bold",
        )
        return ax

    fig = plt.figure(figsize=(13.5, 17))
    ax = draw(fig, [0.495, 0.635, 0.48, 0.315], 0.985)
    variables = [
        "safely_managed_sanitation_pct",
        "EMA_vet_AMU_mg_PCU",
        "WGI_Control_of_Corruption",
        "human_total_DDD_per_1000_day",
    ]
    titles = [
        ("B  Sanitary infrastructure", "Safely managed sanitation (%)"),
        ("C  EMA veterinary antimicrobial use", "EMA veterinary AMU (mg/PCU)"),
        ("D  Governance context", "Control of Corruption (WGI score)"),
        ("E  Human antimicrobial use", "DDD / 1,000 inhabitants / day"),
    ]
    axes = []
    for pos, v, (title, xlabel) in zip(
        [
            [0.105, 0.36, 0.315, 0.205],
            [0.605, 0.36, 0.315, 0.205],
            [0.105, 0.075, 0.315, 0.205],
            [0.605, 0.075, 0.315, 0.205],
        ],
        variables,
        titles,
    ):
        ax = fig.add_axes(pos)
        axes.append(ax)
        ax2 = ax.twinx()
        for ep, aa, c, ls, mult in [
            ("RQ_res", ax, "#3978A8", "-", 1),
            ("HQ_MRL", ax2, "#C77C3D", "--", 100),
        ]:
            z = profiles[(profiles.endpoint == ep) & (profiles.variable == v)]
            assert len(z) == (39 if v == "WGI_Control_of_Corruption" else 41)
            aa.fill_between(
                z.x, z.CI_low * mult, z.CI_high * mult, color=c, alpha=0.13, lw=0
            )
            aa.plot(z.x, z.prediction * mult, color=c, ls=ls, lw=2.5)
            lo = max(0, z.CI_low.min() * mult)
            hi = z.CI_high.max() * mult
            pad = max((hi - lo) * 0.12, 0.001)
            aa.set_ylim(max(0, lo - pad), hi + pad)
            aa.tick_params(axis="y", labelcolor=c, labelsize=12.5)
            aa.set_ylabel(
                (
                    "Predicted RQ_res (unitless)"
                    if ep == "RQ_res"
                    else "HQ/MRL screening probability (%)"
                ),
                color=c,
                fontsize=13,
                labelpad=4,
            )
        ax.set_title(title, loc="left", weight="bold", fontsize=14, pad=13)
        ax.set_xlabel(xlabel, fontsize=13, labelpad=7)
        ax.grid(axis="y", alpha=0.10)
        ax.tick_params(axis="x", labelsize=12.5)
        if v == "EMA_vet_AMU_mg_PCU":
            ax.text(
                0.97,
                0.96,
                "5 European countries\nn = 22 / 23",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11.5,
                color="#555555",
            )
    fig.legend(
        handles=[
            Line2D([], [], color="#3978A8", lw=2.6, label="RQ_res"),
            Line2D([], [], color="#C77C3D", ls="--", lw=2.6, label="HQ/MRL"),
        ],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.25, 0.618),
        fontsize=14,
    )
    fig.savefig(F / "Figure7.png", dpi=300, facecolor="white")
    fig.savefig(F / "Figure7.pdf", facecolor="white")
    plt.close(fig)


FOODS = {
    "meat": "Meat",
    "milk/dairy": "Milk and dairy",
    "egg": "Eggs",
    "aquatic products": "Aquatic foods",
    "honey": "Honey",
    "other animal-derived foods": "Other animal foods",
}


def figure4():
    z = pd.read_csv(OUTPUTS / "five_year_food_summaries.csv")
    ci = pd.read_csv(OUTPUTS / "five_year_detection_intervals.csv")
    tr = pd.read_csv(OUTPUTS / "food_antibiotic_trends.csv")
    pos = tr[tr.positive_estimable]
    drugs = sorted(pos.antibiotic_std.unique())
    foods = list(FOODS)
    fig = plt.figure(figsize=(15, 11), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    axs = [fig.add_subplot(gs[0, i]) for i in range(2)]
    ax = fig.add_subplot(gs[1, :])
    colors = ["#2379AD", "#E69F00", "#009E73", "#D55E00", "#A35AA4", "#657A85"]
    markers = ["o", "s", "^", "D", "v", "P"]
    for a, ep, letter in zip(axs, ["detection", "concentration"], ["A", "B"]):
        scale = 100 if ep == "detection" else 1
        overall = z[z.endpoint.eq(ep) & z.food_group.eq("all_foods")].sort_values(
            "period_start"
        )
        for food, color, marker in zip(foods, colors, markers):
            h = z[z.endpoint.eq(ep) & z.food_group.eq(food)].sort_values("period_start")
            a.plot(
                h.period_start + 2,
                h.estimate * scale,
                color=color,
                marker=marker,
                ms=4,
                lw=1.15,
                label=FOODS[food],
            )
        a.plot(
            overall.period_start + 2,
            overall.estimate * scale,
            "o-",
            color="#183F58",
            lw=2.6,
            label="Overall",
        )
        if ep == "detection":
            a.fill_between(
                ci.period_start + 2,
                ci.CI_low * 100,
                ci.CI_high * 100,
                color="#2a729f",
                alpha=0.16,
                lw=0,
            )
            a.set_ylabel("Detection frequency (%)")
            a.set_ylim(0, 72)
        else:
            a.fill_between(
                overall.period_start + 2,
                overall.q25,
                overall.q75,
                color="#2a729f",
                alpha=0.16,
                lw=0,
            )
            a.set(yscale="log", ylabel="Concentration (µg/kg; log scale)")
        a.set_xticks(
            overall.period_start + 2,
            [
                f"{int(p)}–\n{int(p+4)}" + ("*" if p == 2025 else "")
                for p in overall.period_start
            ],
            fontsize=10,
        )
        a.set_xlabel("Sampling period (*incomplete period)")
        a.grid(axis="y", alpha=0.15)
        a.set_title(
            letter
            + "  "
            + (
                "Five-year detection frequency"
                if ep == "detection"
                else "Positive-mean concentration"
            ),
            loc="left",
            weight="bold",
        )
    mat = np.full((6, len(drugs)), np.nan)
    for row in pos.itertuples():
        mat[foods.index(row.food_group), drugs.index(row.antibiotic_std)] = (
            0
            if row.status.startswith("point_only")
            else 2 if row.positive_FDR_significant else 1
        )
    ax.imshow(
        mat,
        cmap=ListedColormap(["#c5ccce", "#f2c8a9", "#b8413a"]),
        vmin=-0.5,
        vmax=2.5,
        aspect="auto",
    )
    ax.set(
        yticks=range(6),
        yticklabels=list(FOODS.values()),
        xticks=range(len(drugs)),
        xticklabels=drugs,
    )
    ax.tick_params(axis="x", rotation=50, labelsize=11)
    for r in pos.itertuples():
        if r.positive_FDR_significant:
            ax.text(
                drugs.index(r.antibiotic_std),
                foods.index(r.food_group),
                "*",
                ha="center",
                va="center",
                fontsize=18,
                color="white",
            )
    ax.set_title(
        "C  Food–antibiotic combinations with positive estimated trends",
        loc="left",
        weight="bold",
    )
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(fc="#c5ccce", label="Positive point estimate only"),
            Patch(fc="#f2c8a9", label="Positive estimate; q ≥ 0.05"),
            Patch(fc="#b8413a", label="Positive estimate; q < 0.05"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=11
    )
    fig.savefig(F / "Figure4.png", dpi=300, facecolor="white")
    fig.savefig(F / "Figure4.pdf", facecolor="white")
    plt.close(fig)


def figure5():
    h = pd.read_csv(OUTPUTS / "dietary_HQ_records.csv")
    m = pd.read_csv(OUTPUTS / "regulatory_MRL_records.csv")
    q = pd.read_csv(OUTPUTS / "resistance_selection_F100.csv")
    world = world_polygons()
    cm = ListedColormap(["#176fa0", "#dfb635", "#bd3537"])
    marks = ["o", "^", "s", "D", "P", "X"]
    fig = plt.figure(figsize=(14, 19.7))
    audit = []
    specs = [
        (
            "A",
            h,
            "HQ",
            "Dietary hazard quotient",
            [0.1, 1],
            ["≤0.1", ">0.1–1", ">1"],
            "HQ",
        ),
        (
            "B",
            m,
            "C_MRL",
            "Regulatory concentration screen",
            [1, 10],
            ["≤1", ">1–10", ">10"],
            r"HQ$_{MRL}$",
        ),
        (
            "C",
            q,
            "RQ",
            "Resistance-selection quotient",
            [0.1, 1],
            ["≤0.1", ">0.1–1", ">1"],
            r"RQ$_{res}$",
        ),
    ]
    for row, (panel, g, value, title, cut, labs, label) in enumerate(specs):
        display = read(f"risk_map_{panel}_display.csv")
        z = g.merge(display, on="statistic_id", how="inner", validate="one_to_one")
        y = 0.745 - row * 0.235
        ax = fig.add_axes([0.015, y, 0.86, 0.20])
        ax.add_collection(
            PolyCollection(world, facecolors="#f1f3f2", edgecolors="#aebabc", lw=0.18)
        )
        ax.set(xlim=(-170, 190), ylim=(-57, 78))
        ax.set_axis_off()
        ax.set_title(
            panel + "  " + title, loc="left", fontsize=20, weight="bold", pad=8
        )
        norm = BoundaryNorm(
            [0, np.nextafter(cut[0], np.inf), np.nextafter(cut[1], np.inf), np.inf], 3
        )
        drawn = 0
        for level in range(3):
            for food, mk in zip(FOODS, marks):
                zz = z[z.food_group.eq(food) & (norm(z[value]) == level)]
                ax.scatter(
                    zz.display_lon,
                    zz.display_lat,
                    c=zz[value],
                    cmap=cm,
                    norm=norm,
                    marker=mk,
                    s=70,
                    alpha=0.75,
                    edgecolors="white",
                    lw=0.25,
                    zorder=3 + level,
                )
                drawn += len(zz)
        assert drawn == len(z)
        cbax = fig.add_axes([0.89, y + 0.035, 0.014, 0.11])
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cm, norm=BoundaryNorm([0, 1, 2, 3], 3)),
            cax=cbax,
            ticks=[0.5, 1.5, 2.5],
        )
        cb.ax.set_yticklabels(labs, fontsize=17)
        cb.ax.set_title(label, fontsize=18, pad=10)
        audit.append(
            dict(
                panel=panel,
                eligible=len(g),
                plotted=len(z),
                unique_statistics=z.statistic_id.nunique(),
            )
        )
    fig.legend(
        handles=[
            Line2D([], [], marker=mk, ls="", mfc="#71899e", mec="white", ms=9, label=n)
            for mk, n in zip(marks, FOODS.values())
        ],
        loc="upper center",
        bbox_to_anchor=(0.47, 0.283),
        ncol=3,
        frameon=False,
        fontsize=14,
    )
    pair = pd.read_csv(OUTPUTS / "paired_risk_classifications.csv")
    H = pair.HQ_current
    Q = pair.RQ_res
    ax = fig.add_axes([0.08, 0.045, 0.39, 0.18])
    ax.scatter(H, Q, s=20, alpha=0.6, c="#245d88", edgecolors="white", lw=0.3)
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Dietary quotient (HQ)",
        ylabel=r"Resistance-selection quotient (RQ$_{res}$)",
    )
    ax.set_title(
        f"D  Matched statistics (n = {len(pair)})",
        loc="left",
        fontsize=17,
        weight="bold",
    )
    ax.axvline(1, c="#9c422c", lw=1)
    ax.axvline(0.1, c="#9c422c", ls=":", lw=1)
    ax.axhline(1, c="#9c422c", lw=1)
    ax.grid(alpha=0.12)
    matrix = (
        pd.crosstab(H.gt(1), Q.gt(1))
        .reindex(index=[False, True], columns=[False, True], fill_value=0)
        .to_numpy()
    )
    ax = fig.add_axes([0.65, 0.045, 0.29, 0.18])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=265)
    for (i, j), v in np.ndenumerate(matrix):
        ax.text(
            j,
            i,
            str(v),
            ha="center",
            va="center",
            fontsize=25,
            c="white" if v > 150 else "#123d60",
        )
    ax.set(
        xticks=[0, 1],
        xticklabels=[r"RQ$_{res}$ ≤ 1", r"RQ$_{res}$ > 1"],
        yticks=[0, 1],
        yticklabels=["HQ ≤ 1", "HQ > 1"],
    )
    ax.set_title("E  Paired classifications", loc="left", fontsize=17, weight="bold")
    pd.DataFrame(audit).to_csv(OUTPUTS / "risk_map_plot_audit.csv", index=False)
    fig.savefig(F / "Figure5.png", dpi=300, facecolor="white")
    fig.savefig(F / "Figure5.pdf", facecolor="white")
    plt.close(fig)


def run():
    for fn in [figure2, figure3, figure4, figure5, figure6, figure7]:
        fn()
        print(fn.__name__, "saved", flush=True)
