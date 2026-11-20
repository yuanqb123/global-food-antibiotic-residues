"""Supplementary figures S1–S4."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch
from .common import read, OUTPUTS
from .figures import F, world_polygons, FOODS
from .models import vv


def supplementary():
    world = world_polygons()
    cm = ListedColormap(["#176fa0", "#dfb635", "#bd3537"])
    norm = BoundaryNorm(
        [0, np.nextafter(0.1, np.inf), np.nextafter(1, np.inf), np.inf], 3
    )
    display = read("risk_map_C_display.csv")
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), layout="constrained")
    for ax, Fval, letter in zip(axs, [10, 1000], ["A", "B"]):
        g = pd.read_csv(OUTPUTS / f"resistance_selection_F{Fval}.csv").merge(
            display, on="statistic_id", validate="one_to_one"
        )
        ax.add_collection(PolyCollection(world, fc="#f1f3f2", ec="#aebabc", lw=0.18))
        for food, marker in zip(FOODS, ["o", "^", "s", "D", "P", "X"]):
            z = g[g.food_group.eq(food)].sort_values("RQ")
            ax.scatter(
                z.display_lon,
                z.display_lat,
                c=z.RQ,
                cmap=cm,
                norm=norm,
                s=60,
                marker=marker,
                alpha=0.75,
                ec="white",
                lw=0.25,
            )
        ax.set(xlim=(-170, 190), ylim=(-57, 78))
        ax.axis("off")
        ax.set_title(
            f"{letter}  Resistance-selection scenario F = {Fval:,} (n = {len(g)})",
            loc="left",
            weight="bold",
        )
        cb = fig.colorbar(
            plt.cm.ScalarMappable(cmap=cm, norm=BoundaryNorm([0, 1, 2, 3], 3)),
            ax=ax,
            ticks=[0.5, 1.5, 2.5],
            shrink=0.7,
        )
        cb.ax.set_yticklabels(["≤0.1", ">0.1–1", ">1"])
        cb.set_label(r"RQ$_{res}$")
    fig.savefig(F / "FigureS1.png", dpi=250)
    fig.savefig(F / "FigureS1.pdf")
    plt.close(fig)
    labels = (
        read("national_indicator_definitions.csv").set_index("variable").label.to_dict()
    )
    con = pd.concat(
        [
            pd.read_csv(OUTPUTS / f"{ep}_indicator_contributions.csv")
            for ep in ["RQ_res", "HQ_MRL"]
        ]
    )
    order = con.groupby("variable")["share"].mean().sort_values().index.tolist()
    fig, axs = plt.subplots(1, 2, figsize=(17, 10), layout="constrained")
    for ax, ep, letter in zip(axs, ["RQ_res", "HQ_MRL"], ["A", "B"]):
        g = pd.read_csv(OUTPUTS / f"{ep}_SHAP_all_records.csv")
        data = [g[v + "_SHAP"] for v in order]
        ax.boxplot(
            data,
            orientation="horizontal",
            tick_labels=[labels[v] for v in order],
            showfliers=False,
            widths=0.55,
        )
        ax.axvline(0, c="#777", lw=0.8)
        ax.set_xlabel("Signed grouped SHAP value")
        ax.set_title(letter + "  " + ep, loc="left", weight="bold")
    fig.savefig(F / "FigureS2.png", dpi=250)
    fig.savefig(F / "FigureS2.pdf")
    plt.close(fig)
    v = "human_total_DDD_per_1000_day"
    prof = pd.read_csv(OUTPUTS / "response_profiles.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, ep, color, letter in zip(
        axs, ["RQ_res", "HQ_MRL"], ["#3978A8", "#C77C3D"], ["A", "B"]
    ):
        z = np.load(OUTPUTS / f"{ep}_{v}_profiles.npz")
        grid = z["grid"]
        arr = z["predictions"]
        w = z["weights"]
        fold = z["fold"]
        mult = 1 if ep == "RQ_res" else 100
        for f in sorted(set(fold)):
            take = fold == f
            ax.plot(
                grid,
                np.average(arr[take], axis=0, weights=w[take]) * mult,
                lw=1,
                alpha=0.55,
                label=f"Fold {f}",
            )
        h = prof[prof.endpoint.eq(ep) & prof.variable.eq(v)]
        ax.fill_between(h.x, h.CI_low * mult, h.CI_high * mult, color=color, alpha=0.14)
        ax.plot(h.x, h.prediction * mult, color=color, lw=3, label="Pooled")
        ax.set_title(letter + "  " + ep, loc="left", weight="bold")
        ax.set_xlabel("Human antimicrobial use (DDD/1000/day)")
        ax.set_ylabel(
            "Predicted " + ("RQ_res" if ep == "RQ_res" else "HQ/MRL probability (%)")
        )
        ax.legend(frameon=False, fontsize=9)
    fig.savefig(F / "FigureS3.png", dpi=250)
    fig.savefig(F / "FigureS3.pdf")
    plt.close(fig)
    g = pd.read_csv(OUTPUTS / "country_income_values.csv")
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    for ax, ep, letter in zip(axs, ["occurrence", "concentration"], ["A", "B"]):
        h = g[g.endpoint.eq(ep)]
        data = [
            h.loc[h.income_group.eq(c), "value"].dropna()
            * (100 if ep == "occurrence" else 1)
            for c in order
        ]
        ax.boxplot(
            data, tick_labels=[x.replace(" income", "") for x in order], showfliers=True
        )
        ax.set_title(letter + "  " + ep.capitalize(), loc="left", weight="bold")
        ax.set_ylabel(
            "Country detection frequency (%)"
            if ep == "occurrence"
            else "Country median concentration (µg/kg)"
        )
        if ep == "concentration":
            ax.set_yscale("log")
    fig.savefig(F / "FigureS4.png", dpi=250)
    fig.savefig(F / "FigureS4.pdf")
    plt.close(fig)


def run():
    supplementary()
