"""Held-out standardised response profiles across five folds."""

import joblib
import numpy as np
import pandas as pd
from .common import read, save, OUTPUTS
from .models import load


def run():
    grids = read("response_evaluation_grids.csv")
    cohorts = read("response_evaluation_cohorts.csv")
    profiles = []
    contrasts = []
    folds = []
    for ep in ["RQ_res", "HQ_MRL"]:
        g = load(ep)
        co = g.country_iso3.to_numpy()
        src = g.source_id.to_numpy()
        for v in grids.variable.unique():
            grid = (
                grids[grids.endpoint.eq(ep) & grids.variable.eq(v)]
                .sort_values("grid_index")
                .value.to_numpy()
            )
            cohort = cohorts[cohorts.endpoint.eq(ep) & cohorts.variable.eq(v)]
            ids = pd.Index(g.statistic_id).get_indexer(cohort.statistic_id)
            assert (ids >= 0).all()
            w0 = cohort.weight.to_numpy()
            cc = co[ids]
            ss = src[ids]
            pm = np.full((len(g), len(grid)), np.nan)
            fa = np.zeros(len(g), int)
            for fold in range(1, 6):
                z = joblib.load(OUTPUTS / "models" / f"{ep}_fold{fold}.joblib")
                te = z["test_index"]
                obs = te[g[v].iloc[te].notna().to_numpy()]
                if not len(obs):
                    continue
                prep = z["prep"]
                model = z["model"]
                cal = z["calibration"]
                mat = prep.raw(g.iloc[obs])
                j = prep.names.index(v)
                for k, x in enumerate(grid):
                    bb = mat.copy()
                    bb[:, j] = x
                    bb = prep.scaler.transform(bb)
                    p = (
                        np.maximum(0, model.predict(bb))
                        if ep == "RQ_res"
                        else np.clip(
                            cal.predict_proba(model.decision_function(bb)[:, None])[
                                :, 1
                            ],
                            1e-6,
                            1 - 1e-6,
                        )
                    )
                    pm[obs, k] = np.expm1(np.log(10) * p) if ep == "RQ_res" else p
                fa[obs] = fold
            assert np.isfinite(pm[ids]).all()
            arr = pm[ids]
            foldvec = fa[ids]
            mean = np.average(arr, axis=0, weights=w0)
            delta = arr[:, -1] - arr[:, 0]
            boot = {}
            for unit, cluster in [("country", cc), ("source", ss)]:
                rng = np.random.default_rng(20261110)
                cs = np.unique(cluster)
                bp = []
                bd = []
                for _ in range(1000):
                    freq = dict(
                        zip(
                            *np.unique(
                                rng.choice(cs, len(cs), replace=True),
                                return_counts=True,
                            )
                        )
                    )
                    bw = w0 * np.array([freq.get(c, 0) for c in cluster])
                    bp.append(np.average(arr, axis=0, weights=bw))
                    bd.append(np.average(delta, weights=bw))
                lower, upper = np.quantile(bp, [0.025, 0.975], axis=0)
                dlo, dhi = np.quantile(bd, [0.025, 0.975])
                boot[unit] = (lower, upper, dlo, dhi)
            for k, x in enumerate(grid):
                profiles.append(
                    dict(
                        endpoint=ep,
                        variable=v,
                        x=x,
                        prediction=mean[k],
                        CI_low=boot["country"][0][k],
                        CI_high=boot["country"][1][k],
                        source_CI_low=boot["source"][0][k],
                        source_CI_high=boot["source"][1][k],
                    )
                )
            ds = []
            for fold in sorted(set(foldvec)):
                take = foldvec == fold
                dd = float(np.average(delta[take], weights=w0[take]))
                ds.append(dd)
                folds.append(
                    dict(
                        endpoint=ep,
                        variable=v,
                        fold=int(fold),
                        records=int(sum(take)),
                        countries=len(set(cc[take])),
                        sources=len(set(ss[take])),
                        change=dd,
                    )
                )
            contrasts.append(
                dict(
                    endpoint=ep,
                    variable=v,
                    status="evaluated",
                    records=len(ids),
                    countries=len(set(cc)),
                    sources=len(set(ss)),
                    minimum_x=grid[0],
                    maximum_x=grid[-1],
                    change=float(np.average(delta, weights=w0)),
                    CI_low=boot["country"][2],
                    CI_high=boot["country"][3],
                    source_CI_low=boot["source"][2],
                    source_CI_high=boot["source"][3],
                    positive_folds=sum(d > 0 for d in ds),
                    negative_folds=sum(d < 0 for d in ds),
                )
            )
            np.savez_compressed(
                OUTPUTS / f"{ep}_{v}_profiles.npz",
                grid=grid,
                predictions=arr,
                statistic_id=cohort.statistic_id.to_numpy(str),
                weights=w0,
                country_iso3=cc,
                source_id=ss,
                fold=foldvec,
            )
            print(ep, v, "response complete", flush=True)
    save(pd.DataFrame(profiles), "response_profiles.csv")
    save(pd.DataFrame(contrasts), "response_contrasts.csv")
    save(pd.DataFrame(folds), "fold_response_directions.csv")
