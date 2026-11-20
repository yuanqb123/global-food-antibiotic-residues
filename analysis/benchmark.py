"""Seven model families under the same nested random-fold validation."""

import time, json, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, StratifiedKFold
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from .models import Prep, load, metric, vv
from .weighting import weights
from .common import OUTPUTS


def run():
    start = time.time()
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    summary = []
    search = []
    selections = []
    foldmetrics = []
    families = ["RF", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "linear", "SVM"]
    protocol = {
        "outer_seed": 20261110,
        "inner_seed": "20261110 + outer fold",
        "outer": 5,
        "inner": 3,
        "features": vv,
        "families": families,
        "same_frozen_cohorts": True,
        "RQ_target": "log10(1+RQ_res)",
        "criterion": "weighted RMSE for RQ; weighted log loss for HQ/MRL",
        "bootstrap": "1000 paired country-cluster resamples, seed 20261110; fixed OOF predictions",
        "family_selection": "inner folds select family and parameters; per-family OOF results also reported",
    }
    (OUTPUTS / "benchmark_protocol.json").write_text(json.dumps(protocol, indent=2))
    for ep in ["RQ_res", "HQ_MRL"]:
        g = load(ep)
        w = weights(g)
        n = len(g)
        y = (
            np.log10(1 + g.RQ_res.to_numpy())
            if ep == "RQ_res"
            else g.available_path_attention_trigger.to_numpy(float)
        )
        co = g.country_iso3.to_numpy()
        isclass = ep == "HQ_MRL"

        def split(ix, k, seed):
            s = (
                StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
                if isclass
                else KFold(n_splits=k, shuffle=True, random_state=seed)
            )
            return [(ix[a], ix[b]) for a, b in s.split(ix, y[ix])]

        def make(f, j, seed):
            if f == "RF":
                return (RandomForestClassifier if isclass else RandomForestRegressor)(
                    n_estimators=350 if j == 0 else 250,
                    max_depth=7 if j == 0 else None,
                    min_samples_leaf=12 if j == 0 else 3,
                    max_features=0.8,
                    n_jobs=2,
                    random_state=seed,
                )
            if f == "ExtraTrees":
                return (ExtraTreesClassifier if isclass else ExtraTreesRegressor)(
                    n_estimators=250,
                    min_samples_leaf=3 if j == 0 else 12,
                    max_features=1.0,
                    n_jobs=2,
                    random_state=seed,
                )
            if f == "XGBoost":
                return (XGBClassifier if isclass else XGBRegressor)(
                    n_estimators=250,
                    learning_rate=0.04,
                    max_depth=2 if j == 0 else 4,
                    reg_lambda=10 if j == 0 else 1,
                    min_child_weight=1 if j == 0 else 0.1,
                    tree_method="hist",
                    n_jobs=2,
                    random_state=seed,
                    verbosity=0,
                )
            if f == "LightGBM":
                return (LGBMClassifier if isclass else LGBMRegressor)(
                    n_estimators=250,
                    learning_rate=0.04,
                    num_leaves=7 if j == 0 else 15,
                    min_child_samples=20 if j == 0 else 10,
                    reg_lambda=10 if j == 0 else 1,
                    n_jobs=2,
                    random_state=seed,
                    deterministic=True,
                    force_col_wise=True,
                    verbosity=-1,
                )
            if f == "CatBoost":
                return (CatBoostClassifier if isclass else CatBoostRegressor)(
                    iterations=250,
                    learning_rate=0.04,
                    depth=3 if j == 0 else 5,
                    l2_leaf_reg=3 if j == 0 else 10,
                    thread_count=2,
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                    bootstrap_type="No",
                    random_strength=0,
                )
            if f == "linear":
                return (
                    LogisticRegression(C=1 if j == 0 else 10, max_iter=2000)
                    if isclass
                    else Ridge(alpha=10 if j == 0 else 100)
                )
            return (
                SVC(
                    C=10 if j == 0 else 100, kernel="rbf", gamma="scale", cache_size=400
                )
                if isclass
                else SVR(
                    C=10 if j == 0 else 100,
                    kernel="rbf",
                    gamma="scale",
                    epsilon=0.1,
                    cache_size=400,
                )
            )

        def fitpredict(f, j, tr, te, seed):
            pp = Prep(ep).fit(g.iloc[tr])
            a = pp.transform(g.iloc[tr])
            b = pp.transform(g.iloc[te])
            ww = weights(g.iloc[tr])
            m = make(f, j, seed)
            if f == "SVM" and isclass:
                score = np.full(len(tr), np.nan)
                for tt, va in split(tr, 3, seed + 77):
                    prep = Prep(ep).fit(g.iloc[tt])
                    mm = make(f, j, seed)
                    mm.fit(
                        prep.transform(g.iloc[tt]),
                        y[tt],
                        sample_weight=weights(g.iloc[tt]),
                    )
                    score[pd.Index(tr).get_indexer(va)] = mm.decision_function(
                        prep.transform(g.iloc[va])
                    )
                cal = LogisticRegression(C=1e6, max_iter=2000).fit(
                    score[:, None], y[tr], sample_weight=ww
                )
                m.fit(a, y[tr], sample_weight=ww)
                p = cal.predict_proba(m.decision_function(b)[:, None])[:, 1]
            else:
                m.fit(a, y[tr], sample_weight=ww)
                p = m.predict_proba(b)[:, 1] if isclass else m.predict(b)
            return np.clip(p, 1e-6, 1 - 1e-6) if isclass else np.maximum(0, p)

        preds = {
            f: np.full(n, np.nan)
            for f in families + ["nested_selection", "training_mean"]
        }
        fa = np.zeros(n, int)
        criterion = "log_loss" if isclass else "RMSE"
        for outer, (tr, te) in enumerate(split(np.arange(n), 5, 20261110), 1):
            inn = {(f, j): np.full(len(tr), np.nan) for f in families for j in [0, 1]}
            for inner, (tt, va) in enumerate(split(tr, 3, 20261110 + outer), 1):
                for (f, j), pr in inn.items():
                    pr[pd.Index(tr).get_indexer(va)] = fitpredict(
                        f, j, tt, va, 111010 + outer * 10 + inner
                    )
                print(
                    ep, outer, inner, "seconds", round(time.time() - start), flush=True
                )
            losses = {k: metric(y[tr], p, w[tr], ep)[criterion] for k, p in inn.items()}
            best = min(losses, key=losses.get)
            for (f, j), loss in losses.items():
                search.append(
                    dict(
                        endpoint=ep,
                        fold=outer,
                        family=f,
                        configuration=j,
                        loss=loss,
                        criterion=criterion,
                    )
                )
            for f in families:
                j = min([0, 1], key=lambda j: losses[f, j])
                preds[f][te] = fitpredict(f, j, tr, te, 111010 + outer)
                selections.append(
                    dict(
                        endpoint=ep,
                        fold=outer,
                        family=f,
                        configuration=j,
                        inner_loss=losses[f, j],
                        selected_overall=(f, j) == best,
                    )
                )
            preds["nested_selection"][te] = preds[best[0]][te]
            preds["training_mean"][te] = np.average(y[tr], weights=weights(g.iloc[tr]))
            fa[te] = outer
            for f, p in preds.items():
                foldmetrics.append(
                    dict(
                        endpoint=ep,
                        fold=outer,
                        family=f,
                        **metric(y[te], p[te], w[te], ep),
                    )
                )
            pd.DataFrame(search).to_csv(OUTPUTS / "inner_search.csv", index=False)
            pd.DataFrame(selections).to_csv(OUTPUTS / "selections.csv", index=False)
            print(ep, "OUTER", outer, "DONE", best, flush=True)
        out = g[["statistic_id", "country_iso3", "source_id"]].copy()
        out["y"] = y
        out["weight"] = w
        out["fold"] = fa
        for f, p in preds.items():
            assert np.isfinite(p).all()
            out[f] = p
        out.to_csv(OUTPUTS / f"{ep}_OOF.csv", index=False)
        boots = {f: [] for f in preds}
        rng = np.random.default_rng(20261110)
        cs = np.unique(co)
        for _ in range(1000):
            freq = dict(
                zip(
                    *np.unique(
                        rng.choice(cs, len(cs), replace=True), return_counts=True
                    )
                )
            )
            ww = w * np.array([freq.get(c, 0) for c in co])
            take = ww > 0
            for f, p in preds.items():
                boots[f].append(metric(y[take], p[take], ww[take], ep))
        for f, p in preds.items():
            row = dict(
                endpoint=ep,
                family=f,
                records=n,
                countries=len(cs),
                **metric(y, p, w, ep),
            )
            bb = pd.DataFrame(boots[f])
            refb = pd.DataFrame(boots["SVM"])
            for k in bb:
                row[k + "_low"], row[k + "_high"] = np.nanquantile(
                    bb[k], [0.025, 0.975]
                )
                row["delta_" + k + "_low"], row["delta_" + k + "_high"] = (
                    np.nanquantile(bb[k] - refb[k], [0.025, 0.975])
                )
            summary.append(row)
        pd.DataFrame(summary).to_csv(
            OUTPUTS / "model_family_performance.csv", index=False
        )
        pd.DataFrame(foldmetrics).to_csv(
            OUTPUTS / "model_family_fold_metrics.csv", index=False
        )
        print(pd.DataFrame(summary).to_string(index=False), flush=True)
