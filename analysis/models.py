"""Seventeen-indicator random-fold SVR and calibrated SVC analyses."""

import json
import numpy as np
import pandas as pd
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from .common import read, save, load_config, OUTPUTS
from .weighting import weights

CFG = load_config()
vv = CFG["national_indicators"]
base = list(CFG["indicator_transforms"])
trans = CFG["indicator_transforms"]
aq = CFG["aquatic_only"]


class Prep:

    def __init__(self, ep):
        self.ep = ep
        self.cats = [
            "food_group",
            "antibiotic_std",
            "statistic_type",
            "sample_source",
            "analytical_method",
            "sampling_design",
        ] + (
            ["tissue_std", "antibiotic_class_broad", "map_location_level"]
            if ep == "RQ_res"
            else ["marker_match"]
        )

    def national(self, g):
        x = g[vv].copy()
        for v in base:
            if trans[v] == "log1p_z":
                x[v] = np.log1p(x[v])
        return x

    def fit(self, g):
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(
            g[self.cats].fillna("missing")
        )
        self.imyear = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ).fit(g[["year_mid"]])
        self.im = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ).fit(self.national(g))
        self.names = (
            [v for v, c in zip(self.cats, self.enc.categories_) for _ in c]
            + list(self.imyear.get_feature_names_out(["year_mid"]))
            + list(self.im.get_feature_names_out(vv))
        )
        if self.ep == "RQ_res":
            self.names += ["log10_intake_known", "log10_PNEC_known"]
        self.scaler = StandardScaler().fit(self.raw(g))
        return self

    def raw(self, g):
        out = np.c_[
            self.enc.transform(g[self.cats].fillna("missing")),
            self.imyear.transform(g[["year_mid"]]),
            self.im.transform(self.national(g)),
        ]
        if self.ep == "RQ_res":
            out = np.c_[out, g[["log10_intake_known", "log10_PNEC_known"]].to_numpy()]
        return out

    def transform(self, g):
        return self.scaler.transform(self.raw(g))


def metric(y, p, w, ep):
    mse = np.average((y - p) ** 2, weights=w)
    out = dict(RMSE=float(np.sqrt(mse)))
    if ep == "RQ_res":
        out["R2"] = float(
            1 - mse / np.average((y - np.average(y, weights=w)) ** 2, weights=w)
        )
    else:
        out["AUC"] = (
            float(roc_auc_score(y, p, sample_weight=w))
            if len(np.unique(y[w > 0])) == 2
            else np.nan
        )
        out["log_loss"] = float(
            np.average(-y * np.log(p) - (1 - y) * np.log(1 - p), weights=w)
        )
        out["Brier"] = float(mse)
    return out


def load(ep):
    g = read(ep + "_model_input.csv")
    g.loc[~g.food_group.eq("aquatic products"), aq] = np.nan
    if ep == "RQ_res":
        thresholds = read("resistance_selection_thresholds.csv")
        pm = (
            thresholds[thresholds.version.eq("WHO2024")]
            .set_index("antibiotic_std")
            .PNEC_res_ug_L
        )
        g["log10_intake_known"] = np.log10(g.food_intake_kg_day)
        g["log10_PNEC_known"] = np.log10(g.antibiotic_std.map(pm))
    return g


def split(g, y, indices, folds, seed, ep):
    splitter = (
        StratifiedKFold(folds, shuffle=True, random_state=seed)
        if ep == "HQ_MRL"
        else KFold(folds, shuffle=True, random_state=seed)
    )
    return [(indices[a], indices[b]) for a, b in splitter.split(indices, y[indices])]


def fit_predict(g, y, tr, te, C, ep, seed):
    prep = Prep(ep).fit(g.iloc[tr])
    calibration = None
    if ep == "HQ_MRL":
        score = np.full(len(tr), np.nan)
        for tt, va in split(g, y, tr, 3, seed + 77, ep):
            cp = Prep(ep).fit(g.iloc[tt])
            cm = SVC(C=C, kernel="rbf", gamma="scale", cache_size=400)
            cm.fit(cp.transform(g.iloc[tt]), y[tt], sample_weight=weights(g.iloc[tt]))
            score[pd.Index(tr).get_indexer(va)] = cm.decision_function(
                cp.transform(g.iloc[va])
            )
        calibration = LogisticRegression(C=1000000.0, max_iter=2000).fit(
            score[:, None], y[tr], sample_weight=weights(g.iloc[tr])
        )
    model = (
        SVR(C=C, kernel="rbf", gamma="scale", epsilon=0.1, cache_size=400)
        if ep == "RQ_res"
        else SVC(C=C, kernel="rbf", gamma="scale", cache_size=400)
    )
    model.fit(prep.transform(g.iloc[tr]), y[tr], sample_weight=weights(g.iloc[tr]))
    matrix = prep.transform(g.iloc[te])
    prediction = (
        np.maximum(0, model.predict(matrix))
        if ep == "RQ_res"
        else np.clip(
            calibration.predict_proba(model.decision_function(matrix)[:, None])[:, 1],
            1e-06,
            1 - 1e-06,
        )
    )
    return (
        prediction,
        dict(
            prep=prep,
            model=model,
            calibration=calibration,
            train_index=tr,
            test_index=te,
            C=C,
        ),
    )


def run():
    import joblib

    (OUTPUTS / "models").mkdir(exist_ok=True)
    metrics = []
    selections = []
    for ep in ["RQ_res", "HQ_MRL"]:
        g = load(ep)
        y = (
            np.log10(1 + g.RQ_res.to_numpy())
            if ep == "RQ_res"
            else g.available_path_attention_trigger.to_numpy(float)
        )
        weight = weights(g)
        pred = np.full(len(g), np.nan)
        fold_id = np.zeros(len(g), int)
        for outer, (tr, te) in enumerate(
            split(g, y, np.arange(len(g)), 5, 20261110, ep), 1
        ):
            candidates = {C: np.full(len(tr), np.nan) for C in [10, 100]}
            for inner, (tt, va) in enumerate(
                split(g, y, tr, 3, 20261110 + outer, ep), 1
            ):
                for C in candidates:
                    p, _ = fit_predict(g, y, tt, va, C, ep, 111010 + outer * 10 + inner)
                    candidates[C][pd.Index(tr).get_indexer(va)] = p
            key = "RMSE" if ep == "RQ_res" else "log_loss"
            scores = {
                C: metric(y[tr], p, weight[tr], ep)[key] for C, p in candidates.items()
            }
            C = min(scores, key=scores.get)
            p, bundle = fit_predict(g, y, tr, te, C, ep, 111010 + outer)
            pred[te] = p
            fold_id[te] = outer
            joblib.dump(bundle, OUTPUTS / "models" / f"{ep}_fold{outer}.joblib")
            selections.append(dict(endpoint=ep, fold=outer, C=C))
            print(ep, "fold", outer, "C", C, flush=True)
        out = g[
            [
                "statistic_id",
                "canonical_row",
                "sampling_frame_id",
                "source_id",
                "country_iso3",
            ]
        ].copy()
        out["y"] = y
        out["weight"] = weight
        out["prediction"] = pred
        out["fold"] = fold_id
        save(out, ep + "_heldout_predictions.csv")
        metrics.append(
            dict(
                endpoint=ep,
                records=len(g),
                countries=g.country_iso3.nunique(),
                sources=g.source_id.nunique(),
                **metric(y, pred, weight, ep),
            )
        )
    save(pd.DataFrame(metrics), "SVM_performance.csv")
    save(pd.DataFrame(selections), "SVM_selected_parameters.csv")
    return metrics
