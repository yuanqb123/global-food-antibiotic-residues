"""Grouped held-out Kernel SHAP, with exact native-equivalent RBF coalitions."""

import time
import joblib
import numpy as np
import pandas as pd
import shap
from scipy.special import expit
from scipy.stats import rankdata
from shap.utils._legacy import DenseData
from .common import OUTPUTS
from .models import load, vv, base, trans
from .weighting import weights


class FactorizedKernel(shap.KernelExplainer):
    """Official Kernel SHAP sampling and solver, exact batched RBF coalition evaluation."""

    def __init__(self, native, data, svm, cal, groups, kind, **kw):
        self.svm = svm
        self.cal = cal
        self.feature_groups = groups
        self.kind = kind
        self.sv = svm.support_vectors_
        self.gamma = svm._gamma
        self.alpha = svm.dual_coef_[0]
        self.bias = float(svm.intercept_[0])
        self.bglog = self.logs(data.data)
        self.bgbase = self.bglog.sum(axis=0)
        super().__init__(native, data, link="identity", **kw)

    def logs(self, Z):
        Z = np.asarray(Z)
        out = []
        for ix in self.feature_groups:
            a = Z[:, ix]
            s = self.sv[:, ix]
            dist = np.maximum(
                np.sum(a * a, axis=1)[:, None]
                + np.sum(s * s, axis=1)[None, :]
                - 2 * a @ s.T,
                0,
            )
            out.append(-self.gamma * dist)
        return np.asarray(out)

    def post(self, score):
        return (
            np.maximum(0, score)
            if self.kind == "RQ_res"
            else np.clip(
                expit(
                    float(self.cal.coef_[0, 0]) * score + float(self.cal.intercept_[0])
                ),
                1e-6,
                1 - 1e-6,
            )
        )

    def coalition(self, x, masks, individual=False):
        dx = self.logs(np.asarray(x).reshape(1, -1))[:, 0, :]
        delta = dx[:, None, :] - self.bglog
        flat = delta.reshape(len(self.feature_groups), -1)
        result = []
        for k in range(0, len(masks), 128):
            logs = (masks[k : k + 128] @ flat + self.bgbase.reshape(1, -1)).reshape(
                -1, len(self.data.data), len(self.sv)
            )
            score = np.exp(logs) @ self.alpha + self.bias
            pp = self.post(score)
            result.append(pp if individual else pp @ self.data.weights)
        return np.concatenate(result, axis=0)

    def allocate(self):
        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

    def addsample(self, x, m, w):
        self._x = np.asarray(x)[0]
        self.maskMatrix[self.nsamplesAdded] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
        masks = np.zeros(
            (self.nsamplesAdded - self.nsamplesRun, len(self.feature_groups))
        )
        masks[:, self.varyingInds] = self.maskMatrix[
            self.nsamplesRun : self.nsamplesAdded
        ]
        self.ey[self.nsamplesRun : self.nsamplesAdded, 0] = self.coalition(
            self._x, masks
        )
        self.nsamplesRun = self.nsamplesAdded


def run(endpoints=("RQ_res", "HQ_MRL")):
    start = time.time()
    summary = []
    audit = []
    background = []
    for ep in endpoints:
        summary = []
        audit = []
        background = []
        g = load(ep)
        w = weights(g)
        n = len(g)
        co = g.country_iso3.to_numpy()
        source = g.source_id.to_numpy()
        y = (
            np.log10(1 + g.RQ_res.to_numpy())
            if ep == "RQ_res"
            else g.available_path_attention_trigger.to_numpy(float)
        )
        ref = pd.read_csv(OUTPUTS / f"{ep}_heldout_predictions.csv")
        sv = np.full((n, 18), np.nan)
        bases = np.full(n, np.nan)
        pred = np.full(n, np.nan)
        foldvec = np.zeros(n, int)
        for fold in range(1, 6):
            z = joblib.load(OUTPUTS / "models" / f"{ep}_fold{fold}.joblib")
            tr = z["train_index"]
            te = z["test_index"]
            prep = z["prep"]
            model = z["model"]
            cal = z["calibration"]
            a = prep.transform(g.iloc[tr])
            b = prep.transform(g.iloc[te])
            names = prep.names
            assert len(set(tr) & set(te)) == 0

            def native(Z):
                return (
                    np.maximum(0, model.predict(Z))
                    if ep == "RQ_res"
                    else np.clip(
                        cal.predict_proba(model.decision_function(Z)[:, None])[:, 1],
                        1e-6,
                        1 - 1e-6,
                    )
                )

            pp = native(b)
            error = float(np.max(abs(pp - ref.prediction.to_numpy()[te])))
            assert error < 1e-10
            groups = [
                np.array(
                    [
                        j
                        for j, name in enumerate(names)
                        if name == v or name == "missingindicator_" + v
                    ],
                    int,
                )
                for v in vv
            ]
            covered = set(np.concatenate(groups))
            groups.append(np.array([j for j in range(len(names)) if j not in covered]))
            assert sorted(np.concatenate(groups).tolist()) == list(range(a.shape[1]))
            wt = weights(g.iloc[tr])
            rng = np.random.default_rng(20261110 + fold)
            bi = []
            bw = []
            for cc in np.unique(co[tr]):
                ix = np.flatnonzero(co[tr] == cc)
                take = rng.choice(
                    ix, min(2, len(ix)), replace=False, p=wt[ix] / wt[ix].sum()
                )
                bi.extend(take)
                bw.extend([wt[ix].sum() / len(take)] * len(take))
            bi = np.array(bi)
            bw = np.array(bw)
            bw /= bw.sum()
            bg = a[bi]
            bd = DenseData(
                bg, vv + ["Composition and measurement controls"], groups, bw.copy()
            )
            expl = FactorizedKernel(native, bd, model, cal, groups, ep)
            for i, ww in zip(bi, bw):
                background.append(
                    dict(
                        endpoint=ep,
                        fold=fold,
                        statistic_id=g.statistic_id.iloc[tr[i]],
                        country_iso3=co[tr[i]],
                        weight=ww,
                    )
                )
            masks = rng.integers(0, 2, size=(24, 18))
            masks[:2] = [np.zeros(18), np.ones(18)]
            syn = np.tile(bg, (len(masks), 1))
            for k, mask in enumerate(masks):
                for j, ix in enumerate(groups):
                    if mask[j]:
                        syn[k * len(bg) : (k + 1) * len(bg), ix] = b[0, ix]
            err = float(
                np.max(
                    abs(
                        native(syn).reshape(len(masks), len(bg))
                        - expl.coalition(b[0], masks, individual=True)
                    )
                )
            )
            assert err < 1e-8
            native_err = None
            if fold == 1:
                conventional = shap.KernelExplainer(native, bd, link="identity")
                np.random.seed(111010)
                fast = expl.shap_values(b[:1], nsamples=128, l1_reg=0, silent=True)
                np.random.seed(111010)
                slow = conventional.shap_values(
                    b[:1], nsamples=128, l1_reg=0, silent=True
                )
                native_err = float(np.max(abs(fast - slow)))
                assert native_err < 1e-7
            for lo in range(0, len(te), 16):
                hi = min(lo + 16, len(te))
                np.random.seed(20261110 + 1000 * fold + lo)
                sv[te[lo:hi]] = expl.shap_values(
                    b[lo:hi], nsamples=1024, l1_reg=0, silent=True
                )
                print(
                    ep,
                    "fold",
                    fold,
                    "SHAP",
                    hi,
                    "/",
                    len(te),
                    "seconds",
                    round(time.time() - start),
                    flush=True,
                )
            bases[te] = float(expl.expected_value)
            pred[te] = pp
            foldvec[te] = fold
            add = float(np.max(abs(bases[te] + sv[te].sum(axis=1) - pp)))
            assert add < 1e-8
            audit.append(
                dict(
                    endpoint=ep,
                    fold=fold,
                    records=len(te),
                    background_records=len(bg),
                    country_overlap=len(set(co[tr]) & set(co[te])),
                    source_overlap=len(set(source[tr]) & set(source[te])),
                    prediction_reproduction_error=error,
                    coalition_reproduction_error=err,
                    SHAP_additivity_error=add,
                    official_KernelSHAP_comparison_error=native_err,
                )
            )
            np.savez_compressed(
                OUTPUTS / f"{ep}_SHAP_checkpoint.npz",
                SHAP=sv,
                prediction=pred,
                base=bases,
                weights=w,
                statistic_id=g.statistic_id.to_numpy(str),
            )
            pd.DataFrame(audit).to_csv(OUTPUTS / f"{ep}_SHAP_audit.csv", index=False)
        assert np.isfinite(sv).all()
        X = g[vv].copy()
        for v in base:
            if trans[v] == "log1p_z":
                X[v] = np.log1p(X[v])
        out = g[["statistic_id", "country_iso3", "source_id"]].copy()
        out["fold"] = foldvec
        out["y"] = y
        out["weight"] = w
        out["prediction"] = pred
        out["base"] = bases
        for j, v in enumerate(vv):
            out[v + "_SHAP"] = sv[:, j]
            out[v + "_value"] = X[v].to_numpy()
        out["controls_SHAP"] = sv[:, 17]
        out.to_csv(OUTPUTS / f"{ep}_SHAP_all_records.csv", index=False)
        imp = np.average(abs(sv[:, :17]), axis=0, weights=w)
        shares = imp / imp.sum()
        rank = rankdata(-imp, method="min").astype(int)
        cs = np.unique(co)
        rng = np.random.default_rng(20261110)
        bs = []
        br = []
        for _ in range(1000):
            freq = dict(
                zip(
                    *np.unique(
                        rng.choice(cs, len(cs), replace=True), return_counts=True
                    )
                )
            )
            ww = w * np.array([freq.get(c, 0) for c in co])
            ii = np.average(abs(sv[:, :17]), axis=0, weights=ww)
            bs.append(ii / ii.sum())
            br.append(rankdata(-ii, method="min"))
        bs = np.array(bs)
        br = np.array(br)
        for j, v in enumerate(vv):
            summary.append(
                dict(
                    endpoint=ep,
                    variable=v,
                    mean_abs_SHAP=imp[j],
                    share=shares[j],
                    rank=int(rank[j]),
                    share_CI_low=np.quantile(bs[:, j], 0.025),
                    share_CI_high=np.quantile(bs[:, j], 0.975),
                    rank_CI_low=np.quantile(br[:, j], 0.025),
                    rank_CI_high=np.quantile(br[:, j], 0.975),
                    top5_frequency=np.mean(br[:, j] <= 5),
                )
            )
        pd.DataFrame(summary).to_csv(
            OUTPUTS / f"{ep}_indicator_contributions.csv", index=False
        )
        pd.DataFrame(background).to_csv(
            OUTPUTS / f"{ep}_background_records.csv", index=False
        )
        print(
            ep, "Held-out grouped SHAP complete", round(time.time() - start), flush=True
        )
