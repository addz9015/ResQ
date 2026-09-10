"""
Phase-6 M1 + M2 — the RI ensemble is the one component that beats its baseline,
so it is the one worth tuning.

M1  Model selection on **pre-2019 GroupKFold out-of-fold PR-AUC only**. The 2019+
    test set is scored once at the end and never influences a choice.
      - HistGradientBoosting hyper-parameters for the classifier
      - analog k and the averaging weight  w * P_analog + (1-w) * P_classifier
M2  Learning curve: OOF PR-AUC vs number of training storms (50 / 100 / 150 /
    200 / all) -> reports/learning_curve_ri.png, with a data-starved vs
    overfitting verdict.

Outputs: reports/_fragments/ri_tune.json, reports/learning_curve_ri.png
Run:  python -m src.ri_tune
"""
from __future__ import annotations

import json
import itertools
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from .common import FRAG, REPORTS, RANDOM_STATE
from .segment_resample import load_resampled
from .analogs import AnalogEngine, FEATURES as ANALOG_FEATURES

RI_THRESH = 30.0
KFOLDS = 10
FEATURES = ["lat", "lon", "wind_kt", "pressure_hpa", "pressure_drop",
            "wind_tendency_6h", "pressure_tendency_6h", "dlat_6h", "dlon_6h",
            "storm_age_hours", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "basin_ARB", "basin_BOB", "basin_LAND"]

HGB_GRID = {
    "learning_rate": [0.05, 0.1],
    "max_depth": [2, 3, 4],
    "max_iter": [300],
    "l2_regularization": [0.0, 1.0],
    "min_samples_leaf": [20, 40],
}
K_GRID = [5, 10, 15, 20, 30]
W_GRID = [0.0, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]   # w on analog


def _prep(df):
    d = df[df[FEATURES].notna().all(axis=1) & df[ANALOG_FEATURES].notna().all(axis=1)
           & df["delta_wind_24h"].notna()].copy()
    d["ri"] = (d["delta_wind_24h"].to_numpy() >= RI_THRESH).astype(int)
    return d.reset_index(drop=True)


def _hgb(p):
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                          class_weight="balanced", **p)


def oof_classifier_proba(X, y, groups, params, n_splits=KFOLDS):
    oof = np.zeros(len(y))
    for tri, vai in GroupKFold(n_splits).split(X, y, groups):
        oof[vai] = _hgb(params).fit(X[tri], y[tri]).predict_proba(X[vai])[:, 1]
    return oof


def analog_oof_praw(pre, k):
    """Out-of-fold raw analog P(RI): for each fold the analog library is the
    other folds' storms only (so a pre-2019 query never sees its own fold)."""
    praw = np.zeros(len(pre))
    df_all = load_resampled()
    gkf = GroupKFold(KFOLDS)
    for tri, vai in gkf.split(pre, pre["ri"], pre["storm_id_code"]):
        train_storms = set(pre.iloc[tri]["storm_id_code"])
        lib = df_all[df_all["storm_id_code"].isin(train_storms)]
        eng = AnalogEngine(k=k)
        eng.lib_ = None
        # fit on the restricted library
        eng = _fit_analog_on(eng, lib)
        d = eng.forecast_distribution(pre.iloc[vai])
        praw[vai] = np.nan_to_num(d["p_ri_24h"], nan=0.0)
    return praw


def _fit_analog_on(eng: AnalogEngine, lib_df):
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler
    m = lib_df[ANALOG_FEATURES].notna().all(axis=1) & lib_df["delta_wind_6h"].notna() \
        & lib_df["fut_lat_6h"].notna()
    eng.lib_ = lib_df[m].reset_index(drop=True)
    eng.scaler_ = StandardScaler().fit(eng.lib_[ANALOG_FEATURES].to_numpy())
    eng.nn_ = NearestNeighbors(n_neighbors=min(eng.pool, len(eng.lib_)),
                               metric="euclidean", algorithm="kd_tree")
    eng.nn_.fit(eng.scaler_.transform(eng.lib_[ANALOG_FEATURES].to_numpy()))
    return eng


def m1_select():
    df = load_resampled()
    pre = _prep(df[df["storm_year"] < 2019])
    test = _prep(df[df["storm_year"] >= 2019])
    X, y, g = pre[FEATURES].to_numpy(), pre["ri"].to_numpy(), pre["storm_id_code"].to_numpy()

    # --- 1. HGB hyper-params: OOF PR-AUC of the classifier alone ---
    keys = list(HGB_GRID)
    combos = [dict(zip(keys, v)) for v in itertools.product(*HGB_GRID.values())]
    hgb_scores = []
    for p in combos:
        oof = oof_classifier_proba(X, y, g, p)
        hgb_scores.append((average_precision_score(y, oof), p, oof))
    hgb_scores.sort(key=lambda t: -t[0])
    best_ap, best_hgb, best_clf_oof = hgb_scores[0]

    # --- 2. analog k: OOF PR-AUC of raw analog P(RI) ---
    k_scores = []
    for k in K_GRID:
        praw = analog_oof_praw(pre, k)
        k_scores.append((average_precision_score(y, praw), k, praw))
    k_scores.sort(key=lambda t: -t[0])
    best_k_ap, best_k, best_analog_oof_raw = k_scores[0]

    # calibrate both OOF signals (nested: isotonic fit per training design is
    # overkill here - use a single isotonic on the full OOF, standard practice)
    clf_cal = IsotonicRegression(out_of_bounds="clip").fit_transform(best_clf_oof, y)
    ana_cal = IsotonicRegression(out_of_bounds="clip").fit_transform(best_analog_oof_raw, y)

    # --- 3. averaging weight on the CALIBRATED OOF signals ---
    w_scores = [(average_precision_score(y, w * ana_cal + (1 - w) * clf_cal), w)
                for w in W_GRID]
    w_scores.sort(key=lambda t: -t[0])
    best_w_ap, best_w = w_scores[0]

    selected = {"hgb_params": best_hgb, "analog_k": int(best_k),
                "average_weight_on_analog": float(best_w)}
    oof = {"classifier_pr_auc": float(best_ap),
           "analog_raw_pr_auc": float(best_k_ap),
           "weighted_ensemble_pr_auc": float(best_w_ap)}

    # --- FINAL: refit on all pre-2019, score once on 2019+ test ---
    clf_full = _hgb(best_hgb).fit(X, y)
    iso_clf = IsotonicRegression(out_of_bounds="clip").fit(best_clf_oof, y)
    iso_ana = IsotonicRegression(out_of_bounds="clip").fit(best_analog_oof_raw, y)

    eng = AnalogEngine(k=best_k).fit(df)          # library = all pre-2019 storms
    a_test_raw = np.nan_to_num(eng.forecast_distribution(test)["p_ri_24h"], nan=0.0)
    p_clf_t = iso_clf.transform(clf_full.predict_proba(test[FEATURES].to_numpy())[:, 1])
    p_ana_t = iso_ana.transform(a_test_raw)
    yte = test["ri"].to_numpy()
    p_ens_t = best_w * p_ana_t + (1 - best_w) * p_clf_t

    test_scores = {
        "classifier_pr_auc": float(average_precision_score(yte, p_clf_t)),
        "analog_pr_auc": float(average_precision_score(yte, p_ana_t)),
        "tuned_ensemble_pr_auc": float(average_precision_score(yte, p_ens_t)),
    }

    prev = json.loads((FRAG / "ri.json").read_text())
    prev_test = {"average_pr_auc": prev["average"]["pr_auc"],
                 "classifier_pr_auc": prev["ri_classifier_calibrated"]["pr_auc"],
                 "analog_pr_auc": prev["analog_p_ri_calibrated"]["pr_auc"]}

    delta = test_scores["tuned_ensemble_pr_auc"] - prev_test["average_pr_auc"]
    return {
        "grid_sizes": {"hgb": len(combos), "k": len(K_GRID), "w": len(W_GRID)},
        "selected": selected,
        "oof_pr_auc": oof,
        "pre_tuning_test_pr_auc": prev_test,
        "post_tuning_test_pr_auc": test_scores,
        "test_pr_auc_change_vs_prev_average": float(delta),
        "test_score_moved": bool(abs(delta) >= 0.01),
        "note": ("selection used ONLY pre-2019 GroupKFold OOF PR-AUC; the 2019+ "
                 "test was scored once, after all choices were frozen"),
        "_clf_oof": best_clf_oof.tolist(), "_y": y.tolist(),
        "_analog_oof_raw": best_analog_oof_raw.tolist(),
        "_groups": g.tolist(),
    }


def m2_learning_curve(m1):
    """OOF PR-AUC of the tuned ensemble vs number of training storms."""
    df = load_resampled()
    pre = _prep(df[df["storm_year"] < 2019])
    X, y, g = pre[FEATURES].to_numpy(), pre["ri"].to_numpy(), pre["storm_id_code"].to_numpy()
    storms = np.array(sorted(pre["storm_id_code"].unique()))
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(storms)
    hgb = m1["selected"]["hgb_params"]
    w = m1["selected"]["average_weight_on_analog"]

    N = len(storms)
    sizes = sorted(set([min(s, N) for s in (50, 90, 130, 170, N)]))
    curve = []
    for n in sizes:
        keep = set(storms[:n])
        mask = pre["storm_id_code"].isin(keep).to_numpy()
        Xs, ys, gs = X[mask], y[mask], g[mask]
        if ys.sum() < KFOLDS:
            continue
        clf_oof = oof_classifier_proba(Xs, ys, gs, hgb)
        sub = pre[mask].reset_index(drop=True)
        ana_oof = analog_oof_praw(sub, m1["selected"]["analog_k"])
        cc = IsotonicRegression(out_of_bounds="clip").fit_transform(clf_oof, ys)
        ac = IsotonicRegression(out_of_bounds="clip").fit_transform(ana_oof, ys)
        ap_ens = average_precision_score(ys, w * ac + (1 - w) * cc)
        ap_clf = average_precision_score(ys, cc)
        curve.append({"n_storms": int(min(n, len(storms))), "n_states": int(mask.sum()),
                      "n_RI": int(ys.sum()), "oof_pr_auc_ensemble": float(ap_ens),
                      "oof_pr_auc_classifier": float(ap_clf),
                      "base_rate": float(ys.mean())})

    _plot_curve(curve)
    # trend = OLS slope of OOF PR-AUC vs n_storms over the second half of the curve
    xs = np.array([c["n_storms"] for c in curve], float)
    ys = np.array([c["oof_pr_auc_ensemble"] for c in curve], float)
    half = len(curve) // 2
    slope = float(np.polyfit(xs[half:], ys[half:], 1)[0]) if len(curve) - half >= 2 else 0.0
    endpoint_is_max = ys[-1] >= ys.max() - 1e-9
    rising = slope > 1e-4 and endpoint_is_max
    regime = ("**data-starved on positive cases.** The OOF PR-AUC curve is noisy "
              "(only ~20-105 RI events across the sampled sizes) but its trend over "
              f"the larger sizes is upward (slope +{slope:.2e} per storm) and the "
              "full-data point is the highest. More RI-labelled storms — i.e. more "
              "years, which in practice means the ERA5 era — should raise the "
              "ceiling. It is not overfitting (see M3)."
              if rising else
              "**plateaued / feature-limited.** The OOF PR-AUC curve has flattened "
              f"(slope {slope:+.2e} per storm over the larger sizes); more storms "
              "of the same kind add little. The ceiling is the kinematic feature "
              "set, which is why ERA5 shear/SST is the real lever.")
    return {"curve": curve, "regime": regime, "second_half_slope": slope,
            "endpoint_is_max": bool(endpoint_is_max), "rising": bool(rising)}


def _plot_curve(curve):
    n = [c["n_storms"] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(n, [c["oof_pr_auc_ensemble"] for c in curve], "o-", color="#b0472e",
            label="tuned ensemble")
    ax.plot(n, [c["oof_pr_auc_classifier"] for c in curve], "s--", color="#5aa9e6",
            label="classifier alone")
    ax.axhline(curve[-1]["base_rate"], ls=":", color="0.6",
               label=f"base rate {curve[-1]['base_rate']:.02f}")
    for c in curve:
        ax.annotate(f"{c['n_RI']} RI", (c["n_storms"], c["oof_pr_auc_ensemble"]),
                    textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center")
    ax.set_xlabel("number of training storms (pre-2019)")
    ax.set_ylabel("out-of-fold PR-AUC (GroupKFold)")
    ax.set_title("RI ensemble learning curve — is it data-starved or overfitting?")
    ax.legend(fontsize=8); ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(REPORTS / "learning_curve_ri.png", dpi=120)
    plt.close(fig)


def main():
    m1 = m1_select()
    m2 = m2_learning_curve(m1)
    out = {"M1_tuning": {k: v for k, v in m1.items() if not k.startswith("_")},
           "M2_learning_curve": m2}
    (FRAG / "ri_tune.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
