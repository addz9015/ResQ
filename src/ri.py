"""
COMPONENT D + phase-4 FIX 3 - rapid-intensification probability, combined.

Target : RI = wind(t+24h) - wind(t) >= 30 kt on the resampled 6 h grid.
Features (classifier): kinematic only (no ERA5 yet).
Split  : all pre-2019 storms for fitting/calibration (GroupKFold K=10 by
         storm_id_code), 2019+ storms for the final test.

Signals compared on the 2019+ test:
  * analog P(RI)           - fraction of retrieved analogs with Δwind >= 30 kt,
                             isotonic-calibrated on the K=10 out-of-fold values
  * RI classifier          - HistGradientBoosting, isotonic-calibrated OOF
  * simple average of the two calibrated probabilities
  * logistic stack of [analog P(RI), classifier P(RI)]

Reported: ROC-AUC, PR-AUC (vs base rate), Brier. A winner is picked, or we say
the combination does not help.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))

from .common import FRAG, RANDOM_STATE
from .segment_resample import load_resampled
from .analogs import AnalogEngine, FEATURES as ANALOG_FEATURES

RI_THRESH = 30.0
KFOLDS = 10
FEATURES = ["lat", "lon", "wind_kt", "pressure_hpa", "pressure_drop",
            "wind_tendency_6h", "pressure_tendency_6h", "dlat_6h", "dlon_6h",
            "storm_age_hours", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "basin_ARB", "basin_BOB", "basin_LAND"]


def _prep(df):
    d = df[df[FEATURES].notna().all(axis=1) & df[ANALOG_FEATURES].notna().all(axis=1)
           & df["delta_wind_24h"].notna()].copy()
    d["ri"] = (d["delta_wind_24h"].to_numpy() >= RI_THRESH).astype(int)
    return d


def _hgc():
    return HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=300,
                                          learning_rate=0.05, max_depth=4,
                                          class_weight="balanced")


def _metrics(y, p):
    return {"roc_auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "n": int(len(y))}


def run() -> dict:
    df = load_resampled()
    eng = AnalogEngine().fit(df)                       # library = pre-2019 storms

    pre = _prep(df[df["storm_year"] < 2019]).reset_index(drop=True)
    test = _prep(df[df["storm_year"] >= 2019]).reset_index(drop=True)
    ypre, ytest = pre["ri"].to_numpy(), test["ri"].to_numpy()
    grp = pre["storm_id_code"].to_numpy()

    # analog raw P(RI)
    a_pre_raw = eng.forecast_distribution(pre)["p_ri_24h"]
    a_test_raw = eng.forecast_distribution(test)["p_ri_24h"]
    a_pre_raw = np.nan_to_num(a_pre_raw, nan=float(np.nanmean(a_pre_raw)))
    a_test_raw = np.nan_to_num(a_test_raw, nan=float(np.nanmean(a_pre_raw)))

    gkf = GroupKFold(n_splits=KFOLDS)
    # OOF classifier probs + OOF isotonic-calibrated analog probs
    clf_oof = np.zeros(len(pre))
    a_oof_cal = np.zeros(len(pre))
    Xpre = pre[FEATURES].to_numpy()
    for tri, vai in gkf.split(Xpre, groups=grp):
        c = _hgc().fit(Xpre[tri], ypre[tri])
        clf_oof[vai] = c.predict_proba(Xpre[vai])[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(a_pre_raw[tri], ypre[tri])
        a_oof_cal[vai] = iso.transform(a_pre_raw[vai])

    # final models on all pre-2019
    clf_full = _hgc().fit(Xpre, ypre)
    clf_test = clf_full.predict_proba(test[FEATURES].to_numpy())[:, 1]
    iso_clf = IsotonicRegression(out_of_bounds="clip").fit(clf_oof, ypre)
    clf_test_cal = iso_clf.transform(clf_test)

    iso_a = IsotonicRegression(out_of_bounds="clip").fit(a_pre_raw, ypre)
    a_test_cal = iso_a.transform(a_test_raw)

    # combinations
    avg_test = 0.5 * (a_test_cal + clf_test_cal)
    stack = LogisticRegression(max_iter=1000).fit(
        np.column_stack([a_oof_cal, clf_oof]), ypre)
    stack_test = stack.predict_proba(
        np.column_stack([a_test_cal, clf_test_cal]))[:, 1]

    base_rate = float(ypre.mean())
    results = {
        "target": "wind(t+24h) - wind(t) >= 30 kt",
        "split": f"pre-2019 GroupKFold K={KFOLDS} for fit/calibration; test = 2019+",
        "n": {"pre2019": int(len(pre)), "test": int(len(test))},
        "train_base_rate": base_rate,
        "test_base_rate": float(ytest.mean()),
        "pr_auc_baseline": float(ytest.mean()),
        "climatology_brier": float(brier_score_loss(ytest, np.full_like(clf_test, base_rate))),
        "analog_p_ri_calibrated": _metrics(ytest, a_test_cal),
        "ri_classifier_calibrated": _metrics(ytest, clf_test_cal),
        "average": _metrics(ytest, avg_test),
        "logistic_stack": _metrics(ytest, stack_test),
        "stack_coefficients": {"analog": float(stack.coef_[0][0]),
                               "classifier": float(stack.coef_[0][1]),
                               "intercept": float(stack.intercept_[0])},
    }
    cand = {k: results[k] for k in ["analog_p_ri_calibrated", "ri_classifier_calibrated",
                                    "average", "logistic_stack"]}
    results["winner_pr_auc"] = max(cand, key=lambda k: cand[k]["pr_auc"])
    results["winner_brier"] = min(cand, key=lambda k: cand[k]["brier"])
    results["combination_helps"] = bool(
        max(cand["average"]["pr_auc"], cand["logistic_stack"]["pr_auc"])
        > max(cand["analog_p_ri_calibrated"]["pr_auc"],
              cand["ri_classifier_calibrated"]["pr_auc"]))

    # reliability of the winner (5 bins)
    win = cand[results["winner_pr_auc"]]
    pw = {"analog_p_ri_calibrated": a_test_cal, "ri_classifier_calibrated": clf_test_cal,
          "average": avg_test, "logistic_stack": stack_test}[results["winner_pr_auc"]]
    edges = np.linspace(0, 1, 6)
    b = np.clip(np.digitize(pw, edges[1:-1]), 0, 4)
    results["winner_reliability_5bin"] = [
        {"bin": f"{edges[k]:.1f}-{edges[k+1]:.1f}", "n": int((b == k).sum()),
         "mean_pred": float(pw[b == k].mean()) if (b == k).any() else None,
         "empirical": float(ytest[b == k].mean()) if (b == k).any() else None}
        for k in range(5)]

    (FRAG / "ri.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # per-state test predictions for the cluster bootstrap (src/bootstrap.py)
    pd.DataFrame({
        "storm_id_code": test["storm_id_code"].to_numpy(),
        "y_ri": ytest,
        "p_analog": a_test_cal, "p_classifier": clf_test_cal,
        "p_average": avg_test, "p_stack": stack_test,
    }).to_parquet(FRAG / "preds_ri.parquet", index=False)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run()
