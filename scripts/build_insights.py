"""
Assemble app/static/data/insights.json for the dashboard's "Model insights" tab.

Everything here is READ from reports/_fragments/*.json (produced by the pipeline)
plus one cheap permutation-importance fit. No model is tuned or trained at page
load; the frontend just fetches this file.

Run:  python scripts/build_insights.py     (also called by app/run.py --rebuild)
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FRAG = ROOT / "reports" / "_fragments"
OUT = ROOT / "app" / "static" / "data" / "insights.json"
LEADS = ("6h", "12h", "24h")


def j(name, default=None):
    p = FRAG / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def _ci(d):
    return {"point": d["point"], "lo": d["lo"], "hi": d["hi"]}


def primary_table():
    m = j("models.json"); bs = j("bootstrap.json")
    rows = m["primary_intersection"]["rows"]
    n = m["primary_intersection"]["n_per_lead"]
    bt = (bs or {}).get("track_intensity", {})
    out = {"n_per_lead": n, "models": []}
    label = {
        "persistence": "persistence (Δ=0 / no motion)",
        "persistence_motion": "persistence-motion (best track baseline)",
        "true_cliper": "true CLIPER (OLS)",
        "gbm_direct": "GBM — direct",
        "gbm_motion_resid": "GBM — motion + residual",
        "lstm_direct": "LSTM — direct",
        "lstm_motion_resid": "LSTM — motion + residual",
        "analog": "analog-ensemble (k=10)",
    }
    for key, lab in label.items():
        if key not in rows:
            continue
        row = {"model": lab, "key": key, "track": {}, "dwind": {}}
        for l in LEADS:
            tv = rows[key].get("track", {}).get(l)
            dv = rows[key].get("dwind_MAE", {}).get(l)
            tci = bt.get("track", {}).get(key, {}).get(l)
            dkey = {"persistence": "persistence", "gbm_direct": "gbm",
                    "analog": "analog"}.get(key)
            dci = bt.get("intensity", {}).get(dkey, {}).get(l) if dkey else None
            row["track"][l] = {"value": tv, "ci": _ci(tci) if tci else None}
            row["dwind"][l] = {"value": dv, "ci": _ci(dci) if dci else None}
        out["models"].append(row)
    return out


def supported_panel():
    bs = j("bootstrap.json")
    if not bs:
        return {}
    ri = bs["ri"]["paired"]; ti = bs["track_intensity"]
    sup, notsup = [], []

    ac = ri["average_beats_climatology_brier"]
    (sup if ac["sig_either_way"] and ac["improvement"] > 0 else notsup).append(
        "combined P(RI) beats a climatological base-rate forecast (Brier)")
    dc = ri["average_minus_classifier_pr_auc"]
    (sup if dc["significant"] else notsup).append(
        "combined P(RI) beats the classifier alone (PR-AUC)")
    da = ri["average_minus_analog_pr_auc"]
    (sup if da["significant"] else notsup).append(
        "combined P(RI) beats the analog ensemble alone (PR-AUC)")

    t6 = ti["track_paired"]["true_cliper_vs_persistence_motion"]["6h"]
    (sup if t6["sig_either_way"] else notsup).append(
        "persistence-motion beats true CLIPER on track at 6/12 h"
        if t6["improvement"] < 0 else "true CLIPER beats persistence-motion at 6 h")

    any_int = any(ti["intensity_paired"][f"{a}_vs_persistence"][l]["significant"]
                  for a in ("gbm", "analog") for l in LEADS)
    (sup if any_int else notsup).append(
        "any intensity model beats the Δ=0 baseline")
    any_trk = any(ti["track_paired"][f"{a}_vs_persistence_motion"][l]["significant"]
                  for a in ("gbm_direct", "gbm_motion_resid", "lstm_direct",
                            "lstm_motion_resid", "analog", "true_cliper")
                  for l in LEADS)
    (sup if any_trk else notsup).append(
        "any learned track model beats persistence-motion")
    return {"supported": sup, "not_supported": notsup,
            "n_test_storms": bs["ri"]["n_test_storms"],
            "n_RI_positive": bs["ri"]["n_RI_positive"]}


def ri_panel():
    ri = j("ri.json"); bs = j("bootstrap.json")
    m = bs["ri"]["metrics"] if bs else {}
    prauc = {name: _ci(m[f"{name}_pr_auc"]) if f"{name}_pr_auc" in m else None
             for name in ("analog", "classifier", "average", "stack")}
    rocauc = {name: _ci(m[f"{name}_roc_auc"]) if f"{name}_roc_auc" in m else None
              for name in ("analog", "classifier", "average", "stack")}
    return {"pr_auc_ci": prauc, "roc_auc_ci": rocauc,
            "pr_auc_baseline": ri["pr_auc_baseline"],
            "reliability": ri["winner_reliability_5bin"],
            "winner": ri["winner_pr_auc"],
            "climatology_brier": ri["climatology_brier"]}


def coverage_panel():
    u = j("uncertainty.json")
    rows = []
    for l in LEADS:
        for lvl in ("50", "80"):
            e = u["per_target"][f"delta_wind_{l}"][f"nominal_{lvl}"]
            rows.append({"lead": l, "nominal": int(lvl) / 100,
                         "marginal": e["marginal_coverage"],
                         "split_cqr": e["split_cqr_coverage"],
                         "cross_conformal": e.get("cross_conformal_coverage")})
    return {"rows": rows,
            "n_test": u["per_target"]["delta_wind_24h"]["n_test"]}


def perm_importance():
    """Permutation importance for the RI classifier (PR-AUC drop when a feature
    is shuffled). Fit on <=2016 storms, scored on the 2017-2018 slice so nothing
    from 2019+ is touched."""
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import make_scorer, average_precision_score
    sys.path.insert(0, str(ROOT / "scripts"))
    from src.segment_resample import load_resampled
    from src.ri_tune import _prep, FEATURES

    d = _prep(load_resampled())
    tr = d[d["storm_year"] <= 2016]
    va = d[(d["storm_year"] >= 2017) & (d["storm_year"] <= 2018)]
    if va["ri"].nunique() < 2:
        return {"available": False}
    clf = HistGradientBoostingClassifier(random_state=26070, max_iter=300,
                                         learning_rate=0.05, max_depth=4,
                                         class_weight="balanced")
    clf.fit(tr[FEATURES].to_numpy(), tr["ri"].to_numpy())
    r = permutation_importance(
        clf, va[FEATURES].to_numpy(), va["ri"].to_numpy(), n_repeats=20,
        random_state=26070,
        scoring=make_scorer(average_precision_score, needs_proba=True))
    order = np.argsort(r.importances_mean)[::-1]
    return {"available": True,
            "scored_on": "2017-2018 held-out slice", "metric": "PR-AUC drop",
            "features": [{"name": FEATURES[i],
                          "mean": float(r.importances_mean[i]),
                          "std": float(r.importances_std[i])} for i in order]}


def main():
    rt = j("ri_tune.json", {})
    fd = j("fit_diagnostics.json", {})
    out = {
        "generated": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "primary": primary_table(),
        "supported": supported_panel(),
        "ri": ri_panel(),
        "coverage": coverage_panel(),
        "perm_importance": perm_importance(),
        "tuning": rt.get("M1_tuning", {}),
        "learning_curve": rt.get("M2_learning_curve", {}),
        "learning_curve_png": "/static/img/learning_curve_ri.png",
        "fit_diagnostics": fd,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=_np), encoding="utf-8")
    # copy figures the frontend references
    img = OUT.parent.parent / "img"
    img.mkdir(exist_ok=True)
    for fig in ("learning_curve_ri.png", "reliability_uncertainty.png"):
        src = ROOT / "reports" / fig
        if src.exists():
            (img / fig).write_bytes(src.read_bytes())
    print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} KB)")


def _np(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))


if __name__ == "__main__":
    main()
