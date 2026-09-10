"""
COMPONENT B - calibrated predictive uncertainty.

Phase-3 (kept): split conformalized quantile regression (split CQR) on a
three-way storm-disjoint split (train <= 2016, calibrate 2017-2018, test 2019+).

Phase-4 FIX 2: the fixed 2017-2018 calibration bucket is only 18 storms / ~160
states, so the split-CQR offset is noisy. For the Δwind targets we add
CROSS-CONFORMAL prediction:

    K = 10 GroupKFold folds over ALL pre-2019 storms (grouped by storm_id_code)
    for each fold: fit quantile GBM on the other 9, score conformity on the fold
    pool every fold's scores; Q(alpha) = the (1-alpha) empirical quantile
    refit the quantile GBM on all pre-2019 storms; apply the pooled Q on 2019+

Coverage at nominal 50% / 80% is reported for marginal, split-CQR and
cross-conformal, side by side.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from splits import three_way_report  # noqa: E402

from .common import FRAG, LEADS_H, RANDOM_STATE, REPORTS, deg_to_km
from .segment_resample import load_resampled

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
LEVELS = {0.5: (0.25, 0.75), 0.8: (0.10, 0.90)}
FEATURES = ["lat", "lon", "wind_kt", "pressure_hpa", "pressure_drop",
            "wind_tendency_6h", "pressure_tendency_6h", "dlat_6h", "dlon_6h",
            "storm_age_hours", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "basin_ARB", "basin_BOB", "basin_LAND"]


def _along_cross(d):
    out = {}
    for h in LEADS_H:
        dn, de = deg_to_km(d[f"fut_lat_{h}h"] - d["lat"],
                           d[f"fut_lon_{h}h"] - d["lon"], d["lat"])
        hn, he = deg_to_km(d["dlat_6h"], d["dlon_6h"], d["lat"])
        hmag = np.hypot(hn, he).clip(lower=1e-6)
        un, ue = hn / hmag, he / hmag
        out[f"along_{h}h"] = dn * un + de * ue
        out[f"cross_{h}h"] = dn * (-ue) + de * un
    return pd.DataFrame(out, index=d.index)


def pinball(y, q_pred, q):
    e = y - q_pred
    return float(np.mean(np.maximum(q * e, (q - 1) * e)))


def _qgb(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q,
                                         random_state=RANDOM_STATE, max_iter=300,
                                         learning_rate=0.05, max_depth=4)


def _finite_quantile(e, alpha):
    n = len(e)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.max(e)) if k >= n else float(np.sort(e)[k - 1])


def run() -> dict:
    df = load_resampled()
    df = pd.concat([df, _along_cross(df)], axis=1)
    base = df[df[FEATURES].notna().all(axis=1)]
    train = base[base["storm_year"] <= 2016]
    calib = base[(base["storm_year"] >= 2017) & (base["storm_year"] <= 2018)]
    pre = base[base["storm_year"] < 2019]
    test = base[base["storm_year"] >= 2019]

    targets = ([f"delta_wind_{h}h" for h in LEADS_H]
               + [f"along_{h}h" for h in LEADS_H] + [f"cross_{h}h" for h in LEADS_H])

    report = {
        "split": three_way_report(df),
        "n_states_with_features": {"train_le2016": int(len(train)),
                                   "calib_2017_18": int(len(calib)),
                                   "pre2019_total": int(len(pre)),
                                   "test_2019plus": int(len(test))},
        "quantiles": QUANTILES, "per_target": {},
    }
    plot_data = {}

    for tgt in targets:
        tr = train[train[tgt].notna()]
        ca = calib[calib[tgt].notna()]
        te = test[test[tgt].notna()]
        pr = pre[pre[tgt].notna()]
        if len(te) < 20 or len(ca) < 10:
            continue

        # marginal + split CQR (fit on train<=2016, calib on 2017-18)
        qmods = {q: _qgb(q).fit(tr[FEATURES], tr[tgt]) for q in QUANTILES}
        qcal = {q: qmods[q].predict(ca[FEATURES]) for q in QUANTILES}
        qte = {q: qmods[q].predict(te[FEATURES]) for q in QUANTILES}
        yte, yca = te[tgt].to_numpy(), ca[tgt].to_numpy()

        entry = {"n_test": int(len(te)), "n_calib_split": int(len(ca)),
                 "pinball_mean_over_quantiles": float(np.mean(
                     [pinball(yte, qte[q], q) for q in QUANTILES]))}

        is_dwind = tgt.startswith("delta_wind")
        cc_offsets = {}
        if is_dwind:
            # cross-conformal over ALL pre-2019 storms
            gkf = GroupKFold(n_splits=10)
            pooled = {lvl: [] for lvl in LEVELS}
            Xpr, ypr = pr[FEATURES].to_numpy(), pr[tgt].to_numpy()
            grp = pr["storm_id_code"].to_numpy()
            for tri, vai in gkf.split(Xpr, groups=grp):
                fold_q = {}
                for lvl, (qlo, qhi) in LEVELS.items():
                    mlo = _qgb(qlo).fit(Xpr[tri], ypr[tri]).predict(Xpr[vai])
                    mhi = _qgb(qhi).fit(Xpr[tri], ypr[tri]).predict(Xpr[vai])
                    pooled[lvl].append(np.maximum(mlo - ypr[vai], ypr[vai] - mhi))
            for lvl in LEVELS:
                cc_offsets[lvl] = _finite_quantile(np.concatenate(pooled[lvl]), 1 - lvl)
            # refit quantile GBM on all pre-2019 for the test intervals
            qpr = {q: _qgb(q).fit(Xpr, ypr) for q in QUANTILES}
            qte_pre = {q: qpr[q].predict(te[FEATURES]) for q in QUANTILES}

        pd_entry = {}
        for lvl, (qlo, qhi) in LEVELS.items():
            lo_m, hi_m = qte[qlo], qte[qhi]
            cov_marg = float(np.mean((yte >= lo_m) & (yte <= hi_m)))
            off_s = _finite_quantile(np.maximum(qcal[qlo] - yca, yca - qcal[qhi]), 1 - lvl)
            cov_split = float(np.mean((yte >= lo_m - off_s) & (yte <= hi_m + off_s)))
            e = {"marginal_coverage": cov_marg, "split_cqr_coverage": cov_split,
                 "split_cqr_offset": off_s,
                 "median_width_marginal": float(np.median(hi_m - lo_m)),
                 "median_width_split_cqr": float(np.median((hi_m + off_s) - (lo_m - off_s)))}
            triple = [cov_marg, cov_split, None]
            if is_dwind:
                lo_p, hi_p = qte_pre[qlo], qte_pre[qhi]
                off_c = cc_offsets[lvl]
                cov_cc = float(np.mean((yte >= lo_p - off_c) & (yte <= hi_p + off_c)))
                e.update({"cross_conformal_coverage": cov_cc,
                          "cross_conformal_offset": off_c,
                          "median_width_cross_conformal": float(
                              np.median((hi_p + off_c) - (lo_p - off_c)))})
                triple[2] = cov_cc
            entry[f"nominal_{int(lvl*100)}"] = e
            pd_entry[lvl] = triple
        report["per_target"][tgt] = entry
        plot_data[tgt] = pd_entry

    _plot(plot_data)
    (FRAG / "uncertainty.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("coverage on 2019+ test  (marginal / split-CQR / cross-conformal):")
    for t, e in report["per_target"].items():
        row = []
        for lvl in ("50", "80"):
            n = e[f"nominal_{lvl}"]
            cc = n.get("cross_conformal_coverage")
            row.append(f"{lvl}%: {n['marginal_coverage']:.2f}/"
                       f"{n['split_cqr_coverage']:.2f}/"
                       f"{'--' if cc is None else f'{cc:.2f}'}")
        print(f"  {t:15s} {'   '.join(row)}  (n_test={e['n_test']})")
    return report


def _plot(plot_data: dict):
    cols = [("delta_wind", "Δwind (kt)"), ("along", "along-track (km)"),
            ("cross", "cross-track (km)")]
    fig, ax = plt.subplots(3, 3, figsize=(11, 9))
    for i, h in enumerate(LEADS_H):
        for j, (key, lab) in enumerate(cols):
            a = ax[i, j]
            tgt = f"delta_wind_{h}h" if key == "delta_wind" else f"{key}_{h}h"
            a.axhline(0.5, ls="--", c="0.6", lw=1); a.axhline(0.8, ls="--", c="0.6", lw=1)
            if tgt in plot_data:
                m50, s50, c50 = plot_data[tgt][0.5]
                m80, s80, c80 = plot_data[tgt][0.8]
                xs = [0, 1, 2, 4, 5, 6]
                ys = [m50, s50, c50 if c50 is not None else 0,
                      m80, s80, c80 if c80 is not None else 0]
                cols_ = ["#d8b9a6", "#c8794f", "#7a2e18"] * 2
                a.bar(xs, ys, color=cols_)
                a.set_xticks([1, 5]); a.set_xticklabels(["50%", "80%"])
            a.set_ylim(0, 1); a.set_title(f"+{h} h  {lab}", fontsize=9)
            if j == 0:
                a.set_ylabel("empirical coverage")
    fig.suptitle("Interval coverage on 2019+ test — per group: "
                 "marginal | split-CQR | cross-conformal (Δwind only)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(REPORTS / "reliability_uncertainty.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run()
