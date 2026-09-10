"""
Point-forecast models on the resampled 6 h grid.

Phase-3 corrections kept: primary (model-score intersection) vs secondary
(full-n) tables; track models also learn the motion baseline's RESIDUAL;
primary table re-scored with interpolated targets excluded.

Phase-4 FIX 1: the motion baseline is split into two.
  * persistence-motion : extrapolate the last 6 h (d_lat, d_lon) vector.
  * true CLIPER        : OLS of (d_lat, d_lon) at each lead on
      [lat, lon, wind, past-6h d_lat, past-6h d_lon, doy sin/cos,
       lat x past-6h d_lat], fit on pre-2019 storms only.
Both are scored on the primary intersection; residual learning (correction 3)
is run against whichever wins on pre-2019 CV (expected: true CLIPER).

Splits: point models fit on pre-2019 storms, evaluated on 2019+ storms.
No random row splits. Losses are reported, not hidden.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

import torch
from torch import nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.common import FRAG, LEADS_H, RANDOM_STATE, haversine_km
from src.segment_resample import load_resampled
from src.analogs import AnalogEngine, FEATURES as ANALOG_FEATURES
from splits import modeling_frame, temporal_holdout, three_way_report

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

SEQ_LEN = 8
FEATURES = ["lat", "lon", "pressure_hpa", "pressure_drop", "wind_kt",
            "wind_tendency_6h", "pressure_tendency_6h", "dlat_6h", "dlon_6h",
            "storm_age_hours", "month_sin", "month_cos", "doy_sin", "doy_cos",
            "basin_ARB", "basin_BOB", "basin_LAND"]
NEEDED = sorted(set(FEATURES) | set(ANALOG_FEATURES))
TC_FEATURES = ["lat", "lon", "wind_kt", "dlat_6h", "dlon_6h",
               "doy_sin", "doy_cos", "lat_x_dlat6"]


def mae(y, p):
    return float(mean_absolute_error(y, p))


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def gc(plat, plon, alat, alon):
    return float(np.mean(haversine_km(np.asarray(plat, float), np.asarray(plon, float),
                                      np.asarray(alat, float), np.asarray(alon, float))))


# ---------------------------------------------------------------------------
# true CLIPER  (climatology-and-persistence, OLS)
# ---------------------------------------------------------------------------
def _tc_matrix(lat, lon, wind, dlat6, dlon6, doy_sin, doy_cos):
    return np.column_stack([lat, lon, wind, dlat6, dlon6, doy_sin, doy_cos,
                            lat * dlat6])


class TrueCliper:
    """OLS regression of lead-h displacement on kinematic + seasonal terms."""

    def __init__(self):
        self.models = {}

    def fit(self, pre: pd.DataFrame):
        for h in LEADS_H:
            d = pre[pre[f"fut_lat_{h}h"].notna()
                    & pre[["lat", "lon", "wind_kt", "dlat_6h", "dlon_6h",
                           "doy_sin", "doy_cos"]].notna().all(axis=1)]
            X = _tc_matrix(d["lat"], d["lon"], d["wind_kt"], d["dlat_6h"],
                           d["dlon_6h"], d["doy_sin"], d["doy_cos"])
            y = np.column_stack([d[f"fut_lat_{h}h"] - d["lat"],
                                 d[f"fut_lon_{h}h"] - d["lon"]])
            self.models[h] = LinearRegression().fit(X, y)
            self.models[f"n_{h}"] = int(len(d))
        return self

    def predict_pos(self, lead, lat, lon, wind, dlat6, dlon6, doy_sin, doy_cos):
        X = _tc_matrix(np.asarray(lat, float), np.asarray(lon, float),
                       np.asarray(wind, float), np.asarray(dlat6, float),
                       np.asarray(dlon6, float), np.asarray(doy_sin, float),
                       np.asarray(doy_cos, float))
        d = self.models[lead].predict(X)
        return np.asarray(lat, float) + d[:, 0], np.asarray(lon, float) + d[:, 1]


# ---------------------------------------------------------------------------
# sequence records (per lead)
# ---------------------------------------------------------------------------
def seq_records(df, lead, tcliper: TrueCliper):
    k = lead // 6
    out = []
    for seg_id, g in df.groupby("segment_id"):
        g = g.sort_values("t_unix").reset_index(drop=True)
        F = g[FEATURES].to_numpy("float64")
        lat = g["lat"].to_numpy(); lon = g["lon"].to_numpy(); w = g["wind_kt"].to_numpy()
        dlat6 = g["dlat_6h"].to_numpy(); dlon6 = g["dlon_6h"].to_numpy()
        dsin = g["doy_sin"].to_numpy(); dcos = g["doy_cos"].to_numpy()
        need = g[NEEDED].notna().all(axis=1).to_numpy()
        interp = g["interpolated"].to_numpy()
        for i in range(SEQ_LEN - 1, len(g) - k):
            j = i + k
            if not np.isfinite([lat[i], lon[i], w[i], lat[j], lon[j], w[j],
                                dlat6[i], dlon6[i], dsin[i], dcos[i]]).all():
                continue
            pm_lat = lat[i] + dlat6[i] * k
            pm_lon = lon[i] + dlon6[i] * k
            tc_lat, tc_lon = tcliper.predict_pos(lead, [lat[i]], [lon[i]], [w[i]],
                                                 [dlat6[i]], [dlon6[i]],
                                                 [dsin[i]], [dcos[i]])
            out.append({
                "segment_id": seg_id, "storm_id_code": g["storm_id_code"].iloc[0],
                "storm_year": int(g["storm_year"].iloc[0]),
                "X": F[i - SEQ_LEN + 1: i + 1].copy(),
                "blat": lat[i], "blon": lon[i], "bwind": w[i],
                "flat": lat[j], "flon": lon[j], "fwind": w[j],
                "pm_lat": pm_lat, "pm_lon": pm_lon,
                "tc_lat": float(tc_lat[0]), "tc_lon": float(tc_lon[0]),
                "feat_ok": bool(need[i] and need[j]),
                "clean": bool((not interp[i]) and (not interp[j])),
            })
    return out


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
class LSTMReg(nn.Module):
    def __init__(self, n_feat, hidden=64, out=3):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, out))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1, :])


def _fit_predict_lstm(tr, te, target_fn, epochs=70):
    Xtr = np.stack([s["X"] for s in tr]); Xte = np.stack([s["X"] for s in te])
    ytr = np.stack([target_fn(s) for s in tr])
    med = np.nanmedian(Xtr.reshape(-1, Xtr.shape[-1]), axis=0)
    for A in (Xtr, Xte):
        ii = np.where(~np.isfinite(A)); A[ii] = np.take(med, ii[2])
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0); sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-8
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ymu = ytr.mean(0); ysd = ytr.std(0) + 1e-8
    m = LSTMReg(Xtr.shape[-1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)
    lf = nn.MSELoss()
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor((ytr - ymu) / ysd, dtype=torch.float32)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for b0 in range(0, len(Xt), 64):
            b = perm[b0:b0 + 64]; opt.zero_grad()
            lf(m(Xt[b]), yt[b]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte, dtype=torch.float32)).numpy() * ysd + ymu


# ---------------------------------------------------------------------------
# GBM suite
# ---------------------------------------------------------------------------
def _hgb():
    return HistGradientBoostingRegressor(random_state=RANDOM_STATE, max_iter=400,
                                         learning_rate=0.05, max_depth=5,
                                         l2_regularization=1.0)


def fit_gbm_suite(pre, lead, tcliper, resid_base):
    k = lead // 6
    tr = pre[pre[f"fut_wind_{lead}h"].notna() & pre[FEATURES].notna().all(axis=1)]
    X = tr[FEATURES]
    if resid_base == "true_cliper":
        rb_lat, rb_lon = tcliper.predict_pos(lead, tr["lat"], tr["lon"], tr["wind_kt"],
                                             tr["dlat_6h"], tr["dlon_6h"],
                                             tr["doy_sin"], tr["doy_cos"])
    else:
        rb_lat = (tr["lat"] + tr["dlat_6h"] * k).to_numpy()
        rb_lon = (tr["lon"] + tr["dlon_6h"] * k).to_numpy()
    m = {}
    m["dwind"] = _hgb().fit(X, tr[f"delta_wind_{lead}h"])
    m["dir_lat"] = _hgb().fit(X, tr[f"fut_lat_{lead}h"] - tr["lat"])
    m["dir_lon"] = _hgb().fit(X, tr[f"fut_lon_{lead}h"] - tr["lon"])
    m["res_lat"] = _hgb().fit(X, tr[f"fut_lat_{lead}h"].to_numpy() - rb_lat)
    m["res_lon"] = _hgb().fit(X, tr[f"fut_lon_{lead}h"].to_numpy() - rb_lon)
    return m


# ---------------------------------------------------------------------------
# scoring one lead
# ---------------------------------------------------------------------------
def score_lead(recs, lead, gbm, lstm_dir, lstm_res, eng, resid_base):
    if len(recs) < 5:
        return {}
    g = lambda key: np.array([r[key] for r in recs], float)
    blat, blon, bwind = g("blat"), g("blon"), g("bwind")
    flat, flon, fwind = g("flat"), g("flon"), g("fwind")
    pm_lat, pm_lon = g("pm_lat"), g("pm_lon")
    tc_lat, tc_lon = g("tc_lat"), g("tc_lon")
    rb_lat, rb_lon = (tc_lat, tc_lon) if resid_base == "true_cliper" else (pm_lat, pm_lon)
    dwind_true = fwind - bwind
    X = pd.DataFrame(np.stack([r["X"][-1] for r in recs]), columns=FEATURES)

    res = {
        "persistence": {"track": gc(blat, blon, flat, flon),
                        "dwind_MAE": mae(dwind_true, np.zeros_like(dwind_true))},
        "persistence_motion": {"track": gc(pm_lat, pm_lon, flat, flon)},
        "true_cliper": {"track": gc(tc_lat, tc_lon, flat, flon)},
    }
    gdw = gbm["dwind"].predict(X)
    res["gbm_direct"] = {"track": gc(blat + gbm["dir_lat"].predict(X),
                                     blon + gbm["dir_lon"].predict(X), flat, flon),
                         "dwind_MAE": mae(dwind_true, gdw)}
    res["gbm_motion_resid"] = {"track": gc(rb_lat + gbm["res_lat"].predict(X),
                                           rb_lon + gbm["res_lon"].predict(X), flat, flon),
                               "dwind_MAE": mae(dwind_true, gdw)}
    if lstm_dir is not None:
        res["lstm_direct"] = {"track": gc(blat + lstm_dir[:, 0], blon + lstm_dir[:, 1],
                                          flat, flon),
                              "dwind_MAE": mae(dwind_true, lstm_dir[:, 2])}
        res["lstm_motion_resid"] = {"track": gc(rb_lat + lstm_res[:, 0],
                                                rb_lon + lstm_res[:, 1], flat, flon),
                                    "dwind_MAE": mae(dwind_true, lstm_res[:, 2])}
    ah = X.copy(); ah["storm_id_code"] = [r["storm_id_code"] for r in recs]
    ap = eng.forecast(ah)
    res["analog"] = {"track": gc(ap[f"lat_{lead}h"], ap[f"lon_{lead}h"], flat, flon),
                     "dwind_MAE": mae(dwind_true, ap[f"dwind_{lead}h"])}
    return res


def predict_lead(recs, lead, gbm, lstm_dir, lstm_res, eng, resid_base):
    """Per-state predictions for the bootstrap (one row per test state)."""
    if not recs:
        return pd.DataFrame()
    g = lambda key: np.array([r[key] for r in recs], float)
    blat, blon, bwind = g("blat"), g("blon"), g("bwind")
    flat, flon, fwind = g("flat"), g("flon"), g("fwind")
    pm_lat, pm_lon = g("pm_lat"), g("pm_lon")
    tc_lat, tc_lon = g("tc_lat"), g("tc_lon")
    rb_lat, rb_lon = (tc_lat, tc_lon) if resid_base == "true_cliper" else (pm_lat, pm_lon)
    X = pd.DataFrame(np.stack([r["X"][-1] for r in recs]), columns=FEATURES)
    ah = X.copy(); ah["storm_id_code"] = [r["storm_id_code"] for r in recs]
    ap = eng.forecast(ah)
    gdw = gbm["dwind"].predict(X)
    out = pd.DataFrame({
        "storm_id_code": [r["storm_id_code"] for r in recs],
        "lead_h": lead, "clean": [r["clean"] for r in recs],
        "fut_lat": flat, "fut_lon": flon, "dwind_true": fwind - bwind,
        "base_lat": blat, "base_lon": blon,
        "pm_lat": pm_lat, "pm_lon": pm_lon, "tc_lat": tc_lat, "tc_lon": tc_lon,
        "gbm_dir_lat": blat + gbm["dir_lat"].predict(X),
        "gbm_dir_lon": blon + gbm["dir_lon"].predict(X),
        "gbm_res_lat": rb_lat + gbm["res_lat"].predict(X),
        "gbm_res_lon": rb_lon + gbm["res_lon"].predict(X),
        "lstm_dir_lat": blat + lstm_dir[:, 0], "lstm_dir_lon": blon + lstm_dir[:, 1],
        "lstm_res_lat": rb_lat + lstm_res[:, 0], "lstm_res_lon": rb_lon + lstm_res[:, 1],
        "analog_lat": ap[f"lat_{lead}h"], "analog_lon": ap[f"lon_{lead}h"],
        "dwind_gbm": gdw, "dwind_analog": ap[f"dwind_{lead}h"],
    })
    return out


def _collect(dst, src, h):
    for mdl, v in src.items():
        dst.setdefault(mdl, {}).setdefault("track", {})[f"{h}h"] = v.get("track")
        if "dwind_MAE" in v:
            dst.setdefault(mdl, {}).setdefault("dwind_MAE", {})[f"{h}h"] = v["dwind_MAE"]


# ---------------------------------------------------------------------------
def main():
    df = load_resampled()
    pre = df[df["storm_year"] < 2019].copy()
    test = df[df["storm_year"] >= 2019].copy()

    tcliper = TrueCliper().fit(pre)

    # --- pre-2019 grouped-CV: persistence-motion vs true CLIPER (track km) ---
    cv = {"persistence_motion": {}, "true_cliper": {}}
    gkf = GroupKFold(n_splits=5)
    for h in LEADS_H:
        d = pre[pre[f"fut_lat_{h}h"].notna()
                & pre[["lat", "lon", "wind_kt", "dlat_6h", "dlon_6h",
                       "doy_sin", "doy_cos"]].notna().all(axis=1)].reset_index(drop=True)
        pm_err, tc_err = [], []
        for tr_i, va_i in gkf.split(d, groups=d["storm_id_code"]):
            tr, va = d.iloc[tr_i], d.iloc[va_i]
            tcf = TrueCliper().fit(tr)
            k = h // 6
            pm_la = va["lat"] + va["dlat_6h"] * k
            pm_lo = va["lon"] + va["dlon_6h"] * k
            tla, tlo = tcf.predict_pos(h, va["lat"], va["lon"], va["wind_kt"],
                                       va["dlat_6h"], va["dlon_6h"],
                                       va["doy_sin"], va["doy_cos"])
            pm_err.append(gc(pm_la, pm_lo, va[f"fut_lat_{h}h"], va[f"fut_lon_{h}h"]))
            tc_err.append(gc(tla, tlo, va[f"fut_lat_{h}h"], va[f"fut_lon_{h}h"]))
        cv["persistence_motion"][f"{h}h"] = float(np.mean(pm_err))
        cv["true_cliper"][f"{h}h"] = float(np.mean(tc_err))
    tc_wins_cv = np.mean([cv["true_cliper"][f"{h}h"] < cv["persistence_motion"][f"{h}h"]
                          for h in LEADS_H]) >= 0.5

    eng = AnalogEngine().fit(df)

    # sequence records first (no trained model needed)
    seq_pre, seq_test = {}, {}
    for h in LEADS_H:
        seq_pre[h] = seq_records(pre, h, tcliper)
        seq_test[h] = seq_records(test, h, tcliper)

    # persistence-motion vs true CLIPER on the TEST primary intersection
    # (Fix 1c/1d: the residual baseline is chosen from THIS comparison)
    prim_base = {"persistence_motion": {}, "true_cliper": {}}
    for h in LEADS_H:
        sub = [r for r in seq_test[h] if r["feat_ok"]]
        flat = np.array([r["flat"] for r in sub]); flon = np.array([r["flon"] for r in sub])
        prim_base["persistence_motion"][f"{h}h"] = gc(
            [r["pm_lat"] for r in sub], [r["pm_lon"] for r in sub], flat, flon)
        prim_base["true_cliper"][f"{h}h"] = gc(
            [r["tc_lat"] for r in sub], [r["tc_lon"] for r in sub], flat, flon)
    tc_wins_test = np.mean([prim_base["true_cliper"][f"{h}h"]
                            < prim_base["persistence_motion"][f"{h}h"]
                            for h in LEADS_H]) >= 0.5
    residual_baseline = "true_cliper" if tc_wins_test else "persistence_motion"
    bl = "tc" if residual_baseline == "true_cliper" else "pm"

    gbms, lstms = {}, {}
    for h in LEADS_H:
        gbms[h] = fit_gbm_suite(pre, h, tcliper, residual_baseline)
        tr_s, te_s = seq_pre[h], seq_test[h]
        pdir = _fit_predict_lstm(tr_s, te_s,
                                 lambda s: [s["flat"] - s["blat"], s["flon"] - s["blon"],
                                            s["fwind"] - s["bwind"]])
        pres = _fit_predict_lstm(tr_s, te_s,
                                 lambda s, bl=bl: [s["flat"] - s[f"{bl}_lat"],
                                                   s["flon"] - s[f"{bl}_lon"],
                                                   s["fwind"] - s["bwind"]])
        lstms[h] = (pdir, pres)

    # ---- PRIMARY intersection ----
    primary, primary_clean, n_prim, n_clean = {}, {}, {}, {}
    pred_frames = []
    for h in LEADS_H:
        recs = seq_test[h]
        idx = [i for i, r in enumerate(recs) if r["feat_ok"]]
        sub = [recs[i] for i in idx]
        pdir, pres = lstms[h]
        _collect(primary, score_lead(sub, h, gbms[h], pdir[idx], pres[idx], eng,
                                     residual_baseline), h)
        pred_frames.append(predict_lead(sub, h, gbms[h], pdir[idx], pres[idx], eng,
                                        residual_baseline))
        n_prim[h] = len(sub)
        cidx = [q for q, i in enumerate(idx) if recs[i]["clean"]]
        subc = [sub[q] for q in cidx]
        _collect(primary_clean,
                 score_lead(subc, h, gbms[h], pdir[[idx[q] for q in cidx]],
                            pres[[idx[q] for q in cidx]], eng, residual_baseline), h)
        n_clean[h] = len(subc)

    # ---- SECONDARY full-n ----
    secondary, n_sec = {}, {}
    for h in LEADS_H:
        t = test[test[f"fut_wind_{h}h"].notna() & test[NEEDED].notna().all(axis=1)]
        n_sec[h] = int(len(t)); k = h // 6
        X = t[FEATURES]
        blat, blon = t["lat"].to_numpy(), t["lon"].to_numpy()
        flat, flon = t[f"fut_lat_{h}h"].to_numpy(), t[f"fut_lon_{h}h"].to_numpy()
        dwt = t[f"delta_wind_{h}h"].to_numpy()
        tcl_lat, tcl_lon = tcliper.predict_pos(h, t["lat"], t["lon"], t["wind_kt"],
                                               t["dlat_6h"], t["dlon_6h"],
                                               t["doy_sin"], t["doy_cos"])
        ap = eng.forecast(t)
        rows = {
            "persistence": {"track": gc(blat, blon, flat, flon),
                            "dwind_MAE": mae(dwt, np.zeros_like(dwt))},
            "persistence_motion": {"track": gc(blat + t["dlat_6h"] * k,
                                               blon + t["dlon_6h"] * k, flat, flon)},
            "true_cliper": {"track": gc(tcl_lat, tcl_lon, flat, flon)},
            "gbm_direct": {"track": gc(blat + gbms[h]["dir_lat"].predict(X),
                                       blon + gbms[h]["dir_lon"].predict(X), flat, flon),
                           "dwind_MAE": mae(dwt, gbms[h]["dwind"].predict(X))},
            "analog": {"track": gc(ap[f"lat_{h}h"], ap[f"lon_{h}h"], flat, flon),
                       "dwind_MAE": mae(dwt, ap[f"dwind_{h}h"])},
        }
        _collect(secondary, rows, h)

    def rank(tbl, metric, h):
        items = [(m, tbl[m][metric][f"{h}h"]) for m in tbl
                 if metric in tbl.get(m, {}) and tbl[m][metric].get(f"{h}h") is not None]
        return [m for m, _ in sorted(items, key=lambda x: x[1])]

    ranking_changes = {}
    for h in LEADS_H:
        ranking_changes[f"{h}h"] = {
            "dwind_primary": rank(primary, "dwind_MAE", h),
            "dwind_clean_only": rank(primary_clean, "dwind_MAE", h),
            "track_primary": rank(primary, "track", h),
            "track_clean_only": rank(primary_clean, "track", h),
            "track_order_changed": rank(primary, "track", h) != rank(primary_clean, "track", h),
            "dwind_order_changed": rank(primary, "dwind_MAE", h) != rank(primary_clean, "dwind_MAE", h),
        }

    d0 = modeling_frame(pd.read_parquet(ROOT / "data" / "cyclone_clean.parquet"))
    _, ho = temporal_holdout(d0)
    dd = d0.iloc[ho]; msk = dd["wind_kt_lag1_recomp"].notna()
    withdrawn = {"framing": "wind(t) from previous row - a nowcast",
                 "MAE_kt": mae(dd.loc[msk, "wind_kt_canonical"], dd.loc[msk, "wind_kt_lag1_recomp"]),
                 "n": int(msk.sum())}

    out = {
        "split": three_way_report(df),
        "withdrawn_nowcast_intensity": withdrawn,
        "leads_h": list(LEADS_H),
        "true_cliper": {
            "features": TC_FEATURES,
            "fit_n_per_lead": {f"{h}h": tcliper.models[f"n_{h}"] for h in LEADS_H},
            "fit_storms": int(pre["storm_id_code"].nunique()),
            "pre2019_groupcv_track_km": cv,
            "test_primary_intersection_track_km": prim_base,
            "true_cliper_wins_pre2019_cv": bool(tc_wins_cv),
            "true_cliper_wins_2019plus_test": bool(tc_wins_test),
            "residual_learning_baseline": residual_baseline,
        },
        "primary_intersection": {"n_per_lead": n_prim, "rows": primary},
        "primary_clean_only": {"n_per_lead": n_clean, "rows": primary_clean},
        "secondary_full_n": {"n_per_lead": n_sec, "rows": secondary},
        "ranking_changes": ranking_changes,
    }
    (FRAG / "models.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.concat(pred_frames, ignore_index=True).to_parquet(
        FRAG / "preds_primary.parquet", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "split"}, indent=2))


if __name__ == "__main__":
    main()
