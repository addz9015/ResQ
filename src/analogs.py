"""
NEW COMPONENT A - analog retrieval engine.

For any cyclone state (a row of the resampled 6 h grid) find the most similar
past states from *other* storms and use what those analogs did next as a
forecast.

  * feature vector: lat, lon, wind, pressure, pressure_drop, 6 h wind tendency,
    6 h motion vector (dlat, dlon), storm_age, month_sin/cos, basin one-hots
  * StandardScaler fit on TRAINING storms only (first obs year < 2019)
  * kNN over training-storm states (KD-tree euclidean by default, cosine
    optional); analogs from the query's own storm_id_code are always excluded
  * retrieve(): top-k analogs with storm id / name / year, their +6/12/24 h
    delta-wind and displacement, and their storm's eventual peak grade
  * score_holdout(): analog-ensemble mean as a forecast vs persistence and
    CLIPER on the 2019+ holdout - same metrics as everything else

If the analog ensemble does not beat the baselines the report says so plainly.
"""
from __future__ import annotations

import json
import sys
import pathlib

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from splits import temporal_holdout_resampled  # noqa: E402

from .common import (FRAG, HOLDOUT_START_YEAR, LEADS_H, RANDOM_STATE,
                     haversine_km)
from .segment_resample import load_resampled

FEATURES = ["lat", "lon", "wind_kt", "pressure_hpa", "pressure_drop",
            "wind_tendency_6h", "dlat_6h", "dlon_6h", "storm_age_hours",
            "month_sin", "month_cos", "basin_ARB", "basin_BOB", "basin_LAND"]

OUTCOME = ([f"delta_wind_{h}h" for h in LEADS_H]
           + [f"fut_lat_{h}h" for h in LEADS_H]
           + [f"fut_lon_{h}h" for h in LEADS_H])


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


class AnalogEngine:
    def __init__(self, metric: str = "euclidean", k: int = 10, pool: int = 80):
        self.metric = metric
        self.k = k
        self.pool = pool

    def fit(self, df: pd.DataFrame):
        feat_ok = df[FEATURES].notna().all(axis=1)
        # a library state must have finite features and at least a +6 h future,
        # otherwise it is a dead analog that contributes nothing
        outcome_ok = df["delta_wind_6h"].notna() & df["fut_lat_6h"].notna()
        train = df[feat_ok & outcome_ok
                   & (df["storm_year"] < HOLDOUT_START_YEAR)].copy()
        self.lib_ = train.reset_index(drop=True)
        self.scaler_ = StandardScaler().fit(self.lib_[FEATURES].to_numpy())
        algo = "kd_tree" if self.metric == "euclidean" else "brute"
        self.nn_ = NearestNeighbors(n_neighbors=min(self.pool, len(self.lib_)),
                                    metric=self.metric, algorithm=algo)
        self.nn_.fit(self.scaler_.transform(self.lib_[FEATURES].to_numpy()))
        return self

    def _neighbours(self, X: np.ndarray, query_storm_ids: np.ndarray):
        X = np.asarray(X, dtype="float64")
        key = (X.tobytes(), np.asarray(query_storm_ids).tobytes())
        cached = getattr(self, "_nb_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        # standardise, then mean-impute any missing query feature (0 == train mean)
        Xs = np.nan_to_num(self.scaler_.transform(X), nan=0.0)
        _, idx = self.nn_.kneighbors(Xs)
        lib_sid = self.lib_["storm_id_code"].to_numpy()
        out = []
        for row, qsid in zip(idx, query_storm_ids):
            keep, per_storm = [], {}
            for j in row:
                s = lib_sid[j]
                if s == qsid or per_storm.get(s, 0) >= 3:
                    continue
                per_storm[s] = per_storm.get(s, 0) + 1
                keep.append(j)
                if len(keep) >= self.k:
                    break
            out.append(keep)
        self._nb_cache = (key, out)
        return out

    def retrieve(self, df_states: pd.DataFrame) -> list[pd.DataFrame]:
        nb = self._neighbours(df_states[FEATURES].to_numpy(),
                              df_states["storm_id_code"].to_numpy())
        tables = []
        for (_, q), js in zip(df_states.iterrows(), nb):
            a = self.lib_.iloc[js]
            t = pd.DataFrame({
                "storm_id": a["imd_storm_id"].values,
                "name": a["storm_name"].values,
                "year": a["storm_year"].values,
                "analog_wind_kt": a["wind_kt"].values.round(1),
                "d_wind_6h": a["delta_wind_6h"].values,
                "d_wind_12h": a["delta_wind_12h"].values,
                "d_wind_24h": a["delta_wind_24h"].values,
                "disp_6h_km": haversine_km(a["lat"], a["lon"],
                                           a["fut_lat_6h"], a["fut_lon_6h"]).values,
                "disp_24h_km": haversine_km(a["lat"], a["lon"],
                                            a["fut_lat_24h"], a["fut_lon_24h"]).values,
                "eventual_peak_grade_ord": a["peak_grade_ordinal"].values,
            })
            tables.append(t)
        return tables

    def analog_deltas(self, df_states: pd.DataFrame) -> list[dict]:
        """For each query state, the raw arrays of retrieved-analog outcomes.

        Returns one dict per state: {h: {"dwind": np.array, "dlat": .., "dlon": ..}}
        with only the finite analog values kept (variable length <= k).
        """
        nb = self._neighbours(df_states[FEATURES].to_numpy(),
                              df_states["storm_id_code"].to_numpy())
        rows = []
        for js in nb:
            a = self.lib_.iloc[js]
            d = {}
            for h in LEADS_H:
                dw = a[f"delta_wind_{h}h"].to_numpy(dtype="float64")
                dlat = (a[f"fut_lat_{h}h"] - a["lat"]).to_numpy(dtype="float64")
                dlon = (a[f"fut_lon_{h}h"] - a["lon"]).to_numpy(dtype="float64")
                m = np.isfinite(dw)
                d[h] = {"dwind": dw[m], "dlat": dlat[m], "dlon": dlon[m]}
            rows.append(d)
        return rows

    # ---- COMPONENT C: probabilistic analog output ------------------------
    RI_THRESH = 30.0   # kt / 24 h
    RW_THRESH = -30.0

    def forecast_distribution(self, df_states: pd.DataFrame) -> dict:
        """Empirical predictive distribution from the analog ensemble.

        Per lead: deciles of analog Δwind. Plus P(RI) / P(rapid-weakening) at 24 h
        = fraction of retrieved analogs above / below +-30 kt.
        """
        deltas = self.analog_deltas(df_states)
        n = len(df_states)
        dec = np.arange(10, 100, 10)
        out = {f"dwind_deciles_{h}h": np.full((n, len(dec)), np.nan) for h in LEADS_H}
        out["p_ri_24h"] = np.full(n, np.nan)
        out["p_rw_24h"] = np.full(n, np.nan)
        out["n_analogs_24h"] = np.zeros(n, dtype=int)
        for i, d in enumerate(deltas):
            for h in LEADS_H:
                v = d[h]["dwind"]
                if v.size:
                    out[f"dwind_deciles_{h}h"][i] = np.percentile(v, dec)
            v24 = d[24]["dwind"]
            out["n_analogs_24h"][i] = v24.size
            if v24.size:
                out["p_ri_24h"][i] = float(np.mean(v24 >= self.RI_THRESH))
                out["p_rw_24h"][i] = float(np.mean(v24 <= self.RW_THRESH))
        return out

    def forecast(self, df_states: pd.DataFrame) -> dict:
        nb = self._neighbours(df_states[FEATURES].to_numpy(),
                              df_states["storm_id_code"].to_numpy())
        lat0 = df_states["lat"].to_numpy(); lon0 = df_states["lon"].to_numpy()
        pred = {f"dwind_{h}h": np.full(len(df_states), np.nan) for h in LEADS_H}
        pred.update({f"lat_{h}h": np.full(len(df_states), np.nan) for h in LEADS_H})
        pred.update({f"lon_{h}h": np.full(len(df_states), np.nan) for h in LEADS_H})
        for i, js in enumerate(nb):
            a = self.lib_.iloc[js]
            for h in LEADS_H:
                dw = a[f"delta_wind_{h}h"].to_numpy()
                dlat = (a[f"fut_lat_{h}h"] - a["lat"]).to_numpy()
                dlon = (a[f"fut_lon_{h}h"] - a["lon"]).to_numpy()
                if np.isfinite(dw).any():
                    pred[f"dwind_{h}h"][i] = np.nanmean(dw)
                if np.isfinite(dlat).any():
                    pred[f"lat_{h}h"][i] = lat0[i] + np.nanmean(dlat)
                    pred[f"lon_{h}h"][i] = lon0[i] + np.nanmean(dlon)
        return pred


def _score(df_h: pd.DataFrame, pred: dict, label: str) -> dict:
    res = {}
    for h in LEADS_H:
        m = (df_h[f"delta_wind_{h}h"].notna()
             & df_h[f"fut_lat_{h}h"].notna()
             & np.isfinite(pred[f"dwind_{h}h"])
             & np.isfinite(pred[f"lat_{h}h"]))
        d = df_h[m]
        yw = d[f"delta_wind_{h}h"].to_numpy()
        pw = pred[f"dwind_{h}h"][m.to_numpy()]
        gc = haversine_km(pred[f"lat_{h}h"][m.to_numpy()],
                          pred[f"lon_{h}h"][m.to_numpy()],
                          d[f"fut_lat_{h}h"].to_numpy(), d[f"fut_lon_{h}h"].to_numpy())
        res[f"{h}h"] = {
            "n": int(m.sum()),
            "track_km_mean": float(np.mean(gc)),
            "track_km_median": float(np.median(gc)),
            "dwind_MAE_kt": float(mean_absolute_error(yw, pw)),
            "dwind_RMSE_kt": rmse(yw, pw),
        }
    return {"model": label, **res}


def _persistence_pred(df_h):
    p = {}
    for h in LEADS_H:
        p[f"dwind_{h}h"] = np.zeros(len(df_h))
        p[f"lat_{h}h"] = df_h["lat"].to_numpy()
        p[f"lon_{h}h"] = df_h["lon"].to_numpy()
    return p


def _cliper_pred(df_h):
    p = {}
    for h in LEADS_H:
        steps = h / 6.0
        p[f"dwind_{h}h"] = np.zeros(len(df_h))
        p[f"lat_{h}h"] = (df_h["lat"] + df_h["dlat_6h"] * steps).to_numpy()
        p[f"lon_{h}h"] = (df_h["lon"] + df_h["dlon_6h"] * steps).to_numpy()
    return p


def score_holdout(metric: str = "euclidean") -> dict:
    df = load_resampled()
    eng = AnalogEngine(metric=metric).fit(df)

    _, ho = temporal_holdout_resampled(df)
    hold = df.iloc[ho]
    hold = hold[hold[FEATURES].notna().all(axis=1)
                & hold["dlat_6h"].notna()].reset_index(drop=True)

    analog_pred = eng.forecast(hold)
    rows = [
        _score(hold, _persistence_pred(hold), "persistence (dwind=0, no motion)"),
        _score(hold, _cliper_pred(hold), "CLIPER (extrapolate 6h motion)"),
        _score(hold, analog_pred, f"analog-ensemble mean (k=10, {metric})"),
    ]

    # verdict
    a = rows[2]; p = rows[0]; c = rows[1]
    beats = {
        f"{h}h": {
            "track_vs_CLIPER": a[f"{h}h"]["track_km_mean"] < c[f"{h}h"]["track_km_mean"],
            "track_vs_persistence": a[f"{h}h"]["track_km_mean"] < p[f"{h}h"]["track_km_mean"],
            "dwind_vs_persistence": a[f"{h}h"]["dwind_MAE_kt"] < p[f"{h}h"]["dwind_MAE_kt"],
        } for h in LEADS_H
    }
    out = {"library_states": int(len(eng.lib_)),
           "holdout_states_scored": int(len(hold)),
           "table": rows, "analog_beats_baseline": beats}
    (FRAG / "analogs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    _worked_example(eng, hold, name_contains="MANDOUS")
    print(json.dumps(out, indent=2))
    return out


def _worked_example(eng, hold, name_contains="MANDOUS"):
    """Regenerate the worked example as a DISTRIBUTION + P(RI), not a mean."""
    cand = hold[hold["storm_name"].str.contains(name_contains, na=False)
                & hold["delta_wind_24h"].notna()]
    if len(cand) == 0:
        cand = hold[hold["delta_wind_24h"].notna()]
    # the state with the strongest actual 24 h intensification - the case the
    # ensemble mean tends to under-predict
    q = cand.loc[cand["delta_wind_24h"].idxmax()]
    qi = hold.index.get_loc(q.name)
    tbl = eng.retrieve(hold.iloc[[qi]])[0]
    dist = eng.forecast_distribution(hold.iloc[[qi]])
    dec = np.arange(10, 100, 10)
    ex = {
        "query": {"storm": q["imd_storm_id"], "name": q["storm_name"],
                  "year": int(q["storm_year"]), "lat": round(float(q["lat"]), 2),
                  "lon": round(float(q["lon"]), 2),
                  "wind_kt": round(float(q["wind_kt"]), 1),
                  "actual_d_wind_24h": (None if pd.isna(q["delta_wind_24h"])
                                        else float(q["delta_wind_24h"]))},
        "analog_dwind_deciles": {
            f"{h}h": [None if not np.isfinite(v) else round(float(v), 1)
                      for v in dist[f"dwind_deciles_{h}h"][0]]
            for h in LEADS_H},
        "decile_pct": dec.tolist(),
        "p_ri_24h": (None if not np.isfinite(dist["p_ri_24h"][0])
                     else round(float(dist["p_ri_24h"][0]), 3)),
        "p_rapid_weakening_24h": (None if not np.isfinite(dist["p_rw_24h"][0])
                                  else round(float(dist["p_rw_24h"][0]), 3)),
        "n_analogs_with_24h_outcome": int(dist["n_analogs_24h"][0]),
        "analogs": tbl.round(2).to_dict(orient="records"),
    }
    (FRAG / "analogs_example.json").write_text(json.dumps(ex, indent=2), encoding="utf-8")


def score_ri(metric: str = "euclidean") -> dict:
    """Score the analog P(RI) forecast on the 2019+ test storms against a
    climatological base-rate forecast. RI = Δwind over 24 h >= 30 kt."""
    from sklearn.metrics import brier_score_loss

    df = load_resampled()
    eng = AnalogEngine(metric=metric).fit(df)          # library = pre-2019 storms
    lib = eng.lib_
    base_rate = float((lib["delta_wind_24h"] >= AnalogEngine.RI_THRESH).mean())

    _, ho = temporal_holdout_resampled(df)
    test = df.iloc[ho]
    test = test[test[FEATURES].notna().all(axis=1)
                & test["delta_wind_24h"].notna()].reset_index(drop=True)
    y = (test["delta_wind_24h"].to_numpy() >= AnalogEngine.RI_THRESH).astype(int)

    dist = eng.forecast_distribution(test)
    p = dist["p_ri_24h"]
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]

    # reliability, 5 bins
    edges = np.linspace(0, 1, 6)
    binid = np.clip(np.digitize(p, edges[1:-1]), 0, 4)
    rel = []
    for b in range(5):
        m = binid == b
        if m.any():
            rel.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                        "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                        "empirical": float(y[m].mean())})

    out = {
        "definition": "RI = wind(t+24h) - wind(t) >= 30 kt on the 6 h grid",
        "split": "library = pre-2019 storms; scored on 2019+ test storms",
        "n_test": int(len(y)),
        "test_base_rate": float(y.mean()),
        "train_climatology_base_rate": base_rate,
        "analog_p_ri": {
            "brier": float(brier_score_loss(y, p)),
            "reliability_5bin": rel,
        },
        "climatology_forecast": {
            "brier": float(brier_score_loss(y, np.full_like(p, base_rate))),
        },
        "analog_beats_climatology": bool(
            brier_score_loss(y, p) < brier_score_loss(y, np.full_like(p, base_rate))),
    }
    (FRAG / "analogs_ri.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    score_holdout()
    score_ri()
