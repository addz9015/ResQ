"""
Offline forecast engine for the ResQ dashboard.

Wraps the phase-4 models into one object:
  * track    : true CLIPER (OLS) + GBM-learned residual
  * intensity: GBM Δwind regressor, and the analog-ensemble mean
  * cone     : split-CQR 80% along/cross-track half-widths per lead
  * P(RI)    : mean of the isotonic-calibrated analog and classifier probabilities
  * analogs  : the 10 nearest past states with what they did next

Trained once by `app/precompute.py`, pickled to `app/static/data/engine.pkl`,
and reused by the FastAPI backend. No network, no live CDS calls.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.common import haversine_km, deg_to_km, LEADS_H          # noqa: E402
from src.segment_resample import load_resampled                   # noqa: E402
from src.analogs import AnalogEngine, FEATURES as ANALOG_FEATURES  # noqa: E402
from run_models import TrueCliper, fit_gbm_suite, FEATURES as GBM_FEATURES  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

RI_THRESH = 30.0


class ForecastEngine:
    def __init__(self):
        self.leads = list(LEADS_H)

    # -- training -----------------------------------------------------------
    def fit(self):
        df = load_resampled()
        pre = df[df["storm_year"] < 2019].copy()

        self.analog = AnalogEngine().fit(df)
        self.tcliper = TrueCliper().fit(pre)
        # true CLIPER is the best track model on the held-out test; the residual
        # GBM does not beat it, so the dashboard shows true CLIPER directly for
        # track and the GBM only for Δwind.
        self.gbm = {h: fit_gbm_suite(pre, h, self.tcliper, "true_cliper")
                    for h in LEADS_H}

        # along/cross-track 80 % split-CQR half-widths (cone)
        self._fit_cone(df)

        # Δwind 80 % predictive interval per lead (quantile GBM + conformal offset)
        self._fit_dwind_pi(df)

        # RI: isotonic-calibrated analog + classifier, then average
        self._fit_ri(df, pre)
        return self

    def _fit_dwind_pi(self, df):
        """q0.1 / q0.9 quantile GBM for Δwind per lead, widened by the
        cross-conformal offset already computed in reports (uncertainty.json).
        Also carries the empirical coverage note for the 24 h interval."""
        import json as _json
        base = df[df[GBM_FEATURES].notna().all(axis=1)]
        pre = base[base["storm_year"] < 2019]
        unc_p = pathlib.Path(__file__).resolve().parents[2] / "reports" / "_fragments" / "uncertainty.json"
        unc = _json.loads(unc_p.read_text()) if unc_p.exists() else {}
        self.dwind_pi = {}
        for h in LEADS_H:
            tr = pre[pre[f"delta_wind_{h}h"].notna()]
            X, y = tr[GBM_FEATURES], tr[f"delta_wind_{h}h"]
            lo = _qfit(0.1).fit(X, y)
            hi = _qfit(0.9).fit(X, y)
            e = unc.get("per_target", {}).get(f"delta_wind_{h}h", {}).get("nominal_80", {})
            off = e.get("cross_conformal_offset", e.get("split_cqr_offset", 0.0))
            cov = e.get("cross_conformal_coverage", e.get("split_cqr_coverage"))
            self.dwind_pi[h] = {"lo": lo, "hi": hi, "offset": float(off),
                                "empirical_coverage_80": cov}

    def _fit_cone(self, df):
        """80 % split-CQR half-width (km) for along- and cross-track error at each
        lead - the footprint radius of the uncertainty cone."""
        base = df[df[GBM_FEATURES].notna().all(axis=1)]
        tr = base[base["storm_year"] <= 2016]
        ca = base[(base["storm_year"] >= 2017) & (base["storm_year"] <= 2018)]
        self.cone = {}
        for h in LEADS_H:
            comp = _components(base, h)
            trh, cah = tr.join(comp), ca.join(comp)
            hw = {}
            for name in ("along", "cross"):
                col = f"{name}_{h}h"
                t, c = trh[trh[col].notna()], cah[cah[col].notna()]
                lo = _qfit(0.1).fit(t[GBM_FEATURES], t[col])
                hi = _qfit(0.9).fit(t[GBM_FEATURES], t[col])
                clo, chi = lo.predict(c[GBM_FEATURES]), hi.predict(c[GBM_FEATURES])
                e = np.maximum(clo - c[col].to_numpy(), c[col].to_numpy() - chi)
                q = float(np.sort(e)[int(np.ceil((len(e) + 1) * 0.8)) - 1]) if len(e) else 0.0
                width = np.median((chi + q) - (clo - q))
                hw[name] = float(max(width / 2, 5.0))
            self.cone[h] = hw

    def _fit_ri(self, df, pre):
        d = pre[pre[GBM_FEATURES].notna().all(axis=1)
                & pre[ANALOG_FEATURES].notna().all(axis=1)
                & pre["delta_wind_24h"].notna()].reset_index(drop=True)
        y = (d["delta_wind_24h"].to_numpy() >= RI_THRESH).astype(int)
        a_raw = self.analog.forecast_distribution(d)["p_ri_24h"]
        a_raw = np.nan_to_num(a_raw, nan=float(np.nanmean(a_raw)))
        X = d[GBM_FEATURES].to_numpy()
        gkf = GroupKFold(10); clf_oof = np.zeros(len(d))
        for tri, vai in gkf.split(X, groups=d["storm_id_code"].to_numpy()):
            c = _clf().fit(X[tri], y[tri])
            clf_oof[vai] = c.predict_proba(X[vai])[:, 1]
        self.ri_clf = _clf().fit(X, y)
        self.ri_iso_clf = IsotonicRegression(out_of_bounds="clip").fit(clf_oof, y)
        self.ri_iso_analog = IsotonicRegression(out_of_bounds="clip").fit(a_raw, y)
        self.ri_base_rate = float(y.mean())
        # reliability curve of the average, for the gauge background
        avg_oof = 0.5 * (self.ri_iso_analog.transform(a_raw)
                         + self.ri_iso_clf.transform(clf_oof))
        edges = np.linspace(0, 1, 6)
        b = np.clip(np.digitize(avg_oof, edges[1:-1]), 0, 4)
        self.ri_reliability = [
            {"x": float((edges[k] + edges[k + 1]) / 2),
             "pred": float(avg_oof[b == k].mean()) if (b == k).any() else None,
             "obs": float(y[b == k].mean()) if (b == k).any() else None}
            for k in range(5)]

    # -- inference --------------------------------------------------------
    def forecast(self, state: dict) -> dict:
        """state: dict with the GBM_FEATURES keys + optional storm_id_code."""
        row = {k: state.get(k, np.nan) for k in set(GBM_FEATURES) | set(ANALOG_FEATURES)
               | {"doy_sin", "doy_cos"}}
        X = pd.DataFrame([{k: row.get(k, np.nan) for k in GBM_FEATURES}])
        blat, blon = float(state["lat"]), float(state["lon"])
        bwind = float(state["wind_kt"])
        out = {"leads_h": self.leads, "origin": {"lat": blat, "lon": blon, "wind_kt": bwind},
               "track": [], "intensity": []}

        for h in LEADS_H:
            tc_lat, tc_lon = self.tcliper.predict_pos(
                h, [blat], [blon], [bwind], [state.get("dlat_6h", 0.0)],
                [state.get("dlon_6h", 0.0)], [state["doy_sin"]], [state["doy_cos"]])
            g = self.gbm[h]
            flat, flon = float(tc_lat[0]), float(tc_lon[0])   # true CLIPER track
            hw = self.cone[h]
            out["track"].append({"lead_h": h, "lat": flat, "lon": flon,
                                 "cliper_lat": float(tc_lat[0]), "cliper_lon": float(tc_lon[0]),
                                 "along_hw_km": hw["along"], "cross_hw_km": hw["cross"],
                                 "cone": _cone_polygon(blat, blon, flat, flon,
                                                       hw["along"], hw["cross"])})
            dwind_gbm = float(g["dwind"].predict(X)[0])
            pi = self.dwind_pi[h]
            q10 = float(pi["lo"].predict(X)[0]); q90 = float(pi["hi"].predict(X)[0])
            out["intensity"].append({
                "lead_h": h, "dwind_gbm": dwind_gbm, "wind_gbm": bwind + dwind_gbm,
                "dwind_lo": q10 - pi["offset"], "dwind_hi": q90 + pi["offset"],
                "interval_nominal": 0.80,
                "interval_empirical_coverage": pi["empirical_coverage_80"]})

        # analog forecast + distribution + P(RI)
        ah = X.copy()
        for c in ANALOG_FEATURES:
            if c not in ah:
                ah[c] = row.get(c, np.nan)
        ah["storm_id_code"] = state.get("storm_id_code", -999)
        ap = self.analog.forecast(ah)
        dist = self.analog.forecast_distribution(ah)
        for i, h in enumerate(LEADS_H):
            out["intensity"][i]["dwind_analog"] = _f(ap[f"dwind_{h}h"][0])
            out["intensity"][i]["analog_lat"] = _f(ap[f"lat_{h}h"][0])
            out["intensity"][i]["analog_lon"] = _f(ap[f"lon_{h}h"][0])
            out["intensity"][i]["dwind_deciles"] = [
                _f(v) for v in dist[f"dwind_deciles_{h}h"][0]]

        a_raw = dist["p_ri_24h"][0]
        clf_p = float(self.ri_clf.predict_proba(X.to_numpy())[0, 1])
        p_analog = float(self.ri_iso_analog.transform([0.0 if a_raw != a_raw else a_raw])[0])
        p_clf = float(self.ri_iso_clf.transform([clf_p])[0])
        out["p_ri"] = {"analog": p_analog, "classifier": p_clf,
                       "combined": 0.5 * (p_analog + p_clf),
                       "base_rate": self.ri_base_rate,
                       "reliability": self.ri_reliability}
        import json as _json
        out["analogs"] = _json.loads(
            self.analog.retrieve(ah)[0].round(2).to_json(orient="records"))
        return out


# ---- helpers -----------------------------------------------------------
def _f(v):
    return None if v is None or (isinstance(v, float) and v != v) else float(v)


def _qfit(q):
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, random_state=26070,
                                         max_iter=250, learning_rate=0.05, max_depth=4)


def _clf():
    return HistGradientBoostingClassifier(random_state=26070, max_iter=300,
                                          learning_rate=0.05, max_depth=4,
                                          class_weight="balanced")


def _components(base, h):
    dn, de = deg_to_km(base[f"fut_lat_{h}h"] - base["lat"],
                       base[f"fut_lon_{h}h"] - base["lon"], base["lat"])
    hn, he = deg_to_km(base["dlat_6h"], base["dlon_6h"], base["lat"])
    hmag = np.hypot(hn, he).clip(lower=1e-6)
    un, ue = hn / hmag, he / hmag
    return pd.DataFrame({f"along_{h}h": dn * un + de * ue,
                         f"cross_{h}h": dn * (-ue) + de * un}, index=base.index)


def _cone_polygon(blat, blon, flat, flon, along_hw, cross_hw, npts=9):
    """Ellipse-ish cone footprint around the forecast point, oriented along the
    forecast heading. Returns [[lat,lon], ...]."""
    hn, he = deg_to_km(flat - blat, flon - blon, blat)
    hmag = np.hypot(hn, he) or 1e-6
    un, ue = hn / hmag, he / hmag
    xn, xe = -ue, un
    km_lat = 110.574
    km_lon = 111.320 * np.cos(np.radians(flat))
    pts = []
    for a in np.linspace(0, 2 * np.pi, npts):
        dn = along_hw * np.cos(a) * un + cross_hw * np.sin(a) * xn
        de = along_hw * np.cos(a) * ue + cross_hw * np.sin(a) * xe
        pts.append([round(float(flat + dn / km_lat), 4),
                    round(float(flon + de / km_lon), 4)])
    return pts
