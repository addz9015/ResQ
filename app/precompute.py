"""
Build everything the dashboard serves offline:
  * app/static/data/engine.pkl   - the trained ForecastEngine
  * app/static/data/replay.json  - step-by-step hindcasts for every 2019+ storm
                                   (the forecast that WOULD have been issued at
                                   each 6 h step, plus what actually happened)

Run once:  python app/precompute.py
"""
from __future__ import annotations

import json
import pathlib
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.segment_resample import load_resampled           # noqa: E402
from app.backend.engine import ForecastEngine, GBM_FEATURES  # noqa: E402

STATIC = pathlib.Path(__file__).resolve().parent / "static" / "data"
STATIC.mkdir(parents=True, exist_ok=True)


def main():
    print("training ForecastEngine ...")
    eng = ForecastEngine().fit()
    with open(STATIC / "engine.pkl", "wb") as f:
        pickle.dump(eng, f)
    print(f"  -> {STATIC / 'engine.pkl'}")

    df = load_resampled()
    test = df[df["storm_year"] >= 2019].copy()

    storms, replay = [], {}
    for sid, g in test.groupby("imd_storm_id"):
        g = g.sort_values("t_unix").reset_index(drop=True)
        name = str(g["storm_name"].iloc[0])
        year = int(g["storm_year"].iloc[0])
        track = [{"t": int(r.t_unix), "lat": _f(r.lat), "lon": _f(r.lon),
                  "wind": _f(r.wind_kt), "interp": bool(r.interpolated),
                  "seg": str(r.segment_id)}
                 for r in g.itertuples()]
        steps = []
        for i, r in enumerate(g.itertuples()):
            state = {k: getattr(r, k, np.nan) for k in
                     set(GBM_FEATURES) | {"doy_sin", "doy_cos", "dlat_6h", "dlon_6h"}}
            state["storm_id_code"] = float(r.storm_id_code)
            if any(state.get(k) is None or (isinstance(state.get(k), float) and np.isnan(state[k]))
                   for k in GBM_FEATURES):
                continue
            fc = eng.forecast(state)
            actual = {}
            for h in (6, 12, 24):
                k = h // 6
                if i + k < len(g):
                    rr = g.iloc[i + k]
                    actual[f"{h}h"] = {"lat": _f(rr.lat), "lon": _f(rr.lon),
                                       "wind": _f(rr.wind_kt),
                                       "interp": bool(rr.interpolated)}
                else:
                    actual[f"{h}h"] = None
            steps.append({"i": i, "t": int(r.t_unix), "forecast": fc, "actual": actual})
        if not steps:
            continue
        replay[sid] = {"name": name, "year": year, "track": track, "steps": steps}
        storms.append({"id": sid, "name": name, "year": year,
                       "n_points": len(track), "n_forecasts": len(steps)})

    storms.sort(key=lambda s: (s["year"], s["id"]))
    out = {"generated_utc": pd.Timestamp.utcnow().isoformat(),
           "n_storms": len(storms), "storms": storms, "replay": replay}
    (STATIC / "replay.json").write_text(json.dumps(out, default=_json_default),
                                        encoding="utf-8")
    sz = (STATIC / "replay.json").stat().st_size / 1e6
    print(f"  -> {STATIC / 'replay.json'}  ({sz:.1f} MB, {len(storms)} storms, "
          f"{sum(s['n_forecasts'] for s in storms)} hindcast steps)")


def _f(v):
    try:
        v = float(v)
        return None if np.isnan(v) else round(v, 4)
    except (TypeError, ValueError):
        return None


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


if __name__ == "__main__":
    main()
