"""
Build app/static/data/storms.json for the TC-Sentinel UI — the real 2019+
holdout storms from data/cyclone_resampled_6h.parquet. No mock data, no
timestamps invented: every field comes from the parquet.

Run:  python scripts/build_storms.py   (also called by app/run.py --rebuild)
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "cyclone_resampled_6h.parquet"
OUT = ROOT / "app" / "static" / "data" / "storms.json"

# IMD North Indian Ocean intensity scale (3-min sustained wind, kt)
GRADES = [(17, "Low pressure area"), (28, "Depression"), (34, "Deep depression"),
          (48, "Cyclonic storm"), (64, "Severe cyclonic storm"),
          (90, "Very severe cyclonic storm"), (120, "Extremely severe cyclonic storm"),
          (999, "Super cyclonic storm")]


def category(wind_kt):
    if wind_kt is None or np.isnan(wind_kt):
        return "—"
    for hi, name in GRADES:
        if wind_kt < hi:
            return name
    return GRADES[-1][1]


def main():
    d = pd.read_parquet(SRC)
    d = d[d["storm_year"] >= 2019].copy()
    d = d.sort_values(["imd_storm_id", "t_unix"])

    storms, total_steps = [], 0
    for sid, g in d.groupby("imd_storm_id"):
        g = g.reset_index(drop=True)
        basin = ("arabian_sea" if g["basin_ARB"].iloc[0]
                 else "bay_of_bengal" if g["basin_BOB"].iloc[0] else "land")
        steps = []
        for r in g.itertuples():
            steps.append({
                "t": int(r.t_unix),
                "date": pd.Timestamp(r.t_unix, unit="s").strftime("%Y-%m-%d %H:%MZ"),
                "lat": round(float(r.lat), 3), "lon": round(float(r.lon), 3),
                "wind": None if pd.isna(r.wind_kt) else round(float(r.wind_kt), 0),
                "pressure": None if pd.isna(r.pressure_hpa) else round(float(r.pressure_hpa), 0),
                "category": category(None if pd.isna(r.wind_kt) else float(r.wind_kt)),
                "interpolated": bool(r.interpolated),
            })
        peak = g["wind_kt"].max()
        low = g["pressure_hpa"].min()
        name = str(g["storm_name"].iloc[0])
        storms.append({
            "id": sid,                          # imd_storm_id — maps to /api/replay/{id}
            "name": name if name and name != "(unnamed)" else f"Unnamed {sid}",
            "season_year": int(g["storm_year"].iloc[0]),
            "basin": basin,
            "peak_wind": None if pd.isna(peak) else round(float(peak), 0),
            "min_pressure": None if pd.isna(low) else round(float(low), 0),
            "peak_category": category(None if pd.isna(peak) else float(peak)),
            "n_steps": len(steps),
            "first_date": steps[0]["date"], "last_date": steps[-1]["date"],
            "steps": steps,
        })
        total_steps += len(steps)

    storms.sort(key=lambda s: (s["season_year"], s["first_date"]))
    out = {
        "source": "data/cyclone_resampled_6h.parquet — IMD best-track, 6 h grid",
        "note": "2019+ holdout storms. Historical replay only — nothing here is live.",
        "n_storms": len(storms),
        "n_forecast_steps_total": total_steps,
        "storms": storms,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB, {len(storms)} storms, "
          f"{total_steps} steps)")


if __name__ == "__main__":
    main()
