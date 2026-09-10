"""
PROBLEM 2 - irregular time axis.

The raw cyclone table is only 60.8% 3 h steps, 26% 6 h, with gaps up to 8748 h.
This module:

  1. splits every storm into contiguous SEGMENTS wherever the observation gap
     exceeds 12 h, assigning `segment_id`;
  2. resamples each segment onto a uniform 6 h grid - lat / lon / pressure /
     pressure_drop by linear interpolation in real (unix) time, wind by linear
     interpolation with an `interpolated` flag;
  3. drops segments shorter than 5 grid points;
  4. recomputes proper per-6h / per-hour rates on the uniform grid;
  5. writes data/cyclone_resampled_6h.parquet and a report fragment.

Nothing here is used to overwrite the raw table - it is a separate artifact.
Interpolated wind values are flagged so downstream code / the reader can see
exactly which target points are synthetic.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import (CLEAN_PARQUET, FRAG, GRID_SECONDS, HOLDOUT_START_YEAR,
                     LEADS_H, MIN_SEGMENT_POINTS, RESAMPLED_PARQUET,
                     SEGMENT_GAP_SECONDS, storm_metadata)

WIND_OBS_TOL_S = 3600.0  # a grid wind value is "interpolated" if the nearest real
                         # wind observation is more than this far away


def _interp(x_grid, x_obs, y_obs):
    m = np.isfinite(y_obs)
    if m.sum() < 2:
        return np.full_like(x_grid, np.nan, dtype="float64")
    return np.interp(x_grid, x_obs[m], y_obs[m], left=np.nan, right=np.nan)


def build() -> pd.DataFrame:
    clean = pd.read_parquet(CLEAN_PARQUET)
    clean = clean[clean["storm_id_code"].notna()
                  & clean["lat"].notna() & clean["lon"].notna()].copy()
    clean = clean.sort_values(["storm_id_code", "dt_snap_unix"]).reset_index(drop=True)

    meta = storm_metadata()
    peak_grade = clean.groupby("storm_id_code")["imd_grade_ordinal"].max()
    storm_first = clean.groupby("storm_id_code")["dt_snap_unix"].min()
    storm_year = pd.to_datetime(storm_first, unit="s").dt.year

    # ---- segmentation --------------------------------------------------------
    gap = clean.groupby("storm_id_code")["dt_snap_unix"].diff()
    new_seg = (gap > SEGMENT_GAP_SECONDS) | gap.isna()
    clean["seg_no"] = new_seg.groupby(clean["storm_id_code"]).cumsum().astype(int)
    clean["segment_id"] = (clean["storm_id_code"].astype(int).astype(str)
                           + "_" + clean["seg_no"].astype(str))

    raw_seg_sizes = clean.groupby("segment_id").size()

    rows = []
    dropped_short = 0
    for seg_id, g in clean.groupby("segment_id"):
        g = g.sort_values("dt_snap_unix")
        t = g["dt_snap_unix"].to_numpy(dtype="float64")
        if len(t) < 2:
            dropped_short += 1
            continue
        grid = np.arange(t[0], t[-1] + 1.0, GRID_SECONDS)
        if len(grid) < MIN_SEGMENT_POINTS:
            dropped_short += 1
            continue

        sid = g["storm_id_code"].iloc[0]
        lat = _interp(grid, t, g["lat"].to_numpy("float64"))
        lon = _interp(grid, t, g["lon"].to_numpy("float64"))
        pres = _interp(grid, t, g["imd_pressure_hpa"].to_numpy("float64"))
        pdrop = _interp(grid, t, g["imd_pressure_drop"].to_numpy("float64"))
        wind = _interp(grid, t, g["wind_kt_canonical"].to_numpy("float64"))

        w_obs_t = t[np.isfinite(g["wind_kt_canonical"].to_numpy("float64"))]
        if len(w_obs_t):
            near = np.min(np.abs(grid[:, None] - w_obs_t[None, :]), axis=1)
        else:
            near = np.full(len(grid), np.inf)
        interpolated = near > WIND_OBS_TOL_S

        basin = g[["basin_ARB", "basin_BOB", "basin_LAND"]].mode().iloc[0]
        gd = pd.to_datetime(grid, unit="s")
        month = gd.month.to_numpy()
        doy = gd.dayofyear.to_numpy()

        seg = pd.DataFrame({
            "storm_id_code": sid,
            "segment_id": seg_id,
            "imd_storm_id": meta.loc[sid, "imd_storm_id"] if sid in meta.index else None,
            "storm_name": meta.loc[sid, "name"] if sid in meta.index else "(unnamed)",
            "storm_year": int(storm_year.loc[sid]),
            "t_unix": grid,
            "lat": lat, "lon": lon, "wind_kt": wind,
            "interpolated": interpolated,
            "nearest_wind_obs_gap_h": near / 3600.0,
            "pressure_hpa": pres, "pressure_drop": pdrop,
            "storm_age_hours": (grid - float(storm_first.loc[sid])) / 3600.0,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "basin_ARB": int(basin["basin_ARB"]),
            "basin_BOB": int(basin["basin_BOB"]),
            "basin_LAND": int(basin["basin_LAND"]),
            "peak_grade_ordinal": float(peak_grade.loc[sid]),
        })

        # per-grid-step rates (uniform 6 h)
        seg["wind_tendency_6h"] = seg["wind_kt"].diff()
        seg["wind_rate_per_h"] = seg["wind_tendency_6h"] / 6.0
        seg["pressure_tendency_6h"] = seg["pressure_hpa"].diff()
        seg["pressure_rate_per_h"] = seg["pressure_tendency_6h"] / 6.0
        seg["dlat_6h"] = seg["lat"].diff()
        seg["dlon_6h"] = seg["lon"].diff()

        # targets at exactly +6 / +12 / +24 h  (== +1 / +2 / +4 grid steps)
        for h in LEADS_H:
            k = h // 6
            seg[f"fut_lat_{h}h"] = seg["lat"].shift(-k)
            seg[f"fut_lon_{h}h"] = seg["lon"].shift(-k)
            seg[f"fut_wind_{h}h"] = seg["wind_kt"].shift(-k)
            seg[f"delta_wind_{h}h"] = seg[f"fut_wind_{h}h"] - seg["wind_kt"]
            # is the +h wind value itself an interpolation (vs a real IMD fix)?
            fut_interp = seg["interpolated"].shift(-k)
            seg[f"fut_interp_{h}h"] = fut_interp
            # Δwind is "clean" only if BOTH endpoints are real fixes
            # (a missing future point counts as not-clean via `== False`)
            seg[f"delta_wind_{h}h_clean"] = (~seg["interpolated"]) & (fut_interp == False)  # noqa: E712

        rows.append(seg)

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(RESAMPLED_PARQUET, index=False)

    # ---- report fragment ---------------------------------------------------
    grid_sizes = out.groupby("segment_id").size()
    holdout = out[out["storm_year"] >= HOLDOUT_START_YEAR]
    tgt_cols = [f"delta_wind_{h}h" for h in LEADS_H]
    ho_tgt_rows = holdout.dropna(subset=tgt_cols, how="all")

    frag = {
        "raw_rows_in": int(len(clean)),
        "segments_total": int(clean["segment_id"].nunique()),
        "segments_kept": int(grid_sizes.shape[0]),
        "segments_dropped_lt_5_grid_pts": int(dropped_short),
        "resampled_rows_out": int(len(out)),
        "raw_segment_len_points": {
            "min": int(raw_seg_sizes.min()), "p25": float(raw_seg_sizes.quantile(.25)),
            "median": float(raw_seg_sizes.median()),
            "p75": float(raw_seg_sizes.quantile(.75)),
            "max": int(raw_seg_sizes.max()), "mean": round(float(raw_seg_sizes.mean()), 2),
        },
        "grid_segment_len_points": {
            "min": int(grid_sizes.min()), "p25": float(grid_sizes.quantile(.25)),
            "median": float(grid_sizes.median()),
            "p75": float(grid_sizes.quantile(.75)),
            "max": int(grid_sizes.max()), "mean": round(float(grid_sizes.mean()), 2),
        },
        "grid_len_histogram": {
            f"{lo}-{hi}": int(((grid_sizes >= lo) & (grid_sizes < hi)).sum())
            for lo, hi in [(5, 9), (9, 13), (13, 21), (21, 33), (33, 999)]
        },
        "wind_interpolated_fraction_all": round(float(out["interpolated"].mean()), 3),
        "wind_interpolated_fraction_holdout": round(float(holdout["interpolated"].mean()), 3),
        "holdout_segments": int(holdout["segment_id"].nunique()),
        "holdout_storms": int(holdout["storm_id_code"].nunique()),
        "holdout_target_rows_per_lead": {
            f"{h}h": int(holdout[f"delta_wind_{h}h"].notna().sum()) for h in LEADS_H
        },
        "holdout_target_interp_fraction_per_lead": {
            f"{h}h": round(float(
                holdout.loc[holdout[f"delta_wind_{h}h"].notna(), "interpolated"].mean()), 3)
            for h in LEADS_H
        },
    }
    (FRAG / "resample.json").write_text(json.dumps(frag, indent=2), encoding="utf-8")
    print(json.dumps(frag, indent=2))
    return out


def load_resampled() -> pd.DataFrame:
    if not RESAMPLED_PARQUET.exists():
        return build()
    return pd.read_parquet(RESAMPLED_PARQUET)


if __name__ == "__main__":
    build()
