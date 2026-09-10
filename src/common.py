"""Shared helpers for the ResQ cyclone pipeline (SIH 2026 PS26070)."""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FRAG = REPORTS / "_fragments"
RAW = ROOT / "Raw data"

CLEAN_PARQUET = DATA / "cyclone_clean.parquet"
RESAMPLED_PARQUET = DATA / "cyclone_resampled_6h.parquet"

HOLDOUT_START_YEAR = 2019
RANDOM_STATE = 26070
EARTH_R_KM = 6371.0088
GRID_SECONDS = 6 * 3600
SEGMENT_GAP_SECONDS = 12 * 3600          # split a storm wherever the gap exceeds this
MIN_SEGMENT_POINTS = 5                   # drop resampled segments shorter than this
LEADS_H = (6, 12, 24)

for d in (DATA, REPORTS, FRAG):
    d.mkdir(parents=True, exist_ok=True)


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    d = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(d))


def deg_to_km(dlat, dlon, lat_ref):
    """Local flat-earth conversion of a (dlat, dlon) step to (north_km, east_km)."""
    km_per_deg_lat = 111.19492664455873
    north = dlat * km_per_deg_lat
    east = dlon * km_per_deg_lat * np.cos(np.radians(lat_ref))
    return north, east


def along_cross_track_km(pred_lat, pred_lon, true_lat, true_lon,
                         heading_lat, heading_lon, ref_lat):
    """Decompose the prediction error into along-track / cross-track (km).

    heading_* is the storm's recent motion vector (dlat, dlon); the error vector
    (true - pred) is projected onto the unit heading and its left-normal.
    """
    hn, he = deg_to_km(heading_lat, heading_lon, ref_lat)
    hmag = np.hypot(hn, he)
    hmag = np.where(hmag < 1e-6, 1e-6, hmag)
    un, ue = hn / hmag, he / hmag           # unit along-track
    en, ee = -ue, un                        # unit cross-track (left normal)
    dn, de = deg_to_km(true_lat - pred_lat, true_lon - pred_lon, ref_lat)
    along = dn * un + de * ue
    cross = dn * en + de * ee
    return along, cross


def storm_metadata() -> pd.DataFrame:
    """storm_id_code -> imd_storm_id, name, year, peak_grade, peak_wind_kt.

    The mapping is recovered by matching each storm's first (time, lat, lon) in
    the clean table against the raw IMD observation file; all 425 storms match
    exactly.
    """
    clean = pd.read_parquet(CLEAN_PARQUET).dropna(subset=["storm_id_code"]).copy()
    clean["t"] = pd.to_datetime(clean["dt_snap_unix"], unit="s")
    first = (clean.sort_values("t").groupby("storm_id_code").first()
             .reset_index()[["storm_id_code", "t", "lat", "lon"]])

    obs = pd.read_csv(RAW / "cyclone_observations_final.csv")
    obs["t"] = pd.to_datetime(obs["dt_snap"])
    obs_first = (obs.sort_values("t").groupby("imd_storm_id").first()
                 .reset_index()[["imd_storm_id", "t", "lat", "lon"]])

    m = first.merge(obs_first, on=["t", "lat", "lon"], how="left")

    storms = pd.read_csv(RAW / "cyclone_storms_final.csv")
    m = m.merge(storms[["imd_storm_id", "name", "year", "peak_grade",
                        "peak_wind_kt"]],
                on="imd_storm_id", how="left")
    m["name"] = m["name"].fillna("(unnamed)")
    return m.set_index("storm_id_code")[["imd_storm_id", "name", "year",
                                         "peak_grade", "peak_wind_kt"]]
