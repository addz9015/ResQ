"""
ERA5 environmental predictors for the 6 h resampled grid  (src/era5.py).

This is the critical path for real intensity skill. The fetch layer is built now
so it is ready the moment CDS credentials are approved; without them it FAILS
CLEANLY - it writes `data/cyclone_6h_with_env.parquet` with the env columns
present but null, prints exactly what is missing, and exits 0.

Design (when credentials + libs are available):
  * one CDS request per (year, month) for the North Indian Ocean box, cached as
    NetCDF in `data/era5_cache/` so re-runs cost nothing;
  * fields:
      850 & 200 hPa u,v      -> deep-layer vertical wind shear |V850 - V200|
      500 hPa geopotential z -> steering / ridge position (m)
      600 hPa relative hum.  -> mid-level moisture (%)
      sea-surface temp.      -> SST (degC)
  * per grid state: value at the storm centre and the mean over 0-200, 200-500
    and 500-800 km annuli;
  * missing fields are left NULL - never interpolated across gaps.

Run:  python -m src.era5
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

from .common import DATA, FRAG, haversine_km
from .segment_resample import load_resampled

CACHE = DATA / "era5_cache"
OUT_PARQUET = DATA / "cyclone_6h_with_env.parquet"

# North Indian Ocean bounding box  [N, W, S, E]
AREA = [30, 40, -5, 105]
GRID = 0.25
PLEVELS = ["200", "500", "600", "850"]
ANNULI_KM = [(0, 200), (200, 500), (500, 800)]
ENV_FIELDS = ["shear_deep_ms", "steering_z500_m", "rh600_pct", "sst_c"]
ENV_COLUMNS = [f"{f}{sfx}" for f in ENV_FIELDS
               for sfx in ["", "_a200", "_a500", "_a800"]]

CDS_HELP = (
    "ERA5 fetch is not available. To enable it:\n"
    "  1. pip install cdsapi xarray netcdf4\n"
    "  2. create ~/.cdsapirc with your CDS API key\n"
    "     (https://cds.climate.copernicus.eu/how-to-api)\n"
    "  3. accept the 'ERA5 hourly data on pressure/single levels' licence\n"
    "Until then, env columns are written NULL and no model uses them."
)


def _have_stack():
    try:
        import cdsapi  # noqa: F401
        import xarray  # noqa: F401
        return True, None
    except Exception as e:  # pragma: no cover - env dependent
        return False, str(e)


def _creds_present() -> bool:
    if (pathlib.Path.home() / ".cdsapirc").exists():
        return True
    import os
    return bool(os.environ.get("CDSAPI_URL") and os.environ.get("CDSAPI_KEY"))


def _months_needed(df: pd.DataFrame):
    t = pd.to_datetime(df["t_unix"], unit="s")
    return sorted({(d.year, d.month) for d in t})


def _download_month(year: int, month: int) -> pathlib.Path:  # pragma: no cover
    import cdsapi
    CACHE.mkdir(parents=True, exist_ok=True)
    pl = CACHE / f"era5_pl_{year}{month:02d}.nc"
    sl = CACHE / f"era5_sl_{year}{month:02d}.nc"
    c = cdsapi.Client(quiet=True)
    days = [f"{d:02d}" for d in range(1, 32)]
    times = [f"{h:02d}:00" for h in (0, 6, 12, 18)]
    if not pl.exists():
        c.retrieve("reanalysis-era5-pressure-levels", {
            "product_type": "reanalysis", "format": "netcdf",
            "variable": ["u_component_of_wind", "v_component_of_wind",
                         "geopotential", "relative_humidity"],
            "pressure_level": PLEVELS, "year": str(year), "month": f"{month:02d}",
            "day": days, "time": times, "area": AREA, "grid": [GRID, GRID]}, str(pl))
    if not sl.exists():
        c.retrieve("reanalysis-era5-single-levels", {
            "product_type": "reanalysis", "format": "netcdf",
            "variable": ["sea_surface_temperature"],
            "year": str(year), "month": f"{month:02d}", "day": days,
            "time": times, "area": AREA, "grid": [GRID, GRID]}, str(sl))
    return pl, sl


def _annulus_mean(field2d, lats, lons, clat, clon, r0, r1):
    LA, LO = np.meshgrid(lats, lons, indexing="ij")
    d = haversine_km(clat, clon, LA, LO)
    m = (d >= r0) & (d < r1) & np.isfinite(field2d)
    return float(np.nanmean(field2d[m])) if m.any() else np.nan


def _extract_state(ds_pl, ds_sl, t, clat, clon):  # pragma: no cover
    import numpy as np
    sub_pl = ds_pl.sel(time=t, method="nearest")
    sub_sl = ds_sl.sel(time=t, method="nearest")
    lats = ds_pl.latitude.values
    lons = ds_pl.longitude.values

    def at_centre(a):
        return float(a.sel(latitude=clat, longitude=clon, method="nearest").values)

    u850 = sub_pl["u"].sel(level=850); v850 = sub_pl["v"].sel(level=850)
    u200 = sub_pl["u"].sel(level=200); v200 = sub_pl["v"].sel(level=200)
    shear = np.hypot(u850 - u200, v850 - v200)
    z500 = sub_pl["z"].sel(level=500) / 9.80665
    rh600 = sub_pl["r"].sel(level=600)
    sst = sub_sl["sst"] - 273.15

    row = {}
    for name, fld in [("shear_deep_ms", shear), ("steering_z500_m", z500),
                      ("rh600_pct", rh600), ("sst_c", sst)]:
        arr = fld.values
        row[name] = at_centre(fld)
        for (r0, r1), sfx in zip(ANNULI_KM, ["_a200", "_a500", "_a800"]):
            row[name + sfx] = _annulus_mean(arr, lats, lons, clat, clon, r0, r1)
    return row


def build() -> dict:
    grid = load_resampled()
    for col in ENV_COLUMNS:
        grid[col] = np.nan

    have_libs, liberr = _have_stack()
    have_creds = _creds_present()
    fetched_months, failed_months = [], []

    if have_libs and have_creds:  # pragma: no cover - needs credentials
        import xarray as xr
        by_month = {}
        for (yr, mo) in _months_needed(grid):
            try:
                pl, sl = _download_month(yr, mo)
                by_month[(yr, mo)] = (xr.open_dataset(pl), xr.open_dataset(sl))
                fetched_months.append(f"{yr}-{mo:02d}")
            except Exception as e:
                failed_months.append(f"{yr}-{mo:02d}: {e}")
        tt = pd.to_datetime(grid["t_unix"], unit="s")
        for idx, r in grid.iterrows():
            key = (tt[idx].year, tt[idx].month)
            if key not in by_month or not np.isfinite(r["lat"]) or not np.isfinite(r["lon"]):
                continue
            ds_pl, ds_sl = by_month[key]
            vals = _extract_state(ds_pl, ds_sl, np.datetime64(tt[idx]),
                                  float(r["lat"]), float(r["lon"]))
            for k, v in vals.items():
                grid.at[idx, k] = v
        status = "fetched"
    else:
        status = "unavailable"
        missing = []
        if not have_libs:
            missing.append(f"libraries ({liberr})")
        if not have_creds:
            missing.append("CDS credentials (~/.cdsapirc)")
        print("ERA5 STATUS: unavailable -", " and ".join(missing))
        print(CDS_HELP)

    grid.to_parquet(OUT_PARQUET, index=False)

    coverage = {c: round(float(grid[c].notna().mean()), 3) for c in ENV_COLUMNS}
    report = {
        "status": status,
        "output": str(OUT_PARQUET.relative_to(DATA.parent)),
        "n_grid_states": int(len(grid)),
        "months_required": len(_months_needed(grid)),
        "months_fetched": fetched_months,
        "months_failed": failed_months,
        "field_coverage": coverage,
        "note": ("env columns are present but NULL until CDS access is granted; "
                 "no model consumes them yet - `src/era5.py` is the fetch layer "
                 "only."),
        "help": CDS_HELP if status == "unavailable" else None,
    }
    (FRAG / "era5.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "field_coverage"}, indent=2))
    return report


if __name__ == "__main__":
    build()
