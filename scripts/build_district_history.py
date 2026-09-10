"""
Build app/static/data/district_history.json — how often each coastal district
has been within 100 km of a tropical-cyclone track in the IMD archive.

Everything is derived from the local 425-storm archive
(`Raw data/cyclone_observations_final.csv` + `cyclone_storms_final.csv`) and
public district boundaries (GADM 4.1 level-2). No forecast, no external storm
data.

Boundary source
---------------
GADM 4.1 level-2 (https://gadm.org/), file
`gadm41_IND_2.json` from
https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip
Downloaded automatically to data/gadm_cache/ ; the retrieval date and URL are recorded
in the output JSON.

If the boundary download fails, the script falls back to a small fixed list of
named coastal districts with hand-entered centroid lat/lon and marks the whole
output `"mode": "centroid_approx"` so the UI can label it
"approximate, centroid-based". Boundaries are never fabricated.

Distance
--------
Tracks and polygons are projected to a local equirectangular km plane
(x = Δlon·111.32·cos φ0, y = Δlat·110.57) centred on each district; shapely then
gives the track-to-district distance in km. Accurate to ~1 % over the few
hundred km that matters for the 100 km test.

Run:  python scripts/build_district_history.py
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math
import pathlib
import re
import warnings
import zipfile

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="invalid value encountered in distance")

ROOT = pathlib.Path(__file__).resolve().parents[1]
GEO = ROOT / "data" / "gadm_cache"
OUT = ROOT / "app" / "static" / "data" / "district_history.json"
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip"

# States on the Bay of Bengal / Arabian Sea / island groups — the only ones a
# North-Indian-Ocean cyclone can realistically strike.
COASTAL_STATES = {
    "Gujarat", "DamanandDiu", "Maharashtra", "Goa", "Karnataka", "Kerala",
    "TamilNadu", "Puducherry", "AndhraPradesh", "Odisha", "WestBengal",
    "AndamanandNicobar", "Lakshadweep",
}

GRADES = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]
GRADE_LABEL = {
    "D": "Depression", "DD": "Deep Depression", "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm", "VSCS": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm", "SuCS": "Super Cyclonic Storm",
}
SEVERE_PLUS = {"SCS", "VSCS", "ESCS", "SuCS"}
MATCH_KM = 100.0          # "affected" threshold
KEEP_KM = 150.0           # keep a district in the dropdown if any storm within this
SINCE = 1990              # headline sentence counts from this year


def _decamel(s: str) -> str:
    out = []
    for i, ch in enumerate(s):
        if i and ch.isupper() and (not s[i - 1].isupper()):
            out.append(" ")
        out.append(ch)
    return "".join(out).replace(" and ", " and ").strip()


# ─────────────────────────────────────────────────────────────────────────────
# boundaries
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_CENTROIDS = [
    # district, state, lat, lon  — hand-entered, used ONLY if GADM is unavailable
    ("Kachchh", "Gujarat", 23.73, 69.86), ("Jamnagar", "Gujarat", 22.47, 70.06),
    ("Porbandar", "Gujarat", 21.64, 69.61), ("Junagadh", "Gujarat", 21.52, 70.46),
    ("Amreli", "Gujarat", 21.60, 71.22), ("Bhavnagar", "Gujarat", 21.76, 72.15),
    ("Valsad", "Gujarat", 20.61, 72.93), ("Mumbai Suburban", "Maharashtra", 19.18, 72.93),
    ("Raigad", "Maharashtra", 18.52, 73.18), ("Ratnagiri", "Maharashtra", 16.99, 73.31),
    ("Sindhudurg", "Maharashtra", 16.13, 73.66), ("North Goa", "Goa", 15.55, 73.90),
    ("Udupi", "Karnataka", 13.34, 74.75), ("Dakshina Kannada", "Karnataka", 12.85, 75.23),
    ("Kannur", "Kerala", 11.87, 75.37), ("Kozhikode", "Kerala", 11.25, 75.78),
    ("Ernakulam", "Kerala", 10.00, 76.31), ("Alappuzha", "Kerala", 9.50, 76.34),
    ("Thiruvananthapuram", "Kerala", 8.52, 76.94), ("Kanniyakumari", "TamilNadu", 8.19, 77.41),
    ("Thoothukudi", "TamilNadu", 8.80, 78.15), ("Ramanathapuram", "TamilNadu", 9.37, 78.83),
    ("Nagapattinam", "TamilNadu", 10.77, 79.84), ("Cuddalore", "TamilNadu", 11.75, 79.75),
    ("Villupuram", "TamilNadu", 11.94, 79.49), ("Chengalpattu", "TamilNadu", 12.69, 79.98),
    ("Chennai", "TamilNadu", 13.08, 80.27), ("Tiruvallur", "TamilNadu", 13.14, 79.91),
    ("Puducherry", "Puducherry", 11.94, 79.81), ("Nellore", "AndhraPradesh", 14.44, 79.99),
    ("Prakasam", "AndhraPradesh", 15.35, 80.05), ("Krishna", "AndhraPradesh", 16.24, 81.10),
    ("East Godavari", "AndhraPradesh", 16.90, 82.20), ("Visakhapatnam", "AndhraPradesh", 17.69, 83.22),
    ("Srikakulam", "AndhraPradesh", 18.30, 83.90), ("Ganjam", "Odisha", 19.39, 84.86),
    ("Puri", "Odisha", 19.81, 85.83), ("Jagatsinghpur", "Odisha", 20.25, 86.17),
    ("Kendrapara", "Odisha", 20.50, 86.42), ("Bhadrak", "Odisha", 21.06, 86.51),
    ("Balasore", "Odisha", 21.49, 86.93), ("Purba Medinipur", "WestBengal", 21.93, 87.79),
    ("South 24 Parganas", "WestBengal", 22.16, 88.43), ("North 24 Parganas", "WestBengal", 22.62, 88.70),
]


def load_boundaries():
    """Returns (mode, [(name, state, geom_or_None, centroid_lat, centroid_lon), ...])."""
    from shapely.geometry import shape

    GEO.mkdir(parents=True, exist_ok=True)
    jf = GEO / "gadm41_IND_2.json"
    note = None
    if not jf.exists():
        try:
            import requests
            print("[geo] downloading GADM 4.1 level-2 …")
            r = requests.get(GADM_URL, timeout=180)
            r.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(GEO)
        except Exception as e:                       # noqa: BLE001
            note = f"GADM download failed: {e!r}"
            print("[geo]", note)

    if jf.exists():
        gj = json.loads(jf.read_text(encoding="utf-8"))
        rows = []
        for feat in gj["features"]:
            p = feat["properties"]
            if p.get("NAME_1") not in COASTAL_STATES:
                continue
            try:
                g = shape(feat["geometry"])
                c = g.centroid
            except Exception:                        # noqa: BLE001
                continue
            name = _decamel(p["NAME_2"])
            aliases = set()
            vn = p.get("VARNAME_2")
            if vn and vn != "NA":
                for a in re.split(r"[|/,]", vn):
                    a = _decamel(a.strip())
                    if a:
                        aliases.add(a)
            # doubled-consonant spelling variants
            # (Nagappattinam -> Nagapattinam / Nagappatinam / Nagapatinam)
            dbl = list(re.finditer(r"([bcdfghjklmnpqrstvwxyz])\1", name, flags=re.I))
            if dbl:
                aliases.add(re.sub(r"([bcdfghjklmnpqrstvwxyz])\1", r"\1", name, flags=re.I))
                for mm in dbl:
                    aliases.add(name[:mm.start()] + mm.group(1) + name[mm.end():])
            rows.append((name, _decamel(p["NAME_1"]), g,
                         round(c.y, 4), round(c.x, 4), sorted(aliases)))
        if rows:
            return "boundary", rows, {
                "source": "GADM 4.1 level-2 (gadm.org)",
                "url": GADM_URL,
                "retrieved": dt.date.today().isoformat(),
                "note": "district polygons; track-to-polygon distance",
            }

    # fallback
    rows = [(_decamel(n), _decamel(s), None, lat, lon, [])
            for (n, s, lat, lon) in FALLBACK_CENTROIDS]
    return "centroid_approx", rows, {
        "source": "hand-entered coastal-district centroids (GADM unavailable)",
        "url": GADM_URL,
        "retrieved": dt.date.today().isoformat(),
        "note": note or "GADM file absent; distances measured to a single "
                        "centroid point, not a boundary — APPROXIMATE.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# storms
# ─────────────────────────────────────────────────────────────────────────────
def load_storms():
    obs = pd.read_csv(ROOT / "Raw data" / "cyclone_observations_final.csv")
    meta = pd.read_csv(ROOT / "Raw data" / "cyclone_storms_final.csv")
    obs = obs.dropna(subset=["lat", "lon"]).sort_values(["imd_storm_id", "dt_snap"])
    meta = meta.set_index("imd_storm_id")
    storms = []
    for sid, g in obs.groupby("imd_storm_id"):
        if len(g) < 2:
            continue
        m = meta.loc[sid] if sid in meta.index else None
        pk = (m["peak_grade"] if m is not None and isinstance(m["peak_grade"], str)
              else None)
        if pk not in GRADES:
            # derive from peak wind if grade missing
            w = float(g["wind_kt"].max())
            pk = ("SuCS" if w >= 120 else "ESCS" if w >= 90 else "VSCS" if w >= 64
                  else "SCS" if w >= 48 else "CS" if w >= 34 else "DD" if w >= 28
                  else "D")
        storms.append({
            "sid": sid,
            "name": (g["name"].dropna().iloc[0].strip().title()
                     if g["name"].notna().any() and str(g["name"].dropna().iloc[0]).strip()
                     else "(unnamed)"),
            "year": int(g["year"].iloc[0]),
            "peak_grade": pk,
            "peak_wind_kt": (None if pd.isna(g["wind_kt"].max())
                             else int(round(float(g["wind_kt"].max())))),
            "lat": g["lat"].to_numpy(float),
            "lon": g["lon"].to_numpy(float),
        })
    return storms


def _projector(clat, clon):
    kx = 111.32 * math.cos(math.radians(clat))
    ky = 110.57

    def proj_xy(lat, lon):
        return (np.asarray(lon, float) - clon) * kx, (np.asarray(lat, float) - clat) * ky
    return proj_xy


def project_district(geom, clat, clon):
    """District polygon → local km plane, simplified. Done once per district."""
    import shapely
    from shapely.geometry import Point

    if geom is None:
        return Point(0.0, 0.0)                        # centroid mode
    kx = 111.32 * math.cos(math.radians(clat))
    ky = 110.57
    g = shapely.simplify(geom, 0.02, preserve_topology=True)
    if not g.is_valid:
        g = shapely.make_valid(g)
    return shapely.transform(
        g, lambda a: np.column_stack([(a[:, 0] - clon) * kx, (a[:, 1] - clat) * ky]))


def track_distance_km(proj_geom, slat, slon, clat, clon):
    from shapely.geometry import LineString, Point

    kx = 111.32 * math.cos(math.radians(clat))
    ky = 110.57
    sx = (np.asarray(slon, float) - clon) * kx
    sy = (np.asarray(slat, float) - clat) * ky
    track = LineString(np.column_stack([sx, sy])) if sx.size >= 2 else Point(sx[0], sy[0])
    return track.distance(proj_geom)


def main():
    mode, districts, boundary_meta = load_boundaries()
    storms = load_storms()
    print(f"[hist] mode={mode}  districts={len(districts)}  storms={len(storms)}")

    # bbox prefilter per storm
    for s in storms:
        s["bbox"] = (s["lat"].min(), s["lat"].max(), s["lon"].min(), s["lon"].max())

    records = []
    for i, (name, state, geom, clat, clon, aliases) in enumerate(districts):
        # district raw-lat/lon bbox (from geometry if present, else centroid)
        if geom is not None:
            gx0, gy0, gx1, gy1 = geom.bounds          # (minlon, minlat, maxlon, maxlat)
        else:
            gx0 = gx1 = clon
            gy0 = gy1 = clat
        pg = project_district(geom, clat, clon)
        hits = []
        for s in storms:
            la0, la1, lo0, lo1 = s["bbox"]
            # ~2° (~220 km) gate: skip storms whose bbox can't be within KEEP_KM
            if la0 - 2.0 > gy1 or la1 + 2.0 < gy0 or lo0 - 2.0 > gx1 or lo1 + 2.0 < gx0:
                continue
            d = track_distance_km(pg, s["lat"], s["lon"], clat, clon)
            if d <= KEEP_KM:
                # closest point on the track (for the map) — nearest vertex
                kx = 111.32 * math.cos(math.radians(clat))
                dd = np.hypot((s["lon"] - clon) * kx, (s["lat"] - clat) * 110.57)
                j = int(dd.argmin())
                hits.append({
                    "sid": s["sid"], "name": s["name"], "year": s["year"],
                    "peak_grade": s["peak_grade"],
                    "peak_grade_label": GRADE_LABEL[s["peak_grade"]],
                    "peak_wind_kt": s["peak_wind_kt"],
                    "closest_km": round(float(d), 1),
                    "closest_lat": round(float(s["lat"][j]), 2),
                    "closest_lon": round(float(s["lon"][j]), 2),
                    "track": [[round(float(a), 2), round(float(b), 2)]
                              for a, b in zip(s["lat"], s["lon"])],
                })
        if not hits:
            continue
        within100 = [h for h in hits if h["closest_km"] <= MATCH_KM]
        since = [h for h in within100 if h["year"] >= SINCE]
        by_decade = {}
        for h in within100:
            dkey = f"{(h['year'] // 10) * 10}s"
            by_decade.setdefault(dkey, 0)
            by_decade[dkey] += 1
        by_cat = {gr: sum(1 for h in within100 if h["peak_grade"] == gr) for gr in GRADES}
        records.append({
            "district": name, "state": state, "aliases": aliases,
            "centroid": [clat, clon],
            "n_within_100km": len(within100),
            "n_within_100km_since_1990": len(since),
            "n_severe_plus_since_1990": sum(1 for h in since
                                            if h["peak_grade"] in SEVERE_PLUS),
            "n_within_150km": len(hits),
            "by_decade": dict(sorted(by_decade.items())),
            "by_category": by_cat,
            "storms": sorted(within100, key=lambda h: -h["year"]),
        })
        if (i + 1) % 50 == 0:
            print(f"  … {i + 1}/{len(districts)}")

    records.sort(key=lambda r: (r["state"], r["district"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "boundary_source": boundary_meta,
        "storm_source": {
            "file": "Raw data/cyclone_observations_final.csv",
            "note": "IMD/RSMC New Delhi best-track archive, "
                    f"{len(storms)} storms {min(s['year'] for s in storms)}–"
                    f"{max(s['year'] for s in storms)}",
        },
        "method": {
            "match_km": MATCH_KM, "keep_km": KEEP_KM, "since_year": SINCE,
            "distance": "local equirectangular km plane, track-to-district",
            "caveat": "This is what has happened before. It is not a forecast.",
        },
        "n_districts": len(records),
        "districts": records,
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({len(records)} districts with >=1 storm within {KEEP_KM:.0f} km)")


if __name__ == "__main__":
    main()
