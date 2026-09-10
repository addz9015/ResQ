"""
ResQ operational dashboard - FastAPI backend.

Endpoints
  GET  /                     -> the dashboard (static/dashboard.html)
  GET  /api/replay           -> list of 2019+ storms available for replay
  GET  /api/replay/{storm_id}-> full step-by-step hindcast for one storm
  POST /api/forecast         -> forecast for an arbitrary storm state
  GET  /api/health

Everything is served from the pre-computed artifacts in static/data/. The
replay demo makes NO live calls - it just walks the pre-computed steps.

Run:  python app/run.py        (preferred - builds artifacts if missing)
  or: uvicorn app.backend.main:app --port 8000
"""
from __future__ import annotations

import json
import pathlib
import pickle
import re

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

HERE = pathlib.Path(__file__).resolve().parent
STATIC = HERE.parent / "static"
DATA = STATIC / "data"
DISCLAIMER = "ResQ model estimate — not an official warning."

app = FastAPI(title="ResQ cyclone dashboard", version="0.4")

_engine = None
_replay = None


def engine():
    global _engine
    if _engine is None:
        p = DATA / "engine.pkl"
        if not p.exists():
            raise HTTPException(503, "engine.pkl missing - run `python app/precompute.py`")
        with open(p, "rb") as f:
            _engine = pickle.load(f)
    return _engine


def replay_data():
    global _replay
    if _replay is None:
        p = DATA / "replay.json"
        if not p.exists():
            raise HTTPException(503, "replay.json missing - run `python app/precompute.py`")
        _replay = json.loads(p.read_text(encoding="utf-8"))
    return _replay


class StateIn(BaseModel):
    lat: float
    lon: float
    wind_kt: float
    pressure_hpa: float | None = None
    pressure_drop: float | None = None
    wind_tendency_6h: float | None = 0.0
    pressure_tendency_6h: float | None = 0.0
    dlat_6h: float = 0.0
    dlon_6h: float = 0.0
    storm_age_hours: float | None = 24.0
    month: int = 6
    basin: str = "BOB"           # BOB | ARB | LAND
    storm_id_code: float | None = -999.0


@app.get("/api/health")
def health():
    return {"ok": True, "has_engine": (DATA / "engine.pkl").exists(),
            "has_replay": (DATA / "replay.json").exists()}


@app.get("/api/presets")
def presets():
    """Real 2019+ holdout states pre-filled for the Forecast tab, plus a
    MANDOUS-2022 preset. Read from the resampled parquet."""
    import pandas as pd
    import numpy as np
    p = HERE.parents[1] / "data" / "cyclone_resampled_6h.parquet"
    if not p.exists():
        return {"presets": []}
    d = pd.read_parquet(p)
    d = d[(d["storm_year"] >= 2019) & d["dlat_6h"].notna()
          & d["wind_kt"].notna()].copy()
    d["month"] = pd.to_datetime(d["t_unix"], unit="s").dt.month
    d["date"] = pd.to_datetime(d["t_unix"], unit="s").dt.strftime("%Y-%m-%d")
    d["prev_lat"] = d["lat"] - d["dlat_6h"]
    d["prev_lon"] = d["lon"] - d["dlon_6h"]

    def row(r, label):
        return {"label": label, "storm": r["imd_storm_id"], "name": r["storm_name"],
                "lat": round(float(r["lat"]), 2), "lon": round(float(r["lon"]), 2),
                "wind_kt": round(float(r["wind_kt"]), 0),
                "pressure_hpa": (None if pd.isna(r["pressure_hpa"])
                                 else round(float(r["pressure_hpa"]), 0)),
                "prev_lat": round(float(r["prev_lat"]), 2),
                "prev_lon": round(float(r["prev_lon"]), 2),
                "storm_age_hours": round(float(r["storm_age_hours"]), 0),
                "date": r["date"], "month": int(r["month"]),
                "actual_dwind_24h": (None if pd.isna(r.get("delta_wind_24h"))
                                     else round(float(r["delta_wind_24h"]), 0))}

    out = []
    mand = d[d["storm_name"].astype(str).str.contains("MANDOUS", na=False)
             & d["delta_wind_24h"].notna()]
    if len(mand):
        r = mand.loc[mand["delta_wind_24h"].idxmax()]
        out.append(row(r, f"★ MANDOUS 2022 (+{r['delta_wind_24h']:.0f} kt / 24 h — RI case)"))
    # a spread of other 2019+ states: every ~40th, developing storms first
    d2 = d[d["storm_name"] != mand["storm_name"].iloc[0] if len(mand) else slice(None)]
    for _, r in d.sort_values("t_unix").iloc[::45].head(30).iterrows():
        lab = f"{r['storm_name']} {r['imd_storm_id']} — {r['date']}, {r['wind_kt']:.0f} kt"
        out.append(row(r, lab))
    return {"presets": out}


@app.get("/api/replay")
def list_replay():
    d = replay_data()
    return {"n_storms": d["n_storms"], "generated_utc": d["generated_utc"],
            "storms": d["storms"]}


@app.get("/api/replay/{storm_id}")
def get_replay(storm_id: str):
    d = replay_data()
    if storm_id not in d["replay"]:
        raise HTTPException(404, f"no replay for {storm_id}")
    return {"id": storm_id, "disclaimer": DISCLAIMER, **d["replay"][storm_id]}


@app.post("/api/forecast")
def post_forecast(s: StateIn):
    import numpy as np
    doy = 30 * (s.month - 1) + 15
    state = {
        "lat": s.lat, "lon": s.lon, "wind_kt": s.wind_kt,
        "pressure_hpa": s.pressure_hpa if s.pressure_hpa is not None else np.nan,
        "pressure_drop": s.pressure_drop if s.pressure_drop is not None else np.nan,
        "wind_tendency_6h": s.wind_tendency_6h or 0.0,
        "pressure_tendency_6h": s.pressure_tendency_6h or 0.0,
        "dlat_6h": s.dlat_6h, "dlon_6h": s.dlon_6h,
        "storm_age_hours": s.storm_age_hours if s.storm_age_hours is not None else 24.0,
        "month_sin": np.sin(2 * np.pi * s.month / 12),
        "month_cos": np.cos(2 * np.pi * s.month / 12),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "basin_ARB": int(s.basin == "ARB"), "basin_BOB": int(s.basin == "BOB"),
        "basin_LAND": int(s.basin == "LAND"),
        "storm_id_code": s.storm_id_code,
    }
    out = engine().forecast(state)
    out["disclaimer"] = DISCLAIMER
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Official Warnings — paste a bulletin, get it parsed + explained + an overlay
#  There is NO live-fetch endpoint. IMD has no API; the paste box is the feature.
# ─────────────────────────────────────────────────────────────────────────────
_CAT_WIND = {"Depression": 25, "Deep Depression": 30, "Cyclonic Storm": 45,
             "Severe Cyclonic Storm": 55, "Very Severe Cyclonic Storm": 75,
             "Extremely Severe Cyclonic Storm": 100, "Super Cyclonic Storm": 130}


class BulletinText(BaseModel):
    text: str


_MONTHS = ("january february march april may june july august september october "
           "november december").split()


def _hours_to_landfall(issued_at_iso, time_raw):
    """Best-effort whole-hours between the bulletin's issue time and the stated
    landfall time. Returns None unless BOTH are clearly determinable from the
    text — never a guess."""
    import datetime as _dt
    import re as _re
    if not issued_at_iso or not time_raw:
        return None
    try:
        issued = _dt.datetime.fromisoformat(issued_at_iso.replace("Z", "+00:00"))
    except Exception:
        return None
    tl = time_raw.lower()
    m = _re.search(r"(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")", tl)
    if not m:
        return None
    day = int(m.group(1))
    mon = _MONTHS.index(m.group(2)) + 1
    hour = 0
    if "midnight" in tl:
        hour = 0
    elif "noon" in tl or "midday" in tl:
        hour = 12
    elif (hm := _re.search(r"(\d{4})\s*(?:hrs?|hours?)?\s*(ist|utc)?", tl)):
        hour = int(hm.group(1)[:2])
    elif "morning" in tl:
        hour = 6
    elif "evening" in tl or "night" in tl:
        hour = 20
    elif "afternoon" in tl:
        hour = 15
    year = issued.year
    try:
        landfall = _dt.datetime(year, mon, day, hour, tzinfo=issued.tzinfo)
    except Exception:
        return None
    if landfall < issued - _dt.timedelta(days=1):     # wrapped to next year
        landfall = landfall.replace(year=year + 1)
    delta_h = (landfall - issued).total_seconds() / 3600.0
    if not (0 <= delta_h <= 168):
        return None
    return int(round(delta_h))


def _resq_overlay(parsed: dict) -> dict:
    """Seed the ResQ forecast engine from a parsed bulletin's position and
    intensity. Returns the overlay dict used by both the paste and the upload
    endpoints. Never fabricates a bulletin field — it only reads what
    parse_text() already extracted."""
    overlay = {"disclaimer": DISCLAIMER, "available": False,
               "reason": "no usable position/intensity in the pasted text"}
    pos = parsed.get("current_position")
    ci = parsed.get("current_intensity") or {}
    if pos:
        import numpy as np
        wind = ci.get("max_wind_kt") or _CAT_WIND.get(ci.get("category"), 35)
        tr = parsed.get("forecast_track") or []
        dlat = dlon = 0.0
        if len(tr) >= 2:
            dlat = round(tr[1]["lat"] - tr[0]["lat"], 3)
            dlon = round(tr[1]["lon"] - tr[0]["lon"], 3)
        month = 6
        if parsed.get("issued_at"):
            try:
                month = int(parsed["issued_at"][5:7])
            except Exception:
                pass
        basin = "ARB" if pos["lon"] < 78 else "BOB"
        doy = 30 * (month - 1) + 15
        state = {
            "lat": pos["lat"], "lon": pos["lon"], "wind_kt": float(wind),
            "pressure_hpa": ci.get("pressure_hpa") if ci.get("pressure_hpa") else np.nan,
            "pressure_drop": np.nan, "wind_tendency_6h": 0.0,
            "pressure_tendency_6h": 0.0, "dlat_6h": dlat, "dlon_6h": dlon,
            "storm_age_hours": 24.0,
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
            "basin_ARB": int(basin == "ARB"), "basin_BOB": int(basin == "BOB"),
            "basin_LAND": 0, "storm_id_code": -999.0,
        }
        try:
            fc = engine().forecast(state)
            overlay = {
                "disclaimer": DISCLAIMER, "available": True,
                "from": "ResQ model, seeded from the position and intensity in "
                        "the pasted bulletin",
                "track": fc["track"], "intensity": fc["intensity"],
                "p_ri": fc["p_ri"], "analogs": fc["analogs"],
            }
        except Exception as e:                        # noqa: BLE001
            overlay = {"disclaimer": DISCLAIMER, "available": False,
                       "reason": f"forecast engine error: {e!r}"}
    return overlay


def _checklist_prefill(parsed: dict) -> dict | None:
    """Prefill for the Checklist tab — only from fields parse_text() extracted."""
    ci = parsed.get("current_intensity") or {}
    if not ci.get("category"):
        return None
    short = {"Depression": "D", "Deep Depression": "DD", "Cyclonic Storm": "CS",
             "Severe Cyclonic Storm": "SCS", "Very Severe Cyclonic Storm": "VSCS",
             "Extremely Severe Cyclonic Storm": "ESCS",
             "Super Cyclonic Storm": "ESCS"}.get(ci["category"])
    lf_raw = (parsed.get("landfall_estimate") or {}).get("time_raw")
    return {
        "intensity": short,
        "landfall_time_raw": lf_raw,
        "hours_to_landfall": _hours_to_landfall(parsed.get("issued_at"), lf_raw),
    }


@app.post("/api/bulletin/explain")
def bulletin_explain(b: BulletinText):
    import sys as _sys
    _sys.path.insert(0, str(HERE.parents[1]))
    from src.imd_bulletin import parse_text, explain

    parsed = parse_text(b.text)
    return {
        "parsed": parsed,
        "explanation": explain(parsed),
        "resq_overlay": _resq_overlay(parsed),
        "checklist_prefill": _checklist_prefill(parsed),
        "disclaimer_on_overlay_only": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Official Warnings — upload a bulletin PDF / .txt or an RSMC report.
#  The document is classified BEFORE anything is parsed. A multi-storm report is
#  never merged into one bulletin: its storms are listed with page ranges and
#  the user picks the section to parse. parse_text() / explain() are reused
#  unchanged; every "never infer a colour / district / landfall" rule still holds.
#  The upload is processed in memory and discarded — never written to disk.
# ─────────────────────────────────────────────────────────────────────────────
_UPLOAD_MSG = {
    "NO_TEXT_LAYER": "This PDF has no readable text layer — it looks scanned or "
                     "image-only. ResQ does not run OCR. Please copy the bulletin "
                     "text and paste it into the box below.",
    "NOT_A_BULLETIN": "Text was extracted, but it contains no cyclone-bulletin "
                      "markers. Nothing was parsed. If this really is a bulletin, "
                      "paste its text into the box below.",
    "MULTI_STORM_REPORT": "This looks like a multi-storm RSMC report, not a single "
                          "operational bulletin. Pick one storm below to parse just "
                          "that section — the report is not merged into one bulletin.",
    "SINGLE_BULLETIN": "Detected a single cyclone bulletin.",
}


@app.post("/api/bulletin/upload")
async def bulletin_upload(file: UploadFile = File(...),
                          storm: str | None = Form(None)):
    import sys as _sys
    _sys.path.insert(0, str(HERE.parents[1]))
    from src.document_ingest import DocumentError, extract_document
    from src.imd_bulletin import (detect_doc_type, explain, parse_text,
                                  section_text)

    data = await file.read()
    try:
        doc = extract_document(file.filename or "", data)
    except DocumentError as e:
        raise HTTPException(400, f"{file.filename or 'file'}: {e}")
    finally:
        del data

    det = detect_doc_type(doc["text"], doc["page_count"],
                          doc["chars_extracted"], doc["per_page_text"])
    doc_type = det["doc_type"]
    storms_found = det["storms_found"]

    stats = {
        "filename": file.filename,
        "source": "uploaded by user",
        "doc_type": doc_type,
        "page_count": doc["page_count"],
        "chars_extracted": doc["chars_extracted"],
        "per_page_chars": [len(re.sub(r"\s", "", t)) for t in doc["per_page_text"]],
        "detection_reason": det["reason"],
    }
    resp = {
        "doc_type": doc_type,
        "storms_found": storms_found,
        "parsed": None,
        "explanation": None,
        "overlay": None,
        "checklist_prefill": None,
        "disclaimer": DISCLAIMER,
        "message": _UPLOAD_MSG.get(doc_type, ""),
        "extraction_stats": stats,
    }

    if doc_type in ("NO_TEXT_LAYER", "NOT_A_BULLETIN"):
        return resp

    if doc_type == "MULTI_STORM_REPORT":
        if not storm:
            return resp
        chosen = next((s for s in storms_found
                       if s["name"].lower() == storm.strip().lower()), None)
        if chosen is None:
            raise HTTPException(
                404, f"'{storm}' is not one of the storms found in this report")
        sect = section_text(doc["per_page_text"],
                            chosen["first_page"], chosen["last_page"])
        parsed = parse_text(sect)
        resp.update({
            "parsed": parsed,
            "explanation": explain(parsed),
            "overlay": _resq_overlay(parsed),
            "checklist_prefill": _checklist_prefill(parsed),
            "message": f"Parsed the section for {chosen['name']} "
                       f"(pages {chosen['first_page']}–{chosen['last_page']}).",
        })
        resp["selected_storm"] = chosen
        return resp

    # SINGLE_BULLETIN — same shape as POST /api/bulletin/explain
    parsed = parse_text(doc["text"])
    resp.update({
        "parsed": parsed,
        "explanation": explain(parsed),
        "overlay": _resq_overlay(parsed),
        "checklist_prefill": _checklist_prefill(parsed),
    })
    return resp


# ─────────────────────────────────────────────────────────────────────────────
#  Chatbot — retrieval only (offline default, Groq optional rephrasing)
# ─────────────────────────────────────────────────────────────────────────────
class ChatIn(BaseModel):
    question: str
    bulletin_text: str | None = None


@app.post("/api/chat")
def chat(c: ChatIn):
    from app.backend import chat as _chat
    return _chat.answer(c.question, c.bulletin_text)


@app.get("/api/chat/status")
def chat_status():
    from app.backend import chat as _chat
    return _chat.status()


TC_SENTINEL = STATIC / "dashboard.html"


@app.get("/")
def index():
    # the TC-Sentinel UI is the shipped dashboard; the earlier tabbed prototype
    # stays reachable at /prototype
    if TC_SENTINEL.exists():
        return FileResponse(TC_SENTINEL)
    return FileResponse(STATIC / "index.html")


@app.get("/prototype")
def prototype():
    return FileResponse(STATIC / "index.html")


if (STATIC).exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
