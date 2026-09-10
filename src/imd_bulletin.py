"""
IMD bulletin ingestion  (src/imd_bulletin.py)  —  SIH 2026 PS26070.

DESIGN RULE, LOCKED
    ResQ never issues its own warning colour. IMD is the sole authority for
    warning stage, colour and district advisories. Our models only add
    uncertainty, RI probability, analogs and the action layer *on top* of an
    IMD bulletin.

SCHEMA (one bulletin)
    {
      "id":               str,                # <storm>-<issued_at compact>
      "issued_at":        ISO-8601 str,       # exactly as printed on the bulletin
      "source_url":       str,                # where this bulletin was obtained
      "source_document":  str,                # human label of the archived source
      "system_name":      str,
      "imd_storm_id":     str | null,         # link to the 2019+ holdout, if known
      "current_position": {"lat": float, "lon": float, "as_of": ISO str},
      "current_intensity":{"category": str, "max_wind_kt": float|null,
                           "pressure_hpa": float|null},
      "forecast_track":   [{"lead_h": int, "lat": float, "lon": float,
                            "category": str|null}],
      "district_warnings":[{"district": str, "state": str|null, "colour": str,
                            "hazards": [str], "valid_from": ISO str|null,
                            "valid_until": ISO str}],
      "landfall_estimate":{"time": ISO str|null, "place": str|null,
                           "coast": str|null} | null
    }

HARD RULE
    Nothing is ever fetched. `parse_text` extracts only what is literally in the
    pasted text; any field it cannot find is null. It never synthesises a
    bulletin, infers a warning colour / district / landfall time, or carries a
    partial parse forward. A hand-added JSON in data/imd_bulletins/ must pass the
    strict `validate()` (a real district warning with a real colour and
    valid_until) or it is ignored.

USAGE
    python src/imd_bulletin.py --explain FILE   # parse + explain a bulletin text file
    python src/imd_bulletin.py --validate       # re-check every hand-added JSON
    python src/imd_bulletin.py --list           # print what is currently loaded
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "imd_bulletins"          # hand-added schema-valid bulletins

VALID_COLOURS = {"green", "yellow", "orange", "red"}
DISCLAIMER = "ResQ model estimate — not an official warning."

# There is NO automated download path. IMD / RSMC New Delhi expose no API and the
# transient operational bulletins are not archived machine-readably. The feature
# is: the user pastes bulletin text, parse_text() extracts it deterministically,
# explain() restates it. A hand-added schema-valid JSON in data/imd_bulletins/
# is still accepted (validate() / --validate), but nothing is ever fetched.


# ---------------------------------------------------------------------------
# schema validation — strict; a record either passes fully or is rejected
# ---------------------------------------------------------------------------
def _iso(s):
    try:
        dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return True
    except Exception:
        return False


def validate(rec: dict) -> list[str]:
    e = []
    for k in ("id", "issued_at", "source_url", "source_document", "system_name",
              "current_position", "current_intensity", "forecast_track",
              "district_warnings"):
        if k not in rec:
            e.append(f"missing key: {k}")
    if e:
        return e
    if not _iso(rec["issued_at"]):
        e.append("issued_at is not ISO-8601")
    if not str(rec["source_url"]).startswith("http"):
        e.append("source_url is not a URL")
    cp = rec["current_position"]
    if not (isinstance(cp, dict) and _is_num(cp.get("lat")) and _is_num(cp.get("lon"))
            and _iso(cp.get("as_of"))):
        e.append("current_position incomplete")
    dw = rec["district_warnings"]
    if not isinstance(dw, list) or not dw:
        e.append("district_warnings must be a non-empty list "
                 "(a bulletin with no real district warning is not usable)")
    else:
        for i, w in enumerate(dw):
            if not isinstance(w, dict):
                e.append(f"district_warnings[{i}] not an object"); continue
            if not w.get("district"):
                e.append(f"district_warnings[{i}].district missing")
            if str(w.get("colour", "")).lower() not in VALID_COLOURS:
                e.append(f"district_warnings[{i}].colour not one of {sorted(VALID_COLOURS)}")
            if not _iso(w.get("valid_until")):
                e.append(f"district_warnings[{i}].valid_until missing/invalid")
            if not (isinstance(w.get("hazards"), list) and w["hazards"]):
                e.append(f"district_warnings[{i}].hazards must be a non-empty list")
    return e


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ---------------------------------------------------------------------------
def load_all() -> list[dict]:
    if not OUT.exists():
        return []
    recs = []
    for p in sorted(OUT.glob("*.json")):
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not validate(r):
            r.setdefault("id", p.stem)
            recs.append(r)
    return recs


def cmd_validate():
    n_ok = n_bad = 0
    for p in sorted(OUT.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        errs = validate(r)
        if errs:
            n_bad += 1
            print(f"INVALID {p.name}:\n  " + "\n  ".join(errs))
        else:
            n_ok += 1
            print(f"OK      {p.name}  ({len(r['district_warnings'])} district warnings)")
    print(f"\n{n_ok} valid, {n_bad} invalid")
    return 0 if n_bad == 0 else 1


# ═══════════════════════════════════════════════════════════════════════════
#  PASTE-AND-EXPLAIN  —  parse_text(raw) + explain(parsed)
#
#  parse_text: deterministic, regex / rule based. NO LLM, NO inference. Any
#  field not literally in the text is null. It never invents a colour, a
#  district or a landfall time.
#  explain:    rephrases the already-extracted fields in plain English. It adds
#              no facts; a null field becomes "not stated in this bulletin".
# ═══════════════════════════════════════════════════════════════════════════

_GRADE_WORDS = [
    (r"super cyclonic storm", "Super Cyclonic Storm"),
    (r"extremely severe cyclonic storm", "Extremely Severe Cyclonic Storm"),
    (r"very severe cyclonic storm", "Very Severe Cyclonic Storm"),
    (r"severe cyclonic storm", "Severe Cyclonic Storm"),
    (r"cyclonic storm", "Cyclonic Storm"),
    (r"deep depression", "Deep Depression"),
    (r"depression", "Depression"),
]
_COMPASS = (r"(?:north|south|east|west|north-?east|north-?west|south-?east|"
            r"south-?west|north-?north-?east|east-?north-?east)")


def _kmph_to_kt(kmph: float) -> float:
    return round(kmph * 0.539957, 1)


def _first(pattern, text, flags=re.I, group=1):
    m = re.search(pattern, text, flags)
    return m.group(group).strip() if m else None


def _mk_iso_from_ddmmyyyy(d, mo, y, hh="00", mm="00", tz="UTC"):
    try:
        y = int(y);  y += 2000 if y < 100 else 0
        return (f"{y:04d}-{int(mo):02d}-{int(d):02d}T{int(hh):02d}:{int(mm):02d}:00"
                + ("Z" if tz.upper() == "UTC" else ""))
    except Exception:                                # noqa: BLE001
        return None


def _parse_issued_at(text):
    # "BASED ON 0300 UTC OF 09.12.2022"  /  "0300 UTC of 09th December 2022"
    m = re.search(r"based on\s+(\d{2})(\d{2})\s*(utc|ist)\s+of\s+(\d{1,2})[.\-/]"
                  r"(\d{1,2})[.\-/](\d{2,4})", text, re.I)
    if m:
        return _mk_iso_from_ddmmyyyy(m.group(4), m.group(5), m.group(6),
                                     m.group(1), m.group(2), m.group(3)), m.group(0)
    months = ("january february march april may june july august september "
              "october november december").split()
    m = re.search(r"(\d{2})(\d{2})\s*(utc|ist)\s+of\s+(\d{1,2})\w*\s+"
                  r"(" + "|".join(months) + r")\s+(\d{4})", text, re.I)
    if m:
        mo = months.index(m.group(5).lower()) + 1
        return _mk_iso_from_ddmmyyyy(m.group(4), mo, m.group(6),
                                     m.group(1), m.group(2), m.group(3)), m.group(0)
    m = re.search(r"issued (?:at )?(\d{2})(\d{2})\s*hours?\s*(ist|utc)?\s*"
                  r"of\s+(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", text, re.I)
    if m:
        return _mk_iso_from_ddmmyyyy(m.group(3), m.group(4), m.group(5),
                                     m.group(1), m.group(2),
                                     m.group(6) or "UTC"), m.group(0)
    return None, None


def _parse_position(text):
    m = re.search(r"lat(?:itude)?\.?\s*([0-9]{1,2}\.?[0-9]?)\s*[°ºo]?\s*n"
                  r".{0,40}?long(?:itude)?\.?\s*([0-9]{2,3}\.?[0-9]?)\s*[°ºo]?\s*e",
                  text, re.I | re.S)
    if not m:
        m = re.search(r"near\s+([0-9]{1,2}\.[0-9])\s*/\s*([0-9]{2,3}\.[0-9])", text, re.I)
    if m:
        raw = m.group(0).strip()
        return {"lat": float(m.group(1)), "lon": float(m.group(2))}, raw
    return None, None


def _parse_intensity(text):
    cat = raw = None
    for pat, label in _GRADE_WORDS:
        m = re.search(pat, text, re.I)
        if m:
            cat, raw = label, m.group(0)
            break
    wind_kt = wind_raw = None
    m = re.search(r"(?:maximum sustained|max\.?)\s*wind\s*speed\s*of\s*"
                  r"(\d{2,3})(?:\s*[-–]\s*(\d{2,3}))?\s*(kmph|km/h|kt|knots)", text, re.I)
    if m:
        lo, hi, unit = m.group(1), m.group(2), m.group(3).lower()
        val = (int(lo) + int(hi)) / 2 if hi else int(lo)
        wind_kt = _kmph_to_kt(val) if unit.startswith("km") else float(val)
        wind_raw = re.sub(r"\s+", " ", m.group(0))
    pres = pres_raw = None
    m = re.search(r"central pressure\s*(?:is|of|:)?\s*(\d{3,4})\s*(?:hpa|mb|hectopascal)",
                  text, re.I)
    if m:
        pres, pres_raw = float(m.group(1)), re.sub(r"\s+", " ", m.group(0))
    if not any([cat, wind_kt, pres]):
        return None, None
    return ({"category": cat, "max_wind_kt": wind_kt, "pressure_hpa": pres},
            {"category": raw, "wind": wind_raw, "pressure": pres_raw})


def _parse_movement(text):
    m = re.search(r"mov(?:ed|ing)\s+((?:nearly\s+)?[a-z]+(?:-[a-z]+)*wards?)"
                  r"(?:\s+with a speed of\s+(\d{1,3})\s*(?:kmph|km/h))?", text, re.I)
    if m:
        return {"direction": m.group(1).strip(),
                "speed_kmph": (int(m.group(2)) if m.group(2) else None)}, m.group(0).strip()
    return None, None


def _parse_forecast_track(text):
    """The 'Forecast track and intensity' table. Rows look like
       09-12/0300  11.5/82.0  65   (dd-mm/HHMM  lat/lon  maxwind)."""
    rows = []
    for m in re.finditer(r"(\d{1,2})[-/.](\d{1,2})\s*[/ ]\s*(\d{4})\s+"
                         r"([0-9]{1,2}\.[0-9])\s*[/ ]\s*([0-9]{2,3}\.[0-9])\s+"
                         r"(\d{2,3})", text):
        rows.append({
            "when_raw": f"{m.group(1)}-{m.group(2)}/{m.group(3)} UTC",
            "lat": float(m.group(4)), "lon": float(m.group(5)),
            "max_wind_kmph": int(m.group(6)),
        })
    return rows or None


# A colour word within a short span of "warning/alert/message/code", followed
# somewhere before the sentence end by a run of Title-Case district names. The
# district run must come after an explicit "for/of" cue so we don't grab a
# region phrase.
_DISTRICT_COLOUR_RE = re.compile(
    r"\b(red|orange|yellow|green)\b"
    r"(?:\s+(?:message|warning|alert|colou?r\s*code)\b)"
    r"[^.\n]{0,120}?"
    r"\b(?:for|of|over)\s+"
    r"(?:the\s+)?(?:districts?\s+of\s+)?"
    r"((?:[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2}"
    r"(?:\s*,\s*|\s+and\s+|\s*/\s*))+[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+){0,2})",
    re.I | re.M)

_FILLER = re.compile(r"^(the|districts?\s+of|district|force|is\s+in\s+force|in\s+force|"
                     r"be\s+prepared|take\s+action)\s+", re.I)


def _parse_district_warnings(text):
    """Only when a colour word AND district name(s) are LITERALLY present in the
    same clause. Otherwise []. Never inferred."""
    out = []
    for m in _DISTRICT_COLOUR_RE.finditer(text):
        colour = m.group(1).lower()
        chunk = re.sub(r"\s+", " ", m.group(2)).strip(" .,;&")
        # strip trailing 'district(s)/of Tamil Nadu' style tails
        chunk = re.sub(r"\s+(districts?|of\s+[A-Z][A-Za-z ]+)$", "", chunk, flags=re.I)
        valid_until = None
        vm = re.search(r"valid\s+(?:till|until|upto|up to)\s+"
                       r"([0-9]{3,4}\s*(?:hrs?|hours?)?\s*(?:ist|utc)?"
                       r"(?:\s+of\s+[0-9A-Za-z ]{4,25})?)",
                       text[m.start():m.start() + 400], re.I)
        if vm:
            valid_until = re.sub(r"\s+", " ", vm.group(1)).strip()
        for part in re.split(r"\s*,\s*|\s+and\s+|\s*/\s*", chunk):
            name = _FILLER.sub("", part).strip(" .,;&")
            name = re.sub(r"\s+districts?(\s+of)?$", "", name, flags=re.I).strip()
            if 3 <= len(name) <= 40 and re.match(r"[A-Z]", name) and \
               name.lower() not in ("tamil nadu", "andhra pradesh", "west bengal",
                                    "be prepared", "take action"):
                out.append({
                    "district": name, "state": None, "colour": colour,
                    "hazards": [], "valid_from": None, "valid_until": valid_until,
                    "raw_phrase": re.sub(r"\s+", " ", m.group(0)).strip(),
                })
    seen, uniq = set(), []
    for w in out:
        k = (w["district"].lower(), w["colour"])
        if k not in seen:
            seen.add(k)
            uniq.append(w)
    return uniq


def _parse_landfall(text):
    m = re.search(r"cross(?:ing|es)?\s+(?:the\s+)?([A-Za-z ,&]+?\s+coasts?)"
                  r"([^.\n]{0,150}?)\b(?:around|by|on|near|during)\s+"
                  r"([^.\n]{4,80}?)(?:\s+as a\b|\.|\n)", text, re.I)
    if not m:
        return None, None
    place = re.sub(r"\s+", " ", m.group(1)).strip(" ,&")
    between = re.sub(r"\s+", " ", m.group(2)).strip(" ,&")
    when_raw = re.sub(r"\s+", " ", m.group(3)).strip()
    loc = f"{place} ({between})" if between.lower().startswith("between") else place
    return ({"time_raw": when_raw, "place": loc, "coast": place},
            re.sub(r"\s+", " ", m.group(0)).strip())


def parse_text(raw: str) -> dict:
    """Deterministic extraction from pasted IMD bulletin text.
    Fields absent from the text are returned as null. Nothing is inferred."""
    text = (raw or "").replace(" ", " ")
    flat = re.sub(r"[ \t]+", " ", text)
    oneline = re.sub(r"\s+", " ", text)          # newlines collapsed too

    issued_at, issued_raw = _parse_issued_at(flat)
    name = _first(r"['\"‘’]([A-Z][A-Z \-]{2,20})['\"‘’]", oneline)
    grade = None
    for pat, label in _GRADE_WORDS:
        if re.search(pat, flat, re.I):
            grade = label
            break
    system_name = None
    if name and grade:
        system_name = f"{grade} '{name.strip()}'"
    elif name:
        system_name = name.strip()
    elif grade:
        system_name = grade

    pos, pos_raw = _parse_position(oneline)
    inten, inten_raw = _parse_intensity(oneline)
    move, move_raw = _parse_movement(oneline)
    track = _parse_forecast_track(text)
    dws = _parse_district_warnings(text)
    landfall, landfall_raw = _parse_landfall(oneline)

    return {
        "source": "text pasted by user",
        "parsed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "issued_at": issued_at,
        "issued_at_raw": issued_raw,
        "system_name": system_name,
        "system_name_raw": (f"{name} / {grade}" if name or grade else None),
        "current_position": pos,
        "current_position_raw": pos_raw,
        "current_intensity": inten,
        "current_intensity_raw": inten_raw,
        "movement": move,
        "movement_raw": move_raw,
        "forecast_track": track,
        "district_warnings": dws,          # [] unless a colour+district was literal
        "landfall_estimate": landfall,
        "landfall_estimate_raw": landfall_raw,
        "unparsed_note": ("No per-district IMD colour code was found in this text — "
                          "district_warnings is empty. That is expected for RSMC "
                          "track bulletins; colour codes live in the state-level "
                          "warning bulletins."
                          if not dws else None),
    }


_NULL = "not stated in this bulletin"


def explain(parsed: dict) -> dict:
    """Plain-English restatement of fields ALREADY in `parsed`. Adds no facts."""
    p = parsed or {}

    def g(*keys):
        cur = p
        for k in keys:
            if not isinstance(cur, dict) or cur.get(k) is None:
                return None
            cur = cur[k]
        return cur

    lines = {}
    lines["issued_at"] = (f"The bulletin was issued at {p['issued_at']}."
                          if p.get("issued_at") else _NULL)
    lines["system_name"] = (f"The system is {p['system_name']}."
                            if p.get("system_name") else _NULL)

    if g("current_position"):
        lines["current_position"] = (
            f"Its centre was near {p['current_position']['lat']}°N, "
            f"{p['current_position']['lon']}°E.")
    else:
        lines["current_position"] = _NULL

    ci = p.get("current_intensity")
    if ci:
        bits = []
        if ci.get("category"):
            bits.append(f"classified as a {ci['category']}")
        if ci.get("max_wind_kt"):
            bits.append(f"maximum sustained wind about {ci['max_wind_kt']:.0f} kt")
        if ci.get("pressure_hpa"):
            bits.append(f"central pressure {ci['pressure_hpa']:.0f} hPa")
        lines["current_intensity"] = ("It was " + ", ".join(bits) + "."
                                      if bits else _NULL)
    else:
        lines["current_intensity"] = _NULL

    mv = p.get("movement")
    lines["movement"] = (
        (f"It was moving {mv['direction']}"
         + (f" at about {mv['speed_kmph']} km/h." if mv.get("speed_kmph") else "."))
        if mv else _NULL)

    tr = p.get("forecast_track")
    if tr:
        lines["forecast_track"] = (
            f"The bulletin gives a {len(tr)}-point forecast track, ending near "
            f"{tr[-1]['lat']}°N, {tr[-1]['lon']}°E ({tr[-1]['when_raw']}).")
    else:
        lines["forecast_track"] = _NULL

    dws = p.get("district_warnings") or []
    if dws:
        by_c = {}
        for w in dws:
            by_c.setdefault(w["colour"], []).append(w["district"])
        lines["district_warnings"] = "; ".join(
            f"{c.upper()} for {', '.join(sorted(set(ds)))}" for c, ds in by_c.items())
    else:
        lines["district_warnings"] = (
            "No district-level IMD colour warning is stated in this text. "
            "For district colour codes and advisories, see the IMD state "
            "meteorological centre bulletin.")

    lf = p.get("landfall_estimate")
    if lf:
        lines["landfall_estimate"] = (
            f"Landfall is expected near {lf.get('place') or 'the coast'} "
            f"around {lf.get('time_raw') or lf.get('time') or 'a time not stated'}.")
    else:
        lines["landfall_estimate"] = _NULL

    return {
        "disclaimer": DISCLAIMER,
        "source": p.get("source", "text pasted by user"),
        "issued_at": p.get("issued_at"),
        "lines": lines,
    }


def cmd_explain(path):
    raw = pathlib.Path(path).read_text(encoding="utf-8")
    p = parse_text(raw)
    print(json.dumps(p, indent=2, ensure_ascii=False))
    print("\n--- explain ---")
    print(json.dumps(explain(p), indent=2, ensure_ascii=False))
    return 0


def main():
    ap = argparse.ArgumentParser(description="IMD bulletin — paste/parse/explain "
                                             "(no network).")
    ap.add_argument("--validate", action="store_true",
                    help="re-check every hand-added JSON in data/imd_bulletins/")
    ap.add_argument("--list", action="store_true", help="print loaded bulletins")
    ap.add_argument("--explain", metavar="FILE",
                    help="parse + explain a bulletin text file")
    a = ap.parse_args()
    if a.validate:
        sys.exit(cmd_validate())
    if a.explain:
        sys.exit(cmd_explain(a.explain))
    if a.list:
        for r in load_all():
            print(f"{r['id']:14s}  {r['system_name']:12s}  "
                  f"{len(r['district_warnings'])} warnings  src={r.get('source_url','?')}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
