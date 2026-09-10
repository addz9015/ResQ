"""
Build app/static/data/corpus.json — the passage store the chatbot retrieves from.

Sources (local only, no network):
  - reports/*.md                      → one passage per section (## heading)
  - app/static/data/district_history.json → one passage per district
  - app/static/data/storms.json       → one passage per 2019+ holdout storm
  - data/ndma_guidance.json           → one passage per NDMA item (empty if the
                                        guidance download failed)

The chatbot answers ONLY from these passages (plus, at query time, the bulletin
the user has pasted). Every passage keeps its source label so answers can cite
where they came from.

Run:  python scripts/build_corpus.py
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "app" / "static" / "data" / "corpus.json"


def _clean(s: str) -> str:
    s = re.sub(r"`{1,3}", "", s)
    s = re.sub(r"\|", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def from_reports():
    out = []
    for md in sorted((ROOT / "reports").glob("*.md")):
        text = md.read_text(encoding="utf-8", errors="ignore")
        # split on ## / # headings, keep heading with its body
        parts = re.split(r"\n(?=#{1,3}\s)", text)
        for part in parts:
            part = part.strip()
            if len(part) < 80:
                continue
            m = re.match(r"#{1,3}\s+(.*)", part)
            title = m.group(1).strip() if m else md.stem
            body = _clean(part)
            if len(body) > 1400:
                body = body[:1400] + " …"
            out.append({
                "id": f"report:{md.stem}:{len(out)}",
                "source": f"reports/{md.name}",
                "title": f"{md.stem} — {title}",
                "text": body,
            })
    return out


def from_district_history():
    p = ROOT / "app" / "static" / "data" / "district_history.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in d.get("districts", []):
        dec = ", ".join(f"{k}: {v}" for k, v in r["by_decade"].items())
        cats = ", ".join(f"{k} {v}" for k, v in r["by_category"].items() if v)
        names = ", ".join(f"{h['name']} ({h['year']}, {h['peak_grade']}, "
                          f"{h['closest_km']:.0f} km)" for h in r["storms"][:12])
        alias_str = (" Also spelled " + ", ".join(r["aliases"]) + "."
                     if r.get("aliases") else "")
        out.append({
            "id": f"district:{r['district']}",
            "source": "district_history.json",
            "title": f"Cyclone history — {r['district']}, {r['state']}",
            "text": _clean(
                f"{r['district']} district ({r['state']}).{alias_str} Has had "
                f"{r['n_within_100km']} tropical cyclones pass within 100 km in the "
                f"IMD archive; {r['n_within_100km_since_1990']} since 1990, of which "
                f"{r['n_severe_plus_since_1990']} were severe cyclonic storm or "
                f"stronger. By decade: {dec}. By peak category: {cats}. "
                f"Nearest-passing storms: {names}. "
                "This is a historical count, not a forecast."),
        })
    return out


def from_storms():
    p = ROOT / "app" / "static" / "data" / "storms.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for s in d.get("storms", []):
        st = s["steps"]
        out.append({
            "id": f"storm:{s['id']}",
            "source": "storms.json",
            "title": f"Storm record — {s['name']} {s['season_year']}",
            "text": _clean(
                f"{s['name']} ({s['id']}), {s['basin']}, season {s['season_year']}. "
                f"Peak wind {s['peak_wind']} kt, minimum pressure {s['min_pressure']} "
                f"hPa, peak category {s['peak_category']}. "
                f"{s['n_steps']} six-hour track points from {s['first_date']} to "
                f"{s['last_date']}. Start near {st[0]['lat']}°N {st[0]['lon']}°E, "
                f"end near {st[-1]['lat']}°N {st[-1]['lon']}°E. "
                "Held out from model training (2019+ test set)."),
        })
    return out


def from_ndma():
    p = ROOT / "data" / "ndma_guidance.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text(encoding="utf-8"))
    if d.get("status") != "ok":
        return []
    band_label = {"now": "now", "next24": "when a warning is issued",
                  "final12": "during the cyclone", "after": "after it passes"}
    out = []
    for it in d.get("items", []):
        bl = band_label.get(it.get("time_band", ""), it.get("time_band", ""))
        out.append({
            "id": f"ndma:{it['id']}",
            "source": it.get("source_name", "NDMA cyclone guidance"),
            "title": f"NDMA cyclone Do's & Don'ts — {bl}",
            "text": _clean(it["text"]),
            "source_url": it.get("source_url"),
            "retrieved_on": it.get("retrieved_on"),
        })
    return out


def main():
    passages = (from_reports() + from_district_history()
                + from_storms() + from_ndma())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "note": "Retrieval corpus for the chatbot. The bot answers only from "
                "these passages and the bulletin the user has pasted; it never "
                "adds facts and always cites its source.",
        "n_passages": len(passages),
        "by_source": {k: sum(1 for p in passages if p["source"].startswith(k) or
                             p["source"] == k)
                      for k in {p["source"] for p in passages}},
        "passages": passages,
    }, indent=1), encoding="utf-8")
    print(f"wrote {OUT}  ({len(passages)} passages)")
    for src in sorted({p["source"] for p in passages}):
        print(f"  {src:34s} {sum(1 for p in passages if p['source'] == src)}")


if __name__ == "__main__":
    main()
