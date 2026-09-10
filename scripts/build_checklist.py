"""
Build the Preparation Checklist data from **genuinely retrieved** NDMA guidance.

  data/ndma_guidance.json        — every preparedness item, VERBATIM, each with
                                   source_name / source_url / retrieved_on.
  data/ndma_guidance/TEMPLATE.json — schema + one "EXAMPLE — REPLACE" entry, for
                                   hand-transcription if the fetch ever fails.
  data/ndma_guidance/README.md   — how to hand-fill it.
  data/checklist_rules.json      — (intensity, hours, location) -> item ids map.
                                   Contains NO advice text, only the mapping.
  data/ndma_guidance/STATUS.md   — what was fetched, or why nothing was.

All three JSONs are copied to app/static/data/ for the dashboard.

HARD RULE
  Only text actually returned from a named URL is stored. No preparedness item is
  ever written from the author's knowledge or drafted by an LLM. If a fetch
  fails it is reported; it is never back-filled.

NDMA serves its pages from a Drupal REST API:
  https://ndma.gov.in/node/<id>?_format=json   ->  {..., "body":[{"value":"<html>"}]}
Node 118 = "Cyclone: Do's & Dont's"  (the phased do's and don'ts)
Node  87 = "Cyclone"                 (Emergency Kit list, Recover-and-build list)

Run:  python scripts/build_checklist.py
"""
from __future__ import annotations

import datetime as dt
import html
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
GUID = ROOT / "data" / "ndma_guidance.json"
RULES = ROOT / "data" / "checklist_rules.json"
STATUSDIR = ROOT / "data" / "ndma_guidance"
CACHE = ROOT / "data" / "ndma_cache"
APPDATA = ROOT / "app" / "static" / "data"

NDMA_SITE = "https://ndma.gov.in"
NDMA_NODES = [
    # (node id, public page URL, human source name)
    (118, "https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts",
     "NDMA — Cyclone: Do's & Don'ts"),
    (87, "https://ndma.gov.in/Natural-Hazards/Cyclone",
     "NDMA — Cyclone"),
]

INTENSITY_ORDER = ["D", "DD", "CS", "SCS", "VSCS", "ESCS"]
HOURS_TO_BAND = {"72": "now", "48": "now", "24": "next24", "12": "final12", "6": "final12"}

# NDMA section heading  ->  our time band.  Deterministic, from the heading text.
HEADING_BANDS = [
    (r"before the cyclone season", "now"),
    (r"emergency kit", "now"),
    (r"when the cyclone starts", "now"),
    (r"when .*area is under cyclone warning", "next24"),
    (r"when evacuation is instructed", "next24"),
    (r"during (a|the) cyclone", "final12"),
    (r"post-?cyclone|after .*cyclone|recover and build", "after"),
]
# Sub-lists that are navigation / meta, not instructions — skipped entirely.
SKIP_HEADINGS = [r"necessary actions", r"worldwide terminology", r"classifications",
                 r"how cyclones are formed", r"indian context", r"indian meteorological",
                 r"modification and decay", r"mature tropical", r"formation and initial"]

# An item is tagged coastal-only if its text literally mentions the shore/sea.
COASTAL_RE = re.compile(r"\b(beach|beaches|coast|coastal|shore|sea|surge|fishermen|"
                        r"boat|marooned|low-?lying)\b", re.I)


def _http_json(url: str):
    import requests
    r = requests.get(url, timeout=30, headers={"User-Agent": "ResQ/1.0 (SIH build)"})
    r.raise_for_status()
    return r.json()


def _band_for(headings: list[str]) -> str | None:
    for h in reversed(headings):
        hl = h.lower()
        if any(re.search(p, hl) for p in SKIP_HEADINGS):
            return None
        for pat, band in HEADING_BANDS:
            if re.search(pat, hl):
                return band
    return None


def _plain(fragment: str) -> str:
    t = html.unescape(re.sub(r"<[^>]+>", "", fragment))
    return re.sub(r"\s+", " ", t).replace("’", "'").replace("‘", "'").strip()


def _parse_body(body_html: str, source_name: str, source_url: str, retrieved: str):
    """Walk the HTML in order; keep <li> and instructive <p> under a heading that
    maps to a band. Every returned item's text is verbatim from body_html."""
    tokens = re.findall(
        r"<h[1-6][^>]*>.*?</h[1-6]>|<p>\s*<strong>.*?</strong>\s*</p>|"
        r"<strong>.*?</strong>|<li>.*?</li>|<p>.*?</p>",
        body_html, re.S | re.I)
    headings: list[str] = []
    items = []
    seen = set()
    for tok in tokens:
        tl = tok.lower()
        is_heading = (tl.startswith("<h") or "<strong>" in tl and "<li>" not in tl
                      and len(_plain(tok)) < 120)
        if is_heading:
            txt = _plain(tok)
            if txt:
                headings.append(txt)
            continue
        band = _band_for(headings)
        if band is None:
            continue
        # candidate instruction text
        if tok.lower().startswith("<li>"):
            txt = _plain(tok)
        else:  # <p> — keep only if it reads like an instruction
            txt = _plain(tok)
            if not re.match(r"(do not|don'?t|be safe|be sure|remain|stay|listen|"
                            r"keep|switch|leave|head|pack|get |board|provide|"
                            r"ensure|avoid|report|clear|drive|move)\b", txt, re.I):
                continue
        if not (12 <= len(txt) <= 320) or txt.lower() in seen:
            continue
        seen.add(txt.lower())
        items.append({
            "id": f"ndma-{source_url.rsplit('/', 1)[-1][:6].lower()}-{len(items) + 1:02d}",
            "text": txt,
            "source_name": source_name,
            "source_url": source_url,
            "retrieved_on": retrieved,
            "time_band": band,
            "_coastal_only": bool(COASTAL_RE.search(txt)),
        })
    return items


def fetch_ndma():
    """Returns (items, notes[]).  items == [] means nothing usable was retrieved."""
    retrieved = dt.date.today().isoformat()
    items, notes = [], []
    try:
        import requests  # noqa: F401
    except Exception as e:                            # noqa: BLE001
        return [], [f"requests not importable: {e!r}"]

    CACHE.mkdir(parents=True, exist_ok=True)
    for nid, page_url, name in NDMA_NODES:
        api = f"{NDMA_SITE}/node/{nid}?_format=json"
        try:
            j = _http_json(api)
            body = (j.get("body") or [{}])[0].get("value") or ""
            (CACHE / f"node{nid}.json").write_text(json.dumps(j), encoding="utf-8")
            if not body or len(body) < 200:
                notes.append(f"(a) {api} -> 200 but body empty")
                continue
            got = _parse_body(body, name, page_url, retrieved)
            notes.append(f"(a) {api} -> {len(got)} items "
                         f"(node title {j.get('title', [{}])[0].get('value', '?')!r})")
            items.extend(got)
        except Exception as e:                        # noqa: BLE001
            notes.append(f"(a) {api} -> {e!r}")

    # renumber ids uniquely & stably
    for i, it in enumerate(items, 1):
        it["id"] = f"ndma-{i:02d}"
    return items, notes


# ─────────────────────────────────────────────────────────────────────────────
def write_template():
    STATUSDIR.mkdir(parents=True, exist_ok=True)
    (STATUSDIR / "TEMPLATE.json").write_text(json.dumps({
        "status": "ok",
        "source_note": "Every item below MUST be transcribed verbatim from a "
                       "named published source. Do not paraphrase. Do not add "
                       "items from memory.",
        "items": [
            {
                "id": "ndma-01",
                "text": "EXAMPLE — REPLACE: <the exact sentence from the source>",
                "source_name": "EXAMPLE — REPLACE: NDMA — Cyclone: Do's & Don'ts",
                "source_url": "EXAMPLE — REPLACE: https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts",
                "retrieved_on": "EXAMPLE — REPLACE: 2026-01-01",
                "time_band": "now | next24 | final12 | after",
                "_coastal_only": False,
            }
        ],
    }, indent=2), encoding="utf-8")
    (STATUSDIR / "README.md").write_text(
        "# Hand-filling `ndma_guidance.json`\n\n"
        "If `scripts/build_checklist.py` cannot retrieve NDMA guidance, a human "
        "must transcribe it:\n\n"
        "1. Open NDMA's *Cyclone: Do's & Don'ts* "
        "(https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts) and *Cyclone* "
        "(https://ndma.gov.in/Natural-Hazards/Cyclone).\n"
        "2. Copy `TEMPLATE.json` to `../ndma_guidance.json`.\n"
        "3. For **each** published bullet, add an item. `text` must be the exact "
        "sentence — no paraphrase, nothing from memory, nothing from an LLM. "
        "Fill `source_name`, `source_url`, `retrieved_on` (today's date), and "
        "`time_band` from the section it appears under:\n"
        "   - *Before the cyclone season* / *When the cyclone starts* / *Emergency "
        "Kit* -> `now`\n"
        "   - *When your area is under cyclone warning* / *When evacuation is "
        "instructed* -> `next24`\n"
        "   - *During a cyclone* -> `final12`\n"
        "   - *Post-cyclone measures* / *Recover and build* -> `after`\n"
        "4. Set `_coastal_only: true` only if the sentence literally refers to "
        "the coast/beach/sea/surge.\n"
        "5. Run `python scripts/build_checklist.py` — it validates the file, "
        "rebuilds `checklist_rules.json`, and copies both into "
        "`app/static/data/`. No code change is needed.\n",
        encoding="utf-8")


def write_rules(items):
    by_band = {"now": [], "next24": [], "final12": [], "after": []}
    coastal_only = []
    for it in items:
        by_band.setdefault(it["time_band"], []).append(it["id"])
        if it.get("_coastal_only"):
            coastal_only.append(it["id"])
    RULES.write_text(json.dumps({
        "note": "Maps (expected intensity, hours-to-arrival, location) onto NDMA "
                "guidance item ids. Contains no advice text. Every id here must "
                "exist in ndma_guidance.json or it is not rendered.",
        "intensity_order": INTENSITY_ORDER,
        "hours_to_band": HOURS_TO_BAND,
        "bands_cumulative": {
            "now": ["now"],
            "next24": ["now", "next24"],
            "final12": ["now", "next24", "final12"],
        },
        "time_bands": {"now": "Now", "next24": "Next 24 hours",
                       "final12": "Final 12 hours"},
        "map": {
            "by_band": by_band,
            "coastal_only": coastal_only,
            "min_intensity": "D",
            "min_intensity_note": "NDMA's cyclone do's & don'ts do not vary by "
                                  "storm category, so every item applies from "
                                  "Depression upward; the intensity input is "
                                  "surfaced for context and future per-item tuning "
                                  "via item_overrides.",
        },
        "item_overrides": {},
    }, indent=2), encoding="utf-8")


def _status_ok(items, notes):
    srcs = sorted({it["source_name"] for it in items})
    bands = {}
    for it in items:
        bands[it["time_band"]] = bands.get(it["time_band"], 0) + 1
    (STATUSDIR / "STATUS.md").write_text(
        "# NDMA guidance — retrieved\n\n"
        f"_Retrieved {dt.datetime.now(dt.timezone.utc).isoformat()}._\n\n"
        f"**{len(items)} preparedness items**, all verbatim, from:\n\n"
        + "".join(f"- {s}\n" for s in srcs)
        + f"\nBy band: {bands}\n\n"
        "## Fetch log\n\n" + "".join(f"- {n}\n" for n in notes)
        + "\n## Source\n\nNDMA (ndma.gov.in) serves page content from a Drupal "
        "REST API at `/node/<id>?_format=json`. Node 118 is *Cyclone: Do's & "
        "Don'ts*; node 87 is *Cyclone* (Emergency Kit, Recover). Each item stores "
        "the public page URL it corresponds to.\n",
        encoding="utf-8")


def _status_fail(notes):
    (STATUSDIR / "STATUS.md").write_text(
        "# NDMA guidance — NOT obtained\n\n"
        f"_Attempted {dt.datetime.now(dt.timezone.utc).isoformat()}._\n\n"
        "ResQ shows preparedness steps only from NDMA's published guidance and "
        "never writes them itself, so with nothing retrieved the Preparation "
        "Checklist renders an empty state that links to NDMA.\n\n"
        "## Attempts\n\n" + "".join(f"- {n}\n" for n in notes)
        + "\n## To populate by hand\n\nSee `README.md` and `TEMPLATE.json` in this "
        "folder. A hand-filled `../ndma_guidance.json` is accepted with no code "
        "change.\n",
        encoding="utf-8")


def _validate(items):
    ok = []
    for it in items:
        if not isinstance(it.get("text"), str) or len(it["text"]) < 8:
            continue
        if not it.get("source_url", "").startswith("http"):
            continue
        if not it.get("source_name") or not it.get("retrieved_on"):
            continue
        if it.get("time_band") not in ("now", "next24", "final12", "after"):
            continue
        if "EXAMPLE" in it["text"] or "REPLACE" in it["text"]:
            continue
        ok.append(it)
    return ok


def main():
    STATUSDIR.mkdir(parents=True, exist_ok=True)
    APPDATA.mkdir(parents=True, exist_ok=True)
    write_template()

    # An existing committed / hand-filled guidance file we can fall back to.
    existing = None
    if GUID.exists():
        try:
            j = json.loads(GUID.read_text(encoding="utf-8"))
            v = _validate(j.get("items", []))
            if v and j.get("status") == "ok":
                existing = (v, j.get("_source_mode", "committed"))
        except Exception:                            # noqa: BLE001
            pass

    if existing is not None and existing[1] == "hand":
        items, notes, mode = existing[0], [
            f"hand-filled data/ndma_guidance.json accepted ({len(existing[0])} items)"], "hand"
    else:
        fetched, notes = fetch_ndma()
        items, mode = _validate(fetched), "fetched"
        if not items and existing is not None:
            # fetch failed but we already have a good committed file — keep it,
            # do not overwrite with an empty "unavailable" file.
            items, mode = existing[0], existing[1]
            notes.append(f"fetch produced nothing; kept the existing "
                         f"data/ndma_guidance.json ({len(items)} items, "
                         f"mode={mode}).")

    if items:
        GUID.write_text(json.dumps({
            "status": "ok",
            "_source_mode": mode,
            "source": "NDMA — National Disaster Management Authority (ndma.gov.in)",
            "retrieved_on": dt.date.today().isoformat(),
            "n_items": len(items),
            "items": items,
        }, indent=2), encoding="utf-8")
        write_rules(items)
        _status_ok(items, notes)
        print(f"[checklist] {len(items)} NDMA items ({mode}). bands: "
              + ", ".join(sorted({i['time_band'] for i in items})))
    else:
        GUID.write_text(json.dumps({
            "status": "unavailable",
            "source": "NDMA — National Disaster Management Authority (ndma.gov.in)",
            "attempted_on": dt.date.today().isoformat(),
            "n_items": 0,
            "items": [],
        }, indent=2), encoding="utf-8")
        write_rules([])
        _status_fail(notes)
        print("[checklist] NDMA guidance NOT retrieved — empty state. "
              + " | ".join(notes)[:300])

    for f in (GUID, RULES):
        (APPDATA / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[checklist] wrote {GUID.name}, {RULES.name}, "
          f"{STATUSDIR.name}/TEMPLATE.json + README.md")


if __name__ == "__main__":
    main()
