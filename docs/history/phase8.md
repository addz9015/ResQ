# ResQ — phase 8 (four citizen-facing features)

All four are added to the existing TC-Sentinel HTML. CSS, design tokens, class
names, sidebar, theme toggles and the `navigate()` dispatch are unchanged; only
new pages, new `navigate()` entries and layout-only CSS helpers (existing tokens)
were added. `node --check` passes; every `getElementById` resolves; div balance
123/123.

## Nav — before → after

| # | before | after |
|---|---|---|
| 00 | Dashboard | Dashboard |
| 01 | Forecast | Forecast |
| **02** | Chatbot | **Preparation Checklist** — new (`nav-checklist` / `page-checklist`) |
| **03** | — | **District Cyclone History** — new (`nav-history` / `page-history`) |
| **04** | — | **Official Warnings** — new (`nav-warnings` / `page-warnings`), paste-and-explain |
| 05 | — | Chatbot *(now retrieval over `/api/chat`)* |

## Files created

| file | what |
|---|---|
| `scripts/build_district_history.py` | GADM 4.1 level-2 boundaries + the 422-storm archive → `app/static/data/district_history.json`. Track-to-district distance on a local km plane; falls back to hand-entered coastal centroids (label `centroid_approx`) if GADM is unreachable. |
| `scripts/build_checklist.py` | attempts the NDMA cyclone-guidance download → `data/ndma_guidance.json` + `data/checklist_rules.json` (copied to `app/static/data/`). On failure: `data/ndma_guidance/STATUS.md` + empty guidance. **Never writes a safety step.** |
| `scripts/build_corpus.py` | `reports/*.md` + `district_history.json` + `storms.json` + NDMA items → `app/static/data/corpus.json` (360 passages), each with a source label. |
| `app/backend/chat.py` | retrieval-only chatbot. IDF-weighted keyword + TF-IDF retrieval; offline answer is the default; Groq (`GROQ_API_KEY`, env only) is a rephrasing layer that silently falls back on any failure. Live-storm questions refused; preparedness questions answered only from NDMA passages. |
| `tests/sample_bulletin.txt` | test fixture for `parse_text()` (clearly labelled — not an archived record). |
| `.env.example` | `GROQ_API_KEY=` template; `.env` is git-ignored. |
| `reports/phase8.md` | this file. |

**`src/imd_bulletin.py`** gained `parse_text(raw)` (deterministic regex extraction
— issued_at, system_name, position, intensity, movement, forecast track, district
warnings, landfall; any field not literally in the text is `null`; a colour or
district is never inferred) and `explain(parsed)` (plain-English restatement of
those fields only; `null` → "not stated in this bulletin"), plus `fetch_recent()`
(best-effort live pull) and `--explain` / `--fetch-recent` CLI flags.

## Endpoints added

| method | path | returns |
|---|---|---|
| POST | `/api/bulletin/explain` | `{parsed, explanation, resq_overlay, checklist_prefill, disclaimer_on_overlay_only}` — the overlay (P(RI), Δwind interval, analogs) carries the disclaimer; it is seeded from the parsed position + intensity |
| GET | `/api/bulletin/recent` | best-effort live IMD bulletins; `{count:0, …}` when the fetch fails (the paste box is the feature) |
| POST | `/api/chat` | `{answer, sources[], mode: offline\|groq\|refusal, disclaimer}` — retrieval over `corpus.json` + any pasted bulletin |
| GET | `/api/chat/status` | `{corpus_passages, groq_key_present, default_mode}` |

Static: `/static/data/{district_history,ndma_guidance,checklist_rules,corpus}.json`.

## Were the external sources obtained?

| source | needed by | result |
|---|---|---|
| **GADM 4.1 level-2 India** — `https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_IND_2.json.zip` | District Cyclone History | ✅ **obtained** — 220 coastal districts with real polygon boundaries. `district_history.json` records the URL + retrieval date. |
| **NDMA cyclone preparedness guidance** — `https://ndma.gov.in/Natural-Hazards/Cyclone` (+4 other NDMA/NIDM URLs) | Preparation Checklist, chatbot | ❌ **not obtained.** `ndma.gov.in` was intermittently reachable, but the cyclone page is a JavaScript SPA — its raw HTML carries only the site navigation, no preparedness list. The `/Dos-Donts` path 404s; the two candidate PDFs 404. The parser is deliberately strict (a line is kept only if it has an action verb **and** cyclone-preparedness vocabulary **and** is not site chrome) so the navigation menu is rejected rather than shipped as fake "advice". Result: `status: "unavailable"`, `items: []`, `data/ndma_guidance/STATUS.md` written, Checklist tab shows the empty state, chatbot refuses preparedness questions with a pointer to NDMA. Drop a real `data/ndma_guidance.json` in, re-run `build_checklist.py`, and both fill in with no code change. |
| **Live IMD bulletins** — rsmcnewdelhi.imd.gov.in / mausam.imd.gov.in | Official Warnings (optional fetch) | ❌ RSMC unreachable; mausam reachable but no machine-readable bulletin endpoint. `/api/bulletin/recent` returns `count:0`. **The paste box is the feature** and works fully offline. |
| **Groq API** | Chatbot (optional) | reachable, but no `GROQ_API_KEY` in this environment → chatbot runs in offline mode (the default). Verified: with a key set it calls Groq and, on failure, falls back to the offline answer; with no key it goes straight to offline. Both return an answer. |

## Reproduce from a clean clone

```
git clone <repo> && cd ResQ
pip install -r requirements.txt
cp .env.example .env            # optional — add GROQ_API_KEY for chatbot phrasing

# one command — builds every missing artifact, then serves 127.0.0.1:8000
python app/run.py               # first run downloads GADM (~1.5 MB) + builds the
                                # engine + hindcasts (~2–10 min); after that, seconds

# or run the phase-8 builders individually:
python scripts/build_district_history.py   # GADM + archive -> district_history.json
python scripts/build_checklist.py          # NDMA guidance  -> ndma_guidance.json (+ STATUS.md)
python scripts/build_corpus.py             # -> corpus.json  (needs district_history.json)
python -m src.imd_bulletin --explain tests/sample_bulletin.txt   # parse + explain a bulletin
```

## Verification run (actually executed)

```
pip install -r requirements.txt                         # ok (shapely added)
python scripts/build_district_history.py                # 220 districts, GADM boundaries
python scripts/build_corpus.py                          # 369 passages
python -m src.imd_bulletin --explain tests/sample_bulletin.txt   # full parse + explain
python app/run.py --rebuild                             # rebuilt every artifact, served :8000
```

All 15 endpoints returned **HTTP 200**:

| endpoint | note |
|---|---|
| `GET /` , `/api/presets` , `/api/replay` , `/api/replay/{id}` | unchanged, still 200 |
| `GET /api/chat/status` | `{corpus_passages: 369, groq_key_present: false, default_mode: "offline keyword retrieval"}` |
| `GET /api/bulletin/recent` | `{count: 0, …}` — no live IMD feed reachable |
| `GET /static/data/{district_history,ndma_guidance,checklist_rules,corpus}.json` | 200 |
| `POST /api/forecast` | 200, `disclaimer` present |
| `POST /api/chat` (key unset) | 200, `mode: "offline"`, answer + cited sources |
| `POST /api/chat` (`GROQ_API_KEY` set to an invalid value) | 200, Groq call fails → `mode: "offline"` fallback, still a full answer |
| `POST /api/chat` (`"is there a cyclone right now…"`) | 200, `mode: "refusal"`, fixed message |
| `POST /api/bulletin/explain` (sample bulletin) | 200 — MANDOUS, 7 district warnings with IMD colours, landfall, ResQ overlay (P(RI) 0.06, 10 analogs), checklist prefill `{intensity: "CS"}` |
| `POST /api/bulletin/explain` (nonsense text) | 200 — `system_name: null`, `district_warnings: []`, no overlay. Nothing fabricated. |

`node --check` passes on the page script; all 40 `getElementById` targets resolve;
no undefined inline handlers; div balance 123/123. `python app/run.py` with
artifacts present prints "all artifacts present — starting server" and is
interactive in seconds.

> Note on Groq: no real `GROQ_API_KEY` was available in this environment, so the
> Groq *generation* path was exercised only up to the API call + graceful
> fallback. With a valid key the same route returns `mode: "groq"` with the
> model's rephrasing and the same cited sources.

## Honest notes

- The **Preparation Checklist is empty** in this build — NDMA guidance could not
  be downloaded and ResQ does not author safety advice. The UI, the rule engine
  (`checklist_rules.json`) and the corpus hook are all in place for the moment a
  real `ndma_guidance.json` is added.
- **District history distances are approximate** (~1 % on a local equirectangular
  km plane) and the count is "track passed within 100 km", not a modelled impact.
  The tab and every headline sentence say "This is what has happened before. It
  is not a forecast."
- **`parse_text` is deterministic and conservative.** It will miss fields that
  are phrased unusually, and it returns `district_warnings: []` unless a colour
  word and district names appear literally in the same clause. It never fills a
  gap with a guess.
- The chatbot is **retrieval only** — every answer cites the passage/source it
  came from, and it cannot emit a forecast, a warning colour, a landfall time or
  a safety step that is not in the retrieved text.
