# ResQ — phase 9 (finalise the two citizen USPs)

All changes are in the existing TC-Sentinel HTML plus backend/scripts. CSS,
tokens, class names, sidebar, theme toggles and `navigate()` are unchanged;
additions are layout-only. `node --check` passes, every `getElementById`
resolves, div balance 122/122.

## Nav (unchanged from phase 8)

| # | tab |
|---|---|
| 00 | Dashboard |
| 01 | Forecast |
| 02 | Preparation Checklist |
| 03 | District Cyclone History |
| 04 | Official Warnings *(paste-and-explain)* |
| 05 | Chatbot |

## Part 1 — removed the live IMD fetch path

| deleted | where |
|---|---|
| `GET /api/bulletin/recent` + `bulletin_recent()` | `app/backend/main.py` |
| `GET /api/bulletins`, `GET /api/bulletins/{bid}`, `_load_bulletins()` | `app/backend/main.py` (dead since phase 7) |
| `fetch_recent()`, `_extract_any()`, `--fetch-recent` CLI | `src/imd_bulletin.py` |
| `fetch()`, `_download()`, `_pdf_full_text()`, `_parse_storm_warnings()`, `holdout_storm_names()`, `RSMC_ANNUAL_REPORTS`, `RSMC_INDEX`, `--fetch` CLI | `src/imd_bulletin.py` (RSMC annual-report download machinery) |
| the "Or load a recent IMD bulletin" `<select>` + its fetch in `renderWarnings()` | the HTML |

`POST /api/bulletin/explain` is unchanged in behaviour. New Official Warnings
intro: *"Paste an official IMD cyclone bulletin below. ResQ will explain what it
says in plain language and show what our models add. ResQ does not issue
warnings."*

## Part 2 — NDMA guidance: what was actually retrieved

`ndma.gov.in` is a Drupal site with a working REST API. Outcome of each attempt:

| # | attempt | result |
|---|---|---|
| **(a)** | **Inspect the SPA for a content API.** The page markup references `/node/<id>` and `drupalSettings`; `GET /jsonapi` returns a full JSON:API index. `GET /node/118?_format=json` → **"Cyclone: Do's & Dont's"** page body (9.2 KB HTML). `GET /node/87?_format=json` → **"Cyclone"** page (Emergency Kit + Recover-and-build lists). | ✅ **51 preparedness items retrieved, verbatim**, each stored with `source_name`, `source_url`, `retrieved_on`. |
| (b) | NDMA PDF assets for cyclone Do's & Don'ts (`/sites/default/files/IEC/…`, NIDM PDF). | ❌ 404 — but (a) already yields the full text, so not needed. |
| (c) | IMD + SDMA pages as additional sources: `imdtvm.gov.in`, `osdma.org`, `apsdma.ap.gov.in`, `gsdma.org`, `mausam.imd.gov.in/…/cyclone_precautions.php`. | ❌ IMD TVM unreachable; OSDMA 301→JS page; APSDMA 200 but "Do's & Don'ts" is a nav link with no inline content; GSDMA 500; IMD precautions page 404. Nothing structured. NDMA (national authority) is the single source used. |
| (d) | Headless render of the SPA. | not attempted — no headless-render dependency is installed and the spec forbids adding one; (a) made it unnecessary. |

Source URLs recorded in every item and in `data/ndma_guidance/STATUS.md`:
- `https://ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts` (node 118) — 41 items
- `https://ndma.gov.in/Natural-Hazards/Cyclone` (node 87) — 10 items (Emergency Kit)

**Nothing was written from the author's knowledge and no LLM drafted any item.**
Section headings map deterministically to time bands:
*Before the cyclone season* / *When the cyclone starts* / *Emergency Kit* → `now`;
*When your area is under cyclone warning* / *When evacuation is instructed* → `next24`;
*During a cyclone* → `final12`; *Post-cyclone* / *Recover and build* → `after`.

`data/ndma_guidance/TEMPLATE.json` (schema + one `EXAMPLE — REPLACE` entry) and
`data/ndma_guidance/README.md` are written **regardless**, and a hand-filled
`data/ndma_guidance.json` with `"_source_mode": "hand"` is accepted with no code
change.

## Part 3 — Preparation Checklist

- Inputs: expected intensity (Depression → Extremely Severe), hours-to-arrival
  (72+ / 48 / 24 / 12 / 6), location (Coastal / Inland) — all existing `.tc-input`.
- `data/checklist_rules.json` now carries an explicit
  `map.by_band` (item ids per band) + `map.coastal_only` (items whose text
  literally names the coast/beach/sea) + `map.min_intensity`. **No advice text
  in the rules file.** The frontend renders an item only if its id exists in
  `ndma_guidance.json`.
- Output grouped **Now / Next 24 h / Final 12 h** (cumulative by hours-to-arrival)
  plus an **After it passes** group. Each item is a checkbox with
  `Source: <source_name> — <url>, retrieved <date>` beneath it in muted text.
- **Prefill from bulletin:** the Official Warnings tab's "Prefill preparation
  checklist" button passes the parsed intensity and, when the bulletin states a
  landfall day+time that can be resolved against the issue time,
  `hours_to_landfall` (snapped to the nearest dropdown option). Intensity-only
  when the landfall time is not numerically determinable — the tab says so.
- Empty state names the missing source and links it:
  *"Missing: NDMA — Cyclone: Do's & Don'ts (https://…/Dos-Donts). ResQ does not
  write safety steps itself."* — never "coming soon".

## Part 4 — Chatbot with Groq

- **Retrieval-only.** Groq only rephrases retrieved passages.
- **System prompt** now requires: answer only from the passages; say so if they
  don't contain the answer; never state a forecast / warning colour / landfall
  time / safety step not in the passages; name the source of each claim.
- **Post-generation guard** (`chat._guard`), required: discards the Groq
  response and returns the offline answer if it contains
  (i) a warning-colour phrase, (ii) a specific future time (`0230 hrs`,
  `by 12th December`, `midnight`, …), or (iii) a measured figure
  (`185 kmph`, `N km`, `N kt`, `N %`) whose numeric value is absent from every
  retrieved passage. Each discard is logged (`resq.chat` logger,
  `GUARD discarded groq generation: …`).
- **Preparedness questions** are answered **only** from NDMA passages; if none
  are retrieved the bot refuses with a link to `ndma.gov.in/Natural-Hazards/Cyclone/Dos-Donts`.
  Groq is never allowed to answer preparedness from its own knowledge (a fluent
  wrong step is the one failure that could get someone hurt).
- **Live-storm refusal** unchanged.
- **Fallback:** missing key / timeout / rate-limit / guard discard → offline
  keyword retrieval. The endpoint never returns an error; every response has a
  `mode` (`groq` / `offline` / `refusal` / `empty`) surfaced as a badge in the UI
  and sources listed under every answer.
- **Model:** `GROQ_MODEL` env var, default **`openai/gpt-oss-120b`**, then
  `llama-3.3-70b-versatile` / `llama-3.1-8b-instant` as secondaries (tried only
  if the primary errors). gpt-oss reasoning is kept out of `content`
  (`reasoning_format: "hidden"` + a `<think>` strip). **Key:** `GROQ_API_KEY`
  env only; `app/run.py` loads `.env` (git-ignored) with a tiny built-in parser —
  no `python-dotenv` dependency. `.env.example` shipped.

## Part 5 — parser test against a real bulletin

`tests/test_parser.py`:
- `test_parse_real_bulletin_extracts_core_fields` runs against
  `tests/real_bulletin.txt` and asserts `parse_text()` extracts a system name, a
  position inside the North-Indian-Ocean box, an intensity, and **≥1 district
  warning with a valid IMD colour**. **Skips with a clear message if the file is
  absent** — the synthetic fixture is not substituted.
- `test_nonsense_yields_all_null_and_fabricates_nothing` (always runs): a
  meeting-notes paragraph → every field `null`, `district_warnings == []`,
  `explain()` emits the "not stated in this bulletin" marker for each field.
- `test_empty_input_is_safe`, plus a labelled synthetic smoke test.

`tests/test_chat_guard.py`: the guard discards invented colour/time/figure,
keeps a faithful generation, live-storm is refused, preparedness is NDMA-only,
`answer()` never raises.

`pytest.ini` disables the environment's incompatible `pytest-asyncio` build
(ResQ has no async tests) so `python -m pytest tests/` runs clean.

## Files

**Created:** `pytest.ini`, `tests/test_parser.py`, `tests/test_chat_guard.py`,
`data/ndma_guidance/TEMPLATE.json`, `data/ndma_guidance/README.md`,
`reports/phase9.md`.
**Rewritten:** `scripts/build_checklist.py` (Drupal REST harvest + template +
rules map). **Edited:** `src/imd_bulletin.py` (download code removed),
`app/backend/main.py` (routes removed, `_hours_to_landfall` added),
`app/backend/chat.py` (guard, `GROQ_MODEL`, prompt, prep-only), `scripts/build_corpus.py`
(NDMA schema + intent), the TC-Sentinel HTML, `app/run.py`, `requirements.txt`,
`.gitignore`.

## VERIFY — what was run

```
python scripts/build_checklist.py    # 51 NDMA items (fetched), bands now/next24/final12/after
python scripts/build_corpus.py       # 431 passages (51 NDMA + reports + storms + districts)
python -m pytest tests/ -v           # 10 passed, 1 skipped (real_bulletin.txt absent)
python app/run.py --rebuild          # every artifact rebuilt; server on 127.0.0.1:8000
```

`POST /api/chat` — all five cases, HTTP 200, no errors:

| case | key unset | key set (`openai/gpt-oss-120b`) |
|---|---|---|
| **normal** ("how accurate is the 24 h track forecast, does any model beat the baseline?") | `offline` — limitations §2 (33/66/144 km) | **guard fired** → `offline`. gpt-oss wrote "145 km" (rounded 144→145); the figure is absent from the passages so the generation was discarded and the accurate offline answer returned. Logged: `GUARD discarded groq generation: contains a figure '145 km' absent from the passages`. |
| **live-storm** ("is a cyclone going to hit Chennai this week?") | `refusal` — fixed message | `refusal` — fixed message |
| **preparedness** ("what should I do to prepare my house before a cyclone?") | `offline` — verbatim NDMA item, NDMA sources only | **`groq`** — gpt-oss rephrases NDMA items only ("inspect your home and secure any loose tiles… turn off the electrical mains… monitor any warnings"), cites `[4] [3] [1]`, NDMA sources only |
| **not in corpus** ("what is the population of Brazil?") | `offline` — "I don't have anything … that answers that" | same |
| **also verified** | | "what is rapid intensification", "why was the background file excluded", "what does the District Cyclone History tab show", and a question about a **pasted bulletin** ("what districts does this bulletin warn, and what colour?") all returned `mode: groq` with faithful, source-cited answers. |

`node --check` passes; every `getElementById` resolves; div balance 122/122.
`git check-ignore .env` → ignored.

## Reproduce from a clean clone

```
git clone <repo> && cd ResQ
pip install -r requirements.txt
cp .env.example .env               # then paste your GROQ_API_KEY into .env
                                   # (GROQ_MODEL defaults to openai/gpt-oss-120b)

python -m pytest tests/ -v         # 10 passed, 1 skipped without tests/real_bulletin.txt
python app/run.py --rebuild        # loads .env, builds every artifact, serves 127.0.0.1:8000
                                   #   - build_district_history.py  (GADM download, ~90s)
                                   #   - build_checklist.py         (NDMA Drupal REST, 51 items)
                                   #   - build_corpus.py            (431 passages)
                                   #   - engine + hindcasts         (~5-10 min first time)

# drop a real archived IMD bulletin at tests/real_bulletin.txt to un-skip the
# real-bulletin parser test, then: python -m pytest tests/ -v
```

`.env` is git-ignored; the key never enters the repo, code, or any report.
