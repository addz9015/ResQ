# ResQ — phase 7 (end-to-end wiring · one command)

> **Update:** the Official Warnings tab that phase 7 added was **removed** at the
> user's request. No archived IMD bulletin with a machine-parseable
> district / colour / valid-until table could be obtained (see below), the locked
> rule forbids synthesising one, and an empty tab was not worth shipping. The
> ingestion module `src/imd_bulletin.py`, its log `data/imd_bulletins/STATUS.md`,
> and the `/api/bulletins*` routes stay in the repo (routes return an empty list)
> so the tab can be re-enabled if a real bulletin source appears. The rest of
> phase 7 — end-to-end wiring, `disclaimer` fields, one-command startup,
> `reports/limitations.md` — stands.

## Nav — final

| # | tab |
|---|---|
| 00 | Dashboard |
| 01 | Forecast |
| 02 | Chatbot |

(Phase 7 briefly inserted `nav-warnings` / `page-warnings` at 02; that markup,
its `navigate()` wiring, and the `renderWarnings` / `renderOneBulletin` /
`NDMA_CHECKLIST` JS were all removed. `node --check` passes; all `getElementById`
targets resolve; div balance 86/86.)

`navigate()` updated: `navMap` + the page-hide array + `PAGE_LABELS` + a
`if (page === 'warnings') renderWarnings()` branch. All existing CSS, tokens,
class names, the sidebar, both theme toggles, the `navigate()` dispatch and the
embedded Kalpana-1 image are unchanged. Layout-only helpers stay in the appended
`<style>` block, existing tokens only. `node --check` passes; every
`getElementById` resolves; no orphan inline handlers; div balance 106/106.

## Files created

| file | what |
|---|---|
| `src/imd_bulletin.py` | IMD bulletin ingestion — schema, strict validator, RSMC-annual-report fetch + parse, `--fetch` / `--validate` / `--list`. **Never synthesises.** |
| `data/imd_bulletins/STATUS.md` | ingestion log (written; **0 bulletins** — see below) |
| `requirements.txt` | pinned: fastapi, uvicorn, pandas, pyarrow, numpy, scikit-learn, scipy, requests, pdfplumber, matplotlib |
| `requirements-era5.txt` | optional: cdsapi, xarray, netcdf4 — `src/era5.py` still fails cleanly without them |
| `scripts/make_limitations.py` → `reports/limitations.md` | one-page limitations, every number from the fragments/parquet |
| `.gitignore` | excludes the ~14 MB engine, ~6 MB replay, ~245 MB of RSMC PDFs |
| `reports/phase7.md` | this file |

`app/run.py` rewritten as a declarative artifact list — builds only what is
missing, `--rebuild` forces all, serves `127.0.0.1:8000`, interactive in <10 s
when artifacts exist.

## Endpoints (all serve pre-built JSON; no fitting at request time)

| method | path | returns | `disclaimer` field |
|---|---|---|---|
| GET | `/` | the TC-Sentinel HTML | — |
| GET | `/api/presets` | 2019+ holdout states, MANDOUS 2022 first | — |
| POST | `/api/forecast` | track + cone + `dwind_lo/hi` + P(RI) + analogs | ✅ |
| GET | `/api/replay` | holdout storm list | — |
| GET | `/api/replay/{id}` | step-by-step hindcast | ✅ |
| GET | `/api/bulletins` | **new** — parsed IMD fixtures (`[]` if none) | — (metadata) |
| GET | `/api/bulletins/{id}` | **new** — one bulletin + `resq_overlay` (P(RI), cone) | ✅ on the overlay only |
| GET | `/static/data/*.json` | storms, insights, faq | — |
| GET | `/prototype` | the earlier tabbed prototype | — |

## Were real IMD bulletins obtained? **No.**

`python src/imd_bulletin.py --fetch` ran fully:

- Downloaded **3 real, archived RSMC New Delhi annual reports** (80–84 MB each):
  - 2019: `rsmcnewdelhi.imd.gov.in/download.php?path=uploads/report/27/27_60dae9_rsmc-2018.pdf`
  - 2020: `…/uploads/report/27/27_26e77b_rsmc-2020 with damage.pdf`
  - 2022: `…/uploads/report/27/27_501da8_RSMC full report 2022 13 Jan.pdf`
  - (index: `rsmcnewdelhi.imd.gov.in/report.php?internal_menu=Mjc=`)
- Searched all 35 named 2019+ holdout storms. Found FANI, MAHA, AMPHAN, NISARGA,
  GATI, NIVAR, BUREVI, ASANI, SITRANG, MANDOUS in the reports.
- **None contains a structured district / IMD-colour / valid-until warning
  table** — the annual reports narrate the warnings, they do not tabulate them,
  and there is no per-district `valid_until`.
- IMD / RSMC New Delhi expose **no API**. The transient 3-hourly operational
  bulletins that do carry that structure were not archived machine-readably;
  web.archive.org rate-limited automated retrieval.

Per the locked design rule, **nothing was synthesised, no partial parse was
carried into the UI.** `data/imd_bulletins/STATUS.md` records every step. The
Official Warnings tab renders exactly:

> No IMD bulletins loaded. This tab shows official IMD warnings only — it does
> not generate them.

The tab is fully built for the fixtures case (bulletin selector, IMD colour
block with `Source: IMD RSMC New Delhi, issued <time>` + link, district table,
Time-to-Act shown as a **range** from the conformal timing spread, an
NDMA-sourced preparedness checklist keyed to colour/hours/zone, and the ResQ
overlay clearly separated below and labelled *"ResQ model estimate — not an
official warning."*). Drop a schema-valid JSON with a real `source_url` into
`data/imd_bulletins/` and `python src/imd_bulletin.py --validate` will accept it;
the API and tab then serve it with no code change.

**ResQ never issues its own warning colour.** IMD is the sole authority. No claim
of IMD endorsement or partnership anywhere.

## Reproduce from a clean clone

```
git clone <repo> && cd ResQ
pip install -r requirements.txt
# optional LSTM baseline:  pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# --- data + model pipeline (produces reports/_fragments/*.json) ---
python scripts/build_dataset.py
python -m src.segment_resample
python scripts/leakage_audit.py
python -m src.era5                 # writes null env columns unless CDS access added
python -m src.analogs
python -m src.uncertainty
python -m src.ri
python scripts/run_models.py       # ~10 min (LSTM + GBM CV); needs torch
python -m src.bootstrap
python -m src.ri_tune
python -m src.fit_diagnostics

# --- reports ---
python scripts/make_results.py
python scripts/make_corrections.py
python scripts/make_phase4.py
python scripts/make_phase6.py
python scripts/make_limitations.py
python scripts/make_audit.py

# --- dashboard data + IMD ingestion ---
python scripts/build_storms.py
python scripts/build_insights.py
python scripts/build_faq.py
python src/imd_bulletin.py --fetch     # downloads ~245 MB of RSMC PDFs; writes STATUS.md

# --- run (also builds engine.pkl + replay.json on first start, ~10 min) ---
python app/run.py --rebuild            # -> http://127.0.0.1:8000
```

On a clone that already carries `reports/_fragments/` and the parquets, just:
`pip install -r requirements.txt && python app/run.py` (first start builds the
engine + hindcasts; after that it is interactive in seconds).
