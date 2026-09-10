# ResQ — cyclone track & intensity forecasting + evaluation (SIH 2026, PS26070)

ResQ takes IMD/RSMC New Delhi best-track data for the North Indian Ocean, cleans
it onto a uniform 6-hour grid, and builds **track and intensity forecasts that
are honestly compared against strong baselines** — with conformal uncertainty, an
analog-retrieval engine, a rapid-intensification probability, a cluster
bootstrap for significance, and a browser dashboard that replays every 2019+
holdout storm step by step. It also has a citizen layer: an NDMA-sourced
preparedness checklist, a per-district cyclone-history view built from the
425-storm archive, a tool that explains a pasted IMD bulletin in plain language,
and a retrieval-only chatbot.

**ResQ does not issue official cyclone warnings.** IMD (RSMC New Delhi) is the
sole authority for warning stage, colour and district advisories. Every model
output in ResQ is labelled *"ResQ model estimate — not an official warning."*

## What the evidence says

Read these four, in order — they are the current state of the project:

| file | what it is |
|---|---|
| [`reports/audit.md`](reports/audit.md) | **data provenance** — null table, lag/tendency recomputation, gap-based segmentation + 6 h resampling, the storm-disjoint splits, and why `normal_weather_background.csv` is excluded (a classifier separates it from real data with ROC-AUC 1.00) |
| [`reports/results.md`](reports/results.md) | the model comparison on 2019+ test storms — primary/secondary tables, interpolation-sensitivity re-score, marginal vs split-CQR vs cross-conformal coverage, rapid intensification |
| [`reports/significance.md`](reports/significance.md) | 1000-replicate **cluster bootstrap by storm**; 95% CIs and paired-difference significance for every headline number |
| [`reports/limitations.md`](reports/limitations.md) | one page: no ERA5, no learned track model beats persistence-of-motion, 24 h intensity intervals under-cover, only 6 super-cyclones in the archive, ~12 % interpolated wind targets, what the citizen features can and cannot do |

**Headline (2019+ test, ~70 storms):** no learned track model beats the motion
baselines — most are significantly worse. On intensity, the analog/GBM point
estimates edge Δ=0 at 12/24 h but not significantly. The one result that holds at
95 %: the **combined P(RI)** (calibrated analog + classifier, ROC-AUC ≈ 0.84)
beats both climatology and the classifier alone. Real intensity/RI skill needs
environmental fields (the ERA5 fetch layer is built but CDS access was
unavailable).

The phase-by-phase working notes — how each of the above was corrected over
time — are in [`docs/history/`](docs/history/).

## Clean-clone reproduction

```bash
git clone <repo> && cd ResQ
pip install -r requirements.txt          # torch is optional (LSTM baseline only)

# optional — Groq for the chatbot (works fully offline without it):
cp .env.example .env                     # paste your GROQ_API_KEY into .env

python -m pytest tests/ -v               # 17 pass, 1 skips (needs tests/real_bulletin.txt)

python app/run.py --rebuild              # builds every gitignored artifact from the
                                         # committed CSVs + reports/_fragments/, then
                                         # serves http://127.0.0.1:8000 and opens it
```

`python app/run.py` (no `--rebuild`) builds only what is missing, then serves.
First build fetches GADM district boundaries (~1.5 MB) and the NDMA guidance
pages, and trains the forecast engine + 76 replay hindcasts (~5–10 min); after
that it is interactive in seconds.

**Open the dashboard through the server** at `http://127.0.0.1:8000` — not by
opening the `.html` file directly (its panels load data from `/api/*` and
`/static/data/*.json`).

The full research pipeline (the parts `app/run.py` does *not* rebuild — LSTM
training, the bootstrap, etc., which produce `reports/_fragments/*.json`) is the
command block at the end of [`reports/audit.md`](reports/audit.md).

## Repository layout

| path | holds |
|---|---|
| `src/` | the model code — `segment_resample`, `analogs`, `uncertainty` (split-CQR + cross-conformal), `ri` (combined P(RI)), `bootstrap`, `fit_diagnostics`, `imd_bulletin` (paste-and-explain), `era5`, `common` |
| `scripts/` | pipeline drivers (`build_dataset`, `run_models`, `leakage_audit`, `splits`) and builders for the dashboard payloads (`build_storms`, `build_insights`, `build_faq`, `build_district_history`, `build_checklist`, `build_corpus`) and the reports (`make_*`) |
| `app/` | FastAPI backend (`app/backend/`), the retrieval chatbot (`app/backend/chat.py`), `app/precompute.py` (trains the engine + hindcasts), `app/run.py` (one-command startup), and the dashboard HTML in `app/static/` |
| `tests/` | `pytest` — parser tests and the chatbot-guard tests |
| `reports/` | the load-bearing reports (above) + `reports/_fragments/` (pipeline outputs the report builders read — **committed**, because `app/run.py` does not regenerate them) |
| `docs/history/` | the phase-by-phase correction notes |
| `data/` | `ndma_guidance.json`, `checklist_rules.json` (committed so the Checklist works offline); everything else here is a gitignored, rebuildable artifact |
| `Raw data/` | the original IMD best-track CSVs |

## Dashboard tabs

Dashboard (replay every 2019+ holdout storm) · Forecast (run the engine on any
state) · Preparation Checklist (NDMA guidance, verbatim) · District Cyclone
History (how often cyclones passed within 100 km of a district, 1982–2026) ·
Official Warnings (paste an IMD bulletin, get it explained) · Chatbot
(retrieval-only over the reports, the storm archive, district history and NDMA
guidance). The earlier tabbed prototype stays at `/prototype`.

**Chatbot safety:** it answers only from retrieved text and cites the source of
every claim. A post-generation guard discards any Groq answer that invents a
warning colour, a specific future time, a numeric forecast, or a
meaning-inverting negation. Preparedness answers are **assembled from
character-for-character copies** of NDMA items — Groq may only choose which items
to show. Discards are logged to `reports/guard_discards.jsonl`. With no
`GROQ_API_KEY` the chatbot runs on offline keyword retrieval.

## Not in scope

ResQ is a forecasting and evaluation tool, not a warning service. It is not
connected to live data, it does not issue warnings / colour codes / district
advisories, and it has no affiliation with or endorsement by IMD.
