# ResQ dashboard (`app/`)

The TC-Sentinel UI (`app/static/dashboard.html`) wired to the real ResQ models
(SIH 2026, PS26070). Historical replay + a live forecast tool + citizen tabs +
a retrieval chatbot. **Nothing in it is live and it does not issue warnings.**

## Run

```bash
python app/run.py            # builds what's missing, serves http://127.0.0.1:8000
python app/run.py --rebuild  # force-rebuild every artifact first
```

`run.py` loads `.env` (for `GROQ_API_KEY` / `GROQ_MODEL`, both optional), builds
any missing artifact from the committed CSVs + `reports/_fragments/`, then serves
the dashboard. First build trains the `ForecastEngine` + 76 replay hindcasts
(~5–10 min) and fetches GADM boundaries + NDMA guidance; later starts are
seconds.

## Tabs

- **Dashboard** — every 2019+ holdout storm, with operational metrics carrying
  95 % cluster-bootstrap CIs.
- **Forecast** — enter or prefill a storm state → track + 80 % conformal cone +
  Δwind interval + P(RI) + 10 nearest analogs.
- **Preparation Checklist** — NDMA cyclone Do's & Don'ts, shown **verbatim**,
  grouped Now / Next 24 h / Final 12 h by hours-to-arrival, each with its
  source URL.
- **District Cyclone History** — how often a cyclone passed within 100 km of a
  district (1982–2026), by decade and peak category, with the tracks on a map.
  A historical count, not a forecast.
- **Official Warnings** — paste an IMD bulletin; `src/imd_bulletin.py` extracts
  its fields deterministically and restates them in plain language, then ResQ's
  own uncertainty / P(RI) / analogs are shown separately below.
- **Chatbot** — retrieval-only over `reports/`, the storm archive, district
  history and the NDMA guidance; cites every source; refuses live-storm
  questions. Groq (if `GROQ_API_KEY` is set) only rephrases retrieved text and
  is bounded by a post-generation guard; preparedness answers are assembled from
  exact NDMA strings.

The earlier tabbed prototype is still at `/prototype`.

## API

| method | path | purpose |
|---|---|---|
| GET  | `/` | the dashboard |
| GET  | `/api/health` | artifact status |
| GET  | `/api/presets` | 2019+ holdout states for the Forecast tab |
| GET  | `/api/replay` · `/api/replay/{id}` | replayable storms · one storm's step-by-step hindcast |
| POST | `/api/forecast` | forecast for an arbitrary state (see `StateIn` in `backend/main.py`) |
| POST | `/api/bulletin/explain` | parse + explain a pasted IMD bulletin + a ResQ overlay |
| POST | `/api/chat` · GET `/api/chat/status` | the retrieval chatbot |
| GET  | `/static/data/*.json` | the pre-built dashboard payloads |

Every response carrying a model output includes
`"disclaimer": "ResQ model estimate — not an official warning."`

```bash
curl -s localhost:8000/api/forecast -H 'content-type: application/json' -d '{
  "lat": 12.0, "lon": 88.0, "wind_kt": 45, "pressure_hpa": 990,
  "dlat_6h": 0.3, "dlon_6h": -0.5, "month": 11, "basin": "BOB"}'
```

## Files

```
app/
  run.py                 entry point (loads .env, builds artifacts, serves)
  precompute.py          trains the engine, builds replay.json
  backend/
    main.py              FastAPI app + routes
    engine.py            ForecastEngine (track + intensity + cone + P(RI) + analogs)
    chat.py              retrieval-only chatbot (offline default; Groq optional)
  static/
    dashboard.html       the TC-Sentinel dashboard (served at /)
    index.html           the earlier tabbed prototype (served at /prototype)
    data/                generated payloads — engine.pkl, replay.json, *.json (gitignored)
    img/                 report figures copied in by build_insights.py (gitignored)
```
