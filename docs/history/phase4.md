# ResQ — phase-4 changes

_Three fixes, one data-acquisition layer, one dashboard. Before/after numbers below; full tables in `reports/results.md`. All test numbers are on the 2019+ storms._

## Fix 1 — the motion baseline was misnamed

**Before:** phase-3 called its baseline "CLIPER". It was only persistence-of-motion — extrapolate the last 6 h `(Δlat, Δlon)`.

**After:**
- renamed **persistence-motion**;
- added **true CLIPER**: OLS of lead-h displacement on `[lat, lon, wind_kt, dlat_6h, dlon_6h, doy_sin, doy_cos, lat_x_dlat6]`, fit on 237 pre-2019 storms;
- both scored on the primary intersection;
- Correction-3 residual learning now targets whichever wins the 2019+ primary intersection (**persistence motion**).

| baseline, track km | CV 6/12/24h | test 6/12/24h |
|---|---|---|
| persistence-motion | 60 / 114 / 221 | 33 / 66 / 144 |
| true CLIPER | 56 / 100 / 184 | 39 / 74 / 147 |

**Result:** true CLIPER wins the pre-2019 CV (184 vs 221 km at 24 h) but **not the held-out 2019+ test** — persistence-motion is level or better at 6/12 h. `true_cliper_wins_pre2019_cv = True`, `true_cliper_wins_2019plus_test = False`. The residual models (gbm 152, lstm 175 km at 24 h) still do not beat the bare baseline (144 km). **Phase-3 conclusion holds and is stronger:** with kinematic features there is no learnable structure in either motion baseline's residual.

## Fix 2 — conformal calibration was data-starved

**Before:** split-CQR calibrated on the fixed 2017–2018 bucket — 238 states / 18 storms. The 24 h Δwind 80% interval reached only ~76% coverage.

**After:** cross-conformal for the Δwind targets — K=10 GroupKFold over all 2560 pre-2019 states, pooled conformity scores, quantile GBM refit on all pre-2019.

| Δwind lead | nominal | marginal | split-CQR | cross-conformal |
|---|---|--:|--:|--:|
| 6h | 50% | 0.67 | 0.67 | 0.66 |
| 6h | 80% | 0.74 | 0.82 | 0.82 |
| 12h | 50% | 0.53 | 0.55 | 0.56 |
| 12h | 80% | 0.72 | 0.83 | 0.81 |
| 24h | 50% | 0.42 | 0.53 | 0.50 |
| 24h | 80% | 0.60 | 0.76 | 0.75 |

**Result:** cross-conformal 24 h nominal-80% coverage = **75%** — within a point of split-CQR (76%) and still short of 80%. **The extra calibration data (237 storms vs 18) does not close the gap**, so the shortfall is not calibration-set size — it is conditional coverage (the 24 h quantile intervals are too narrow for the hard cases; a global conformal offset cannot fix that). Both methods do lift the marginal 60% to ~76%. Honest answer to 'does it reach nominal at 24 h': no.

## Fix 3 — combine the two RI signals

**Before:** phase-3 reported analog P(RI) and the classifier separately (PR-AUC ≈ 0.23 and 0.16); no combination.

**After:** both isotonic-calibrated on K=10 pre-2019 out-of-fold predictions, then a simple average and a logistic stack.

| forecast | ROC-AUC | PR-AUC | Brier |
|---|--:|--:|--:|
| analog (calibrated) | 0.787 | 0.187 | 0.0532 |
| classifier (calibrated) | 0.796 | 0.189 | 0.0526 |
| average | 0.842 | 0.247 | 0.0522 |
| logistic stack | 0.824 | 0.231 | 0.0563 |

**Result:** the **simple average has the best point estimate** — ROC-AUC 0.84, PR-AUC 0.25, Brier 0.0522, ahead of both inputs on all three metrics. The logistic stack over-weights the classifier and is worse on Brier.

Phase-5 cluster bootstrap (`reports/significance.md`): average **significantly** beats the classifier (+0.057 PR-AUC [95% CI 0.009, 0.127]) but **not** the analog ensemble alone (+0.060 [-0.012, 0.198], crosses 0) — only 48 RI cases in the test set. Use the average as a robust default; the analog-alone vs average question needs more positive examples (i.e. ERA5-era data) to settle.

## Data — ERA5 environmental predictors (`src/era5.py`)

Built now, on the critical path. Status: **unavailable**.
- 16 env columns (shear / steering z500 / RH600 / SST × centre + 200/500/800 km annuli) written to `data\cyclone_6h_with_env.parquet`, currently NULL.
- 223 monthly CDS requests would be needed; cached per month in `data/era5_cache/`.
- Fails cleanly: prints the exact missing pieces (cdsapi/xarray libs + `~/.cdsapirc`), writes null columns, exits 0. No model consumes them yet.
- Missing fields are left null — never interpolated across gaps.

## New deliverable — operational dashboard (`app/`)

`python app/run.py` → builds the engine + replay hindcasts (~2 min) and serves **http://127.0.0.1:8000**. Offline, no CDS calls.
- **FastAPI**: `POST /api/forecast` (state → track + cone + Δwind + P(RI) + top-10 analogs), `GET /api/replay/{storm_id}` (step-by-step hindcast), `GET /api/replay` (storm list).
- **Leaflet frontend**: observed track, forecast track (true CLIPER), 80% conformal cone from the along/cross-track quantiles, the actual future revealed as you step, analog table, and a P(RI) gauge with the climatological base rate marked.
- **Replay mode** steps a 2019+ holdout storm forward 6 h at a time, showing the forecast that *would* have been issued and the track error once each verifying fix is reached. This is the demo.
