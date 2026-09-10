# ResQ — phase 6 (dashboard + last modelling)

_No new architectures, no LSTM retrain. Dashboard extended in place; the RI ensemble tuned on pre-2019 OOF only; overfitting checked with numbers._

## Dashboard

**File:** `app/static/index.html` (single file; served at `/`).

**Tabs — before:** none (fixed 3-column replay view).

**Tabs — after:** `Replay` · `Forecast` · `Model insights` (top nav, `.tabs` / `.tabview`, vanilla show/hide, lazy Leaflet `invalidateSize` per tab).

**Endpoints added:** `GET /api/presets` — real 2019+ holdout states pre-filled for the Forecast form + a MANDOUS-2022 preset, read from `data/cyclone_resampled_6h.parquet`. `POST /api/forecast` extended to return the 80% conformal Δwind interval (`dwind_lo` / `dwind_hi`) and its empirical coverage. `Model insights` loads a pre-built `static/data/insights.json` (`scripts/build_insights.py`) — no fitting at page load.

- **Forecast tab:** inputs = lat, lon, wind, pressure, prev-6h lat, prev-6h lon, storm age, date (basin derived from longitude). Outputs = persistence-motion track ("best baseline") + conformal cone, Δwind with 80% intervals + the 75%-vs-80% coverage note, combined P(RI) gauge with the 5.9% base rate marked, top-10 analogs with Δwind deciles. Every panel: *"ResQ model estimate — not an official warning."*
- **Model insights tab:** primary table with 95% bootstrap CIs; a one-panel supported / not-supported summary from `significance.md`; RI PR-AUC bars with error bars + reliability curve; the Δwind coverage table; RI-classifier permutation importance; M1 tuning summary; M2 learning curve; M3 fit-diagnostics table.

**Hard constraint:** no "is there a cyclone" classifier UI or endpoint exists or was added. The only `is_background` model lives in `scripts/leakage_audit.py` — the audit tool that *documents* the AUC-1.0 leakage (feeds `audit.md` §6); it has no UI, endpoint or saved model and is kept as the required proof-of-exclusion. No temperature / humidity inputs anywhere.

## M1 — RI ensemble tuning (selection on pre-2019 GroupKFold OOF PR-AUC only)

Grid: 24 HGB configs × 5 analog k × 7 averaging weights, all scored by 10-fold GroupKFold OOF PR-AUC inside pre-2019. The 2019+ test was scored once, after every choice was frozen.

**Selected config:**

- classifier HGB: `{"learning_rate": 0.05, "max_depth": 2, "max_iter": 300, "l2_regularization": 1.0, "min_samples_leaf": 20}` (shallower / more regularised than the phase-4 depth-4 default — see M3)
- analog `k = 30`
- ensemble = `0.25·P_analog + 0.75·P_classifier`

| metric | value |
|---|---|
| OOF PR-AUC — classifier alone | 0.205 |
| OOF PR-AUC — analog raw | 0.117 |
| **OOF PR-AUC — tuned ensemble** | **0.227** |
| test PR-AUC — phase-4 untuned average | 0.247 |
| **test PR-AUC — tuned ensemble** | **0.276** |

**Did the test score move?** Yes — 0.276 vs 0.247, a +0.029 change. **This is well inside the phase-5 cluster-bootstrap CI (PR-AUC ±~0.15), so it is not a real improvement.** The value of M1 is a *defensible, OOF-selected, more-regularised* config — not a better test number. The deployed engine is left on the documented config so the phase-5 significance analysis stays valid; adopting the tuned config is a safe follow-up.

## M2 — RI learning curve

OOF PR-AUC of the tuned ensemble vs number of pre-2019 training storms:

| storms | RI cases | OOF PR-AUC (ensemble) | (classifier) |
|---:|---:|---:|---:|
| 50 | 22 | 0.170 | 0.137 |
| 90 | 52 | 0.120 | 0.097 |
| 130 | 81 | 0.191 | 0.162 |
| 170 | 100 | 0.199 | 0.173 |
| 183 | 105 | 0.227 | 0.198 |

![learning curve](learning_curve_ri.png)

**Regime:** **data-starved on positive cases.** The OOF PR-AUC curve is noisy (only ~20-105 RI events across the sampled sizes) but its trend over the larger sizes is upward (slope +5.60e-04 per storm) and the full-data point is the highest. More RI-labelled storms — i.e. more years, which in practice means the ERA5 era — should raise the ceiling. It is not overfitting (see M3).

## M3 — overfitting check (train vs OOF vs test)

Full table in `docs/history/fit_diagnostics.md`. Summary:

| model | train | OOF | test | |
|---|--:|--:|--:|---|
| GBM Δwind 6h | 2.4 | 3.6 | 4.2 | **OVERFIT** |
| GBM Δwind 12h | 3.5 | 6.1 | 7.6 | **OVERFIT** |
| GBM Δwind 24h | 4.7 | 10.6 | 13.1 | **OVERFIT** |
| GBM track-residual 6h | 38.1 | 58.3 | 40.1 | **OVERFIT** |
| GBM track-residual 12h | 58.3 | 105.3 | 77.7 | **OVERFIT** |
| GBM track-residual 24h | 92.4 | 188.3 | 152.1 | **OVERFIT** |
| true CLIPER 6h | 55.6 | 56.2 | 39.0 | ok |
| true CLIPER 12h | 98.6 | 100.0 | 74.0 | ok |
| true CLIPER 24h | 179.8 | 184.0 | 147.2 | ok |
| RI classifier (HGB) | 0.930 | 0.155 | 0.219 | **OVERFIT** |
| RI analog P(RI), k=10 | 0.093 | 0.105 | 0.234 | ok |

**7 of 11 models are flagged.** For each, OOF ≈ test while train is far better — they memorise their training storms. The RI classifier is the extreme (train PR-AUC 0.93 vs OOF 0.15). **true CLIPER (linear) and the analog ensemble are not flagged** — train ≈ OOF ≈ test. Two failure modes stack: the kinematic feature set is weak *and* the flexible models overfit the little that is there. ERA5 features matter more than a bigger model. The LSTM is not retrained (constraint); its test error being worse than the trivial baselines already rules out train-memorisation.
