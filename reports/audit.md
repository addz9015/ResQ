# ResQ data audit — SIH 2026 PS26070 (cyclone ID / classification / prediction, IMD)

_Every number is reproducible with the scripts listed at the end. No rows were fabricated; blanks were left blank. Resampled-grid wind targets are linear interpolations between real IMD fixes and are flagged as such. This file covers data provenance; model results are in `reports/results.md`, the corrections history in `docs/history/`, and the demo in `app/`._

## Contents
0. Corrections — numbers withdrawn from the first audit (phase-3 corrections: `docs/history/corrections.md`)
1. Null table (before/after cleaning)
2. Lag / tendency recomputation & mismatch rate
3. Segmentation & 6 h resampling
4. Splits
5. Baselines vs models (results: `reports/results.md`)
6. Why `normal_weather_background.csv` was excluded
7. How to reproduce

---

## 0. Corrections — what was withdrawn from the first audit

Three problems in the first pass were fixed before any further modelling; the affected numbers are withdrawn.

**0.1 The intensity task was misframed as a nowcast.** The original "intensity" number predicted `wind(t)` from the previous row's wind. Its persistence MAE of **1.99 kt** (n=1778, 2019+ holdout) measures how smooth the label is over one step, not forecast skill — it is withdrawn.
The intensity task is now **Δwind at +6 / +12 / +24 h** (`wind(t+h) − wind(t)`, by real elapsed time from `dt_snap_unix`), with the baseline **Δ = 0**, reported at the same leads as the track metric. `wind_tendency_3h` / `pressure_tendency_3h` are **not** used as predictors for any intensity target (they reconstruct a one-step target — see §2).

**0.2 The time axis was treated as uniform 3 h — it is not.** Only 60.8% of steps are 3 h. The old LSTM stacked 8 consecutive rows regardless of the real gaps between them, so a "sequence" could span anywhere from 21 h to several months. All sequence-model numbers from that run are withdrawn and replaced with the segmented / resampled version in §3 and §5.

**0.3 Three image-structure columns (>99% null) were kept in the first pass.** `eye_confidence_0to2`, `storm_touches_frame_edge`, `pct_storm_boundary_at_frame_edge_or_nodata` are now dropped entirely — a missingness indicator on a column that is 99.8% empty mostly encodes *which era / batch* a row came from, which is leakage, not signal.


---

### 1. Null table — before vs after cleaning

Source: `training_ready_dataset(all 3 combined).csv` — 7656 rows x 49 cols -> clean `cyclone_clean.parquet` — 7656 rows x 33 cols (no rows dropped).

Columns dropped (21): `eye_to_core_diameter_ratio`, `aux_sst_warm_pct`, `aux_sst_cool_pct`, `aux_rainfall_heavy_pct`, `aux_rainfall_moderate_pct`, `oci`, `oci_diameter`, `core_to_shield_area_ratio`, `dense_core_circularity_0to1`, `symmetry_score_0to1`, `cloud_cover_pct_of_valid_frame`, `image_to_observation_gap_hours`, `image_filename_code`, `storm_name_code`, `has_image`, `pipeline_grayscale`, `pipeline_no_image`, `pipeline_visible`, `eye_confidence_0to2`, `storm_touches_frame_edge`, `pct_storm_boundary_at_frame_edge_or_nodata`

Columns renamed: `wind_tendency_3h` -> `wind_tendency_step`, `pressure_tendency_3h` -> `pressure_tendency_step`, `wind_kt_tendency_3h_recomp` -> `wind_kt_tendency_step_recomp`, `pressure_tendency_3h_recomp` -> `pressure_tendency_step_recomp`

| column | %null before | %null after | status |
|---|---:|---:|---|
| `eye_to_core_diameter_ratio` | 100.00 | — | DROPPED |
| `aux_rainfall_heavy_pct` | 99.99 | — | DROPPED |
| `aux_rainfall_moderate_pct` | 99.99 | — | DROPPED |
| `aux_sst_cool_pct` | 99.97 | — | DROPPED |
| `aux_sst_warm_pct` | 99.97 | — | DROPPED |
| `cloud_cover_pct_of_valid_frame` | 99.78 | — | DROPPED |
| `core_to_shield_area_ratio` | 99.76 | — | DROPPED |
| `dense_core_circularity_0to1` | 99.76 | — | DROPPED |
| `eye_confidence_0to2` | 99.76 | — | DROPPED |
| `image_filename_code` | 99.76 | — | DROPPED |
| `image_to_observation_gap_hours` | 99.76 | — | DROPPED |
| `pct_storm_boundary_at_frame_edge_or_nodata` | 99.76 | — | DROPPED |
| `storm_touches_frame_edge` | 99.76 | — | DROPPED |
| `symmetry_score_0to1` | 99.76 | — | DROPPED |
| `oci` | 93.48 | — | DROPPED |
| `oci_diameter` | 93.48 | — | DROPPED |
| `storm_name_code` | 58.16 | — | DROPPED |
| `has_image` | 0.00 | — | DROPPED |
| `pipeline_grayscale` | 0.00 | — | DROPPED |
| `pipeline_no_image` | 0.00 | — | DROPPED |
| `pipeline_visible` | 0.00 | — | DROPPED |
| `pressure_tendency_step` | 18.09 | 18.09 | kept |
| `pressure_lag1` | 17.52 | 17.52 | kept |
| `wind_tendency_step` | 11.96 | 11.96 | kept |
| `wind_kt_lag1` | 11.46 | 11.46 | kept |
| `imd_pressure_hpa` | 11.32 | 11.32 | kept |
| `imd_pressure_drop` | 7.88 | 7.88 | kept |
| `storm_age_hours` | 4.45 | 4.45 | kept |
| `wind_kt_canonical` | 2.74 | 2.74 | kept |
| `imd_grade_ordinal` | 1.87 | 1.87 | kept |
| `severity_tier_ordinal` | 1.87 | 1.87 | kept |
| `date_suspect` | 1.45 | 1.45 | kept |
| `storm_id_code` | 1.45 | 1.45 | kept |
| `lon` | 0.17 | 0.17 | kept |
| `lat` | 0.01 | 0.01 | kept |
| `basin_ARB` | 0.00 | 0.00 | kept |
| `basin_BOB` | 0.00 | 0.00 | kept |
| `basin_LAND` | 0.00 | 0.00 | kept |
| `doy_cos` | 0.00 | 0.00 | kept |
| `doy_sin` | 0.00 | 0.00 | kept |
| `dt_snap_hour` | 0.00 | 0.00 | kept |
| `dt_snap_unix` | 0.00 | 0.00 | kept |
| `has_wind_label` | 0.00 | 0.00 | kept |
| `month_cos` | 0.00 | 0.00 | kept |
| `month_sin` | 0.00 | 0.00 | kept |
| `oci_diameter_missing` | 0.00 | 0.00 | kept |
| `pressure_missing` | 0.00 | 0.00 | kept |
| `wind_label_source_IMD` | 0.00 | 0.00 | kept |
| `wind_label_source_none` | 0.00 | 0.00 | kept |
| `dt_since_prev_sec` | — | 7.00 | kept |
| `pressure_lag1_recomp` | — | 15.31 | kept |
| `pressure_tendency_step_recomp` | — | 15.88 | kept |
| `wind_kt_lag1_recomp` | — | 9.22 | kept |
| `wind_kt_tendency_step_recomp` | — | 9.73 | kept |


---

### 2. Lag / tendency recomputation

Recomputed per `storm_id_code`, sorted by `dt_snap_unix`: `*_lag1_recomp = value.shift(1)`, `*_tendency_recomp = value - lag1_recomp` (step-to-step, not time-normalised).

**Cadence check** — the shipped columns assume a 3 h step (10800 s); actual gaps:

- rows with a predecessor: 7120
- exactly 3 h apart: 4584 (60.76%)
- largest gap: 8748.0 h
- gap-hours distribution: `{"3h": 4584, "6h": 1999, "9h": 264, "12h": 27, "15h": 193, "18h": 4, "21h": 2, "24h": 9, "27h": 1, "30h": 1, "33h": 1, "39h": 1, "48h": 1, "51h": 2, "597h": 1, "630h": 1, "675h": 1, "699h": 2, "702h": 6, "711h": 1, "723h": 8, "726h": 7, "729h": 1, "1266h": 1, "6201h": 1, "8748h": 1}`

> The cadence is **not** uniformly 3 h (only ~61% of steps are), so the shipped `*_tendency_3h` columns are mislabelled — they are raw step-to-step differences, not 3-hour-normalised rates. They are renamed to `*_tendency_step` in the clean table and a proper 6 h grid is built in §3.

**Shipped vs recomputed** (match tolerance 1e-6):

| shipped column | rows compared | mismatches | mismatch rate | shipped-only nulls filled by recomp | max abs diff |
|---|---:|---:|---:|---:|---:|
| `wind_kt_lag1` | 6779 | 0 | 0.0% | 171 | 0.0 |
| `wind_tendency_3h` | 6740 | 0 | 0.0% | 171 | 0.0 |
| `pressure_lag1` | 6315 | 0 | 0.0% | 169 | 0.0 |
| `pressure_tendency_3h` | 6271 | 0 | 0.0% | 169 | 0.0 |

Column renames in the clean table (the cadence is not 3 h, so the "3h" label was removed): `wind_tendency_3h` -> `wind_tendency_step`, `pressure_tendency_3h` -> `pressure_tendency_step`.

New columns written to the parquet: `wind_kt_lag1_recomp`, `wind_kt_tendency_step_recomp`, `pressure_lag1_recomp`, `pressure_tendency_step_recomp`, `dt_since_prev_sec`. These are step-to-step differences on the raw irregular axis and are kept for provenance only. Proper per-hour rates are computed on the resampled 6 h grid (`data/cyclone_resampled_6h.parquet`, see `src/segment_resample.py`); no `*_tendency` column is used as a predictor for an intensity target.


---

### 3. Segmentation & 6 h resampling (`src/segment_resample.py`)

Each storm is split into contiguous **segments** wherever the observation gap exceeds 12 h. Segments are then resampled onto a uniform **6 h** grid: `lat`, `lon`, `pressure`, `pressure_drop` by linear interpolation in unix time; `wind_kt` by linear interpolation carrying an `interpolated` flag (True when the nearest real wind fix is > 1 h away). Segments with fewer than 5 grid points are dropped.

| quantity | value |
|---|---:|
| raw rows in (valid storm id + lat/lon) | 7531 |
| segments total | 671 |
| segments kept (≥ 5 grid points) | 317 |
| segments dropped (< 5 grid points) | 354 |
| resampled rows out | 4469 |
| wind interpolated fraction (all / holdout) | 0.201 / 0.126 |
| kept-segment length, grid points (min/median/max) | 5 / 13 / 53 |

Kept-segment length histogram (grid points): `{"5-9": 81, "9-13": 73, "13-21": 105, "21-33": 49, "33-999": 9}`.

Raw-segment length before resampling (points): median 5, mean 11.22, max 97 — the 354 dropped segments are overwhelmingly 1–3-point track fragments.

Per-hour rates (`wind_rate_per_h`, `pressure_rate_per_h`), 6 h tendencies and the 6 h motion vector (`dlat_6h`, `dlon_6h`) are recomputed on this uniform grid. Output: `data/cyclone_resampled_6h.parquet`.


---

### 4. Splits (`scripts/splits.py`)

No random row-level split anywhere.

- **Two-way (point forecast models):** storms with first observation in **year ≥ 2019** are the test set; earlier storms train. Segments inherit their storm's side.
- **Three-way (conformal + RI, phase 3):** storm-disjoint — train ≤ 2016 (219 storms / 3041 grid rows), calibration 2017–2018 (18 / 257), test ≥ 2019 (76 / 1171). Full storm-id lists per bucket are in `reports/_fragments/models.json → split`.
- `GroupShuffleSplit` on `storm_id_code` is still available in `scripts/splits.py` for CV but is not used in any headline table.


---

### 5. Baselines vs models — see `reports/results.md`, `docs/history/phase4.md`

The headline numbers live in **`reports/results.md`** (primary intersection, secondary full-n, interpolation sensitivity, marginal/split-CQR/cross-conformal coverage, rapid intensification). **`docs/history/corrections.md`** covers the four phase-3 corrections and **`docs/history/phase4.md`** the phase-4 fixes (true CLIPER, cross-conformal, combined P(RI), the ERA5 layer, the dashboard).

Where things stand on the 2019+ test storms:
- **Track:** the motion baselines (persistence-motion; true CLIPER, an OLS fit) are unbeaten at 6/12/24 h. Every learned track model — direct or residual, GBM or LSTM — loses to the better baseline. No exploitable structure in the residual given kinematics.
- **Intensity (Δwind):** the analog ensemble and a direct GBM beat the Δ = 0 baseline at 12/24 h; the LSTM beats nothing.
- **Uncertainty:** split-CQR and cross-conformal both lift the 24 h Δwind 80% interval from 60% to ~76% coverage; neither reaches nominal (conditional-coverage limit, not calibration data).
- **Rapid intensification:** the **average** of the calibrated analog and classifier P(RI) — ROC-AUC ≈ 0.84, PR-AUC ≈ 0.25 on a 6% base rate — beats either alone and is the one learned model that clearly beats its trivial baseline. Still needs ERA5 to be operational.
- **Demo:** `python app/run.py` serves a Leaflet dashboard that replays each 2019+ storm 6 h at a time with the forecast that would have been issued, the conformal cone, analogs and the P(RI) gauge.


---

### 6. Why `normal_weather_background.csv` was excluded

The file has **9000 rows**. It is **not used anywhere in training, validation, or the holdout** — only here. A prior audit flagged it as synthetic; the checks below reproduce that from the file itself.

**6.1 A classifier separates background from cyclone perfectly, even with all provenance columns removed.**

- `HistGradientBoostingClassifier`, 5-fold stratified CV, target = `is_background`. Provenance dropped (`storm_id_code`, `dt_snap_unix`, `wind_kt_canonical`, `imd_grade_ordinal`, `wind_label_source_*`, `has_image`, `pipeline_*`, image structure columns, …).
- Features used: `lat`, `lon`, `imd_pressure_hpa`, `imd_pressure_drop`, `storm_age_hours`, `month_sin`, `month_cos`, `doy_sin`, `doy_cos`, `dt_snap_hour`, `wind_kt_lag1`, `pressure_lag1`, `wind_tendency_3h`, `pressure_tendency_3h`, `basin_ARB`, `basin_BOB`, `basin_LAND`.
- **ROC-AUC = 1.0000**.
- Using only `lat`, `lon`, `month_sin`, `month_cos`: **ROC-AUC = 0.9985**.

**6.2 The synthetic rows are numerically distinguishable by construction.**

| column | integer-value fraction — cyclone | — background |
|---|---:|---:|
| `wind_kt_canonical` | 1.000 | 0.120 |
| `imd_pressure_hpa` | 1.000 | 0.106 |
| `wind_kt_lag1` | 1.000 | 0.117 |
| `pressure_lag1` | 1.000 | 0.108 |

- Real IMD data is quantised (5-kt winds, whole-hPa pressure): **100%** integer; background rows are continuous floats (~11%).
- `wind_kt_canonical` range: cyclone **15–140 kt**, background **0.5–27.0 kt** — every background row is below tropical-storm force.
- `corr(wind_kt_canonical, wind_kt_lag1)`: cyclone **0.968**, background **0.527** — background lag columns carry no real dynamics.
- `storm_id_code` = **[-1]** for every background row; sampled timestamps jump by up to 738 days — no real tracks in the file.

**6.3 Consequence.** Any supervised model trained on cyclone + background learns the synthetic/real boundary (AUC 1.0) instead of meteorology. Excluded from every split. A real negative class would have to come from genuine IMD / reanalysis fair-weather observations at the same quantisation and cadence.


---

### 7. How to reproduce

```
python scripts/build_dataset.py      # -> data/cyclone_clean.parquet, null + lag fragments
python -m src.segment_resample       # -> data/cyclone_resampled_6h.parquet
python scripts/leakage_audit.py      # -> reports/_fragments/leakage.json
python -m src.era5                   # -> data/cyclone_6h_with_env.parquet (null until CDS access)
python -m src.analogs                # -> analogs.json, analogs_ri.json, example
python -m src.uncertainty            # marginal/split-CQR/cross-conformal -> uncertainty.json, PNG
python -m src.ri                     # combined P(RI) -> reports/_fragments/ri.json
python scripts/run_models.py         # true CLIPER + primary/secondary/residual -> models.json
python -m src.bootstrap              # cluster bootstrap by storm -> reports/significance.md
python -m src.ri_tune                # M1 tuning + M2 learning curve -> ri_tune.json, learning_curve_ri.png
python -m src.fit_diagnostics        # M3 train/OOF/test -> docs/history/fit_diagnostics.md
python scripts/make_results.py       # -> reports/results.md
python scripts/make_corrections.py   # -> docs/history/corrections.md
python scripts/make_phase4.py        # -> docs/history/phase4.md
python scripts/make_phase6.py        # -> docs/history/phase6.md
python scripts/make_limitations.py   # -> reports/limitations.md
python scripts/make_audit.py         # -> reports/audit.md (this file)
python scripts/build_district_history.py  # -> app/static/data/district_history.json
python scripts/build_checklist.py    # -> data/ndma_guidance.json (NDMA Drupal REST harvest)
python scripts/build_corpus.py       # -> app/static/data/corpus.json (chatbot retrieval)
python scripts/build_storms.py       # -> app/static/data/storms.json (2019+ tracks)
python scripts/build_faq.py          # -> app/static/data/faq.json (chatbot)
python scripts/build_insights.py     # -> app/static/data/insights.json (metrics + CIs)
python app/run.py --rebuild          # builds every artifact above, serves 127.0.0.1:8000
                                     #   tabs: Dashboard / Forecast / Preparation Checklist /
                                     #         District Cyclone History / Official Warnings / Chatbot
```
