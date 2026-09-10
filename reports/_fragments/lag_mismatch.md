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
