# ResQ — phase-3 corrections

_What was wrong in the phase-2 `results.md`, what changed, and the numeric effect. Companion to `reports/results.md` and `reports/audit.md`. Phase-4 refined corrections 1–3 further — see `docs/history/phase4.md`; numbers here are the current re-run, with the phase-4 baseline names ("CLIPER" of phase-3 = **persistence-motion**)._

## Correction 1 — the model comparison was unfair

**Wrong:** phase-2 put the LSTM (n = 431 sequences, which require 12 consecutive 6 h grid points) in the same table as persistence / CLIPER / GBM / analog (n = 791 at 24 h). Different, easier subsets for the non-sequence models — the comparison was meaningless.

**Changed:** two explicit tables.
- **Primary (`results.md` §1)** — every model scored on the *same* states: end of a valid 8-step window, future point present, GBM/analog features present. n = 591 / 533 / 431 at 6 / 12 / 24 h.
- **Secondary (`results.md` §2)** — persistence / CLIPER / GBM / analog on all test states (n = 1019 / 943 / 791), clearly labelled as a different, larger sample with no LSTM row.

**Effect:** the intersection is a harder subset (more still-developing storms). Persistence Δwind MAE at 24 h is 14.11 kt on the intersection vs 12.20 kt on the full sample; CLIPER track at 24 h is 144.5 vs 155.4 km. Model *ranking* is the same in both tables — CLIPER best on track, analog/GBM best on intensity — so the phase-2 conclusion survives, but only the primary table supports it.

## Correction 2 — uncertainty calibration used the test set

**Wrong:** phase-2 fit the quantile GBM on pre-2019 storms and then read interval coverage straight off the 2019+ storms — no held-out calibration, so "calibrated uncertainty" was calibrated on nothing.

**Changed:** storm-disjoint three-way split — train ≤ 2016 (219 storms), calibration 2017–2018 (18 storms), test ≥ 2019 (76 storms). Split conformalized quantile regression: quantile GBM on train, conformity scores `E_i = max(q_lo−y, y−q_hi)` on the calibration storms, offset `Q` = the `(1−α)` empirical quantile of `E`, test interval `[q_lo−Q, q_hi+Q]`. (Storm ids per bucket: `models.json → split`.)

**Effect — Δwind interval coverage (nominal → marginal → CQR):**

| lead | nominal 80% | marginal | CQR | nominal 50% | marginal | CQR |
|---|---|--:|--:|---|--:|--:|
| 6h | 0.80 | 0.74 | **0.82** | 0.50 | 0.67 | **0.67** |
| 12h | 0.80 | 0.72 | **0.83** | 0.50 | 0.53 | **0.55** |
| 24h | 0.80 | 0.60 | **0.76** | 0.50 | 0.42 | **0.53** |

CQR **partly fixes** the 24 h over-confidence: nominal-80% coverage 60% → 76% (widening the median interval 22 → 27 kt). It does not reach 80% — the calibration set is only 162 states so the offset is noisy. Track intervals were already close to nominal and stay there.

## Correction 3 — track models predicted position, not CLIPER's residual

**Wrong:** the LSTM (and any GBM track model) predicted the absolute displacement, so it had to re-learn the linear motion CLIPER already gives for free — and lost to CLIPER by a wide margin.

**Changed:** both the LSTM and a new GBM are retargeted to predict `(actual_lat − cliper_lat, actual_lon − cliper_lon)` at each lead; the forecast is `CLIPER + predicted residual`. Direct-prediction variants are kept side by side (`results.md` §1).

**Effect — track error, km (primary table):**

| model | 6h | 12h | 24h |
|---|--:|--:|--:|
| persistence-motion (bare) | 32.6 | 66.0 | 144.5 |
| GBM direct | 41.3 | 78.8 | 161.8 |
| GBM motion+residual | 40.1 | 77.7 | 152.1 |
| LSTM direct | 53.2 | 104.3 | 186.1 |
| LSTM motion+residual | 48.6 | 97.4 | 174.8 |

Residual learning **helps** — LSTM 24 h 186 → 175 km, GBM 24 h 162 → 152 km — but **neither beats plain CLIPER** (144 km). This is itself the finding: with kinematic features there is no systematic, learnable structure in CLIPER's errors. A learned track model will need environmental predictors (steering flow, ridge position) to add skill.

## Correction 4 — interpolated targets

**Wrong:** ~12–20% of resampled wind values are linear interpolations between real IMD fixes; phase-2 scored against them without checking whether that flattered the models.

**Changed:** the primary table is re-scored with test target points excluded whenever either Δwind endpoint is an interpolated wind (`results.md` §3).

**Effect:** dropping interpolated targets removes 19 of 431 states at 24 h. Every model's track and Δwind number moves by < 1.5 km / < 0.3 kt. Ranking changes only at 24h, and only a swap between the two worst LSTM variants; the top of every ranking is unchanged. Interpolation is not inflating the results.

## New components C & D — rapid intensification

- **Component C** turns the analog ensemble into a distribution (deciles of analog Δwind) and reads off `P(RI)` = fraction of analogs with Δwind ≥ +30 kt / 24 h, plus `P(rapid weakening)`.
- **Component D** (`src/ri.py`) is a dedicated HGB classifier + isotonic calibration for the same RI label, kinematic features only, same three-way split.
- On the 2019+ test set (base rate 6.1%): analog P(RI) ROC-AUC **0.79** / PR-AUC **0.19**, classifier **0.80** / **0.19**. Both beat climatology on Brier (0.0532 and 0.0526 vs 0.0570). The analog method wins on discrimination; neither is operationally strong without environmental fields.
