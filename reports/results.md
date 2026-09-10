# ResQ model comparison — 2019+ test storms

_SIH 2026 PS26070, phase-4. Split is storm-disjoint: point-forecast models are fit on **pre-2019 storms** (237 storms) and evaluated on **2019+ storms** (76 storms). Conformal calibration and the RI models use a further GroupKFold split inside pre-2019 (Fix 2 / Fix 3). No random row splits, no CV numbers in these tables. Storm lists per bucket: `reports/_fragments/models.json → split`._

> **Read `reports/significance.md` alongside this file.** A 1000-replicate cluster bootstrap **by storm** (~70 test storms) shows most of the point-estimate differences below are **within noise**. What holds up at 95%: the combined P(RI) beats climatology and beats the classifier alone; persistence-motion beats true CLIPER on track at 6/12 h. What does **not**: any intensity model vs Δ=0, any learned track model vs persistence-motion, and the combined P(RI) vs the analog ensemble alone.

## 0. The motion baseline, split in two (Fix 1)

"CLIPER" in phase-3 was **persistence-of-motion only** — extrapolate the last 6 h `(Δlat, Δlon)` vector. It is renamed **persistence-motion**. **True CLIPER** is now an OLS regression of lead-h displacement on `[lat, lon, wind_kt, dlat_6h, dlon_6h, doy_sin, doy_cos, lat_x_dlat6]`, fit on 237 pre-2019 storms (2704/2487/2033 states).

Track error, km — pre-2019 grouped 5-fold CV **vs** the 2019+ test (primary intersection):

| baseline | CV 6h | CV 12h | CV 24h | test 6h | test 12h | test 24h |
|---|--:|--:|--:|--:|--:|--:|
| persistence-motion | 59.8 | 114.0 | 220.6 | 32.6 | 66.0 | 144.5 |
| true CLIPER | 56.2 | 100.2 | 183.6 | 39.0 | 74.0 | 147.2 |

> **True CLIPER wins the pre-2019 CV at every lead (184 vs 221 km at 24 h)** — but on the held-out 2019+ storms **persistence-motion is as good or better at 6/12 h** (32.6 vs 39.0 km at 6 h), the two converging by 24 h. The OLS fit picks up pre-2019 climatology that does not fully carry forward. Both are kept as scored baselines; the Correction-3 residual is learned against **persistence motion** (the winner of the 2019+ primary intersection, per the Fix-1d rule).

## 1. Primary comparison — model-score intersection

Every model scored on the **same** test states: end of a valid 8-step sequence window, future point present, all GBM/analog features present.

| model | track 6h | 12h | 24h | Δwind MAE 6h | 12h | 24h |
|---|--:|--:|--:|--:|--:|--:|
| persistence (Δ=0 / no motion) | 75.2 | 145.9 | 272.9 | 3.96 | 7.71 | 14.11 |
| persistence-motion (extrapolate last 6 h vector) | 32.6 | 66.0 | 144.5 | — | — | — |
| true CLIPER (OLS climatology+persistence) | 39.0 | 74.0 | 147.2 | — | — | — |
| GBM — direct (Δposition, Δwind) | 41.3 | 78.8 | 161.8 | 4.22 | 7.60 | 13.14 |
| GBM — motion baseline + learned residual | 40.1 | 77.7 | 152.1 | 4.22 | 7.60 | 13.14 |
| LSTM — direct | 53.2 | 104.3 | 186.1 | 4.70 | 9.40 | 16.59 |
| LSTM — motion baseline + learned residual | 48.6 | 97.4 | 174.8 | 4.96 | 11.13 | 16.47 |
| analog-ensemble mean (k=10) | 46.8 | 88.1 | 181.1 | 3.83 | 7.14 | 13.32 |

_n per lead: 6h=591, 12h=533, 24h=431. Track = mean great-circle km; intensity = MAE of Δwind vs actual `wind(t+h) − wind(t)` (kt)._

## 2. Secondary comparison — full test sample

persistence / persistence-motion / true CLIPER / GBM / analog on **all** scorable test states — larger n, different sample, **not comparable to the LSTM rows above.**

| model | track 6h | 12h | 24h | Δwind MAE 6h | 12h | 24h |
|---|--:|--:|--:|--:|--:|--:|
| persistence (Δ=0 / no motion) | 75.7 | 148.6 | 286.4 | 3.27 | 6.43 | 12.20 |
| persistence-motion (extrapolate last 6 h vector) | 33.7 | 70.5 | 155.4 | — | — | — |
| true CLIPER (OLS climatology+persistence) | 38.9 | 74.9 | 150.8 | — | — | — |
| GBM — direct (Δposition, Δwind) | 40.8 | 79.3 | 162.8 | 3.41 | 6.13 | 11.04 |
| analog-ensemble mean (k=10) | 45.4 | 87.0 | 177.4 | 3.26 | 5.99 | 11.20 |

_n per lead: 6h=1019, 12h=943, 24h=791. Track = mean great-circle km; intensity = MAE of Δwind vs actual `wind(t+h) − wind(t)` (kt)._

## 3. Interpolation sensitivity (Correction 4)

Primary table re-scored with test target points where either Δwind endpoint is an interpolated wind **excluded** (n = 543/494/412 vs 591/533/431).

| model | track 24h (primary → clean) | Δwind MAE 24h (primary → clean) |
|---|--:|--:|
| persistence (Δ=0 / no motion) | 272.9 → 273.5 | 14.11 → 14.24 |
| persistence-motion (extrapolate last 6 h vector) | 144.5 → 144.2 | — |
| true CLIPER (OLS climatology+persistence) | 147.2 → 146.5 | — |
| GBM — direct (Δposition, Δwind) | 161.8 → 160.3 | 13.14 → 13.34 |
| GBM — motion baseline + learned residual | 152.1 → 150.8 | 13.14 → 13.34 |
| LSTM — direct | 186.1 → 185.2 | 16.59 → 16.64 |
| LSTM — motion baseline + learned residual | 174.8 → 175.4 | 16.47 → 16.69 |
| analog-ensemble mean (k=10) | 181.1 → 180.9 | 13.32 → 13.54 |

**Ranking changes only at 24h**, among lower-ranked models; the leaders (true CLIPER for track, analog/GBM for intensity) are unaffected.

## 4. Predictive-interval coverage — marginal / split-CQR / cross-conformal (Fix 2)

The fixed 2017–2018 calibration bucket is only 238 states. For the **Δwind** targets we add **cross-conformal** prediction: K=10 GroupKFold over all 2560 pre-2019 states, pooled conformity scores, quantile GBM refit on all pre-2019. Track intervals keep split-CQR (already near nominal).

| target | lead | nominal | marginal | split-CQR | **cross-conformal** | n test |
|---|---|--:|--:|--:|--:|--:|
| Δwind | 6h | 50% | 0.67 | 0.67 | **0.66** | 1019 |
| Δwind | 6h | 80% | 0.74 | 0.82 | **0.82** | 1019 |
| Δwind | 12h | 50% | 0.53 | 0.55 | **0.56** | 943 |
| Δwind | 12h | 80% | 0.72 | 0.83 | **0.81** | 943 |
| Δwind | 24h | 50% | 0.42 | 0.53 | **0.50** | 791 |
| Δwind | 24h | 80% | 0.60 | 0.76 | **0.75** | 791 |
| along-track | 6h | 50% | 0.52 | 0.52 | — | 1019 |
| along-track | 6h | 80% | 0.90 | 0.85 | — | 1019 |
| along-track | 12h | 50% | 0.49 | 0.48 | — | 943 |
| along-track | 12h | 80% | 0.88 | 0.78 | — | 943 |
| along-track | 24h | 50% | 0.43 | 0.49 | — | 791 |
| along-track | 24h | 80% | 0.83 | 0.79 | — | 791 |
| cross-track | 6h | 50% | 0.55 | 0.56 | — | 1019 |
| cross-track | 6h | 80% | 0.83 | 0.85 | — | 1019 |
| cross-track | 12h | 50% | 0.53 | 0.57 | — | 943 |
| cross-track | 12h | 80% | 0.83 | 0.87 | — | 943 |
| cross-track | 24h | 50% | 0.46 | 0.50 | — | 791 |
| cross-track | 24h | 80% | 0.80 | 0.86 | — | 791 |

> **Does cross-conformal reach nominal at 24 h?** **No.** The 24 h Δwind nominal-80% interval: marginal 60% → split-CQR 76% → cross-conformal **75%**. Nominal-50%: 42% → 53% → **50%**. Cross-conformal (all 237 pre-2019 storms) lands **within a point of split-CQR (18 storms)** — the extra calibration data does *not* close the gap, so the shortfall is not calibration-set size. It is **conditional coverage**: the quantile GBM's 24 h intervals are too narrow for the genuinely uncertain cases and a single global conformal offset can't fix that without over-widening the easy ones. Both methods do lift the marginal from 60% to ~76%. Diagram: `reports/reliability_uncertainty.png`.

## 5. Rapid intensification — combined P(RI), 24 h ≥ +30 kt (Fix 3)

Test base rate **6.1%** (791 states). All probabilities isotonic-calibrated on the K=10 pre-2019 out-of-fold predictions.

| forecast | ROC-AUC | PR-AUC (base 0.061) | Brier |
|---|--:|--:|--:|
| climatology | 0.500 | 0.061 | 0.0570 |
| analog P(RI), calibrated | 0.787 | 0.187 | 0.0532 |
| RI classifier, calibrated | 0.796 | 0.189 | 0.0526 |
| **average of the two** | 0.842 | 0.247 | 0.0522 |
| logistic stack | 0.824 | 0.231 | 0.0563 |

> **The simple average has the best point estimate** — ROC-AUC 0.84, PR-AUC 0.25, Brier 0.0522 (analog 0.19 PR-AUC, classifier 0.19). The logistic stack over-weights the classifier and is worse on Brier.

> **Significance (cluster bootstrap by storm, `reports/significance.md`):** average vs classifier +0.057 PR-AUC [95% CI 0.009, 0.127] — **significant**. Average vs analog +0.060 [-0.012, 0.198] — **NOT significant** (CI crosses 0). With only 48 RI-positive test cases the intervals are wide; the average is a safe default (never worse) but its edge over the analog ensemble alone is not established.

> Either way, PR-AUC 0.25 on a 6.1% base rate is ~4× chance — usable as a flag, not a precise probability, until ERA5 shear/SST arrive.

## 6. Probabilistic analog output — worked example (component C)

Query: **2022-013** (MANDOUS, 2022), (8.9°N, 85.0°E), 30.0 kt — the 2019+ test state with the strongest actual 24 h intensification (+20 kt). 10 analogs with a 24 h outcome.

| lead | p10 | p20 | p30 | p40 | p50 | p60 | p70 | p80 | p90 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 6h | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.5 |
| 12h | 0 | 0 | 0 | 0 | 0 | 0 | 1.5 | 5 | 6 |
| 24h | 0 | 0 | 3.5 | 5 | 5 | 5 | 6.5 | 11 | 15.5 |

**P(RI ≥ +30 kt / 24 h) = 0%** from this ensemble; mean forecast ≈ +6 kt. Even the p90 analog (15.5 kt) falls short of the actual +20 kt — kinematic analogs miss the environmental setup, which is exactly why the combined P(RI) in §5 (adding the classifier) and, ultimately, ERA5 matter.

## 7. What beats its baseline (primary table)

**6h:** (best motion baseline = 32.6 km)
- track: GBM — direct (Δposition, Δwind) loses to it (41.3 km)
- track: GBM — motion baseline + learned residual loses to it (40.1 km)
- track: LSTM — direct loses to it (53.2 km)
- track: LSTM — motion baseline + learned residual loses to it (48.6 km)
- track: analog-ensemble mean (k=10) loses to it (46.8 km)
- intensity: GBM — direct (Δposition, Δwind) loses to Δ=0 (4.22 vs 3.96 kt)
- intensity: LSTM — direct loses to Δ=0 (4.70 vs 3.96 kt)
- intensity: analog-ensemble mean (k=10) beats Δ=0 (3.83 vs 3.96 kt)

**12h:** (best motion baseline = 66.0 km)
- track: GBM — direct (Δposition, Δwind) loses to it (78.8 km)
- track: GBM — motion baseline + learned residual loses to it (77.7 km)
- track: LSTM — direct loses to it (104.3 km)
- track: LSTM — motion baseline + learned residual loses to it (97.4 km)
- track: analog-ensemble mean (k=10) loses to it (88.1 km)
- intensity: GBM — direct (Δposition, Δwind) beats Δ=0 (7.60 vs 7.71 kt)
- intensity: LSTM — direct loses to Δ=0 (9.40 vs 7.71 kt)
- intensity: analog-ensemble mean (k=10) beats Δ=0 (7.14 vs 7.71 kt)

**24h:** (best motion baseline = 144.5 km)
- track: GBM — direct (Δposition, Δwind) loses to it (161.8 km)
- track: GBM — motion baseline + learned residual loses to it (152.1 km)
- track: LSTM — direct loses to it (186.1 km)
- track: LSTM — motion baseline + learned residual loses to it (174.8 km)
- track: analog-ensemble mean (k=10) loses to it (181.1 km)
- intensity: GBM — direct (Δposition, Δwind) beats Δ=0 (13.14 vs 14.11 kt)
- intensity: LSTM — direct loses to Δ=0 (16.59 vs 14.11 kt)
- intensity: analog-ensemble mean (k=10) beats Δ=0 (13.32 vs 14.11 kt)

**Bottom line.** No learned track model beats the better of the two motion baselines at any lead — the GBM/LSTM residual models (now learning the residual of the 2019+-winning baseline) still add no skill, and against true CLIPER's residual they are slightly *worse* than the bare fit, i.e. the residual is noise. On intensity, analog and the direct GBM beat Δ=0 at 12/24 h; the LSTM beats nothing. The **combined P(RI) flag (§5) is the one place a learned model clearly beats the trivial baseline** (PR-AUC 0.25 vs 0.06).

## 8. ERA5 environmental predictors — status

`src/era5.py` is built and wired to `data/cyclone_6h_with_env.parquet` (16 env columns). Current status: **unavailable** — credentials/libraries not yet available in this environment; columns are present but null and no model consumes them. The moment CDS access is approved, `python -m src.era5` populates shear / steering / RH600 / SST at the storm centre and 200/500/800 km annuli; this is the expected route to real RI and intensity skill.
