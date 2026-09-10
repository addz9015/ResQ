# ResQ — statistical significance (cluster bootstrap by storm)

_Phase-5 TASK 1. 1000 bootstrap replicates, resampling the **2019+ test storms** with replacement (not states — adjacent 6 h states are ~0.97 autocorrelated). Point estimate = value on the real test set; interval = 2.5/97.5 percentiles of the replicates. Paired differences are computed within each replicate. `src/bootstrap.py`._


## Headline — what survives the bootstrap

**Statistically supported (95% cluster bootstrap):**
- the combined P(RI) beats a climatological base-rate forecast on Brier
- the combined P(RI) beats the classifier alone (PR-AUC)
- persistence-motion beats true CLIPER on track at 6/12 h

**NOT statistically supported — point estimate only, CI crosses zero:**
- the combined P(RI) beats the analog ensemble alone (PR-AUC)
- any intensity model beats the Δ=0 baseline (point estimates favour analog/GBM at 12/24 h but every CI crosses zero)
- any learned track model beats persistence-motion (all are significantly worse or tie)

> The test set is ~70 storms / ~48 RI cases. Most phase-4 "wins" on track and intensity are within noise at this sample size; the RI combination and the RI-vs-climatology gap are the results that hold up. This does not overturn the *direction* of the phase-4 findings — it bounds the confidence in them.

## Rapid intensification — P(RI ≥ +30 kt / 24 h)

791 test states across 71 storms, 48 RI-positive. Replicates with no positive case give undefined AUC/PR-AUC and are dropped (count shown).

| model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| analog | 0.787 [95% CI 0.697, 0.862] | 0.187 [95% CI 0.093, 0.317] | 0.053 [95% CI 0.031, 0.078] |
| classifier | 0.796 [95% CI 0.662, 0.892] | 0.189 [95% CI 0.073, 0.390] | 0.053 [95% CI 0.032, 0.075] |
| average | 0.842 [95% CI 0.730, 0.913] | 0.247 [95% CI 0.104, 0.488] | 0.052 [95% CI 0.031, 0.076] |
| stack | 0.824 [95% CI 0.704, 0.910] | 0.231 [95% CI 0.087, 0.466] | 0.056 [95% CI 0.032, 0.084] |
| climatology (rate 0.059) | 0.500 | 0.059 | 0.057 [95% CI 0.033, 0.084] |

_(AUC/PR-AUC intervals from 1000/1000 valid replicates.)_

**Paired differences (PR-AUC, higher = better):**

- average beats analog by +0.060 PR-AUC [95% CI -0.012, +0.198] — not significant (CI crosses 0)
- average beats classifier by +0.057 PR-AUC [95% CI +0.009, +0.127] — **significant**
- average beats stack by +0.016 PR-AUC [95% CI -0.006, +0.044] — not significant (CI crosses 0)

**Brier vs climatology (lower = better):**

- average beats climatology by +0.0048 [95% CI +0.0012, +0.0089] — **significant**
- analog beats climatology by +0.0038 [95% CI +0.0012, +0.0066] — **significant**

## Track error, km (primary model-score intersection)

| model | 6 h | 12 h | 24 h |
|---|---|---|---|
| persistence_motion | 32.6 [95% CI 29.4, 36.2] | 66.0 [95% CI 59.1, 74.0] | 144.5 [95% CI 128.1, 167.3] |
| true_cliper | 39.0 [95% CI 36.1, 42.3] | 74.0 [95% CI 68.7, 80.5] | 147.2 [95% CI 134.3, 161.2] |
| gbm_direct | 41.3 [95% CI 38.7, 44.5] | 78.8 [95% CI 73.2, 85.0] | 161.8 [95% CI 148.8, 176.7] |
| gbm_motion_resid | 40.1 [95% CI 37.4, 43.3] | 77.7 [95% CI 71.6, 84.4] | 152.1 [95% CI 136.2, 169.6] |
| lstm_direct | 53.2 [95% CI 49.5, 57.2] | 104.3 [95% CI 93.3, 115.6] | 186.1 [95% CI 167.6, 205.0] |
| lstm_motion_resid | 48.6 [95% CI 43.5, 54.1] | 97.4 [95% CI 89.0, 108.1] | 174.8 [95% CI 156.5, 197.1] |
| analog | 46.8 [95% CI 43.4, 50.2] | 88.1 [95% CI 82.1, 94.3] | 181.1 [95% CI 166.1, 196.5] |

_n per lead: 6h=591, 12h=533, 24h=431._

**Paired differences vs persistence-motion** (each: `improvement km [95% CI] — verdict`; a *negative* improvement means the model is worse):

- 6h: true_cliper beats persistence-motion by -6.4 km [95% CI -9.2, -3.9] — **significant — persistence-motion is better**
- 12h: true_cliper beats persistence-motion by -8.0 km [95% CI -13.8, -2.0] — **significant — persistence-motion is better**
- 24h: true_cliper beats persistence-motion by -2.7 km [95% CI -17.6, +12.8] — not significant (CI crosses 0)
- 6h: gbm_direct beats persistence-motion by -8.7 km [95% CI -11.9, -5.8] — **significant — persistence-motion is better**
- 12h: gbm_direct beats persistence-motion by -12.8 km [95% CI -20.0, -5.6] — **significant — persistence-motion is better**
- 24h: gbm_direct beats persistence-motion by -17.3 km [95% CI -33.9, +1.2] — not significant (CI crosses 0)
- 6h: gbm_motion_resid beats persistence-motion by -7.6 km [95% CI -10.5, -4.9] — **significant — persistence-motion is better**
- 12h: gbm_motion_resid beats persistence-motion by -11.7 km [95% CI -19.8, -4.3] — **significant — persistence-motion is better**
- 24h: gbm_motion_resid beats persistence-motion by -7.6 km [95% CI -24.8, +10.2] — not significant (CI crosses 0)
- 6h: lstm_direct beats persistence-motion by -20.6 km [95% CI -25.0, -16.4] — **significant — persistence-motion is better**
- 12h: lstm_direct beats persistence-motion by -38.3 km [95% CI -51.9, -25.8] — **significant — persistence-motion is better**
- 24h: lstm_direct beats persistence-motion by -41.7 km [95% CI -63.7, -18.1] — **significant — persistence-motion is better**
- 6h: lstm_motion_resid beats persistence-motion by -16.0 km [95% CI -21.6, -11.0] — **significant — persistence-motion is better**
- 12h: lstm_motion_resid beats persistence-motion by -31.4 km [95% CI -39.7, -22.8] — **significant — persistence-motion is better**
- 24h: lstm_motion_resid beats persistence-motion by -30.4 km [95% CI -41.7, -16.3] — **significant — persistence-motion is better**
- 6h: analog beats persistence-motion by -14.2 km [95% CI -17.2, -11.0] — **significant — persistence-motion is better**
- 12h: analog beats persistence-motion by -22.1 km [95% CI -29.3, -15.3] — **significant — persistence-motion is better**
- 24h: analog beats persistence-motion by -36.6 km [95% CI -55.6, -15.5] — **significant — persistence-motion is better**
- 6h: gbm_motion_resid beats true CLIPER by -1.1 km [95% CI -3.2, +0.9] — not significant (CI crosses 0)
- 12h: gbm_motion_resid beats true CLIPER by -3.7 km [95% CI -8.6, +0.8] — not significant (CI crosses 0)
- 24h: gbm_motion_resid beats true CLIPER by -4.9 km [95% CI -13.8, +4.8] — not significant (CI crosses 0)

## Intensity — Δwind MAE, kt (primary intersection)

| model | 6 h | 12 h | 24 h |
|---|---|---|---|
| persistence | 3.96 [95% CI 3.03, 4.89] | 7.71 [95% CI 5.92, 9.61] | 14.11 [95% CI 10.42, 18.02] |
| gbm | 4.22 [95% CI 3.32, 5.14] | 7.60 [95% CI 5.64, 9.45] | 13.14 [95% CI 9.66, 16.13] |
| analog | 3.83 [95% CI 3.21, 4.47] | 7.14 [95% CI 5.94, 8.49] | 13.32 [95% CI 10.91, 16.02] |

**Paired differences vs persistence (Δ=0):**

- 6h: gbm beats persistence by -0.26 kt [95% CI -1.28, +0.64] — not significant (CI crosses 0)
- 12h: gbm beats persistence by +0.11 kt [95% CI -2.13, +2.13] — not significant (CI crosses 0)
- 24h: gbm beats persistence by +0.97 kt [95% CI -2.24, +4.41] — not significant (CI crosses 0)
- 6h: analog beats persistence by +0.13 kt [95% CI -0.29, +0.58] — not significant (CI crosses 0)
- 12h: analog beats persistence by +0.57 kt [95% CI -0.35, +1.48] — not significant (CI crosses 0)
- 24h: analog beats persistence by +0.79 kt [95% CI -1.44, +2.97] — not significant (CI crosses 0)
