# ResQ — limitations

_One page. Every number traces to `reports/_fragments/*.json` or the parquet; see `reports/results.md`, `significance.md`, `fit_diagnostics.md`, `audit.md` for the working._

## 1. No environmental fields
The models use **kinematic features only** — position, wind, pressure, motion, storm age, season. There is **no ERA5** (vertical wind shear, SST, mid-level humidity, steering flow) and no satellite structure. `src/era5.py` is built and wired to `data/cyclone_6h_with_env.parquet` but its status is **unavailable** — CDS credentials / libraries were not available, so those columns are NULL and no model consumes them. This is the single biggest cap on intensity and rapid-intensification skill.

## 2. No learned track model beats persistence-of-motion
On the 2019+ test the best track baseline is persistence-motion (extrapolate the last 6 h vector): **33 / 66 / 144 km** at 6 / 12 / 24 h (24 h 95% CI 128–167 km). Every learned track model — GBM or LSTM, direct or CLIPER-residual — is **as good or significantly worse** (`reports/significance.md`). Track guidance in the UI is the baseline, not a model.

## 3. 24 h intensity intervals under-cover
The 80 %-nominal predictive interval for the 24 h wind change empirically covers only **75%** on the test set (marginal 60% → split-CQR 76% → cross-conformal 75%). Both conformal methods lift coverage but neither reaches nominal — the gap is conditional coverage (intervals too tight for the hard cases), not calibration-set size.

## 4. Rare high-end events
Only **6 super cyclonic storms** (peak wind ≥ 120 kt) exist in the entire 425-storm IMD archive used here, and just **48 rapid-intensification cases** in the 71-storm test set. The models have almost no examples of the most dangerous behaviour, and the learning curve (`docs/history/phase6.md` M2) is still rising — this is a data-starved regime for the high end.

## 5. Interpolated wind targets
The 6 h grid is built by resampling irregular IMD fixes. About **12% of the 6/12/24 h wind targets** used in evaluation are linear interpolations between real fixes (flagged `interpolated` in the parquet). Correction-4 in `reports/results.md` re-scores with those excluded and the model ranking does not change, but the absolute error numbers include some interpolated ground truth.

## 6. Overfitting in the flexible models
`docs/history/fit_diagnostics.md` flags **7 of the fitted models** (the GBM Δwind and track-residual regressors, the RI classifier) as overfitting — train score far above out-of-fold, OOF ≈ test. The trivial / linear models (persistence, true CLIPER, analog) do not overfit. Two effects stack: weak features **and** flexible models memorising the little signal there is.

## 7. The synthetic background file is excluded
`normal_weather_background.csv` (9000 rows) is used **nowhere** in training or evaluation. A classifier separates it from real cyclone data with **ROC-AUC 1.00** even after dropping provenance columns — it is distinguishable by construction (continuous vs 5-kt-quantised winds, all below tropical-storm force, sentinel storm ids). Any model trained on it would learn the synthetic/real boundary instead of meteorology. Full reproduction: `reports/audit.md` §6.

## 8. What the citizen features can and cannot do
- **Preparation Checklist** shows steps only from NDMA's published cyclone guidance. **51 items** are loaded, harvested verbatim from NDMA's Drupal REST API (nodes 118 *Cyclone: Do's & Don'ts* and 87 *Cyclone*), each stored with its source URL and retrieval date. NDMA's guidance is not graded by cyclone category, so every item shows from Depression upward; the intensity input is context only. Coastal-vs-inland filtering is a literal text match on coast/beach/sea/surge, not a judgement.
- **District Cyclone History** uses real GADM level-2 boundaries; distance is a ~1%-accurate equirectangular approximation and the count is "a track passed within 100 km", **not** a modelled impact. It is a historical record, not a forecast.
- **Official Warnings** parses bulletin text the user pastes (`src/imd_bulletin.py` `parse_text`, deterministic). It never infers a colour, district or landfall time that is not literally in the text, and returns `null` for anything it cannot find. **There is no live IMD feed** — IMD/RSMC New Delhi have no API and the transient operational bulletins are not archived machine-readably; the paste box is the feature.
- **Chatbot** is retrieval-only over `reports/`, the storm archive, district history and the loaded NDMA guidance; every answer cites its source. A post-generation guard discards any Groq response that invents a warning colour, a specific future time or a numeric forecast, falling back to the offline answer. Preparedness questions are answered only from NDMA passages. Groq is optional phrasing, off by default.

## 9. Not a warning service
ResQ replays historical storms and evaluates models. It is **not** connected to live data, it does **not** issue warnings, colour codes or district advisories, and it has **no affiliation with or endorsement by IMD**. For live cyclone warnings, see IMD.
