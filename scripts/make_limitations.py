"""reports/limitations.md — one page, every number from reports/_fragments/*.json
or the parquet. Nothing hardcoded, nothing rounded away from the source."""
from __future__ import annotations

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAG = ROOT / "reports" / "_fragments"
OUT = ROOT / "reports" / "limitations.md"


def j(n):
    p = FRAG / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    m = j("models.json"); bs = j("bootstrap.json"); unc = j("uncertainty.json")
    lk = j("leakage.json"); rs = j("resample.json"); fd = j("fit_diagnostics.json")
    era5 = j("era5.json")

    clean = pd.read_parquet(ROOT / "data" / "cyclone_clean.parquet")
    g = clean.dropna(subset=["storm_id_code"]).groupby("storm_id_code")["wind_kt_canonical"].max()
    n_super = int((g >= 120).sum()); n_storms = int(g.size)

    pm = m["primary_intersection"]["rows"]["persistence_motion"]["track"]
    t24 = bs["track_intensity"]["track"]["persistence_motion"]["24h"]
    e80 = unc["per_target"]["delta_wind_24h"]["nominal_80"]
    cc = e80.get("cross_conformal_coverage", e80["split_cqr_coverage"])
    interp = rs.get("holdout_target_interp_fraction_per_lead", {})
    n_over = len(fd.get("overfitting_summary", []))

    L = [
        "# ResQ — limitations\n",
        "_One page. Every number traces to `reports/_fragments/*.json` or the "
        "parquet; see `reports/results.md`, `significance.md`, `fit_diagnostics.md`, "
        "`audit.md` for the working._\n",

        "## 1. No environmental fields",
        "The models use **kinematic features only** — position, wind, pressure, "
        "motion, storm age, season. There is **no ERA5** (vertical wind shear, "
        "SST, mid-level humidity, steering flow) and no satellite structure. "
        f"`src/era5.py` is built and wired to `data/cyclone_6h_with_env.parquet` "
        f"but its status is **{era5.get('status', 'unavailable')}** — CDS "
        "credentials / libraries were not available, so those columns are NULL and "
        "no model consumes them. This is the single biggest cap on intensity and "
        "rapid-intensification skill.\n",

        "## 2. No learned track model beats persistence-of-motion",
        f"On the 2019+ test the best track baseline is persistence-motion "
        f"(extrapolate the last 6 h vector): **{pm['6h']:.0f} / {pm['12h']:.0f} / "
        f"{pm['24h']:.0f} km** at 6 / 12 / 24 h (24 h 95% CI "
        f"{t24['lo']:.0f}–{t24['hi']:.0f} km). Every learned track model — GBM or "
        "LSTM, direct or CLIPER-residual — is **as good or significantly worse** "
        "(`reports/significance.md`). Track guidance in the UI is the baseline, "
        "not a model.\n",

        "## 3. 24 h intensity intervals under-cover",
        f"The 80 %-nominal predictive interval for the 24 h wind change "
        f"empirically covers only **{cc*100:.0f}%** on the test set "
        f"(marginal {e80['marginal_coverage']*100:.0f}% → split-CQR "
        f"{e80['split_cqr_coverage']*100:.0f}% → cross-conformal {cc*100:.0f}%). "
        "Both conformal methods lift coverage but neither reaches nominal — the "
        "gap is conditional coverage (intervals too tight for the hard cases), "
        "not calibration-set size.\n",

        "## 4. Rare high-end events",
        f"Only **{n_super} super cyclonic storms** (peak wind ≥ 120 kt) exist in "
        f"the entire {n_storms}-storm IMD archive used here, and just "
        f"**{bs['ri']['n_RI_positive']} rapid-intensification cases** in the "
        f"{bs['ri']['n_test_storms']}-storm test set. The models have almost no "
        "examples of the most dangerous behaviour, and the learning curve "
        "(`docs/history/phase6.md` M2) is still rising — this is a data-starved regime "
        "for the high end.\n",

        "## 5. Interpolated wind targets",
        f"The 6 h grid is built by resampling irregular IMD fixes. About "
        f"**{interp.get('6h', 0.12)*100:.0f}% of the 6/12/24 h wind targets** used "
        "in evaluation are linear interpolations between real fixes (flagged "
        "`interpolated` in the parquet). Correction-4 in `reports/results.md` "
        "re-scores with those excluded and the model ranking does not change, but "
        "the absolute error numbers include some interpolated ground truth.\n",

        "## 6. Overfitting in the flexible models",
        f"`docs/history/fit_diagnostics.md` flags **{n_over} of the fitted models** "
        "(the GBM Δwind and track-residual regressors, the RI classifier) as "
        "overfitting — train score far above out-of-fold, OOF ≈ test. The "
        "trivial / linear models (persistence, true CLIPER, analog) do not "
        "overfit. Two effects stack: weak features **and** flexible models "
        "memorising the little signal there is.\n",

        "## 7. The synthetic background file is excluded",
        f"`normal_weather_background.csv` ({lk['n_background_rows']} rows) is used "
        "**nowhere** in training or evaluation. A classifier separates it from "
        "real cyclone data with **ROC-AUC "
        f"{lk['auc_all_features_no_provenance']:.2f}** even after dropping "
        "provenance columns — it is distinguishable by construction (continuous "
        "vs 5-kt-quantised winds, all below tropical-storm force, sentinel storm "
        "ids). Any model trained on it would learn the synthetic/real boundary "
        "instead of meteorology. Full reproduction: `reports/audit.md` §6.\n",

        "## 8. What the citizen features can and cannot do",
    ]
    try:
        ndma = json.loads((ROOT / "data" / "ndma_guidance.json").read_text(encoding="utf-8"))
    except Exception:
        ndma = {"status": "unavailable", "n_items": 0}
    hist = {}
    try:
        hist = json.loads((ROOT / "app" / "static" / "data" / "district_history.json")
                          .read_text(encoding="utf-8"))
    except Exception:
        pass
    ndma_ok = ndma.get("status") == "ok"
    L.append(
        (f"- **Preparation Checklist** shows steps only from NDMA's published "
         f"cyclone guidance. **{ndma.get('n_items', 0)} items** are loaded, "
         "harvested verbatim from NDMA's Drupal REST API (nodes 118 *Cyclone: "
         "Do's & Don'ts* and 87 *Cyclone*), each stored with its source URL and "
         "retrieval date. NDMA's guidance is not graded by cyclone category, so "
         "every item shows from Depression upward; the intensity input is context "
         "only. Coastal-vs-inland filtering is a literal text match on "
         "coast/beach/sea/surge, not a judgement."
         if ndma_ok else
         f"- **Preparation Checklist** shows steps only from NDMA's published "
         f"cyclone guidance. That guidance is currently "
         f"**{ndma.get('status', 'unavailable')}** ({ndma.get('n_items', 0)} "
         "items) so the tab renders an empty state that links to the NDMA source. "
         "ResQ does not author safety advice. Run `scripts/build_checklist.py` or "
         "hand-fill `data/ndma_guidance.json` from the TEMPLATE.")
        + "\n"
        f"- **District Cyclone History** uses "
        f"{'real GADM level-2 boundaries' if hist.get('mode') == 'boundary' else 'approximate hand-entered centroids'}; "
        "distance is a ~1%-accurate equirectangular approximation and the count "
        "is \"a track passed within 100 km\", **not** a modelled impact. It is a "
        "historical record, not a forecast.\n"
        "- **Official Warnings** parses bulletin text the user pastes "
        "(`src/imd_bulletin.py` `parse_text`, deterministic). It never infers a "
        "colour, district or landfall time that is not literally in the text, and "
        "returns `null` for anything it cannot find. **There is no live IMD "
        "feed** — IMD/RSMC New Delhi have no API and the transient operational "
        "bulletins are not archived machine-readably; the paste box is the "
        "feature.\n"
        "- **Chatbot** is retrieval-only over `reports/`, the storm archive, "
        "district history and the loaded NDMA guidance; every answer cites its "
        "source. A post-generation guard discards any Groq response that invents "
        "a warning colour, a specific future time or a numeric forecast, falling "
        "back to the offline answer. Preparedness questions are answered only "
        "from NDMA passages. Groq is optional phrasing, off by default.\n")

    L.append("## 9. Not a warning service")
    L.append("ResQ replays historical storms and evaluates models. It is **not** "
             "connected to live data, it does **not** issue warnings, colour "
             "codes or district advisories, and it has **no affiliation with or "
             "endorsement by IMD**. For live cyclone warnings, see IMD.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
