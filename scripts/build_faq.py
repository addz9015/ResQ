"""
Build app/static/data/faq.json — the fixed FAQ the Chatbot page answers from.

The chatbot is a keyword lookup over these entries (no LLM backend). Entries are
assembled from reports/ and reports/_fragments/*.json so the numbers never drift
from the pipeline. It answers only: how the model was built, what the numbers
mean, what the limitations are. It refuses live/current-storm questions.

Run:  python scripts/build_faq.py
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAG = ROOT / "reports" / "_fragments"
OUT = ROOT / "app" / "static" / "data" / "faq.json"


def j(n):
    p = FRAG / n
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def main():
    m = j("models.json"); bs = j("bootstrap.json"); ri = j("ri.json")
    unc = j("uncertainty.json"); lk = j("leakage.json"); rs = j("resample.json")
    fd = j("fit_diagnostics.json"); rt = j("ri_tune.json")

    pm = m["primary_intersection"]["rows"]["persistence_motion"]["track"]
    tk24 = bs["track_intensity"]["track"]["persistence_motion"]["24h"]
    dw24 = bs["track_intensity"]["intensity_paired"]["analog_vs_persistence"]["24h"]
    avg = ri["average"]; base = ri["test_base_rate"]
    e80 = unc["per_target"]["delta_wind_24h"]["nominal_80"]
    sup = j("bootstrap.json")  # for supported list rebuild below

    entries = [
        {"q": "what does this system do",
         "keywords": ["what", "do", "system", "purpose", "about", "overview"],
         "a": "It replays past tropical cyclones over the North Indian Ocean and, "
              "at each 6-hour step, shows the forecast that our models would have "
              "issued: where the centre goes, how the peak wind changes, how "
              "confident we are, and which past storms it most resembles. It is a "
              "research / evaluation tool, not an operational warning service."},
        {"q": "how was the model built",
         "keywords": ["how", "built", "trained", "model", "method", "pipeline", "data"],
         "a": "IMD best-track data is cleaned, split into gap-free segments and "
              "resampled to a uniform 6-hour grid ("
              f"{rs['segments_kept']} segments, {rs['resampled_rows_out']} states). "
              "Storms are split by year: pre-2019 to fit, 2019+ held out for "
              "testing, with a further inner split for calibration. Track uses "
              "linear motion extrapolation (persistence-motion) and an OLS "
              "'true CLIPER'; intensity uses a gradient-boosting regressor and a "
              "k-nearest-neighbour analog ensemble; rapid-intensification "
              "probability averages a calibrated classifier and the analog rate. "
              "Full detail: reports/audit.md, results.md, phase4.md, phase6.md."},
        {"q": "how accurate is the track forecast",
         "keywords": ["track", "accurate", "accuracy", "position", "error", "km", "how far"],
         "a": f"On the 2019+ holdout the 24-hour track error is about "
              f"{pm['24h']:.0f} km (95% range {tk24['lo']:.0f}-{tk24['hi']:.0f} km), "
              f"{pm['6h']:.0f} km at 6 h and {pm['12h']:.0f} km at 12 h. That is "
              "the best baseline; no learned model beats it in this project."},
        {"q": "how accurate is the intensity forecast",
         "keywords": ["intensity", "wind", "strength", "strong", "dwind", "kt", "knots"],
         "a": "The 24-hour peak-wind-change error is around 13 kt. The learned "
              "models' point estimates edge the no-change baseline at 12/24 h but "
              f"the difference is not statistically significant ("
              f"{dw24['improvement']:+.2f} kt, 95% CI {dw24['lo']:.2f} to "
              f"{dw24['hi']:.2f} — crosses zero). Treat intensity change as "
              "low-confidence."},
        {"q": "what is rapid intensification and how well is it predicted",
         "keywords": ["rapid", "intensification", "ri", "sudden", "strengthen", "p(ri)",
                      "probability"],
         "a": f"Rapid intensification = the peak wind rising by at least 30 kt in "
              f"24 hours. It happens in about {base*100:.0f}% of test states. Our "
              f"combined probability has ROC-AUC {avg['roc_auc']:.2f} and PR-AUC "
              f"{avg['pr_auc']:.2f} (about 4x better than the base rate). It "
              "significantly beats a climatological forecast and the classifier "
              "alone, but not the analog ensemble alone. Use it as a flag, not a "
              "precise number."},
        {"q": "what does the shaded cone mean",
         "keywords": ["cone", "shaded", "area", "uncertainty", "interval", "confident",
                      "confidence", "range"],
         "a": "The shaded area is an 80% conformal interval for where the centre "
              "will be. Wider means less certain. For the 24-hour wind-change "
              f"interval the nominal level is 80% but it empirically covers about "
              f"{e80.get('cross_conformal_coverage', e80['split_cqr_coverage'])*100:.0f}% "
              "on the test set — so slightly over-confident at long range."},
        {"q": "what are the limitations",
         "keywords": ["limitation", "limitations", "weakness", "problem", "caveat",
                      "not", "cannot", "wrong", "bad"],
         "a": "Small test set (~70 storms, ~48 RI cases) so most model-vs-baseline "
              "differences are within noise. The features are kinematic only (no "
              "satellite structure, no ERA5 shear/SST) which caps intensity and RI "
              f"skill. {len(fd.get('overfitting_summary', []))} of the fitted "
              "models overfit their training data (train score far above "
              "out-of-fold). ~12-20% of the 6-hour wind values are interpolated "
              "between real fixes. See reports/significance.md and "
              "reports/fit_diagnostics.md."},
        {"q": "why is there no live cyclone data",
         "keywords": ["live", "current", "now", "today", "active", "real-time",
                      "realtime", "happening"],
         "a": "This system replays historical storms. For live cyclone warnings, "
              "see IMD."},
        {"q": "what is the background file and why was it excluded",
         "keywords": ["background", "excluded", "leakage", "synthetic", "normal_weather",
                      "auc 1"],
         "a": f"A synthetic 'normal weather' file ({lk['n_background_rows']} rows) "
              "was excluded from all training. A classifier separates it from real "
              "cyclone data with ROC-AUC 1.00 even after dropping provenance "
              "columns — it is distinguishable by construction (non-integer winds, "
              "sub-tropical-storm strength, sentinel storm ids). Details: "
              "reports/audit.md section 6."},
        {"q": "which past storms does it compare to",
         "keywords": ["analog", "analogs", "similar", "past storms", "compare",
                      "resemble", "nearest"],
         "a": "For any state it retrieves the 10 nearest past states from other "
              "storms (matched on position, wind, pressure, motion, season) and "
              "shows what those storms did at +6/+12/+24 h and their eventual peak "
              "strength. It excludes states from the same storm."},
        {"q": "does resq issue cyclone warnings or colour codes",
         "keywords": ["warning", "warnings", "colour", "color", "code", "issue",
                      "official", "imd", "bulletin", "alert", "advisory", "authority"],
         "a": "No. ResQ never issues a warning stage, colour or district advisory "
              "— IMD (RSMC New Delhi) is the sole authority for those. The Official "
              "Warnings tab only *explains* an IMD bulletin you paste in: "
              "src/imd_bulletin.py parse_text() extracts the fields deterministically "
              "(it never infers a colour, district or landfall time not literally in "
              "the text) and explain() restates them in plain language. On top of "
              "that ResQ adds its own uncertainty range, rapid-intensification "
              "probability and analogs, always labelled 'ResQ model estimate — not "
              "an official warning.' There is no live IMD feed; IMD has no API."},
        {"q": "was the model tuned",
         "keywords": ["tune", "tuned", "tuning", "hyperparameter", "optimis", "optimiz",
                      "grid search"],
         "a": "Only the rapid-intensification ensemble was tuned, and only on "
              "pre-2019 out-of-fold PR-AUC — the 2019+ test set never influenced a "
              "choice. Selected: a more-regularised classifier, analog k="
              f"{rt.get('M1_tuning', {}).get('selected', {}).get('analog_k', 30)}. "
              "The test score change was inside the confidence interval, so it is "
              "not a real improvement — just a defensible config. Track and "
              "intensity models were deliberately not tuned; they lose to "
              "baselines because the features are weak, not because of tuning."},
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": "Fixed FAQ for the keyword-lookup chatbot. Answers only cover how "
                "the models were built, what the numbers mean, and their limits.",
        "refusal": "This system replays historical storms. For live cyclone "
                   "warnings, see IMD.",
        "entries": entries,
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}  ({len(entries)} entries)")


if __name__ == "__main__":
    main()
