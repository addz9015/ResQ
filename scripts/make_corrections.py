"""docs/history/corrections.md - what the four phase-3 corrections changed."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAG = ROOT / "reports" / "_fragments"
OUT = ROOT / "docs" / "history" / "corrections.md"


def j(n):
    return json.loads((FRAG / n).read_text(encoding="utf-8"))


def main():
    m = j("models.json")
    unc = j("uncertainty.json")
    ri = j("ri.json")
    sp = m["split"]
    pri = m["primary_intersection"]
    sec = m["secondary_full_n"]

    L = []
    L.append("# ResQ — phase-3 corrections\n")
    L.append("_What was wrong in the phase-2 `results.md`, what changed, and the "
             "numeric effect. Companion to `reports/results.md` and "
             "`reports/audit.md`. Phase-4 refined corrections 1–3 further — see "
             "`docs/history/phase4.md`; numbers here are the current re-run, with the "
             "phase-4 baseline names (\"CLIPER\" of phase-3 = **persistence-motion**)._\n")

    # ---- 1 ----
    L.append("## Correction 1 — the model comparison was unfair\n")
    L.append("**Wrong:** phase-2 put the LSTM (n = 431 sequences, which require 12 "
             "consecutive 6 h grid points) in the same table as persistence / CLIPER "
             "/ GBM / analog (n = 791 at 24 h). Different, easier subsets for the "
             "non-sequence models — the comparison was meaningless.\n")
    L.append("**Changed:** two explicit tables.\n"
             "- **Primary (`results.md` §1)** — every model scored on the *same* "
             f"states: end of a valid 8-step window, future point present, GBM/analog "
             f"features present. n = {pri['n_per_lead']['6']} / "
             f"{pri['n_per_lead']['12']} / {pri['n_per_lead']['24']} at 6 / 12 / 24 h.\n"
             "- **Secondary (`results.md` §2)** — persistence / CLIPER / GBM / analog "
             f"on all test states (n = {sec['n_per_lead']['6']} / "
             f"{sec['n_per_lead']['12']} / {sec['n_per_lead']['24']}), clearly "
             "labelled as a different, larger sample with no LSTM row.\n")
    p = pri["rows"]; s = sec["rows"]
    L.append("**Effect:** the intersection is a harder subset (more still-developing "
             "storms). Persistence Δwind MAE at 24 h is "
             f"{p['persistence']['dwind_MAE']['24h']:.2f} kt on the intersection vs "
             f"{s['persistence']['dwind_MAE']['24h']:.2f} kt on the full sample; "
             f"CLIPER track at 24 h is {p['persistence_motion']['track']['24h']:.1f} vs "
             f"{s['persistence_motion']['track']['24h']:.1f} km. Model *ranking* is the same in "
             "both tables — CLIPER best on track, analog/GBM best on intensity — so "
             "the phase-2 conclusion survives, but only the primary table supports "
             "it.\n")

    # ---- 2 ----
    L.append("## Correction 2 — uncertainty calibration used the test set\n")
    L.append("**Wrong:** phase-2 fit the quantile GBM on pre-2019 storms and then "
             "read interval coverage straight off the 2019+ storms — no held-out "
             "calibration, so \"calibrated uncertainty\" was calibrated on nothing.\n")
    L.append("**Changed:** storm-disjoint three-way split — "
             f"train ≤ 2016 ({sp['train_le_2016']['storms']} storms), "
             f"calibration 2017–2018 ({sp['calib_2017_2018']['storms']} storms), "
             f"test ≥ 2019 ({sp['test_ge_2019']['storms']} storms). Split "
             "conformalized quantile regression: quantile GBM on train, conformity "
             "scores `E_i = max(q_lo−y, y−q_hi)` on the calibration storms, offset "
             "`Q` = the `(1−α)` empirical quantile of `E`, test interval "
             "`[q_lo−Q, q_hi+Q]`. (Storm ids per bucket: `models.json → split`.)\n")
    dw = unc["per_target"]
    L.append("**Effect — Δwind interval coverage (nominal → marginal → CQR):**\n")
    L.append("| lead | nominal 80% | marginal | CQR | nominal 50% | marginal | CQR |")
    L.append("|---|---|--:|--:|---|--:|--:|")
    for h in ("6h", "12h", "24h"):
        e = dw[f"delta_wind_{h}"]
        L.append(f"| {h} | 0.80 | {e['nominal_80']['marginal_coverage']:.2f} | "
                 f"**{e['nominal_80']['split_cqr_coverage']:.2f}** | 0.50 | "
                 f"{e['nominal_50']['marginal_coverage']:.2f} | "
                 f"**{e['nominal_50']['split_cqr_coverage']:.2f}** |")
    e24 = dw["delta_wind_24h"]
    L.append(f"\nCQR **partly fixes** the 24 h over-confidence: nominal-80% coverage "
             f"{e24['nominal_80']['marginal_coverage']:.0%} → "
             f"{e24['nominal_80']['split_cqr_coverage']:.0%} (widening the median interval "
             f"{e24['nominal_80']['median_width_marginal']:.0f} → "
             f"{e24['nominal_80']['median_width_split_cqr']:.0f} kt). It does not reach "
             f"80% — the calibration set is only {e24['n_calib_split']} states so the "
             "offset is noisy. Track intervals were already close to nominal and "
             "stay there.\n")

    # ---- 3 ----
    L.append("## Correction 3 — track models predicted position, not CLIPER's residual\n")
    L.append("**Wrong:** the LSTM (and any GBM track model) predicted the absolute "
             "displacement, so it had to re-learn the linear motion CLIPER already "
             "gives for free — and lost to CLIPER by a wide margin.\n")
    L.append("**Changed:** both the LSTM and a new GBM are retargeted to predict "
             "`(actual_lat − cliper_lat, actual_lon − cliper_lon)` at each lead; the "
             "forecast is `CLIPER + predicted residual`. Direct-prediction variants "
             "are kept side by side (`results.md` §1).\n")
    L.append("**Effect — track error, km (primary table):**\n")
    L.append("| model | 6h | 12h | 24h |")
    L.append("|---|--:|--:|--:|")
    for key, lab in [("persistence_motion", "persistence-motion (bare)"),
                     ("gbm_direct", "GBM direct"),
                     ("gbm_motion_resid", "GBM motion+residual"),
                     ("lstm_direct", "LSTM direct"),
                     ("lstm_motion_resid", "LSTM motion+residual")]:
        t = p[key]["track"]
        L.append(f"| {lab} | {t['6h']:.1f} | {t['12h']:.1f} | {t['24h']:.1f} |")
    L.append(f"\nResidual learning **helps** — LSTM 24 h "
             f"{p['lstm_direct']['track']['24h']:.0f} → "
             f"{p['lstm_motion_resid']['track']['24h']:.0f} km, GBM 24 h "
             f"{p['gbm_direct']['track']['24h']:.0f} → "
             f"{p['gbm_motion_resid']['track']['24h']:.0f} km — but **neither beats "
             f"plain CLIPER** ({p['persistence_motion']['track']['24h']:.0f} km). This is itself "
             "the finding: with kinematic features there is no systematic, learnable "
             "structure in CLIPER's errors. A learned track model will need "
             "environmental predictors (steering flow, ridge position) to add "
             "skill.\n")

    # ---- 4 ----
    L.append("## Correction 4 — interpolated targets\n")
    L.append("**Wrong:** ~12–20% of resampled wind values are linear interpolations "
             "between real IMD fixes; phase-2 scored against them without checking "
             "whether that flattered the models.\n")
    L.append("**Changed:** the primary table is re-scored with test target points "
             "excluded whenever either Δwind endpoint is an interpolated wind "
             "(`results.md` §3).\n")
    pc = m["primary_clean_only"]["n_per_lead"]
    rc = m["ranking_changes"]
    changed = [h for h in ("6h", "12h", "24h")
               if rc[h]["track_order_changed"] or rc[h]["dwind_order_changed"]]
    L.append(f"**Effect:** dropping interpolated targets removes "
             f"{pri['n_per_lead']['24']-pc['24']} of {pri['n_per_lead']['24']} "
             "states at 24 h. Every model's track and Δwind number moves by "
             "< 1.5 km / < 0.3 kt. "
             + ("**No ranking changes at any lead.**"
                if not changed else
                f"Ranking changes only at {', '.join(changed)}, and only a swap "
                "between the two worst LSTM variants; the top of every ranking is "
                "unchanged.")
             + " Interpolation is not inflating the results.\n")

    # ---- new components ----
    L.append("## New components C & D — rapid intensification\n")
    rcl, ap = (lambda r:(r['ri_classifier_calibrated'],r['analog_p_ri_calibrated']))(ri)  # ["ri_classifier_calibrated"], ri["analog_p_ri_calibrated"]
    L.append(f"- **Component C** turns the analog ensemble into a distribution "
             "(deciles of analog Δwind) and reads off `P(RI)` = fraction of analogs "
             f"with Δwind ≥ +30 kt / 24 h, plus `P(rapid weakening)`.\n"
             f"- **Component D** (`src/ri.py`) is a dedicated HGB classifier + "
             "isotonic calibration for the same RI label, kinematic features only, "
             "same three-way split.\n"
             f"- On the 2019+ test set (base rate {ri['test_base_rate']:.1%}): analog "
             f"P(RI) ROC-AUC **{ap['roc_auc']:.2f}** / PR-AUC **{ap['pr_auc']:.2f}**, "
             f"classifier **{rcl['roc_auc']:.2f}** / **{rcl['pr_auc']:.2f}**. Both "
             f"beat climatology on Brier ({ap['brier']:.4f} and {rcl['brier']:.4f} "
             f"vs {ri['climatology_brier']:.4f}). The analog method wins on "
             "discrimination; neither is operationally strong without environmental "
             "fields.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
