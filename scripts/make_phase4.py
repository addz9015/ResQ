"""docs/history/phase4.md - what each phase-4 fix changed, with before/after numbers."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAG = ROOT / "reports" / "_fragments"
OUT = ROOT / "docs" / "history" / "phase4.md"


def j(n):
    return json.loads((FRAG / n).read_text(encoding="utf-8"))


def main():
    m = j("models.json")
    unc = j("uncertainty.json")
    ri = j("ri.json")
    era5 = j("era5.json")
    tc = m["true_cliper"]
    cv, ts = tc["pre2019_groupcv_track_km"], tc["test_primary_intersection_track_km"]
    p = m["primary_intersection"]["rows"]

    L = []
    L.append("# ResQ — phase-4 changes\n")
    L.append("_Three fixes, one data-acquisition layer, one dashboard. Before/after "
             "numbers below; full tables in `reports/results.md`. All test numbers "
             "are on the 2019+ storms._\n")

    # FIX 1
    L.append("## Fix 1 — the motion baseline was misnamed\n")
    L.append("**Before:** phase-3 called its baseline \"CLIPER\". It was only "
             "persistence-of-motion — extrapolate the last 6 h `(Δlat, Δlon)`.\n")
    L.append("**After:**\n"
             "- renamed **persistence-motion**;\n"
             "- added **true CLIPER**: OLS of lead-h displacement on "
             f"`[{', '.join(tc['features'])}]`, fit on {tc['fit_storms']} pre-2019 "
             "storms;\n"
             "- both scored on the primary intersection;\n"
             "- Correction-3 residual learning now targets whichever wins the 2019+ "
             f"primary intersection (**{tc['residual_learning_baseline'].replace('_',' ')}**).\n")
    L.append("| baseline, track km | CV 6/12/24h | test 6/12/24h |")
    L.append("|---|---|---|")
    for k, lab in [("persistence_motion", "persistence-motion"), ("true_cliper", "true CLIPER")]:
        L.append(f"| {lab} | {cv[k]['6h']:.0f} / {cv[k]['12h']:.0f} / {cv[k]['24h']:.0f} "
                 f"| {ts[k]['6h']:.0f} / {ts[k]['12h']:.0f} / {ts[k]['24h']:.0f} |")
    L.append("")
    L.append(f"**Result:** true CLIPER wins the pre-2019 CV "
             f"({cv['true_cliper']['24h']:.0f} vs {cv['persistence_motion']['24h']:.0f} km "
             f"at 24 h) but **not the held-out 2019+ test** — persistence-motion is "
             f"level or better at 6/12 h. `true_cliper_wins_pre2019_cv = "
             f"{tc['true_cliper_wins_pre2019_cv']}`, "
             f"`true_cliper_wins_2019plus_test = {tc['true_cliper_wins_2019plus_test']}`. "
             "The residual models "
             f"(gbm {p['gbm_motion_resid']['track']['24h']:.0f}, "
             f"lstm {p['lstm_motion_resid']['track']['24h']:.0f} km at 24 h) still "
             f"do not beat the bare baseline "
             f"({min(ts['true_cliper']['24h'], ts['persistence_motion']['24h']):.0f} km). "
             "**Phase-3 conclusion holds and is stronger:** with kinematic features "
             "there is no learnable structure in either motion baseline's residual.\n")

    # FIX 2
    L.append("## Fix 2 — conformal calibration was data-starved\n")
    L.append(f"**Before:** split-CQR calibrated on the fixed 2017–2018 bucket — "
             f"{unc['n_states_with_features']['calib_2017_18']} states / 18 storms. "
             "The 24 h Δwind 80% interval reached only ~76% coverage.\n")
    L.append("**After:** cross-conformal for the Δwind targets — K=10 GroupKFold "
             f"over all {unc['n_states_with_features']['pre2019_total']} pre-2019 "
             "states, pooled conformity scores, quantile GBM refit on all pre-2019.\n")
    L.append("| Δwind lead | nominal | marginal | split-CQR | cross-conformal |")
    L.append("|---|---|--:|--:|--:|")
    for h in ("6h", "12h", "24h"):
        for lvl in ("50", "80"):
            e = unc["per_target"][f"delta_wind_{h}"][f"nominal_{lvl}"]
            L.append(f"| {h} | {lvl}% | {e['marginal_coverage']:.2f} | "
                     f"{e['split_cqr_coverage']:.2f} | "
                     f"{e.get('cross_conformal_coverage', float('nan')):.2f} |")
    e80 = unc["per_target"]["delta_wind_24h"]["nominal_80"]
    cc = e80.get("cross_conformal_coverage", 0)
    L.append(f"\n**Result:** cross-conformal 24 h nominal-80% coverage = "
             f"**{cc:.0%}** — within a point of split-CQR "
             f"({e80['split_cqr_coverage']:.0%}) and still short of 80%. "
             "**The extra calibration data (237 storms vs 18) does not close the "
             "gap**, so the shortfall is not calibration-set size — it is "
             "conditional coverage (the 24 h quantile intervals are too narrow for "
             "the hard cases; a global conformal offset cannot fix that). Both "
             f"methods do lift the marginal {e80['marginal_coverage']:.0%} to "
             "~76%. Honest answer to 'does it reach nominal at 24 h': no.\n")

    # FIX 3
    L.append("## Fix 3 — combine the two RI signals\n")
    L.append("**Before:** phase-3 reported analog P(RI) and the classifier "
             "separately (PR-AUC ≈ 0.23 and 0.16); no combination.\n")
    L.append("**After:** both isotonic-calibrated on K=10 pre-2019 out-of-fold "
             "predictions, then a simple average and a logistic stack.\n")
    L.append("| forecast | ROC-AUC | PR-AUC | Brier |")
    L.append("|---|--:|--:|--:|")
    for k, lab in [("analog_p_ri_calibrated", "analog (calibrated)"),
                   ("ri_classifier_calibrated", "classifier (calibrated)"),
                   ("average", "average"), ("logistic_stack", "logistic stack")]:
        e = ri[k]
        L.append(f"| {lab} | {e['roc_auc']:.3f} | {e['pr_auc']:.3f} | {e['brier']:.4f} |")
    a = ri["average"]
    L.append(f"\n**Result:** the **simple average has the best point estimate** — "
             f"ROC-AUC {a['roc_auc']:.2f}, PR-AUC {a['pr_auc']:.2f}, Brier "
             f"{a['brier']:.4f}, ahead of both inputs on all three metrics. The "
             "logistic stack over-weights the classifier and is worse on Brier.\n")
    if (FRAG / "bootstrap.json").exists():
        b = j("bootstrap.json")["ri"]["paired"]
        da = b["average_minus_analog_pr_auc"]; dc = b["average_minus_classifier_pr_auc"]
        L.append(f"Phase-5 cluster bootstrap (`reports/significance.md`): "
                 f"average **significantly** beats the classifier "
                 f"(+{dc['improvement']:.3f} PR-AUC [95% CI {dc['lo']:.3f}, "
                 f"{dc['hi']:.3f}]) but **not** the analog ensemble alone "
                 f"(+{da['improvement']:.3f} [{da['lo']:.3f}, {da['hi']:.3f}], crosses "
                 f"0) — only {j('bootstrap.json')['ri']['n_RI_positive']} RI cases in "
                 "the test set. Use the average as a robust default; the "
                 "analog-alone vs average question needs more positive examples "
                 "(i.e. ERA5-era data) to settle.\n")

    # DATA
    L.append("## Data — ERA5 environmental predictors (`src/era5.py`)\n")
    L.append(f"Built now, on the critical path. Status: **{era5['status']}**.\n"
             f"- {len(era5['field_coverage'])} env columns "
             "(shear / steering z500 / RH600 / SST × centre + 200/500/800 km "
             f"annuli) written to `{era5['output']}`, currently NULL.\n"
             f"- {era5['months_required']} monthly CDS requests would be needed; "
             "cached per month in `data/era5_cache/`.\n"
             "- Fails cleanly: prints the exact missing pieces "
             "(cdsapi/xarray libs + `~/.cdsapirc`), writes null columns, exits 0. "
             "No model consumes them yet.\n"
             "- Missing fields are left null — never interpolated across gaps.\n")

    # APP
    L.append("## New deliverable — operational dashboard (`app/`)\n")
    L.append("`python app/run.py` → builds the engine + replay hindcasts (~2 min) "
             "and serves **http://127.0.0.1:8000**. Offline, no CDS calls.\n"
             "- **FastAPI**: `POST /api/forecast` (state → track + cone + Δwind + "
             "P(RI) + top-10 analogs), `GET /api/replay/{storm_id}` (step-by-step "
             "hindcast), `GET /api/replay` (storm list).\n"
             "- **Leaflet frontend**: observed track, forecast track (true CLIPER), "
             "80% conformal cone from the along/cross-track quantiles, the actual "
             "future revealed as you step, analog table, and a P(RI) gauge with the "
             "climatological base rate marked.\n"
             "- **Replay mode** steps a 2019+ holdout storm forward 6 h at a time, "
             "showing the forecast that *would* have been issued and the track "
             "error once each verifying fix is reached. This is the demo.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
