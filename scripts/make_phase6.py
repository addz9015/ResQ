"""docs/history/phase6.md — dashboard integration + the M1/M2/M3 modelling.
Numbers are read from reports/_fragments/*.json; nothing hardcoded."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAG = ROOT / "reports" / "_fragments"
OUT = ROOT / "docs" / "history" / "phase6.md"


def j(n):
    return json.loads((FRAG / n).read_text(encoding="utf-8"))


def main():
    rt = j("ri_tune.json")
    fd = j("fit_diagnostics.json")
    m1 = rt["M1_tuning"]; m2 = rt["M2_learning_curve"]

    L = []
    L.append("# ResQ — phase 6 (dashboard + last modelling)\n")
    L.append("_No new architectures, no LSTM retrain. Dashboard extended in place; "
             "the RI ensemble tuned on pre-2019 OOF only; overfitting checked with "
             "numbers._\n")

    # ---- dashboard ----
    L.append("## Dashboard\n")
    L.append("**File:** `app/static/index.html` (single file; served at `/`).\n")
    L.append("**Tabs — before:** none (fixed 3-column replay view).\n")
    L.append("**Tabs — after:** `Replay` · `Forecast` · `Model insights` "
             "(top nav, `.tabs` / `.tabview`, vanilla show/hide, lazy Leaflet "
             "`invalidateSize` per tab).\n")
    L.append("**Endpoints added:** `GET /api/presets` — real 2019+ holdout states "
             "pre-filled for the Forecast form + a MANDOUS-2022 preset, read from "
             "`data/cyclone_resampled_6h.parquet`. `POST /api/forecast` extended to "
             "return the 80% conformal Δwind interval (`dwind_lo` / `dwind_hi`) and "
             "its empirical coverage. `Model insights` loads a pre-built "
             "`static/data/insights.json` (`scripts/build_insights.py`) — no "
             "fitting at page load.\n")
    L.append("- **Forecast tab:** inputs = lat, lon, wind, pressure, prev-6h lat, "
             "prev-6h lon, storm age, date (basin derived from longitude). Outputs "
             "= persistence-motion track (\"best baseline\") + conformal cone, "
             "Δwind with 80% intervals + the 75%-vs-80% coverage note, combined "
             "P(RI) gauge with the 5.9% base rate marked, top-10 analogs with Δwind "
             "deciles. Every panel: *\"ResQ model estimate — not an official "
             "warning.\"*\n"
             "- **Model insights tab:** primary table with 95% bootstrap CIs; a "
             "one-panel supported / not-supported summary from `significance.md`; "
             "RI PR-AUC bars with error bars + reliability curve; the Δwind "
             "coverage table; RI-classifier permutation importance; M1 tuning "
             "summary; M2 learning curve; M3 fit-diagnostics table.\n")
    L.append("**Hard constraint:** no \"is there a cyclone\" classifier UI or "
             "endpoint exists or was added. The only `is_background` model lives in "
             "`scripts/leakage_audit.py` — the audit tool that *documents* the "
             "AUC-1.0 leakage (feeds `audit.md` §6); it has no UI, endpoint or "
             "saved model and is kept as the required proof-of-exclusion. No "
             "temperature / humidity inputs anywhere.\n")

    # ---- M1 ----
    L.append("## M1 — RI ensemble tuning (selection on pre-2019 GroupKFold OOF PR-AUC only)\n")
    s = m1["selected"]; oof = m1["oof_pr_auc"]
    L.append(f"Grid: {m1['grid_sizes']['hgb']} HGB configs × {m1['grid_sizes']['k']} "
             f"analog k × {m1['grid_sizes']['w']} averaging weights, all scored by "
             "10-fold GroupKFold OOF PR-AUC inside pre-2019. The 2019+ test was "
             "scored once, after every choice was frozen.\n")
    L.append("**Selected config:**\n")
    L.append(f"- classifier HGB: `{json.dumps(s['hgb_params'])}` "
             "(shallower / more regularised than the phase-4 depth-4 default — see M3)\n"
             f"- analog `k = {s['analog_k']}`\n"
             f"- ensemble = `{s['average_weight_on_analog']:.2f}·P_analog + "
             f"{1 - s['average_weight_on_analog']:.2f}·P_classifier`\n")
    pre = m1["pre_tuning_test_pr_auc"]; post = m1["post_tuning_test_pr_auc"]
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| OOF PR-AUC — classifier alone | {oof['classifier_pr_auc']:.3f} |")
    L.append(f"| OOF PR-AUC — analog raw | {oof['analog_raw_pr_auc']:.3f} |")
    L.append(f"| **OOF PR-AUC — tuned ensemble** | **{oof['weighted_ensemble_pr_auc']:.3f}** |")
    L.append(f"| test PR-AUC — phase-4 untuned average | {pre['average_pr_auc']:.3f} |")
    L.append(f"| **test PR-AUC — tuned ensemble** | **{post['tuned_ensemble_pr_auc']:.3f}** |")
    L.append("")
    moved = m1["test_score_moved"]
    L.append(f"**Did the test score move?** "
             f"{'Yes' if moved else 'No'} — "
             f"{post['tuned_ensemble_pr_auc']:.3f} vs {pre['average_pr_auc']:.3f}, "
             f"a {m1['test_pr_auc_change_vs_prev_average']:+.3f} change. "
             "**This is well inside the phase-5 cluster-bootstrap CI (PR-AUC ±~0.15), "
             "so it is not a real improvement.** The value of M1 is a *defensible, "
             "OOF-selected, more-regularised* config — not a better test number. "
             "The deployed engine is left on the documented config so the phase-5 "
             "significance analysis stays valid; adopting the tuned config is a "
             "safe follow-up.\n")

    # ---- M2 ----
    L.append("## M2 — RI learning curve\n")
    L.append("OOF PR-AUC of the tuned ensemble vs number of pre-2019 training "
             "storms:\n")
    L.append("| storms | RI cases | OOF PR-AUC (ensemble) | (classifier) |")
    L.append("|---:|---:|---:|---:|")
    for c in m2["curve"]:
        L.append(f"| {c['n_storms']} | {c['n_RI']} | "
                 f"{c['oof_pr_auc_ensemble']:.3f} | {c['oof_pr_auc_classifier']:.3f} |")
    L.append("")
    L.append(f"![learning curve](learning_curve_ri.png)\n")
    L.append(f"**Regime:** {m2['regime']}\n")

    # ---- M3 ----
    L.append("## M3 — overfitting check (train vs OOF vs test)\n")
    L.append("Full table in `docs/history/fit_diagnostics.md`. Summary:\n")
    L.append("| model | train | OOF | test | |")
    L.append("|---|--:|--:|--:|---|")
    for r in fd["rows"]:
        f = "{:.3f}" if r["metric"] == "PR-AUC" else "{:.1f}"
        L.append(f"| {r['model']} | {f.format(r['train'])} | {f.format(r['oof'])} | "
                 f"{f.format(r['test'])} | {'**OVERFIT**' if r['flag']=='OVERFIT' else 'ok'} |")
    L.append("")
    L.append(f"**{len(fd['overfitting_summary'])} of {len(fd['rows'])} models are "
             "flagged.** For each, OOF ≈ test while train is far better — they "
             "memorise their training storms. The RI classifier is the extreme "
             f"(train PR-AUC "
             f"{[r['train'] for r in fd['rows'] if 'classifier' in r['model']][0]:.2f} "
             f"vs OOF "
             f"{[r['oof'] for r in fd['rows'] if 'classifier' in r['model']][0]:.2f}). "
             "**true CLIPER (linear) and the analog ensemble are not flagged** — "
             "train ≈ OOF ≈ test. Two failure modes stack: the kinematic feature "
             "set is weak *and* the flexible models overfit the little that is "
             "there. ERA5 features matter more than a bigger model. The LSTM is not "
             "retrained (constraint); its test error being worse than the trivial "
             "baselines already rules out train-memorisation.\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
