"""Assemble reports/audit.md from the fragments produced by the other scripts.
Pure formatting - no modelling here."""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAG = ROOT / "reports" / "_fragments"
OUT = ROOT / "reports" / "audit.md"


def j(name):
    return json.loads((FRAG / name).read_text(encoding="utf-8"))


def f(name):
    return (FRAG / name).read_text(encoding="utf-8")


def km(x):
    return "—" if x is None else f"{x:.1f}"


def kt(x):
    return "—" if x is None else f"{x:.2f}"


def main() -> None:
    m = j("models.json")
    lk = j("leakage.json")
    rs = j("resample.json")
    wd = m["withdrawn_nowcast_intensity"]
    sp = m["split"]

    out = []
    out.append("# ResQ data audit — SIH 2026 PS26070 (cyclone ID / classification / prediction, IMD)\n")
    out.append("_Every number is reproducible with the scripts listed at the end. "
               "No rows were fabricated; blanks were left blank. Resampled-grid "
               "wind targets are linear interpolations between real IMD fixes and "
               "are flagged as such. This file covers data provenance; model "
               "results are in `reports/results.md`, the corrections history in "
               "`docs/history/`, and the demo in `app/`._\n")
    out.append("## Contents\n"
               "0. Corrections — numbers withdrawn from the first audit "
               "(phase-3 corrections: `docs/history/corrections.md`)\n"
               "1. Null table (before/after cleaning)\n"
               "2. Lag / tendency recomputation & mismatch rate\n"
               "3. Segmentation & 6 h resampling\n"
               "4. Splits\n"
               "5. Baselines vs models (results: `reports/results.md`)\n"
               "6. Why `normal_weather_background.csv` was excluded\n"
               "7. How to reproduce\n")

    # ---- 0. corrections --------------------------------------------------
    out.append("---\n")
    out.append("## 0. Corrections — what was withdrawn from the first audit\n")
    out.append("Three problems in the first pass were fixed before any further "
               "modelling; the affected numbers are withdrawn.\n")
    out.append("**0.1 The intensity task was misframed as a nowcast.** The original "
               "\"intensity\" number predicted `wind(t)` from the previous row's "
               f"wind. Its persistence MAE of **{wd['MAE_kt']:.2f} kt** "
               f"(n={wd['n']}, 2019+ holdout) measures how smooth the label is "
               "over one step, not forecast skill — it is withdrawn.\n"
               "The intensity task is now **Δwind at +6 / +12 / +24 h** "
               "(`wind(t+h) − wind(t)`, by real elapsed time from `dt_snap_unix`), "
               "with the baseline **Δ = 0**, reported at the same leads as the track "
               "metric. `wind_tendency_3h` / `pressure_tendency_3h` are **not** used "
               "as predictors for any intensity target (they reconstruct a "
               "one-step target — see §2).\n")
    out.append("**0.2 The time axis was treated as uniform 3 h — it is not.** Only "
               "60.8% of steps are 3 h. The old LSTM stacked 8 consecutive rows "
               "regardless of the real gaps between them, so a \"sequence\" could "
               "span anywhere from 21 h to several months. All sequence-model "
               "numbers from that run are withdrawn and replaced with the "
               "segmented / resampled version in §3 and §5.\n")
    out.append("**0.3 Three image-structure columns (>99% null) were kept in the "
               "first pass.** `eye_confidence_0to2`, `storm_touches_frame_edge`, "
               "`pct_storm_boundary_at_frame_edge_or_nodata` are now dropped "
               "entirely — a missingness indicator on a column that is 99.8% empty "
               "mostly encodes *which era / batch* a row came from, which is "
               "leakage, not signal.\n")

    out.append("\n---\n")
    out.append(f("null_table.md"))
    out.append("\n---\n")
    out.append(f("lag_mismatch.md"))

    # ---- 3. segmentation ----------------------------------------------
    out.append("\n---\n")
    out.append("### 3. Segmentation & 6 h resampling (`src/segment_resample.py`)\n")
    out.append(f"Each storm is split into contiguous **segments** wherever the "
               f"observation gap exceeds 12 h. Segments are then resampled onto a "
               f"uniform **6 h** grid: `lat`, `lon`, `pressure`, `pressure_drop` by "
               f"linear interpolation in unix time; `wind_kt` by linear "
               f"interpolation carrying an `interpolated` flag "
               f"(True when the nearest real wind fix is > 1 h away). Segments with "
               f"fewer than 5 grid points are dropped.\n")
    out.append("| quantity | value |")
    out.append("|---|---:|")
    out.append(f"| raw rows in (valid storm id + lat/lon) | {rs['raw_rows_in']} |")
    out.append(f"| segments total | {rs['segments_total']} |")
    out.append(f"| segments kept (≥ 5 grid points) | {rs['segments_kept']} |")
    out.append(f"| segments dropped (< 5 grid points) | {rs['segments_dropped_lt_5_grid_pts']} |")
    out.append(f"| resampled rows out | {rs['resampled_rows_out']} |")
    out.append(f"| wind interpolated fraction (all / holdout) | "
               f"{rs['wind_interpolated_fraction_all']} / "
               f"{rs['wind_interpolated_fraction_holdout']} |")
    gl = rs["grid_segment_len_points"]
    out.append(f"| kept-segment length, grid points (min/median/max) | "
               f"{gl['min']} / {gl['median']:.0f} / {gl['max']} |")
    out.append("")
    out.append(f"Kept-segment length histogram (grid points): "
               f"`{json.dumps(rs['grid_len_histogram'])}`.\n")
    out.append(f"Raw-segment length before resampling (points): median "
               f"{rs['raw_segment_len_points']['median']:.0f}, mean "
               f"{rs['raw_segment_len_points']['mean']}, max "
               f"{rs['raw_segment_len_points']['max']} — the {rs['segments_dropped_lt_5_grid_pts']} "
               "dropped segments are overwhelmingly 1–3-point track fragments.\n")
    out.append("Per-hour rates (`wind_rate_per_h`, `pressure_rate_per_h`), 6 h "
               "tendencies and the 6 h motion vector (`dlat_6h`, `dlon_6h`) are "
               "recomputed on this uniform grid. Output: "
               "`data/cyclone_resampled_6h.parquet`.\n")

    out.append("\n---\n")
    out.append("### 4. Splits (`scripts/splits.py`)\n")
    out.append("No random row-level split anywhere.\n")
    out.append("- **Two-way (point forecast models):** storms with first observation "
               "in **year ≥ 2019** are the test set; earlier storms train. Segments "
               "inherit their storm's side.\n"
               "- **Three-way (conformal + RI, phase 3):** storm-disjoint — "
               f"train ≤ 2016 ({sp['train_le_2016']['storms']} storms / "
               f"{sp['train_le_2016']['grid_rows']} grid rows), calibration "
               f"2017–2018 ({sp['calib_2017_2018']['storms']} / "
               f"{sp['calib_2017_2018']['grid_rows']}), test ≥ 2019 "
               f"({sp['test_ge_2019']['storms']} / {sp['test_ge_2019']['grid_rows']}). "
               "Full storm-id lists per bucket are in "
               "`reports/_fragments/models.json → split`.\n"
               "- `GroupShuffleSplit` on `storm_id_code` is still available in "
               "`scripts/splits.py` for CV but is not used in any headline table.\n")

    out.append("\n---\n")
    out.append("### 5. Baselines vs models — see `reports/results.md`, `docs/history/phase4.md`\n")
    out.append("The headline numbers live in **`reports/results.md`** (primary "
               "intersection, secondary full-n, interpolation sensitivity, "
               "marginal/split-CQR/cross-conformal coverage, rapid intensification). "
               "**`docs/history/corrections.md`** covers the four phase-3 corrections and "
               "**`docs/history/phase4.md`** the phase-4 fixes (true CLIPER, "
               "cross-conformal, combined P(RI), the ERA5 layer, the dashboard).\n")
    out.append("Where things stand on the 2019+ test storms:\n"
               "- **Track:** the motion baselines (persistence-motion; true "
               "CLIPER, an OLS fit) are unbeaten at 6/12/24 h. Every learned track "
               "model — direct or residual, GBM or LSTM — loses to the better "
               "baseline. No exploitable structure in the residual given kinematics.\n"
               "- **Intensity (Δwind):** the analog ensemble and a direct GBM beat "
               "the Δ = 0 baseline at 12/24 h; the LSTM beats nothing.\n"
               "- **Uncertainty:** split-CQR and cross-conformal both lift the 24 h "
               "Δwind 80% interval from 60% to ~76% coverage; neither reaches "
               "nominal (conditional-coverage limit, not calibration data).\n"
               "- **Rapid intensification:** the **average** of the calibrated "
               "analog and classifier P(RI) — ROC-AUC ≈ 0.84, PR-AUC ≈ 0.25 on a "
               "6% base rate — beats either alone and is the one learned model that "
               "clearly beats its trivial baseline. Still needs ERA5 to be "
               "operational.\n"
               "- **Demo:** `python app/run.py` serves a Leaflet dashboard that "
               "replays each 2019+ storm 6 h at a time with the forecast that would "
               "have been issued, the conformal cone, analogs and the P(RI) gauge.\n")

    # ---- 6. background (unchanged content) --------------------------------
    out.append("\n---\n")
    out.append("### 6. Why `normal_weather_background.csv` was excluded\n")
    out.append(f"The file has **{lk['n_background_rows']} rows**. It is **not used "
               "anywhere in training, validation, or the holdout** — only here. A "
               "prior audit flagged it as synthetic; the checks below reproduce that "
               "from the file itself.\n")
    out.append("**6.1 A classifier separates background from cyclone perfectly, even "
               "with all provenance columns removed.**\n")
    out.append(f"- `HistGradientBoostingClassifier`, 5-fold stratified CV, target = "
               f"`is_background`. Provenance dropped (`storm_id_code`, "
               f"`dt_snap_unix`, `wind_kt_canonical`, `imd_grade_ordinal`, "
               f"`wind_label_source_*`, `has_image`, `pipeline_*`, image structure "
               f"columns, …).\n"
               f"- Features used: {', '.join('`'+c+'`' for c in lk['features_used'])}.\n"
               f"- **ROC-AUC = {lk['auc_all_features_no_provenance']:.4f}**.\n"
               f"- Using only `lat`, `lon`, `month_sin`, `month_cos`: "
               f"**ROC-AUC = {lk['auc_lat_lon_month_only']:.4f}**.\n")
    out.append("**6.2 The synthetic rows are numerically distinguishable by construction.**\n")
    ivf = lk["integer_value_fraction"]
    out.append("| column | integer-value fraction — cyclone | — background |")
    out.append("|---|---:|---:|")
    for c, v in ivf.items():
        out.append(f"| `{c}` | {v['cyclone']:.3f} | {v['background']:.3f} |")
    wr = lk["wind_kt_canonical_range"]
    out.append("")
    out.append(f"- Real IMD data is quantised (5-kt winds, whole-hPa pressure): "
               f"**100%** integer; background rows are continuous floats (~11%).\n"
               f"- `wind_kt_canonical` range: cyclone **{wr['cyclone'][0]:.0f}–"
               f"{wr['cyclone'][1]:.0f} kt**, background **{wr['background'][0]:.1f}–"
               f"{wr['background'][1]:.1f} kt** — every background row is below "
               "tropical-storm force.\n"
               f"- `corr(wind_kt_canonical, wind_kt_lag1)`: cyclone "
               f"**{lk['lag_autocorr_wind_vs_lag1']['cyclone']:.3f}**, background "
               f"**{lk['lag_autocorr_wind_vs_lag1']['background']:.3f}** — background "
               "lag columns carry no real dynamics.\n"
               f"- `storm_id_code` = **{lk['storm_id_code_unique_in_background']}** "
               "for every background row; sampled timestamps jump by up to "
               f"{lk['background_timestamp_gap_days']['max']:.0f} days — no real "
               "tracks in the file.\n")
    out.append("**6.3 Consequence.** Any supervised model trained on cyclone + "
               "background learns the synthetic/real boundary (AUC 1.0) instead of "
               "meteorology. Excluded from every split. A real negative class would "
               "have to come from genuine IMD / reanalysis fair-weather observations "
               "at the same quantisation and cadence.\n")

    out.append("\n---\n")
    out.append("### 7. How to reproduce\n")
    out.append("```\n"
               "python scripts/build_dataset.py      # -> data/cyclone_clean.parquet, null + lag fragments\n"
               "python -m src.segment_resample       # -> data/cyclone_resampled_6h.parquet\n"
               "python scripts/leakage_audit.py      # -> reports/_fragments/leakage.json\n"
               "python -m src.era5                   # -> data/cyclone_6h_with_env.parquet (null until CDS access)\n"
               "python -m src.analogs                # -> analogs.json, analogs_ri.json, example\n"
               "python -m src.uncertainty            # marginal/split-CQR/cross-conformal -> uncertainty.json, PNG\n"
               "python -m src.ri                     # combined P(RI) -> reports/_fragments/ri.json\n"
               "python scripts/run_models.py         # true CLIPER + primary/secondary/residual -> models.json\n"
               "python -m src.bootstrap              # cluster bootstrap by storm -> reports/significance.md\n"
               "python -m src.ri_tune                # M1 tuning + M2 learning curve -> ri_tune.json, learning_curve_ri.png\n"
               "python -m src.fit_diagnostics        # M3 train/OOF/test -> docs/history/fit_diagnostics.md\n"
               "python scripts/make_results.py       # -> reports/results.md\n"
               "python scripts/make_corrections.py   # -> docs/history/corrections.md\n"
               "python scripts/make_phase4.py        # -> docs/history/phase4.md\n"
               "python scripts/make_phase6.py        # -> docs/history/phase6.md\n"
               "python scripts/make_limitations.py   # -> reports/limitations.md\n"
               "python scripts/make_audit.py         # -> reports/audit.md (this file)\n"
               "python scripts/build_district_history.py  # -> app/static/data/district_history.json\n"
               "python scripts/build_checklist.py    # -> data/ndma_guidance.json (NDMA Drupal REST harvest)\n"
               "python scripts/build_corpus.py       # -> app/static/data/corpus.json (chatbot retrieval)\n"
               "python scripts/build_storms.py       # -> app/static/data/storms.json (2019+ tracks)\n"
               "python scripts/build_faq.py          # -> app/static/data/faq.json (chatbot)\n"
               "python scripts/build_insights.py     # -> app/static/data/insights.json (metrics + CIs)\n"
               "python app/run.py --rebuild          # builds every artifact above, serves 127.0.0.1:8000\n"
               "                                     #   tabs: Dashboard / Forecast / Preparation Checklist /\n"
               "                                     #         District Cyclone History / Official Warnings / Chatbot\n"
               "```\n")

    OUT.write_text("\n".join(out), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
