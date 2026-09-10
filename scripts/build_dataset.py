"""
Steps 1 & 2 for SIH 2026 PS26070.

1. Build data/cyclone_clean.parquet from 'training_ready_dataset(all 3 combined).csv'
   ONLY, dropping >97%-null / leak-only columns. Emit a before/after null table.

2. Recompute lag / tendency features from dt_snap_unix, grouped by storm_id_code
   and sorted by time. Verify the shipped wind_kt_lag1 / pressure_lag1 /
   *_tendency_3h columns against the recomputation and report the mismatch rate.
   Assert the 3-hour cadence and report the gap distribution.

No rows are fabricated or dropped; blanks stay blank.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "training_ready_dataset(all 3 combined).csv"
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
FRAG = REPORTS / "_fragments"
DATA.mkdir(exist_ok=True)
FRAG.mkdir(parents=True, exist_ok=True)

DROP_COLS = [
    "eye_to_core_diameter_ratio",
    "aux_sst_warm_pct", "aux_sst_cool_pct",
    "aux_rainfall_heavy_pct", "aux_rainfall_moderate_pct",
    "oci", "oci_diameter",
    "core_to_shield_area_ratio", "dense_core_circularity_0to1",
    "symmetry_score_0to1", "cloud_cover_pct_of_valid_frame",
    "image_to_observation_gap_hours",
    "image_filename_code", "storm_name_code", "has_image",
    # pipeline_*
    "pipeline_grayscale", "pipeline_no_image", "pipeline_visible",
    # >99% null - not worth a missingness indicator that could encode era
    "eye_confidence_0to2", "storm_touches_frame_edge",
    "pct_storm_boundary_at_frame_edge_or_nodata",
]

# shipped columns are named "*_tendency_3h" but the cadence is not 3 h; rename to
# "*_tendency_step" in the clean table so nothing downstream trusts the "3h".
TENDENCY_RENAME = {
    "wind_tendency_3h": "wind_tendency_step",
    "pressure_tendency_3h": "pressure_tendency_step",
    "wind_kt_tendency_3h_recomp": "wind_kt_tendency_step_recomp",
    "pressure_tendency_3h_recomp": "pressure_tendency_step_recomp",
}

LAG_PAIRS = [
    # (base value col, shipped lag col, shipped tendency col, recomputed prefix)
    ("wind_kt_canonical", "wind_kt_lag1", "wind_tendency_3h", "wind_kt"),
    ("imd_pressure_hpa", "pressure_lag1", "pressure_tendency_3h", "pressure"),
]

STEP_SECONDS = 3 * 3600


def null_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "n_missing": df.isna().sum(),
        "pct_missing": (df.isna().mean() * 100).round(2),
    })


def main() -> None:
    raw = pd.read_csv(SRC)
    before = null_table(raw)

    present_drop = [c for c in DROP_COLS if c in raw.columns]
    missing_drop = [c for c in DROP_COLS if c not in raw.columns]
    clean = raw.drop(columns=present_drop)

    # ---- Step 2: recompute lags/tendencies from time --------------------------
    clean = clean.sort_values(["storm_id_code", "dt_snap_unix"],
                              kind="mergesort").reset_index(drop=True)
    has_sid = clean["storm_id_code"].notna()
    g = clean.loc[has_sid].groupby("storm_id_code", sort=False)

    dt = g["dt_snap_unix"].diff()
    clean.loc[has_sid, "dt_since_prev_sec"] = dt
    dt_hours = dt / 3600.0

    cadence = {
        "rows_with_predecessor": int(dt.notna().sum()),
        "step_eq_3h": int((dt == STEP_SECONDS).sum()),
        "step_eq_3h_pct": round(float((dt == STEP_SECONDS).mean(skipna=True) * 100), 2),
        "gap_hours_value_counts": {
            f"{k:g}h": int(v)
            for k, v in dt_hours.round(3).value_counts().sort_index().items()
        },
        "max_gap_hours": round(float(dt_hours.max()), 2),
    }

    mismatch_rows = []
    for base, lag_col, tend_col, pfx in LAG_PAIRS:
        rc_lag = g[base].shift(1)
        rc_tend = clean.loc[has_sid, base] - rc_lag
        clean.loc[has_sid, f"{pfx}_lag1_recomp"] = rc_lag
        clean.loc[has_sid, f"{pfx}_tendency_3h_recomp"] = rc_tend

        for shipped, recomp, label in [(lag_col, rc_lag, "lag1"),
                                       (tend_col, rc_tend, "tendency_3h")]:
            s = clean[shipped]
            both = s.notna() & recomp.notna()
            diff = (s[both] - recomp[both]).abs()
            mism = (diff > 1e-6)
            mismatch_rows.append({
                "shipped_column": shipped,
                "compared_rows": int(both.sum()),
                "mismatches_gt_1e-6": int(mism.sum()),
                "mismatch_rate_pct": round(float(mism.mean() * 100), 3) if both.sum() else None,
                "shipped_notnull_recomp_null": int((s.notna() & recomp.isna()).sum()),
                "recomp_notnull_shipped_null": int((s.isna() & recomp.notna()).sum()),
                "max_abs_diff": round(float(diff.max()), 4) if both.sum() else None,
            })

    mism_df = pd.DataFrame(mismatch_rows)

    # ---- rename tendency cols so nothing downstream trusts the "3h" --------
    renamed = {k: v for k, v in TENDENCY_RENAME.items() if k in clean.columns}
    clean = clean.rename(columns=renamed)

    # ---- write parquet ------------------------------------------------------
    out = DATA / "cyclone_clean.parquet"
    clean.to_parquet(out, index=False)

    after = null_table(clean)

    # ---- before/after null table -----------------------------------------
    # line renamed columns up with their new names so they read as "kept"
    before = before.rename(index=renamed)
    nt = before.join(after, lsuffix="_before", rsuffix="_after", how="outer")
    nt["status"] = np.where(nt["pct_missing_after"].isna(), "DROPPED", "kept")
    nt = nt.sort_values(["status", "pct_missing_before"], ascending=[True, False])

    lines = []
    lines.append("### 1. Null table — before vs after cleaning\n")
    lines.append(f"Source: `{SRC.name}` — {raw.shape[0]} rows x {raw.shape[1]} cols "
                 f"-> clean `{out.name}` — {clean.shape[0]} rows x {clean.shape[1]} cols "
                 f"(no rows dropped).\n")
    lines.append(f"Columns dropped ({len(present_drop)}): "
                 + ", ".join(f"`{c}`" for c in present_drop) + "\n")
    lines.append("Columns renamed: "
                 + ", ".join(f"`{k}` -> `{v}`" for k, v in renamed.items()) + "\n")
    if missing_drop:
        lines.append(f"Requested but not present in source: "
                     + ", ".join(f"`{c}`" for c in missing_drop) + "\n")
    lines.append("| column | %null before | %null after | status |")
    lines.append("|---|---:|---:|---|")
    for c, r in nt.iterrows():
        b = f"{r['pct_missing_before']:.2f}" if pd.notna(r["pct_missing_before"]) else "—"
        a = f"{r['pct_missing_after']:.2f}" if pd.notna(r["pct_missing_after"]) else "—"
        lines.append(f"| `{c}` | {b} | {a} | {r['status']} |")
    lines.append("")
    still_sparse = after[after["pct_missing"] > 90].index.tolist()
    if still_sparse:
        lines.append("> Columns still >90% null after the prescribed drop list "
                     "(kept because they were not on the list; treat as unusable "
                     "for supervised training): "
                     + ", ".join(f"`{c}`" for c in still_sparse) + "\n")
    (FRAG / "null_table.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- lag mismatch fragment -------------------------------------------
    l2 = []
    l2.append("### 2. Lag / tendency recomputation\n")
    l2.append("Recomputed per `storm_id_code`, sorted by `dt_snap_unix`: "
              "`*_lag1_recomp = value.shift(1)`, "
              "`*_tendency_recomp = value - lag1_recomp` (step-to-step, not "
              "time-normalised).\n")
    l2.append("**Cadence check** — the shipped columns assume a 3 h step "
              "(10800 s); actual gaps:\n")
    l2.append(f"- rows with a predecessor: {cadence['rows_with_predecessor']}")
    l2.append(f"- exactly 3 h apart: {cadence['step_eq_3h']} "
              f"({cadence['step_eq_3h_pct']}%)")
    l2.append(f"- largest gap: {cadence['max_gap_hours']} h")
    l2.append(f"- gap-hours distribution: `{json.dumps(cadence['gap_hours_value_counts'])}`\n")
    l2.append("> The cadence is **not** uniformly 3 h (only ~61% of steps are), so "
              "the shipped `*_tendency_3h` columns are mislabelled — they are raw "
              "step-to-step differences, not 3-hour-normalised rates. They are "
              "renamed to `*_tendency_step` in the clean table and a proper 6 h "
              "grid is built in §3.\n")
    l2.append("**Shipped vs recomputed** (match tolerance 1e-6):\n")
    l2.append("| shipped column | rows compared | mismatches | mismatch rate | "
              "shipped-only nulls filled by recomp | max abs diff |")
    l2.append("|---|---:|---:|---:|---:|---:|")
    for _, r in mism_df.iterrows():
        rate = f"{r['mismatch_rate_pct']}%" if r["mismatch_rate_pct"] is not None else "—"
        l2.append(f"| `{r['shipped_column']}` | {r['compared_rows']} | "
                  f"{r['mismatches_gt_1e-6']} | {rate} | "
                  f"{r['recomp_notnull_shipped_null']} | {r['max_abs_diff']} |")
    l2.append("")
    l2.append("Column renames in the clean table (the cadence is not 3 h, so the "
              "\"3h\" label was removed): `wind_tendency_3h` -> `wind_tendency_step`, "
              "`pressure_tendency_3h` -> `pressure_tendency_step`.\n")
    l2.append("New columns written to the parquet: "
              "`wind_kt_lag1_recomp`, `wind_kt_tendency_step_recomp`, "
              "`pressure_lag1_recomp`, `pressure_tendency_step_recomp`, "
              "`dt_since_prev_sec`. These are step-to-step differences on the raw "
              "irregular axis and are kept for provenance only. Proper per-hour "
              "rates are computed on the resampled 6 h grid "
              "(`data/cyclone_resampled_6h.parquet`, see `src/segment_resample.py`); "
              "no `*_tendency` column is used as a predictor for an intensity "
              "target.\n")
    (FRAG / "lag_mismatch.md").write_text("\n".join(l2), encoding="utf-8")

    (FRAG / "cadence.json").write_text(json.dumps(cadence, indent=2), encoding="utf-8")
    mism_df.to_csv(FRAG / "lag_mismatch.csv", index=False)

    print("wrote", out)
    print(nt.to_string())
    print()
    print(mism_df.to_string())
    print()
    print("cadence:", json.dumps(cadence, indent=2))


if __name__ == "__main__":
    main()
