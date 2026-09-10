"""
Step 6 support: reproduce the leakage that makes normal_weather_background.csv
unusable for supervised training, so reports/audit.md can document the exclusion
with numbers anyone can regenerate.

This is the ONLY place the background file is read.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRAG = ROOT / "reports" / "_fragments"
FRAG.mkdir(parents=True, exist_ok=True)

CYCLONE = ROOT / "training_ready_dataset(all 3 combined).csv"
BACKGROUND = ROOT / "normal_weather_background.csv"

# everything that could tag which file a row came from
PROVENANCE = [
    "storm_id_code", "storm_name_code", "dt_snap_unix", "image_filename_code",
    "wind_kt_canonical", "has_wind_label", "imd_grade_ordinal",
    "severity_tier_ordinal", "wind_label_source_IMD", "wind_label_source_none",
    "date_suspect", "has_image", "image_to_observation_gap_hours",
    "oci", "oci_diameter", "core_to_shield_area_ratio",
    "dense_core_circularity_0to1", "symmetry_score_0to1",
    "cloud_cover_pct_of_valid_frame", "eye_confidence_0to2",
    "eye_to_core_diameter_ratio", "storm_touches_frame_edge",
    "pct_storm_boundary_at_frame_edge_or_nodata",
    "aux_sst_warm_pct", "aux_sst_cool_pct", "aux_rainfall_heavy_pct",
    "aux_rainfall_moderate_pct", "oci_diameter_missing", "pressure_missing",
    "pipeline_grayscale", "pipeline_no_image", "pipeline_visible",
]

MINIMAL = ["lat", "lon", "month_sin", "month_cos"]


def int_frac(s: pd.Series) -> float:
    s = s.dropna()
    return float(np.mean(np.isclose(s, s.round()))) if len(s) else float("nan")


def auc_cv(X: pd.DataFrame, y: np.ndarray) -> float:
    clf = HistGradientBoostingClassifier(random_state=26070, max_iter=300)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=26070)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def main() -> None:
    cy = pd.read_csv(CYCLONE)
    bg = pd.read_csv(BACKGROUND)
    both = pd.concat([cy.assign(is_background=0), bg.assign(is_background=1)],
                     ignore_index=True)
    y = both["is_background"].to_numpy()

    feats_all = [c for c in both.columns
                 if c not in PROVENANCE + ["is_background"]]
    X_all = both[feats_all].select_dtypes(include="number")
    X_min = both[MINIMAL]

    report = {
        "n_cyclone_rows": int(len(cy)),
        "n_background_rows": int(len(bg)),
        "auc_all_features_no_provenance": auc_cv(X_all, y),
        "features_used": list(X_all.columns),
        "auc_lat_lon_month_only": auc_cv(X_min, y),
        "integer_value_fraction": {
            c: {"cyclone": round(int_frac(cy[c]), 3),
                "background": round(int_frac(bg[c]), 3)}
            for c in ["wind_kt_canonical", "imd_pressure_hpa",
                      "wind_kt_lag1", "pressure_lag1"]
        },
        "wind_kt_canonical_range": {
            "cyclone": [float(cy.wind_kt_canonical.min()),
                        float(cy.wind_kt_canonical.max())],
            "background": [float(bg.wind_kt_canonical.min()),
                          float(bg.wind_kt_canonical.max())],
        },
        "lag_autocorr_wind_vs_lag1": {
            "cyclone": float(cy.dropna(subset=["wind_kt_canonical", "wind_kt_lag1"])
                             [["wind_kt_canonical", "wind_kt_lag1"]].corr().iloc[0, 1]),
            "background": float(bg.dropna(subset=["wind_kt_canonical", "wind_kt_lag1"])
                                [["wind_kt_canonical", "wind_kt_lag1"]].corr().iloc[0, 1]),
        },
        "storm_id_code_unique_in_background": bg.storm_id_code.dropna().unique().tolist(),
        "background_timestamp_gap_days": {
            "median": float(np.median(np.diff(np.sort(bg.dt_snap_unix.unique())) / 86400)),
            "max": float(np.max(np.diff(np.sort(bg.dt_snap_unix.unique())) / 86400)),
        },
    }
    (FRAG / "leakage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
