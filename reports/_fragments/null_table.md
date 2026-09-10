### 1. Null table — before vs after cleaning

Source: `training_ready_dataset(all 3 combined).csv` — 7656 rows x 49 cols -> clean `cyclone_clean.parquet` — 7656 rows x 33 cols (no rows dropped).

Columns dropped (21): `eye_to_core_diameter_ratio`, `aux_sst_warm_pct`, `aux_sst_cool_pct`, `aux_rainfall_heavy_pct`, `aux_rainfall_moderate_pct`, `oci`, `oci_diameter`, `core_to_shield_area_ratio`, `dense_core_circularity_0to1`, `symmetry_score_0to1`, `cloud_cover_pct_of_valid_frame`, `image_to_observation_gap_hours`, `image_filename_code`, `storm_name_code`, `has_image`, `pipeline_grayscale`, `pipeline_no_image`, `pipeline_visible`, `eye_confidence_0to2`, `storm_touches_frame_edge`, `pct_storm_boundary_at_frame_edge_or_nodata`

Columns renamed: `wind_tendency_3h` -> `wind_tendency_step`, `pressure_tendency_3h` -> `pressure_tendency_step`, `wind_kt_tendency_3h_recomp` -> `wind_kt_tendency_step_recomp`, `pressure_tendency_3h_recomp` -> `pressure_tendency_step_recomp`

| column | %null before | %null after | status |
|---|---:|---:|---|
| `eye_to_core_diameter_ratio` | 100.00 | — | DROPPED |
| `aux_rainfall_heavy_pct` | 99.99 | — | DROPPED |
| `aux_rainfall_moderate_pct` | 99.99 | — | DROPPED |
| `aux_sst_cool_pct` | 99.97 | — | DROPPED |
| `aux_sst_warm_pct` | 99.97 | — | DROPPED |
| `cloud_cover_pct_of_valid_frame` | 99.78 | — | DROPPED |
| `core_to_shield_area_ratio` | 99.76 | — | DROPPED |
| `dense_core_circularity_0to1` | 99.76 | — | DROPPED |
| `eye_confidence_0to2` | 99.76 | — | DROPPED |
| `image_filename_code` | 99.76 | — | DROPPED |
| `image_to_observation_gap_hours` | 99.76 | — | DROPPED |
| `pct_storm_boundary_at_frame_edge_or_nodata` | 99.76 | — | DROPPED |
| `storm_touches_frame_edge` | 99.76 | — | DROPPED |
| `symmetry_score_0to1` | 99.76 | — | DROPPED |
| `oci` | 93.48 | — | DROPPED |
| `oci_diameter` | 93.48 | — | DROPPED |
| `storm_name_code` | 58.16 | — | DROPPED |
| `has_image` | 0.00 | — | DROPPED |
| `pipeline_grayscale` | 0.00 | — | DROPPED |
| `pipeline_no_image` | 0.00 | — | DROPPED |
| `pipeline_visible` | 0.00 | — | DROPPED |
| `pressure_tendency_step` | 18.09 | 18.09 | kept |
| `pressure_lag1` | 17.52 | 17.52 | kept |
| `wind_tendency_step` | 11.96 | 11.96 | kept |
| `wind_kt_lag1` | 11.46 | 11.46 | kept |
| `imd_pressure_hpa` | 11.32 | 11.32 | kept |
| `imd_pressure_drop` | 7.88 | 7.88 | kept |
| `storm_age_hours` | 4.45 | 4.45 | kept |
| `wind_kt_canonical` | 2.74 | 2.74 | kept |
| `imd_grade_ordinal` | 1.87 | 1.87 | kept |
| `severity_tier_ordinal` | 1.87 | 1.87 | kept |
| `date_suspect` | 1.45 | 1.45 | kept |
| `storm_id_code` | 1.45 | 1.45 | kept |
| `lon` | 0.17 | 0.17 | kept |
| `lat` | 0.01 | 0.01 | kept |
| `basin_ARB` | 0.00 | 0.00 | kept |
| `basin_BOB` | 0.00 | 0.00 | kept |
| `basin_LAND` | 0.00 | 0.00 | kept |
| `doy_cos` | 0.00 | 0.00 | kept |
| `doy_sin` | 0.00 | 0.00 | kept |
| `dt_snap_hour` | 0.00 | 0.00 | kept |
| `dt_snap_unix` | 0.00 | 0.00 | kept |
| `has_wind_label` | 0.00 | 0.00 | kept |
| `month_cos` | 0.00 | 0.00 | kept |
| `month_sin` | 0.00 | 0.00 | kept |
| `oci_diameter_missing` | 0.00 | 0.00 | kept |
| `pressure_missing` | 0.00 | 0.00 | kept |
| `wind_label_source_IMD` | 0.00 | 0.00 | kept |
| `wind_label_source_none` | 0.00 | 0.00 | kept |
| `dt_since_prev_sec` | — | 7.00 | kept |
| `pressure_lag1_recomp` | — | 15.31 | kept |
| `pressure_tendency_step_recomp` | — | 15.88 | kept |
| `wind_kt_lag1_recomp` | — | 9.22 | kept |
| `wind_kt_tendency_step_recomp` | — | 9.73 | kept |
