"""
Split module for the ResQ cyclone dataset (SIH 2026 PS26070).

Two split strategies, both grouped by storm so no storm's timesteps ever
straddle a train/eval boundary:

  * cv_splits()        -> GroupShuffleSplit on storm_id_code (for cross-validation)
  * temporal_holdout() -> fixed holdout = every storm whose FIRST observation
                          falls in year >= 2019

There are deliberately NO random row-level splits in this module. Callers must
pass a frame that still carries `storm_id_code` and `dt_snap_unix`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

HOLDOUT_START_YEAR = 2019
RANDOM_STATE = 26070


def storm_start_year(df: pd.DataFrame) -> pd.Series:
    """Year of each storm's earliest observation, broadcast back to every row."""
    yr = pd.to_datetime(df["dt_snap_unix"], unit="s").dt.year
    return df.assign(_yr=yr).groupby("storm_id_code")["_yr"].transform("min")


def modeling_frame(df: pd.DataFrame, target: str = "wind_kt_canonical") -> pd.DataFrame:
    """Rows usable for supervised modelling: real storm id + non-null target."""
    m = df["storm_id_code"].notna() & df[target].notna()
    return df.loc[m].copy()


def temporal_holdout(df: pd.DataFrame):
    """Return (train_idx, holdout_idx) as positional numpy arrays.

    holdout = storms that start in HOLDOUT_START_YEAR or later.
    """
    start_yr = storm_start_year(df).to_numpy()
    pos = np.arange(len(df))
    is_holdout = start_yr >= HOLDOUT_START_YEAR
    return pos[~is_holdout], pos[is_holdout]


def temporal_holdout_resampled(df: pd.DataFrame):
    """(train_idx, holdout_idx) for the resampled 6 h table, which already carries
    a per-row `storm_year` (year of the storm's first raw observation)."""
    pos = np.arange(len(df))
    is_holdout = df["storm_year"].to_numpy() >= HOLDOUT_START_YEAR
    return pos[~is_holdout], pos[is_holdout]


TRAIN_MAX_YEAR = 2016      # three-way split: train <= 2016
CALIB_YEARS = (2017, 2018)  #                  calibration 2017-2018
# test = HOLDOUT_START_YEAR (2019) and later


def three_way_split_resampled(df: pd.DataFrame):
    """(train_idx, calib_idx, test_idx) positional arrays for the resampled grid.

    Storm-disjoint by construction (buckets are defined on `storm_year`, which is
    constant within a storm). Used for conformal calibration so the 2019+ test
    storms are never touched by fitting or calibration.
    """
    y = df["storm_year"].to_numpy()
    pos = np.arange(len(df))
    train = pos[y <= TRAIN_MAX_YEAR]
    calib = pos[(y >= CALIB_YEARS[0]) & (y <= CALIB_YEARS[1])]
    test = pos[y >= HOLDOUT_START_YEAR]
    return train, calib, test


def three_way_report(df: pd.DataFrame) -> dict:
    y = df["storm_year"]
    out = {}
    for name, mask in [("train_le_2016", y <= TRAIN_MAX_YEAR),
                       ("calib_2017_2018", (y >= CALIB_YEARS[0]) & (y <= CALIB_YEARS[1])),
                       ("test_ge_2019", y >= HOLDOUT_START_YEAR)]:
        sub = df[mask]
        out[name] = {
            "storms": int(sub["storm_id_code"].nunique()),
            "segments": int(sub["segment_id"].nunique()),
            "grid_rows": int(len(sub)),
            "storm_ids": sorted(sub["imd_storm_id"].dropna().unique().tolist()),
        }
    return out


def cv_splits(df: pd.DataFrame, n_splits: int = 5, test_size: float = 0.2):
    """Yield (train_idx, test_idx) positional arrays, grouped on storm_id_code.

    Intended to be run on the *training* portion of the temporal split so the
    post-2019 holdout is never touched during model selection.
    """
    groups = df["storm_id_code"].to_numpy()
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=test_size,
                            random_state=RANDOM_STATE)
    x_dummy = np.zeros(len(df))
    for tr, te in gss.split(x_dummy, groups=groups):
        yield tr, te


if __name__ == "__main__":
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "data" / "cyclone_clean.parquet"
    d = pd.read_parquet(p)
    d = modeling_frame(d)
    tr, ho = temporal_holdout(d)
    print(f"modeling rows: {len(d)}")
    print(f"temporal train: {len(tr)} rows / {d.iloc[tr].storm_id_code.nunique()} storms")
    print(f"temporal holdout (>= {HOLDOUT_START_YEAR}): {len(ho)} rows / "
          f"{d.iloc[ho].storm_id_code.nunique()} storms")
    for i, (a, b) in enumerate(cv_splits(d.iloc[tr])):
        ga = set(d.iloc[tr].iloc[a].storm_id_code)
        gb = set(d.iloc[tr].iloc[b].storm_id_code)
        assert not (ga & gb), "storm leaked across CV fold"
        print(f"  fold {i}: train {len(a)} / test {len(b)}  (disjoint storms ok)")
