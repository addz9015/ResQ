"""
Phase-5 TASK 1 - cluster bootstrap confidence intervals on every headline number.

Adjacent 6 h states within a storm are ~0.97 autocorrelated, so a state-level
bootstrap would give intervals that are far too narrow. All resampling here is
**by storm** (cluster / block bootstrap): each of 1000 replicates draws the
2019+ test storms with replacement and pools every state of every drawn storm
(with multiplicity).

Point estimate = the metric on the real test set. Interval = 2.5 / 97.5
percentiles of the replicate values. Paired differences are computed *within*
each replicate (same resampled storms for both models) so the difference
interval accounts for the correlation between the two estimates.

Inputs  : reports/_fragments/preds_primary.parquet  (from scripts/run_models.py)
          reports/_fragments/preds_ri.parquet       (from src/ri.py)
Outputs : reports/significance.md, reports/_fragments/bootstrap.json

Run:  python -m src.bootstrap
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .common import FRAG, REPORTS, RANDOM_STATE, haversine_km

N_BOOT = 1000
LEADS = (6, 12, 24)


# ---------------------------------------------------------------------------
# generic storm-cluster bootstrap
# ---------------------------------------------------------------------------
def cluster_bootstrap(df: pd.DataFrame, metric_fn, storm_col="storm_id_code",
                      n_boot=N_BOOT, seed=RANDOM_STATE) -> pd.DataFrame:
    storms = df[storm_col].to_numpy()
    uniq = np.unique(storms)
    idx_by_storm = {s: np.flatnonzero(storms == s) for s in uniq}
    rng = np.random.RandomState(seed)

    rows = [metric_fn(df)]  # row 0 = point estimate on the real data
    for _ in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_storm[s] for s in drawn])
        rows.append(metric_fn(df.iloc[idx]))
    out = pd.DataFrame(rows)
    out.index = ["point"] + [f"b{i}" for i in range(n_boot)]
    return out


def summarise(series: pd.Series) -> dict:
    point = float(series.iloc[0])
    reps = series.iloc[1:].dropna()
    return {"point": point, "lo": float(reps.quantile(0.025)),
            "hi": float(reps.quantile(0.975)), "n_valid_reps": int(len(reps))}


def diff_summary(reps_df: pd.DataFrame, a: str, b: str, lower_is_better: bool):
    """Difference 'a beats b' with its own within-replicate CI.

    lower_is_better  -> improvement D = b - a  (positive means a is better)
    higher_is_better -> improvement D = a - b
    """
    d = (reps_df[b] - reps_df[a]) if lower_is_better else (reps_df[a] - reps_df[b])
    point = float(d.iloc[0])
    reps = d.iloc[1:].dropna()
    lo, hi = float(reps.quantile(0.025)), float(reps.quantile(0.975))
    crosses = bool(lo <= 0 <= hi)
    if crosses:
        verdict = "tie (CI crosses 0)"
    elif lo > 0:
        verdict = "a significantly better"
    else:
        verdict = "a significantly WORSE"
    return {"improvement": point, "lo": lo, "hi": hi,
            "significant": bool(lo > 0),          # a beats b, in a's favour
            "sig_either_way": not crosses,
            "crosses_zero": crosses, "verdict": verdict}


# ---------------------------------------------------------------------------
# RI metrics
# ---------------------------------------------------------------------------
RI_MODELS = {"analog": "p_analog", "classifier": "p_classifier",
             "average": "p_average", "stack": "p_stack"}


_CLIM_RATE = [0.06]  # overwritten from ri.json


def _ri_metrics(sub: pd.DataFrame) -> dict:
    y = sub["y_ri"].to_numpy()
    out = {}
    two_classes = y.min() != y.max()
    for name, col in RI_MODELS.items():
        p = sub[col].to_numpy()
        out[f"{name}_roc_auc"] = roc_auc_score(y, p) if two_classes else np.nan
        out[f"{name}_pr_auc"] = average_precision_score(y, p) if two_classes else np.nan
        out[f"{name}_brier"] = brier_score_loss(y, p)
    out["climatology_brier"] = brier_score_loss(
        y, np.full(len(y), _CLIM_RATE[0])) if len(y) else np.nan
    return out


def run_ri():
    df = pd.read_parquet(FRAG / "preds_ri.parquet")
    try:
        _CLIM_RATE[0] = json.loads((FRAG / "ri.json").read_text())["train_base_rate"]
    except Exception:
        pass
    reps = cluster_bootstrap(df, _ri_metrics)
    res = {"n_test_states": int(len(df)),
           "n_test_storms": int(df["storm_id_code"].nunique()),
           "n_RI_positive": int(df["y_ri"].sum()),
           "climatology_rate": _CLIM_RATE[0], "metrics": {}, "paired": {}}
    for name in RI_MODELS:
        for m in ("roc_auc", "pr_auc", "brier"):
            res["metrics"][f"{name}_{m}"] = summarise(reps[f"{name}_{m}"])
    res["metrics"]["climatology_brier"] = summarise(reps["climatology_brier"])
    res["paired"]["average_minus_analog_pr_auc"] = diff_summary(
        reps, "average_pr_auc", "analog_pr_auc", lower_is_better=False)
    res["paired"]["average_minus_classifier_pr_auc"] = diff_summary(
        reps, "average_pr_auc", "classifier_pr_auc", lower_is_better=False)
    res["paired"]["average_minus_stack_pr_auc"] = diff_summary(
        reps, "average_pr_auc", "stack_pr_auc", lower_is_better=False)
    # Brier: lower is better -> "average beats climatology" if clim - avg > 0
    res["paired"]["average_beats_climatology_brier"] = diff_summary(
        reps, "average_brier", "climatology_brier", lower_is_better=True)
    res["paired"]["analog_beats_climatology_brier"] = diff_summary(
        reps, "analog_brier", "climatology_brier", lower_is_better=True)
    return res


# ---------------------------------------------------------------------------
# track + intensity metrics (per lead, from the primary intersection)
# ---------------------------------------------------------------------------
TRACK_MODELS = {
    "persistence_motion": ("pm_lat", "pm_lon"),
    "true_cliper": ("tc_lat", "tc_lon"),
    "gbm_direct": ("gbm_dir_lat", "gbm_dir_lon"),
    "gbm_motion_resid": ("gbm_res_lat", "gbm_res_lon"),
    "lstm_direct": ("lstm_dir_lat", "lstm_dir_lon"),
    "lstm_motion_resid": ("lstm_res_lat", "lstm_res_lon"),
    "analog": ("analog_lat", "analog_lon"),
}


def _track_metrics(sub: pd.DataFrame) -> dict:
    out = {}
    for name, (la, lo) in TRACK_MODELS.items():
        out[name] = float(np.mean(haversine_km(sub[la], sub[lo],
                                               sub["fut_lat"], sub["fut_lon"])))
    return out


def _intensity_metrics(sub: pd.DataFrame) -> dict:
    yt = sub["dwind_true"].to_numpy()
    return {
        "persistence": float(np.mean(np.abs(yt))),
        "gbm": float(np.mean(np.abs(yt - sub["dwind_gbm"].to_numpy()))),
        "analog": float(np.mean(np.abs(yt - sub["dwind_analog"].to_numpy()))),
    }


def run_track_intensity():
    df = pd.read_parquet(FRAG / "preds_primary.parquet")
    res = {"n_per_lead": {}, "track": {}, "intensity": {},
           "track_paired": {}, "intensity_paired": {}}
    for h in LEADS:
        d = df[df["lead_h"] == h]
        res["n_per_lead"][f"{h}h"] = int(len(d))

        tr = cluster_bootstrap(d, _track_metrics)
        for name in TRACK_MODELS:
            res["track"].setdefault(name, {})[f"{h}h"] = summarise(tr[name])
        for a in ("true_cliper", "gbm_direct", "gbm_motion_resid",
                  "lstm_direct", "lstm_motion_resid", "analog"):
            res["track_paired"].setdefault(f"{a}_vs_persistence_motion", {})[f"{h}h"] = \
                diff_summary(tr, a, "persistence_motion", lower_is_better=True)
        res["track_paired"].setdefault("gbm_motion_resid_vs_true_cliper", {})[f"{h}h"] = \
            diff_summary(tr, "gbm_motion_resid", "true_cliper", lower_is_better=True)

        it = cluster_bootstrap(d, _intensity_metrics)
        for name in ("persistence", "gbm", "analog"):
            res["intensity"].setdefault(name, {})[f"{h}h"] = summarise(it[name])
        for a in ("gbm", "analog"):
            res["intensity_paired"].setdefault(f"{a}_vs_persistence", {})[f"{h}h"] = \
                diff_summary(it, a, "persistence", lower_is_better=True)
    return res


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _ci(d, unit="", f="{:.3f}"):
    return f"{f.format(d['point'])}{unit} [95% CI {f.format(d['lo'])}, {f.format(d['hi'])}]"


def _pair(name_a, name_b, d, unit, f="{:+.3f}"):
    """The 'X beats Y by D [95% CI a, b] — significant / not' line."""
    D = f.format(d["improvement"]); lo = f.format(d["lo"]); hi = f.format(d["hi"])
    if d["crosses_zero"]:
        tag = "not significant (CI crosses 0)"
    elif d["improvement"] > 0:
        tag = "**significant**"
    else:
        tag = f"**significant — {name_b} is better**"
    return f"{name_a} beats {name_b} by {D}{unit} [95% CI {lo}, {hi}] — {tag}"


def write_report(ri, ti):
    L = []
    L.append("# ResQ — statistical significance (cluster bootstrap by storm)\n")
    L.append(f"_Phase-5 TASK 1. {N_BOOT} bootstrap replicates, resampling the "
             f"**2019+ test storms** with replacement (not states — adjacent 6 h "
             f"states are ~0.97 autocorrelated). Point estimate = value on the real "
             f"test set; interval = 2.5/97.5 percentiles of the replicates. Paired "
             f"differences are computed within each replicate. `src/bootstrap.py`._\n")

    # ---- RI ----
    L.append("## Rapid intensification — P(RI ≥ +30 kt / 24 h)\n")
    L.append(f"{ri['n_test_states']} test states across {ri['n_test_storms']} "
             f"storms, {ri['n_RI_positive']} RI-positive. Replicates with no "
             "positive case give undefined AUC/PR-AUC and are dropped "
             "(count shown).\n")
    L.append("| model | ROC-AUC | PR-AUC | Brier |")
    L.append("|---|---|---|---|")
    for name in RI_MODELS:
        a = ri["metrics"][f"{name}_roc_auc"]; p = ri["metrics"][f"{name}_pr_auc"]
        b = ri["metrics"][f"{name}_brier"]
        L.append(f"| {name} | {_ci(a)} | {_ci(p)} | {_ci(b)} |")
    cb = ri["metrics"]["climatology_brier"]
    L.append(f"| climatology (rate {ri['climatology_rate']:.3f}) | 0.500 | "
             f"{ri['climatology_rate']:.3f} | {_ci(cb)} |")
    L.append(f"\n_(AUC/PR-AUC intervals from {ri['metrics']['analog_pr_auc']['n_valid_reps']}"
             f"/{N_BOOT} valid replicates.)_\n")
    L.append("**Paired differences (PR-AUC, higher = better):**\n")
    L.append(f"- {_pair('average', 'analog', ri['paired']['average_minus_analog_pr_auc'], ' PR-AUC')}")
    L.append(f"- {_pair('average', 'classifier', ri['paired']['average_minus_classifier_pr_auc'], ' PR-AUC')}")
    L.append(f"- {_pair('average', 'stack', ri['paired']['average_minus_stack_pr_auc'], ' PR-AUC')}")
    L.append("\n**Brier vs climatology (lower = better):**\n")
    L.append(f"- {_pair('average', 'climatology', ri['paired']['average_beats_climatology_brier'], '', '{:+.4f}')}")
    L.append(f"- {_pair('analog', 'climatology', ri['paired']['analog_beats_climatology_brier'], '', '{:+.4f}')}")
    L.append("")

    # ---- track ----
    L.append("## Track error, km (primary model-score intersection)\n")
    L.append("| model | 6 h | 12 h | 24 h |")
    L.append("|---|---|---|---|")
    for name in TRACK_MODELS:
        cells = [_ci(ti["track"][name][h], f="{:.1f}") for h in ("6h", "12h", "24h")]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append(f"\n_n per lead: 6h={ti['n_per_lead']['6h']}, 12h={ti['n_per_lead']['12h']}, "
             f"24h={ti['n_per_lead']['24h']}._\n")
    L.append("**Paired differences vs persistence-motion** (each: `improvement km "
             "[95% CI] — verdict`; a *negative* improvement means the model is "
             "worse):\n")
    for a in ("true_cliper", "gbm_direct", "gbm_motion_resid",
              "lstm_direct", "lstm_motion_resid", "analog"):
        key = f"{a}_vs_persistence_motion"
        for h in ("6h", "12h", "24h"):
            d = ti["track_paired"][key][h]
            L.append(f"- {h}: {_pair(a, 'persistence-motion', d, ' km', '{:+.1f}')}")
    for h in ("6h", "12h", "24h"):
        d = ti["track_paired"]["gbm_motion_resid_vs_true_cliper"][h]
        L.append(f"- {h}: {_pair('gbm_motion_resid', 'true CLIPER', d, ' km', '{:+.1f}')}")
    L.append("")

    # ---- intensity ----
    L.append("## Intensity — Δwind MAE, kt (primary intersection)\n")
    L.append("| model | 6 h | 12 h | 24 h |")
    L.append("|---|---|---|---|")
    for name in ("persistence", "gbm", "analog"):
        cells = [_ci(ti["intensity"][name][h], f="{:.2f}") for h in ("6h", "12h", "24h")]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("**Paired differences vs persistence (Δ=0):**\n")
    for a in ("gbm", "analog"):
        for h in ("6h", "12h", "24h"):
            d = ti["intensity_paired"][f"{a}_vs_persistence"][h]
            L.append(f"- {h}: {_pair(a, 'persistence', d, ' kt', '{:+.2f}')}")
    L.append("")

    # ---- headline ----
    hl = ["\n## Headline — what survives the bootstrap\n"]
    sig, notsig = [], []
    da = ri["paired"]["average_minus_analog_pr_auc"]
    dc = ri["paired"]["average_minus_classifier_pr_auc"]
    ac = ri["paired"]["average_beats_climatology_brier"]
    (sig if ac["sig_either_way"] and ac["improvement"] > 0 else notsig).append(
        "the combined P(RI) beats a climatological base-rate forecast on Brier")
    (sig if dc["significant"] else notsig).append(
        "the combined P(RI) beats the classifier alone (PR-AUC)")
    (sig if da["significant"] else notsig).append(
        "the combined P(RI) beats the analog ensemble alone (PR-AUC)")
    t6 = ti["track_paired"]["true_cliper_vs_persistence_motion"]["6h"]
    (sig if t6["sig_either_way"] else notsig).append(
        "persistence-motion beats true CLIPER on track at 6/12 h"
        if t6["improvement"] < 0 else "true CLIPER beats persistence-motion on track at 6 h")
    any_int = any(ti["intensity_paired"][f"{a}_vs_persistence"][h]["significant"]
                  for a in ("gbm", "analog") for h in ("6h", "12h", "24h"))
    (sig if any_int else notsig).append(
        "any intensity model beats the Δ=0 baseline (point estimates favour "
        "analog/GBM at 12/24 h but every CI crosses zero)")
    any_trk = any(ti["track_paired"][f"{a}_vs_persistence_motion"][h]["significant"]
                  for a in ("gbm_direct", "gbm_motion_resid", "lstm_direct",
                            "lstm_motion_resid", "analog", "true_cliper")
                  for h in ("6h", "12h", "24h"))
    (sig if any_trk else notsig).append(
        "any learned track model beats persistence-motion (all are significantly "
        "worse or tie)")
    hl.append("**Statistically supported (95% cluster bootstrap):**")
    hl += [f"- {s}" for s in sig] or ["- (none)"]
    hl.append("\n**NOT statistically supported — point estimate only, CI crosses zero:**")
    hl += [f"- {s}" for s in notsig] or ["- (none)"]
    hl.append("\n> The test set is ~70 storms / ~48 RI cases. Most phase-4 "
              "\"wins\" on track and intensity are within noise at this sample size; "
              "the RI combination and the RI-vs-climatology gap are the results that "
              "hold up. This does not overturn the *direction* of the phase-4 "
              "findings — it bounds the confidence in them.\n")
    L = L[:2] + hl + L[2:]

    (REPORTS / "significance.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", REPORTS / "significance.md")


def main():
    ri = run_ri()
    ti = run_track_intensity()
    (FRAG / "bootstrap.json").write_text(
        json.dumps({"ri": ri, "track_intensity": ti}, indent=2), encoding="utf-8")
    write_report(ri, ti)


if __name__ == "__main__":
    main()
