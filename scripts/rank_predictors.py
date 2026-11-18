"""
rank_predictors.py
Ranks every available predictor (draft slot, Combine results, NGS scores and
the tracking metrics) by how well it predicts NFL outcomes. Exploratory: no
multiple-testing correction.

Playing time (all four position groups)
  auc_alone      cross-validated AUC for starter status (10+ starts), predictor alone
  auc_gain       AUC with draft pick + UDFA flag + predictor, minus AUC with draft pick + UDFA flag
  spearman_snaps rank correlation with career snaps per season
  (5-fold cross-validation repeated 20 times; logistic regression)

Per-snap efficiency (WR yards per route run, 100+ routes; DL pressure rate, 100+ rushes)
  spearman       rank correlation with the efficiency outcome
  partial        rank correlation after removing log(draft pick) (and edge vs interior for DL)

Writes results/predictor_ranking_volume.csv and results/predictor_ranking_efficiency.csv.
Run extract_features.py first.
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

import common as cm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

LABEL = {
    "log_pick": "Draft pick (log)", "udfa": "Undrafted", "forty": "40-yard dash", "ten_yd_split": "10-yard split",
    "vertical": "Vertical jump", "broad_jump": "Broad jump", "three_cone": "3-cone", "short_shuttle": "Short shuttle",
    "bench_reps": "Bench reps", "combine_weight": "Weight", "combine_height": "Height", "arm_length": "Arm length",
    "hand_size": "Hand size", "wing_span": "Wingspan", "ngs_athleticism_score": "NGS athleticism score",
    "ngs_college_production_score": "NGS college production score", "ngs_final_score": "NGS final score",
    "s_cut": "Tracking: cut speed (pooled)", "s_cut_z": "Tracking: cut speed (drill-standardised)",
    "peak_any_drill": "Tracking: peak speed", "gamma_all_oe": "Tracking: gamma-OE (all drills)",
    "gamma_pr_oe": "Tracking: gamma-OE (pass-rush drill)", "db_p90": "Tracking: DB break speed",
    "ol_oe": "Tracking: OL gamma-OE",
}
STANDARD = ["log_pick", "udfa", "forty", "ten_yd_split", "vertical", "broad_jump", "three_cone", "short_shuttle",
            "bench_reps", "combine_weight", "combine_height", "arm_length", "hand_size", "wing_span",
            "ngs_athleticism_score", "ngs_college_production_score", "ngs_final_score"]
TRACKING = {"wr": ["s_cut", "s_cut_z", "peak_any_drill"], "dl": ["gamma_all_oe", "gamma_pr_oe"],
            "db": ["db_p90"], "ol": ["ol_oe"]}
MIN_N = 40


def zscore(d, cols):
    X = d[cols].astype(float)
    return ((X - X.mean()) / X.std(ddof=0)).to_numpy()


def cv_auc(X, y, reps=20):
    folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=0)
    scores = []
    for tr, te in folds.split(X, y):
        m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        scores.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(scores))


def volume_ranking():
    rows = []
    for g in ("wr", "dl", "db", "ol"):
        df = pd.read_parquet(os.path.join(cm.OUT_DIR, f"features_{g}.parquet"))
        snaps = df["career_offensive_snaps"] if g in ("wr", "ol") else df["career_defensive_snaps"]
        df["snaps_per_season"] = snaps / df["seasons"]
        for p in STANDARD + TRACKING[g]:
            d = df.dropna(subset=[p]).reset_index(drop=True)
            if len(d) < MIN_N or d[p].nunique() < 2:
                continue
            y = d["starter"].to_numpy()
            if not 5 <= y.sum() <= len(y) - 5:
                continue
            alone = cv_auc(zscore(d, [p]), y)
            gain = np.nan
            if p not in ("log_pick", "udfa"):
                gain = cv_auc(zscore(d, ["log_pick", "udfa", p]), y) - cv_auc(zscore(d, ["log_pick", "udfa"]), y)
            rows.append(dict(group=g.upper(), predictor=LABEL[p], N=len(d), auc_alone=alone, auc_gain=gain,
                             spearman_snaps=stats.spearmanr(d[p], d["snaps_per_season"])[0]))
    return pd.DataFrame(rows).sort_values(["group", "auc_alone"], ascending=[True, False])


def partial_rank_corr(d, x, y, controls):
    r = d[[x, y]].rank()
    Z = np.column_stack([np.ones(len(d))] + [d[c].rank() if d[c].nunique() > 2 else d[c] for c in controls])
    resid = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return stats.pearsonr(resid(r[x].to_numpy()), resid(r[y].to_numpy()))


def efficiency_ranking():
    rows = []
    specs = [("wr", "yprr", "routes", ["log_pick"], "WR yards per route run"),
             ("dl", "pressure_rate", "rush_snaps", ["log_pick", "edge"], "DL pressure rate")]
    for g, y, volume, controls, outcome in specs:
        df = pd.read_parquet(os.path.join(cm.OUT_DIR, f"features_{g}.parquet"))
        df = df[df[volume] >= 100].copy()
        df["edge"] = df["nfl_position"].isin(["DE", "OLB"]).astype(float)
        for p in STANDARD + TRACKING[g]:
            if p in ("udfa", "log_pick"):
                continue
            d = df.dropna(subset=[p, y])
            if len(d) < 30 or d[p].nunique() < 2:
                continue
            rho, p_rho = stats.spearmanr(d[p], d[y])
            part, p_part = partial_rank_corr(d, p, y, controls)
            rows.append(dict(outcome=outcome, predictor=LABEL[p], N=len(d), spearman=rho, p=p_rho,
                             partial=part, p_partial=p_part, controls=" + ".join(controls)))
    out = pd.DataFrame(rows)
    out["abs_partial"] = out["partial"].abs()
    return out.sort_values(["outcome", "abs_partial"], ascending=[False, False]).drop(columns="abs_partial")


def main():
    os.makedirs(cm.RESULTS_DIR, exist_ok=True)
    vol = volume_ranking()
    eff = efficiency_ranking()
    vol.to_csv(os.path.join(cm.RESULTS_DIR, "predictor_ranking_volume.csv"), index=False, float_format="%.4f")
    eff.to_csv(os.path.join(cm.RESULTS_DIR, "predictor_ranking_efficiency.csv"), index=False, float_format="%.4f")
    print("== Playing time: starter AUC alone, and gain over draft pick ==")
    print(vol.round(3).to_string(index=False))
    print("\n== Per-snap efficiency: rank correlations ==")
    print(eff.round(3).to_string(index=False))
    print(f"\nEfficiency correlations tested: {len(eff)}")


if __name__ == "__main__":
    main()
