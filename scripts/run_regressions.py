"""
run_regressions.py
Every model, test and summary number quoted in REPORT.md, report_assigned.tex
and presentation.tex. Run scripts/extract_features.py first.

For each (position group, Combine metric, outcome) two specifications are fit:
  raw         metric + the original covariate (40 time, or weight for DL)
  controlled  raw + log(draft pick) + UDFA flag + draft-class fixed effects
              (+ position fixed effects for DL, DB and OL)
OLS uses heteroskedasticity-robust (HC1) standard errors. Holm's correction is
applied across the whole family of controlled tests.

Writes results/model_results.csv, results/descriptives.txt.
"""
import io
import os
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import common as cm

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)

CONTROLS = "log_pick + udfa + C(draft_year)"
MIN_ROUTES, MIN_TARGETS, MIN_RUSHES = 100, 20, 100


def load(k):
    return pd.read_parquet(os.path.join(cm.OUT_DIR, f"features_{k}.parquet"))


def fit(formula, d, kind):
    if kind == "ols":
        return smf.ols(formula, d).fit(cov_type="HC1")
    return smf.logit(formula, d).fit(disp=False, maxiter=200)


def model_rows(group, metric, label, outcome, olabel, kind, d, covar, extra_fe=""):
    rows = []
    d = d.dropna(subset=[metric, outcome] + covar.split(" + "))
    for spec, rhs in [("raw", f"{metric} + {covar}"),
                      ("controlled", f"{metric} + {covar} + {CONTROLS}{extra_fe}")]:
        m = fit(f"{outcome} ~ {rhs}", d, kind)
        rows.append(dict(group=group, metric=label, outcome=olabel, spec=spec, model=kind, N=int(m.nobs),
                         coef=m.params[metric], se=m.bse[metric], p=m.pvalues[metric],
                         odds_ratio=np.exp(m.params[metric]) if kind == "logit" else np.nan,
                         r2=getattr(m, "rsquared", np.nan) if kind == "ols" else m.prsquared))
    return rows


def all_models(wr, dl, db, ol):
    R = []
    wr_eff = wr[wr.routes >= MIN_ROUTES]
    wr_sep = wr[wr.n_separation >= MIN_TARGETS]
    for metric, label in [("s_cut", "WR cut speed (pooled)"), ("s_cut_z", "WR cut speed (drill-standardised)")]:
        R += model_rows("WR", metric, label, "career_offensive_snaps", "Career snaps", "ols", wr, "forty")
        R += model_rows("WR", metric, label, "starter", "Starter (>=10 starts)", "logit", wr, "forty")
        R += model_rows("WR", metric, label, "yprr", f"Yards per route run (>={MIN_ROUTES} routes)", "ols", wr_eff, "forty")
        R += model_rows("WR", metric, label, "targets_per_route", f"Targets per route (>={MIN_ROUTES} routes)", "ols", wr_eff, "forty")
        R += model_rows("WR", metric, label, "separation", f"Separation at throw (>={MIN_TARGETS} targets)", "ols", wr_sep, "forty")
    dl_eff = dl[dl.rush_snaps >= MIN_RUSHES]
    pos = " + C(nfl_position)"
    for metric, label in [("gamma_all_oe", "DL gamma-OE (all drills)"), ("gamma_pr_oe", "DL gamma-OE (pass-rush drill)")]:
        R += model_rows("DL", metric, label, "starter", "Starter (>=10 starts)", "logit", dl, "combine_weight", pos)
        R += model_rows("DL", metric, label, "career_defensive_snaps", "Career snaps", "ols", dl, "combine_weight", pos)
        R += model_rows("DL", metric, label, "pressure_rate", f"Pressure rate (>={MIN_RUSHES} rushes)", "ols", dl_eff, "combine_weight", pos)
        R += model_rows("DL", metric, label, "get_off", f"Get-off time, s (>={MIN_RUSHES} rushes)", "ols", dl_eff, "combine_weight", pos)
    R += model_rows("DB", "db_p90", "DB break speed (p90)", "career_defensive_snaps", "Career snaps", "ols", db, "forty", pos)
    R += model_rows("DB", "db_p90", "DB break speed (p90)", "starter", "Starter (>=10 starts)", "logit", db, "forty", pos)
    R += model_rows("OL", "ol_oe", "OL gamma-OE (mirror/wave)", "career_offensive_snaps", "Career snaps", "ols", ol, "forty", pos)
    R += model_rows("OL", "ol_oe", "OL gamma-OE (mirror/wave)", "starter", "Starter (>=10 starts)", "logit", ol, "forty", pos)
    res = pd.DataFrame(R)
    ctl = res.spec == "controlled"
    res.loc[ctl, "p_holm"] = multipletests(res.loc[ctl, "p"], method="holm")[1]
    return res


def replication(wr, dl):
    """The original report's exact specifications (classical SEs)."""
    print("== Original specifications, classical standard errors ==")
    c = wr.dropna(subset=["forty"])
    m = smf.ols("career_offensive_snaps ~ s_cut + forty", c).fit()
    print(f"WR OLS snaps: b_cut={m.params.s_cut:.2f} se={m.bse.s_cut:.2f} p={m.pvalues.s_cut:.4f} "
          f"b_forty={m.params.forty:.2f} p_forty={m.pvalues.forty:.3f} R2={m.rsquared:.3f} N={int(m.nobs)}")
    m = smf.logit("starter ~ s_cut + forty", c).fit(disp=False)
    print(f"WR logit: b_cut={m.params.s_cut:.3f} se={m.bse.s_cut:.3f} p={m.pvalues.s_cut:.4f} OR={np.exp(m.params.s_cut):.2f} "
          f"b_forty={m.params.forty:.3f} p_forty={m.pvalues.forty:.4f} N={int(m.nobs)}")
    m = smf.logit("starter ~ gamma_all_oe + combine_weight", dl).fit(disp=False)
    print(f"DL logit: b={m.params.gamma_all_oe:.3f} se={m.bse.gamma_all_oe:.3f} p={m.pvalues.gamma_all_oe:.4f} "
          f"OR={np.exp(m.params.gamma_all_oe):.2f} N={int(m.nobs)} starters={int(dl.starter.sum())}")


def descriptives(wr, dl):
    print("\n== Draft capital ==")
    c = wr.dropna(subset=["forty"])
    r_pick = smf.ols(f"career_offensive_snaps ~ {CONTROLS}", c).fit().rsquared
    r_both = smf.ols(f"career_offensive_snaps ~ s_cut + forty + {CONTROLS}", c).fit().rsquared
    r_cut = smf.ols("career_offensive_snaps ~ s_cut + forty", c).fit().rsquared
    print(f"WR snaps R2: combine only={r_cut:.3f}  draft only={r_pick:.3f}  both={r_both:.3f}  (N={len(c)})")
    m = smf.ols("s_cut ~ log_pick + udfa", wr).fit(cov_type="HC1")
    print(f"WR s_cut ~ log_pick: b={m.params.log_pick:.3f} p={m.pvalues.log_pick:.4f}  corr={wr[['s_cut','log_pick']].corr().iloc[0,1]:.3f}")
    m = smf.ols("gamma_all_oe ~ log_pick + udfa", dl).fit(cov_type="HC1")
    print(f"DL gamma_all_oe ~ log_pick: b={m.params.log_pick:.3f} p={m.pvalues.log_pick:.4f}")
    print("WR draft-day counts:", wr.draft_day.value_counts().to_dict())

    print("\n== WR tiers by career starts (all WRs, pooled cut speed) ==")
    tier = pd.cut(wr.career_games_started, [-1, 2, 19, 10**6], labels=["Fringe (<=2 starts)", "Middle (3-19)", "Top (>=20)"])
    print(wr.groupby(tier, observed=False).agg(N=("nfl_id", "size"), mean_s_cut=("s_cut", "mean"),
                                               sd_s_cut=("s_cut", "std"), mean_draft_pick=("draft_overall_pick", "mean"),
                                               udfa_share=("udfa", "mean")).round(2).to_string())
    a, b = wr.loc[wr.career_games_started >= 20, "s_cut"], wr.loc[wr.career_games_started <= 2, "s_cut"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    print(f"Welch t (Top vs Fringe) = {t:.3f}, p = {p:.6f}, Cohen's d (pooled SD) = {(a.mean()-b.mean())/sp:.2f}")

    print("\n== WR starter rate by pooled cut-speed band (raw, not pick-adjusted) ==")
    band = pd.cut(wr.s_cut, [0, 16, 17, 99], right=False, labels=["<16", "16-17", ">=17"])
    print(wr.groupby(band, observed=False).agg(N=("nfl_id", "size"), starter_rate=("starter", "mean"),
                                               mean_pick_or_262=("log_pick", lambda x: np.exp(x).mean())).round(3).to_string())

    print("\n== WR efficiency sample ==")
    print(f"WRs with >={MIN_ROUTES} regular-season routes: {(wr.routes >= MIN_ROUTES).sum()} "
          f"(with a 40 time: {((wr.routes >= MIN_ROUTES) & wr.forty.notna()).sum()})")
    print(f"DL with >={MIN_RUSHES} pass rushes: {(dl.rush_snaps >= MIN_RUSHES).sum()}")


def case_studies(wr, dl, ct):
    print("\n== Four-receiver case study ==")
    # nfl_id: Nacua, Smith-Njigba, Thomas Jr. (listed as "Brian Thomas"), Johnston
    ids = [56042, 55884, 57144, 55885]
    cols = ["display_name", "draft_overall_pick", "career_games_started", "ap_all_pro_1st_team", "pro_bowl_original_ballot",
            "s_cut", "s_cut_z", "n_cut_drills", "peak_any_drill", "routes", "yprr"]
    print(wr.set_index("nfl_id").loc[ids].reset_index()[cols].round(2).to_string(index=False))

    print("\n== Comeback route (one rep each) ==")
    for name in ["Puka Nacua", "Quentin Johnston"]:
        r = wr[wr.display_name == name].iloc[0]
        g = (ct.filter((pl.col("nfl_id") == int(r.nfl_id)) & (pl.col("drill_name") == "COMEBACK_ROUTE_RIGHT"))
               .sort("time").to_pandas())
        kmh = g.s.to_numpy() * cm.YDS_TO_KMH
        brk = int(np.argmax(g.x.to_numpy()))
        lo = brk - 5 + int(np.argmin(kmh[brk - 5:brk + 6]))
        kg = r.combine_weight * cm.LB_TO_KG
        peak = kmh[:brk + 1].max()
        dsdt = np.diff(g.s.to_numpy()) / 0.1
        print(f"{name}: reps={g.event_id.nunique()} weight={r.combine_weight:.0f} lb ({kg:.1f} kg) | stem peak {peak:.1f} km/h | "
              f"slowest {kmh[lo]:.1f} km/h | peak braking {dsdt.min():.2f} yd/s^2 | "
              f"speed 0.5/1.0 s after slowest {kmh[min(lo+5,len(kmh)-1)]:.1f}/{kmh[min(lo+10,len(kmh)-1)]:.1f} km/h | "
              f"momentum at stem peak {kg*peak/3.6:.0f} kg m/s | shed in cut {kg*(peak-kmh[lo])/3.6:.0f} kg m/s")

    print("\n== Named defensive linemen ==")
    print(dl[dl.display_name.isin(["Braden Fiske", "Laiatu Latu", "Gervon Dexter"])]
          [["display_name", "combine_weight", "draft_overall_pick", "gamma_all_oe", "gamma_pr_oe", "career_games_started",
            "pressure_rate"]].round(2).to_string(index=False))


def run_models():
    os.makedirs(cm.RESULTS_DIR, exist_ok=True)
    wr, dl, db, ol = (load(k) for k in ("wr", "dl", "db", "ol"))
    res = all_models(wr, dl, db, ol)
    res.to_csv(os.path.join(cm.RESULTS_DIR, "model_results.csv"), index=False, float_format="%.6g")
    show = res.copy()
    for c in ["coef", "se", "odds_ratio", "r2"]:
        show[c] = show[c].round(3)
    show["p"] = show["p"].round(4)
    show["p_holm"] = show["p_holm"].round(3)
    print(show.to_string(index=False))

    buf = io.StringIO()
    with redirect_stdout(buf):
        replication(wr, dl)
        descriptives(wr, dl)
        case_studies(wr, dl, cm.combine_tracking())
    print(buf.getvalue())
    with open(os.path.join(cm.RESULTS_DIR, "descriptives.txt"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())


if __name__ == "__main__":
    run_models()
