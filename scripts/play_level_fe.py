"""
play_level_fe.py
Runs results/play_fe_preregistration.md: within-player, play-level test of
whether holding speed (idea 1) or producing turning force (idea 2) through the
last cut before the throw goes with more separation at the throw.

Inputs: data/game_cuts.parquet (game_cod_features.py), player_play.csv, games.csv.
Writes data/play_level.parquet, results/play_fe_models.csv,
results/play_fe_conformal.csv and results/play_fe_summary.txt.
"""
import os

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

import common as cm
import cod_features as cf

MIN_GROUP, MIN_PLAYS = 30, 30
ALPHA, N_PERM, SEED = 0.20, 200, 2027
FE = ("C(nfl_id) + C(route_ran) + C(team_coverage_type) + C(receiver_alignment) + C(down) + C(opp_season)")
COVARS = "turn_deg + t_after_snap + yards_to_go + cushion + dropback"
PP_COLS = ["game_id", "play_id", "nfl_id", "route_ran", "target", "separation_at_pass_forward", "cushion",
           "team_coverage_man_zone", "team_coverage_type", "receiver_alignment", "down", "yards_to_go",
           "team_abbr", "dropback_duration"]


def zs(s):
    return (s - s.mean()) / s.std(ddof=1)


def build():
    base = cm.player_base()
    pos = base[["nfl_id", "nfl_position", "display_name"]].copy()
    pos["pos_group"] = pos.nfl_position.map(cf.POS_GROUP)
    pp = cm.player_play_regular_season(usecols=PP_COLS)
    drop = pp.groupby(["game_id", "play_id"]).dropback_duration.max().rename("dropback").reset_index()
    games = pd.read_csv(cm.path("games.csv"))

    cuts = pd.read_parquet(os.path.join(cm.OUT_DIR, "game_cuts.parquet"))
    cuts = cuts[cuts.pass_play].merge(pos, on="nfl_id")
    cuts = cuts[cuts.pos_group.isin(["WR", "TE"])]
    cuts = cuts.merge(pp[["game_id", "play_id", "nfl_id", "route_ran"]], on=["game_id", "play_id", "nfl_id"])
    cuts = cuts[cuts.route_ran.notna()].copy()
    cuts["angle_bin"] = pd.cut(cuts.turn_deg, cf.ANGLE_BINS, right=False).astype(str)
    g = cuts.groupby(["pos_group", "route_ran", "angle_bin"])
    cuts = cuts[g.apex.transform("size") >= MIN_GROUP].copy()
    cuts["apex_z"] = cuts.groupby(["pos_group", "route_ran", "angle_bin"]).apex.transform(zs)
    cuts["entry_z"] = cuts.groupby(["pos_group", "route_ran", "angle_bin"]).entry.transform(zs)
    cuts["n_cuts"] = cuts.groupby(["game_id", "play_id", "nfl_id"]).apex.transform("size")

    last = cuts.sort_values("t_after_snap").groupby(["game_id", "play_id", "nfl_id"]).tail(1)
    d = last.merge(pp.drop(columns=["route_ran", "dropback_duration"]), on=["game_id", "play_id", "nfl_id"])
    d = d.merge(drop, on=["game_id", "play_id"], how="left").merge(games[["game_id", "home_team_abbr",
                                                                           "visitor_team_abbr"]], on="game_id")
    d["opp_season"] = np.where(d.team_abbr == d.home_team_abbr, d.visitor_team_abbr, d.home_team_abbr) \
        + "_" + d.season.astype(str)
    d["turn_force"] = d.f_turn / 100.0
    d["separation"] = d.separation_at_pass_forward
    d["targeted"] = d.target.astype(float)
    d["man"] = (d.team_coverage_man_zone == "MAN_COVERAGE").astype(int)
    d["cut_to_throw"] = d.dropback - d.t_after_snap
    need = ["separation", "cushion", "dropback", "team_coverage_type", "receiver_alignment", "down", "yards_to_go"]
    d = d.dropna(subset=need)
    n = d.groupby("nfl_id").size()
    d = d[d.nfl_id.isin(n[n >= MIN_PLAYS].index)].copy()
    d.to_parquet(os.path.join(cm.OUT_DIR, "play_level.parquet"), index=False)
    return d


def fit(d, y, treat, extra=""):
    f = f"{y} ~ {treat} + {COVARS} + {FE}{extra}"
    return smf.ols(f, d).fit(cov_type="cluster", cov_kwds={"groups": d.nfl_id.to_numpy()})


def models(d):
    wr, te = d[d.pos_group == "WR"], d[d.pos_group == "TE"]
    plan = [("apex_z", "separation", "WR, all coverage", wr, True),
            ("turn_force", "separation", "WR, all coverage", wr, True),
            ("apex_z", "separation", "WR, man", wr[wr.man == 1], False),
            ("apex_z", "separation", "WR, zone", wr[wr.man == 0], False),
            ("turn_force", "separation", "WR, man", wr[wr.man == 1], False),
            ("turn_force", "separation", "WR, zone", wr[wr.man == 0], False),
            ("apex_z", "targeted", "WR, all coverage", wr, False),
            ("turn_force", "targeted", "WR, all coverage", wr, False),
            ("apex_z", "separation", "TE, all coverage", te, False),
            ("turn_force", "separation", "TE, all coverage", te, False)]
    rows = []
    for treat, y, label, s, primary in plan:
        m = fit(s, y, treat)
        lo, hi = m.conf_int().loc[treat]
        rows.append(dict(treatment=treat, outcome=y, sample=label, n_plays=int(m.nobs), n_players=s.nfl_id.nunique(),
                         coef=m.params[treat], ci_lo=lo, ci_hi=hi, p=m.pvalues[treat], primary=primary,
                         sd_treat_within=s.groupby("nfl_id")[treat].transform(lambda v: v - v.mean()).std()))
    for label, s in [("WR, all coverage", wr), ("WR, man", wr[wr.man == 1])]:   # both treatments together
        m = fit(s, "separation", "apex_z + turn_force")
        for treat in ["apex_z", "turn_force"]:
            lo, hi = m.conf_int().loc[treat]
            rows.append(dict(treatment=f"{treat} (joint)", outcome="separation", sample=label, n_plays=int(m.nobs),
                             n_players=s.nfl_id.nunique(), coef=m.params[treat], ci_lo=lo, ci_hi=hi,
                             p=m.pvalues[treat], primary=False))
    R = pd.DataFrame(rows)
    R["p_holm"] = np.nan
    for mask in [R.primary, ~R.primary]:
        R.loc[mask, "p_holm"] = multipletests(R.loc[mask, "p"], method="holm")[1]
    return R


def robustness(d):
    """Post-hoc checks (not in the plan) on the idea-1 estimate, WR."""
    wr = d[d.pos_group == "WR"]
    man = wr[wr.man == 1]
    fe_game = FE.replace("C(opp_season)", "C(game_id)")
    checks = [
        ("plan model (reference)", wr, "apex_z", FE, ""),
        ("+ entry speed into the cut", wr, "apex_z", FE, " + entry_z"),
        ("+ entry speed, man coverage", man, "apex_z", FE, " + entry_z"),
        ("placebo: entry speed alone", wr, "entry_z", FE, ""),
        ("cut >= 0.3 s before the throw", wr[wr.cut_to_throw >= 0.3], "apex_z", FE, ""),
        ("cut >= 0.6 s before the throw", wr[wr.cut_to_throw >= 0.6], "apex_z", FE, ""),
        ("single-cut plays only", wr[wr.n_cuts == 1], "apex_z", FE, ""),
        ("game FE instead of opponent x season", wr, "apex_z", fe_game, ""),
    ]
    rows = []
    for label, s, treat, fe, extra in checks:
        m = smf.ols(f"separation ~ {treat} + {COVARS} + {fe}{extra}", s).fit(
            cov_type="cluster", cov_kwds={"groups": s.nfl_id.to_numpy()})
        lo, hi = m.conf_int().loc[treat]
        rows.append(dict(check=label, treatment=treat, n_plays=int(m.nobs), coef=m.params[treat], ci_lo=lo, ci_hi=hi,
                         p=m.pvalues[treat]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- conformal (CV+)
def cv_plus(X, y, outer, rng, inner_k=5):
    n = len(y)
    lo, hi = np.empty(n), np.empty(n)
    for test in outer:
        train = np.setdiff1d(np.arange(n), test)
        folds = np.array_split(rng.permutation(train), inner_k)
        P, R = [], []
        for f in folds:
            tr = np.setdiff1d(train, f)
            b, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            R.append(np.abs(y[f] - X[f] @ b))
            P.append(np.repeat((X[test] @ b)[None, :], len(f), 0))
        R, P = np.concatenate(R)[:, None], np.concatenate(P, 0)
        m = len(R)
        lo[test] = np.sort(P - R, 0)[max(int(np.floor(ALPHA * (m + 1))) - 1, 0)]
        hi[test] = np.sort(P + R, 0)[min(int(np.ceil((1 - ALPHA) * (m + 1))) - 1, m - 1)]
    return lo, hi


def conformal(d):
    wr = d[d.pos_group == "WR"].reset_index(drop=True)
    out = []
    for treat in ["apex_z", "turn_force"]:
        _, Xf = patsy.dmatrices(f"separation ~ {treat} + {COVARS} + {FE}", wr, return_type="dataframe")
        _, Xb = patsy.dmatrices(f"separation ~ {COVARS} + {FE}", wr, return_type="dataframe")
        j = list(Xf.columns).index(treat)
        Xf, Xb = Xf.to_numpy(float), Xb.to_numpy(float)
        y = wr.separation.to_numpy(float)
        rng = np.random.default_rng(SEED)
        outer = np.array_split(rng.permutation(len(y)), 10)
        state = rng.bit_generator.state

        def width(X):
            rng.bit_generator.state = state
            lo, hi = cv_plus(X, y, outer, rng)
            return np.mean(hi - lo), np.mean((y >= lo) & (y <= hi))

        wb, cb = width(Xb)
        wf, cf_ = width(Xf)
        prng = np.random.default_rng(SEED + 1)
        null = np.empty(N_PERM)
        for b in range(N_PERM):
            Xp = Xf.copy()
            Xp[:, j] = prng.permutation(Xp[:, j])
            null[b] = wb - width(Xp)[0]
        out.append(dict(treatment=treat, n_plays=len(y), coverage_base=cb, coverage_with=cf_, width_base=wb,
                        width_with=wf, width_change_pct=(wf - wb) / wb * 100,
                        null_median_change_pct=-np.median(null) / wb * 100,
                        perm_p=(1 + np.sum(null >= wb - wf)) / (N_PERM + 1)))
        print(f"conformal {treat}: {out[-1]}", flush=True)
    return pd.DataFrame(out)


def main():
    d = build()
    R = models(d)
    B = robustness(d)
    C = conformal(d)
    R.to_csv(os.path.join(cm.RESULTS_DIR, "play_fe_models.csv"), index=False)
    B.to_csv(os.path.join(cm.RESULTS_DIR, "play_fe_robustness.csv"), index=False)
    C.to_csv(os.path.join(cm.RESULTS_DIR, "play_fe_conformal.csv"), index=False)
    wr = d[d.pos_group == "WR"]
    L = ["Play-level fixed-effects test: last cut before the throw -> separation at the throw",
         "Plan: results/play_fe_preregistration.md", ""]
    L.append(f"Receivers: {wr.nfl_id.nunique()} players, {len(wr):,} plays (median "
             f"{wr.groupby('nfl_id').size().median():.0f} per player); man coverage {wr.man.mean():.0%}")
    L.append(f"Tight ends: {d[d.pos_group == 'TE'].nfl_id.nunique()} players, {(d.pos_group == 'TE').sum():,} plays")
    L.append(f"Separation: mean {wr.separation.mean():.2f} yd, SD {wr.separation.std():.2f}; "
             f"turning force mean {wr.f_turn.mean():.0f} N, within-player SD {wr.turn_force.sub(wr.groupby('nfl_id').turn_force.transform('mean')).std() * 100:.0f} N; "
             f"apex_z within-player SD {wr.apex_z.sub(wr.groupby('nfl_id').apex_z.transform('mean')).std():.2f}")
    L.append(f"Correlation of apex_z and turning force within play: {wr.apex_z.corr(wr.turn_force):+.2f}")
    L.append("")
    L.append("ESTIMATES (player FE + route, coverage, alignment, down, opponent x season FE; clustered by player)")
    unit = {"apex_z": "per 1 SD", "turn_force": "per 100 N"}
    for r in R.itertuples():
        u = unit.get(r.treatment.split(" ")[0], "")
        y = "yd separation" if r.outcome == "separation" else "probability of target"
        tag = "PRIMARY " if r.primary else "        "
        L.append(f"  {tag}{r.treatment:20s} {r.sample:18s} plays={r.n_plays:5d}  {r.coef:+.3f} {y} {u} "
                 f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}], p = {r.p:.3f}, Holm {r.p_holm:.3f}")
    L.append("")
    L.append("ROBUSTNESS (post hoc, not in the plan; WR separation, per 1 SD)")
    for b in B.itertuples():
        L.append(f"  {b.check:38s} {b.treatment:8s} plays={b.n_plays:5d}  {b.coef:+.3f} [{b.ci_lo:+.3f}, {b.ci_hi:+.3f}], "
                 f"p = {b.p:.3f}")
    L.append("")
    L.append("CONFORMAL (80% CV+ intervals for separation; WR, all coverage)")
    for c in C.itertuples():
        L.append(f"  {c.treatment:10s} coverage {c.coverage_base:.2f} -> {c.coverage_with:.2f}; width {c.width_base:.3f} -> "
                 f"{c.width_with:.3f} yd ({c.width_change_pct:+.2f}%; noise {c.null_median_change_pct:+.2f}%), perm p = {c.perm_p:.3f}")
    txt = "\n".join(L)
    with open(os.path.join(cm.RESULTS_DIR, "play_fe_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
