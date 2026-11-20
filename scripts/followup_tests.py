"""
followup_tests.py
Runs results/followup_preregistration.md (T1-T7) on the treatments built by
followup_features.py. Writes results/followup_tests.csv and
results/followup_summary.txt.
"""
import os

import numpy as np
import pandas as pd
import patsy
from statsmodels.stats.multitest import multipletests

import common as cm
import cod_features as cf
import distribution_checks as DC
import play_level_fe as P

SEED = 2027
YD = 0.9144                 # provider s is yd/s; converted to m/s
B_MED = 4999
SKILL = {"WR", "TE", "RB", "DB"}
PP_COLS = ["game_id", "play_id", "nfl_id", "team_abbr", "lined_up_position", "down", "yards_to_go",
           "possession_team", "yardline_side", "yardline_number", "play_direction", "expected_points_added",
           "pass_result", "pass_length", "play_nullified_by_penalty", "team_coverage_type", "route_ran", "target",
           "yards_after_catch", "expected_yards_after_catch", "player_get_off", "time_to_pressure", "sack",
           "blitzing", "dropback_duration", "pressure_allowed", "peak_pressure_probability_allowed",
           "coverage_assignment", "team_coverage_man_zone"]
ROWS = []


def within_sd(d, col, unit):
    return float((d[col] - d.groupby(unit)[col].transform("mean")).std())


def est(test, label, d, y, treat, rhs, cluster, unit, primary=False, wild=True, scale=1.0, unit_name=""):
    """OLS with fixed effects; CR1 by cluster plus restricted wild cluster bootstrap."""
    yv, X = patsy.dmatrices(f"{y} ~ {treat} + {rhs}", d, return_type="dataframe")
    sub = d.loc[X.index]
    pt = DC.Partial(X.drop(columns=[treat]).to_numpy(float), sub[cluster].to_numpy())
    yt, xt = pt.M(yv.to_numpy(float).ravel()), pt.M(X[treat].to_numpy(float))
    if wild:
        r = DC.wild_cluster(pt, yt, xt)
        lo, hi, p = *r["ci_wild"], r["p_wild"]
        b, se = r["coef"], r["se_cluster"]
    else:
        b, se = pt.fit(yt, xt)
        lo, hi = b - 1.96 * se, b + 1.96 * se
        from scipy import stats
        p = 2 * stats.t.sf(abs(b / se), pt.G - 1)
    sd = within_sd(sub, treat, unit)
    row = dict(test=test, estimate=label, outcome=y, treatment=treat, n=len(sub), units=sub[unit].nunique(),
               clusters=pt.G, coef=b, se=se, ci_lo=lo, ci_hi=hi, p=p, primary=primary, within_sd=sd,
               per_sd=b * sd, per_sd_lo=lo * sd, per_sd_hi=hi * sd, per_unit=b * scale, unit_label=unit_name)
    ROWS.append(row)
    print(f"  {test} {label}: {b:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f} n={len(sub)} units={row['units']}", flush=True)
    return row


def opp_season(d, games):
    g = games[["game_id", "season", "home_team_abbr", "visitor_team_abbr"]]
    d = d.merge(g, on="game_id", how="left", suffixes=("", "_g"))
    own = d.team_abbr
    d["opp"] = np.where(own == d.home_team_abbr, d.visitor_team_abbr, d.home_team_abbr)
    d["opp_season"] = d.opp + "_" + d.season.astype(str)
    d["home"] = (own == d.home_team_abbr).astype(int)
    return d


def min_units(d, unit, k):
    n = d.groupby(unit)[unit].transform("size")
    return d[n >= k].copy()


# ---------------------------------------------------------------- T1 mediation
def t1(pl, pp):
    wr = pl[pl.pos_group == "WR"].merge(
        pp[["game_id", "play_id", "nfl_id", "expected_points_added", "pass_result", "play_nullified_by_penalty"]],
        on=["game_id", "play_id", "nfl_id"])
    wr = wr[wr.play_nullified_by_penalty != "Y"].reset_index(drop=True)
    wr["recv_epa"] = wr.expected_points_added * wr.targeted
    rhs = f"{P.COVARS} + {P.FE}"
    a = est("T1", "a: apex_z -> separation", wr, "separation", "apex_z", rhs, "nfl_id", "nfl_id")
    b = est("T1", "b: separation -> receiver EPA | apex_z", wr, "recv_epa", "separation", f"apex_z + {rhs}",
            "nfl_id", "nfl_id")
    est("T1", "total: apex_z -> receiver EPA", wr, "recv_epa", "apex_z", rhs, "nfl_id", "nfl_id")
    est("T1", "direct: apex_z -> receiver EPA | separation", wr, "recv_epa", "apex_z", f"separation + {rhs}",
        "nfl_id", "nfl_id")
    tg = wr[wr.targeted == 1].copy()
    tg["complete"] = (tg.pass_result == "C").astype(float)
    tg = min_units(tg, "nfl_id", 2)
    est("T1", "completion | target: apex_z", tg, "complete", "apex_z", P.COVARS + " + C(nfl_id) + C(route_ran) + "
        "C(team_coverage_type) + C(down)", "nfl_id", "nfl_id")

    # cluster bootstrap of a*b
    ya, Xa = patsy.dmatrices(f"separation ~ apex_z + {rhs}", wr, return_type="dataframe")
    yb, Xb = patsy.dmatrices(f"recv_epa ~ separation + apex_z + {rhs}", wr, return_type="dataframe")
    assert (Xa.index == Xb.index).all()
    ja, jb = list(Xa.columns).index("apex_z"), list(Xb.columns).index("separation")
    Xa, Xb, ya, yb = Xa.to_numpy(float), Xb.to_numpy(float), ya.to_numpy(float).ravel(), yb.to_numpy(float).ravel()
    g = pd.factorize(wr.loc[wr.index, "nfl_id"])[0]
    members = [np.flatnonzero(g == k) for k in range(g.max() + 1)]
    rng = np.random.default_rng(SEED)
    ab = np.empty(B_MED)
    for r in range(B_MED):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(members), len(members))])
        ca = np.linalg.lstsq(Xa[idx], ya[idx], rcond=None)[0][ja]
        cb = np.linalg.lstsq(Xb[idx], yb[idx], rcond=None)[0][jb]
        ab[r] = ca * cb
    obs = a["coef"] * b["coef"]
    p = min(1.0, 2 * min(np.mean(ab <= 0), np.mean(ab >= 0)))
    lo, hi = np.percentile(ab, [2.5, 97.5])
    ROWS.append(dict(test="T1", estimate="indirect a*b (primary)", outcome="recv_epa", treatment="apex_z",
                     n=len(wr), units=wr.nfl_id.nunique(), clusters=wr.nfl_id.nunique(), coef=obs, ci_lo=lo,
                     ci_hi=hi, p=p, primary=True, within_sd=1.0, per_sd=obs, per_sd_lo=lo, per_sd_hi=hi))
    print(f"  T1 indirect: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}] p={p:.4f}", flush=True)


# ---------------------------------------------------------------- T2 catch speed -> YAC
def t2(pp, fu, base, games):
    c = pp[pp.yards_after_catch.notna() & (pp.play_nullified_by_penalty != "Y")]
    c = c.merge(fu[["game_id", "play_id", "nfl_id", "s_caught"]], on=["game_id", "play_id", "nfl_id"])
    c = c.merge(base[["nfl_id", "pos_group"]], on="nfl_id")
    c = c[c.pos_group.isin(["WR", "TE", "RB"]) & c.s_caught.notna()].copy()
    c["s_caught"] = c.s_caught * YD
    c = opp_season(c, games)
    c["route"] = c.route_ran.fillna("NONE")
    c["cov"] = c.team_coverage_type.fillna("UNK")
    c["air"] = c.pass_length.fillna(0)
    c["to_goal"] = np.where(c.yardline_side == c.possession_team, 100 - c.yardline_number, c.yardline_number)
    c["yacoe"] = c.yards_after_catch - c.expected_yards_after_catch
    c["yac5"] = (c.yards_after_catch >= 5).astype(float)
    c = c.dropna(subset=["to_goal", "down", "yards_to_go"])
    c = min_units(c, "nfl_id", 2)
    print(f"T2: {len(c):,} catches, {c.nfl_id.nunique()} players", flush=True)
    if len(c) < 300 or c.nfl_id.nunique() < 20:
        print("T2 below minimum: descriptive only")
        return
    rhs = "air + yards_to_go + to_goal + C(nfl_id) + C(route) + C(cov) + C(down) + C(opp_season)"
    est("T2", "catch speed -> YAC (primary)", c, "yards_after_catch", "s_caught", rhs, "nfl_id", "nfl_id",
        primary=True, scale=1.0, unit_name="yd per m/s")
    est("T2", "catch speed -> YAC over expected", c.dropna(subset=["yacoe"]), "yacoe", "s_caught", rhs,
        "nfl_id", "nfl_id")
    est("T2", "catch speed -> P(YAC >= 5)", c, "yac5", "s_caught", rhs, "nfl_id", "nfl_id")


# ---------------------------------------------------------------- T3 rusher get-off -> pressure
def t3(pp, games):
    r = pp[pp.player_get_off.notna() & (pp.play_nullified_by_penalty != "Y")].copy()
    drop = pp.groupby(["game_id", "play_id"]).dropback_duration.max().rename("dropback").reset_index()
    r = r.merge(drop, on=["game_id", "play_id"], how="left")
    r = opp_season(r, games)
    r["getoff_fast"] = -r.player_get_off
    r["pressure"] = (r.time_to_pressure.notna() | (r.sack.fillna(0) > 0)).astype(float)
    r["sack_any"] = (r.sack.fillna(0) > 0).astype(float)
    r["blitz"] = r.blitzing.fillna(False).astype(float)
    r = r.dropna(subset=["down", "yards_to_go"])
    r = min_units(r, "nfl_id", 30)
    print(f"T3: {len(r):,} rushes, {r.nfl_id.nunique()} rushers, pressure rate {r.pressure.mean():.3f}", flush=True)
    rhs = "yards_to_go + blitz + C(nfl_id) + C(lined_up_position) + C(down) + C(opp_season)"
    est("T3", "get-off (faster) -> pressure (primary)", r, "pressure", "getoff_fast", rhs, "nfl_id", "nfl_id",
        primary=True, scale=0.1, unit_name="per 0.1 s faster")
    est("T3", "get-off -> sack", r, "sack_any", "getoff_fast", rhs, "nfl_id", "nfl_id")
    pr = min_units(r[r.time_to_pressure.notna()], "nfl_id", 2)
    est("T3", "get-off -> time to pressure | pressure", pr, "time_to_pressure", "getoff_fast", rhs, "nfl_id",
        "nfl_id")
    est("T3", "get-off -> pressure, + dropback", r.dropna(subset=["dropback"]), "pressure", "getoff_fast",
        "dropback + " + rhs, "nfl_id", "nfl_id")


# ---------------------------------------------------------------- T4 OL get-off -> pressure allowed
def t4(pp, fu, games):
    o = pp[pp.lined_up_position.isin(["G", "T", "C"]) & pp.peak_pressure_probability_allowed.notna()
           & (pp.play_nullified_by_penalty != "Y")]
    o = o.merge(fu[["game_id", "play_id", "nfl_id", "ol_getoff", "dx10"]], on=["game_id", "play_id", "nfl_id"])
    o = o.dropna(subset=["ol_getoff", "down", "yards_to_go"]).copy()
    o = opp_season(o, games)
    o["ol_fast"] = -o.ol_getoff
    o["retreat"] = np.where(o.play_direction == "right", -o.dx10, o.dx10)   # yards toward own end zone
    o["pressure_allowed_f"] = o.pressure_allowed.astype(float)
    o = min_units(o, "nfl_id", 30)
    print(f"T4: {len(o):,} snaps, {o.nfl_id.nunique()} linemen", flush=True)
    rhs = "yards_to_go + C(nfl_id) + C(lined_up_position) + C(down) + C(opp_season)"
    est("T4", "OL get-off (faster) -> peak pressure prob (primary)", o, "peak_pressure_probability_allowed",
        "ol_fast", rhs, "nfl_id", "nfl_id", primary=True, scale=0.1, unit_name="per 0.1 s faster")
    est("T4", "OL get-off -> pressure allowed", o.dropna(subset=["pressure_allowed_f"]), "pressure_allowed_f",
        "ol_fast", rhs, "nfl_id", "nfl_id")
    est("T4", "set depth at 1.0 s -> peak pressure prob", o, "peak_pressure_probability_allowed", "retreat", rhs,
        "nfl_id", "nfl_id", scale=1.0, unit_name="per yard")


# ---------------------------------------------------------------- T5 shoulders
def t5(pl, pp, sh):
    wr = pl[pl.pos_group == "WR"].merge(sh, on=["game_id", "play_id", "nfl_id"])
    wr = wr.merge(pp[["game_id", "play_id", "nfl_id", "play_nullified_by_penalty"]],
                  on=["game_id", "play_id", "nfl_id"])
    wr = wr[wr.play_nullified_by_penalty != "Y"].reset_index(drop=True)
    wr["shoulder_z"] = wr.groupby(["route_ran", "angle_bin"]).shoulder_lead.transform(P.zs)
    print(f"T5: {len(wr):,} routes; shoulder lead median {wr.shoulder_lead.median():.1f} deg, "
          f"IQR [{wr.shoulder_lead.quantile(.25):.1f}, {wr.shoulder_lead.quantile(.75):.1f}]", flush=True)
    rhs = f"{P.COVARS} + {P.FE}"
    est("T5", "early shoulder turn -> separation (primary)", wr, "separation", "shoulder_z", rhs, "nfl_id",
        "nfl_id", primary=True)
    est("T5", "early shoulder turn -> separation | apex_z", wr, "separation", "shoulder_z", f"apex_z + {rhs}",
        "nfl_id", "nfl_id")
    est("T5", "early shoulder turn, man", wr[wr.man == 1], "separation", "shoulder_z", rhs, "nfl_id", "nfl_id")
    est("T5", "early shoulder turn, zone", wr[wr.man == 0], "separation", "shoulder_z", rhs, "nfl_id", "nfl_id")
    return wr


# ---------------------------------------------------------------- T6 head to head
def t6(pl, pp, paths):
    man_rows = pp[pp.coverage_assignment == "MAN"][["game_id", "play_id", "nfl_id"]]
    wr = pl[(pl.pos_group == "WR") & (pl.man == 1)].merge(
        pp[["game_id", "play_id", "nfl_id", "play_nullified_by_penalty"]], on=["game_id", "play_id", "nfl_id"])
    wr = wr[wr.play_nullified_by_penalty != "Y"]
    P_ = {k: g for k, g in paths.groupby(["game_id", "play_id", "nfl_id"])}
    defs = man_rows.groupby(["game_id", "play_id"]).nfl_id.apply(list).to_dict()
    out = []
    for r in wr.itertuples(index=False):
        rp = P_.get((r.game_id, r.play_id, r.nfl_id))
        k = int(round(r.t_after_snap * 10))
        if rp is None or k >= len(rp):
            continue
        best = None
        for dn in defs.get((r.game_id, r.play_id), []):
            dp = P_.get((r.game_id, r.play_id, dn))
            if dp is None or k >= len(dp):
                continue
            dist = np.hypot(rp.x.to_numpy()[:k + 1] - dp.x.to_numpy()[:k + 1],
                            rp.y.to_numpy()[:k + 1] - dp.y.to_numpy()[:k + 1])
            if best is None or dist.mean() < best[0]:
                best = (dist.mean(), dist[-1], dn, float(dp.s.to_numpy()[k]))
        if best is None or best[1] > 3.0:          # yards
            continue
        out.append(dict(game_id=r.game_id, play_id=r.play_id, nfl_id=r.nfl_id, def_id=best[2],
                        dist_at_cut=best[1], wr_s=float(rp.s.to_numpy()[k]) * YD, def_s=best[3] * YD))
    h = pd.DataFrame(out)
    if h.empty:
        print("T6: no matched plays")
        return None
    h = wr.merge(h, on=["game_id", "play_id", "nfl_id"])
    h["gap"] = h.wr_s - h.def_s
    h = min_units(h, "nfl_id", 2)
    print(f"T6: {len(h):,} matched man plays, {h.nfl_id.nunique()} receivers, {h.def_id.nunique()} defenders; "
          f"median gap {h.gap.median():+.2f} m/s", flush=True)
    if len(h) < 200 or h.nfl_id.nunique() < 20:
        print("T6 below minimum: descriptive only")
        ROWS.append(dict(test="T6", estimate="below minimum (descriptive only)", n=len(h),
                         units=h.nfl_id.nunique(), primary=True, p=np.nan))
        return h
    keep_opp = h.groupby("opp_season").size().median() >= 5
    fe = "C(nfl_id) + C(route_ran) + C(team_coverage_type) + C(receiver_alignment) + C(down)" + \
         (" + C(opp_season)" if keep_opp else "")
    print(f"  T6 opponent x season FE kept: {keep_opp}")
    rhs = f"{P.COVARS} + {fe}"
    est("T6", "speed gap -> separation (primary)", h, "separation", "gap", rhs, "nfl_id", "nfl_id", primary=True,
        scale=1.0, unit_name="yd per m/s")
    est("T6", "receiver speed | defender speed", h, "separation", "wr_s", f"def_s + {rhs}", "nfl_id", "nfl_id")
    est("T6", "defender speed | receiver speed", h, "separation", "def_s", f"wr_s + {rhs}", "nfl_id", "nfl_id")
    hd = min_units(h, "def_id", 2)
    if hd.def_id.nunique() >= 10:
        rhs_d = f"{P.COVARS} + C(def_id) + C(route_ran) + C(team_coverage_type) + C(receiver_alignment) + C(down)"
        est("T6", "speed gap -> separation, defender FE", hd, "separation", "gap", rhs_d, "def_id", "def_id")
    return h


# ---------------------------------------------------------------- T7 rest
def t7(fu, pp, base, games, pl):
    g = games[games.season_type == "REG"].copy()
    g["date"] = pd.to_datetime((g.game_id // 100).astype(str), format="%Y%m%d")
    tg = pd.concat([g[["game_id", "season", "week", "date", "home_team_abbr"]].rename(columns={"home_team_abbr": "team"}),
                    g[["game_id", "season", "week", "date", "visitor_team_abbr"]].rename(columns={"visitor_team_abbr": "team"})])
    tg = tg.sort_values(["team", "season", "date"])
    tg["rest"] = tg.groupby(["team", "season"]).date.diff().dt.days
    tg["home"] = tg.game_id.map(dict(zip(g.game_id, g.home_team_abbr))) == tg.team

    pg = fu[fu.has_snap].groupby(["game_id", "nfl_id"]).agg(peak=("max_s", "max"), n_plays=("max_s", "size")).reset_index()
    pg["peak"] = pg.peak * cm.YDS_TO_KMH
    team = pp.groupby(["game_id", "nfl_id"]).team_abbr.first().rename("team").reset_index()
    d = pg.merge(team, on=["game_id", "nfl_id"]).merge(tg, on=["game_id", "team"])
    d = d.merge(base[["nfl_id", "pos_group"]], on="nfl_id")
    d = d[(d.week >= 2) & (d.n_plays >= 15) & d.rest.notna()].copy()
    d["short"] = (d.rest <= 5).astype(float)
    d["long"] = (d.rest >= 9).astype(float)
    d["home"] = d.home.astype(int)
    d["season_week"] = d.season.astype(str) + "_" + d.week.astype(str)
    d["team_season"] = d.team + "_" + d.season.astype(str)
    skill = min_units(d[d.pos_group.isin(SKILL)], "nfl_id", 2)
    print(f"T7: {len(skill):,} skill player-games ({int(skill.short.sum())} short, {int(skill.long.sum())} long), "
          f"{skill.nfl_id.nunique()} players, {skill.team_season.nunique()} team-seasons", flush=True)
    rhs = "long + home + C(nfl_id) + C(season_week)"
    est("T7", "short rest -> peak speed, skill (primary)", skill, "peak", "short", rhs, "team_season", "nfl_id",
        primary=True, scale=1.0, unit_name="km/h")
    est("T7", "long rest -> peak speed, skill", skill, "peak", "long", "short + home + C(nfl_id) + C(season_week)",
        "team_season", "nfl_id")
    est("T7", "short rest -> peak speed, skill, + plays", skill, "peak", "short", "n_plays + " + rhs,
        "team_season", "nfl_id")
    allp = min_units(d, "nfl_id", 2)
    est("T7", "short rest -> peak speed, all positions", allp, "peak", "short", rhs, "team_season", "nfl_id")
    a = pl[pl.pos_group == "WR"].groupby(["game_id", "nfl_id"]).agg(apex_g=("apex_z", "mean"),
                                                                   k=("apex_z", "size")).reset_index()
    wa = min_units(d.merge(a[a.k >= 3], on=["game_id", "nfl_id"]), "nfl_id", 2)
    if len(wa) > 100:
        est("T7", "short rest -> receiver mean apex_z", wa, "apex_g", "short", rhs, "team_season", "nfl_id")


def main():
    base = cm.player_base()
    base["pos_group"] = base.nfl_position.map(cf.POS_GROUP)
    games = pd.read_csv(cm.path("games.csv"))
    pp = cm.player_play_regular_season(usecols=PP_COLS)
    pl = pd.read_parquet(os.path.join(cm.OUT_DIR, "play_level.parquet"))
    fu = pd.read_parquet(os.path.join(cm.OUT_DIR, "fu_player_play.parquet"))
    sh = pd.read_parquet(os.path.join(cm.OUT_DIR, "fu_shoulders.parquet"))
    paths = pd.read_parquet(os.path.join(cm.OUT_DIR, "fu_paths.parquet"))

    t1(pl, pp)
    t2(pp, fu, base, games)
    t3(pp, games)
    t4(pp, fu, games)
    t5(pl, pp, sh)
    h = t6(pl, pp, paths)
    t7(fu, pp, base, games, pl)

    R = pd.DataFrame(ROWS)
    R["p_holm"] = np.nan
    prim = R.primary & R.p.notna()
    R.loc[prim, "p_holm"] = multipletests(R.loc[prim, "p"], method="holm")[1]
    for t in R.test.unique():
        m = (R.test == t) & ~R.primary & R.p.notna()
        if m.sum():
            R.loc[m, "p_holm"] = multipletests(R.loc[m, "p"], method="holm")[1]
    R.to_csv(os.path.join(cm.RESULTS_DIR, "followup_tests.csv"), index=False)
    if h is not None:
        h.to_parquet(os.path.join(cm.OUT_DIR, "fu_head_to_head.parquet"), index=False)
    lines = ["Follow-up causal tests (results/followup_preregistration.md)",
             "coef [95% wild-bootstrap CI] (T1 indirect: cluster bootstrap); per SD = per 1 SD within unit", ""]
    for r in R.itertuples():
        if pd.isna(r.p):
            lines.append(f"{r.test}  {r.estimate}  n={r.n}")
            continue
        tag = "PRIMARY " if r.primary else ""
        lines.append(f"{r.test}  {tag}{r.estimate}: {r.coef:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], p = {r.p:.4f}, "
                     f"Holm {r.p_holm:.4f}; per SD {r.per_sd:+.4f}; n = {r.n:,}, units = {r.units}")
    txt = "\n".join(lines)
    with open(os.path.join(cm.RESULTS_DIR, "followup_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
