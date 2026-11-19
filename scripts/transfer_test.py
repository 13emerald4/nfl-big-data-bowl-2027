"""
transfer_test.py
Runs results/transfer_preregistration.md: do the Combine change-of-direction
metrics track the same quantities measured in games?

Inputs: data/cod_events.parquet and data/features_cod.parquet (cod_features.py),
data/game_cuts.parquet (game_cod_features.py), player_play.csv.

Writes results/transfer_correlations.csv, results/transfer_outcomes.csv and
results/transfer_summary.txt.
"""
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

import common as cm
import cod_features as cf

MIN_CUTS = 30
MIN_GROUP = 30
CONTROLS = "log_pick + udfa + C(draft_year)"
PRIMARY = [("cod_speed_z", "g_route_speed_z", "WR", False),
           ("force_oe_z", "g_force_oe_z", "ALL", True),
           ("turn_oe_z", "g_turn_oe_z", "ALL", True)]
SECONDARY = [("cod_retention_z", "g_route_retention_z", "WR", False),
             ("tc_cut_speed_z", "g_cut_speed_z", "ALL", False),
             ("brake_oe_z", "g_brake_oe_z", "ALL", True),
             ("force_abs_z", "g_force_n", "ALL", False),
             ("force_per_kg_z", "g_force_per_kg", "ALL", True)]


def zs(s):
    return (s - s.mean()) / s.std(ddof=1)


def sb(r):
    return 2 * r / (1 + r)


# ---------------------------------------------------------------- game metrics
def load_cuts(base, pp):
    cuts = pd.read_parquet(os.path.join(cm.OUT_DIR, "game_cuts.parquet"))
    cuts = cuts.merge(pp[["game_id", "play_id", "nfl_id", "route_ran"]], on=["game_id", "play_id", "nfl_id"], how="left")
    pos = base[["nfl_id", "nfl_position", "combine_weight"]].copy()
    pos["pos_group"] = pos.nfl_position.map(cf.POS_GROUP).fillna("OTHER")
    pos["mass_kg"] = pos.combine_weight * cm.LB_TO_KG
    cuts = cuts.merge(pos[["nfl_id", "pos_group", "mass_kg"]], on="nfl_id")
    cuts["angle_bin"] = pd.cut(cuts.turn_deg, cf.ANGLE_BINS, right=False).astype(str)
    return cuts


def add_z(cuts):
    """Peer z-scores per cut: route x angle (receivers on routes), position x angle (all)."""
    c = cuts.copy()
    g = c.groupby(["pos_group", "angle_bin"])
    ok = g.apex.transform("size") >= MIN_GROUP
    c.loc[ok, "pos_speed_z"] = g.apex.transform(zs)[ok]
    wr = (c.pos_group == "WR") & c.route_ran.notna()
    g = c[wr].groupby(["route_ran", "angle_bin"])
    size = g.apex.transform("size")
    c.loc[wr, "route_speed_z"] = np.where(size >= MIN_GROUP, g.apex.transform(zs), np.nan)
    c.loc[wr, "route_retention_z"] = np.where(size >= MIN_GROUP, g.retention.transform(zs), np.nan)
    return c


def player_metrics(c):
    """One row per player with >= MIN_CUTS cuts."""
    n = c.groupby("nfl_id").size()
    keep = n[n >= MIN_CUTS].index
    c = c[c.nfl_id.isin(keep)]
    out = c.groupby("nfl_id").agg(
        g_cuts=("apex", "size"), g_cut_speed_z=("pos_speed_z", "mean"),
        g_force_n=("f_peak", "median"), g_turn_n=("f_turn", "median"), g_brake_n=("f_brake", "median"),
        pos_group=("pos_group", "first"), mass_kg=("mass_kg", "first"))
    wr = c[c.route_speed_z.notna()].groupby("nfl_id")
    rs = wr.agg(g_route_speed_z=("route_speed_z", "mean"), g_route_retention_z=("route_retention_z", "mean"),
                g_route_cuts=("route_speed_z", "size"))
    out = out.join(rs[rs.g_route_cuts >= MIN_CUTS])
    out["g_force_per_kg"] = out.g_force_n / out.mass_kg
    for src, dst in [("g_force_n", "g_force_oe_z"), ("g_turn_n", "g_turn_oe_z"), ("g_brake_n", "g_brake_oe_z")]:
        r, _ = cf.residual_on_mass(out.reset_index(), src, "mass_kg")
        out[dst] = zs(pd.Series(r.to_numpy(), index=out.index))
    return out.reset_index()


def game_reliability(c):
    """Odd vs even plays (in time order) within player, Spearman-Brown corrected."""
    c = c.sort_values(["nfl_id", "game_id", "play_id"]).copy()
    c["play_key"] = c.game_id * 100_000 + c.play_id
    c["half"] = c.groupby("nfl_id").play_key.transform(lambda s: pd.factorize(s)[0] % 2)
    rel = {}
    halves = [player_metrics(c[c.half == h]).set_index("nfl_id") for h in (0, 1)]
    for col in ["g_route_speed_z", "g_route_retention_z", "g_cut_speed_z", "g_force_oe_z", "g_turn_oe_z",
                "g_brake_oe_z", "g_force_n", "g_force_per_kg"]:
        j = halves[0][[col]].join(halves[1][[col]], rsuffix="_2").dropna()
        r = j.iloc[:, 0].corr(j.iloc[:, 1])
        rel[col] = dict(r_half=r, reliability=sb(r), n=len(j))
    return rel


# ---------------------------------------------------------------- Combine reliability
def combine_reliability():
    ev = pd.read_parquet(os.path.join(cm.OUT_DIR, "cod_events.parquet"))
    base = cm.player_base()
    rel = {}
    s1, wr, per_drill = cf.idea1_scores(ev)
    pdd = per_drill.sort_values(["nfl_id", "drill"]).copy()
    pdd["half"] = pdd.groupby("nfl_id").cumcount() % 2
    for col, name in [("apex_z", "cod_speed_z"), ("retention_z", "cod_retention_z")]:
        h = pdd.groupby(["nfl_id", "half"])[col].mean().unstack().dropna()
        rel[name] = sb(h[0].corr(h[1]))
    ag = ev[ev.drill.isin(cf.AGILITY_DRILLS)]
    per = ag.groupby(["nfl_id", "drill"]).agg(f_peak=("f_peak", "median"), f_brake=("f_brake", "median"),
                                             f_turn=("f_turn", "median")).reset_index()
    per = per.merge(base[["nfl_id", "combine_weight"]], on="nfl_id")
    per["mass_kg"] = per.combine_weight * cm.LB_TO_KG
    for d, g in per.groupby("drill"):
        for col in ["f_peak", "f_brake", "f_turn"]:
            r, _ = cf.residual_on_mass(g, col, "mass_kg")
            per.loc[g.index, col + "_oe"] = r
        per.loc[g.index, "f_abs"] = g.f_peak
        per.loc[g.index, "f_kg"] = g.f_peak / g.mass_kg
    for col, name in [("f_peak_oe", "force_oe_z"), ("f_turn_oe", "turn_oe_z"), ("f_brake_oe", "brake_oe_z"),
                      ("f_abs", "force_abs_z"), ("f_kg", "force_per_kg_z")]:
        w = per.pivot(index="nfl_id", columns="drill", values=col).dropna()
        rel[name] = sb(w.iloc[:, 0].corr(w.iloc[:, 1]))
    rel["tc_cut_speed_z"] = np.nan                     # too few repeated three-cone reps to estimate
    return rel


# ---------------------------------------------------------------- transfer
def fisher_ci(r, n, k=0):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3 - k)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


def partial_on_group(d, a, b):
    X = pd.get_dummies(d.pos_group, drop_first=True, dtype=float)
    X.insert(0, "const", 1.0)
    ra = d[a] - X @ np.linalg.lstsq(X, d[a], rcond=None)[0]
    rb = d[b] - X @ np.linalg.lstsq(X, d[b], rcond=None)[0]
    return ra, rb, X.shape[1] - 1


def transfer(feat, rel_c, rel_g):
    rows = []
    for (cmb, gm, grp, partial), primary in [(p, True) for p in PRIMARY] + [(s, False) for s in SECONDARY]:
        d = feat[feat.pos_group != "QB"] if grp == "ALL" else feat[feat.pos_group == grp]
        d = d.dropna(subset=[cmb, gm])
        a, b, k = (partial_on_group(d, cmb, gm) if partial else (d[cmb], d[gm], 0))
        r, p = stats.pearsonr(a, b)
        lo, hi = fisher_ci(r, len(d), k)
        rc, rg = rel_c.get(cmb, np.nan), rel_g[gm]["reliability"]
        dis = r / np.sqrt(rc * rg) if rc > 0 and rg > 0 else np.nan
        rows.append(dict(combine=cmb, game=gm, group=grp, partial_on_position=partial, n=len(d), r=r, ci_lo=lo,
                         ci_hi=hi, p=p, rel_combine=rc, rel_game=rg, r_disattenuated=dis, primary=primary))
        if partial:                                    # also the plain correlation, for reference
            r0, _ = stats.pearsonr(d[cmb], d[gm])
            rows[-1]["r_without_partial"] = r0
    T = pd.DataFrame(rows)
    T["p_holm"] = np.nan
    for mask in [T.primary, ~T.primary]:
        T.loc[mask, "p_holm"] = multipletests(T.loc[mask, "p"], method="holm")[1]
    return T


# ---------------------------------------------------------------- outcomes
def season_yprr(pp):
    games = pd.read_csv(cm.path("games.csv"))[["game_id", "season"]]
    r = pp[pp.route_ran.notna()].merge(games, on="game_id")
    s = r.groupby(["nfl_id", "season"]).agg(routes=("route_ran", "size"), rec_yards=("rec_yards", "sum")).reset_index()
    snaps = pp.merge(games, on="game_id").groupby(["nfl_id", "season"]).size().rename("snaps").reset_index()
    return s, snaps


def outcomes(cuts, base, feat_c, pp):
    """Rookie-season game metric -> later-season outcome, draft-controlled."""
    s, snaps = season_yprr(pp)
    dy = base.set_index("nfl_id").draft_year
    cuts = cuts.copy()
    cuts["rookie"] = cuts.season == cuts.nfl_id.map(dy)
    rook = player_metrics(cuts[cuts.rookie])
    later_s = s[s.season > s.nfl_id.map(dy)].groupby("nfl_id")[["routes", "rec_yards"]].sum()
    later_s["yprr_later"] = later_s.rec_yards / later_s.routes
    later_n = snaps[snaps.season > snaps.nfl_id.map(dy)].groupby("nfl_id").agg(snaps_later=("snaps", "sum"))
    rook_s = s[s.season == s.nfl_id.map(dy)].set_index("nfl_id")
    d = (base[["nfl_id", "log_pick", "udfa", "draft_year"]].merge(rook, on="nfl_id")
         .merge(later_s[["routes", "yprr_later"]], left_on="nfl_id", right_index=True, how="left")
         .merge(later_n, left_on="nfl_id", right_index=True, how="left")
         .merge(rook_s[["routes"]].rename(columns={"routes": "rookie_routes"}), left_on="nfl_id", right_index=True, how="left")
         .merge(feat_c[["nfl_id", "cod_speed_z", "force_oe_z", "turn_oe_z"]], on="nfl_id", how="left"))
    d = d[d.draft_year < cm.LAST_SEASON].copy()
    d["later_seasons"] = cm.LAST_SEASON - d.draft_year      # exposure; a player cut after year 1 counts as 0 snaps
    d["snaps_later"] = d.snaps_later.fillna(0)
    rows = []
    wr = d[(d.pos_group == "WR") & (d.routes >= 100) & (d.rookie_routes >= 100)]
    for m in ["g_route_speed_z", "g_route_retention_z", "cod_speed_z"]:
        x = wr.dropna(subset=[m])
        if len(x) >= 15:
            f = smf.ols(f"yprr_later ~ {m} + {CONTROLS}", x).fit(cov_type="HC1")
            lo, hi = f.conf_int().loc[m]
            rows.append(dict(metric=m, outcome="YPRR, later seasons (WR)", n=len(x), coef=f.params[m],
                             ci_lo=lo, ci_hi=hi, p=f.pvalues[m], model="OLS HC1"))
    allp = d[d.pos_group != "QB"].copy()
    allp["log_later_seasons"] = np.log(allp.later_seasons)
    for m in ["g_force_oe_z", "g_turn_oe_z", "g_cut_speed_z", "force_oe_z", "turn_oe_z"]:
        x = allp.dropna(subset=[m])
        if len(x) >= 25:
            f = smf.glm(f"snaps_later ~ {m} + C(pos_group) + {CONTROLS}", x, family=sm.families.Poisson(),
                        offset=x.log_later_seasons).fit(cov_type="HC1", maxiter=200)
            lo, hi = f.conf_int().loc[m]
            rows.append(dict(metric=m, outcome="Snaps per later season (all)", n=len(x), coef=f.params[m],
                             ci_lo=lo, ci_hi=hi, p=f.pvalues[m], model="Poisson QMLE HC1",
                             pct_per_sd=(np.exp(f.params[m]) - 1) * 100))
    O = pd.DataFrame(rows)
    O["p_holm"] = multipletests(O.p, method="holm")[1] if len(O) else []
    return O


# ---------------------------------------------------------------- main
def main():
    base = cm.player_base()
    base = base[base.combine_weight.notna()]
    pp = cm.player_play_regular_season(usecols=["game_id", "play_id", "nfl_id", "route_ran", "rec_yards"])
    cuts = add_z(load_cuts(base, pp))
    gm = player_metrics(cuts)
    rel_g = game_reliability(cuts)
    rel_c = combine_reliability()
    feat_c = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_cod.parquet"))
    feat = feat_c.drop(columns=["pos_group"]).merge(gm, on="nfl_id", how="inner")
    T = transfer(feat, rel_c, rel_g)
    O = outcomes(cuts, base, feat_c, pp)
    T.to_csv(os.path.join(cm.RESULTS_DIR, "transfer_correlations.csv"), index=False)
    O.to_csv(os.path.join(cm.RESULTS_DIR, "transfer_outcomes.csv"), index=False)

    L = ["Combine -> game transfer of change-of-direction metrics",
         "Plan: results/transfer_preregistration.md", ""]
    L.append(f"In-game cuts: {len(cuts):,} from {cuts.nfl_id.nunique()} players; "
             f"{len(gm)} players with >= {MIN_CUTS} cuts (median {gm.g_cuts.median():.0f})")
    L.append(cuts.groupby("pos_group").agg(players=("nfl_id", "nunique"), cuts=("apex", "size"),
                                           median_apex_ms=("apex", "median"), median_turn_deg=("turn_deg", "median"),
                                           median_turn_force_n=("f_turn", "median")).round(1).to_string())
    L.append("")
    L.append("Reliability (split-half, Spearman-Brown)")
    for k, v in rel_g.items():
        L.append(f"  game  {k:20s} {v['reliability']:.2f} (N = {v['n']})")
    for k, v in rel_c.items():
        L.append(f"  combine {k:18s} {v:.2f}" if pd.notna(v) else f"  combine {k:18s} n/a")
    L.append("")
    L.append("TRANSFER (Pearson r, 95% CI; primary Holm across three)")
    for t in T.itertuples():
        tag = "PRIMARY " if t.primary else "        "
        extra = f"; plain r = {t.r_without_partial:+.2f}" if t.partial_on_position else ""
        dis = f"{t.r_disattenuated:+.2f}" if pd.notna(t.r_disattenuated) else "n/a"
        L.append(f"  {tag}{t.combine:16s} vs {t.game:20s} {t.group:3s} N={t.n:3d}  r = {t.r:+.2f} "
                 f"[{t.ci_lo:+.2f}, {t.ci_hi:+.2f}], p = {t.p:.3f}, Holm {t.p_holm:.3f}; "
                 f"disattenuated {dis}{' (partial on position)' if t.partial_on_position else ''}{extra}")
    L.append("")
    L.append("OUTCOMES (rookie-season metric -> later seasons; draft-controlled; exploratory)")
    for o in O.itertuples():
        pct = f", {o.pct_per_sd:+.1f}% per SD" if pd.notna(getattr(o, "pct_per_sd", np.nan)) else ""
        L.append(f"  {o.metric:20s} -> {o.outcome:30s} N={o.n:3d}  coef {o.coef:+.3f} [{o.ci_lo:+.3f}, {o.ci_hi:+.3f}]"
                 f"{pct}, p = {o.p:.3f}, Holm {o.p_holm:.2f}")
    txt = "\n".join(L)
    with open(os.path.join(cm.RESULTS_DIR, "transfer_summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
