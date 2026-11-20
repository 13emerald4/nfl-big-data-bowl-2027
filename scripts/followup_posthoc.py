"""
followup_posthoc.py
Checks added after reading followup_tests.py output (NOT pre-registered):
  T3: pass plays only (pressure cannot occur on a run, and rushers may get off
      slower when reading run), and dropback-length bins.
  T4: the OL result runs against the hypothesis; add dropback duration and
      split by dropback length to see whether long dropbacks drive it.
  T6: descriptive rank correlation on the 156 matched plays.
Writes results/followup_posthoc.txt.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as cm
import followup_tests as T


def main():
    games = pd.read_csv(cm.path("games.csv"))
    pp = cm.player_play_regular_season(usecols=T.PP_COLS)
    fu = pd.read_parquet(os.path.join(cm.OUT_DIR, "fu_player_play.parquet"))
    drop = pp.groupby(["game_id", "play_id"]).dropback_duration.max().rename("dropback").reset_index()
    passp = pp.groupby(["game_id", "play_id"]).pass_result.first().rename("play_pass_result").reset_index()

    # T3
    r = pp[pp.player_get_off.notna() & (pp.play_nullified_by_penalty != "Y")].copy()
    r = r.merge(drop, on=["game_id", "play_id"], how="left").merge(passp, on=["game_id", "play_id"], how="left")
    r = T.opp_season(r, games)
    r["getoff_fast"] = -r.player_get_off
    r["pressure"] = (r.time_to_pressure.notna() | (r.sack.fillna(0) > 0)).astype(float)
    r["blitz"] = r.blitzing.fillna(False).astype(float)
    r = r.dropna(subset=["down", "yards_to_go"])
    print("T3 rows by pass_result present:", r.play_pass_result.notna().value_counts().to_dict())
    rp = T.min_units(r[r.play_pass_result.notna()], "nfl_id", 30)
    rhs = "yards_to_go + blitz + C(nfl_id) + C(lined_up_position) + C(down) + C(opp_season)"
    T.est("T3 post-hoc", "get-off -> pressure, pass plays only", rp, "pressure", "getoff_fast", rhs, "nfl_id", "nfl_id")
    rr = T.min_units(r[r.play_pass_result.isna()], "nfl_id", 2)
    print(f"  non-pass rows: {len(rr):,}, pressure rate {rr.pressure.mean():.4f}, "
          f"mean get-off {rr.player_get_off.mean():.3f} s vs pass {rp.player_get_off.mean():.3f} s")

    # T4
    o = pp[pp.lined_up_position.isin(["G", "T", "C"]) & pp.peak_pressure_probability_allowed.notna()
           & (pp.play_nullified_by_penalty != "Y")]
    o = o.merge(fu[["game_id", "play_id", "nfl_id", "ol_getoff", "dx10"]], on=["game_id", "play_id", "nfl_id"])
    o = o.drop(columns=["dropback_duration"]).merge(drop, on=["game_id", "play_id"], how="left")
    o = T.opp_season(o.dropna(subset=["ol_getoff", "down", "yards_to_go", "dropback"]).copy(), games)
    o["ol_fast"] = -o.ol_getoff
    o = T.min_units(o, "nfl_id", 30)
    rhs = "yards_to_go + C(nfl_id) + C(lined_up_position) + C(down) + C(opp_season)"
    T.est("T4 post-hoc", "OL get-off -> peak pressure prob, + dropback", o, "peak_pressure_probability_allowed",
          "ol_fast", "dropback + " + rhs, "nfl_id", "nfl_id")
    q = o.dropback.quantile([1 / 3, 2 / 3]).to_numpy()
    for lab, s in [("dropback short third", o[o.dropback <= q[0]]), ("dropback long third", o[o.dropback > q[1]])]:
        T.est("T4 post-hoc", f"OL get-off -> peak pressure prob, {lab}", T.min_units(s, "nfl_id", 10),
              "peak_pressure_probability_allowed", "ol_fast", rhs, "nfl_id", "nfl_id")
    print(f"  corr(OL get-off speed, dropback) = {np.corrcoef(o.ol_fast, o.dropback)[0, 1]:+.3f}")

    # T6 descriptive
    h = pd.read_parquet(os.path.join(cm.OUT_DIR, "fu_head_to_head.parquet"))
    rho, p = stats.spearmanr(h.gap, h.separation)
    rho_w, p_w = stats.spearmanr(h.gap - h.groupby("nfl_id").gap.transform("mean"),
                                 h.separation - h.groupby("nfl_id").separation.transform("mean"))
    T.ROWS.append(dict(test="T6 descriptive", estimate=f"Spearman gap vs separation, raw rho={rho:+.3f} (p={p:.3f}); "
                       f"within receiver rho={rho_w:+.3f} (p={p_w:.3f})", n=len(h), units=h.nfl_id.nunique(), p=np.nan))

    lines = ["Post-hoc checks of the follow-up tests (NOT pre-registered)", ""]
    for x in T.ROWS:
        if pd.isna(x.get("p")):
            lines.append(f"{x['test']}  {x['estimate']}  n={x['n']}, receivers={x['units']}")
        else:
            lines.append(f"{x['test']}  {x['estimate']}: {x['coef']:+.4f} [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}], "
                         f"p = {x['p']:.4f}; per SD {x['per_sd']:+.4f}; n = {x['n']:,}, units = {x['units']}")
    with open(os.path.join(cm.RESULTS_DIR, "followup_posthoc.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))



def t2_separation():
    """T2 post-hoc: is catch speed only a proxy for being open? Control separation at the throw."""
    games = pd.read_csv(cm.path("games.csv"))
    base = cm.player_base()
    base["pos_group"] = base.nfl_position.map(T.cf.POS_GROUP)
    pp = cm.player_play_regular_season(usecols=T.PP_COLS + ["separation_at_pass_forward"])
    fu = pd.read_parquet(os.path.join(cm.OUT_DIR, "fu_player_play.parquet"))
    c = pp[pp.yards_after_catch.notna() & (pp.play_nullified_by_penalty != "Y")]
    c = c.merge(fu[["game_id", "play_id", "nfl_id", "s_caught"]], on=["game_id", "play_id", "nfl_id"])
    c = c.merge(base[["nfl_id", "pos_group"]], on="nfl_id")
    c = c[c.pos_group.isin(["WR", "TE", "RB"]) & c.s_caught.notna()].copy()
    c["s_caught"] = c.s_caught * T.YD
    c = T.opp_season(c, games)
    c["route"], c["cov"], c["air"] = c.route_ran.fillna("NONE"), c.team_coverage_type.fillna("UNK"), c.pass_length.fillna(0)
    c["to_goal"] = np.where(c.yardline_side == c.possession_team, 100 - c.yardline_number, c.yardline_number)
    c = T.min_units(c.dropna(subset=["to_goal", "down", "yards_to_go", "separation_at_pass_forward"]), "nfl_id", 2)
    rhs = "air + yards_to_go + to_goal + C(nfl_id) + C(route) + C(cov) + C(down) + C(opp_season)"
    T.est("T2 post-hoc", "catch speed -> YAC, same sample, no separation", c, "yards_after_catch", "s_caught", rhs,
          "nfl_id", "nfl_id")
    T.est("T2 post-hoc", "catch speed -> YAC, + separation at the throw", c, "yards_after_catch", "s_caught",
          "separation_at_pass_forward + " + rhs, "nfl_id", "nfl_id")
    with open(os.path.join(cm.RESULTS_DIR, "followup_posthoc.txt"), "a", encoding="utf-8") as fh:
        for x in T.ROWS[-2:]:
            fh.write(f"{x['test']}  {x['estimate']}: {x['coef']:+.4f} [{x['ci_lo']:+.4f}, {x['ci_hi']:+.4f}], "
                     f"p = {x['p']:.4f}; n = {x['n']:,}, units = {x['units']}\n")


if __name__ == "__main__":
    main()
    t2_separation()
