"""
extract_features.py
Builds one feature table per position group from 10 Hz Combine tracking and
joins regular-season outcomes (volume and per-snap efficiency).

Writes data/features_{wr,dl,db,ol}.parquet (git-ignored).

Feature definitions
  WR  s_cut      median speed (km/h) over all frames of the four cutting drills
                 (curl, comeback, speed out, slant), as in the original report.
      s_cut_z    drill-standardised cut speed: the player's median speed in each
                 cutting drill, z-scored against every WR who ran that drill,
                 averaged over the drills he ran. Removes drill-mix bias (players
                 did not all run the same drills).
  DL  gamma_all  90th-percentile |a| (km/h/s) over every Combine drill (original).
      gamma_pr   90th-percentile |a| within each PASS_RUSH_DRILL rep, averaged
                 over reps. Every DL prospect ran this drill.
      *_oe       residual of the above on Combine weight ("Gamma over expected").
  DB  db_p90     90th-percentile speed in the 45-degree back-pedal transition drill.
  OL  ol_oe      90th-percentile |a| in mirror/wave drills, residualised on weight.
Note: the `a` column is a magnitude (never negative), so 90th-percentile |a|
counts hard braking as well as acceleration.
"""
import os

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.formula.api as smf

import common as cm


def wr_features(ct, base, pp):
    ids = base.loc[base.nfl_position.isin(cm.WR_POS), "nfl_id"].tolist()
    cuts = (ct.filter(pl.col("nfl_id").is_in(ids) & pl.col("drill_name").is_in(list(cm.WR_CUT_DRILLS)))
              .with_columns(pl.col("drill_name").replace_strict(cm.WR_CUT_DRILLS).alias("drill"),
                            (pl.col("s") * cm.YDS_TO_KMH).alias("kmh")))
    pooled = cuts.group_by("nfl_id").agg(pl.col("kmh").median().alias("s_cut"),
                                         pl.col("kmh").max().alias("s_cut_max")).to_pandas()
    per = cuts.group_by(["nfl_id", "drill"]).agg(pl.col("kmh").median().alias("med")).to_pandas()
    per["z"] = per.groupby("drill")["med"].transform(lambda x: (x - x.mean()) / x.std())
    z = per.groupby("nfl_id").agg(s_cut_z=("z", "mean"), n_cut_drills=("drill", "nunique")).reset_index()
    peak = (ct.filter(pl.col("nfl_id").is_in(ids)).group_by("nfl_id")
              .agg((pl.col("s").max() * cm.YDS_TO_KMH).alias("peak_any_drill")).to_pandas())

    routes = pp[pp.nfl_id.isin(ids) & pp.route_ran.notna()]
    eff = routes.groupby("nfl_id").agg(
        routes=("route_ran", "size"), targets=("target", "sum"),
        rec_yards=("rec_yards", lambda x: x.fillna(0).sum()),
        separation=("separation_at_pass_forward", "mean"),
        n_separation=("separation_at_pass_forward", "count")).reset_index()
    eff["yprr"] = eff.rec_yards / eff.routes
    eff["targets_per_route"] = eff.targets / eff.routes

    df = (base[base.nfl_position.isin(cm.WR_POS)].merge(pooled, on="nfl_id").merge(z, on="nfl_id")
          .merge(peak, on="nfl_id").merge(eff, on="nfl_id", how="left"))
    for c in ["routes", "targets", "rec_yards"]:
        df[c] = df[c].fillna(0)
    return df


def residualise(df, col, out):
    ok = df[col].notna() & df["combine_weight"].notna()
    m = smf.ols(f"{col} ~ combine_weight", df[ok]).fit()
    df.loc[ok, out] = m.resid
    return m


def dl_features(ct, base, pp):
    ids = base.loc[base.nfl_position.isin(cm.DL_POS), "nfl_id"].tolist()
    dl = ct.filter(pl.col("nfl_id").is_in(ids))
    g_all = dl.group_by("nfl_id").agg((pl.col("a") * cm.YDS_TO_KMH).quantile(0.9).alias("gamma_all"),
                                      pl.col("drill_name").n_unique().alias("n_drills")).to_pandas()
    g_pr = (dl.filter(pl.col("drill_name") == cm.DL_DRILL)
              .group_by(["nfl_id", "event_id"]).agg((pl.col("a") * cm.YDS_TO_KMH).quantile(0.9).alias("g"))
              .group_by("nfl_id").agg(pl.col("g").mean().alias("gamma_pr"), pl.len().alias("pr_reps")).to_pandas())
    df = (base[base.nfl_position.isin(cm.DL_POS)].merge(g_all, on="nfl_id").merge(g_pr, on="nfl_id", how="left")
          .dropna(subset=["combine_weight"]))
    residualise(df, "gamma_all", "gamma_all_oe")
    residualise(df, "gamma_pr", "gamma_pr_oe")

    rush = pp[pp.nfl_id.isin(ids) & pp.player_get_off.notna()]
    eff = rush.groupby("nfl_id").agg(rush_snaps=("player_get_off", "size"), get_off=("player_get_off", "mean"),
                                     pressures=("time_to_pressure", "count")).reset_index()
    eff["pressure_rate"] = eff.pressures / eff.rush_snaps
    return df.merge(eff, on="nfl_id", how="left")


def db_features(ct, base):
    ids = base.loc[base.nfl_position.isin(cm.DB_POS), "nfl_id"].tolist()
    f = (ct.filter(pl.col("nfl_id").is_in(ids) & (pl.col("drill_name") == cm.DB_DRILL))
           .group_by("nfl_id").agg((pl.col("s") * cm.YDS_TO_KMH).quantile(0.9).alias("db_p90")).to_pandas())
    return base[base.nfl_position.isin(cm.DB_POS)].merge(f, on="nfl_id")


def ol_features(ct, base):
    ids = base.loc[base.nfl_position.isin(cm.OL_POS), "nfl_id"].tolist()
    f = (ct.filter(pl.col("nfl_id").is_in(ids) & pl.col("drill_name").is_in(cm.OL_DRILLS))
           .group_by("nfl_id").agg((pl.col("a") * cm.YDS_TO_KMH).quantile(0.9).alias("ol_a")).to_pandas())
    df = base[base.nfl_position.isin(cm.OL_POS)].merge(f, on="nfl_id").dropna(subset=["combine_weight"])
    residualise(df, "ol_a", "ol_oe")
    return df


def extract_all():
    os.makedirs(cm.OUT_DIR, exist_ok=True)
    print(f"Reading competition data from {os.path.abspath(cm.DATA_DIR)}")
    ct = cm.combine_tracking()
    base = cm.player_base()
    pp = cm.player_play_regular_season()
    out = {"wr": wr_features(ct, base, pp), "dl": dl_features(ct, base, pp),
           "db": db_features(ct, base), "ol": ol_features(ct, base)}
    for k, df in out.items():
        df.to_parquet(os.path.join(cm.OUT_DIR, f"features_{k}.parquet"), index=False)
        print(f"{k.upper()} cohort: {len(df)} prospects")


if __name__ == "__main__":
    extract_all()
