"""Distribution of receivers' speed at the last cut before the throw (Stage 3 sample).

Reads data/play_level.parquet (built by play_level_fe.py) and writes
last_cut_speed_distribution.png and results/last_cut_speed.txt. Descriptive only:
the open-vs-tight split mixes routes and receivers; the within-receiver model in
play_level_fe.py is the estimate.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

import common as cm
import make_field_plot as mfp
from make_field_plot import plt, AQUA, BLUE, ORANGE, INK, INK2

KMH = 3.6
GRID_KMH = np.linspace(0, 35, 400)


def ecdf(v):
    v = np.sort(np.asarray(v))
    return v, np.arange(1, len(v) + 1) / len(v)


def load():
    d = pd.read_parquet(os.path.join(cm.OUT_DIR, "play_level.parquet"))
    wr = d[d.pos_group == "WR"].copy()
    wr["kmh"] = wr.apex * KMH
    wr["sharp"] = wr.turn_deg >= 90
    wr["open"] = wr.separation > 2
    return wr


def summary(wr):
    k = wr.kmh
    within = k - wr.groupby(["route_ran", "angle_bin"]).kmh.transform("mean")
    own = k - wr.groupby("nfl_id").kmh.transform("mean")
    op, ti = wr[wr.open].kmh, wr[~wr.open].kmh
    ks = stats.ks_2samp(op, ti)
    q = lambda s, p: float(s.quantile(p))
    lines = [
        "Speed at the slowest point of the last cut before the throw, WRs, Stage 3 sample",
        f"  N = {len(wr):,} routes, {wr.nfl_id.nunique()} receivers",
        f"  median {k.median():.1f} km/h, IQR [{q(k, .25):.1f}, {q(k, .75):.1f}], "
        f"10th-90th pct [{q(k, .1):.1f}, {q(k, .9):.1f}], mean {k.mean():.1f}, SD {k.std():.1f}",
        f"  share below 10 km/h {(k < 10).mean():.0%}, above 20 km/h {(k > 20).mean():.0%}",
        f"  turn 45-90 deg: n = {(~wr.sharp).sum():,}, median {wr[~wr.sharp].kmh.median():.1f} km/h; "
        f"turn 90-180 deg: n = {wr.sharp.sum():,}, median {wr[wr.sharp].kmh.median():.1f} km/h",
        f"  SD within route x angle bin (the scale of 1 SD of apex_z): {within.std():.1f} km/h",
        f"  SD within receiver: {own.std():.1f} km/h (total {k.std():.1f})",
        f"  open (> 2 yd, n = {len(op):,}) median {op.median():.1f} km/h vs tight (<= 2 yd, n = {len(ti):,}) "
        f"{ti.median():.1f} km/h; KS D = {ks.statistic:.3f}, p = {ks.pvalue:.1e} (descriptive, not adjusted)",
    ]
    return lines, dict(within_sd=within.std(), med=k.median())


def figure(wr, s):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.25})
    a = axes[0]
    for mask, color, lab in [(~wr.sharp, AQUA, "Turn 45–90°"), (wr.sharp, INK2, "Turn 90–180°")]:
        v = wr[mask].kmh.to_numpy()
        kde = stats.gaussian_kde(v)
        dens = kde(GRID_KMH) + kde(-GRID_KMH)  # reflect at 0: speed cannot be negative
        a.plot(GRID_KMH, dens, color=color, lw=2, label=f"{lab} ({len(v):,} routes)")
    a.set_xlim(0, 30)
    a.set_ylim(bottom=0)
    a.set_title("Density (PDF), by sharpness of the cut", loc="left", fontsize=10.5, color=INK)
    a.set_xlabel("Speed at the slowest point of the last cut (km/h)")
    a.set_ylabel("Density")
    a.legend(fontsize=8.5, loc="upper right")
    a = axes[1]
    for mask, color, lab in [(wr.open, BLUE, "Open at the throw (> 2 yd)"), (~wr.open, ORANGE, "Tight at the throw (≤ 2 yd)")]:
        v = wr[mask].kmh
        x, p = ecdf(v)
        a.plot(x, p, color=color, lw=2, label=f"{lab}, median {v.median():.1f}")
    a.set_xlim(0, 30)
    a.set_title("Cumulative distribution (CDF), by outcome", loc="left", fontsize=10.5, color=INK)
    a.set_xlabel("Speed at the slowest point of the last cut (km/h)")
    a.set_ylabel("Share of routes at or below")
    a.legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"Receivers slow to a median {s['med']:.1f} km/h at the last cut; "
                 f"one SD within route and angle is about {s['within_sd']:.1f} km/h",
                 x=0.07, ha="left", y=1.05, fontsize=11.5, color=INK)
    fig.text(0.07, 0.975, f"{wr.nfl_id.nunique()} receivers, {len(wr):,} routes in the Stage 3 sample. "
             "The outcome split is descriptive: it mixes routes and receivers", fontsize=8.5, color=INK2)
    mfp.save(fig, "last_cut_speed_distribution.png")


def main():
    wr = load()
    lines, s = summary(wr)
    figure(wr, s)
    txt = "\n".join(lines)
    with open(os.path.join(cm.RESULTS_DIR, "last_cut_speed.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
