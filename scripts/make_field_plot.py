"""
make_field_plot.py
Static publication figures (written to the repo root and copied to figures/):
  1. draft_controls.png              test statistic per model, before vs after draft controls
  2. wr_speed_pdf_cdf.png            cutting-drill speed density, four case-study receivers
  3. route_kinematics_field_plot.png comeback route: paths and speed through the break
  4. mass_vs_gamma_scatter.png       mass vs 90th-percentile acceleration (DL)
  5. play_level_cut_speed.png        within-player effect of speed through the cut on separation
Run extract_features.py and run_regressions.py first (and play_level_fe.py for figure 5).
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy.stats as stats
import statsmodels.formula.api as smf

import common as cm

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
})
CASE = {56042: ("Puka Nacua (pick 177)", BLUE), 55884: ("Jaxon Smith-Njigba (pick 20)", ORANGE),
        57144: ("Brian Thomas Jr. (pick 23)", AQUA), 55885: ("Quentin Johnston (pick 21)", YELLOW)}


def save(fig, name):
    out = os.path.join(cm.REPO_DIR, name)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    os.makedirs(os.path.join(cm.REPO_DIR, "figures"), exist_ok=True)
    shutil.copy(out, os.path.join(cm.REPO_DIR, "figures", name))
    plt.close(fig)
    print(f"Saved: {out}")


def draft_controls():
    res = pd.read_csv(os.path.join(cm.RESULTS_DIR, "model_results.csv"))
    res["z"] = res["coef"] / res["se"]
    res["row"] = res["metric"] + "  |  " + res["outcome"]
    rows = list(dict.fromkeys(res["row"]))[::-1]
    y = {r: i for i, r in enumerate(rows)}
    fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(rows) + 1.4))
    ax.axvspan(-1.96, 1.96, color="#efeeea", zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.8)
    for r in rows:
        a = res[(res.row == r) & (res.spec == "raw")].z.iloc[0]
        b = res[(res.row == r) & (res.spec == "controlled")].z.iloc[0]
        ax.plot([a, b], [y[r], y[r]], color=GRID, lw=2, zorder=1)
    for spec, color, lab in [("raw", BLUE, "Combine metric alone (+ 40 time or weight)"),
                             ("controlled", ORANGE, "+ draft pick, UDFA flag, draft class (and position)")]:
        d = res[res.spec == spec]
        ax.scatter(d.z, [y[r] for r in d.row], s=48, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3, label=lab)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5, color=INK)
    ax.set_xlabel("Test statistic (coefficient / standard error); shaded band = not significant at 5%")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, fontsize=9)
    ax.set_title("Every Combine metric's link to NFL outcomes, before and after draft controls", loc="left",
                 fontsize=11.5, color=INK, pad=28)
    ax.grid(axis="y", visible=False)
    save(fig, "draft_controls.png")


def speed_pdf_cdf(ct):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    grid = np.linspace(0, 32, 300)
    for nid, (label, color) in CASE.items():
        s = (ct.filter((pl.col("nfl_id") == nid) & pl.col("drill_name").is_in(list(cm.WR_CUT_DRILLS)))["s"].to_numpy()
             * cm.YDS_TO_KMH)
        a1.plot(grid, stats.gaussian_kde(s)(grid), color=color, lw=2, label=label)
        xs = np.sort(s)
        a2.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=color, lw=2, label=label)
    a1.set_title("Speed density, cutting drills only", loc="left", fontsize=10.5)
    a1.set_xlabel("Speed (km/h)"); a1.set_ylabel("Density")
    a2.set_title("Cumulative share of frames", loc="left", fontsize=10.5)
    a2.set_xlabel("Speed (km/h)"); a2.set_ylabel("Share of frames at or below speed")
    a2.legend(loc="lower right", fontsize=8.5)
    fig.text(0.01, -0.02, "Cutting drills: curl, comeback, speed out, slant. Smith-Njigba's tracked cutting work is slant and "
                          "speed out only; the others ran all four.",
             fontsize=8.5, color=INK2)
    save(fig, "wr_speed_pdf_cdf.png")


def route_plot(ct, wr):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3), gridspec_kw={"width_ratios": [1.25, 1]})
    a1.set_facecolor("#e9f1ea")
    for x in range(20, 46, 5):
        a1.axvline(x, color="white", lw=1.6 if x % 10 == 0 else 0.8, zorder=0)
    for nid, color in [(56042, BLUE), (55885, YELLOW)]:
        name = wr.loc[wr.nfl_id == nid, "display_name"].iloc[0]
        g = (ct.filter((pl.col("nfl_id") == nid) & (pl.col("drill_name") == "COMEBACK_ROUTE_RIGHT")).sort("time").to_pandas())
        kmh = g.s.to_numpy() * cm.YDS_TO_KMH
        brk = int(np.argmax(g.x.to_numpy()))
        lo = brk - 5 + int(np.argmin(kmh[brk - 5:brk + 6]))
        a1.plot(g.x, g.y, color=color, lw=2, label=name, zorder=2)
        a1.scatter(g.x.iloc[lo], g.y.iloc[lo], s=60, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        t = (np.arange(len(kmh)) - lo) / 10
        a2.plot(t, kmh, color=color, lw=2, label=name)
        dsdt = np.diff(g.s.to_numpy()) / 0.1
        a2.text(1.75, 27 if nid == 56042 else 4,
                f"{name.split()[-1]}: slowest {kmh[lo]:.1f} km/h\npeak braking {dsdt.min():.2f} yd/s²\n"
                f"{kmh[min(lo + 10, len(kmh) - 1)]:.1f} km/h one second later",
                fontsize=8, color=INK, va="top" if nid == 56042 else "bottom",
                bbox=dict(boxstyle="square,pad=0.3", fc=SURFACE, ec=color, lw=1.2))
    a1.set_aspect("equal"); a1.grid(False)
    a1.set_xlabel("Field x (yards)"); a1.set_ylabel("Field y (yards)")
    a1.set_title("Comeback route paths (dot = slowest frame)", loc="left", fontsize=10.5)
    a1.legend(loc="upper left", fontsize=8.5)
    a2.axvline(0, color=INK2, lw=0.8, ls="--")
    a2.set_xlabel("Seconds from slowest frame of the break"); a2.set_ylabel("Speed (km/h)")
    a2.set_title("Speed into and out of the break (one rep each)", loc="left", fontsize=10.5)
    a2.set_xlim(right=4.4)
    save(fig, "route_kinematics_field_plot.png")


def mass_gamma(dl):
    fig, ax = plt.subplots(figsize=(8.5, 5))
    m = smf.ols("gamma_all ~ combine_weight", dl).fit()
    for flag, color, lab in [(0, BLUE, "Fewer than 10 starts"), (1, ORANGE, "10+ starts")]:
        d = dl[dl.starter == flag]
        ax.scatter(d.combine_weight, d.gamma_all, s=42, color=color, edgecolor=SURFACE, linewidth=1.2, label=lab, zorder=3)
    w = np.linspace(dl.combine_weight.min() - 5, dl.combine_weight.max() + 5, 100)
    ax.plot(w, m.params.Intercept + m.params.combine_weight * w, color=INK2, lw=2, ls="--",
            label=f"Expected given weight (slope {m.params.combine_weight:.3f}, R² {m.rsquared:.2f})")
    offsets = {"Laiatu Latu": (-22, 0.55), "Braden Fiske": (14, 0.45), "Gervon Dexter": (14, 0.45)}
    for _, r in dl[dl.display_name.isin(list(offsets))].iterrows():
        dx, dy = offsets[r.display_name]
        ax.annotate(f"{r.display_name} ({r.gamma_all_oe:+.2f} over expected)", (r.combine_weight, r.gamma_all),
                    xytext=(r.combine_weight + dx, r.gamma_all + dy), fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_xlabel("Combine weight (lb)")
    ax.set_ylabel("90th-percentile |acceleration|, all drills (km/h/s)")
    ax.set_title(f"Defensive linemen (N={len(dl)}): heavier players post lower acceleration", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_ylim(top=dl.gamma_all.max() + 1.0)
    save(fig, "mass_vs_gamma_scatter.png")


def play_level_effects():
    """Within-player estimates: separation at the throw per 1 SD of speed held through the last cut."""
    m = pd.read_csv(os.path.join(cm.RESULTS_DIR, "play_fe_models.csv"))
    b = pd.read_csv(os.path.join(cm.RESULTS_DIR, "play_fe_robustness.csv"))
    sel = lambda s: m[(m.treatment == "apex_z") & (m.outcome == "separation") & (m["sample"] == s)].iloc[0]
    chk = lambda c: b[b.check == c].iloc[0]
    rows = [("Receivers, all coverage (primary test)", sel("WR, all coverage"), BLUE),
            ("Receivers, man coverage", sel("WR, man"), BLUE),
            ("Receivers, zone coverage", sel("WR, zone"), BLUE),
            ("Tight ends, all coverage", sel("TE, all coverage"), BLUE),
            ("Controlling for speed going into the cut", chk("+ entry speed into the cut"), BLUE),
            ("Cuts at least 0.6 s before the throw", chk("cut >= 0.6 s before the throw"), BLUE),
            ("Single-cut plays only", chk("single-cut plays only"), BLUE),
            ("Game fixed effects instead of opponent", chk("game FE instead of opponent x season"), BLUE),
            ("Placebo: speed going into the cut, alone", chk("placebo: entry speed alone"), INK2)]
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.axvline(0, color=INK2, lw=0.8)
    for i, (label, r, color) in enumerate(rows):
        y = len(rows) - 1 - i - (0.5 if i >= 4 else 0) - (0.5 if i == 8 else 0)
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([r.coef], [y], s=64, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(r.ci_hi + 0.012, y, f"{r.coef:+.2f}", va="center", fontsize=8.5, color=INK2)
        ax.text(-0.04, y, label, va="center", ha="right", fontsize=8.8, color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks([])
    ax.set_xlim(-0.12, 0.36)
    ax.set_xlabel("Yards of separation at the throw per 1 SD of speed held through the last cut (95% CI)")
    ax.set_title("Comparing each receiver with himself, holding speed through the cut goes with more separation",
                 loc="left", fontsize=11, color=INK, x=-0.62, pad=22)
    ax.text(-0.62, 1.025, "41 receivers, 4,450 regular-season routes; player, route, coverage, alignment, down and "
            "opponent fixed effects; clustered by player", transform=ax.transAxes, fontsize=8.3, color=INK2)
    ax.grid(axis="y", visible=False)
    save(fig, "play_level_cut_speed.png")


def make_all_plots():
    ct = cm.combine_tracking()
    wr = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_wr.parquet"))
    dl = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_dl.parquet"))
    draft_controls()
    speed_pdf_cdf(ct)
    route_plot(ct, wr)
    mass_gamma(dl)
    play_level_effects()


if __name__ == "__main__":
    make_all_plots()
