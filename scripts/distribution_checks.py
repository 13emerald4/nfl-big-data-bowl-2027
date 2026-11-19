"""
distribution_checks.py
Checks of the cut results that do not lean on normal approximations, plus the
distribution figures.

Play-level test (data/play_level.parquet, from play_level_fe.py)
  * wild cluster restricted bootstrap (Rademacher weights by receiver) p-value
    and test-inversion 95% interval, because 41 clusters is few for
    analytic clustered standard errors;
  * Freedman-Lane permutation: residuals of the model without the treatment
    are shuffled within each receiver, so the null keeps the player structure;
  * conformal CV+ width test with the treatment shuffled within receiver;
  * distribution regression: the effect on P(separation > c) for c = 1..6 yd,
    i.e. on the whole CDF rather than the mean.
Transfer test: Spearman correlations with bootstrap percentile intervals.
Stage 1: HC3 standard errors and pairs-bootstrap intervals.

Figures: separation_distribution.png, combine_vs_game_cuts.png,
permutation_nulls.png. Text: results/distribution_checks.txt.
"""
import os

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

import common as cm
import cod_features as cf
import play_level_fe as P
import make_field_plot as mfp
from make_field_plot import plt, BLUE, ORANGE, INK, INK2, GRID, SURFACE

RNG = np.random.default_rng(2027)
B_WILD, B_PERM, B_CONF, B_BOOT = 9999, 4999, 200, 4999
THRESHOLDS = [1, 2, 3, 4, 5, 6]


# ---------------------------------------------------------------- linear algebra helpers
class Partial:
    """Frisch-Waugh projection off the controls and fixed effects, plus CR1 clustered inference."""

    def __init__(self, Z, groups):
        U, s, _ = np.linalg.svd(Z, full_matrices=False)
        self.Q = U[:, s > s.max() * 1e-10]
        self.k = self.Q.shape[1] + 1
        self.g, self.G = pd.factorize(groups)[0], len(np.unique(groups))
        self.n = Z.shape[0]

    def M(self, v):
        return v - self.Q @ (self.Q.T @ v)

    def scale(self):
        return self.G / (self.G - 1) * (self.n - 1) / (self.n - self.k)

    def fit(self, yt, xt):
        sxx = xt @ xt
        b = xt @ yt / sxx
        sg = np.bincount(self.g, weights=xt * (yt - b * xt), minlength=self.G)
        return b, np.sqrt(self.scale() * (sg ** 2).sum() / sxx ** 2)


def matrices(d, treat, y="separation"):
    yv, X = patsy.dmatrices(f"{y} ~ {treat} + {P.COVARS} + {P.FE}", d, return_type="dataframe")
    rows = X.index
    Z = X.drop(columns=[treat]).to_numpy(float)
    return yv.to_numpy(float).ravel(), X[treat].to_numpy(float), Z, d.loc[rows]


def wild_cluster(pt, yt, xt, B=B_WILD):
    """Restricted wild cluster bootstrap (Rademacher) p-value at 0 and a test-inversion 95% interval."""
    b, se = pt.fit(yt, xt)
    sxx = xt @ xt
    Sxx = np.bincount(pt.g, weights=xt * xt, minlength=pt.G)
    W = RNG.choice([-1.0, 1.0], size=(B, pt.G))

    def pval(b0):
        ur = yt - b0 * xt
        S = np.bincount(pt.g, weights=xt * ur, minlength=pt.G)
        d = (W @ S) / sxx                                  # beta* - b0
        s_star = W * S - d[:, None] * Sxx
        se_star = np.sqrt(pt.scale() * (s_star ** 2).sum(1) / sxx ** 2)
        t_obs = (b - b0) / se
        return np.mean(np.abs(d / se_star) >= abs(t_obs))

    grid = np.linspace(b - 5 * se, b + 5 * se, 401)
    ps = np.array([pval(g) for g in grid])
    keep = grid[ps > 0.05]
    return dict(coef=b, se_cluster=se, ci_cluster=(b - 1.96 * se, b + 1.96 * se),
                p_wild=pval(0.0), ci_wild=(keep.min(), keep.max()))


def freedman_lane(pt, yt, xt, B=B_PERM):
    """Shuffle reduced-model residuals within receiver; statistic = clustered t."""
    b, se = pt.fit(yt, xt)
    t_obs = b / se
    idx = [np.flatnonzero(pt.g == g) for g in range(pt.G)]
    null = np.empty(B)
    for r in range(B):
        perm = np.arange(pt.n)
        for ix in idx:
            perm[ix] = RNG.permutation(ix)
        yp = pt.M(yt[perm])
        bb, ss = pt.fit(yp, xt)
        null[r] = bb / ss
    return t_obs, null, (1 + np.sum(np.abs(null) >= abs(t_obs))) / (B + 1)


def conformal_within(d):
    """CV+ width change from apex_z vs the same column shuffled within receiver."""
    d = d.reset_index(drop=True)
    _, Xf = patsy.dmatrices(f"separation ~ apex_z + {P.COVARS} + {P.FE}", d, return_type="dataframe")
    _, Xb = patsy.dmatrices(f"separation ~ {P.COVARS} + {P.FE}", d, return_type="dataframe")
    j = list(Xf.columns).index("apex_z")
    Xf, Xb = Xf.to_numpy(float), Xb.to_numpy(float)
    yv = d.separation.to_numpy(float)
    g = pd.factorize(d.nfl_id)[0]
    rng = np.random.default_rng(P.SEED)
    outer = np.array_split(rng.permutation(len(yv)), 10)
    state = rng.bit_generator.state

    def width(X):
        rng.bit_generator.state = state
        lo, hi = P.cv_plus(X, yv, outer, rng)
        return np.mean(hi - lo)

    wb, wf = width(Xb), width(Xf)
    null = np.empty(B_CONF)
    prng = np.random.default_rng(P.SEED + 7)
    for r in range(B_CONF):
        Xp = Xf.copy()
        col = Xp[:, j].copy()
        for gg in np.unique(g):
            ix = np.flatnonzero(g == gg)
            col[ix] = col[prng.permutation(ix)]
        Xp[:, j] = col
        null[r] = (width(Xp) - wb) / wb * 100
        print(f"  conformal within-receiver shuffle {r + 1}/{B_CONF}", end="\r", flush=True)
    obs = (wf - wb) / wb * 100
    return obs, null, (1 + np.sum(null <= obs)) / (B_CONF + 1)


# ---------------------------------------------------------------- play level
def play_level(lines):
    d = pd.read_parquet(os.path.join(cm.OUT_DIR, "play_level.parquet"))
    wr = d[d.pos_group == "WR"].reset_index(drop=True)
    out = {}
    lines.append("PLAY LEVEL (receivers; same model as play_level_fe.py)")
    for label, s, treat in [("apex_z, all coverage", wr, "apex_z"), ("apex_z, man", wr[wr.man == 1], "apex_z"),
                            ("apex_z, zone", wr[wr.man == 0], "apex_z"), ("turn_force, all coverage", wr, "turn_force")]:
        y, x, Z, rows = matrices(s.reset_index(drop=True), treat)
        pt = Partial(Z, rows.nfl_id.to_numpy())
        yt, xt = pt.M(y), pt.M(x)
        w = wild_cluster(pt, yt, xt)
        t_obs, null, p_fl = freedman_lane(pt, yt, xt)
        out[label] = dict(w, t=t_obs, null=null, p_fl=p_fl, n=len(y), G=pt.G)
        lines.append(f"  {label:26s} N={len(y):5d}, {pt.G} receivers: coef {w['coef']:+.3f}; "
                     f"analytic cluster CI [{w['ci_cluster'][0]:+.3f}, {w['ci_cluster'][1]:+.3f}]; "
                     f"wild-bootstrap CI [{w['ci_wild'][0]:+.3f}, {w['ci_wild'][1]:+.3f}], p = {w['p_wild']:.4f}; "
                     f"Freedman-Lane within-receiver p = {p_fl:.4f}")
    # distribution regression: effect on P(separation > c)
    lines.append("")
    lines.append("DISTRIBUTION REGRESSION: effect of +1 SD apex_z on P(separation > c), same fixed effects")
    y, x, Z, rows = matrices(wr, "apex_z")
    pt = Partial(Z, rows.nfl_id.to_numpy())
    xt = pt.M(x)
    dr = []
    for c in THRESHOLDS:
        yc = (y > c).astype(float)
        w = wild_cluster(pt, pt.M(yc), xt, B=4999)
        dr.append(dict(c=c, base=yc.mean(), coef=w["coef"], lo=w["ci_wild"][0], hi=w["ci_wild"][1], p=w["p_wild"]))
        lines.append(f"  c = {c} yd: baseline {yc.mean():.2f}; {w['coef'] * 100:+.1f} points per SD "
                     f"[{w['ci_wild'][0] * 100:+.1f}, {w['ci_wild'][1] * 100:+.1f}] (wild bootstrap), p = {w['p_wild']:.4f}")
    dr = pd.DataFrame(dr)
    # adjusted separation for the figure: residual on everything but apex_z, plus the mean
    yb, X = patsy.dmatrices(f"separation ~ {P.COVARS} + {P.FE}", wr, return_type="dataframe")
    adj = pt.M(yb.to_numpy(float).ravel()) + y.mean()
    dev = rows.apex_z - rows.groupby("nfl_id").apex_z.transform("mean")
    third = pd.qcut(dev, 3, labels=["slow", "mid", "fast"])
    fig_data = pd.DataFrame({"adj": adj, "third": third.to_numpy(), "raw": y})
    for t in ["fast", "slow"]:
        v = fig_data[fig_data.third == t]
        lines.append(f"  {t:4s} third of the receiver's own cuts: adjusted separation mean {v.adj.mean():.2f}, "
                     f"P(> 3 yd) {np.mean(v.adj > 3):.3f}, P(> 5 yd) {np.mean(v.adj > 5):.3f}")
    ks = stats.ks_2samp(fig_data[fig_data.third == "fast"].adj, fig_data[fig_data.third == "slow"].adj)
    lines.append(f"  two-sample KS, fast vs slow third (ignores clustering, descriptive): D = {ks.statistic:.3f}, p = {ks.pvalue:.4f}")
    return out, dr, fig_data, wr


# ---------------------------------------------------------------- transfer, Stage 1
def spearman_boot(a, b, B=B_BOOT):
    a, b = np.asarray(a), np.asarray(b)
    rho = stats.spearmanr(a, b)[0]
    n = len(a)
    bs = np.array([stats.spearmanr(a[i], b[i])[0] for i in (RNG.integers(0, n, n) for _ in range(B))])
    return rho, np.percentile(bs, [2.5, 97.5])


def transfer(lines):
    import transfer_test as T
    base = cm.player_base()
    base = base[base.combine_weight.notna()]
    pp = cm.player_play_regular_season(usecols=["game_id", "play_id", "nfl_id", "route_ran", "rec_yards"])
    cuts = T.add_z(T.load_cuts(base, pp))
    del pp
    gm = T.player_metrics(cuts)
    feat = pd.read_parquet(os.path.join(cm.OUT_DIR, "features_cod.parquet")).drop(columns=["pos_group"]).merge(gm, on="nfl_id")
    lines.append("")
    lines.append("TRANSFER, rank-based (Spearman; 95% percentile bootstrap)")
    for cmb, g, grp, partial in T.PRIMARY + T.SECONDARY:
        d = feat[feat.pos_group != "QB"] if grp == "ALL" else feat[feat.pos_group == grp]
        d = d.dropna(subset=[cmb, g])
        a, b = d[cmb].rank(), d[g].rank()
        if partial:
            a, b, _ = T.partial_on_group(d.assign(**{cmb: a, g: b}), cmb, g)
        rho, ci = spearman_boot(a, b)
        lines.append(f"  {cmb:16s} vs {g:20s} {grp:3s} N={len(d):3d}: rho = {rho:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
                     f"{' (partial on position)' if partial else ''}")
    return cuts, feat


def stage1(lines):
    import cod_force_models as M
    df = M.load()
    S = M.samples(df)
    lines.append("")
    lines.append("STAGE 1 (Combine cut metrics -> careers), small-sample checks")
    d = S[("WR", "yprr")].dropna(subset=["cod_speed_z", "yprr", "log_pick", "udfa", "draft_year"]).reset_index(drop=True)
    f = f"yprr ~ {M.rhs_for('cod_speed_z', 'WR', True, [])}"
    m1, m3 = smf.ols(f, d).fit(cov_type="HC1"), smf.ols(f, d).fit(cov_type="HC3")
    bs = []
    for _ in range(B_BOOT):
        i = RNG.integers(0, len(d), len(d))
        try:
            bs.append(smf.ols(f, d.iloc[i]).fit().params["cod_speed_z"])
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    lines.append(f"  cod_speed_z -> YPRR (N={len(d)}): {m1.params.cod_speed_z:+.3f}; HC1 p = {m1.pvalues.cod_speed_z:.3f}, "
                 f"HC3 p = {m3.pvalues.cod_speed_z:.3f} [{m3.conf_int().loc['cod_speed_z'][0]:+.3f}, "
                 f"{m3.conf_int().loc['cod_speed_z'][1]:+.3f}]; pairs bootstrap [{lo:+.3f}, {hi:+.3f}]")
    d = S[("ALL", "snaps_per_season")].dropna(subset=["force_oe_z", "log_pick", "udfa", "draft_year"]).reset_index(drop=True)
    rhs = M.rhs_for("force_oe_z", "ALL", True, [])
    m = M.fit("poisson", "snaps_per_season", rhs, d)
    bs = []
    for _ in range(1999):
        i = RNG.integers(0, len(d), len(d))
        try:
            bs.append(M.fit("poisson", "snaps_per_season", rhs, d.iloc[i].reset_index(drop=True)).params["force_oe_z"])
        except Exception:
            pass
    lo, hi = np.percentile(bs, [2.5, 97.5])
    lines.append(f"  force_oe_z -> snaps per season (N={len(d)}): {(np.exp(m.params.force_oe_z) - 1) * 100:+.1f}% per SD, "
                 f"robust p = {m.pvalues.force_oe_z:.3f}; pairs bootstrap [{(np.exp(lo) - 1) * 100:+.1f}%, {(np.exp(hi) - 1) * 100:+.1f}%]")


# ---------------------------------------------------------------- figures
def ecdf(v):
    v = np.sort(np.asarray(v))
    return v, np.arange(1, len(v) + 1) / len(v)


def fig_separation(fd, dr):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), gridspec_kw={"wspace": 0.32})
    grid = np.linspace(-3, 12, 400)
    for t, color, lab in [("fast", BLUE, "Fastest third of his own cuts"), ("slow", ORANGE, "Slowest third of his own cuts")]:
        v = fd[fd.third == t].adj.to_numpy()
        axes[0].plot(grid, stats.gaussian_kde(v)(grid), color=color, lw=2, label=lab)
        x, p = ecdf(v)
        axes[1].plot(x, p, color=color, lw=2, label=lab)
    axes[0].set_title("Density (PDF)", loc="left", fontsize=10.5, color=INK)
    axes[0].set_xlabel("Adjusted separation at the throw (yd)")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8.5, loc="upper right")
    axes[1].set_title("Cumulative distribution (CDF)", loc="left", fontsize=10.5, color=INK)
    axes[1].set_xlabel("Adjusted separation at the throw (yd)")
    axes[1].set_ylabel("Share of plays at or below")
    axes[1].set_xlim(-3, 12)
    a = axes[2]
    a.axhline(0, color=INK2, lw=0.8)
    a.vlines(dr.c, dr.lo * 100, dr.hi * 100, color=BLUE, lw=2)
    a.scatter(dr.c, dr.coef * 100, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    for r in dr.itertuples():
        a.text(r.c + 0.12, r.coef * 100, f"{r.coef * 100:+.1f}", va="center", fontsize=8.3, color=INK2)
    a.set_xticks(dr.c)
    a.set_xticklabels([f">{c}" for c in dr.c])
    a.set_title("Effect on P(separation > c), per 1 SD", loc="left", fontsize=10.5, color=INK)
    a.set_xlabel("Separation threshold c (yd)")
    a.set_ylabel("Percentage points (95% wild-bootstrap CI)")
    a.grid(axis="x", visible=False)
    fig.suptitle("Holding speed through the cut turns tight coverage into open targets, not wide-open ones",
                 x=0.06, ha="left", y=1.04, fontsize=11.5, color=INK)
    fig.text(0.06, 0.965, "41 receivers, 4,450 routes. Separation is adjusted for route, coverage, alignment, down, opponent, "
             "cut timing and the receiver himself; plays split by the receiver's own cut speed", fontsize=8.5, color=INK2)
    mfp.save(fig, "separation_distribution.png")


def fig_combine_vs_game(cuts):
    ev = pd.read_parquet(os.path.join(cm.OUT_DIR, "cod_events.parquet"))
    comb = ev[ev.drill.isin(cf.WR_ROUTE_DRILLS) & ev.complete]
    game = cuts[(cuts.pos_group == "WR") & cuts.pass_play]
    ids = np.intersect1d(comb.nfl_id.unique(), game.nfl_id.unique())
    comb, game = comb[comb.nfl_id.isin(ids)], game[game.nfl_id.isin(ids)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"wspace": 0.25})
    for v, color, lab in [(comb, BLUE, f"Combine route drills ({len(comb):,} cuts)"),
                          (game, ORANGE, f"Games, pass plays ({len(game):,} cuts)")]:
        x, p = ecdf(v.apex * 3.6)
        axes[0].plot(x, p, color=color, lw=2, label=lab)
        x, p = ecdf(v.turn_deg)
        axes[1].plot(x, p, color=color, lw=2, label=lab)
    axes[0].set_xlabel("Speed at the slowest point of the cut (km/h)")
    axes[0].set_ylabel("Share of cuts at or below")
    axes[0].set_title("Speed through the cut", loc="left", fontsize=10.5, color=INK)
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[1].set_xlabel("Turn angle (degrees)")
    axes[1].set_title("Sharpness of the cut", loc="left", fontsize=10.5, color=INK)
    fig.suptitle(f"The same {len(ids)} receivers cut differently at the Combine and in games",
                 x=0.07, ha="left", y=1.02, fontsize=11.5, color=INK)
    mfp.save(fig, "combine_vs_game_cuts.png")
    med = lambda s: float(np.median(s))
    return dict(n=len(ids), comb_speed=med(comb.apex * 3.6), game_speed=med(game.apex * 3.6),
                comb_turn=med(comb.turn_deg), game_turn=med(game.turn_deg))


def fig_nulls(pl, conf):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"wspace": 0.25})
    r = pl["apex_z, all coverage"]
    axes[0].hist(r["null"], bins=60, color=GRID, edgecolor=SURFACE)
    axes[0].axvline(r["t"], color=BLUE, lw=2)
    axes[0].text(r["t"], axes[0].get_ylim()[1] * 0.9, f"  observed t = {r['t']:.2f}\n  p = {r['p_fl']:.4f}",
                 color=INK, fontsize=8.5, va="top")
    axes[0].set_title("Freedman-Lane, residuals shuffled within receiver", loc="left", fontsize=10, color=INK)
    axes[0].set_xlabel("Clustered t statistic under the null")
    axes[0].set_ylabel("Count")
    obs, null, p = conf
    axes[1].hist(null, bins=30, color=GRID, edgecolor=SURFACE)
    axes[1].axvline(obs, color=BLUE, lw=2)
    axes[1].text(obs, axes[1].get_ylim()[1] * 0.9, f"  observed {obs:+.2f}%\n  p = {p:.3f}", color=INK, fontsize=8.5, va="top")
    axes[1].set_title("Conformal width, cut speed shuffled within receiver", loc="left", fontsize=10, color=INK)
    axes[1].set_xlabel("Change in mean 80% interval width (%)")
    fig.suptitle("Permutation null distributions for the within-receiver cut-speed effect", x=0.07, ha="left",
                 y=1.03, fontsize=11.5, color=INK)
    mfp.save(fig, "permutation_nulls.png")


def main():
    lines = ["Distribution-free and small-sample checks of the cut results", ""]
    pl, dr, fd, wr = play_level(lines)
    print("play level done", flush=True)
    fig_separation(fd, dr)
    conf = conformal_within(wr)
    lines.append(f"  conformal CV+ width with apex_z: {conf[0]:+.2f}% vs within-receiver shuffles "
                 f"(median {np.median(conf[1]):+.2f}%), permutation p = {conf[2]:.3f}")
    fig_nulls(pl, conf)
    cuts, _ = transfer(lines)
    s = fig_combine_vs_game(cuts)
    lines.append(f"  Combine vs game cuts, same {s['n']} receivers: median speed at the cut {s['comb_speed']:.1f} vs "
                 f"{s['game_speed']:.1f} km/h; median turn {s['comb_turn']:.0f} vs {s['game_turn']:.0f} degrees")
    del cuts
    stage1(lines)
    dr.to_csv(os.path.join(cm.RESULTS_DIR, "distribution_regression.csv"), index=False)
    txt = "\n".join(lines)
    with open(os.path.join(cm.RESULTS_DIR, "distribution_checks.txt"), "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print("\n" + txt)


if __name__ == "__main__":
    main()
